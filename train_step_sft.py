#!/usr/bin/env python3
"""
Phase 1: Step-conditioned SFT — train next reasoning line given prefix.

Starts from best_exp4_gsm8k_ma_ent.pt by default. Best checkpoint selected on gsm8k_dev.

Usage:
  python prepare_step_sft.py
  python train_step_sft.py --use_gpu
  python train_step_sft.py --use_gpu --init_checkpoint best_exp4_gsm8k_ma_ent.pt --eval_limit 50
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import time

import torch
import torch.nn.functional as F
from einops import rearrange
from torch.utils.data import DataLoader
from tqdm import tqdm

from checkpoint_utils import (
  add_checkpoint_args,
  checkpoint_path,
  cleanup_incomplete_checkpoints,
  prune_old_checkpoints,
  save_model,
  set_checkpoint_filepath,
)
from gpt_datasets import StepReasoningDataset, mask_labels_for_step
from optimizer import AdamW
from reasoning_generation import ReasoningGPT, add_arguments, seed_everything
from train_with_eval import (
  GSM8K_DEV_JSONL,
  evaluate_dev,
  metrics_path,
  summary_path,
)

TQDM_DISABLE = False
DEFAULT_INIT = "best_exp4_gsm8k_ma_ent.pt"
DEFAULT_TRAIN = os.path.join("data", "step_sft_train.jsonl")
DEFAULT_TAG = "exp8_step_sft"


def best_checkpoint_path(checkpoint_tag: str) -> str:
  return f"best_{checkpoint_tag}.pt"


def cleanup_experiment_checkpoints(checkpoint_tag: str):
  for pattern in (f"*_{checkpoint_tag}.pt", f"best_{checkpoint_tag}.pt"):
    for path in glob.glob(pattern):
      os.remove(path)
      print(f"removed {path}")


def train_one_epoch(model, dataloader, optimizer, device):
  model.train()
  train_loss = 0.0
  num_batches = 0

  for batch in tqdm(dataloader, desc="train", disable=TQDM_DISABLE):
    b_ids = batch["token_ids"].to(device)
    b_mask = batch["attention_mask"].to(device)

    optimizer.zero_grad()
    logits = model(b_ids, b_mask)
    logits = rearrange(logits[:, :-1].contiguous(), "b t d -> (b t) d")
    labels = b_ids[:, 1:].clone()
    labels = mask_labels_for_step(
        labels, b_mask, batch["loss_starts"], batch["loss_ends"]
    )

    loss = F.cross_entropy(
        logits, labels.flatten(), ignore_index=-100, reduction="mean"
    )
    loss.backward()
    optimizer.step()

    train_loss += loss.item()
    num_batches += 1

  return train_loss / max(num_batches, 1)


def train(args):
  device = torch.device("cuda") if args.use_gpu else torch.device("cpu")

  if args.fresh:
    cleanup_experiment_checkpoints(args.checkpoint_tag)

  dataset = StepReasoningDataset(args.train_path)
  dataloader = DataLoader(
      dataset,
      shuffle=True,
      batch_size=args.batch_size,
      collate_fn=dataset.collate_fn,
  )

  args = add_arguments(args)
  set_checkpoint_filepath(args)
  cleanup_incomplete_checkpoints(args)

  model = ReasoningGPT(args).to(device)
  if args.init_checkpoint:
    saved = torch.load(args.init_checkpoint, weights_only=False)
    model.load_state_dict(saved["model"])
    print(f"Initialized weights from {args.init_checkpoint}")
  else:
    print("Training from pretrained GPT-2 (no init checkpoint).")

  optimizer = AdamW(model.parameters(), lr=args.lr)

  best_acc = -1.0
  best_epoch = -1
  patience_left = args.patience
  history = []
  finished_epoch = -1

  print(
      f"Phase 1 step-SFT tag={args.checkpoint_tag} "
      f"examples={len(dataset)} epochs={args.epochs}"
  )

  for epoch in range(args.epochs):
    t0 = time.time()
    train_loss = train_one_epoch(model, dataloader, optimizer, device)
    elapsed = time.time() - t0
    finished_epoch = epoch
    print(f"Epoch {epoch}: train_loss={train_loss:.4f} ({elapsed:.0f}s)")

    should_eval = (epoch % args.eval_every == 0) or (epoch == args.epochs - 1)
    if should_eval:
      metrics = evaluate_dev(
          model,
          device,
          dev_path=args.dev_path,
          temperature=args.temperature,
          top_p=args.top_p,
          max_new_tokens=args.max_new_tokens,
          greedy=args.greedy,
          limit=args.eval_limit,
      )
      metrics["epoch"] = epoch
      metrics["train_loss"] = train_loss
      metrics["checkpoint_tag"] = args.checkpoint_tag
      history.append(metrics)

      with open(metrics_path(args.checkpoint_tag, epoch), "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

      acc = metrics["exact_accuracy"]
      print(
          f"  eval: exact_accuracy={acc:.3f} "
          f"format_valid={metrics['format_valid_rate']:.3f}"
      )

      if acc > best_acc:
        best_acc = acc
        best_epoch = epoch
        patience_left = args.patience
        best_path = best_checkpoint_path(args.checkpoint_tag)
        save_model(model, optimizer, args, best_path)
        print(f"  ** new best -> {best_path} (acc={acc:.3f})")
      else:
        patience_left -= 1
        print(f"  no improvement (patience left: {patience_left})")

      if patience_left <= 0:
        print(f"Early stopping at epoch {epoch}.")
        break

    prune_old_checkpoints(epoch, args)
    save_model(model, optimizer, args, checkpoint_path(epoch, args))

  summary = {
      "phase": "step_sft",
      "checkpoint_tag": args.checkpoint_tag,
      "train_path": args.train_path,
      "init_checkpoint": args.init_checkpoint,
      "best_epoch": best_epoch,
      "best_exact_accuracy": best_acc,
      "best_checkpoint": best_checkpoint_path(args.checkpoint_tag),
      "history": history,
      "finished_epoch": finished_epoch,
  }
  with open(summary_path(args.checkpoint_tag), "w", encoding="utf-8") as f:
    json.dump(summary, f, indent=2)
  print(json.dumps(summary, indent=2))


def get_args():
  parser = argparse.ArgumentParser()
  parser.add_argument("--train_path", type=str, default=DEFAULT_TRAIN)
  parser.add_argument("--dev_path", type=str, default=GSM8K_DEV_JSONL)
  parser.add_argument("--epochs", type=int, default=10)
  parser.add_argument("--eval_every", type=int, default=2)
  parser.add_argument("--patience", type=int, default=4)
  parser.add_argument("--lr", type=float, default=2e-6)
  parser.add_argument("--batch_size", type=int, default=8)
  parser.add_argument("--use_gpu", action="store_true")
  parser.add_argument("--seed", type=int, default=11711)
  parser.add_argument("--fresh", action="store_true", default=True)
  parser.add_argument("--no_fresh", action="store_true")
  parser.add_argument("--temperature", type=float, default=0.7)
  parser.add_argument("--top_p", type=float, default=0.9)
  parser.add_argument("--max_new_tokens", type=int, default=256)
  parser.add_argument("--greedy", action="store_true")
  parser.add_argument("--eval_limit", type=int, default=0)
  parser.add_argument("--model_size", default="gpt2",
                      choices=["gpt2", "gpt2-medium", "gpt2-large", "gpt2-xl"])
  add_checkpoint_args(parser, default_tag=DEFAULT_TAG, default_init=DEFAULT_INIT)
  return parser.parse_args()


if __name__ == "__main__":
  args = get_args()
  if args.no_fresh:
    args.fresh = False
  seed_everything(args.seed)
  train(args)
