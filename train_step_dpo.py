#!/usr/bin/env python3
"""
Phase 2: Step-level DPO — prefer gold next step over planning-focused rejected step.

Starts from Phase 1 best checkpoint by default; reference model is frozen copy at init.

Usage:
  python prepare_step_dpo.py
  python train_step_dpo.py --use_gpu
  python train_step_dpo.py --use_gpu --init_checkpoint best_exp8_step_sft.pt --beta 0.2
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import time

import torch
import torch.nn.functional as F
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
from gpt_datasets import StepDPODataset
from optimizer import AdamW
from reasoning_generation import ReasoningGPT, add_arguments, seed_everything
from train_with_eval import (
  GSM8K_DEV_JSONL,
  evaluate_dev,
  metrics_path,
  summary_path,
)

TQDM_DISABLE = False
DEFAULT_INIT = "best_exp8_step_sft.pt"
DEFAULT_TRAIN = os.path.join("data", "step_dpo_train.jsonl")
DEFAULT_TAG = "exp9_step_dpo"


def best_checkpoint_path(checkpoint_tag: str) -> str:
  return f"best_{checkpoint_tag}.pt"


def cleanup_experiment_checkpoints(checkpoint_tag: str):
  for pattern in (f"*_{checkpoint_tag}.pt", f"best_{checkpoint_tag}.pt"):
    for path in glob.glob(pattern):
      os.remove(path)
      print(f"removed {path}")


def sequence_logprob(model, input_ids, attention_mask, prompt_len):
  """Sum log-prob of completion tokens (positions >= prompt_len)."""
  logits = model(input_ids, attention_mask)
  log_probs = F.log_softmax(logits[:, :-1, :], dim=-1)
  targets = input_ids[:, 1:]
  token_logps = log_probs.gather(2, targets.unsqueeze(-1)).squeeze(-1)

  batch_size, seq_len = token_logps.shape
  positions = torch.arange(seq_len, device=token_logps.device).unsqueeze(0)
  # Predict token at index t uses logit at t-1; completion starts at prompt_len.
  completion_mask = (positions >= prompt_len - 1) & (attention_mask[:, 1:] == 1)
  masked = token_logps * completion_mask.float()
  return masked.sum(dim=1)


def dpo_loss(policy_c, policy_r, ref_c, ref_r, beta):
  logits = beta * ((policy_c - policy_r) - (ref_c - ref_r))
  return -F.logsigmoid(logits).mean()


def train_one_epoch(model, ref_model, dataloader, optimizer, device, beta):
  model.train()
  ref_model.eval()
  train_loss = 0.0
  num_batches = 0

  for batch in tqdm(dataloader, desc="dpo", disable=TQDM_DISABLE):
    c_ids = batch["chosen_ids"].to(device)
    c_mask = batch["chosen_mask"].to(device)
    r_ids = batch["rejected_ids"].to(device)
    r_mask = batch["rejected_mask"].to(device)
    prompt_lens = batch["prompt_lens"].to(device)

    optimizer.zero_grad()

    with torch.no_grad():
      ref_c = sequence_logprob(ref_model, c_ids, c_mask, prompt_lens)
      ref_r = sequence_logprob(ref_model, r_ids, r_mask, prompt_lens)

    pol_c = sequence_logprob(model, c_ids, c_mask, prompt_lens)
    pol_r = sequence_logprob(model, r_ids, r_mask, prompt_lens)

    loss = dpo_loss(pol_c, pol_r, ref_c, ref_r, beta)
    loss.backward()
    optimizer.step()

    train_loss += loss.item()
    num_batches += 1

  return train_loss / max(num_batches, 1)


def train(args):
  device = torch.device("cuda") if args.use_gpu else torch.device("cpu")

  if args.fresh:
    cleanup_experiment_checkpoints(args.checkpoint_tag)

  dataset = StepDPODataset(args.train_path)
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
  ref_model = ReasoningGPT(args).to(device)

  if args.init_checkpoint:
    saved = torch.load(args.init_checkpoint, weights_only=False)
    model.load_state_dict(saved["model"])
    ref_model.load_state_dict(saved["model"])
    print(f"Initialized policy + ref from {args.init_checkpoint}")
  else:
    raise SystemExit("--init_checkpoint is required for step DPO")

  for p in ref_model.parameters():
    p.requires_grad = False

  optimizer = AdamW(model.parameters(), lr=args.lr)

  best_acc = -1.0
  best_epoch = -1
  patience_left = args.patience
  history = []
  finished_epoch = -1

  print(
      f"Phase 2 step-DPO tag={args.checkpoint_tag} "
      f"pairs={len(dataset)} beta={args.beta} epochs={args.epochs}"
  )

  for epoch in range(args.epochs):
    t0 = time.time()
    train_loss = train_one_epoch(
        model, ref_model, dataloader, optimizer, device, args.beta
    )
    elapsed = time.time() - t0
    finished_epoch = epoch
    print(f"Epoch {epoch}: dpo_loss={train_loss:.4f} ({elapsed:.0f}s)")

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
      metrics["dpo_loss"] = train_loss
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
      "phase": "step_dpo",
      "checkpoint_tag": args.checkpoint_tag,
      "train_path": args.train_path,
      "init_checkpoint": args.init_checkpoint,
      "beta": args.beta,
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
  parser.add_argument("--beta", type=float, default=0.2)
  parser.add_argument("--epochs", type=int, default=4)
  parser.add_argument("--eval_every", type=int, default=1)
  parser.add_argument("--patience", type=int, default=3)
  parser.add_argument("--lr", type=float, default=1e-6)
  parser.add_argument("--batch_size", type=int, default=4)
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
