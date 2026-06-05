"""
Train GPT-2 on synthetic arithmetic data (Stage 1) — PEMDAS multi-step examples.

Uses the same checkpoints as reasoning_generation.py (--checkpoint_tag, default: reasoning).
Epoch indices continue across stages: arithmetic epochs 0..N-1, then GSM8K picks up at N.

Usage:
  python prepare_arithmetic.py
  bash scripts/fresh_start.sh   # optional: delete old .pt files
  python arithmetic_pretrain.py --use_gpu --epochs 10 --lr 1e-4
  python reasoning_generation.py --use_gpu --epochs 60 --lr 5e-6   # GSM8K after 10 arithmetic epochs
"""

import argparse

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
  resolve_training_start,
  save_model,
  set_checkpoint_filepath,
)
from gpt_datasets import ReasoningDataset, mask_labels_to_reasoning_only
from optimizer import AdamW
from reasoning_generation import ReasoningGPT, add_arguments, seed_everything

TQDM_DISABLE = False


def train(args):
  device = torch.device('cuda') if args.use_gpu else torch.device('cpu')
  dataset = ReasoningDataset(args.arithmetic_path)
  dataloader = DataLoader(
      dataset, shuffle=True, batch_size=args.batch_size, collate_fn=dataset.collate_fn
  )

  args = add_arguments(args)
  cleanup_incomplete_checkpoints(args)

  model = ReasoningGPT(args).to(device)
  start_epoch, latest_epoch = resolve_training_start(
      model, args, init_checkpoint=args.init_checkpoint
  )

  optimizer = AdamW(model.parameters(), lr=args.lr)

  if start_epoch >= args.epochs:
    print(
        f"Arithmetic stage already complete through global epoch {latest_epoch} "
        f"(target --epochs {args.epochs})."
    )
    return

  for epoch in range(start_epoch, args.epochs):
    model.train()
    train_loss = 0
    num_batches = 0

    for batch in tqdm(dataloader, desc=f'arithmetic-{epoch}', disable=TQDM_DISABLE):
      b_ids = batch['token_ids'].to(device)
      b_mask = batch['attention_mask'].to(device)

      optimizer.zero_grad()
      logits = model(b_ids, b_mask)
      logits = rearrange(logits[:, :-1].contiguous(), 'b t d -> (b t) d')
      labels = b_ids[:, 1:].clone()
      labels = mask_labels_to_reasoning_only(labels, b_mask, batch['reasoning_starts'])
      loss = F.cross_entropy(
          logits, labels.flatten(), ignore_index=-100, reduction='mean'
      )
      loss.backward()
      optimizer.step()

      train_loss += loss.item()
      num_batches += 1

    print(f"Epoch {epoch}: train loss :: {train_loss / num_batches:.3f}")
    prune_old_checkpoints(epoch, args)
    save_model(model, optimizer, args, checkpoint_path(epoch, args))


def get_args():
  parser = argparse.ArgumentParser()
  parser.add_argument('--arithmetic_path', type=str, default='data/arithmetic_pretrain.txt')
  parser.add_argument('--seed', type=int, default=11711)
  parser.add_argument('--epochs', type=int, default=5,
                      help='Global epoch index limit for this stage (e.g. 5 runs epochs 0-4).')
  parser.add_argument('--use_gpu', action='store_true')
  parser.add_argument('--batch_size', type=int, default=16)
  parser.add_argument('--lr', type=float, default=1e-4)
  parser.add_argument('--model_size', type=str, default='gpt2',
                      choices=['gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'])
  add_checkpoint_args(parser)
  return parser.parse_args()


if __name__ == '__main__':
  args = get_args()
  set_checkpoint_filepath(args)
  seed_everything(args.seed)
  train(args)
