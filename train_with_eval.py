#!/usr/bin/env python3
"""
Train ReasoningGPT from pretrained GPT-2 with periodic MultiArith eval and best-checkpoint tracking.

Experiments (see prepare_experiment_data.py + scripts/run_overnight_experiments.sh):
  exp1–exp3  from-scratch data ablations
  exp4       arithmetic curriculum → exp3 mix
  exp5       GSM8K + MultiArith aug + entity
  exp6       exp3 mix + subsampled PEMDAS arithmetic
  exp7       arithmetic curriculum → GSM8K + entity + subsampled MA (best on GSM8K dev)
  exp10      exp4 curriculum → GSM8K + entity + subsampled MA aug (best on GSM8K dev)
  exp11      sequential: arith → entity → MA aug → GSM8K (best on GSM8K dev)
  exp14      exp10-style mix with GSM8K 4500 + entity stage-1 (mask_target auto)
  exp15      exp4 mix + planning entity (5-7 step LLM), best on GSM8K dev

Each run:
  - starts from pretrained GPT-2 (no resume unless --resume)
  - evaluates on data/multiarith_dev.jsonl every --eval_every epochs
  - keeps best_{checkpoint_tag}.pt by exact_accuracy
  - early-stops after --patience evals without improvement

Usage:
  python prepare_experiment_data.py

  python train_with_eval.py --experiment exp1 --use_gpu
  python train_with_eval.py --experiment exp2 --use_gpu
  python train_with_eval.py --experiment exp3 --use_gpu
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

import gsm8k_eval
from checkpoint_utils import (
  add_checkpoint_args,
  checkpoint_path,
  cleanup_incomplete_checkpoints,
  prune_old_checkpoints,
  resolve_training_start,
  save_model,
  set_checkpoint_filepath,
)
from gpt_datasets import ReasoningDataset, mask_labels_from_start
from optimizer import AdamW
from reasoning_generation import ReasoningGPT, add_arguments, seed_everything

TQDM_DISABLE = False
DEV_JSONL = os.path.join('data', 'multiarith_dev.jsonl')
GSM8K_DEV_JSONL = os.path.join('data', 'gsm8k_dev.jsonl')

EXPERIMENT_CONFIGS = {
    'exp1': {
        'train_path': 'data/experiments/exp1_gsm8k_train.txt',
        'checkpoint_tag': 'exp1_gsm8k',
        'epochs': 30,
        'eval_every': 2,
        'patience': 5,
        'lr': 5e-6,
    },
    'exp2': {
        'train_path': 'data/experiments/exp2_gsm8k_multiarith_train.txt',
        'checkpoint_tag': 'exp2_gsm8k_ma',
        'epochs': 30,
        'eval_every': 2,
        'patience': 5,
        'lr': 5e-6,
    },
    'exp3': {
        'train_path': 'data/experiments/exp3_gsm8k_multiarith_entity_train.txt',
        'checkpoint_tag': 'exp3_gsm8k_ma_ent',
        'epochs': 32,
        'eval_every': 2,
        'patience': 5,
        'lr': 5e-6,
    },
    'exp4': {
        'train_path': 'data/experiments/exp3_gsm8k_multiarith_entity_train.txt',
        'checkpoint_tag': 'exp4_gsm8k_ma_ent',
        'epochs': 32,
        'eval_every': 2,
        'patience': 5,
        'lr': 5e-6,
    },
    'exp5': {
        'train_path': 'data/experiments/exp5_gsm8k_ma_aug_entity_train.txt',
        'checkpoint_tag': 'exp5_gsm8k_ma_aug_ent',
        'epochs': 36,
        'eval_every': 2,
        'patience': 6,
        'lr': 5e-6,
    },
    'exp6': {
        'train_path': 'data/experiments/exp6_gsm8k_ma_ent_arith_train.txt',
        'checkpoint_tag': 'exp6_gsm8k_ma_ent_arith',
        'epochs': 34,
        'eval_every': 2,
        'patience': 5,
        'lr': 5e-6,
    },
    'exp7': {
        'train_path': 'data/experiments/exp7_gsm8k_ma_sub_ent_train.txt',
        'checkpoint_tag': 'exp7_gsm8k_ma_sub_ent',
        'dev_path': GSM8K_DEV_JSONL,
        'epochs': 32,
        'eval_every': 4,
        'patience': 5,
        'lr': 5e-6,
    },
    'exp10': {
        'train_path': 'data/experiments/exp10_gsm8k_ma_aug_sub_ent_train.txt',
        'checkpoint_tag': 'exp10_gsm8k_ma_aug_sub_ent',
        'dev_path': GSM8K_DEV_JSONL,
        'epochs': 32,
        'eval_every': 4,
        'patience': 5,
        'lr': 5e-6,
    },
    'exp11': {
        'train_path': 'data/gsm8k_sft_train.txt',
        'checkpoint_tag': 'exp11_gsm8k',
        'dev_path': GSM8K_DEV_JSONL,
        'epochs': 32,
        'eval_every': 4,
        'patience': 5,
        'lr': 5e-6,
    },
    'exp14': {
        'train_path': 'data/experiments/exp14_gsm8k4500_ent1_ma_aug_train.txt',
        'checkpoint_tag': 'exp14_gsm8k4500_ent1_ma',
        'dev_path': GSM8K_DEV_JSONL,
        'epochs': 32,
        'eval_every': 4,
        'patience': 5,
        'lr': 5e-6,
        'mask_target': 'auto',
    },
    'exp15': {
        'train_path': 'data/experiments/exp15_gsm8k_ma_ent_planning_train.txt',
        'checkpoint_tag': 'exp15_gsm8k_ma_ent_planning',
        'dev_path': GSM8K_DEV_JSONL,
        'epochs': 32,
        'eval_every': 4,
        'patience': 5,
        'lr': 5e-6,
    },
}


def load_dev(path: str) -> list:
  records = []
  with open(path, 'r', encoding='utf-8') as f:
    for line in f:
      records.append(json.loads(line))
  return records


@torch.no_grad()
def evaluate_dev(
    model: ReasoningGPT,
    device: torch.device,
    *,
    dev_path: str = DEV_JSONL,
    temperature: float = 0.7,
    top_p: float = 0.9,
    max_new_tokens: int = 256,
    greedy: bool = False,
    limit: int = 0,
) -> dict:
  model.eval()
  dev = load_dev(dev_path)
  if limit > 0:
    dev = dev[:limit]

  eval_records = []
  for rec in dev:
    prompt = f"Question: {rec['question']}\n\nReasoning:\n"
    enc = model.tokenizer(
        prompt, return_tensors='pt', truncation=True, max_length=512
    ).to(device)
    _, generated = model.generate(
        enc['input_ids'],
        temperature=temperature,
        top_p=top_p,
        max_length=max_new_tokens,
        sample=not greedy,
    )
    continuation = generated[len(prompt):] if generated.startswith(prompt) else generated
    eval_records.append({'generation': continuation, 'gold': rec['gold_answer']})

  metrics = gsm8k_eval.evaluate(eval_records)
  if 'gsm8k' in os.path.basename(dev_path):
    metrics['dataset'] = 'gsm8k_dev'
  else:
    metrics['dataset'] = 'multiarith'
  metrics['n_dev'] = len(dev)
  metrics['dev_path'] = dev_path
  return metrics


def evaluate_multiarith(*args, **kwargs):
  """Backward-compatible alias."""
  return evaluate_dev(*args, **kwargs)


def best_checkpoint_path(checkpoint_tag: str) -> str:
  return f'best_{checkpoint_tag}.pt'


def metrics_path(checkpoint_tag: str, epoch: int) -> str:
  out_dir = os.path.join('outputs', checkpoint_tag)
  os.makedirs(out_dir, exist_ok=True)
  return os.path.join(out_dir, f'metrics_epoch_{epoch}.json')


def summary_path(checkpoint_tag: str) -> str:
  out_dir = os.path.join('outputs', checkpoint_tag)
  os.makedirs(out_dir, exist_ok=True)
  return os.path.join(out_dir, 'training_summary.json')


def cleanup_experiment_checkpoints(checkpoint_tag: str):
  for pattern in (f'*_{checkpoint_tag}.pt', f'best_{checkpoint_tag}.pt'):
    for path in glob.glob(pattern):
      os.remove(path)
      print(f'removed {path}')


def train_one_epoch(model, dataloader, optimizer, device, mask_prompt: bool):
  model.train()
  train_loss = 0.0
  num_batches = 0

  for batch in tqdm(dataloader, desc='train', disable=TQDM_DISABLE):
    b_ids = batch['token_ids'].to(device)
    b_mask = batch['attention_mask'].to(device)

    optimizer.zero_grad()
    logits = model(b_ids, b_mask)
    logits = rearrange(logits[:, :-1].contiguous(), 'b t d -> (b t) d')
    labels = b_ids[:, 1:].clone()

    if mask_prompt:
      labels = mask_labels_from_start(labels, b_mask, batch['loss_starts'])
    else:
      labels[b_mask[:, 1:] == 0] = -100

    loss = F.cross_entropy(
        logits, labels.flatten(), ignore_index=-100, reduction='mean'
    )
    loss.backward()
    optimizer.step()

    train_loss += loss.item()
    num_batches += 1

  return train_loss / max(num_batches, 1)


def train(args):
  device = torch.device('cuda') if args.use_gpu else torch.device('cpu')

  if args.fresh:
    cleanup_experiment_checkpoints(args.checkpoint_tag)

  mask_target = args.mask_target if args.mask_prompt else None
  dataset = ReasoningDataset(
      args.train_path,
      mask_prompt=args.mask_prompt,
      mask_target=mask_target,
  )
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
  if args.resume:
    start_epoch, latest_epoch = resolve_training_start(model, args)
  elif args.init_checkpoint:
    saved = torch.load(args.init_checkpoint, weights_only=False)
    model.load_state_dict(saved['model'])
    start_epoch, latest_epoch = 0, -1
    print(f'Initialized weights from {args.init_checkpoint}; starting at epoch 0')
  else:
    start_epoch, latest_epoch = 0, -1
    print('Training from pretrained GPT-2 (no checkpoint resume).')

  optimizer = AdamW(model.parameters(), lr=args.lr)

  best_acc = -1.0
  best_epoch = -1
  patience_left = args.patience
  history = []
  finished_epoch = start_epoch - 1

  print(
      f"Experiment tag={args.checkpoint_tag} "
      f"train={args.train_path} examples={len(dataset)} "
      f"epochs={args.epochs} eval_every={args.eval_every}"
  )

  for epoch in range(start_epoch, args.epochs):
    t0 = time.time()
    train_loss = train_one_epoch(model, dataloader, optimizer, device, args.mask_prompt)
    elapsed = time.time() - t0
    finished_epoch = epoch
    print(f'Epoch {epoch}: train_loss={train_loss:.4f} ({elapsed:.0f}s)')

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
      metrics['epoch'] = epoch
      metrics['train_loss'] = train_loss
      metrics['checkpoint_tag'] = args.checkpoint_tag
      history.append(metrics)

      met_file = metrics_path(args.checkpoint_tag, epoch)
      with open(met_file, 'w', encoding='utf-8') as f:
        json.dump(metrics, f, indent=2)

      acc = metrics['exact_accuracy']
      print(
          f'  eval epoch {epoch}: exact_accuracy={acc:.3f} '
          f'format_valid={metrics["format_valid_rate"]:.3f} '
          f'no_answer={metrics["no_answer_rate"]:.3f}'
      )

      if acc > best_acc:
        best_acc = acc
        best_epoch = epoch
        patience_left = args.patience
        best_path = best_checkpoint_path(args.checkpoint_tag)
        save_model(model, optimizer, args, best_path)
        print(f'  ** new best -> {best_path} (epoch {epoch}, acc={acc:.3f})')
      else:
        patience_left -= 1
        print(f'  no improvement (patience left: {patience_left})')

      if patience_left <= 0:
        print(f'Early stopping at epoch {epoch} (no improvement for {args.patience} evals).')
        break

    prune_old_checkpoints(epoch, args)
    save_model(model, optimizer, args, checkpoint_path(epoch, args))

  summary = {
      'checkpoint_tag': args.checkpoint_tag,
      'train_path': args.train_path,
      'init_checkpoint': getattr(args, 'init_checkpoint', None),
      'best_epoch': best_epoch,
      'best_exact_accuracy': best_acc,
      'best_checkpoint': best_checkpoint_path(args.checkpoint_tag),
      'history': history,
      'finished_epoch': finished_epoch,
  }
  with open(summary_path(args.checkpoint_tag), 'w', encoding='utf-8') as f:
    json.dump(summary, f, indent=2)

  print('\n=== Done ===')
  print(json.dumps(summary, indent=2))
  print(f'Summary -> {summary_path(args.checkpoint_tag)}')


def apply_experiment_preset(args):
  if not args.experiment:
    return args
  if args.experiment not in EXPERIMENT_CONFIGS:
    raise ValueError(f'Unknown experiment {args.experiment!r}; choose {list(EXPERIMENT_CONFIGS)}')

  cfg = EXPERIMENT_CONFIGS[args.experiment]
  args.train_path = cfg['train_path']
  args.checkpoint_tag = cfg['checkpoint_tag']
  args.epochs = cfg['epochs']
  args.eval_every = cfg['eval_every']
  args.patience = cfg['patience']
  args.lr = cfg['lr']
  if cfg.get('dev_path'):
    args.dev_path = cfg['dev_path']
  if cfg.get('mask_target'):
    args.mask_target = cfg['mask_target']
  return args


def get_args():
  parser = argparse.ArgumentParser()
  parser.add_argument(
      '--experiment',
      choices=list(EXPERIMENT_CONFIGS.keys()),
      default=None,
      help='Preset: exp1–exp7, exp10–exp11, exp14–exp15 (see EXPERIMENT_CONFIGS).',
  )
  parser.add_argument('--train_path', type=str, default=None)
  parser.add_argument('--dev_path', type=str, default=DEV_JSONL)
  parser.add_argument('--epochs', type=int, default=30)
  parser.add_argument('--eval_every', type=int, default=2)
  parser.add_argument('--patience', type=int, default=5)
  parser.add_argument('--lr', type=float, default=5e-6)
  parser.add_argument('--batch_size', type=int, default=8)
  parser.add_argument('--use_gpu', action='store_true')
  parser.add_argument('--seed', type=int, default=11711)
  parser.add_argument('--mask_prompt', action='store_true', default=True)
  parser.add_argument('--no_mask_prompt', action='store_true')
  parser.add_argument('--mask_target', type=str, default='reasoning',
                      choices=['reasoning', 'entities', 'entities_reasoning', 'auto'])
  parser.add_argument('--fresh', action='store_true', default=True)
  parser.add_argument('--no_fresh', action='store_true')
  parser.add_argument('--resume', action='store_true')
  parser.add_argument('--temperature', type=float, default=0.7)
  parser.add_argument('--top_p', type=float, default=0.9)
  parser.add_argument('--max_new_tokens', type=int, default=256)
  parser.add_argument('--greedy', action='store_true')
  parser.add_argument('--eval_limit', type=int, default=0)
  parser.add_argument('--model_size', default='gpt2',
                      choices=['gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'])
  add_checkpoint_args(parser, default_tag='exp1_gsm8k')
  return parser.parse_args()


if __name__ == '__main__':
  args = get_args()
  args = apply_experiment_preset(args)
  if args.no_mask_prompt:
    args.mask_prompt = False
  if args.no_fresh:
    args.fresh = False
  if args.experiment is None and args.train_path is None:
    raise SystemExit('Provide --experiment or --train_path')
  if args.train_path is None:
    args.train_path = EXPERIMENT_CONFIGS[args.experiment]['train_path']
  seed_everything(args.seed)
  train(args)
