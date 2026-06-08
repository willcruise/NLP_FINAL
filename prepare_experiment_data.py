#!/usr/bin/env python3
"""
Build merged training files for from-scratch SFT experiments.

Experiments:
  exp1  GSM8K only
  exp2  GSM8K + MultiArith (clean train split)
  exp3  GSM8K + MultiArith + entity stage-2
  exp4  (uses exp3 file — arithmetic curriculum applied at train time)
  exp5  GSM8K + MultiArith augmented + entity
  exp6  exp3 mix + 3000 subsampled PEMDAS arithmetic examples
  exp7  GSM8K + entity + subsampled MultiArith (arith curriculum at train time)
  exp10 exp4-style arith curriculum → GSM8K + entity + subsampled MA aug (best on GSM8K dev)
  exp11_ma  aug MA subsample only (for sequential exp11 MA stage)
  exp11     sequential curriculum: arith → entity → MA → GSM8K (see scripts/run_exp11.sh)

Outputs under data/experiments/

Usage:
  python prepare_experiment_data.py
  python prepare_experiment_data.py --experiments exp5,exp6
"""

import argparse
import os
import random
import re

DATA_DIR = os.path.join('data', 'experiments')

EXPERIMENTS = {
    'exp1': {
        'out': 'exp1_gsm8k_train.txt',
        'sources': ['data/gsm8k_sft_train.txt'],
    },
    'exp2': {
        'out': 'exp2_gsm8k_multiarith_train.txt',
        'sources': [
            'data/gsm8k_sft_train.txt',
            'data/multiarith_sft_train.txt',
        ],
    },
    'exp3': {
        'out': 'exp3_gsm8k_multiarith_entity_train.txt',
        'sources': [
            'data/gsm8k_sft_train.txt',
            'data/multiarith_sft_train.txt',
            'data/entity_stage2_train.txt',
        ],
    },
    'exp5': {
        'out': 'exp5_gsm8k_ma_aug_entity_train.txt',
        'sources': [
            'data/gsm8k_sft_train.txt',
            'data/multiarith_sft_train_aug.txt',
            'data/entity_stage2_train.txt',
        ],
    },
    'exp6': {
        'out': 'exp6_gsm8k_ma_ent_arith_train.txt',
        'sources': [
            'data/gsm8k_sft_train.txt',
            'data/multiarith_sft_train.txt',
            'data/entity_stage2_train.txt',
        ],
        'arithmetic_path': 'data/arithmetic_pretrain.txt',
        'arithmetic_subsample': 3000,
    },
    'exp7': {
        'out': 'exp7_gsm8k_ma_sub_ent_train.txt',
        'sources': [
            'data/gsm8k_sft_train.txt',
            'data/entity_stage2_train.txt',
        ],
        'subsampled_sources': [
            {'path': 'data/multiarith_sft_train.txt', 'n': 150, 'seed_offset': 2},
        ],
    },
    'exp10': {
        'out': 'exp10_gsm8k_ma_aug_sub_ent_train.txt',
        'sources': [
            'data/gsm8k_sft_train.txt',
            'data/entity_stage2_train.txt',
        ],
        'subsampled_sources': [
            {
                'path': 'data/multiarith_sft_train_aug.txt',
                'n': 600,
                'seed_offset': 3,
            },
        ],
    },
    'exp11_ma': {
        'out': 'exp11_ma_aug_sub_train.txt',
        'sources': [],
        'subsampled_sources': [
            {
                'path': 'data/multiarith_sft_train_aug.txt',
                'n': 600,
                'seed_offset': 3,
            },
        ],
    },
}


def load_blocks(path: str) -> list:
  with open(path, 'r', encoding='utf-8') as f:
    text = f.read()
  if '<|endoftext|>' in text:
    blocks = [b.strip() for b in text.split('<|endoftext|>') if b.strip()]
  else:
    blocks = re.split(r'\n\s*\d+\s*\n', text)
    blocks = [b.strip() for b in blocks if b.strip()]
  return blocks


def subsample_blocks(blocks: list, max_blocks: int, seed: int) -> list:
  if len(blocks) <= max_blocks:
    return blocks
  rng = random.Random(seed)
  return rng.sample(blocks, max_blocks)


def write_blocks(blocks, out_path: str, shuffle_seed=None):
  if shuffle_seed is not None:
    rng = random.Random(shuffle_seed)
    blocks = blocks[:]
    rng.shuffle(blocks)

  os.makedirs(os.path.dirname(out_path), exist_ok=True)
  with open(out_path, 'w', encoding='utf-8') as f:
    for i, block in enumerate(blocks):
      f.write(f'{i}\n\n{block}\n\n<|endoftext|>\n\n')


def apply_env_overrides(name: str, cfg: dict) -> dict:
  if name in ('exp10', 'exp11_ma'):
    n = os.environ.get('MA_AUG_SUBSAMPLE')
    if n:
      subs = [dict(item) for item in cfg.get('subsampled_sources', [])]
      if subs:
        subs[0]['n'] = int(n)
      cfg = {**cfg, 'subsampled_sources': subs}
  return cfg


def build_experiment(name: str, cfg: dict, seed: int):
  blocks = []
  for src in cfg['sources']:
    if not os.path.exists(src):
      raise FileNotFoundError(f'Missing source for {name}: {src}')
    part = load_blocks(src)
    print(f'{name}: {len(part):4d} blocks from {src}')
    blocks.extend(part)

  for item in cfg.get('subsampled_sources', []):
    src = item['path']
    if not os.path.exists(src):
      raise FileNotFoundError(f'Missing source for {name}: {src}')
    part = load_blocks(src)
    n = item['n']
    offset = item.get('seed_offset', 0)
    part = subsample_blocks(part, n, seed + offset)
    print(f'{name}: {len(part):4d} blocks subsampled from {src} (n={n})')
    blocks.extend(part)

  arith_path = cfg.get('arithmetic_path')
  if arith_path:
    if not os.path.exists(arith_path):
      raise FileNotFoundError(
          f'Missing {arith_path} for {name}. Run: '
          f'python prepare_arithmetic.py --num_examples 20000'
      )
    arith = load_blocks(arith_path)
    n = cfg.get('arithmetic_subsample', 3000)
    arith = subsample_blocks(arith, n, seed + 1)
    print(f'{name}: {len(arith):4d} blocks subsampled from {arith_path}')
    blocks.extend(arith)

  out_path = os.path.join(DATA_DIR, cfg['out'])
  write_blocks(blocks, out_path, shuffle_seed=seed)
  print(f'{name}: wrote {len(blocks)} total -> {out_path}\n')


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument(
      '--experiments',
      type=str,
      default='exp1,exp2,exp3',
      help='Comma-separated experiment ids to build (exp4 uses exp3 file).',
  )
  parser.add_argument('--seed', type=int, default=11711)
  args = parser.parse_args()

  os.makedirs(DATA_DIR, exist_ok=True)
  names = [x.strip() for x in args.experiments.split(',') if x.strip()]

  for name in names:
    if name == 'exp4':
      print('exp4: uses data/experiments/exp3_gsm8k_multiarith_entity_train.txt (no merge step)\n')
      continue
    if name == 'exp7':
      print('exp7: uses data/experiments/exp7_gsm8k_ma_sub_ent_train.txt (arith curriculum at train time)\n')
    if name == 'exp10':
      print('exp10: uses data/experiments/exp10_gsm8k_ma_aug_sub_ent_train.txt (arith curriculum at train time)\n')
    if name == 'exp11_ma':
      print('exp11_ma: MA aug subsample only -> data/experiments/exp11_ma_aug_sub_train.txt\n')
    if name not in EXPERIMENTS:
      raise ValueError(
          f'Unknown experiment {name!r}; choose from {list(EXPERIMENTS)} + exp4 + exp7 + exp10'
      )
    cfg = apply_env_overrides(name, EXPERIMENTS[name])
    build_experiment(name, cfg, args.seed)


if __name__ == '__main__':
  main()
