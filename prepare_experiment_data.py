#!/usr/bin/env python3
"""
Build merged training files for the three from-scratch SFT experiments.

Experiments:
  exp1  GSM8K only
  exp2  GSM8K + MultiArith (clean train split, no dev leakage)
  exp3  GSM8K + MultiArith + entity stage-2 data (Entities + Reasoning)

Outputs (under data/experiments/):
  exp1_gsm8k_train.txt
  exp2_gsm8k_multiarith_train.txt
  exp3_gsm8k_multiarith_entity_train.txt

Usage:
  python prepare_experiment_data.py
"""

import os
import random
import re

DATA_DIR = os.path.join('data', 'experiments')

EXPERIMENTS = {
    'exp1': {
        'out': 'exp1_gsm8k_train.txt',
        'sources': [
            'data/gsm8k_sft_train.txt',
        ],
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


def write_blocks(blocks, out_path: str, shuffle_seed=None):
  if shuffle_seed is not None:
    rng = random.Random(shuffle_seed)
    blocks = blocks[:]
    rng.shuffle(blocks)

  os.makedirs(os.path.dirname(out_path), exist_ok=True)
  with open(out_path, 'w', encoding='utf-8') as f:
    for i, block in enumerate(blocks):
      f.write(f'{i}\n\n{block}\n\n<|endoftext|>\n\n')


def main():
  os.makedirs(DATA_DIR, exist_ok=True)
  seed = 11711

  for name, cfg in EXPERIMENTS.items():
    blocks = []
    for src in cfg['sources']:
      if not os.path.exists(src):
        raise FileNotFoundError(f'Missing source for {name}: {src}')
      part = load_blocks(src)
      print(f'{name}: {len(part):4d} blocks from {src}')
      blocks.extend(part)

    out_path = os.path.join(DATA_DIR, cfg['out'])
    write_blocks(blocks, out_path, shuffle_seed=seed)
    print(f'{name}: wrote {len(blocks)} total -> {out_path}\n')


if __name__ == '__main__':
  main()
