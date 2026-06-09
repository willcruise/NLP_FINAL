#!/usr/bin/env python3
"""
Build the merged training files for the MultiArith factor-ablation (M4/M5).

The ablation isolates how much PEMDAS (arith init), entity, and plan each help
on the MultiArith dev set. Most arms reuse existing files; only the entity-mix
arms need merging here:

  M4 (+entity):       multiarith_sft_train_aug.txt   + entity_stage2_train.txt   (no plan)
  M5 (+plan+entity):  data/triple/ma_plan_aug.txt    + data/triple/entity_plan.txt (skeleton plan)

Entity is mixed in as additional training data (deterministic, no LLM), matching
the exp3/exp5 philosophy. Output blocks are renumbered and shuffled.

Usage:
  python3 prepare_ma_ablation.py
"""

from __future__ import annotations

import argparse
import os
import random
import re

OUT_DIR = os.path.join('data', 'ma_ablation')

ARMS = {
    'm4_aug_entity': [
        'data/multiarith_sft_train_aug.txt',
        'data/entity_stage2_train.txt',
    ],
    'm5_plan_entity': [
        'data/triple/ma_plan_aug.txt',
        'data/triple/entity_plan.txt',
    ],
}


def load_blocks(path: str) -> list:
  """Return block bodies (leading id line + <|endoftext|> stripped)."""
  with open(path, 'r', encoding='utf-8') as f:
    text = f.read()
  bodies = []
  for block in text.split('<|endoftext|>'):
    block = block.strip()
    if not block:
      continue
    idx = block.find('Question:')  # drops the leading id line
    if idx == -1:
      continue
    bodies.append(block[idx:].strip())
  return bodies


def write_blocks(blocks: list, out_path: str, seed: int):
  blocks = blocks[:]
  random.Random(seed).shuffle(blocks)
  os.makedirs(os.path.dirname(out_path) or '.', exist_ok=True)
  with open(out_path, 'w', encoding='utf-8') as f:
    for i, block in enumerate(blocks):
      f.write(f'{i}\n\n{block}\n\n<|endoftext|>\n\n')
  print(f'  -> wrote {len(blocks)} blocks to {out_path}')


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('--seed', type=int, default=11711)
  args = parser.parse_args()

  os.makedirs(OUT_DIR, exist_ok=True)
  for name, sources in ARMS.items():
    blocks = []
    for src in sources:
      part = load_blocks(src)
      print(f'{name}: {len(part):4d} blocks from {src}')
      blocks.extend(part)
    write_blocks(blocks, os.path.join(OUT_DIR, f'{name}.txt'), args.seed)
  print('Done.')


if __name__ == '__main__':
  main()
