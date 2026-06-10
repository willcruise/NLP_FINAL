#!/usr/bin/env python3
"""
Build the 2×2 plan-position ablation data (same MA 3061 problems, four formats).

Confound in the original M2 vs M3 comparison: skeleton plan was OUTSIDE Reasoning
while number-ful plan was INSIDE Reasoning (first line after Reasoning:\\n).
This script isolates position vs numbers:

  Arm   plan text   plan position
  P_SO  skeleton    outside Reasoning   (= old M2)
  P_SI  skeleton    inside  Reasoning   (NEW — fair skeleton baseline)
  P_NO  number-ful  outside Reasoning   (NEW)
  P_NI  number-ful  inside  Reasoning   (= old M3, GitHub file as-is)

Source: data/multiarith_sft_train_plan_aug.txt (yoonBot number-ful inline).

Usage:
  python3 prepare_plan_position_ablation.py
"""

from __future__ import annotations

import argparse
import os
import random
import re

from prepare_triple_data import (
    GITHUB_MA_PLAN,
    derive_plan,
    load_blocks,
    parse_body,
    write_blocks,
    _strip_inline_plan,
)

OUT_DIR = os.path.join('data', 'ma_ablation')

INLINE_PLAN_RE = re.compile(r'^\s*(Plan:[^\n]*)', re.MULTILINE)


def _extract_inline_plan(reasoning: str) -> str | None:
  m = re.match(r'^\s*(Plan:[^\n]*)', reasoning)
  return m.group(1).strip() if m else None


def to_skeleton_outside(body: str) -> str | None:
  """Skeleton plan as a separate block before Reasoning (old M2 format)."""
  parsed = parse_body(body)
  reasoning = _strip_inline_plan(parsed['reasoning'])
  if not parsed['question'] or not reasoning or parsed['answer'] is None:
    return None
  plan = derive_plan(reasoning)
  parts = [
      f"Question: {parsed['question']}", '',
      plan, '',
      'Reasoning:', reasoning, f"#### {parsed['answer']}",
  ]
  return '\n'.join(parts)


def to_skeleton_inline(body: str) -> str | None:
  """Skeleton plan as the first line inside Reasoning (matches eval prompt)."""
  parsed = parse_body(body)
  reasoning = _strip_inline_plan(parsed['reasoning'])
  if not parsed['question'] or not reasoning or parsed['answer'] is None:
    return None
  plan = derive_plan(reasoning)
  parts = [
      f"Question: {parsed['question']}", '',
      'Reasoning:', plan, reasoning, f"#### {parsed['answer']}",
  ]
  return '\n'.join(parts)


def to_number_outside(body: str) -> str | None:
  """Keep GitHub's number-ful plan text but move it outside Reasoning."""
  parsed = parse_body(body)
  inline_plan = _extract_inline_plan(parsed['reasoning'])
  reasoning = _strip_inline_plan(parsed['reasoning'])
  if not parsed['question'] or not inline_plan or not reasoning or parsed['answer'] is None:
    return None
  parts = [
      f"Question: {parsed['question']}", '',
      inline_plan, '',
      'Reasoning:', reasoning, f"#### {parsed['answer']}",
  ]
  return '\n'.join(parts)


def build_all(src: str, seed: int):
  os.makedirs(OUT_DIR, exist_ok=True)
  specs = {
      'plan_skel_outside': to_skeleton_outside,
      'plan_skel_inline': to_skeleton_inline,
      'plan_num_outside': to_number_outside,
  }
  for name, fn in specs.items():
    blocks = []
    dropped = 0
    for body in load_blocks(src):
      block = fn(body)
      if block is None:
        dropped += 1
      else:
        blocks.append(block)
    out = os.path.join(OUT_DIR, f'{name}.txt')
    write_blocks(blocks, out, shuffle_seed=seed)
    print(f'  {name}: {len(blocks)} blocks' +
          (f' ({dropped} skipped)' if dropped else ''))

  # P_NI reuses the GitHub file verbatim — just record the path.
  print(f'  plan_num_inline: use {src} ({len(load_blocks(src))} blocks, unchanged)')


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('--seed', type=int, default=11711)
  parser.add_argument('--source', default=GITHUB_MA_PLAN)
  args = parser.parse_args()
  print(f'Building plan-position ablation from {args.source}')
  build_all(args.source, args.seed)
  print('Done.')


if __name__ == '__main__':
  main()
