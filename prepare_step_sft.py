#!/usr/bin/env python3
"""
Build step-conditioned SFT data: prefix -> next reasoning line.

Inputs:
  data/entity_stage2_train.txt
  data/gsm8k_dpo_source.jsonl  (optional extra GSM8K w/o entities)

Output:
  data/step_sft_train.jsonl   {prefix, target, question, step_index, source_id}

Usage:
  python prepare_step_sft.py
  python prepare_step_sft.py --entity_path data/entity_stage2_train.txt --dpo_jsonl data/gsm8k_dpo_source.jsonl
"""

from __future__ import annotations

import argparse
import json
import os
import random

from gsm8k_steps import expand_example_to_steps, parse_entity_stage2_block

DEFAULT_ENTITY = os.path.join("data", "entity_stage2_train.txt")
DEFAULT_DPO = os.path.join("data", "gsm8k_dpo_source.jsonl")
DEFAULT_OUT = os.path.join("data", "step_sft_train.jsonl")


def load_entity_blocks(path: str) -> list[dict]:
  with open(path, encoding="utf-8") as f:
    text = f.read()
  blocks = [b.strip() for b in text.split("<|endoftext|>") if b.strip()]
  examples = []
  for i, block in enumerate(blocks):
    parsed = parse_entity_stage2_block(block)
    if parsed:
      parsed["source_id"] = f"entity_{i}"
      examples.append(parsed)
  return examples


def load_dpo_jsonl(path: str) -> list[dict]:
  examples = []
  with open(path, encoding="utf-8") as f:
    for line in f:
      rec = json.loads(line)
      examples.append(
          {
              "question": rec["question"],
              "entities": None,
              "reasoning": rec["gold_reasoning"],
              "gold": rec["gold_answer"],
              "source_id": f"dpo_{rec['id']}",
          }
      )
  return examples


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument("--entity_path", default=DEFAULT_ENTITY)
  parser.add_argument("--dpo_jsonl", default=DEFAULT_DPO)
  parser.add_argument("--output", default=DEFAULT_OUT)
  parser.add_argument("--include_dpo", action="store_true", default=True)
  parser.add_argument("--no_dpo", action="store_true")
  parser.add_argument("--shuffle_seed", type=int, default=11711)
  args = parser.parse_args()

  if args.no_dpo:
    args.include_dpo = False

  examples = load_entity_blocks(args.entity_path)
  if args.include_dpo and os.path.isfile(args.dpo_jsonl):
    examples.extend(load_dpo_jsonl(args.dpo_jsonl))

  records = []
  for ex in examples:
    pairs = expand_example_to_steps(ex)
    for step_idx, pair in enumerate(pairs):
      records.append(
          {
              "source_id": ex["source_id"],
              "step_index": step_idx,
              "prefix": pair["prefix"],
              "target": pair["target"],
              "question": pair["question"],
          }
      )

  rng = random.Random(args.shuffle_seed)
  rng.shuffle(records)

  os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
  with open(args.output, "w", encoding="utf-8") as f:
    for rec in records:
      f.write(json.dumps(rec, ensure_ascii=False) + "\n")

  n_problems = len({r["source_id"] for r in records})
  print(f"Wrote {len(records)} step examples from {n_problems} problems -> {args.output}")


if __name__ == "__main__":
  main()
