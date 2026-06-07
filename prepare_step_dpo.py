#!/usr/bin/env python3
"""
Build step-level DPO pairs: same prefix, chosen gold step vs planning-focused rejected step.

Inputs:
  data/entity_stage2_train.txt
  data/gsm8k_dpo_source.jsonl

Output:
  data/step_dpo_train.jsonl
    {prompt, chosen, rejected, reject_type, source_id, step_index}

Usage:
  python prepare_step_dpo.py
  python prepare_step_dpo.py --max_rejections_per_step 2
"""

from __future__ import annotations

import argparse
import json
import os
import random

from gsm8k_steps import (
    build_prefix,
    expand_example_to_steps,
    generate_planning_rejections,
    parse_entity_stage2_block,
    split_reasoning_steps,
)

DEFAULT_ENTITY = os.path.join("data", "entity_stage2_train.txt")
DEFAULT_DPO = os.path.join("data", "gsm8k_dpo_source.jsonl")
DEFAULT_OUT = os.path.join("data", "step_dpo_train.jsonl")


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


def build_dpo_records(example: dict, rng: random.Random, max_rejections_per_step: int) -> list[dict]:
  question = example["question"]
  entities = example.get("entities")
  steps = split_reasoning_steps(example["reasoning"], example.get("gold"))
  records = []

  prior: list[str] = []
  for step_idx, step in enumerate(steps):
    prefix = build_prefix(question, entities, prior)
    chosen = step if step.endswith("\n") else f"{step}\n"
    rejections = generate_planning_rejections(chosen, question, prior, rng)
    seen = {chosen.rstrip("\n")}
    added = 0
    for rejected, reject_type in rejections:
      key = rejected.rstrip("\n")
      if key in seen or key == chosen.rstrip("\n"):
        continue
      seen.add(key)
      records.append(
          {
              "source_id": example["source_id"],
              "step_index": step_idx,
              "prompt": prefix,
              "chosen": chosen,
              "rejected": rejected,
              "reject_type": reject_type,
          }
      )
      added += 1
      if added >= max_rejections_per_step:
        break
    prior.append(step)

  return records


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument("--entity_path", default=DEFAULT_ENTITY)
  parser.add_argument("--dpo_jsonl", default=DEFAULT_DPO)
  parser.add_argument("--output", default=DEFAULT_OUT)
  parser.add_argument("--include_dpo", action="store_true", default=True)
  parser.add_argument("--no_dpo", action="store_true")
  parser.add_argument("--max_rejections_per_step", type=int, default=2)
  parser.add_argument("--shuffle_seed", type=int, default=11711)
  args = parser.parse_args()

  if args.no_dpo:
    args.include_dpo = False

  examples = load_entity_blocks(args.entity_path)
  if args.include_dpo and os.path.isfile(args.dpo_jsonl):
    examples.extend(load_dpo_jsonl(args.dpo_jsonl))

  rng = random.Random(args.shuffle_seed)
  records = []
  for ex in examples:
    records.extend(build_dpo_records(ex, rng, args.max_rejections_per_step))

  rng.shuffle(records)
  os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
  with open(args.output, "w", encoding="utf-8") as f:
    for rec in records:
      f.write(json.dumps(rec, ensure_ascii=False) + "\n")

  by_type = {}
  for r in records:
    by_type[r["reject_type"]] = by_type.get(r["reject_type"], 0) + 1
  print(f"Wrote {len(records)} DPO pairs -> {args.output}")
  print("By reject_type:", by_type)


if __name__ == "__main__":
  main()
