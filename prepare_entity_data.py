#!/usr/bin/env python3
"""
Synthetic entity-binding word problems for staged CoT training.

Outputs:
  data/entity_stage1_train.txt  — Question + Entities (stage 1: bind numbers to entities)
  data/entity_stage2_train.txt  — Question + Entities + Reasoning + #### (stage 2)
  data/gsm8k_small_held_out_entities.txt — held-out prompts with Entities:\\n (optional)

Usage:
  python prepare_entity_data.py
  python prepare_entity_data.py --num_examples 5000 --held_out_path data/gsm8k_small_held_out.txt
"""

import argparse
import random
import re


NAMES = [
    'Alice', 'Bob', 'Lisa', 'Sandra', 'John', 'Mary', 'Katie', 'James',
    'Emma', 'Noah', 'Mia', 'Liam', 'Zoe', 'Leo', 'Nina', 'Omar',
]


def block_stage1(example_id, question, entities_lines):
  entities = '\n'.join(f'- {line}' for line in entities_lines)
  return (
      f"{example_id}\n\n"
      f"Question: {question}\n\n"
      f"Entities:\n{entities}\n\n"
      f"<|endoftext|>\n\n"
  )


def block_stage2(example_id, question, entities_lines, reasoning, answer):
  entities = '\n'.join(f'- {line}' for line in entities_lines)
  return (
      f"{example_id}\n\n"
      f"Question: {question}\n\n"
      f"Entities:\n{entities}\n\n"
      f"Reasoning:\n{reasoning}\n#### {answer}\n\n"
      f"<|endoftext|>\n\n"
  )


def gen_twice_as_many(rng):
  a, x = rng.choice(NAMES), rng.randint(5, 40)
  b = rng.choice([n for n in NAMES if n != a])
  bx = 2 * x
  total = x + bx
  q = (
      f"{a} has {x} items. {b} has twice as many as {a}. "
      f"How many items do they have in total?"
  )
  entities = [
      f"{a}.count = {x}",
      f"{b}.count = 2 * {a}.count = {bx}",
      f"total = {a}.count + {b}.count = {total}",
  ]
  reasoning = (
      f"{b} has twice as many as {a}, so {b} has <<2*{x}={bx}>>{bx} items.\n"
      f"Together they have {x} + {bx} = <<{x}+{bx}={total}>>{total} items."
  )
  return q, entities, reasoning, total


def gen_three_agent_offset(rng):
  billy, john, mom = 'Billy', 'John', 'their mother'
  b = rng.randint(5, 25)
  j = 2 * b
  m = j + rng.randint(5, 15)
  total = b + j + m
  q = (
      f"If {billy} rode his bike {b} times, {john} rode twice as many times, "
      f"and {mom} rode {m - j} times more than {john}, "
      f"how many times did they ride in total?"
  )
  entities = [
      f"{billy}.rides = {b}",
      f"{john}.rides = 2 * {billy}.rides = {j}",
      f"{mom}.rides = {john}.rides + {m - j} = {m}",
      f"total.rides = {billy}.rides + {john}.rides + {mom}.rides = {total}",
  ]
  reasoning = (
      f"{john} rode twice as many as {billy}: <<2*{b}={j}>>{j}.\n"
      f"{mom} rode {m - j} more than {john}: {j} + {m - j} = <<{j}+{m - j}={m}>>{m}.\n"
      f"Total rides: {b} + {j} + {m} = <<{b}+{j}+{m}={total}>>{total}."
  )
  return q, entities, reasoning, total


def gen_percent_difference(rng):
  p1 = rng.randint(10, 40)
  p2 = rng.randint(p1 + 5, 50)
  total = rng.choice([100, 200, 500, 1000])
  c1 = total * p1 // 100
  c2 = total * p2 // 100
  diff = c2 - c1
  q = (
      f"{p1}% of the vets in a state recommend Puppy Kibble. "
      f"{p2}% recommend Yummy Dog Kibble. "
      f"If there are {total} vets, how many more recommend Yummy than Puppy?"
  )
  entities = [
      f"vets.total = {total}",
      f"PuppyKibble.rate = {p1}%",
      f"YummyKibble.rate = {p2}%",
      f"PuppyKibble.count = {p1}% of {total} = {c1}",
      f"YummyKibble.count = {p2}% of {total} = {c2}",
      f"difference = YummyKibble.count - PuppyKibble.count = {diff}",
  ]
  reasoning = (
      f"Puppy Kibble: {p1}% of {total} = <<{p1}/100*{total}={c1}>>{c1} vets.\n"
      f"Yummy Kibble: {p2}% of {total} = <<{p2}/100*{total}={c2}>>{c2} vets.\n"
      f"Difference: {c2} - {c1} = <<{c2}-{c1}={diff}>>{diff}."
  )
  return q, entities, reasoning, diff


def gen_part_whole_height(rng):
  name = rng.choice(NAMES)
  h = rng.choice([48, 60, 72, 84])
  legs = h // 3
  head = h // 4
  rest = h - legs - head
  q = (
      f"{name}'s legs are 1/3 of her height. Her head is 1/4 of her height. "
      f"She is {h} inches tall. How long is the rest of her body?"
  )
  entities = [
      f"{name}.height = {h} inches",
      f"{name}.legs = 1/3 * height = {legs} inches",
      f"{name}.head = 1/4 * height = {head} inches",
      f"{name}.rest = height - legs - head = {rest} inches",
  ]
  reasoning = (
      f"Legs: <<1/3*{h}={legs}>>{legs} inches.\n"
      f"Head: <<1/4*{h}={head}>>{head} inches.\n"
      f"Rest: {h} - {legs} - {head} = <<{h}-{legs}-{head}={rest}>>{rest} inches."
  )
  return q, entities, reasoning, rest


def gen_collection_chain(rng):
  lisa = rng.choice(NAMES)
  sandra = rng.choice([n for n in NAMES if n != lisa])
  start = rng.randint(8, 20)
  sandra_brought = rng.randint(10, 30)
  cousin = sandra_brought // 5
  mom = 3 * start + rng.randint(5, 12)
  total = start + sandra_brought + cousin + mom
  q = (
      f"{lisa} started with {start} pairs of socks. "
      f"{sandra} brought {sandra_brought} pairs. "
      f"A cousin brought one-fifth as many pairs as {sandra}. "
      f"{lisa}'s mom brought 8 more than three times the number {lisa} started with. "
      f"How many pairs does {lisa} have now?"
  )
  mom_extra = mom - 3 * start
  entities = [
      f"{lisa}.start = {start}",
      f"{sandra}.brought = {sandra_brought}",
      f"cousin.brought = {sandra}.brought / 5 = {cousin}",
      f"{lisa}.mom.brought = 3 * {lisa}.start + {mom_extra} = {mom}",
      f"{lisa}.total = start + sandra + cousin + mom = {total}",
  ]
  reasoning = (
      f"Cousin brought <<{sandra_brought}/5={cousin}>>{cousin} pairs.\n"
      f"Mom brought 3*{start} + {mom_extra} = <<3*{start}+{mom_extra}={mom}>>{mom} pairs.\n"
      f"Total: {start} + {sandra_brought} + {cousin} + {mom} = "
      f"<<{start}+{sandra_brought}+{cousin}+{mom}={total}>>{total} pairs."
  )
  return q, entities, reasoning, total


def gen_rate_times_units(rng):
  name = rng.choice(NAMES)
  hours_per_day = rng.randint(2, 6)
  days = rng.randint(5, 7)
  pct = rng.choice([20, 25, 30, 40])
  weekly = hours_per_day * days
  math_hours = weekly * pct // 100
  q = (
      f"{name} goes to school {hours_per_day} hours a day for {days} days a week. "
      f"She spends {pct}% of that time in math class. "
      f"How many hours per week does she spend in math class?"
  )
  entities = [
      f"{name}.hours_per_day = {hours_per_day}",
      f"{name}.days_per_week = {days}",
      f"{name}.weekly_school_hours = hours_per_day * days = {weekly}",
      f"{name}.math_percent = {pct}%",
      f"{name}.math_hours = {pct}% of weekly_school_hours = {math_hours}",
  ]
  reasoning = (
      f"Weekly school hours: {hours_per_day} * {days} = <<{hours_per_day}*{days}={weekly}>>{weekly}.\n"
      f"Math: {pct}% of {weekly} = <<{pct}/100*{weekly}={math_hours}>>{math_hours} hours."
  )
  return q, entities, reasoning, math_hours


GENERATORS = [
    gen_twice_as_many,
    gen_three_agent_offset,
    gen_percent_difference,
    gen_part_whole_height,
    gen_collection_chain,
    gen_rate_times_units,
]


def write_training_data(num_examples, seed, stage1_path, stage2_path):
  rng = random.Random(seed)
  s1_parts, s2_parts = [], []
  for i in range(num_examples):
    gen = rng.choice(GENERATORS)
    q, entities, reasoning, answer = gen(rng)
    s1_parts.append(block_stage1(i, q, entities))
    s2_parts.append(block_stage2(i, q, entities, reasoning, answer))

  with open(stage1_path, 'w', encoding='utf-8') as f:
    f.writelines(s1_parts)
  with open(stage2_path, 'w', encoding='utf-8') as f:
    f.writelines(s2_parts)
  print(f"Wrote {num_examples} examples -> {stage1_path}")
  print(f"Wrote {num_examples} examples -> {stage2_path}")


def write_entity_held_out(src_path, dst_path):
  """Insert Entities:\\n before Reasoning:\\n in prompt-only held-out file."""
  with open(src_path, 'r', encoding='utf-8') as f:
    text = f.read()

  blocks = re.split(r'\n\s*(\d+)\s*\n', text)
  if blocks and not blocks[0].strip():
    blocks = blocks[1:]

  out = []
  i = 0
  while i < len(blocks) - 1:
    ex_id = blocks[i].strip()
    body = blocks[i + 1].strip()
    i += 2
    if 'Entities:' not in body:
      body = body.replace(
          'Reasoning:\n',
          'Entities:\n\nReasoning:\n',
          1,
      )
    out.append(f"{ex_id}\n\n{body}\n")
  with open(dst_path, 'w', encoding='utf-8') as f:
    f.write('\n'.join(out))
  print(f"Wrote entity held-out prompts -> {dst_path} ({len(out)} examples)")


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('--num_examples', type=int, default=3000)
  parser.add_argument('--seed', type=int, default=11711)
  parser.add_argument('--stage1_path', type=str, default='data/entity_stage1_train.txt')
  parser.add_argument('--stage2_path', type=str, default='data/entity_stage2_train.txt')
  parser.add_argument('--held_out_src', type=str, default='data/gsm8k_small_held_out.txt')
  parser.add_argument('--held_out_path', type=str, default='data/gsm8k_small_held_out_entities.txt')
  args = parser.parse_args()

  write_training_data(args.num_examples, args.seed, args.stage1_path, args.stage2_path)
  write_entity_held_out(args.held_out_src, args.held_out_path)


if __name__ == '__main__':
  main()
