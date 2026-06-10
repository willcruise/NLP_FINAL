#!/usr/bin/env python3
"""
Build the "triple" unified-format training data that combines the three
chain-of-thought error remedies (Wei et al., 2022, Appendix D.2) in one block:

  * Entities:  bind words -> symbols          -> fixes "symbol mapping error"
  * Plan:      enumerate every solution step   -> fixes "one step missing error"
  * Reasoning: verifiable <<expr=result>> calc -> fixes "calculator error"
               (PEMDAS arithmetic is supplied by the Stage-0 arith init checkpoint)

The transform is fully deterministic (no LLM): the Plan header is derived from
the calculator tags already present in each example's Reasoning, and existing
Entities blocks are preserved. This matches the project's no-LLM augmentation
philosophy and keeps dev splits untouched.

Output block format (id + <|endoftext|> added by writer):

  Question: ...

  Entities:            (only if the source block had one)
  - ...

  Reasoning:
  Plan: Solve in N steps. (1) add; (2) subtract; then give the final answer.
  ... <<expr=result>>result
  #### answer

  The skeleton Plan (number-free) is the FIRST line inside Reasoning so it is
  (a) covered by the default 'reasoning' loss mask and (b) generated at inference,
  since the eval harness prompts with 'Question: ...\\n\\nReasoning:\\n'.

Usage:
  python prepare_triple_data.py                 # build all curriculum files
  python prepare_triple_data.py --only stage1   # just the MultiArith Stage-1 file
"""

from __future__ import annotations

import argparse
import os
import random
import re

DATA_DIR = os.path.join('data', 'triple')

# Stage-1 source: the original yoonBot sh-branch plan+numaug MultiArith data
# (Plan line lives inside Reasoning, with operand numbers — used verbatim).
GITHUB_MA_PLAN = os.path.join('data', 'multiarith_sft_train_plan_aug.txt')

# Each calc tag looks like <<expr=result>>result ; we only need the lhs expr.
CALC_TAG_RE = re.compile(r'<<([^=>]+)=[^>]*>>')
OP_VERB = {'+': 'add', '-': 'subtract', '*': 'multiply', '/': 'divide'}


def load_blocks(path: str) -> list:
  """Return list of block bodies (leading id line and <|endoftext|> stripped)."""
  with open(path, 'r', encoding='utf-8') as f:
    text = f.read()
  raw = [b.strip() for b in text.split('<|endoftext|>') if b.strip()]
  bodies = []
  for block in raw:
    # Drop a leading numeric id line if present.
    block = re.sub(r'^\s*\d+\s*\n+', '', block, count=1)
    idx = block.find('Question:')
    if idx == -1:
      continue
    bodies.append(block[idx:].strip())
  return bodies


def _step_verb(expr: str) -> str:
  """Map an arithmetic expression to a short natural-language verb phrase."""
  ops = [c for c in expr if c in OP_VERB]
  if not ops:
    return 'compute'
  verbs = []
  for op in ops:
    verb = OP_VERB[op]
    if verb not in verbs:
      verbs.append(verb)
  return ' and '.join(verbs)


def derive_plan(reasoning: str) -> str:
  """Deterministically build a Plan header from the reasoning's calc tags."""
  exprs = CALC_TAG_RE.findall(reasoning)
  if not exprs:
    # Fallback: count non-empty reasoning lines (excluding the answer line).
    lines = [ln for ln in reasoning.splitlines() if ln.strip()
             and not ln.strip().startswith('####')]
    n = max(1, len(lines))
    steps = '; '.join(f'({i + 1}) compute' for i in range(n))
    unit = 'step' if n == 1 else 'steps'
    return f'Plan: Solve in {n} {unit}. {steps}; then give the final answer.'

  steps = []
  for i, expr in enumerate(exprs):
    steps.append(f'({i + 1}) {_step_verb(expr)}')
  n = len(exprs)
  unit = 'step' if n == 1 else 'steps'
  return f'Plan: Solve in {n} {unit}. {"; ".join(steps)}; then give the final answer.'


def parse_body(body: str) -> dict:
  """Split a block body into question / entities / reasoning / answer."""
  q_match = re.search(r'Question:\s*(.*?)(?=\n\s*(?:Entities:|Plan:|Reasoning:)|\Z)',
                      body, flags=re.DOTALL)
  question = q_match.group(1).strip() if q_match else ''

  ent_match = re.search(r'Entities:\s*(.*?)(?=\n\s*(?:Plan:|Reasoning:)|\Z)',
                        body, flags=re.DOTALL)
  entities = ent_match.group(1).strip() if ent_match else None

  reas_match = re.search(r'Reasoning:\s*(.*)\Z', body, flags=re.DOTALL)
  reasoning_full = reas_match.group(1).strip() if reas_match else ''

  ans_match = re.search(r'####\s*([^\n]+)', reasoning_full)
  answer = ans_match.group(1).strip() if ans_match else None
  reasoning = re.split(r'\n?####', reasoning_full, maxsplit=1)[0].strip()

  return {
      'question': question,
      'entities': entities,
      'reasoning': reasoning,
      'answer': answer,
  }


def _strip_inline_plan(reasoning: str) -> str:
  """Remove a leading inline 'Plan: ...' line (yoonBot GitHub MA format)."""
  return re.sub(r'^\s*Plan:[^\n]*\n?', '', reasoning, count=1).strip()


def transform_block(body: str, *, keep_entities: bool = True) -> str | None:
  """Normalize one block to the unified format:

  Question -> [Entities] -> Reasoning: { Plan line, then steps } -> #### answer.
  The skeleton Plan (number-free) is the FIRST line inside the Reasoning block
  (same position as yoonBot's number-ful Plan) so training loss and greedy eval
  stay aligned.  Any pre-existing inline 'Plan:' line is stripped first.
  """
  parsed = parse_body(body)
  reasoning = _strip_inline_plan(parsed['reasoning'])
  if not parsed['question'] or not reasoning or parsed['answer'] is None:
    return None

  plan = derive_plan(reasoning)
  parts = [f"Question: {parsed['question']}", '']
  if keep_entities and parsed['entities']:
    parts += ['Entities:', parsed['entities'], '']
  parts += ['Reasoning:', plan, reasoning, f"#### {parsed['answer']}"]
  return '\n'.join(parts)


def transform_file(src: str, *, keep_entities: bool = True) -> list:
  if not os.path.exists(src):
    raise FileNotFoundError(f'Missing source: {src}')
  out = []
  dropped = 0
  for body in load_blocks(src):
    block = transform_block(body, keep_entities=keep_entities)
    if block is None:
      dropped += 1
      continue
    out.append(block)
  print(f'  {src}: {len(out)} blocks transformed' +
        (f' ({dropped} skipped)' if dropped else ''))
  return out


def subsample(blocks: list, n: int, seed: int) -> list:
  if n <= 0 or len(blocks) <= n:
    return blocks
  return random.Random(seed).sample(blocks, n)


def write_blocks(blocks: list, out_path: str, shuffle_seed: int | None = None):
  if shuffle_seed is not None:
    blocks = blocks[:]
    random.Random(shuffle_seed).shuffle(blocks)
  os.makedirs(os.path.dirname(out_path) or '.', exist_ok=True)
  with open(out_path, 'w', encoding='utf-8') as f:
    for i, block in enumerate(blocks):
      f.write(f'{i}\n\n{block}\n\n<|endoftext|>\n\n')
  print(f'  -> wrote {len(blocks)} blocks to {out_path}')


def build_stage1(seed: int):
  """Stage 1: yoonBot sh-branch MultiArith data, plan normalized to unified format."""
  print('[stage1] MultiArith plan (GitHub data, plan format normalized)')
  blocks = transform_file(GITHUB_MA_PLAN)
  write_blocks(blocks, os.path.join(DATA_DIR, 'ma_plan_aug.txt'), shuffle_seed=seed)
  return blocks


def build_stage2(seed: int, ma_anchor: int):
  """Stage 2: GSM8K+Plan and entity+Plan, plus a MultiArith plan anchor.

  All sources go through the same transform, so every block shares one plan
  convention (skeleton, number-free, Plan as first line inside Reasoning).
  """
  print('[stage2] component files')
  gsm8k = transform_file('data/gsm8k_sft_train.txt')
  entity = transform_file('data/entity_stage2_train.txt')  # has Entities + Plan
  ma = transform_file(GITHUB_MA_PLAN)
  ma_sub = subsample(ma, ma_anchor, seed + 7)

  write_blocks(gsm8k, os.path.join(DATA_DIR, 'gsm8k_plan.txt'), shuffle_seed=seed)
  write_blocks(entity, os.path.join(DATA_DIR, 'entity_plan.txt'), shuffle_seed=seed)

  full = gsm8k + entity + ma_sub
  print(f'[stage2] FULL mix = gsm8k({len(gsm8k)}) + entity({len(entity)}) '
        f'+ ma_anchor({len(ma_sub)}) = {len(full)}')
  write_blocks(full, os.path.join(DATA_DIR, 'stage2_full.txt'), shuffle_seed=seed)


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('--only', choices=['stage1', 'stage2'], default=None,
                      help='Build only one curriculum stage (default: both).')
  parser.add_argument('--seed', type=int, default=11711)
  parser.add_argument('--ma_anchor', type=int, default=600,
                      help='How many MultiArith plan blocks to mix into Stage 2 '
                           'as a plan-format anchor.')
  args = parser.parse_args()

  os.makedirs(DATA_DIR, exist_ok=True)
  if args.only in (None, 'stage1'):
    build_stage1(args.seed)
  if args.only in (None, 'stage2'):
    build_stage2(args.seed, args.ma_anchor)
  print('Done.')


if __name__ == '__main__':
  main()
