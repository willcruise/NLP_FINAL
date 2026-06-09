#!/usr/bin/env python3
"""
Entity-binding word-problem generation for staged CoT training.

By default this script uses a strong LLM (OpenAI-compatible chat completions API)
to generate high-quality, diverse examples with explicit Entities bindings.

Profiles:
  short     — 2-4 arithmetic steps (legacy entity_stage2_train.txt)
  planning  — 5-7 GSM8K-style planning steps (entity_planning_stage2_train.txt)

Outputs:
  data/entity_stage1_train.txt           — short profile stage 1
  data/entity_stage2_train.txt           — short profile stage 2
  data/entity_planning_stage1_train.txt  — planning profile stage 1
  data/entity_planning_stage2_train.txt  — planning profile stage 2
  data/gsm8k_small_held_out_entities.txt — held-out prompts with Entities:\\n (optional)

Examples:
  # Planning entity data for exp15 (strong LLM, 3000 examples)
  OPENAI_API_KEY=... python prepare_entity_data.py --profile planning --num_examples 3000

  # Resume after rate-limit interrupt (appends from existing file)
  OPENAI_API_KEY=... python prepare_entity_data.py --profile planning --resume --request_delay_s 1.5

  # Legacy short entity
  OPENAI_API_KEY=... python prepare_entity_data.py --profile short --num_examples 500

  # Synthetic fallback mode
  python prepare_entity_data.py --generator synthetic --num_examples 3000
"""

import argparse
import json
import os
import random
import re
import time
from typing import List, Optional, Tuple

import requests


NAMES = [
    'Alice', 'Bob', 'Lisa', 'Sandra', 'John', 'Mary', 'Katie', 'James',
    'Emma', 'Noah', 'Mia', 'Liam', 'Zoe', 'Leo', 'Nina', 'Omar',
]

LLM_SHORT_SYSTEM_PROMPT = """You generate high-quality grade-school arithmetic word problems.
Return ONLY valid JSON with this schema:
{
  "question": string,
  "entities": [string, ...],
  "reasoning": string,
  "answer": integer
}

Rules:
- Problem must require 2-4 arithmetic steps.
- Use realistic narrative with entity tracking (people/objects/rates/percent/part-whole).
- "entities" must bind all key quantities and relationships in concise symbolic style.
- "reasoning" must be correct and include GSM8K-style calculator tags like <<a+b=c>>c.
- Do NOT include "####" in reasoning.
- answer must be an integer matching reasoning.
- No markdown, no extra text, JSON only.
"""

LLM_PLANNING_SYSTEM_PROMPT = """You generate GSM8K-style grade-school math word problems that require multi-step PLANNING.
Return ONLY valid JSON with this schema:
{
  "question": string,
  "entities": [string, ...],
  "reasoning": string,
  "answer": integer
}

Rules:
- Problem must require 5-7 distinct arithmetic steps (not fewer).
- Narrative should resemble GSM8K: multiple quantities, rates, fractions, percent, or unit conversions.
- "entities" must list the planning subgoals: bind every key quantity, intermediate target, and relationship BEFORE solving.
  Example entity lines: "discount_rate = 10%", "subtotal_before_discount = sum(item_costs)", "step3 = apply tax to discounted_total"
- "reasoning" must be 5-7 sentences (one per step). Each sentence performs ONE planning step and ends with a GSM8K calculator tag <<expr=result>>result.
- Reasoning must follow the entity plan in order; do not skip intermediate quantities.
- Do NOT include "####" in reasoning.
- answer must be a positive integer matching the final reasoning step.
- No markdown, no extra text, JSON only.
"""

PROFILE_DEFAULTS = {
    'short': {
        'system_prompt': LLM_SHORT_SYSTEM_PROMPT,
        'min_calc_steps': 2,
        'max_calc_steps': 4,
        'stage1_path': 'data/entity_stage1_train.txt',
        'stage2_path': 'data/entity_stage2_train.txt',
        'llm_max_tokens': 700,
    },
    'planning': {
        'system_prompt': LLM_PLANNING_SYSTEM_PROMPT,
        'min_calc_steps': 5,
        'max_calc_steps': 7,
        'stage1_path': 'data/entity_planning_stage1_train.txt',
        'stage2_path': 'data/entity_planning_stage2_train.txt',
        'llm_max_tokens': 1400,
    },
}

SHORT_HINTS = [
    "twice as many / multiplicative comparison",
    "three entities with offset relation",
    "percent comparison from total population",
    "part-whole body/length decomposition",
    "collection accumulation chain with relatives",
    "time-rate-percent composition",
    "discount then tax sequence",
    "work-rate aggregation across days",
]

PLANNING_HINTS = [
    "shopping cart: item prices, 10% member discount, then 8% tax — 6+ steps",
    "work-week: hourly rate, partial days, overtime multiplier, weekly total",
    "farm animals: nested ratios (sheep to cows to chickens) then difference",
    "baking: batches, servings per batch, guests eating fractions of servings",
    "travel: two legs at different speeds, rest stop, total time or distance",
    "fundraising: per-person goal, partial completion, remainder split across days",
    "classroom supplies: packs per student, teacher bonus packs, total cost at unit price",
    "garden: rows, plants per row, fraction died, replant cost per plant",
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


def _extract_json_object(text: str) -> dict:
  text = text.strip()
  if text.startswith("```"):
    text = text.strip("`")
    text = text.replace("json\n", "", 1).strip()
  try:
    return json.loads(text)
  except json.JSONDecodeError:
    m = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not m:
      raise
    return json.loads(m.group(0))


_CALC_TAG_RE = re.compile(r'<<[^>]+>>')


def _count_calc_steps(reasoning: str) -> int:
  return len(_CALC_TAG_RE.findall(reasoning))


def _validate_example(
    obj: dict,
    *,
    min_calc_steps: int = 2,
    max_calc_steps: int = 4,
) -> Tuple[str, List[str], str, int]:
  question = str(obj.get("question", "")).strip()
  entities = obj.get("entities", [])
  reasoning = str(obj.get("reasoning", "")).strip()
  answer = obj.get("answer", None)

  if not question:
    raise ValueError("Missing question")
  if not isinstance(entities, list) or not entities:
    raise ValueError("entities must be non-empty list")
  entities = [str(x).strip() for x in entities if str(x).strip()]
  if not entities:
    raise ValueError("entities list empty after cleanup")
  if not reasoning:
    raise ValueError("Missing reasoning")
  if "####" in reasoning:
    reasoning = reasoning.split("####", 1)[0].strip()

  n_steps = _count_calc_steps(reasoning)
  if n_steps < min_calc_steps or n_steps > max_calc_steps:
    raise ValueError(
        f"reasoning needs {min_calc_steps}-{max_calc_steps} calc steps, got {n_steps}"
    )

  try:
    answer_int = int(answer)
  except Exception as exc:
    raise ValueError(f"answer must be integer, got: {answer}") from exc
  if answer_int <= 0:
    raise ValueError(f"answer must be positive, got: {answer_int}")

  return question, entities, reasoning, answer_int


def _parse_retry_after(resp: requests.Response, attempt: int) -> float:
  raw = resp.headers.get('Retry-After')
  if raw:
    try:
      return max(1.0, float(raw))
    except ValueError:
      pass
  return min(120.0, 2.0 ** attempt)


def _llm_chat_completion(
    *,
    api_key: str,
    base_url: str,
    model: str,
    seed: int,
    temperature: float,
    max_tokens: int,
    timeout_s: int,
    user_prompt: str,
    system_prompt: str,
    rate_limit_retries: int = 25,
) -> str:
  url = base_url.rstrip("/") + "/chat/completions"
  headers = {
      "Authorization": f"Bearer {api_key}",
      "Content-Type": "application/json",
  }
  payload = {
      "model": model,
      "temperature": temperature,
      "max_tokens": max_tokens,
      "seed": seed,
      "response_format": {"type": "json_object"},
      "messages": [
          {"role": "system", "content": system_prompt},
          {"role": "user", "content": user_prompt},
      ],
  }
  last_err = None
  for rate_attempt in range(rate_limit_retries):
    resp = requests.post(url, headers=headers, json=payload, timeout=timeout_s)
    if resp.status_code in (429, 500, 502, 503, 504):
      wait_s = _parse_retry_after(resp, rate_attempt)
      print(
          f"[rate-limit] HTTP {resp.status_code}; sleeping {wait_s:.1f}s "
          f"(attempt {rate_attempt + 1}/{rate_limit_retries})"
      )
      time.sleep(wait_s)
      last_err = requests.HTTPError(f"{resp.status_code} for {url}", response=resp)
      continue
    try:
      resp.raise_for_status()
    except requests.HTTPError as exc:
      last_err = exc
      raise
    data = resp.json()
    return data["choices"][0]["message"]["content"]
  raise RuntimeError(
      f"LLM request failed after {rate_limit_retries} rate-limit retries: {last_err}"
  )


def _make_llm_prompt(template_hint: str, profile: str, feedback: Optional[str] = None) -> str:
  if profile == 'planning':
    extra = (
        "Write reasoning with 5 to 7 sentences. "
        "Each sentence must contain exactly one GSM8K calculator tag <<expr=result>>result. "
        "Do not combine multiple calculations into one sentence."
    )
  else:
    extra = "Keep quantities small/moderate and answer integer."
  prompt = (
      "Generate one problem with this style hint: "
      f"{template_hint}. {extra}"
  )
  if feedback:
    prompt += f"\n\nCORRECTION (previous attempt was invalid): {feedback}"
  return prompt


def _planning_retry_feedback(err: Exception) -> str:
  msg = str(err)
  m = re.search(r'got (\d+)', msg)
  got = m.group(1) if m else "too few"
  need = "5-7"
  return (
      f"The reasoning had {got} calculator <<>> steps but needs {need}. "
      "Add more intermediate planning steps (subtotals, unit conversions, "
      "partial counts) so each of 5-7 sentences has one <<expr=result>>result tag."
  )


def gen_with_llm(
    i: int,
    args,
    profile_cfg: dict,
    *,
    attempt: int = 0,
    feedback: Optional[str] = None,
) -> Tuple[str, List[str], str, int]:
  hints = PLANNING_HINTS if args.profile == 'planning' else SHORT_HINTS
  prompt = _make_llm_prompt(hints[i % len(hints)], args.profile, feedback=feedback)
  if args.request_delay_s > 0:
    time.sleep(args.request_delay_s)
  temperature = args.llm_temperature
  if args.profile == 'planning' and attempt > 0:
    temperature = min(1.0, args.llm_temperature + 0.05 * attempt)
  content = _llm_chat_completion(
      api_key=args.api_key,
      base_url=args.base_url,
      model=args.model,
      seed=args.seed + i * 1000 + attempt * 17,
      temperature=temperature,
      max_tokens=args.llm_max_tokens,
      timeout_s=args.timeout_s,
      user_prompt=prompt,
      system_prompt=profile_cfg['system_prompt'],
      rate_limit_retries=args.rate_limit_retries,
  )
  parsed = _extract_json_object(content)
  return _validate_example(
      parsed,
      min_calc_steps=profile_cfg['min_calc_steps'],
      max_calc_steps=profile_cfg['max_calc_steps'],
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


def _count_existing_examples(path: str) -> int:
  if not os.path.exists(path):
    return 0
  with open(path, 'r', encoding='utf-8') as f:
    text = f.read()
  return len([b for b in text.split('<|endoftext|>') if b.strip()])


def write_training_data(num_examples, seed, stage1_path, stage2_path, args, profile_cfg: dict):
  rng = random.Random(seed)
  failures = 0
  retry_budget = args.max_retries
  if args.profile == 'planning':
    retry_budget = max(retry_budget, 12)

  start_i = 0
  if args.resume:
    start_i = _count_existing_examples(stage2_path)
    if start_i >= num_examples:
      print(f"Resume: already have {start_i} examples (target {num_examples}); nothing to do.")
      return
    if start_i > 0:
      print(f"Resume: continuing from example {start_i} / {num_examples}")
  elif _count_existing_examples(stage2_path) > 0:
    print(
        f"Note: {stage2_path} already has {_count_existing_examples(stage2_path)} examples. "
        "Pass --resume to append, or delete the file to regenerate from scratch."
    )

  file_mode = 'a' if start_i > 0 else 'w'
  os.makedirs(os.path.dirname(stage1_path) or '.', exist_ok=True)
  os.makedirs(os.path.dirname(stage2_path) or '.', exist_ok=True)

  with open(stage1_path, file_mode, encoding='utf-8') as f1, \
      open(stage2_path, file_mode, encoding='utf-8') as f2:
    for i in range(start_i, num_examples):
      if args.generator == 'llm':
        ok = False
        last_err = None
        feedback = None
        for attempt in range(retry_budget):
          try:
            q, entities, reasoning, answer = gen_with_llm(
                i, args, profile_cfg, attempt=attempt, feedback=feedback
            )
            ok = True
            break
          except requests.HTTPError as exc:
            last_err = exc
            if getattr(exc.response, 'status_code', None) in (429, 500, 502, 503, 504):
              raise RuntimeError(
                  f"Rate limit persisted at example {i}. "
                  f"Re-run with --resume to continue. Last error: {exc}"
              ) from exc
            if args.profile == 'planning':
              feedback = _planning_retry_feedback(exc)
            if attempt < retry_budget - 1 and (attempt + 1) % 3 == 0:
              print(f"[retry] example {i}: attempt {attempt + 1}/{retry_budget} ({exc})")
          except Exception as exc:
            last_err = exc
            if args.profile == 'planning':
              feedback = _planning_retry_feedback(exc)
            if attempt < retry_budget - 1 and (attempt + 1) % 3 == 0:
              print(f"[retry] example {i}: attempt {attempt + 1}/{retry_budget} ({exc})")
        if not ok:
          if args.profile == 'planning':
            raise RuntimeError(
                f"LLM planning generation failed at example {i} after {retry_budget} retries: {last_err}"
            )
          failures += 1
          print(f"[warn] LLM generation failed at example {i}: {last_err}. Falling back to synthetic.")
          gen = rng.choice(GENERATORS)
          q, entities, reasoning, answer = gen(rng)
      else:
        if args.profile == 'planning':
          raise ValueError(
              "Planning profile requires --generator llm (synthetic templates are 2-4 steps only)."
          )
        gen = rng.choice(GENERATORS)
        q, entities, reasoning, answer = gen(rng)

      f1.write(block_stage1(i, q, entities))
      f2.write(block_stage2(i, q, entities, reasoning, answer))
      f1.flush()
      f2.flush()

      if (i + 1) % 50 == 0 or i == num_examples - 1:
        print(f"Generated {i + 1}/{num_examples}")

  final_count = _count_existing_examples(stage2_path)
  print(f"Wrote {final_count} examples -> {stage1_path}")
  print(f"Wrote {final_count} examples -> {stage2_path}")
  if failures:
    print(f"LLM failures with synthetic fallback: {failures}")


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
  parser.add_argument(
      '--profile',
      choices=['short', 'planning'],
      default='short',
      help='short=2-4 step entity; planning=5-7 step GSM8K-style planning entity.',
  )
  parser.add_argument('--num_examples', type=int, default=500)
  parser.add_argument('--seed', type=int, default=11711)
  parser.add_argument('--stage1_path', type=str, default=None)
  parser.add_argument('--stage2_path', type=str, default=None)
  parser.add_argument('--held_out_src', type=str, default='data/gsm8k_small_held_out.txt')
  parser.add_argument('--held_out_path', type=str, default='data/gsm8k_small_held_out_entities.txt')
  parser.add_argument('--skip_held_out', action='store_true')
  parser.add_argument('--generator', choices=['llm', 'synthetic'], default='llm')
  parser.add_argument('--model', type=str, default='gpt-4.1')
  parser.add_argument('--base_url', type=str, default='https://api.openai.com/v1')
  parser.add_argument('--api_key', type=str, default=None)
  parser.add_argument('--llm_temperature', type=float, default=0.7)
  parser.add_argument('--llm_max_tokens', type=int, default=None)
  parser.add_argument('--timeout_s', type=int, default=90)
  parser.add_argument('--max_retries', type=int, default=4)
  parser.add_argument('--min_calc_steps', type=int, default=None)
  parser.add_argument('--max_calc_steps', type=int, default=None)
  parser.add_argument(
      '--resume',
      action='store_true',
      help='Append from existing stage files (count blocks in stage2_path).',
  )
  parser.add_argument(
      '--request_delay_s',
      type=float,
      default=0.0,
      help='Sleep before each LLM request (reduces 429 rate limits).',
  )
  parser.add_argument(
      '--rate_limit_retries',
      type=int,
      default=25,
      help='Per-request retries on HTTP 429/5xx with exponential backoff.',
  )
  args = parser.parse_args()

  profile_cfg = dict(PROFILE_DEFAULTS[args.profile])
  if args.min_calc_steps is not None:
    profile_cfg['min_calc_steps'] = args.min_calc_steps
  if args.max_calc_steps is not None:
    profile_cfg['max_calc_steps'] = args.max_calc_steps
  if args.stage1_path is None:
    args.stage1_path = profile_cfg['stage1_path']
  if args.stage2_path is None:
    args.stage2_path = profile_cfg['stage2_path']
  if args.llm_max_tokens is None:
    args.llm_max_tokens = profile_cfg['llm_max_tokens']
  if args.profile == 'planning' and args.num_examples == 500:
    args.num_examples = 3000
  if args.profile == 'planning' and args.request_delay_s == 0.0:
    args.request_delay_s = 1.0

  if args.generator == 'llm' and not args.api_key:
    args.api_key = None
    for env_name in ('OPENAI_API_KEY',):
      value = __import__('os').environ.get(env_name)
      if value:
        args.api_key = value
        break
    if not args.api_key:
      raise ValueError(
          "LLM mode requires API key. Set --api_key or OPENAI_API_KEY."
      )

  print(
      f"Profile={args.profile} examples={args.num_examples} "
      f"steps={profile_cfg['min_calc_steps']}-{profile_cfg['max_calc_steps']} "
      f"model={args.model}"
  )
  write_training_data(
      args.num_examples, args.seed, args.stage1_path, args.stage2_path, args, profile_cfg
  )
  if not args.skip_held_out:
    write_entity_held_out(args.held_out_src, args.held_out_path)


if __name__ == '__main__':
  main()
