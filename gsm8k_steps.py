#!/usr/bin/env python3
"""
GSM8K reasoning step utilities.

Step definition (v1): each non-empty line under Reasoning: is one step.
The final #### answer line is the last step when present in the chain.
"""

from __future__ import annotations

import re
from typing import Iterable

from gsm8k_eval import extract_gold_answer, normalize_number

REASONING_DELIMITER = "Reasoning:\n"
ENTITIES_DELIMITER = "Entities:\n"

CALC_TAG_RE = re.compile(r"<<([^=]+)=([^>]+)>>")
NUMBER_RE = re.compile(r"-?\$?\d[\d,]*(?:\.\d+)?%?")

_BLOCK_ENTITY_RE = re.compile(
    r"Question:\s*(.*?)\s*Entities:\s*(.*?)\s*Reasoning:\s*(.*)",
    re.DOTALL,
)
_BLOCK_QUESTION_RE = re.compile(
    r"Question:\s*(.*?)\s*Reasoning:\s*(.*)",
    re.DOTALL,
)


def split_reasoning_steps(reasoning_text: str, final_answer=None) -> list[str]:
  """Split reasoning into steps (one line each). Appends #### if missing."""
  lines = [ln.strip() for ln in reasoning_text.strip().split("\n") if ln.strip()]
  steps: list[str] = []
  for ln in lines:
    if ln.startswith("####"):
      steps.append(ln)
      return steps
    steps.append(ln)

  if final_answer is not None:
    if isinstance(final_answer, float) and final_answer == int(final_answer):
      ans = str(int(final_answer))
    else:
      ans = str(final_answer).rstrip("0").rstrip(".") if isinstance(final_answer, float) else str(final_answer)
    steps.append(f"#### {ans}")
  return steps


def parse_entity_stage2_block(block: str) -> dict | None:
  m = _BLOCK_ENTITY_RE.search(block)
  if not m:
    return None
  question = m.group(1).strip()
  entities = m.group(2).strip()
  reasoning = m.group(3).strip()
  gold = extract_gold_answer(reasoning)
  if gold is None:
    return None
  return {
      "question": question,
      "entities": entities,
      "reasoning": reasoning,
      "gold": gold,
  }


def parse_question_reasoning_block(block: str) -> dict | None:
  m = _BLOCK_QUESTION_RE.search(block)
  if not m:
    return None
  question = m.group(1).strip()
  reasoning = m.group(2).strip()
  gold = extract_gold_answer(reasoning)
  if gold is None:
    return None
  return {
      "question": question,
      "entities": None,
      "reasoning": reasoning,
      "gold": gold,
  }


def build_prefix(question: str, entities: str | None, prior_steps: Iterable[str]) -> str:
  parts = [f"Question: {question}\n\n"]
  if entities:
    parts.append(f"Entities:\n{entities}\n\n")
  parts.append(REASONING_DELIMITER)
  parts.extend(f"{step}\n" if not step.endswith("\n") else step for step in prior_steps)
  return "".join(parts)


def expand_example_to_steps(example: dict) -> list[dict]:
  """Expand one problem into prefix/target pairs for step-conditioned SFT or DPO."""
  question = example["question"]
  entities = example.get("entities")
  steps = split_reasoning_steps(example["reasoning"], example.get("gold"))
  if not steps:
    return []

  pairs = []
  prefix = build_prefix(question, entities, [])
  for step in steps:
    target = step if step.endswith("\n") else f"{step}\n"
    pairs.append(
        {
            "question": question,
            "entities": entities,
            "prefix": prefix,
            "target": target,
            "step": target.rstrip("\n"),
        }
    )
    prefix += target
  return pairs


def extract_numbers_from_text(text: str) -> list[float]:
  nums = []
  for raw in NUMBER_RE.findall(text):
    val = normalize_number(raw)
    if val is not None:
      nums.append(val)
  return nums


def extract_calc_results(steps: list[str]) -> list[str]:
  results = []
  for step in steps:
    for m in CALC_TAG_RE.finditer(step):
      results.append(m.group(2).strip())
  return results


def _replace_first_number(line: str, new_num_str: str) -> str | None:
  m = NUMBER_RE.search(line)
  if not m:
    return None
  return line[: m.start()] + new_num_str + line[m.end() :]


def _replace_number_matching(line: str, old: str, new: str) -> str | None:
  if old not in line:
    return None
  return line.replace(old, new, 1)


def generate_planning_rejections(
    chosen: str,
    question: str,
    prior_steps: list[str],
    rng,
) -> list[tuple[str, str]]:
  """
  Planning-focused rejected steps (not arithmetic-typo focused).
  Returns list of (rejected_text, reject_type).
  """
  rejections: list[tuple[str, str]] = []
  q_nums = extract_numbers_from_text(question)
  prior_results = extract_calc_results(prior_steps)
  chosen_stripped = chosen.rstrip("\n")

  # wrong_quantity: swap first numeric literal with a different question number
  if q_nums and NUMBER_RE.search(chosen_stripped):
    for cand in rng.sample(q_nums, k=min(3, len(q_nums))):
      cand_str = str(int(cand)) if cand == int(cand) else str(cand)
      rejected = _replace_first_number(chosen_stripped, cand_str)
      if rejected and rejected != chosen_stripped:
        rejections.append((rejected + "\n", "wrong_quantity"))
        break

  # wrong_reference: replace a prior intermediate result with an unrelated question number
  if prior_results and q_nums:
    for ref in prior_results:
      for cand in q_nums:
        cand_str = str(int(cand)) if cand == int(cand) else str(cand)
        if ref == cand_str:
          continue
        rejected = _replace_number_matching(chosen_stripped, ref, cand_str)
        if rejected and rejected != chosen_stripped:
          rejections.append((rejected + "\n", "wrong_reference"))
          break
      if any(t == "wrong_reference" for _, t in rejections):
        break

  # wrong_operation_plan: swap common planning verbs (light heuristic)
  op_swaps = [
      ("total", "difference"),
      ("sum", "product"),
      ("add", "multiply"),
      ("combined", "remaining"),
      ("altogether", "left"),
      ("per week", "per day"),
  ]
  for a, b in op_swaps:
    if a in chosen_stripped.lower():
      idx = chosen_stripped.lower().find(a)
      rejected = chosen_stripped[:idx] + b + chosen_stripped[idx + len(a) :]
      if rejected != chosen_stripped:
        rejections.append((rejected + "\n", "wrong_operation_plan"))
        break

  # skip_step proxy: emit #### early with a plausible wrong answer from question nums
  if q_nums and not chosen_stripped.startswith("####"):
    wrong = rng.choice(q_nums)
    ans = str(int(wrong)) if wrong == int(wrong) else str(wrong)
    rejections.append((f"#### {ans}\n", "skip_step"))

  return rejections
