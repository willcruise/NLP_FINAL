"""Shared GSM8K parsing and answer extraction utilities."""

import re


def extract_final_answer(text: str):
  """Extract numeric answer after GSM8K #### marker."""
  if text is None or '####' not in text:
    return None
  ans = text.split('####')[-1].strip()
  if not ans:
    return None
  # Take first token and strip non-numeric chars except . and -
  token = ans.split()[0]
  token = re.sub(r'[^\d\.\-]', '', token)
  return token if token else None


def answers_match(pred: str, gold: str) -> bool:
  if pred is None or gold is None:
    return False
  try:
    return abs(float(pred) - float(gold)) < 1e-6
  except (ValueError, TypeError):
    return str(pred).strip() == str(gold).strip()


def parse_question_and_gold(example_text: str):
  """
  Parse one training example into question, full reasoning+answer, and gold numeric answer.
  """
  q_match = re.search(r'Question:\s*(.*?)\n\nReasoning:\s*', example_text, re.DOTALL)
  if not q_match:
    return None, None, None
  question = q_match.group(1).strip()
  reasoning_start = q_match.end()
  reasoning = example_text[reasoning_start:].strip()
  # Drop trailing EOS token text if present
  if reasoning.endswith('<|endoftext|>'):
    reasoning = reasoning[:-len('<|endoftext|>')].strip()
  gold = extract_final_answer(reasoning)
  return question, reasoning, gold


def build_prompt(example_id, question: str) -> str:
  return f"""{example_id}

Question: {question}

Reasoning:

"""


def format_train_example(example_id, question: str, reasoning: str, eos: str = '<|endoftext|>') -> str:
  return f"""{example_id}

Question: {question}

Reasoning:
{reasoning}

{eos}

"""


def load_gsm8k_train_examples(file_path: str, eos_token: str = '<|endoftext|>'):
  """Load GSM8K-style train file into list of dicts."""
  with open(file_path, 'r', encoding='utf-8') as f:
    text = f.read()

  raw_examples = [e.strip() for e in text.split(eos_token) if e.strip()]
  examples = []
  for raw in raw_examples:
    id_match = re.match(r'^(\d+)\s*\n', raw)
    ex_id = int(id_match.group(1)) if id_match else len(examples)
    question, reasoning, gold = parse_question_and_gold(raw)
    if question is None or gold is None:
      continue
    examples.append({
      'id': ex_id,
      'question': question,
      'reasoning': reasoning,
      'gold_answer': gold,
      'prompt': build_prompt(ex_id, question),
    })
  return examples
