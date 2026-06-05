"""
Generate synthetic arithmetic pretraining data with PEMDAS (order of operations).

Most examples use Question / Reasoning / #### format (same as GSM8K) with multi-step
chains that respect * / before + - and parentheses.

Usage:
  python prepare_arithmetic.py --num_examples 50000 --output data/arithmetic_pretrain.txt
"""

import argparse
import ast
import operator
import random


_BIN_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.floordiv,
    ast.FloorDiv: operator.floordiv,
}


def calc_step(expression: str, result: int) -> str:
  """GSM8K-style intermediate step."""
  compact = expression.replace(' ', '')
  return f"{expression} = <<{compact}={result}>>{result}"


def safe_eval_expr(expr: str) -> int:
  """Evaluate an integer arithmetic expression (PEMDAS via Python ast)."""
  node = ast.parse(expr, mode='eval').body

  def _eval(n):
    if isinstance(n, ast.Constant):
      if not isinstance(n.value, int):
        raise ValueError('non-int constant')
      return n.value
    if isinstance(n, ast.BinOp):
      op = _BIN_OPS.get(type(n.op))
      if op is None:
        raise ValueError('bad op')
      left, right = _eval(n.left), _eval(n.right)
      if isinstance(n.op, (ast.Div, ast.FloorDiv)):
        if right == 0 or left % right != 0:
          raise ValueError('bad div')
        return left // right
      return op(left, right)
    if isinstance(n, ast.UnaryOp) and isinstance(n.op, ast.USub):
      return -_eval(n.operand)
    raise ValueError('unsupported')

  return _eval(node)


def format_example(example_id, question, reasoning):
  return (
      f"{example_id}\n\n"
      f"Question: {question}\n\n"
      f"Reasoning:\n{reasoning}\n\n"
      f"<|endoftext|>\n\n"
  )


def format_plain(example_id, line):
  return f"{example_id}\n\n{line}\n\n<|endoftext|>\n\n"


def _try_plain_pemdas(rng, max_attempts=20):
  templates = [
    '{a} + {b} * {c}',
    '{a} * {b} + {c}',
    '{a} + {b} / {c}',
    '({a} + {b}) * {c}',
    '{a} * ({b} + {c})',
    '{a} + {b} * {c} - {d}',
    '{a} * {b} + {c} * {d}',
  ]
  for _ in range(max_attempts):
    tpl = rng.choice(templates)
    a, b, c, d = rng.randint(2, 20), rng.randint(2, 12), rng.randint(2, 12), rng.randint(1, 15)
    if '/' in tpl:
      c = rng.randint(2, 12)
      b = c * rng.randint(2, 12)
    if tpl.startswith('(') and '- 0' not in tpl:
      pass
    if '({b} + {c})' in tpl or '({a} + {b})' in tpl:
      pass
    if '- {d}' in tpl:
      expr = tpl.format(a=a, b=b, c=c, d=d)
      try:
        result = safe_eval_expr(expr)
      except ValueError:
        continue
      if result < 0:
        continue
    else:
      expr = tpl.format(a=a, b=b, c=c, d=d)
      try:
        result = safe_eval_expr(expr)
      except ValueError:
        continue
      if result < 0:
        continue
    return f"{expr} = {result}"
  return None


def gen_add_mul(rng):
  a, b, c = rng.randint(2, 40), rng.randint(2, 12), rng.randint(2, 12)
  bc, ans = b * c, a + b * c
  expr = f"{a} + {b} * {c}"
  question = f"What is {expr}?"
  reasoning = (
      f"{calc_step(f'{b} * {c}', bc)}\n"
      f"{calc_step(f'{a} + {bc}', ans)}\n"
      f"#### {ans}"
  )
  return question, reasoning


def gen_mul_add(rng):
  a, b, c = rng.randint(2, 12), rng.randint(2, 12), rng.randint(2, 40)
  ab, ans = a * b, a * b + c
  expr = f"{a} * {b} + {c}"
  question = f"What is {expr}?"
  reasoning = (
      f"{calc_step(f'{a} * {b}', ab)}\n"
      f"{calc_step(f'{ab} + {c}', ans)}\n"
      f"#### {ans}"
  )
  return question, reasoning


def gen_mul_sub(rng):
  a, b, c = rng.randint(2, 12), rng.randint(2, 12), rng.randint(1, 50)
  ab = a * b
  if ab <= c:
    return gen_mul_add(rng)
  ans = ab - c
  expr = f"{a} * {b} - {c}"
  question = f"What is {expr}?"
  reasoning = (
      f"{calc_step(f'{a} * {b}', ab)}\n"
      f"{calc_step(f'{ab} - {c}', ans)}\n"
      f"#### {ans}"
  )
  return question, reasoning


def gen_add_div(rng):
  a = rng.randint(2, 40)
  divisor = rng.randint(2, 12)
  quotient = rng.randint(2, 12)
  dividend = divisor * quotient
  ans = a + quotient
  expr = f"{a} + {dividend} / {divisor}"
  question = f"What is {expr}?"
  reasoning = (
      f"{calc_step(f'{dividend} / {divisor}', quotient)}\n"
      f"{calc_step(f'{a} + {quotient}', ans)}\n"
      f"#### {ans}"
  )
  return question, reasoning


def gen_paren_add_mul(rng):
  a, b, c = rng.randint(2, 20), rng.randint(2, 20), rng.randint(2, 12)
  inner, ans = a + b, (a + b) * c
  expr = f"({a} + {b}) * {c}"
  question = f"What is {expr}?"
  reasoning = (
      f"{calc_step(f'{a} + {b}', inner)}\n"
      f"{calc_step(f'{inner} * {c}', ans)}\n"
      f"#### {ans}"
  )
  return question, reasoning


def gen_paren_sub_mul(rng):
  a = rng.randint(10, 40)
  b = rng.randint(2, a - 1)
  c = rng.randint(2, 12)
  inner, ans = a - b, (a - b) * c
  expr = f"({a} - {b}) * {c}"
  question = f"What is {expr}?"
  reasoning = (
      f"{calc_step(f'{a} - {b}', inner)}\n"
      f"{calc_step(f'{inner} * {c}', ans)}\n"
      f"#### {ans}"
  )
  return question, reasoning


def gen_mul_paren_add(rng):
  a = rng.randint(2, 12)
  b, c = rng.randint(2, 20), rng.randint(2, 20)
  inner, ans = b + c, a * (b + c)
  expr = f"{a} * ({b} + {c})"
  question = f"What is {expr}?"
  reasoning = (
      f"{calc_step(f'{b} + {c}', inner)}\n"
      f"{calc_step(f'{a} * {inner}', ans)}\n"
      f"#### {ans}"
  )
  return question, reasoning


def gen_add_mul_sub(rng):
  a, b, c = rng.randint(2, 30), rng.randint(2, 12), rng.randint(2, 12)
  d = rng.randint(1, 20)
  bc = b * c
  abc = a + bc
  if abc <= d:
    return gen_add_mul(rng)
  ans = abc - d
  expr = f"{a} + {b} * {c} - {d}"
  question = f"What is {expr}?"
  reasoning = (
      f"{calc_step(f'{b} * {c}', bc)}\n"
      f"{calc_step(f'{a} + {bc}', abc)}\n"
      f"{calc_step(f'{abc} - {d}', ans)}\n"
      f"#### {ans}"
  )
  return question, reasoning


def gen_mul_mul_add(rng):
  a, b, c, d = rng.randint(2, 10), rng.randint(2, 10), rng.randint(2, 10), rng.randint(2, 10)
  ab, cd, ans = a * b, c * d, a * b + c * d
  expr = f"{a} * {b} + {c} * {d}"
  question = f"What is {expr}?"
  reasoning = (
      f"{calc_step(f'{a} * {b}', ab)}\n"
      f"{calc_step(f'{c} * {d}', cd)}\n"
      f"{calc_step(f'{ab} + {cd}', ans)}\n"
      f"#### {ans}"
  )
  return question, reasoning


def gen_sub_mul_add(rng):
  a = rng.randint(20, 60)
  b, c = rng.randint(2, 12), rng.randint(2, 12)
  d = rng.randint(2, 40)
  bc = b * c
  if a <= bc:
    return gen_mul_add(rng)
  ans = a - bc + d
  expr = f"{a} - {b} * {c} + {d}"
  question = f"What is {expr}?"
  mid = a - bc
  reasoning = (
      f"{calc_step(f'{b} * {c}', bc)}\n"
      f"{calc_step(f'{a} - {bc}', mid)}\n"
      f"{calc_step(f'{mid} + {d}', ans)}\n"
      f"#### {ans}"
  )
  return question, reasoning


PEMDAS_GENERATORS = [
  gen_add_mul,
  gen_mul_add,
  gen_mul_sub,
  gen_add_div,
  gen_paren_add_mul,
  gen_paren_sub_mul,
  gen_mul_paren_add,
  gen_add_mul_sub,
  gen_mul_mul_add,
  gen_sub_mul_add,
]


def generate_pemdas_example(rng):
  return rng.choice(PEMDAS_GENERATORS)(rng)


def generate_single_op_plain(rng):
  op = rng.choice(['+', '-', '*'])
  if op == '+':
    a, b = rng.randint(1, 99), rng.randint(1, 99)
    c = a + b
  elif op == '-':
    a = rng.randint(10, 99)
    b = rng.randint(1, a)
    c = a - b
  else:
    a, b = rng.randint(2, 12), rng.randint(2, 12)
    c = a * b
  return f"{a} {op} {b} = {c}"


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('--num_examples', type=int, default=50000)
  parser.add_argument('--output', type=str, default='data/arithmetic_pretrain.txt')
  parser.add_argument('--seed', type=int, default=11711)
  parser.add_argument('--pemdas_fraction', type=float, default=0.85,
                      help='Fraction of multi-step PEMDAS Question/Reasoning examples.')
  parser.add_argument('--plain_fraction', type=float, default=0.15,
                      help='Fraction of plain one-line examples (single-op + PEMDAS).')
  args = parser.parse_args()

  if args.pemdas_fraction + args.plain_fraction > 1.0 + 1e-6:
    raise ValueError('pemdas_fraction + plain_fraction must be <= 1.0')

  rng = random.Random(args.seed)
  num_pemdas = int(args.num_examples * args.pemdas_fraction)
  num_plain = args.num_examples - num_pemdas

  blocks = []
  idx = 0

  for _ in range(num_pemdas):
    question, reasoning = generate_pemdas_example(rng)
    blocks.append(format_example(idx, question, reasoning))
    idx += 1

  for _ in range(num_plain):
    if rng.random() < 0.5:
      line = generate_single_op_plain(rng)
    else:
      line = _try_plain_pemdas(rng)
      if line is None:
        line = generate_single_op_plain(rng)
    blocks.append(format_plain(idx, line))
    idx += 1

  rng.shuffle(blocks)
  with open(args.output, 'w', encoding='utf-8') as f:
    f.writelines(blocks)

  print(f"Wrote {len(blocks)} examples to {args.output}")
  print(f"  PEMDAS (Question/Reasoning): {num_pemdas}")
  print(f"  Plain one-liners: {num_plain}")


if __name__ == '__main__':
  main()
