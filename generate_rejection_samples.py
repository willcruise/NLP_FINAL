"""
Generate rejection-sampling training data: keep model solutions whose #### answer matches gold.

Run in a separate terminal after GSM8K SFT (or arithmetic + SFT) checkpoint exists.

Usage:
  python generate_rejection_samples.py --use_gpu \
    --checkpoint_tag reasoning \
    --train_path data/gsm8k_small_train.txt \
    --output_path data/rejection_train.txt \
    --num_samples 10
"""

import argparse
import os

import torch
import torch.nn.functional as F
from tqdm import tqdm

from checkpoint_utils import find_latest_checkpoint_any, set_checkpoint_filepath
from gsm8k_utils import (
  answers_match,
  extract_final_answer,
  format_train_example,
  load_gsm8k_train_examples,
)
from reasoning_generation import ReasoningGPT, add_arguments, seed_everything


@torch.no_grad()
def generate_reasoning(
    model,
    prompt_ids,
    temperature=0.8,
    top_p=0.9,
    max_new_tokens=256,
):
  """Sample continuation after prompt; returns generated text (prompt + new tokens)."""
  device = model.get_device()
  token_ids = prompt_ids.to(device)
  attention_mask = torch.ones(token_ids.shape, dtype=torch.int64, device=device)
  max_context = model.gpt.pos_embedding.num_embeddings
  prompt_len = token_ids.size(1)

  for _ in range(max_new_tokens):
    if token_ids.size(1) >= max_context:
      break

    logits = model(token_ids, attention_mask)[:, -1, :] / temperature
    probs = F.softmax(logits, dim=-1)

    sorted_probs, sorted_indices = torch.sort(probs, descending=True)
    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
    top_p_mask = cumulative_probs <= top_p
    top_p_mask[..., 1:] = top_p_mask[..., :-1].clone()
    top_p_mask[..., 0] = True
    filtered_probs = sorted_probs * top_p_mask
    filtered_probs = filtered_probs / filtered_probs.sum(dim=-1, keepdim=True)

    sampled_index = torch.multinomial(filtered_probs, num_samples=1)
    sampled_token = sorted_indices.gather(dim=-1, index=sampled_index)

    if sampled_token.item() == model.tokenizer.eos_token_id:
      break

    token_ids = torch.cat([token_ids, sampled_token], dim=1)
    attention_mask = torch.cat(
        [attention_mask, torch.ones((1, 1), dtype=torch.int64, device=device)], dim=1
    )

  return model.tokenizer.decode(token_ids[0].cpu().tolist())


def extract_generated_reasoning(full_text: str, prompt: str) -> str:
  """Return only the model-generated reasoning (strip prompt prefix)."""
  if full_text.startswith(prompt):
    return full_text[len(prompt):].strip()
  if 'Reasoning:' in full_text:
    return full_text.split('Reasoning:', 1)[-1].strip()
  return full_text.strip()


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('--train_path', type=str, default='data/gsm8k_small_train.txt')
  parser.add_argument('--output_path', type=str, default='data/rejection_train.txt')
  parser.add_argument('--checkpoint', type=str, default=None,
                      help='Path to .pt checkpoint. If omitted, uses latest for --checkpoint_tag.')
  parser.add_argument('--checkpoint_tag', type=str, default='reasoning',
                      help='Same tag as arithmetic_pretrain / reasoning_generation.')
  parser.add_argument('--checkpoint_glob', type=str, default=None,
                      help='Optional legacy glob; ignored if --checkpoint is set.')
  parser.add_argument('--num_samples', type=int, default=10,
                      help='Number of sampled solutions per question.')
  parser.add_argument('--temperature', type=float, default=0.8)
  parser.add_argument('--top_p', type=float, default=0.9)
  parser.add_argument('--max_new_tokens', type=int, default=256)
  parser.add_argument('--keep_all_correct', action='store_true',
                      help='Keep every correct sample; default keeps first correct only.')
  parser.add_argument('--use_gpu', action='store_true')
  parser.add_argument('--seed', type=int, default=11711)
  parser.add_argument('--max_examples', type=int, default=None,
                      help='Limit number of train questions (for debugging).')
  args = parser.parse_args()
  set_checkpoint_filepath(args)

  seed_everything(args.seed)
  device = torch.device('cuda') if args.use_gpu else torch.device('cpu')

  if args.checkpoint:
    ckpt_path = args.checkpoint
  elif args.checkpoint_glob:
    import glob
    paths = glob.glob(args.checkpoint_glob)
    if not paths:
      ckpt_path = None
    else:
      ckpt_path = max(paths, key=lambda p: int(os.path.basename(p).split('_', 1)[0]))
  else:
    ckpt_path, _ = find_latest_checkpoint_any(args)

  if ckpt_path is None:
    raise FileNotFoundError(
        f"No checkpoint found (checkpoint={args.checkpoint}, tag={args.checkpoint_tag})"
    )
  print(f"Loading checkpoint: {ckpt_path}")

  saved = torch.load(ckpt_path, weights_only=False)
  model_args = add_arguments(saved['args'])
  model = ReasoningGPT(model_args)
  model.load_state_dict(saved['model'])
  model = model.to(device)
  model.eval()

  examples = load_gsm8k_train_examples(args.train_path)
  if args.max_examples:
    examples = examples[:args.max_examples]
  print(f"Loaded {len(examples)} training questions")

  accepted = []
  stats = {'questions': 0, 'samples_tried': 0, 'accepted': 0, 'no_correct': 0}

  for ex in tqdm(examples, desc='rejection-sampling'):
    stats['questions'] += 1
    encoding = model.tokenizer(
        ex['prompt'],
        return_tensors='pt',
        truncation=True,
        max_length=900,
    )
    prompt_ids = encoding['input_ids']

    found_any = False
    for _ in range(args.num_samples):
      stats['samples_tried'] += 1
      full_text = generate_reasoning(
          model,
          prompt_ids,
          temperature=args.temperature,
          top_p=args.top_p,
          max_new_tokens=args.max_new_tokens,
      )
      reasoning = extract_generated_reasoning(full_text, ex['prompt'])
      pred = extract_final_answer(reasoning)

      if answers_match(pred, ex['gold_answer']):
        stats['accepted'] += 1
        found_any = True
        accepted.append(format_train_example(ex['id'], ex['question'], reasoning))
        if not args.keep_all_correct:
          break

    if not found_any:
      stats['no_correct'] += 1

  os.makedirs(os.path.dirname(args.output_path) or '.', exist_ok=True)
  with open(args.output_path, 'w', encoding='utf-8') as f:
    f.writelines(accepted)

  print(f"Wrote {len(accepted)} accepted examples to {args.output_path}")
  print(
      f"Stats: questions={stats['questions']}, samples_tried={stats['samples_tried']}, "
      f"accepted={stats['accepted']}, questions_with_no_correct={stats['no_correct']}"
  )


if __name__ == '__main__':
  main()
