#!/usr/bin/env python3
"""
Entity-scaffolded reasoning: stage-1 (Entities) then stage-2 (Reasoning) training and inference.

Pipeline (from best SFT checkpoint, e.g. 46_reasoning.pt):
  python prepare_entity_data.py
  python entity_reasoning.py train --stage 1 --init_checkpoint 46_reasoning.pt --epochs 56 --use_gpu
  python entity_reasoning.py train --stage 2 --epochs 66 --use_gpu
  python entity_reasoning.py generate --checkpoint 65_reasoning.pt --use_gpu

Stage 1: loss on Entities block only (--mask_target entities)
Stage 2: loss on Reasoning block; Entities in context (--mask_target reasoning)
Inference: generate Entities, then Reasoning (two-stage).
"""

import argparse
import os
import random

import numpy as np
import torch
import torch.nn.functional as F
from einops import rearrange
from torch.utils.data import DataLoader
from tqdm import tqdm

from checkpoint_utils import (
  add_checkpoint_args,
  checkpoint_path,
  cleanup_incomplete_checkpoints,
  find_latest_checkpoint_any,
  prune_old_checkpoints,
  resolve_training_start,
  save_model,
  set_checkpoint_filepath,
)
from gpt_datasets import ENTITIES_DELIMITER, REASONING_DELIMITER, ReasoningDataset, mask_labels_from_start
from optimizer import AdamW
from reasoning_generation import ReasoningGPT, add_arguments, seed_everything

TQDM_DISABLE = False

STAGE_CONFIG = {
  '1': {
      'default_path': 'data/entity_stage1_train.txt',
      'mask_target': 'entities',
  },
  '2': {
      'default_path': 'data/entity_stage2_train.txt',
      'mask_target': 'reasoning',
  },
}


def train_entity_stage(args):
  device = torch.device('cuda') if args.use_gpu else torch.device('cpu')
  cfg = STAGE_CONFIG[args.stage]

  data_path = args.reasoning_path or cfg['default_path']
  mask_target = args.mask_target or cfg['mask_target']

  dataset = ReasoningDataset(data_path, mask_target=mask_target)
  dataloader = DataLoader(
      dataset, shuffle=True, batch_size=args.batch_size, collate_fn=dataset.collate_fn
  )

  held_out = ReasoningDataset(args.held_out_reasoning_path)
  print(f"Stage {args.stage}: training on {data_path} (mask_target={mask_target})")

  args = add_arguments(args)
  cleanup_incomplete_checkpoints(args)

  model = ReasoningGPT(args).to(device)
  start_epoch, latest_epoch = resolve_training_start(
      model, args, init_checkpoint=args.init_checkpoint
  )
  optimizer = AdamW(model.parameters(), lr=args.lr)

  if start_epoch >= args.epochs:
    print(f"Already at epoch {latest_epoch} >= {args.epochs}; skip training.")
    return

  for epoch in range(start_epoch, args.epochs):
    model.train()
    train_loss = 0
    num_batches = 0

    for batch in tqdm(dataloader, desc=f'entity-stage{args.stage}-{epoch}', disable=TQDM_DISABLE):
      b_ids = batch['token_ids'].to(device)
      b_mask = batch['attention_mask'].to(device)

      optimizer.zero_grad()
      logits = model(b_ids, b_mask)
      logits = rearrange(logits[:, :-1].contiguous(), 'b t d -> (b t) d')
      labels = b_ids[:, 1:].clone()
      labels = mask_labels_from_start(labels, b_mask, batch['loss_starts'])
      loss = F.cross_entropy(logits, labels.flatten(), ignore_index=-100, reduction='mean')
      loss.backward()
      optimizer.step()

      train_loss += loss.item()
      num_batches += 1

    print(f"Epoch {epoch}: train loss :: {train_loss / num_batches:.3f}")

    model.eval()
    demo = list(held_out)[0][1]
    if ENTITIES_DELIMITER not in demo:
      demo = demo.replace(REASONING_DELIMITER, f'{ENTITIES_DELIMITER}\n{REASONING_DELIMITER}', 1)
    if args.stage == '1':
      prompt = demo.split(REASONING_DELIMITER)[0] + ENTITIES_DELIMITER
    else:
      prompt = demo.split(REASONING_DELIMITER)[0] + REASONING_DELIMITER
    enc = model.tokenizer(prompt, return_tensors='pt', truncation=True, max_length=900)
    _, text = model.generate(
        enc['input_ids'].to(device),
        temperature=args.temperature,
        top_p=args.top_p,
        max_length=args.max_gen_length,
        sample=not args.greedy,
    )
    print("Demo generation:\n", text[:600], "\n")

    prune_old_checkpoints(epoch, args)
    save_model(model, optimizer, args, checkpoint_path(epoch, args))


def _question_from_block(block_text):
  if 'Question:' not in block_text:
    return block_text
  q = block_text.split('Question:', 1)[1]
  if ENTITIES_DELIMITER in q:
    q = q.split(ENTITIES_DELIMITER)[0]
  elif REASONING_DELIMITER in q:
    q = q.split(REASONING_DELIMITER)[0]
  elif 'Reasoning:' in q:
    q = q.split('Reasoning:')[0]
  return 'Question:' + q.strip()


@torch.no_grad()
def generate_two_stage(model, prompt_text, device, args):
  """Generate Entities then Reasoning for one held-out block."""
  base = _question_from_block(prompt_text)
  entity_prompt = base + '\n\n' + ENTITIES_DELIMITER

  enc1 = model.tokenizer(entity_prompt, return_tensors='pt', truncation=True, max_length=900)
  _, after_entities = model.generate(
      enc1['input_ids'].to(device),
      temperature=args.temperature,
      top_p=args.top_p,
      max_length=args.max_entity_tokens,
      sample=not args.greedy,
  )

  if REASONING_DELIMITER in after_entities:
    head, _ = after_entities.split(REASONING_DELIMITER, 1)
    entity_block = head
  else:
    entity_block = after_entities

  if ENTITIES_DELIMITER in entity_block:
    entities_body = entity_block.split(ENTITIES_DELIMITER, 1)[1].strip()
  else:
    entities_body = entity_block[len(entity_prompt):].strip()

  reason_prompt = base + '\n\n' + ENTITIES_DELIMITER + entities_body + '\n\n' + REASONING_DELIMITER
  enc2 = model.tokenizer(reason_prompt, return_tensors='pt', truncation=True, max_length=900)
  _, full = model.generate(
      enc2['input_ids'].to(device),
      temperature=args.temperature,
      top_p=args.top_p,
      max_length=args.max_gen_length,
      sample=not args.greedy,
  )
  return full


@torch.no_grad()
def generate_entity_submission(args):
  device = torch.device('cuda') if args.use_gpu else torch.device('cpu')

  if args.checkpoint:
    ckpt_path = args.checkpoint
  else:
    ckpt_path, _ = find_latest_checkpoint_any(args)
  if ckpt_path is None:
    raise FileNotFoundError('No checkpoint found')

  saved = torch.load(ckpt_path, weights_only=False)
  print(f"Loaded {ckpt_path}")
  model = ReasoningGPT(saved['args'])
  model.load_state_dict(saved['model'])
  model = model.to(device)
  model.eval()

  held_out = ReasoningDataset(args.held_out_reasoning_path)
  os.makedirs(os.path.dirname(args.reasoning_out) or '.', exist_ok=True)

  with open(args.reasoning_out, 'w', encoding='utf-8') as f:
    f.write('--Generated Reasonings (two-stage entity) --\n\n')
    for ex_id, block in held_out:
      text = generate_two_stage(model, block, device, args)
      f.write(f'\n{ex_id}\n')
      f.write(text + '\n\n')
      print(f'--- id {ex_id} ---\n{text[:400]}...\n')

  print(f"Wrote {args.reasoning_out}")


def build_parser():
  parser = argparse.ArgumentParser(description='Entity-scaffolded reasoning train/generate')
  sub = parser.add_subparsers(dest='command', required=True)

  p_train = sub.add_parser('train', help='Stage 1 or 2 SFT')
  p_train.add_argument('--stage', choices=['1', '2'], required=True)
  p_train.add_argument('--reasoning_path', type=str, default=None)
  p_train.add_argument('--held_out_reasoning_path', type=str,
                       default='data/gsm8k_small_held_out_entities.txt')
  p_train.add_argument('--mask_target', type=str, default=None,
                       choices=['reasoning', 'entities', 'entities_reasoning'])
  p_train.add_argument('--epochs', type=int, default=56)
  p_train.add_argument('--lr', type=float, default=5e-6)
  p_train.add_argument('--batch_size', type=int, default=8)
  p_train.add_argument('--use_gpu', action='store_true')
  p_train.add_argument('--seed', type=int, default=11711)
  p_train.add_argument('--temperature', type=float, default=0.7)
  p_train.add_argument('--top_p', type=float, default=0.9)
  p_train.add_argument('--max_gen_length', type=int, default=256)
  p_train.add_argument('--greedy', action='store_true')
  p_train.add_argument('--model_size', default='gpt2',
                       choices=['gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'])
  add_checkpoint_args(p_train)

  p_gen = sub.add_parser('generate', help='Two-stage held-out generation')
  p_gen.add_argument('--checkpoint', type=str, default=None)
  p_gen.add_argument('--held_out_reasoning_path', type=str,
                     default='data/gsm8k_small_held_out_entities.txt')
  p_gen.add_argument('--reasoning_out', type=str, default='outputs/generated_reasoning_entities.txt')
  p_gen.add_argument('--use_gpu', action='store_true')
  p_gen.add_argument('--temperature', type=float, default=0.7)
  p_gen.add_argument('--top_p', type=float, default=0.9)
  p_gen.add_argument('--max_entity_tokens', type=int, default=128)
  p_gen.add_argument('--max_gen_length', type=int, default=256)
  p_gen.add_argument('--greedy', action='store_true')
  add_checkpoint_args(p_gen)

  return parser


def main():
  parser = build_parser()
  args = parser.parse_args()
  set_checkpoint_filepath(args)
  seed_everything(getattr(args, 'seed', 11711))

  if args.command == 'train':
    train_entity_stage(args)
  elif args.command == 'generate':
    generate_entity_submission(args)


if __name__ == '__main__':
  main()
