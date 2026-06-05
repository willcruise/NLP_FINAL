# !/usr/bin/env python3


"""
This file contains our Dataset class for Quora paraphrase detection. You may want to modify this file to train on
additional sources of data, or if you change how the Quora dataset is processed (i.e. data augmentation, etc.).
"""

import csv

import re
import torch

from torch.utils.data import Dataset
from transformers import GPT2Tokenizer


def preprocess_string(s):
  return ' '.join(s.lower()
                  .replace('.', ' .')
                  .replace('?', ' ?')
                  .replace(',', ' ,')
                  .replace('\'', ' \'')
                  .split())


class ParaphraseDetectionDataset(Dataset):
  def __init__(self, dataset, args):
    self.dataset = dataset
    self.p = args
    self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    self.tokenizer.pad_token = self.tokenizer.eos_token

  def __len__(self):
    return len(self.dataset)

  def __getitem__(self, idx):
    return self.dataset[idx]

  def collate_fn(self, all_data):
    sent1 = [x[0] for x in all_data]
    sent2 = [x[1] for x in all_data]
    # labels = torch.LongTensor([x[2] for x in all_data])
    labels = ['yes' if label == 1 else 'no' for label in [x[2] for x in all_data]]
    labels = self.tokenizer(labels, return_tensors='pt', padding=True, truncation=True)['input_ids']
    sent_ids = [x[3] for x in all_data]

    cloze_style_sents = [f'Question 1: "{s1}"\nQuestion 2: "{s2}\nAre these questions asking the same thing?\n' for
                         (s1, s2) in zip(sent1, sent2)]
    encoding = self.tokenizer(cloze_style_sents, return_tensors='pt', padding=True, truncation=True, max_length=900)

    token_ids = torch.LongTensor(encoding['input_ids'])
    attention_mask = torch.LongTensor(encoding['attention_mask'])

    batched_data = {
      'token_ids': token_ids,
      'attention_mask': attention_mask,
      'labels': labels,
      'sent_ids': sent_ids
    }

    return batched_data


class ParaphraseDetectionTestDataset(Dataset):
  def __init__(self, dataset, args):
    self.dataset = dataset
    self.p = args
    self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    self.tokenizer.pad_token = self.tokenizer.eos_token

  def __len__(self):
    return len(self.dataset)

  def __getitem__(self, idx):
    return self.dataset[idx]

  def collate_fn(self, all_data):
    sent1 = [x[0] for x in all_data]
    sent2 = [x[1] for x in all_data]
    sent_ids = [x[2] for x in all_data]

    cloze_style_sents = [f'Is "{s1}" a paraphrase of "{s2}"? Answer "yes" or "no": ' for (s1, s2) in
                         zip(sent1, sent2)]

    encoding = self.tokenizer(cloze_style_sents, return_tensors='pt', padding=True, truncation=True)

    token_ids = torch.LongTensor(encoding['input_ids'])
    attention_mask = torch.LongTensor(encoding['attention_mask'])

    batched_data = {
      'token_ids': token_ids,
      'attention_mask': attention_mask,
      'sent_ids': sent_ids
    }

    return batched_data


def load_paraphrase_data(paraphrase_filename, split='train'):
  paraphrase_data = []
  if split == 'test':
    with open(paraphrase_filename, 'r') as fp:
      for record in csv.DictReader(fp, delimiter='\t'):
        sent_id = record['id'].lower().strip()
        paraphrase_data.append((preprocess_string(record['sentence1']),
                                preprocess_string(record['sentence2']),
                                sent_id))

  else:
    with open(paraphrase_filename, 'r') as fp:
      for record in csv.DictReader(fp, delimiter='\t'):
        try:
          sent_id = record['id'].lower().strip()
          paraphrase_data.append((preprocess_string(record['sentence1']),
                                  preprocess_string(record['sentence2']),
                                  int(float(record['is_duplicate'])), sent_id))
        except:
          pass

  print(f"Loaded {len(paraphrase_data)} {split} examples from {paraphrase_filename}")
  return paraphrase_data


class SonnetsDataset(Dataset):
  def __init__(self, file_path):
    self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

    self.tokenizer.pad_token = self.tokenizer.eos_token
    self.sonnets = self._load_sonnets(file_path)

  def _load_sonnets(self, file_path):
    """Reads the file and extracts individual sonnets."""
    with open(file_path, 'r', encoding='utf-8') as f:
      text = f.read()

    # Split sonnets based on numbering pattern (e.g., "\n\n1\n\n")
    sonnets = re.split(r'\n\s*\d+\s*\n', text)[1:]  # Remove header text

    # Strip leading/trailing spaces
    return [s.strip() for s in sonnets]

  def __len__(self):
    return len(self.sonnets)

  def __getitem__(self, idx):
    return (idx, self.sonnets[idx])

  def collate_fn(self, all_data):
    idx = [example[0] for example in all_data]
    sonnets = [example[1] for example in all_data]

    encoding = self.tokenizer(sonnets, return_tensors='pt', padding=True, truncation=True)
    token_ids = torch.LongTensor(encoding['input_ids'])
    attention_mask = torch.LongTensor(encoding['attention_mask'])

    batched_data = {
      'token_ids': token_ids,
      'attention_mask': attention_mask,
      'sent_ids': idx
    }

    return batched_data
  


REASONING_DELIMITER = "Reasoning:\n"
ENTITIES_DELIMITER = "Entities:\n"

# Where CE loss starts when mask_prompt is enabled.
#   reasoning          — after Reasoning:\\n (default GSM8K CoT)
#   entities           — after Entities:\\n (stage-1 entity binding)
#   entities_reasoning — after Entities:\\n (entities + reasoning + ####)
MASK_TARGETS = ('reasoning', 'entities', 'entities_reasoning')


def _prefix_through_delimiter(text, delimiter):
  idx = text.find(delimiter)
  if idx != -1:
    return text[:idx + len(delimiter)]
  # Loose fallback without trailing newline.
  bare = delimiter.rstrip('\n')
  idx = text.find(bare)
  if idx == -1:
    return None
  end = idx + len(bare)
  if end < len(text) and text[end] == '\n':
    end += 1
  return text[:end]


def get_loss_token_start(text, tokenizer, mask_target='reasoning', max_length=900):
  """
  Token index where supervised loss begins (first predicted token after the masked prefix).
  """
  if mask_target == 'entities' or mask_target == 'entities_reasoning':
    prefix = _prefix_through_delimiter(text, ENTITIES_DELIMITER)
    if prefix is None:
      prefix = _prefix_through_delimiter(text, REASONING_DELIMITER)
  else:
    prefix = _prefix_through_delimiter(text, REASONING_DELIMITER)

  if prefix is None:
    return 0

  full_ids = tokenizer(
      text,
      truncation=True,
      max_length=max_length,
      add_special_tokens=True,
  )["input_ids"]
  prefix_ids = tokenizer(
      prefix,
      truncation=True,
      max_length=max_length,
      add_special_tokens=True,
  )["input_ids"]

  if len(prefix_ids) <= len(full_ids) and full_ids[:len(prefix_ids)] == prefix_ids:
    return len(prefix_ids)

  shared = 0
  for i, tok in enumerate(prefix_ids):
    if i < len(full_ids) and full_ids[i] == tok:
      shared = i + 1
    else:
      break
  return shared


def get_reasoning_token_start(text, tokenizer, max_length=900):
  """Backward-compatible alias (loss starts at Reasoning:\\n)."""
  return get_loss_token_start(text, tokenizer, mask_target='reasoning', max_length=max_length)


def mask_labels_from_start(labels, attention_mask, loss_starts):
  """Mask labels before loss_starts so CE is computed only on the completion region."""
  for i in range(labels.size(0)):
    start = int(loss_starts[i].item())
    seq_len = int(attention_mask[i].sum().item())
    start = min(start, seq_len)
    if start > 1:
      labels[i, :start - 1] = -100
  labels[attention_mask[:, 1:] == 0] = -100
  return labels


def mask_labels_to_reasoning_only(labels, attention_mask, reasoning_starts):
  """Backward-compatible alias."""
  return mask_labels_from_start(labels, attention_mask, reasoning_starts)


class ReasoningDataset(Dataset):

  PROMPT_DELIMITER = REASONING_DELIMITER

  def __init__(self, file_path, mask_prompt=False, mask_target=None):
    """
    mask_target: 'reasoning' | 'entities' | 'entities_reasoning' (requires mask_prompt).
    """
    if mask_target is not None and mask_target not in MASK_TARGETS:
      raise ValueError(f"mask_target must be one of {MASK_TARGETS}, got {mask_target!r}")
    if mask_target is not None:
      mask_prompt = True
    elif mask_prompt:
      mask_target = 'reasoning'

    self.max_length = 512
    self.mask_prompt = mask_prompt
    self.mask_target = mask_target
    self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    self.tokenizer.pad_token = self.tokenizer.eos_token

    self.examples = self._load_examples(file_path)
    print("Loaded examples:", len(self.examples))

    if self.has_eos:

      missing_eos = 0

      for i, example in enumerate(self.examples):
        if not example.endswith(self.tokenizer.eos_token):
            missing_eos += 1
            print(f"Missing EOS in example {i}")

      print("Examples missing EOS:", missing_eos)

    else:
      print("Held-out dataset detected; EOS check skipped.")

    lengths = [
      len(self.tokenizer(x)["input_ids"])
      for x in self.examples
    ]
    print("Min length:", min(lengths))
    print("Max length:", max(lengths))
    print("Average length:", sum(lengths)/len(lengths))


    print("First example preview:")
    print(repr(self.examples[0][:300]))

    print("EOS token id:", self.tokenizer.eos_token_id)

    first_ids = self.tokenizer(self.examples[0])["input_ids"]
    print("Last 20 tokens:", first_ids[-20:])

    for i in range(3):
      print(repr(self.examples[i][-30:]))

  def _load_examples(self, file_path):

    with open(file_path, 'r', encoding='utf-8') as f:
        text = f.read()

    self.has_eos = "<|endoftext|>" in text

    if self.has_eos:
        examples = [
            e.strip() + self.tokenizer.eos_token
            for e in text.split("<|endoftext|>")
            if e.strip()
        ]
    else:
        examples = re.split(r'\n\s*\d+\s*\n', text)
        examples = [e.strip() for e in examples if e.strip()]

    return examples

  def __len__(self):
    return len(self.examples)

  def __getitem__(self, idx):
    return (idx, self.examples[idx])

  def collate_fn(self, all_data):
    idx = [example[0] for example in all_data]
    texts = [example[1] for example in all_data]

    max_length = 900
    encoding = self.tokenizer(
        texts,
        return_tensors='pt',
        padding=True,
        truncation=True,
        max_length=max_length
    )

    target = self.mask_target or 'reasoning'
    loss_starts = torch.LongTensor([
        get_loss_token_start(text, self.tokenizer, mask_target=target, max_length=max_length)
        for text in texts
    ])

    batched_data = {
        'token_ids': torch.LongTensor(encoding['input_ids']),
        'attention_mask': torch.LongTensor(encoding['attention_mask']),
        'reasoning_starts': loss_starts,
        'loss_starts': loss_starts,
        'sent_ids': idx
    }

    return batched_data
