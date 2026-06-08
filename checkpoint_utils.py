"""
Shared checkpoint paths for arithmetic pretrain and GSM8K reasoning (one model, one timeline).

Checkpoints are named: {epoch}_{checkpoint_tag}.pt
Both arithmetic_pretrain.py and reasoning_generation.py use the same --checkpoint_tag (default: reasoning).
"""

import glob
import os

import torch

MAX_CHECKPOINTS = 10

# Older runs before unification
LEGACY_CHECKPOINT_GLOBS = [
    '*_arithmetic-*.pt',
    '*_*-*-reasoning.pt',
]


def add_checkpoint_args(parser, default_tag='reasoning', default_init=None):
  parser.add_argument(
      '--checkpoint_tag',
      type=str,
      default=default_tag,
      help='Shared checkpoint basename for arithmetic + GSM8K stages (files: {epoch}_{tag}.pt).',
  )
  parser.add_argument(
      '--init_checkpoint',
      type=str,
      default=default_init,
      help='Optional .pt path; used only when no checkpoint exists for --checkpoint_tag.',
  )
  return parser


def set_checkpoint_filepath(args):
  """Set args.filepath from args.checkpoint_tag."""
  tag = getattr(args, 'checkpoint_tag', 'reasoning')
  args.filepath = f'{tag}.pt'
  return args.filepath


def checkpoint_path(epoch, args):
  return f'{epoch}_{args.filepath}'


def _epoch_from_path(path):
  return int(os.path.basename(path).split('_', 1)[0])


def find_latest_checkpoint(args):
  """Latest checkpoint for unified args.filepath, or (None, -1)."""
  paths = glob.glob(f'*_{args.filepath}')
  if not paths:
    return None, -1
  latest_path = max(paths, key=_epoch_from_path)
  return latest_path, _epoch_from_path(latest_path)


def find_latest_checkpoint_any(args):
  """
  Latest checkpoint across unified tag and legacy arithmetic/reasoning filenames.
  """
  paths = list(glob.glob(f'*_{args.filepath}'))
  for pattern in LEGACY_CHECKPOINT_GLOBS:
    paths.extend(glob.glob(pattern))
  paths = list(set(paths))
  if not paths:
    return None, -1
  latest_path = max(paths, key=_epoch_from_path)
  epoch = _epoch_from_path(latest_path)
  if not os.path.basename(latest_path).endswith(args.filepath):
    print(
        f"Note: using legacy checkpoint {latest_path} (unified pattern *_{args.filepath} not found)."
    )
  return latest_path, epoch


def cleanup_incomplete_checkpoints(args):
  for path in glob.glob(f'*_{args.filepath}.tmp'):
    os.remove(path)
    print(f"removed incomplete checkpoint {path}")


def _free_bytes(path='.'):
  st = os.statvfs(path)
  return st.f_bavail * st.f_frsize


def save_model(model, optimizer, args, filepath, min_free_gb=1.0):
  save_info = {
      'model': model.state_dict(),
      'args': args,
  }
  temp_path = f'{filepath}.tmp'
  free_gb = _free_bytes(os.path.dirname(os.path.abspath(filepath)) or '.') / (1024 ** 3)
  if free_gb < min_free_gb:
    raise RuntimeError(
        f'Not enough disk space to save {filepath}: {free_gb:.2f} GB free '
        f'(need ~{min_free_gb:.1f} GB). Delete old checkpoints or free disk space.'
    )
  try:
    torch.save(save_info, temp_path)
    os.replace(temp_path, filepath)
    print(f"save the model to {filepath}")
  except OSError as exc:
    if os.path.exists(temp_path):
      os.remove(temp_path)
    raise RuntimeError(
        f'Failed to write checkpoint {filepath} ({free_gb:.2f} GB free on disk). '
        'Delete old .pt files or free disk space, then resume training.'
    ) from exc
  except Exception:
    if os.path.exists(temp_path):
      os.remove(temp_path)
    raise


def prune_old_checkpoints(epoch, args):
  old_epoch = epoch - MAX_CHECKPOINTS
  if old_epoch < 0:
    return
  old_path = checkpoint_path(old_epoch, args)
  if os.path.exists(old_path):
    os.remove(old_path)
    print(f"removed old checkpoint {old_path}")


def resolve_training_start(model, args, init_checkpoint=None):
  """
  Load weights from latest unified/legacy checkpoint or init_checkpoint.
  Returns (start_epoch, latest_epoch).
  """
  latest_path, latest_epoch = find_latest_checkpoint_any(args)
  start_epoch = latest_epoch + 1 if latest_path else 0

  if latest_path:
    saved = torch.load(latest_path, weights_only=False)
    model.load_state_dict(saved['model'])
    print(f"Resumed weights from {latest_path}; starting at epoch {start_epoch}")
  elif init_checkpoint:
    saved = torch.load(init_checkpoint, weights_only=False)
    model.load_state_dict(saved['model'])
    print(f"Initialized weights from {init_checkpoint}; starting at epoch {start_epoch}")
  else:
    print("No checkpoint found; training from pretrained GPT-2")

  return start_epoch, latest_epoch
