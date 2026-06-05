#!/usr/bin/env bash
# Remove corrupted / overfit checkpoints and start a new training run.
# Run from repo root: bash scripts/fresh_start.sh

set -euo pipefail

echo "Removing unified checkpoints..."
rm -fv *_reasoning.pt *_reasoning.pt.tmp 2>/dev/null || true

echo "Removing legacy checkpoints..."
rm -fv *_arithmetic-*.pt *_*-reasoning.pt 2>/dev/null || true

echo "Done. Disk usage:"
ls -lh *_reasoning.pt 2>/dev/null || echo "  (no *_reasoning.pt files remain)"

echo ""
echo "Suggested fresh pipeline:"
echo "  python prepare_arithmetic.py --num_examples 50000"
echo "  python arithmetic_pretrain.py --use_gpu --epochs 10 --lr 1e-4 --checkpoint_tag reasoning"
echo "  python reasoning_generation.py --use_gpu --epochs 60 --lr 5e-6 --checkpoint_tag reasoning"
