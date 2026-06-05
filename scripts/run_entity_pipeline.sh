#!/usr/bin/env bash
# Entity-scaffolded reasoning pipeline (run on server in tmux).
#
# Prerequisite: best GSM8K SFT checkpoint, e.g. 46_reasoning.pt
#   mv 47_* 48_* 49_* 55_* ~/checkpoint_backup/   # so init/resume is clean
#
set -euo pipefail
cd "$(dirname "$0")/.."

BEST_CKPT="${BEST_CKPT:-46_reasoning.pt}"
STAGE1_EPOCHS="${STAGE1_EPOCHS:-56}"   # +10 from epoch 46
STAGE2_EPOCHS="${STAGE2_EPOCHS:-66}"   # +10 more

echo "=== 1. Synthetic entity data ==="
python prepare_entity_data.py --num_examples 3000

echo "=== 2. Stage 1: Entities only (from ${BEST_CKPT}) ==="
python entity_reasoning.py train --stage 1 --use_gpu \
  --init_checkpoint "${BEST_CKPT}" \
  --epochs "${STAGE1_EPOCHS}" \
  --lr 5e-6 \
  --checkpoint_tag reasoning

echo "=== 3. Stage 2: Reasoning given Entities ==="
python entity_reasoning.py train --stage 2 --use_gpu \
  --epochs "${STAGE2_EPOCHS}" \
  --lr 3e-6 \
  --checkpoint_tag reasoning

echo "=== 4. Two-stage held-out generation ==="
python entity_reasoning.py generate --use_gpu \
  --checkpoint "$((STAGE2_EPOCHS - 1))_reasoning.pt" \
  --held_out_reasoning_path data/gsm8k_small_held_out_entities.txt \
  --reasoning_out outputs/generated_reasoning_entities.txt

echo "=== 5. Optional: MultiArith eval on latest checkpoint ==="
LATEST=$((STAGE2_EPOCHS - 1))
python eval_multiarith.py --checkpoint "${LATEST}_reasoning.pt" --use_gpu || true

echo "Done."
