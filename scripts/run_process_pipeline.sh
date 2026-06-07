#!/usr/bin/env bash
# Process supervision pipeline: step-conditioned SFT (Phase 1) + step DPO (Phase 2).
#
# Starts from best_exp4_gsm8k_ma_ent.pt (exp4 best).
# Best checkpoints selected on gsm8k_dev (500).
#
# Usage:
#   bash scripts/run_process_pipeline.sh
#   bash scripts/run_process_pipeline.sh --smoke
#
set -euo pipefail
cd "$(dirname "$0")/.."

SMOKE=0
INIT_CKPT="${INIT_CKPT:-best_exp4_gsm8k_ma_ent.pt}"

for arg in "$@"; do
  case "$arg" in
    --smoke) SMOKE=1 ;;
    *) echo "Unknown arg: $arg"; exit 1 ;;
  esac
done

EXTRA=()
if [[ "$SMOKE" -eq 1 ]]; then
  EXTRA+=(--eval_limit 20 --epochs 2)
  DPO_EXTRA=(--eval_limit 20 --epochs 1)
else
  DPO_EXTRA=()
fi

echo "=== 0. Prepare step SFT + DPO data ==="
python3 prepare_step_sft.py
python3 prepare_step_dpo.py

echo "=== 1. Phase 1: step-conditioned SFT (from ${INIT_CKPT}) ==="
python3 train_step_sft.py \
  --use_gpu \
  --init_checkpoint "${INIT_CKPT}" \
  "${EXTRA[@]}"

echo "=== 2. Phase 2: step-level DPO (from best_exp8_step_sft.pt) ==="
python3 train_step_dpo.py \
  --use_gpu \
  --init_checkpoint best_exp8_step_sft.pt \
  "${DPO_EXTRA[@]}"

echo "=== Done ==="
echo "Phase 1 best: best_exp8_step_sft.pt"
echo "Phase 2 best: best_exp9_step_dpo.pt"
