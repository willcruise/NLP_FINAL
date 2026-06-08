#!/usr/bin/env bash
# exp11: Sequential curriculum (no mix) — same data budget as exp10, different order.
#
#   Stage A: PEMDAS arithmetic (20k, 8 ep)
#   Stage B: Entity stage 1 — Q → Entities (entity_stage1, 3k)
#   Stage C: Entity stage 2 — Entities → Reasoning (entity_stage2, 3k)
#   Stage D: Aug MA subsample only (600)
#   Stage E: GSM8K only (3k) — best checkpoint on GSM8K dev (500)
#
# Compare vs exp10 (mix, held-out 5.0%) on:
#   python eval_gsm8k_heldout.py --checkpoint best_exp11_gsm8k.pt --use_gpu
#
# Ablation note (exp1–exp4 are MIXED SFT, not sequential):
#   exp1 vs exp2  → effect of adding clean MA (no entity)
#   exp2 vs exp3  → effect of adding entity (MA present in both)
#   exp3 vs exp4  → effect of arithmetic curriculum (same mix file)
#   exp10 vs exp11 → mix vs sequential (same sources, different order)
# Entity-only / MA-only without GSM8K were NOT run in exp1–exp4.
#
# Usage:
#   bash scripts/run_exp11.sh
#   bash scripts/run_exp11.sh --smoke
#   SKIP_ENTITY_STAGE1=1 bash scripts/run_exp11.sh   # stage2 only (faster)
#
set -euo pipefail
cd "$(dirname "$0")/.."

SMOKE=0
ARITH_EXAMPLES="${ARITH_EXAMPLES:-20000}"
ARITH_EPOCHS="${ARITH_EPOCHS:-8}"
ARITH_LR="${ARITH_LR:-1e-4}"
ENT1_EPOCHS="${ENT1_EPOCHS:-8}"
ENT2_EPOCHS="${ENT2_EPOCHS:-12}"
MA_EPOCHS="${MA_EPOCHS:-8}"
MA_LR="${MA_LR:-5e-6}"
GSM8K_EPOCHS="${GSM8K_EPOCHS:-32}"
GSM8K_LR="${GSM8K_LR:-5e-6}"
MA_AUG_SUBSAMPLE="${MA_AUG_SUBSAMPLE:-600}"
SKIP_ENTITY_STAGE1="${SKIP_ENTITY_STAGE1:-0}"

for arg in "$@"; do
  case "$arg" in
    --smoke) SMOKE=1 ;;
    *) echo "Unknown arg: $arg"; exit 1 ;;
  esac
done

latest_checkpoint() {
  local tag="$1"
  python3 -c "
import glob, os, sys
tag = sys.argv[1]
paths = glob.glob(f'*_{tag}.pt')
if not paths:
    raise SystemExit(f'No checkpoint found for tag {tag}')
print(max(paths, key=lambda p: int(os.path.basename(p).split('_', 1)[0])))
" "$tag"
}

rm_stage_checkpoints() {
  local tag="$1"
  rm -f [0-9]*_"${tag}".pt best_"${tag}".pt 2>/dev/null || true
}

echo "=== 0. Prepare data ==="
export MA_AUG_SUBSAMPLE
python prepare_arithmetic.py --num_examples "${ARITH_EXAMPLES}" --output data/arithmetic_pretrain.txt
python prepare_experiment_data.py --experiments exp11_ma

gsm8k_extra=()
if [[ "$SMOKE" -eq 1 ]]; then
  ENT1_EPOCHS=2
  ENT2_EPOCHS=2
  MA_EPOCHS=2
  GSM8K_EPOCHS=2
  ARITH_EPOCHS=2
  gsm8k_extra+=(--epochs 2 --eval_limit 20 --patience 99 --eval_every 2)
fi

rm_stage_checkpoints exp11_arith
rm_stage_checkpoints exp11_ent1
rm_stage_checkpoints exp11_ent2
rm_stage_checkpoints exp11_ma

echo ""
echo "=== Stage A: PEMDAS arithmetic (${ARITH_EPOCHS} epochs) ==="
python arithmetic_pretrain.py --use_gpu \
  --epochs "${ARITH_EPOCHS}" \
  --lr "${ARITH_LR}" \
  --batch_size 16 \
  --checkpoint_tag exp11_arith

init_ckpt="$(latest_checkpoint exp11_arith)"

if [[ "$SKIP_ENTITY_STAGE1" -eq 0 ]]; then
  echo ""
  echo "=== Stage B: Entity stage 1 — Q → Entities (${ENT1_EPOCHS} epochs) ==="
  python entity_reasoning.py train --stage 1 --use_gpu \
    --init_checkpoint "${init_ckpt}" \
    --epochs "${ENT1_EPOCHS}" \
    --lr 5e-6 \
    --checkpoint_tag exp11_ent1
  init_ckpt="$(latest_checkpoint exp11_ent1)"
else
  echo ""
  echo "=== Stage B: skipped (SKIP_ENTITY_STAGE1=1) ==="
fi

echo ""
echo "=== Stage C: Entity stage 2 — Reasoning (${ENT2_EPOCHS} epochs) ==="
python entity_reasoning.py train --stage 2 --use_gpu \
  --init_checkpoint "${init_ckpt}" \
  --epochs "${ENT2_EPOCHS}" \
  --lr 5e-6 \
  --checkpoint_tag exp11_ent2

init_ckpt="$(latest_checkpoint exp11_ent2)"

echo ""
echo "=== Stage D: Aug MA subsample only (${MA_EPOCHS} epochs) ==="
python reasoning_generation.py --use_gpu \
  --skip_submission \
  --mask_prompt \
  --init_checkpoint "${init_ckpt}" \
  --reasoning_path data/experiments/exp11_ma_aug_sub_train.txt \
  --epochs "${MA_EPOCHS}" \
  --lr "${MA_LR}" \
  --checkpoint_tag exp11_ma

init_ckpt="$(latest_checkpoint exp11_ma)"

echo ""
echo "=== Stage E: GSM8K only (${GSM8K_EPOCHS} epochs, best on GSM8K dev) ==="
python train_with_eval.py \
  --experiment exp11 \
  --use_gpu \
  --fresh \
  --init_checkpoint "${init_ckpt}" \
  --epochs "${GSM8K_EPOCHS}" \
  --lr "${GSM8K_LR}" \
  "${gsm8k_extra[@]}"

echo ""
echo "=== Done ==="
ls -lh best_exp11*.pt 2>/dev/null || true
[[ -f outputs/exp11_gsm8k/training_summary.json ]] && \
  cat outputs/exp11_gsm8k/training_summary.json
