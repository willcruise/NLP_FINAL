#!/usr/bin/env bash
# exp13: PEMDAS → entity (stage1+2) → GSM8K — isolate entity utility (no MA).
#
# Compare vs exp12 (MA only) and exp11 (both) on GSM8K held-out.
#
# Usage:
#   bash scripts/run_exp13_ent_gsm8k.sh
#   bash scripts/run_exp13_ent_gsm8k.sh --smoke
#   SKIP_ENTITY_STAGE1=1 bash scripts/run_exp13_ent_gsm8k.sh
#
set -euo pipefail
cd "$(dirname "$0")/.."

SMOKE=0
ARITH_EXAMPLES="${ARITH_EXAMPLES:-20000}"
ARITH_EPOCHS="${ARITH_EPOCHS:-8}"
ARITH_LR="${ARITH_LR:-1e-4}"
ENT1_EPOCHS="${ENT1_EPOCHS:-8}"
ENT2_EPOCHS="${ENT2_EPOCHS:-12}"
GSM8K_EPOCHS="${GSM8K_EPOCHS:-32}"
GSM8K_LR="${GSM8K_LR:-5e-6}"
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

gsm8k_extra=()
if [[ "$SMOKE" -eq 1 ]]; then
  ARITH_EPOCHS=2
  ENT1_EPOCHS=2
  ENT2_EPOCHS=2
  GSM8K_EPOCHS=2
  gsm8k_extra+=(--eval_limit 20 --patience 99 --eval_every 2)
fi

echo "=== 0. Prepare data ==="
python prepare_arithmetic.py --num_examples "${ARITH_EXAMPLES}" --output data/arithmetic_pretrain.txt

rm_stage_checkpoints exp13_arith
rm_stage_checkpoints exp13_ent1
rm_stage_checkpoints exp13_ent2

echo ""
echo "=== Stage A: PEMDAS arithmetic (${ARITH_EPOCHS} epochs) ==="
python arithmetic_pretrain.py --use_gpu \
  --epochs "${ARITH_EPOCHS}" \
  --lr "${ARITH_LR}" \
  --batch_size 16 \
  --checkpoint_tag exp13_arith

init_ckpt="$(latest_checkpoint exp13_arith)"

if [[ "$SKIP_ENTITY_STAGE1" -eq 0 ]]; then
  echo ""
  echo "=== Stage B: Entity stage 1 — Q → Entities (${ENT1_EPOCHS} epochs) ==="
  python entity_reasoning.py train --stage 1 --use_gpu \
    --init_checkpoint "${init_ckpt}" \
    --epochs "${ENT1_EPOCHS}" \
    --lr 5e-6 \
    --checkpoint_tag exp13_ent1
  init_ckpt="$(latest_checkpoint exp13_ent1)"
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
  --checkpoint_tag exp13_ent2

init_ckpt="$(latest_checkpoint exp13_ent2)"

echo ""
echo "=== Stage D: GSM8K only (${GSM8K_EPOCHS} epochs, best on GSM8K dev) ==="
python train_with_eval.py \
  --experiment exp13 \
  --use_gpu \
  --fresh \
  --init_checkpoint "${init_ckpt}" \
  --epochs "${GSM8K_EPOCHS}" \
  --lr "${GSM8K_LR}" \
  "${gsm8k_extra[@]}"

echo ""
echo "=== Done ==="
ls -lh best_exp13*.pt 2>/dev/null || true
[[ -f outputs/exp13_ent_gsm8k/training_summary.json ]] && \
  cat outputs/exp13_ent_gsm8k/training_summary.json
