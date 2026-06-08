#!/usr/bin/env bash
# exp12: PEMDAS → aug MA (600) → GSM8K — isolate MA utility (no entity).
#
# Compare vs exp13 (entity only) and exp11 (both) on GSM8K held-out.
#
# Usage:
#   bash scripts/run_exp12_ma_gsm8k.sh
#   bash scripts/run_exp12_ma_gsm8k.sh --smoke
#
set -euo pipefail
cd "$(dirname "$0")/.."

SMOKE=0
ARITH_EXAMPLES="${ARITH_EXAMPLES:-20000}"
ARITH_EPOCHS="${ARITH_EPOCHS:-8}"
ARITH_LR="${ARITH_LR:-1e-4}"
MA_EPOCHS="${MA_EPOCHS:-8}"
MA_LR="${MA_LR:-5e-6}"
GSM8K_EPOCHS="${GSM8K_EPOCHS:-32}"
GSM8K_LR="${GSM8K_LR:-5e-6}"
MA_AUG_SUBSAMPLE="${MA_AUG_SUBSAMPLE:-600}"

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
  MA_EPOCHS=2
  GSM8K_EPOCHS=2
  gsm8k_extra+=(--eval_limit 20 --patience 99 --eval_every 2)
fi

echo "=== 0. Prepare data ==="
export MA_AUG_SUBSAMPLE
python prepare_arithmetic.py --num_examples "${ARITH_EXAMPLES}" --output data/arithmetic_pretrain.txt
python prepare_experiment_data.py --experiments exp11_ma

rm_stage_checkpoints exp12_arith
rm_stage_checkpoints exp12_ma

echo ""
echo "=== Stage A: PEMDAS arithmetic (${ARITH_EPOCHS} epochs) ==="
python arithmetic_pretrain.py --use_gpu \
  --epochs "${ARITH_EPOCHS}" \
  --lr "${ARITH_LR}" \
  --batch_size 16 \
  --checkpoint_tag exp12_arith

init_ckpt="$(latest_checkpoint exp12_arith)"

echo ""
echo "=== Stage B: Aug MA subsample only (${MA_EPOCHS} epochs) ==="
python reasoning_generation.py --use_gpu \
  --skip_submission \
  --mask_prompt \
  --init_checkpoint "${init_ckpt}" \
  --reasoning_path data/experiments/exp11_ma_aug_sub_train.txt \
  --epochs "${MA_EPOCHS}" \
  --lr "${MA_LR}" \
  --checkpoint_tag exp12_ma

init_ckpt="$(latest_checkpoint exp12_ma)"

echo ""
echo "=== Stage C: GSM8K only (${GSM8K_EPOCHS} epochs, best on GSM8K dev) ==="
python train_with_eval.py \
  --experiment exp12 \
  --use_gpu \
  --fresh \
  --init_checkpoint "${init_ckpt}" \
  --epochs "${GSM8K_EPOCHS}" \
  --lr "${GSM8K_LR}" \
  "${gsm8k_extra[@]}"

echo ""
echo "=== Done ==="
ls -lh best_exp12*.pt 2>/dev/null || true
[[ -f outputs/exp12_ma_gsm8k/training_summary.json ]] && \
  cat outputs/exp12_ma_gsm8k/training_summary.json
