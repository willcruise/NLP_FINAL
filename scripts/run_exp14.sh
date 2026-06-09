#!/usr/bin/env bash
# exp14: exp10-style mix with augmented GSM8K (4500) + entity stage-1 (no Reasoning) + MA aug 600.
# Best checkpoint selected on GSM8K dev (500), same as exp10.
#
# Train mix (data/experiments/exp14_gsm8k4500_ent1_ma_aug_train.txt):
#   GSM8K SFT 4,500 + entity stage-1 3,000 + MultiArith aug ~600 (subsampled)
#
# Usage (server, tmux recommended):
#   bash scripts/run_exp14.sh
#   bash scripts/run_exp14.sh --smoke
#   MA_AUG_SUBSAMPLE=400 bash scripts/run_exp14.sh
#
set -euo pipefail
cd "$(dirname "$0")/.."

SMOKE=0
ARITH_EXAMPLES="${ARITH_EXAMPLES:-20000}"
ARITH_EPOCHS="${ARITH_EPOCHS:-8}"
ARITH_LR="${ARITH_LR:-1e-4}"
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

echo "=== 0. GSM8K SFT 4500 (former DPO pool + original 3k; dev 500 unchanged) ==="
python make_splits.py --n_sft 4500 --sft_out gsm8k_sft_train_4500.txt --sft_only

echo ""
echo "=== 1. Prepare exp14 training data (MA aug subsample=${MA_AUG_SUBSAMPLE}) ==="
export MA_AUG_SUBSAMPLE
python prepare_arithmetic.py --num_examples "${ARITH_EXAMPLES}" --output data/arithmetic_pretrain.txt
python prepare_experiment_data.py --experiments exp14

extra=()
if [[ "$SMOKE" -eq 1 ]]; then
  extra+=(--epochs 2 --eval_limit 20 --patience 99 --eval_every 2)
  ARITH_EPOCHS=2
fi

rm -f [0-9]*_exp14_arith.pt 2>/dev/null || true

echo ""
echo "=== Stage A: PEMDAS arithmetic (${ARITH_EPOCHS} epochs) ==="
python arithmetic_pretrain.py --use_gpu \
  --epochs "${ARITH_EPOCHS}" \
  --lr "${ARITH_LR}" \
  --batch_size 16 \
  --checkpoint_tag exp14_arith

init_ckpt="$(latest_checkpoint exp14_arith)"
echo ""
echo "=== Stage B: exp14 mix from ${init_ckpt} (best on GSM8K dev, mask_target=auto) ==="
python train_with_eval.py \
  --experiment exp14 \
  --use_gpu \
  --fresh \
  --init_checkpoint "${init_ckpt}" \
  "${extra[@]}"

echo ""
echo "=== Done ==="
ls -lh best_exp14*.pt 2>/dev/null || true
[[ -f outputs/exp14_gsm8k4500_ent1_ma/training_summary.json ]] && \
  cat outputs/exp14_gsm8k4500_ent1_ma/training_summary.json
