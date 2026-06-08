#!/usr/bin/env bash
# exp10: exp4-style arithmetic curriculum → GSM8K + entity + subsampled MA aug.
# Best checkpoint selected on GSM8K dev (500), not MultiArith.
#
# Ablation vs exp4: replace clean MultiArith (~510) with subsampled augmented MA.
# Train mix (data/experiments/exp10_gsm8k_ma_aug_sub_ent_train.txt):
#   GSM8K SFT 3,000 + entity 3,000 + MultiArith aug ~600 (subsampled)
#
# Usage (server, tmux recommended):
#   bash scripts/run_exp10.sh
#   bash scripts/run_exp10.sh --smoke
#   MA_AUG_SUBSAMPLE=400 bash scripts/run_exp10.sh
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

echo "=== 0. Prepare exp10 training data (MA aug subsample=${MA_AUG_SUBSAMPLE}) ==="
export MA_AUG_SUBSAMPLE
python prepare_arithmetic.py --num_examples "${ARITH_EXAMPLES}" --output data/arithmetic_pretrain.txt
python prepare_experiment_data.py --experiments exp10

extra=()
if [[ "$SMOKE" -eq 1 ]]; then
  extra+=(--epochs 2 --eval_limit 20 --patience 99 --eval_every 2)
  ARITH_EPOCHS=2
fi

rm -f [0-9]*_exp10_arith.pt 2>/dev/null || true

echo ""
echo "=== Stage A: PEMDAS arithmetic (${ARITH_EPOCHS} epochs) ==="
python arithmetic_pretrain.py --use_gpu \
  --epochs "${ARITH_EPOCHS}" \
  --lr "${ARITH_LR}" \
  --batch_size 16 \
  --checkpoint_tag exp10_arith

init_ckpt="$(latest_checkpoint exp10_arith)"
echo ""
echo "=== Stage B: exp10 mix from ${init_ckpt} (best on GSM8K dev) ==="
python train_with_eval.py \
  --experiment exp10 \
  --use_gpu \
  --fresh \
  --init_checkpoint "${init_ckpt}" \
  "${extra[@]}"

echo ""
echo "=== Done ==="
ls -lh best_exp10*.pt 2>/dev/null || true
[[ -f outputs/exp10_gsm8k_ma_aug_sub_ent/training_summary.json ]] && \
  cat outputs/exp10_gsm8k_ma_aug_sub_ent/training_summary.json
