#!/usr/bin/env bash
# exp7: exp4-style arithmetic curriculum → GSM8K-heavy mix (MA subsampled).
# Best checkpoint selected on GSM8K dev (500), not MultiArith.
#
# Train mix (data/experiments/exp7_gsm8k_ma_sub_ent_train.txt):
#   GSM8K SFT 3,000 + entity 3,000 + MultiArith 150 (subsampled from 543)
#
# Usage (server, tmux recommended):
#   bash scripts/run_exp7.sh
#   bash scripts/run_exp7.sh --smoke
#
set -euo pipefail
cd "$(dirname "$0")/.."

SMOKE=0
ARITH_EXAMPLES="${ARITH_EXAMPLES:-20000}"
ARITH_EPOCHS="${ARITH_EPOCHS:-8}"
ARITH_LR="${ARITH_LR:-1e-4}"

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

echo "=== 0. Prepare exp7 training data ==="
python prepare_arithmetic.py --num_examples "${ARITH_EXAMPLES}" --output data/arithmetic_pretrain.txt
python prepare_experiment_data.py --experiments exp7

extra=()
if [[ "$SMOKE" -eq 1 ]]; then
  extra+=(--epochs 2 --eval_limit 20 --patience 99 --eval_every 2)
  ARITH_EPOCHS=2
fi

rm -f [0-9]*_exp7_arith.pt 2>/dev/null || true

echo ""
echo "=== Stage A: PEMDAS arithmetic (${ARITH_EPOCHS} epochs) ==="
python arithmetic_pretrain.py --use_gpu \
  --epochs "${ARITH_EPOCHS}" \
  --lr "${ARITH_LR}" \
  --batch_size 16 \
  --checkpoint_tag exp7_arith

init_ckpt="$(latest_checkpoint exp7_arith)"
echo ""
echo "=== Stage B: exp7 mix from ${init_ckpt} (best on GSM8K dev) ==="
python train_with_eval.py \
  --experiment exp7 \
  --use_gpu \
  --fresh \
  --init_checkpoint "${init_ckpt}" \
  "${extra[@]}"

echo ""
echo "=== Done ==="
ls -lh best_exp7*.pt 2>/dev/null || true
[[ -f outputs/exp7_gsm8k_ma_sub_ent/training_summary.json ]] && \
  cat outputs/exp7_gsm8k_ma_sub_ent/training_summary.json
