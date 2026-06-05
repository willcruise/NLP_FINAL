#!/usr/bin/env bash
# Overnight experiments exp4, exp5, exp6 (MultiArith dev eval + best checkpoint each).
#
#  exp4  arithmetic curriculum (20k PEMDAS, 8 ep) → exp3 mix SFT (~2–3 h total)
#  exp5  GSM8K + MultiArith augmented + entity (~3–4 h)
#  exp6  exp3 mix + 3000 subsampled arithmetic (~3–4 h)
#
# Estimated total: ~8–12 h on a single GPU (run in tmux).
#
# Usage:
#   tmux new -s overnight
#   bash scripts/run_overnight_experiments.sh
#   bash scripts/run_overnight_experiments.sh exp5
#   bash scripts/run_overnight_experiments.sh --smoke
#
set -euo pipefail
cd "$(dirname "$0")/.."

SMOKE=0
ONLY=""
ARITH_EXAMPLES="${ARITH_EXAMPLES:-20000}"
ARITH_EPOCHS="${ARITH_EPOCHS:-8}"
ARITH_LR="${ARITH_LR:-1e-4}"
ARITH_MIX="${ARITH_MIX:-3000}"

for arg in "$@"; do
  case "$arg" in
    --smoke) SMOKE=1 ;;
    exp4|exp5|exp6) ONLY="$arg" ;;
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

echo "=== 0. Prepare shared data ==="
python prepare_arithmetic.py --num_examples "${ARITH_EXAMPLES}" --output data/arithmetic_pretrain.txt
python prepare_experiment_data.py --experiments exp3,exp5,exp6

run_exp4() {
  echo ""
  echo "=========================================="
  echo "=== exp4: arithmetic curriculum → exp3 ==="
  echo "=========================================="

  local extra=()
  if [[ "$SMOKE" -eq 1 ]]; then
    extra+=(--epochs 2 --eval_limit 10 --patience 99)
    ARITH_EPOCHS=2
  fi

  rm -f [0-9]*_exp4_arith.pt 2>/dev/null || true

  echo "--- Stage A: PEMDAS arithmetic (${ARITH_EPOCHS} epochs) ---"
  python arithmetic_pretrain.py --use_gpu \
    --epochs "${ARITH_EPOCHS}" \
    --lr "${ARITH_LR}" \
    --batch_size 16 \
    --checkpoint_tag exp4_arith

  local init_ckpt
  init_ckpt="$(latest_checkpoint exp4_arith)"
  echo "--- Stage B: exp3 mix from ${init_ckpt} ---"

  python train_with_eval.py \
    --experiment exp4 \
    --use_gpu \
    --fresh \
    --init_checkpoint "${init_ckpt}" \
    "${extra[@]}"
}

run_exp5() {
  echo ""
  echo "=========================================="
  echo "=== exp5: GSM8K + MA aug + entity ==="
  echo "=========================================="

  local extra=()
  if [[ "$SMOKE" -eq 1 ]]; then
    extra+=(--epochs 1 --eval_limit 10 --patience 99)
  fi

  python train_with_eval.py \
    --experiment exp5 \
    --use_gpu \
    --fresh \
    "${extra[@]}"
}

run_exp6() {
  echo ""
  echo "=========================================="
  echo "=== exp6: exp3 + arithmetic mix ==="
  echo "=========================================="

  local extra=()
  if [[ "$SMOKE" -eq 1 ]]; then
    extra+=(--epochs 1 --eval_limit 10 --patience 99)
  fi

  python train_with_eval.py \
    --experiment exp6 \
    --use_gpu \
    --fresh \
    "${extra[@]}"
}

if [[ -n "$ONLY" ]]; then
  case "$ONLY" in
    exp4) run_exp4 ;;
    exp5) run_exp5 ;;
    exp6) run_exp6 ;;
  esac
else
  run_exp4
  run_exp5
  run_exp6
fi

echo ""
echo "=== Best checkpoints ==="
ls -lh best_exp4*.pt best_exp5*.pt best_exp6*.pt 2>/dev/null || true
echo ""
echo "=== Summaries ==="
for f in outputs/exp4_gsm8k_ma_ent/training_summary.json \
         outputs/exp5_gsm8k_ma_aug_ent/training_summary.json \
         outputs/exp6_gsm8k_ma_ent_arith/training_summary.json; do
  [[ -f "$f" ]] && echo "$f" && cat "$f" && echo ""
done

echo "Done."
