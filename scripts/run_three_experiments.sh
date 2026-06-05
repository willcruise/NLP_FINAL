#!/usr/bin/env bash
# Three from-scratch SFT experiments with MultiArith eval + best checkpoint tracking.
#
#  exp1  pretrained GPT-2 + GSM8K only
#  exp2  pretrained GPT-2 + GSM8K + MultiArith
#  exp3  pretrained GPT-2 + GSM8K + MultiArith + entity
#
# Each run keeps best_{tag}.pt by dev exact_accuracy and early-stops on plateau.
#
# Usage (server, tmux recommended):
#   bash scripts/run_three_experiments.sh
#   bash scripts/run_three_experiments.sh exp2
#   bash scripts/run_three_experiments.sh --smoke
#
set -euo pipefail
cd "$(dirname "$0")/.."

SMOKE=0
ONLY=""

for arg in "$@"; do
  case "$arg" in
    --smoke) SMOKE=1 ;;
    exp1|exp2|exp3) ONLY="$arg" ;;
    *) echo "Unknown arg: $arg"; exit 1 ;;
  esac
done

echo "=== 0. Build merged training files ==="
python prepare_experiment_data.py

run_exp() {
  local name="$1"
  echo ""
  echo "=========================================="
  echo "=== Training ${name} ==="
  echo "=========================================="

  local extra=()
  if [[ "$SMOKE" -eq 1 ]]; then
    extra+=(--epochs 1 --eval_limit 10 --patience 99)
  fi

  python train_with_eval.py \
    --experiment "${name}" \
    --use_gpu \
    --fresh \
    "${extra[@]}"
}

if [[ -n "$ONLY" ]]; then
  run_exp "$ONLY"
else
  run_exp exp1
  run_exp exp2
  run_exp exp3
fi

echo ""
echo "=== Best checkpoints ==="
ls -lh best_exp*.pt 2>/dev/null || true
echo ""
echo "=== Summaries ==="
for f in outputs/exp*/training_summary.json; do
  [[ -f "$f" ]] && echo "$f" && cat "$f" && echo ""
done

echo "Done."
