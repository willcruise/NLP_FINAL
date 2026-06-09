#!/usr/bin/env bash
# Evaluate best exp1–exp6, exp7, exp10–exp11 checkpoints on GSM8K small held-out.
#
# Usage (server):
#   bash scripts/eval_all_experiments_gsm8k_heldout.sh
#   bash scripts/eval_all_experiments_gsm8k_heldout.sh --smoke
#   bash scripts/eval_all_experiments_gsm8k_heldout.sh exp4 exp6
#
set -euo pipefail
cd "$(dirname "$0")/.."

SMOKE=0
LIMIT=0
EXTRA_ARGS=()
ONLY=()

for arg in "$@"; do
  case "$arg" in
    --smoke) SMOKE=1 ;;
    --greedy) EXTRA_ARGS+=(--greedy) ;;
    exp[1-6]|exp7|exp10|exp11|exp14|exp15) ONLY+=("$arg") ;;
    *) echo "Unknown arg: $arg"; exit 1 ;;
  esac
done

if [[ "$SMOKE" -eq 1 ]]; then
  LIMIT=5
fi

declare -A CHECKPOINTS=(
  [exp1]="best_exp1_gsm8k.pt"
  [exp2]="best_exp2_gsm8k_ma.pt"
  [exp3]="best_exp3_gsm8k_ma_ent.pt"
  [exp4]="best_exp4_gsm8k_ma_ent.pt"
  [exp5]="best_exp5_gsm8k_ma_aug_ent.pt"
  [exp6]="best_exp6_gsm8k_ma_ent_arith.pt"
  [exp7]="best_exp7_gsm8k_ma_sub_ent.pt"
  [exp10]="best_exp10_gsm8k_ma_aug_sub_ent.pt"
  [exp11]="best_exp11_gsm8k.pt"
  [exp14]="best_exp14_gsm8k4500_ent1_ma.pt"
  [exp15]="best_exp15_gsm8k_ma_ent_planning.pt"
)

run_one() {
  local tag="$1"
  local ckpt="${CHECKPOINTS[$tag]}"
  if [[ ! -f "$ckpt" ]]; then
    echo "=== Skip $tag: missing $ckpt ==="
    return 0
  fi
  echo "=== Evaluating $tag ($ckpt) on GSM8K held-out ==="
  cmd=(python eval_gsm8k_heldout.py --checkpoint "$ckpt" --use_gpu "${EXTRA_ARGS[@]}")
  if [[ "$LIMIT" -gt 0 ]]; then
    cmd+=(--limit "$LIMIT")
  fi
  "${cmd[@]}"
}

if [[ ${#ONLY[@]} -gt 0 ]]; then
  for tag in "${ONLY[@]}"; do
    run_one "$tag"
  done
else
  for tag in exp1 exp2 exp3 exp4 exp5 exp6 exp7 exp10 exp11 exp14 exp15; do
    run_one "$tag"
  done
fi

echo "=== Done. Metrics in outputs/*_gsm8k_heldout_metrics.json ==="
