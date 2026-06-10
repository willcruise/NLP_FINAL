#!/usr/bin/env bash
# Evaluate MA ablation checkpoints on GSM8K small held-out (200 prompts, train idx 5000+).
#
# Usage:
#   bash scripts/eval_ma_abl_gsm8k_heldout.sh           # M3 only (default)
#   bash scripts/eval_ma_abl_gsm8k_heldout.sh M0 M3 M6   # subset
#   bash scripts/eval_ma_abl_gsm8k_heldout.sh --smoke  # first 5 examples
#
set -euo pipefail
cd "$(dirname "$0")/.."

SMOKE=0
LIMIT=0
ONLY=()

for arg in "$@"; do
  case "$arg" in
    --smoke) SMOKE=1 ;;
    M[0-6]) ONLY+=("$arg") ;;
    *) echo "Unknown arg: $arg (use M0..M6 or --smoke)"; exit 1 ;;
  esac
done

if [[ "$SMOKE" -eq 1 ]]; then
  LIMIT=5
fi

if [[ ${#ONLY[@]} -eq 0 ]]; then
  ONLY=(M3)
fi

checkpoint_for() {
  case "$1" in
    M0) echo "best_ma_abl_M0.pt" ;;
    M1) echo "best_ma_abl_M1.pt" ;;
    M2) echo "best_ma_abl_M2.pt" ;;
    M3) echo "best_ma_abl_M3.pt" ;;
    M4) echo "best_ma_abl_M4.pt" ;;
    M5) echo "best_ma_abl_M5.pt" ;;
    M6) echo "best_ma_abl_M6.pt" ;;
    *) echo ""; return 1 ;;
  esac
}

run_one() {
  local arm="$1"
  local ckpt
  ckpt="$(checkpoint_for "$arm")" || { echo "Unknown arm: $arm"; return 1; }
  if [[ ! -f "$ckpt" ]]; then
    echo "=== Skip $arm: missing $ckpt ==="
    return 0
  fi
  echo "=== Evaluating $arm ($ckpt) on GSM8K held-out (greedy) ==="
  cmd=(python3 eval_gsm8k_heldout.py --checkpoint "$ckpt" --use_gpu --greedy)
  if [[ "$LIMIT" -gt 0 ]]; then
    cmd+=(--limit "$LIMIT")
  fi
  "${cmd[@]}"
}

for arm in "${ONLY[@]}"; do
  run_one "$arm"
done

echo "Done. Metrics -> outputs/best_ma_abl_*_gsm8k_heldout_metrics.json"
