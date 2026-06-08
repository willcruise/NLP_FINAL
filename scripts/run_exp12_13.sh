#!/usr/bin/env bash
# Run exp12 (MA→GSM8K) then exp13 (entity→GSM8K) for MA vs entity ablation.
#
# Usage:
#   bash scripts/run_exp12_13.sh
#   bash scripts/run_exp12_13.sh --smoke
#
set -euo pipefail
cd "$(dirname "$0")/.."

args=()
for arg in "$@"; do
  args+=("$arg")
done

echo "========== exp12: PEMDAS → MA → GSM8K =========="
bash scripts/run_exp12_ma_gsm8k.sh "${args[@]}"

echo ""
echo "========== exp13: PEMDAS → entity → GSM8K =========="
bash scripts/run_exp13_ent_gsm8k.sh "${args[@]}"

echo ""
echo "========== Compare held-out =========="
echo "  python eval_gsm8k_heldout.py --checkpoint best_exp12_ma_gsm8k.pt --use_gpu"
echo "  python eval_gsm8k_heldout.py --checkpoint best_exp13_ent_gsm8k.pt --use_gpu"
echo "  bash scripts/eval_all_experiments_gsm8k_heldout.sh exp12 exp13 exp10 exp11"
