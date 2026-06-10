#!/usr/bin/env bash
# Rebuild skeleton plan data (Plan inside Reasoning) and rerun MA-ablation arms.
#
# Usage:
#   bash scripts/run_m2_skeleton_inline.sh           # M2 + M5
#   bash scripts/run_m2_skeleton_inline.sh M2        # M2 only
#   bash scripts/run_m2_skeleton_inline.sh M5        # M5 only
#   bash scripts/run_m2_skeleton_inline.sh --smoke M5
#
set -euo pipefail
cd "$(dirname "$0")/.."

ARMS=(M2 M5)
EXTRA=()
for arg in "$@"; do
  case "$arg" in
    --smoke) EXTRA=(--smoke) ;;
    M2|M5) ARMS=("$arg") ;;
    *) echo "Unknown arg: $arg  (use M2, M5, or --smoke)"; exit 1 ;;
  esac
done

echo "=== 1. Regenerate triple data (skeleton plan inline) ==="
python3 prepare_triple_data.py

echo ""
echo "=== 2. Regenerate M5 mix (ma_plan_aug + entity_plan) ==="
python3 prepare_ma_ablation.py

echo ""
echo "=== 3. Rerun ${ARMS[*]} ==="
bash scripts/run_ma_ablation.sh "${EXTRA[@]}" "${ARMS[@]}"

echo ""
echo "=== Results ==="
python3 -c "
import json, os
for arm in ['M2','M3','M5']:
    p = f'outputs/ma_abl_{arm}/training_summary.json'
    if os.path.exists(p):
        s = json.load(open(p))
        print(f'{arm}: acc={s[\"best_exact_accuracy\"]*100:.1f}%  ep={s[\"best_epoch\"]}')
"
