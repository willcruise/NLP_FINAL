#!/usr/bin/env bash
# exp15: exp4-style mix with planning entity (5-7 step LLM) + clean MultiArith.
# Best checkpoint on GSM8K dev (500) — combines exp4 data coverage with exp10 selection.
#
# Train mix (data/experiments/exp15_gsm8k_ma_ent_planning_train.txt):
#   GSM8K SFT 3,000 + MultiArith clean ~510 + planning entity stage2 3,000
#
# Prerequisites:
#   OPENAI_API_KEY for planning entity generation (or pre-built entity_planning_stage2_train.txt)
#
# Set API key (pick one):
#   export OPENAI_API_KEY='sk-...'
#   OPENAI_API_KEY='sk-...' bash scripts/run_exp15.sh
# Non-OpenAI compatible endpoint:
#   export OPENAI_API_KEY='...'
#   export ENTITY_BASE_URL='https://your-host/v1'
#   export ENTITY_MODEL='your-model'
#
# Usage:
#   bash scripts/run_exp15.sh
#   bash scripts/run_exp15.sh --smoke
#   SKIP_ENTITY_GEN=1 bash scripts/run_exp15.sh          # reuse existing planning entity file
#   ENTITY_RESUME=1 bash scripts/run_exp15.sh            # continue entity gen after 429
#   REQUEST_DELAY_S=2.0 bash scripts/run_exp15.sh        # slower API calls
#   ENTITY_EXAMPLES=100 bash scripts/run_exp15.sh --smoke
#
set -euo pipefail
cd "$(dirname "$0")/.."

SMOKE=0
SKIP_ENTITY_GEN="${SKIP_ENTITY_GEN:-0}"
ENTITY_EXAMPLES="${ENTITY_EXAMPLES:-3000}"
ENTITY_MODEL="${ENTITY_MODEL:-gpt-4.1}"
ENTITY_RESUME="${ENTITY_RESUME:-0}"
REQUEST_DELAY_S="${REQUEST_DELAY_S:-1.5}"
ARITH_EXAMPLES="${ARITH_EXAMPLES:-40000}"
ARITH_EPOCHS="${ARITH_EPOCHS:-12}"
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

if [[ "$SMOKE" -eq 1 ]]; then
  ENTITY_EXAMPLES=100
  extra=(--epochs 2 --eval_limit 20 --patience 99 --eval_every 2)
  ARITH_EPOCHS=2
  ARITH_EXAMPLES=2000
else
  extra=()
fi

if [[ "$SKIP_ENTITY_GEN" -eq 0 ]]; then
  if [[ -z "${OPENAI_API_KEY:-}" ]]; then
    echo "ERROR: OPENAI_API_KEY is not set."
    echo "  export OPENAI_API_KEY='sk-...'"
    echo "  bash scripts/run_exp15.sh"
    exit 1
  fi
  echo "=== 0. Planning entity data (${ENTITY_EXAMPLES} examples, ${ENTITY_MODEL}) ==="
  ENTITY_EXAMPLES="${ENTITY_EXAMPLES}" ENTITY_MODEL="${ENTITY_MODEL}" \
    ENTITY_BASE_URL="${ENTITY_BASE_URL:-https://api.openai.com/v1}" \
    ENTITY_RESUME="${ENTITY_RESUME}" REQUEST_DELAY_S="${REQUEST_DELAY_S}" \
    bash scripts/generate_entity_planning.sh
else
  echo "=== 0. Skip entity generation (SKIP_ENTITY_GEN=1) ==="
  [[ -f data/entity_planning_stage2_train.txt ]] || {
    echo "Missing data/entity_planning_stage2_train.txt"; exit 1;
  }
fi

echo ""
echo "=== 1. PEMDAS pretrain data (${ARITH_EXAMPLES} examples) + exp15 mix ==="
python3 prepare_arithmetic.py --num_examples "${ARITH_EXAMPLES}" --output data/arithmetic_pretrain.txt
python3 prepare_experiment_data.py --experiments exp15

rm -f [0-9]*_exp15_arith.pt 2>/dev/null || true

echo ""
echo "=== Stage A: PEMDAS arithmetic (${ARITH_EPOCHS} epochs, ${ARITH_EXAMPLES} examples) ==="
python3 arithmetic_pretrain.py --use_gpu \
  --epochs "${ARITH_EPOCHS}" \
  --lr "${ARITH_LR}" \
  --batch_size 16 \
  --checkpoint_tag exp15_arith

init_ckpt="$(latest_checkpoint exp15_arith)"
echo ""
echo "=== Stage B: exp15 mix from ${init_ckpt} (best on GSM8K dev) ==="
python3 train_with_eval.py \
  --experiment exp15 \
  --use_gpu \
  --fresh \
  --init_checkpoint "${init_ckpt}" \
  "${extra[@]}"

echo ""
echo "=== Done ==="
ls -lh best_exp15*.pt 2>/dev/null || true
[[ -f outputs/exp15_gsm8k_ma_ent_planning/training_summary.json ]] && \
  cat outputs/exp15_gsm8k_ma_ent_planning/training_summary.json

echo ""
echo "Held-out eval:"
echo "  python3 eval_gsm8k_heldout.py --checkpoint best_exp15_gsm8k_ma_ent_planning.pt --use_gpu"
echo "MultiArith eval:"
echo "  python3 eval_multiarith.py --checkpoint best_exp15_gsm8k_ma_ent_planning.pt --use_gpu"
