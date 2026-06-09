#!/usr/bin/env bash
# Generate 5-7 step planning entity data with a strong LLM (OpenAI-compatible API).
#
# Requires: OPENAI_API_KEY (or pass --api_key to prepare_entity_data.py)
#
# Set API key:
#   export OPENAI_API_KEY='sk-...'
#   bash scripts/generate_entity_planning.sh
#
# Usage:
#   bash scripts/generate_entity_planning.sh
#   ENTITY_EXAMPLES=500 bash scripts/generate_entity_planning.sh   # smoke
#   ENTITY_RESUME=1 bash scripts/generate_entity_planning.sh       # continue after 429
#   REQUEST_DELAY_S=2.0 bash scripts/generate_entity_planning.sh   # slower, fewer 429s
#
set -euo pipefail
cd "$(dirname "$0")/.."

ENTITY_EXAMPLES="${ENTITY_EXAMPLES:-3000}"
ENTITY_MODEL="${ENTITY_MODEL:-gpt-4.1}"
ENTITY_BASE_URL="${ENTITY_BASE_URL:-https://api.openai.com/v1}"
ENTITY_RESUME="${ENTITY_RESUME:-0}"
REQUEST_DELAY_S="${REQUEST_DELAY_S:-1.5}"

if [[ -z "${OPENAI_API_KEY:-}" ]]; then
  echo "ERROR: OPENAI_API_KEY is not set."
  echo "  export OPENAI_API_KEY='sk-...'"
  exit 1
fi

resume_args=()
if [[ "${ENTITY_RESUME}" -eq 1 ]]; then
  resume_args+=(--resume)
fi

echo "=== Generate planning entity (${ENTITY_EXAMPLES} examples, model=${ENTITY_MODEL}, delay=${REQUEST_DELAY_S}s) ==="
python3 prepare_entity_data.py \
  --profile planning \
  --generator llm \
  --num_examples "${ENTITY_EXAMPLES}" \
  --model "${ENTITY_MODEL}" \
  --base_url "${ENTITY_BASE_URL}" \
  --request_delay_s "${REQUEST_DELAY_S}" \
  --skip_held_out \
  "${resume_args[@]}"

echo "=== Done ==="
ls -lh data/entity_planning_stage2_train.txt
