#!/usr/bin/env bash
# Plan position × numbers 2×2 ablation on MultiArith dev (greedy).
#
# Isolates the confound in old M2 vs M3 (skeleton/outside vs number-ful/inline).
# All arms share weak PEMDAS init (ma_abl_arith) and lr/batch with MA ablation.
#
#   Arm   plan text   position    train file                          prior result
#   P_SO  skeleton    outside     ma_ablation/plan_skel_outside.txt   M2  34.4%
#   P_SI  skeleton    inside      ma_ablation/plan_skel_inline.txt     NEW
#   P_NO  number-ful  outside     ma_ablation/plan_num_outside.txt    NEW
#   P_NI  number-ful  inside      multiarith_sft_train_plan_aug.txt    M3 100.0%
#
# Marginal reads (after this run):
#   P_SI − P_SO  = position effect (skeleton, no numbers)
#   P_NI − P_NO  = position effect (number-ful)
#   P_SI − P_NI  = number effect (both inline)
#   P_SO − P_NO  = number effect (both outside)
#
# Usage:
#   bash scripts/run_plan_position_ablation.sh              # all 4 arms
#   bash scripts/run_plan_position_ablation.sh P_SI P_NO  # only new arms
#   bash scripts/run_plan_position_ablation.sh --smoke P_SI
#
set -euo pipefail
cd "$(dirname "$0")/.."

SMOKE=0
ARITH_EXAMPLES="${ARITH_EXAMPLES:-20000}"
ARITH_EPOCHS="${ARITH_EPOCHS:-8}"
ARITH_LR="${ARITH_LR:-1e-4}"
MA_EPOCHS="${MA_EPOCHS:-40}"
MA_LR="${MA_LR:-1e-5}"
BATCH="${BATCH:-8}"
EVAL_EVERY="${EVAL_EVERY:-2}"
PATIENCE="${PATIENCE:-8}"
# Optional: skip arith pretrain and use an existing .pt (e.g. 7_ma_abl_arith.pt)
INIT_CHECKPOINT="${INIT_CHECKPOINT:-}"

ARMS=()
for arg in "$@"; do
  case "$arg" in
    --smoke) SMOKE=1 ;;
    P_SO|P_SI|P_NO|P_NI) ARMS+=("$arg") ;;
    *) echo "Unknown arg: $arg"; exit 1 ;;
  esac
done
if [[ ${#ARMS[@]} -eq 0 ]]; then
  ARMS=(P_SO P_SI P_NO P_NI)
fi

smoke_extra=()
if [[ "$SMOKE" -eq 1 ]]; then
  ARITH_EPOCHS=2
  MA_EPOCHS=2
  smoke_extra+=(--eval_limit 20 --patience 99 --eval_every 1)
fi

latest_checkpoint() {
  local tag="$1"
  python3 -c "
import glob, os, sys
tag = sys.argv[1]
paths = [p for p in glob.glob(f'*_{tag}.pt')
         if os.path.basename(p).split('_', 1)[0].isdigit()]
if not paths:
    raise SystemExit(f'No checkpoint found for tag {tag}')
print(max(paths, key=lambda p: int(os.path.basename(p).split('_', 1)[0])))
" "$tag"
}

rm_stage_checkpoints() {
  local tag="$1"
  rm -f [0-9]*_"${tag}".pt best_"${tag}".pt 2>/dev/null || true
}

prune_arm_epochs() {
  local tag="$1"
  rm -f [0-9]*_"${tag}".pt 2>/dev/null || true
}

has_checkpoint() {
  local tag="$1"
  compgen -G "[0-9]*_${tag}.pt" > /dev/null 2>&1
}

keep_latest_only() {
  local tag="$1"
  python3 -c "
import glob, os, sys
tag = sys.argv[1]
paths = [p for p in glob.glob(f'[0-9]*_{tag}.pt')
         if os.path.basename(p).split('_', 1)[0].isdigit()]
if not paths:
    raise SystemExit(0)
keep = max(paths, key=lambda p: int(os.path.basename(p).split('_', 1)[0]))
for p in paths:
    if p != keep:
        os.remove(p); print(f'  pruned {p}')
" "$tag"
}

echo "=== 0. Build plan-position data ==="
python3 prepare_plan_position_ablation.py

echo ""
echo "=== 1. Weak PEMDAS init ==="
ARITH=""
if [[ -n "$INIT_CHECKPOINT" ]]; then
  ARITH="$INIT_CHECKPOINT"
  echo "Using INIT_CHECKPOINT -> ${ARITH}"
elif has_checkpoint ma_abl_arith; then
  ARITH="$(latest_checkpoint ma_abl_arith)"
  echo "Reusing ma_abl_arith -> ${ARITH}"
else
  echo "ma_abl_arith not found; running weak PEMDAS (${ARITH_EPOCHS} epochs)..."
  [[ -f data/arithmetic_pretrain.txt ]] || \
    python3 prepare_arithmetic.py --num_examples "${ARITH_EXAMPLES}" \
      --output data/arithmetic_pretrain.txt
  rm_stage_checkpoints ma_abl_arith
  python3 arithmetic_pretrain.py --use_gpu \
    --epochs "${ARITH_EPOCHS}" \
    --lr "${ARITH_LR}" \
    --batch_size 16 \
    --checkpoint_tag ma_abl_arith
  keep_latest_only ma_abl_arith
  ARITH="$(latest_checkpoint ma_abl_arith)"
  echo "trained ma_abl_arith -> ${ARITH}"
fi

run_arm() {
  local arm="$1" train tag
  case "$arm" in
    P_SO) train="data/ma_ablation/plan_skel_outside.txt" ;;
    P_SI) train="data/ma_ablation/plan_skel_inline.txt" ;;
    P_NO) train="data/ma_ablation/plan_num_outside.txt" ;;
    P_NI) train="data/multiarith_sft_train_plan_aug.txt" ;;
  esac
  tag="plan_abl_${arm}"
  rm_stage_checkpoints "$tag"

  local cmd=(python3 train_with_eval.py --use_gpu --fresh
    --train_path "$train" --checkpoint_tag "$tag"
    --dev_path data/multiarith_dev.jsonl --greedy
    --mask_target reasoning
    --init_checkpoint "$ARITH"
    --epochs "$MA_EPOCHS" --lr "$MA_LR" --batch_size "$BATCH"
    --eval_every "$EVAL_EVERY" --patience "$PATIENCE")
  cmd+=("${smoke_extra[@]}")

  echo ""
  echo "=== Arm ${arm} (train=${train}) ==="
  echo "+ ${cmd[*]}"
  "${cmd[@]}"
}

for arm in "${ARMS[@]}"; do
  run_arm "$arm"
  prune_arm_epochs "plan_abl_${arm}"
done

echo ""
echo "=== Done. MultiArith dev best_exact_accuracy ==="
python3 -c "
import json, os

# include prior MA-ablation M2/M3 for side-by-side
prior = {'P_SO': ('M2', 'outputs/ma_abl_M2/training_summary.json'),
         'P_NI': ('M3', 'outputs/ma_abl_M3/training_summary.json')}
rows = []
for arm in ['P_SO','P_SI','P_NO','P_NI']:
    p = f'outputs/plan_abl_{arm}/training_summary.json'
    if os.path.exists(p):
        s = json.load(open(p))
        rows.append((arm, s['best_exact_accuracy'], s['best_epoch'], 'this run'))
    elif arm in prior:
        label, pp = prior[arm]
        if os.path.exists(pp):
            s = json.load(open(pp))
            rows.append((arm, s['best_exact_accuracy'], s['best_epoch'], f'prior {label}'))

print(f'{\"Arm\":<5} {\"Acc\":>7} {\"Ep\":>4}  source')
for arm, acc, ep, src in rows:
    print(f'{arm:<5} {acc*100:6.1f}% {ep:>4}  {src}')

if len(rows) == 4:
    d = {a: acc for a, acc, _, _ in rows}
    print()
    print('Marginal effects:')
    print(f'  P_SI-P_SO position (skeleton):  {(d[\"P_SI\"]-d[\"P_SO\"])*100:+.1f}%p')
    print(f'  P_NI-P_NO position (number):    {(d[\"P_NI\"]-d[\"P_NO\"])*100:+.1f}%p')
    print(f'  P_SI-P_NI numbers (inline):       {(d[\"P_SI\"]-d[\"P_NI\"])*100:+.1f}%p')
    print(f'  P_SO-P_NO numbers (outside):      {(d[\"P_SO\"]-d[\"P_NO\"])*100:+.1f}%p')
"
