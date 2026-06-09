#!/usr/bin/env bash
# MultiArith factor-ablation: isolate the marginal MA contribution of PEMDAS
# (arith init), plan (skeleton vs number-ful), and entity mixing, and try to
# reproduce yoonBot's 0.911 MA result.
#
# Every arm trains and is evaluated (greedy) on data/multiarith_dev.jsonl.
#
#   Arm  what               init          plan          entity  train file
#   M0   base (floor)       vanilla GPT2  -             no      multiarith_sft_train_aug.txt
#   M1   +PEMDAS            arith(weak)   -             no      multiarith_sft_train_aug.txt
#   M2   +plan(skeleton)    arith(weak)   number-free   no      triple/ma_plan_aug.txt
#   M3   +plan(number-ful)  arith(weak)   number-ful    no      multiarith_sft_train_plan_aug.txt
#   M4   +entity            arith(weak)   -             yes     ma_ablation/m4_aug_entity.txt
#   M5   +plan+entity       arith(weak)   number-free   yes     ma_ablation/m5_plan_entity.txt
#   M6   reproduction       arith(strong) number-ful    no      multiarith_sft_train_plan_aug.txt
#
#   Marginal effects:  M1-M0 = PEMDAS,  M2-M1 = plan(skeleton),
#                      M3-M2 = numbers in plan,  M4-M1 = entity,  M6 = 0.911 target.
#
# Usage:
#   bash scripts/run_ma_ablation.sh                 # all arms M0..M6
#   bash scripts/run_ma_ablation.sh M0 M1 M2        # subset
#   bash scripts/run_ma_ablation.sh --smoke M1      # quick sanity run
#
set -euo pipefail
cd "$(dirname "$0")/.."

SMOKE=0
ARITH_EXAMPLES="${ARITH_EXAMPLES:-20000}"
ARITH_EPOCHS="${ARITH_EPOCHS:-8}"          # weak PEMDAS (M1-M5)
ARITH_STRONG_EPOCHS="${ARITH_STRONG_EPOCHS:-30}"  # strong PEMDAS (M6, 0.911 target)
ARITH_LR="${ARITH_LR:-1e-4}"
MA_EPOCHS="${MA_EPOCHS:-40}"
MA_LR="${MA_LR:-1e-5}"
BATCH="${BATCH:-8}"
EVAL_EVERY="${EVAL_EVERY:-2}"
PATIENCE="${PATIENCE:-8}"

ARMS=()
for arg in "$@"; do
  case "$arg" in
    --smoke) SMOKE=1 ;;
    M0|M1|M2|M3|M4|M5|M6) ARMS+=("$arg") ;;
    *) echo "Unknown arg: $arg"; exit 1 ;;
  esac
done
if [[ ${#ARMS[@]} -eq 0 ]]; then
  ARMS=(M0 M1 M2 M3 M4 M5 M6)
fi

smoke_extra=()
if [[ "$SMOKE" -eq 1 ]]; then
  ARITH_EPOCHS=2
  ARITH_STRONG_EPOCHS=2
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

has_checkpoint() {
  local tag="$1"
  compgen -G "[0-9]*_${tag}.pt" > /dev/null 2>&1
}

# Keep only the highest-epoch per-epoch checkpoint for a tag (drop the rest).
# Used after staging arith so the shared init occupies ~1 file (~0.5GB), not ~5GB.
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

# After an arm finishes, drop its per-epoch checkpoints (keep best_ only) so disk
# does not grow across the 7 arms (~0.5GB retained per finished arm).
prune_arm_epochs() {
  local tag="$1"
  rm -f [0-9]*_"${tag}".pt 2>/dev/null || true
}

# --- which staged inits do the requested arms need? ---
need_arith=0
need_strong=0
for arm in "${ARMS[@]}"; do
  case "$arm" in
    M1|M2|M3|M4|M5) need_arith=1 ;;
    M6) need_strong=1 ;;
  esac
done

echo "=== 0. Prepare data (entity-mix arms M4/M5) ==="
[[ -f data/arithmetic_pretrain.txt ]] || \
  python3 prepare_arithmetic.py --num_examples "${ARITH_EXAMPLES}" --output data/arithmetic_pretrain.txt
python3 prepare_ma_ablation.py

# --- Stage A: PEMDAS arithmetic pretraining (shared init for M1-M5) ---
ARITH=""
if [[ "$need_arith" -eq 1 ]]; then
  if has_checkpoint ma_abl_arith; then
    echo ""
    echo "=== Stage A: reuse existing weak PEMDAS checkpoint (ma_abl_arith) ==="
  else
    echo ""
    echo "=== Stage A: PEMDAS arithmetic, weak (${ARITH_EPOCHS} epochs) ==="
    rm_stage_checkpoints ma_abl_arith
    python3 arithmetic_pretrain.py --use_gpu \
      --epochs "${ARITH_EPOCHS}" \
      --lr "${ARITH_LR}" \
      --batch_size 16 \
      --checkpoint_tag ma_abl_arith
  fi
  keep_latest_only ma_abl_arith
  ARITH="$(latest_checkpoint ma_abl_arith)"
  echo "weak arith init -> ${ARITH}"
fi

# --- Stage A': strong PEMDAS pretraining (M6 reproduction target) ---
ARITH_STRONG=""
if [[ "$need_strong" -eq 1 ]]; then
  if has_checkpoint ma_abl_arith_strong; then
    echo ""
    echo "=== Stage A': reuse existing strong PEMDAS checkpoint (ma_abl_arith_strong) ==="
  else
    echo ""
    echo "=== Stage A': PEMDAS arithmetic, strong (${ARITH_STRONG_EPOCHS} epochs) ==="
    rm_stage_checkpoints ma_abl_arith_strong
    python3 arithmetic_pretrain.py --use_gpu \
      --epochs "${ARITH_STRONG_EPOCHS}" \
      --lr "${ARITH_LR}" \
      --batch_size 16 \
      --checkpoint_tag ma_abl_arith_strong
  fi
  keep_latest_only ma_abl_arith_strong
  ARITH_STRONG="$(latest_checkpoint ma_abl_arith_strong)"
  echo "strong arith init -> ${ARITH_STRONG}"
fi

run_arm() {
  local arm="$1"
  local init train mask tag
  case "$arm" in
    M0) init="";             train="data/multiarith_sft_train_aug.txt";       mask="reasoning" ;;
    M1) init="$ARITH";       train="data/multiarith_sft_train_aug.txt";       mask="reasoning" ;;
    M2) init="$ARITH";       train="data/triple/ma_plan_aug.txt";             mask="reasoning" ;;
    M3) init="$ARITH";       train="data/multiarith_sft_train_plan_aug.txt";  mask="reasoning" ;;
    M4) init="$ARITH";       train="data/ma_ablation/m4_aug_entity.txt";      mask="entities_reasoning" ;;
    M5) init="$ARITH";       train="data/ma_ablation/m5_plan_entity.txt";     mask="entities_reasoning" ;;
    M6) init="$ARITH_STRONG"; train="data/multiarith_sft_train_plan_aug.txt"; mask="reasoning" ;;
  esac
  tag="ma_abl_${arm}"
  rm_stage_checkpoints "$tag"

  local cmd=(python3 train_with_eval.py --use_gpu --fresh
    --train_path "$train" --checkpoint_tag "$tag"
    --dev_path data/multiarith_dev.jsonl --greedy
    --mask_target "$mask"
    --epochs "$MA_EPOCHS" --lr "$MA_LR" --batch_size "$BATCH"
    --eval_every "$EVAL_EVERY" --patience "$PATIENCE")
  cmd+=("${smoke_extra[@]}")
  if [[ -n "$init" ]]; then cmd+=(--init_checkpoint "$init"); fi

  echo ""
  echo "=== Arm ${arm} (train=${train}, mask=${mask}, init=${init:-vanilla}) ==="
  echo "+ ${cmd[*]}"
  "${cmd[@]}"
}

for arm in "${ARMS[@]}"; do
  run_arm "$arm"
  prune_arm_epochs "ma_abl_${arm}"   # keep best_ only; free ~5GB before next arm
done

echo ""
echo "=== Done. MultiArith dev best_exact_accuracy per arm ==="
python3 -c "
import json, os
for arm in ['M0','M1','M2','M3','M4','M5','M6']:
    p = f'outputs/ma_abl_{arm}/training_summary.json'
    if not os.path.exists(p):
        continue
    s = json.load(open(p))
    print(f'{arm:3s}  acc={s[\"best_exact_accuracy\"]:.3f}  '
          f'epoch={s[\"best_epoch\"]:>2}  {os.path.basename(s[\"train_path\"])}')
"
