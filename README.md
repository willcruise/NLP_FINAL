# Sogang NLP Final Project: Fine-tuning GPT-2 for Chain-of-Thought Reasoning

Fine-tune GPT-2 on GSM8K, MultiArith, and synthetic entity/arithmetic data. Experiments ablate training mixes, PEMDAS pretraining, planning entities, and step-level process supervision.

## Setup

```bash
conda env create -f env.yml
conda activate nlp   # or your env name
bash setup.sh
```

Checkpoints (`.pt`), generation logs (`.txt`), and run logs are gitignored. Experiment **metrics JSON** under `outputs/` are tracked.

## Project layout

| Path | Description |
|------|-------------|
| `train_with_eval.py` | Main SFT loop with dev eval, best-checkpoint tracking, early stopping |
| `arithmetic_pretrain.py` | PEMDAS arithmetic curriculum (Stage A for exp4, exp7, exp10–15, MA ablation) |
| `prepare_experiment_data.py` | Build merged training files in `data/experiments/` |
| `prepare_entity_data.py` | Entity reasoning data (stage 1/2, planning) |
| `prepare_arithmetic.py` | Synthetic PEMDAS pretraining data |
| `prepare_ma_ablation.py` | MultiArith factor-ablation train files |
| `eval_multiarith.py` | MultiArith dev eval (greedy) |
| `eval_gsm8k_heldout.py` | GSM8K small held-out eval (200 prompts, idx 5000+) |
| `train_step_sft.py` / `train_step_dpo.py` | Step-conditioned SFT and step-level DPO |
| `scripts/` | End-to-end experiment runners and batch eval |
| `data/experiments/` | Merged SFT training corpora per experiment |
| `outputs/` | Per-run metrics (`training_summary.json`, `*_metrics.json`) |

## Experiments

### Data ablations (exp1–exp3)

| Exp | Training data | Script |
|-----|---------------|--------|
| exp1 | GSM8K only | `scripts/run_three_experiments.sh exp1` |
| exp2 | GSM8K + MultiArith | `scripts/run_three_experiments.sh exp2` |
| exp3 | GSM8K + MultiArith + entity | `scripts/run_three_experiments.sh exp3` |

Dev eval: MultiArith (`data/multiarith_dev.jsonl`).

### Curriculum & mix (exp4–exp7, exp10–15)

| Exp | Description | Script |
|-----|-------------|--------|
| exp4 | PEMDAS (8 ep) → exp3 mix | `scripts/run_overnight_experiments.sh exp4` |
| exp5 | GSM8K + MA aug + entity | `scripts/run_overnight_experiments.sh exp5` |
| exp6 | exp3 mix + subsampled PEMDAS | `scripts/run_overnight_experiments.sh exp6` |
| exp7 | PEMDAS → GSM8K + entity + subsampled MA | `scripts/run_exp7.sh` |
| exp10 | exp4-style → GSM8K + entity + MA aug (best on GSM8K dev) | `scripts/run_exp10.sh` |
| exp11 | Sequential: PEMDAS → entity → MA → GSM8K | `scripts/run_exp11.sh` |
| exp12 | PEMDAS → MA → GSM8K (no entity) | `scripts/run_exp12_ma_gsm8k.sh` |
| exp13 | PEMDAS → entity → GSM8K (no MA) | `scripts/run_exp13_ent_gsm8k.sh` |
| exp14 | GSM8K 4500 + entity stage-1 + MA aug | `scripts/run_exp14.sh` |
| exp15 | exp4 mix + planning entity (LLM) | `scripts/run_exp15.sh` |

### Process supervision (exp8–exp9)

| Exp | Description | Script |
|-----|-------------|--------|
| exp8 | Step-conditioned SFT | `scripts/run_process_pipeline.sh` (SFT stage) |
| exp9 | Step-level DPO | `scripts/run_process_pipeline.sh` (DPO stage) |

### MultiArith ablations

| Study | Description | Script |
|-------|-------------|--------|
| MA ablation M0–M6 | PEMDAS / plan / entity marginal effects | `scripts/run_ma_ablation.sh` |
| Plan position 2×2 | Skeleton inline vs outside × numbers | `scripts/run_plan_position_ablation.sh` |

## Experiment configurations & results

All runs start from pretrained **GPT-2 (124M)** unless an init checkpoint is given. Training uses **AdamW**, `seed=11711`, prompt tokens masked from the loss (`--mask_prompt`), and best-checkpoint selection by dev `exact_accuracy` with early stopping.

> "Best epoch / Ran to" = epoch of the best checkpoint vs. last epoch executed (early stop may cut before the configured max). "Best acc" is dev `exact_accuracy` at the best epoch.

### Common SFT defaults (`train_with_eval.py`)

| Setting | Value |
|---------|-------|
| Optimizer | AdamW |
| Batch size | 8 |
| Seed | 11711 |
| Sampling (eval) | greedy, or `temperature=0.7`, `top_p=0.9`, `max_new_tokens=256` |
| Base model | gpt2 (124M) |

### PEMDAS arithmetic pretraining (`arithmetic_pretrain.py`, Stage A)

| Setting | Value |
|---------|-------|
| Examples | 20,000 (`prepare_arithmetic.py`); exp15: 40,000 |
| Epochs | 8 (weak); 12 (exp15); 30 (MA M6 "strong") |
| LR | 1e-4 |
| Batch size | 16 |

### Data ablations (exp1–exp3) — dev: MultiArith (90)

| Exp | Command | Epochs (max) | eval_every | patience | LR | Best epoch / Ran to | Best acc |
|-----|---------|-------------|-----------|----------|----|--------------------|----------|
| exp1 | `bash scripts/run_three_experiments.sh exp1` | 30 | 2 | 5 | 5e-6 | 4 / 14 (early stop) | 0.033 |
| exp2 | `bash scripts/run_three_experiments.sh exp2` | 30 | 2 | 5 | 5e-6 | 28 / 29 | 0.133 |
| exp3 | `bash scripts/run_three_experiments.sh exp3` | 32 | 2 | 5 | 5e-6 | 30 / 31 | 0.289 |

### Curriculum & mix (exp4–exp7) — dev: MultiArith (exp4–6), GSM8K (exp7)

| Exp | Command | Stages (epochs @ lr) | SFT max ep | Best ep / Ran to | Best acc |
|-----|---------|----------------------|-----------|-----------------|----------|
| exp4 | `bash scripts/run_overnight_experiments.sh exp4` | PEMDAS 8@1e-4 → exp3 mix | 32 | 28 / 31 | **0.711** |
| exp5 | `bash scripts/run_overnight_experiments.sh exp5` | GSM8K+MA-aug+entity | 36 (patience 6) | 35 / 35 | 0.511 |
| exp6 | `bash scripts/run_overnight_experiments.sh exp6` | exp3 mix + 3k subsampled PEMDAS | 34 | 33 / 33 | 0.533 |
| exp7 | `bash scripts/run_exp7.sh` | PEMDAS 8@1e-4 → GSM8K+entity+sub-MA | 32 (eval 4) | 28 / 31 | 0.024 |

exp4–6 use `eval_every=2`, `patience=5`, `lr=5e-6`. exp7 uses `eval_every=4`, `patience=5`, `lr=5e-6` and selects on GSM8K dev (harder, lower acc).

### Sequential / isolation pipelines (exp10–exp15) — dev: GSM8K (500)

All Stage-E GSM8K SFT: `epochs=32`, `eval_every=4`, `patience=5`, `lr=5e-6`.

| Exp | Command | Stages (epochs @ lr) | Best ep / Ran to | Best acc |
|-----|---------|----------------------|-----------------|----------|
| exp10 | `bash scripts/run_exp10.sh` | PEMDAS 8@1e-4 → GSM8K+entity+MA-aug(600) mix | 20 / 31 | 0.042 |
| exp11 | `bash scripts/run_exp11.sh` | PEMDAS 8@1e-4 → ent1 8@5e-6 → ent2 12@5e-6 → MA-aug-sub 8@5e-6 → GSM8K | 0 / 20 (early stop) | 0.020 |
| exp12 | `bash scripts/run_exp12_ma_gsm8k.sh` | PEMDAS 8 → MA 8@5e-6 → GSM8K (no entity) | 20 / 31 | 0.026 |
| exp13 | `bash scripts/run_exp13_ent_gsm8k.sh` | PEMDAS 8 → ent1 8 → ent2 12 → GSM8K (no MA) | 16 / 31 | 0.024 |
| exp14 | `bash scripts/run_exp14.sh` | PEMDAS 8 → GSM8K-4500 + entity-stage1 + MA-aug(600), `mask_target=auto` | 24 / 31 | 0.034 |
| exp15 | `OPENAI_API_KEY=... bash scripts/run_exp15.sh` | PEMDAS 12@1e-4 (40k) + planning-entity (3k, LLM) mix | (not recorded) | — |

### Process supervision (exp8–exp9) — dev: GSM8K (500)

Run both phases: `bash scripts/run_process_pipeline.sh`

| Exp | Phase | Init | Epochs | eval_every | patience | LR | extra | Best ep / Ran to | Best acc |
|-----|-------|------|--------|-----------|----------|----|-------|-----------------|----------|
| exp8 | step-conditioned SFT (`train_step_sft.py`) | `best_exp4_gsm8k_ma_ent.pt` | 10 | 2 | 4 | 2e-6 | batch 8 | 6 / 9 | 0.036 |
| exp9 | step-level DPO (`train_step_dpo.py`) | `best_exp8_step_sft.pt` | 4 | 1 | 3 | 1e-6 | batch 4, `beta=0.2` | 2 / 3 | 0.012 |

### MultiArith factor ablation (M0–M6) — dev: MultiArith, greedy

`bash scripts/run_ma_ablation.sh` — SFT: `epochs=40`, `lr=1e-5`, `batch=8`, `eval_every=2`, `patience=8`.

| Arm | Init (PEMDAS) | Train data | mask_target | Best ep / Ran to | Best acc |
|-----|---------------|-----------|-------------|-----------------|----------|
| M0 | vanilla GPT-2 | `multiarith_sft_train_aug` | reasoning | 38 / 39 | 0.789 |
| M1 | weak (8 ep) | `multiarith_sft_train_aug` | reasoning | 22 / 38 | 0.933 |
| M2 | weak (8 ep) | `triple/ma_plan_aug` (skeleton, inline) | reasoning | 36 / 39 | 0.900 |
| M3 | weak (8 ep) | `multiarith_sft_train_plan_aug` (number-ful) | reasoning | 32 / 39 | **1.000** |
| M4 | weak (8 ep) | `ma_ablation/m4_aug_entity` | entities_reasoning | 20 / 36 | 0.989 |
| M5 | weak (8 ep) | `ma_ablation/m5_plan_entity` | entities_reasoning | 39 / 39 | 0.922 |
| M6 | strong (30 ep) | `multiarith_sft_train_plan_aug` | reasoning | 28 / 39 | 0.989 |

Marginal effects: `M1−M0`=PEMDAS, `M2−M1`=plan(skeleton), `M3−M2`=numbers in plan, `M4−M1`=entity, `M6`=strong-PEMDAS reproduction target.

### Plan position 2×2 ablation — dev: MultiArith, greedy

`bash scripts/run_plan_position_ablation.sh` — shared weak PEMDAS (8 ep) init; SFT `epochs=40`, `lr=1e-5`, `batch=8`, `eval_every=2`, `patience=8`. Arms: `P_SO` (skeleton/outside), `P_SI` (skeleton/inline), `P_NO` (number-ful/outside), `P_NI` (number-ful/inline).

### Triple-stage runs — dev: MultiArith, greedy

| Tag | Init | Train data | Best ep / Ran to | Best acc |
|-----|------|-----------|-----------------|----------|
| triple_ma | PEMDAS 8 ep | `triple/ma_plan_aug` | 34 / 39 | 0.344 |
| triple_full | `best_triple_ma` | `triple/stage2_full` | 19 / 19 | 0.032 |
| triple_full_ext | `best_triple_full_ep19` | `triple/stage2_full` | 0 / 12 | 0.030 |

### Entity pipeline (legacy two-stage)

`bash scripts/run_entity_pipeline.sh` — Stage 1 (Q→Entities) `lr=5e-6`, Stage 2 (Reasoning|Entities) `lr=3e-6`, 3,000 synthetic examples.

## Evaluation

```bash
# MultiArith dev
python eval_multiarith.py --checkpoint best_exp4_gsm8k_ma_ent.pt --use_gpu

# GSM8K held-out (all main experiments)
bash scripts/eval_all_experiments_gsm8k_heldout.sh

# MA ablation checkpoints on GSM8K held-out
bash scripts/eval_ma_abl_gsm8k_heldout.sh
```

Metrics are written to `outputs/*_metrics.json` and per-run folders under `outputs/exp*/`.

## Smoke tests

Most scripts accept `--smoke` for a short sanity run (few epochs / limited eval).
