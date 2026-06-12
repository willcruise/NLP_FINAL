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
