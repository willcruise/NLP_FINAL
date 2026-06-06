#!/usr/bin/env python3
"""
Evaluate a trained ReasoningGPT checkpoint on the GSM8K small held-out set.

Prompts: data/gsm8k_small_held_out.txt  (GSM8K train indices 5000+)
Gold:    data/gsm8k_small_held_out.jsonl if present, else HuggingFace gsm8k main train.

Usage:
    python eval_gsm8k_heldout.py --checkpoint best_exp4_gsm8k_ma_ent.pt --use_gpu
    python eval_gsm8k_heldout.py --checkpoint best_exp1_gsm8k.pt --write_gold_jsonl

Outputs:
    outputs/<name>_gsm8k_heldout_metrics.json
    outputs/<name>_gsm8k_heldout_generations.txt
"""

from __future__ import annotations

import argparse
import json
import os
import re

import torch

from reasoning_generation import ReasoningGPT, seed_everything
import gsm8k_eval

HELD_OUT_PROMPTS = os.path.join("data", "gsm8k_small_held_out.txt")
HELD_OUT_GOLD = os.path.join("data", "gsm8k_small_held_out.jsonl")
HELD_OUT_HF_OFFSET = 5000


def load_held_out_prompts(path: str) -> list[tuple[int, str]]:
    """Return list of (id, prompt_text) from the held-out prompt file."""
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()
    blocks = re.split(r"\n\s*\d+\s*\n", text)
    blocks = [b.strip() for b in blocks if b.strip()]
    return list(enumerate(blocks))


def load_gold_jsonl(path: str) -> dict[int, float]:
    gold = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            gold[rec["id"]] = rec["gold_answer"]
    return gold


def load_gold_from_hf(num_examples: int, offset: int = HELD_OUT_HF_OFFSET) -> dict[int, float]:
    from datasets import load_dataset

    ds = load_dataset("gsm8k", "main")["train"].select(range(offset, offset + num_examples))
    gold = {}
    for i, ex in enumerate(ds):
        gold[i] = gsm8k_eval.extract_gold_answer(ex["answer"])
    return gold


def write_gold_jsonl(
    prompts: list[tuple[int, str]],
    gold: dict[int, float],
    path: str,
) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for idx, prompt in prompts:
            m = re.search(r"Question:\s*(.+?)\n\nReasoning:", prompt, re.DOTALL)
            question = m.group(1).strip() if m else prompt
            rec = {"id": idx, "question": question, "gold_answer": gold[idx]}
            f.write(json.dumps(rec) + "\n")


def resolve_gold(
    prompts: list[tuple[int, str]],
    gold_jsonl: str | None,
    hf_offset: int,
    write_gold_jsonl_flag: bool,
) -> dict[int, float]:
    path = gold_jsonl or HELD_OUT_GOLD
    if os.path.isfile(path):
        gold = load_gold_jsonl(path)
        print(f"Loaded gold from {path}")
    else:
        print(
            f"No gold file at {path}; loading from HuggingFace gsm8k main "
            f"train[{hf_offset}:{hf_offset + len(prompts)}]..."
        )
        gold = load_gold_from_hf(len(prompts), offset=hf_offset)

    missing = [idx for idx, _ in prompts if gold.get(idx) is None]
    if missing:
        raise ValueError(f"Missing gold for held-out ids: {missing[:5]}{'...' if len(missing) > 5 else ''}")

    if write_gold_jsonl_flag:
        out_path = gold_jsonl or HELD_OUT_GOLD
        write_gold_jsonl(prompts, gold, out_path)
        print(f"Wrote gold labels -> {out_path}")

    return gold


def main():
    args = get_args()
    seed_everything(args.seed)
    device = torch.device("cuda") if args.use_gpu else torch.device("cpu")

    if os.path.isdir(args.checkpoint):
        from transformers import GPT2LMHeadModel, GPT2Tokenizer

        model = GPT2LMHeadModel.from_pretrained(args.checkpoint).to(device)
        tokenizer = GPT2Tokenizer.from_pretrained(args.checkpoint)
        tokenizer.pad_token = tokenizer.eos_token
        use_hf = True
    else:
        saved = torch.load(args.checkpoint, weights_only=False)
        model_args = saved["args"]
        model = ReasoningGPT(model_args).to(device)
        model.load_state_dict(saved["model"])
        model.eval()
        use_hf = False

    prompt_path = args.held_out_path
    prompts = load_held_out_prompts(prompt_path)
    if args.limit:
        prompts = prompts[: args.limit]

    gold = resolve_gold(
        prompts,
        args.gold_jsonl,
        args.hf_offset,
        args.write_gold_jsonl,
    )

    print(f"Evaluating on {len(prompts)} GSM8K held-out examples...")

    eval_records = []
    gen_lines = []

    for idx, prompt in prompts:
        if use_hf:
            enc = tokenizer(
                prompt, return_tensors="pt", truncation=True, max_length=512
            ).to(device)
            with torch.no_grad():
                out_ids = model.generate(
                    **enc,
                    max_new_tokens=args.max_new_tokens,
                    do_sample=not args.greedy,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    pad_token_id=tokenizer.eos_token_id,
                )
            generated = tokenizer.decode(out_ids[0], skip_special_tokens=True)
        else:
            enc = model.tokenizer(
                prompt, return_tensors="pt", truncation=True, max_length=512
            ).to(device)
            _, generated = model.generate(
                enc["input_ids"],
                temperature=args.temperature,
                top_p=args.top_p,
                max_length=args.max_new_tokens,
                sample=not args.greedy,
            )

        continuation = generated[len(prompt):] if generated.startswith(prompt) else generated
        eval_records.append({"generation": continuation, "gold": gold[idx]})
        gen_lines.append(
            f"\n=== held-out {idx} (gold={gold[idx]}) ===\n{generated}\n"
        )

    metrics = gsm8k_eval.evaluate(eval_records)
    metrics["checkpoint"] = args.checkpoint
    metrics["dataset"] = "gsm8k_small_held_out"
    metrics["num_examples"] = len(prompts)
    metrics["held_out_path"] = prompt_path

    name = os.path.splitext(os.path.basename(args.checkpoint.rstrip("/\\")))[0]
    os.makedirs("outputs", exist_ok=True)
    met_path = os.path.join("outputs", f"{name}_gsm8k_heldout_metrics.json")
    gen_path = os.path.join("outputs", f"{name}_gsm8k_heldout_generations.txt")

    with open(met_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    with open(gen_path, "w", encoding="utf-8") as f:
        f.writelines(gen_lines)

    print(json.dumps(metrics, indent=2))
    print(f"Metrics     -> {met_path}")
    print(f"Generations -> {gen_path}")


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True,
                        help=".pt file or HuggingFace model directory")
    parser.add_argument("--held_out_path", type=str, default=HELD_OUT_PROMPTS,
                        help="Prompt-only held-out file (default: data/gsm8k_small_held_out.txt)")
    parser.add_argument("--gold_jsonl", type=str, default=None,
                        help="Optional gold labels jsonl; defaults to data/gsm8k_small_held_out.jsonl")
    parser.add_argument("--hf_offset", type=int, default=HELD_OUT_HF_OFFSET,
                        help="GSM8K train start index when loading gold from HuggingFace")
    parser.add_argument("--write_gold_jsonl", action="store_true",
                        help="Write resolved gold labels to --gold_jsonl or default path")
    parser.add_argument("--use_gpu", action="store_true")
    parser.add_argument("--seed", type=int, default=11711)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--greedy", action="store_true",
                        help="Use greedy decoding instead of nucleus sampling.")
    parser.add_argument("--limit", type=int, default=0,
                        help="If >0, only evaluate first N held-out items (smoke test).")
    return parser.parse_args()


if __name__ == "__main__":
    main()
