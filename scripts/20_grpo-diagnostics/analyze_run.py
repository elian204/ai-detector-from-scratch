#!/usr/bin/env python3
"""Build the Phase 2 reward-hacking table from an instrumented GRPO run.

The training verifier is the same model the policy was tutored against, so its
score rising proves nothing on its own -- it is the student marking its own exam.
Reward hacking is the pattern where the *training* score rises while independent
measures of writing quality do not:

    step  ->  reward (train verifier)        rises
              P_human (train verifier)       rises
              P_human (held-out detector)    flat or falls   <- overfitting
              mean token logprob (base LM)   falls           <- degeneration
              distinct-3-gram ratio          falls           <- repetition
              |word_count - target|          grows           <- length gaming

Columns are read from `rollouts.jsonl`, which records every rollout rather than
the first three, and which carries a float32 reference verifier score that was
logged during training but never entered a gradient. That reference column is
what separates "the detector saturated" from "bfloat16 destroyed the ranking".

Held-out detection and base-LM fluency are computed here, after the fact, from
the saved texts. Nothing in this script can influence a training run.
"""

import argparse
import json
from pathlib import Path
import sys

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parents[1]


def read_rollouts(run_dir):
    path = run_dir / "rollouts.jsonl"
    if not path.is_file():
        raise SystemExit(f"No rollouts.jsonl in {run_dir}")
    records = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            records.append(json.loads(line))
    if not records:
        raise SystemExit(f"{path} is empty")
    return records


def distinct_ngram_ratio(text, n=3):
    """Unique n-grams over total n-grams: 1.0 is no repetition, lower is more.

    Computed over whitespace tokens, which is coarse but matches how the reward's
    length term counts words.
    """
    tokens = text.split()
    if len(tokens) < n:
        return None
    grams = [tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]
    return len(set(grams)) / len(grams)


def mean_or_none(values):
    values = [value for value in values if value is not None]
    return sum(values) / len(values) if values else None


def score_held_out(texts, model_name, device, batch_size):
    """Score every rollout with a detector that was never the training verifier."""
    sys.path.insert(0, str(PROJECT_DIR / "src"))
    from ai_detector import load_classifier

    classifier = load_classifier(model_name, device=device)
    # The trainer scores empty text as "."; mirror that so the columns align.
    safe = [text if text.strip() else "." for text in texts]
    return classifier.score_many(safe, batch_size=batch_size)


def score_fluency(texts, model_name, device, batch_size):
    """Mean per-token logprob of each text under the FROZEN base policy.

    This is the degeneration check. GRPO with no KL term has nothing anchoring
    it to the base model's distribution, so text can drift into strings the base
    model finds very unlikely while the detector still rates them as human.
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_name, dtype=torch.float32)
    model.to(device)
    model.eval()

    results = []
    for start in range(0, len(texts), batch_size):
        batch = [text if text.strip() else "." for text in texts[start : start + batch_size]]
        encoded = tokenizer(
            batch,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
        )
        encoded = {key: value.to(device) for key, value in encoded.items()}
        with torch.inference_mode():
            logits = model(**encoded).logits.float()
        targets = encoded["input_ids"][:, 1:]
        mask = encoded["attention_mask"][:, 1:].bool()
        logprobs = logits[:, :-1].log_softmax(dim=-1)
        chosen = logprobs.gather(2, targets.unsqueeze(-1)).squeeze(-1)
        for row, row_mask in zip(chosen, mask, strict=True):
            kept = row[row_mask]
            results.append(float(kept.mean()) if kept.numel() else None)
    return results


def summarize_window(records, held_out, fluency, offset):
    """Aggregate one window of steps into a single table row."""
    rollouts = []
    for record in records:
        rollouts.extend(record["rollouts"])

    index = offset
    held_out_values = []
    fluency_values = []
    for _ in rollouts:
        if held_out is not None:
            held_out_values.append(held_out[index])
        if fluency is not None:
            fluency_values.append(fluency[index])
        index += 1

    row = {
        "step_first": records[0]["step"],
        "step_last": records[-1]["step"],
        "rollouts": len(rollouts),
        "reward_train": mean_or_none([r["reward"] for r in rollouts]),
        "human_train": mean_or_none([r["human_probability"] for r in rollouts]),
        "human_train_fp32": mean_or_none([r["human_probability_fp32"] for r in rollouts]),
        "reward_fp32": mean_or_none([r["reward_fp32"] for r in rollouts]),
        "p_ai_exactly_one_frac": mean_or_none(
            [1.0 if r["ai_probability"] == 1.0 else 0.0 for r in rollouts]
        ),
        "word_count": mean_or_none([r["word_count"] for r in rollouts]),
        "length_score": mean_or_none([r["length_score"] for r in rollouts]),
        "abs_word_error": mean_or_none(
            [
                abs(r["word_count"] - record["target_words"])
                for record in records
                for r in record["rollouts"]
                if record["target_words"]
            ]
        ),
        "distinct_3gram": mean_or_none(
            [distinct_ngram_ratio(r["text"]) for r in rollouts]
        ),
        # Rollout diversity. GRPO's entire signal is the spread of rewards WITHIN
        # a group, so a policy that samples the same text four times produces
        # zero advantage and no gradient -- not because there is nothing to
        # learn, but because it has stopped exploring. Observed in run B: at
        # step 192 all four rollouts were byte-identical despite temperature 0.8
        # and top_p 0.9.
        "distinct_texts_per_group": mean_or_none(
            [len({r["text"] for r in record["rollouts"]}) for record in records]
        ),
        "identical_group_frac": mean_or_none(
            [
                1.0 if len({r["text"] for r in record["rollouts"]}) == 1 else 0.0
                for record in records
            ]
        ),
        "empty_frac": mean_or_none([1.0 if r["is_empty"] else 0.0 for r in rollouts]),
        "cap_hit_frac": mean_or_none(
            [1.0 if r["hit_token_cap"] else 0.0 for r in rollouts]
        ),
        "zero_adv_frac": mean_or_none(
            [1.0 if record["is_zero_advantage"] else 0.0 for record in records]
        ),
        "stepped_frac": mean_or_none(
            [1.0 if record["optimizer_stepped"] else 0.0 for record in records]
        ),
        "grad_norm_preclip": mean_or_none(
            [record["grad_norm_preclip"] for record in records]
        ),
        "moved_since_start": records[-1]["movement"]["moved_since_start_fraction"],
        "mean_abs_dw": records[-1]["movement"]["mean_abs_change_since_start"],
        "step_seconds": mean_or_none([record["step_seconds"] for record in records]),
    }
    if held_out is not None:
        row["ai_held_out"] = mean_or_none(held_out_values)
        row["human_held_out"] = mean_or_none(
            [1.0 - value for value in held_out_values if value is not None]
        )
    if fluency is not None:
        row["base_logprob"] = mean_or_none(fluency_values)
    return row, index


def build_parser():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description=__doc__,
    )
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument(
        "--window",
        type=int,
        default=50,
        help="Steps aggregated per table row.",
    )
    parser.add_argument(
        "--held-out-model",
        default="logreg",
        help="Detector that was NOT the training verifier. Use 'none' to skip.",
    )
    parser.add_argument("--held-out-device", default="cpu")
    parser.add_argument(
        "--fluency-model",
        default="none",
        help=(
            "Frozen LM for the degeneration check, e.g. Qwen/Qwen3-0.6B-Base. "
            "Use 'none' to skip (it needs a GPU and a few minutes)."
        ),
    )
    parser.add_argument("--fluency-device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--output-prefix", type=Path, default=None)
    return parser


def main():
    args = build_parser().parse_args()
    run_dir = args.run_dir.expanduser().resolve()
    records = read_rollouts(run_dir)
    texts = [r["text"] for record in records for r in record["rollouts"]]
    print(f"{run_dir.name}: {len(records)} steps, {len(texts)} rollouts", file=sys.stderr)

    held_out = None
    if args.held_out_model.lower() != "none":
        print(f"scoring held-out detector: {args.held_out_model}", file=sys.stderr)
        held_out = score_held_out(
            texts, args.held_out_model, args.held_out_device, args.batch_size
        )

    fluency = None
    if args.fluency_model.lower() != "none":
        print(f"scoring fluency under: {args.fluency_model}", file=sys.stderr)
        fluency = score_fluency(
            texts, args.fluency_model, args.fluency_device, args.batch_size
        )

    rows = []
    offset = 0
    for start in range(0, len(records), args.window):
        window = records[start : start + args.window]
        row, offset = summarize_window(window, held_out, fluency, offset)
        rows.append(row)

    prefix = args.output_prefix or (run_dir / "phase2")
    prefix.parent.mkdir(parents=True, exist_ok=True)

    import csv

    csv_path = prefix.with_name(prefix.name + "-table.csv")
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    # Compact markdown view of the columns the hacking claim rests on.
    headline = [
        ("steps", lambda r: f"{r['step_first']}-{r['step_last']}"),
        ("reward", lambda r: f"{r['reward_train']:.4f}"),
        ("human_train", lambda r: f"{r['human_train']:.4f}"),
        ("human_fp32", lambda r: f"{r['human_train_fp32']:.4f}"),
        ("P_AI==1", lambda r: f"{r['p_ai_exactly_one_frac']:.2f}"),
        ("words", lambda r: f"{r['word_count']:.1f}"),
        ("dist3g", lambda r: f"{r['distinct_3gram']:.3f}" if r["distinct_3gram"] else "-"),
        ("cap%", lambda r: f"{r['cap_hit_frac']:.2f}"),
        ("uniq/grp", lambda r: f"{r['distinct_texts_per_group']:.2f}"),
        ("collapsed", lambda r: f"{r['identical_group_frac']:.2f}"),
        ("moved", lambda r: f"{r['moved_since_start']:.3f}"),
        ("|dw|", lambda r: f"{r['mean_abs_dw']:.2e}"),
    ]
    if held_out is not None:
        headline.insert(4, ("human_heldout", lambda r: f"{r['human_held_out']:.4f}"))
    if fluency is not None:
        headline.insert(-2, ("base_logprob", lambda r: f"{r['base_logprob']:.3f}"))

    lines = [
        "| " + " | ".join(name for name, _ in headline) + " |",
        "| " + " | ".join("---" for _ in headline) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(render(row) for _, render in headline) + " |")
    table = "\n".join(lines)

    md_path = prefix.with_name(prefix.name + "-table.md")
    md_path.write_text(table + "\n", encoding="utf-8")
    print(table)
    print(f"\nwrote {csv_path}\nwrote {md_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
