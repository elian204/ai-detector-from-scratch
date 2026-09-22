#!/usr/bin/env python3
"""Draw a blind sample of rollouts for independent hand labelling.

Why blind. The hand taxonomy is the only measurement in this project that is not
automatic, so it is the one most easily contaminated by knowing the answer. If a
rater can see the step number or the reward, "step 400, reward 0.97" invites the
label the story wants. So the texts are written out shuffled, stripped of every
number, and the key is written to a separate file the rater does not open.

Sampling is stratified across step windows so early and late behaviour are
equally represented, rather than dominated by whichever phase has more steps.

Two raters can label the same file independently and their agreement is then
worth reporting; a single rater's labels are worth much less.
"""

import argparse
import json
from pathlib import Path
import random


# Categories from the project's Phase 2 plan, plus a catch-all. A rollout can
# show several at once, so labels are a comma-separated list, not a single pick.
TAXONOMY = [
    "repetition",        # verbatim or near-verbatim repeated spans
    "gibberish",         # not well-formed language
    "fake_human",        # affected slang, typos, chattiness as "human" costume
    "topic_abandoned",   # does not answer the question asked
    "truncated",         # cut off mid-sentence or mid-word
    "list_scaffold",     # degenerates into "1. A. 2. B." enumeration frames
    "acceptable",        # a genuine, on-topic answer
]


def read_rollouts(run_dir):
    path = run_dir / "rollouts.jsonl"
    records = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not records:
        raise SystemExit(f"{path} is empty")
    return records


def stratified_sample(records, total, windows, rng):
    """Take an equal number of rollouts from each of `windows` step bands."""
    if not records:
        return []
    last_step = records[-1]["step"]
    edges = [
        (round(index * last_step / windows) + 1, round((index + 1) * last_step / windows))
        for index in range(windows)
    ]
    per_window = max(1, total // windows)

    chosen = []
    for low, high in edges:
        band = [
            (record, rollout)
            for record in records
            if low <= record["step"] <= high
            for rollout in record["rollouts"]
        ]
        if not band:
            continue
        chosen.extend(rng.sample(band, min(per_window, len(band))))
    return chosen


def build_parser():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description=__doc__,
    )
    parser.add_argument("--run-dir", type=Path, action="append", required=True,
                        help="May be repeated to pool several runs into one blind set.")
    parser.add_argument("--total", type=int, default=20)
    parser.add_argument("--windows", type=int, default=5)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--rater",
        default="rater1",
        help="Name for the blank label sheet, so two raters do not collide.",
    )
    return parser


def main():
    args = build_parser().parse_args()
    rng = random.Random(args.seed)

    pooled = []
    for run_dir in args.run_dir:
        run_dir = run_dir.expanduser().resolve()
        records = read_rollouts(run_dir)
        for record, rollout in stratified_sample(
            records, args.total // len(args.run_dir), args.windows, rng
        ):
            pooled.append((run_dir.name, record, rollout))

    rng.shuffle(pooled)

    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    # --- the blind file: text only, no run, no step, no score ---------------
    lines = [
        "# Blind rollout sample",
        "",
        "Label each item with one or more categories, comma separated:",
        "",
        *(f"- `{name}`" for name in TAXONOMY),
        "",
        "Write labels in the sheet next to this file. Do not open `key.json`",
        "until both raters are done.",
        "",
        "---",
        "",
    ]
    for index, (_, record, rollout) in enumerate(pooled, start=1):
        lines.append(f"## Item {index}")
        lines.append("")
        lines.append(f"**Question asked:** {record['question']}")
        lines.append("")
        lines.append(f"**Requested length:** about {record['target_words']} words")
        lines.append("")
        lines.append("```text")
        lines.append(rollout["text"] if rollout["text"].strip() else "(empty output)")
        lines.append("```")
        lines.append("")
    (output_dir / "blind-sample.md").write_text("\n".join(lines), encoding="utf-8")

    # --- the blank sheet ----------------------------------------------------
    sheet = ["item,labels,notes"]
    sheet.extend(f"{index}," for index in range(1, len(pooled) + 1))
    (output_dir / f"labels-{args.rater}.csv").write_text(
        "\n".join(sheet) + "\n", encoding="utf-8"
    )

    # --- the key, to be opened only after labelling -------------------------
    key = [
        {
            "item": index,
            "run": run_name,
            "step": record["step"],
            "prompt_id": record["prompt_id"],
            "target_words": record["target_words"],
            "word_count": rollout["word_count"],
            "gen_tokens": rollout["gen_tokens"],
            "hit_token_cap": rollout["hit_token_cap"],
            "ai_probability": rollout["ai_probability"],
            "human_probability": rollout["human_probability"],
            "length_score": rollout["length_score"],
            "reward": rollout["reward"],
            "advantage": rollout["advantage"],
        }
        for index, (run_name, record, rollout) in enumerate(pooled, start=1)
    ]
    (output_dir / "key.json").write_text(
        json.dumps(key, indent=2) + "\n", encoding="utf-8"
    )

    print(f"{len(pooled)} items -> {output_dir}")
    print(f"  blind-sample.md      the texts, shuffled and unlabelled")
    print(f"  labels-{args.rater}.csv{' ' * max(0, 14 - len(args.rater))}blank sheet to fill in")
    print(f"  key.json             step/reward/scores -- open only after labelling")


if __name__ == "__main__":
    main()
