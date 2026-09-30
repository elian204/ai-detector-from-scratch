#!/usr/bin/env python3
"""Plot the reward-hacking panels from one or more analyzed GRPO runs.

Input is the `*-table.csv` written by `analyze_run.py`, so the aggregation has a
single source of truth and the figure cannot disagree with the table.

The four panels are chosen to make the argument falsifiable at a glance:

  1. Detector scores. The training verifier and a held-out detector on the same
     texts. If the training curve rises and the held-out curve does not, that is
     verifier overfitting. If both rise, the exploit transfers -- a stronger
     claim, and the one our runs actually support.
  2. Text degeneration. Distinct-3-gram ratio (falling = more repetition) and the
     fraction of rollouts truncated at the token cap (rising = the policy has
     stopped emitting EOS).
  3. What the reward is actually made of. Once (1 - P_AI) saturates near 1.0, the
     reward equals the length term, so these two curves converging on each other
     shows the objective collapsing onto a word-count heuristic.
  4. Whether the optimizer is doing anything. Fraction of probed parameters that
     have moved, and the fraction of steps skipped for zero advantage. This is
     what distinguishes "the algorithm found nothing" from "the arithmetic could
     not represent the update" or "there was no signal left to learn from".
"""

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def read_table(path):
    with Path(path).open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise SystemExit(f"{path} is empty")

    def column(name):
        values = []
        for row in rows:
            raw = row.get(name, "")
            values.append(float(raw) if raw not in ("", None) else None)
        return values

    # Plot against the midpoint of each aggregation window.
    steps = [
        (float(row["step_first"]) + float(row["step_last"])) / 2.0 for row in rows
    ]
    return steps, column


def build_parser():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description=__doc__,
    )
    parser.add_argument(
        "--table",
        action="append",
        required=True,
        metavar="LABEL=PATH",
        help="Repeatable, e.g. --table 'bfloat16=run-A-bf16/phase2-table.csv'",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--title", default="No-KL GRPO against a frozen AI-text detector")
    parser.add_argument("--dpi", type=int, default=150)
    return parser


def main():
    args = build_parser().parse_args()

    series = []
    for entry in args.table:
        if "=" not in entry:
            raise SystemExit(f"--table needs LABEL=PATH, got: {entry}")
        label, path = entry.split("=", 1)
        series.append((label, *read_table(path)))

    figure, axes = plt.subplots(2, 2, figsize=(13, 8.5))
    styles = ["-", "--", ":", "-."]

    def draw(axis, specs, title, ylabel, ylim=None):
        for index, (label, steps, column) in enumerate(series):
            style = styles[index % len(styles)]
            for name, pretty, colour in specs:
                values = column(name)
                pairs = [(s, v) for s, v in zip(steps, values) if v is not None]
                if not pairs:
                    continue
                axis.plot(
                    [s for s, _ in pairs],
                    [v for _, v in pairs],
                    style,
                    color=colour,
                    label=f"{pretty} ({label})",
                    linewidth=1.9,
                )
        axis.set_title(title, fontsize=11)
        axis.set_xlabel("training step")
        axis.set_ylabel(ylabel)
        if ylim:
            axis.set_ylim(*ylim)
        axis.grid(alpha=0.25)
        axis.legend(fontsize=7.5, loc="best")

    draw(
        axes[0][0],
        [
            ("human_train", "P_human, training verifier", "#c0392b"),
            ("human_held_out", "P_human, held-out detector", "#2471a3"),
            ("reward_train", "reward", "#7d3c98"),
        ],
        "1. Both detectors are fooled: the exploit transfers",
        "probability / reward",
        (0, 1.05),
    )
    draw(
        axes[0][1],
        [
            ("distinct_3gram", "distinct-3-gram ratio", "#b9770e"),
            ("cap_hit_frac", "fraction truncated at token cap", "#196f3d"),
            ("identical_group_frac", "groups fully mode-collapsed", "#884ea0"),
        ],
        "2. Degeneration: repetition, truncation, mode collapse",
        "ratio / fraction",
        (0, 1.05),
    )
    draw(
        axes[1][0],
        [
            ("reward_train", "reward", "#7d3c98"),
            ("length_score", "length term alone", "#d68910"),
        ],
        "3. The reward collapses onto the length heuristic",
        "reward",
        (0, 1.05),
    )
    draw(
        axes[1][1],
        [
            ("moved_since_start", "fraction of params moved", "#1a5276"),
            ("zero_adv_frac", "fraction of steps with zero advantage", "#922b21"),
        ],
        "4. Is the optimizer doing anything?",
        "fraction",
        (-0.03, 1.05),
    )

    figure.suptitle(args.title, fontsize=13)
    figure.tight_layout(rect=(0, 0, 1, 0.97))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, dpi=args.dpi)
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
