&nbsp;
# Robustness and calibration evaluation

[← Stage 18: Reinforcement learning](../18_reinforcement-learning/) · [Project index](../../README.md)

This extension turns a set of detector scores into a reproducible evaluation
report. It is designed for held-out and out-of-distribution datasets, where a
single accuracy value can hide class imbalance, poor calibration, or a failure
on one source or text-length range.

The report includes:

- confusion-matrix metrics at a frozen decision threshold;
- ROC AUC and average precision, which do not depend on one threshold;
- Brier score, log loss, equal-width reliability bins, ECE, and maximum
  calibration error;
- a fixed threshold curve from 0 to 1;
- word-count and user-selected categorical slices; and
- optional row-level predictions for inspecting errors.

These metrics characterize a labeled dataset. They do not turn the detector
score into proof of authorship.

&nbsp;
## Input format

Use a CSV, JSONL, or NDJSON file with at least `text` and `label` fields.
Labels may be `human` / `ai` or `0` / `1`. Extra categorical fields such as
`source`, `domain`, `generator`, or `edit_level` can be evaluated with repeated
`--slice-column` arguments.

To evaluate already-computed probabilities, add a score column with values
from 0 to 1. The committed example is deliberately tiny and illustrative; it
is useful for checking the command, not for drawing conclusions about a
detector:

```bash
uv run python scripts/19_robustness-evaluation/evaluate_detector.py \
  scripts/19_robustness-evaluation/example-scored-data.csv \
  --score-column ai_probability \
  --slice-column source \
  --min-slice-size 1 \
  --output robustness-report.json \
  --predictions-output robustness-predictions.csv
```

Use `--score-scale percent` when the supplied score column runs from 0 to 100.
Column names can be changed with `--text-column` and `--label-column`.

&nbsp;
## Score a local detector

After downloading an artifact as described in stage 15, omit the score column
and choose a model:

```bash
uv run python scripts/19_robustness-evaluation/evaluate_detector.py \
  path/to/held-out.csv \
  --model logreg \
  --slice-column domain \
  --slice-column generator \
  --output robustness-report.json
```

Transformer models also accept `--device` and `--batch-size`. By default,
length slices use whitespace word counts and the ranges 0–99, 100–249,
250–499, 500–999, and 1000+. Change the starts with
`--length-boundaries 0,50,100,250,500`, or disable this analysis with
`--no-length-slice`.

Slices with fewer than 20 rows are retained in the report but marked as
skipped. Set `--min-slice-size` to a value appropriate for a pre-declared
evaluation plan; do not lower it merely to make a noisy result look decisive.

&nbsp;
## Select a threshold without test leakage

The default threshold is 0.5. If the operating goal requires another
threshold, select it on a separate development file and freeze it before the
final evaluation:

```bash
uv run python scripts/19_robustness-evaluation/evaluate_detector.py \
  path/to/final-test.csv \
  --model logreg \
  --threshold-data path/to/development.csv \
  --threshold-objective balanced_accuracy \
  --output final-test-report.json
```

Supported objectives are `balanced_accuracy`, `f1_ai`, and `youden_j`. The
development file is scored with the same model or precomputed score column as
the final input. The selected threshold, objective value, development path,
and development row count are recorded in the report.

Do not use the final test file as `--threshold-data`. If calibration itself is
being changed, fit that calibration on a training or development split and
then measure it here on untouched data.

&nbsp;
## Suggested robustness protocol

Create datasets with provenance and immutable IDs, then pre-register the
slices and comparisons that matter. Useful first cases include a new text
domain, generators absent from training, lightly and moderately edited text,
short notes, and paraphrased or translated text. Keep seed documents and their
derived variants in the same split so near-duplicates cannot leak between
development and final testing.

Always report sample and class counts beside rates. Compare a frozen baseline
and candidate on the same rows, and inspect row-level errors before claiming
that a metric movement generalizes.
