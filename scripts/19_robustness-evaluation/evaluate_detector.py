#!/usr/bin/env python3
"""Evaluate a local detector or precomputed scores on labeled text data."""

import argparse
import csv
import hashlib
import json
import math
import sys
from datetime import UTC, datetime
from itertools import pairwise
from pathlib import Path

from ai_detector import (
    ArtifactNotReadyError,
    evaluate_predictions,
    evaluate_slices,
    load_classifier,
    model_registry,
    normalize_binary_label,
    select_threshold,
    threshold_curve,
    validate_text,
    word_count_bucket,
)

DERIVED_COLUMNS = (
    "evaluation_ai_probability",
    "evaluation_predicted_label",
    "evaluation_is_correct",
    "evaluation_word_count",
)


def parse_boundaries(value):
    try:
        boundaries = tuple(int(item.strip()) for item in value.split(","))
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "length boundaries must be comma-separated integers"
        ) from error
    if (
        not boundaries
        or boundaries[0] != 0
        or any(item < 0 for item in boundaries)
        or any(right <= left for left, right in pairwise(boundaries))
    ):
        raise argparse.ArgumentTypeError(
            "length boundaries must start at 0 and increase"
        )
    return boundaries


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description=(
            "Measure classification, ranking, calibration, threshold, and "
            "slice performance on labeled CSV or JSONL text data."
        )
    )
    parser.add_argument("input", type=Path, help="Labeled .csv or .jsonl file")
    scoring = parser.add_mutually_exclusive_group(required=True)
    scoring.add_argument(
        "--model",
        choices=sorted(model_registry()),
        help="Local detector used to create scores",
    )
    scoring.add_argument(
        "--score-column",
        help="Evaluate probabilities already present in this column",
    )
    parser.add_argument(
        "--score-scale",
        choices=("probability", "percent"),
        default="probability",
        help="Scale of --score-column values (default: probability)",
    )
    parser.add_argument("--text-column", default="text")
    parser.add_argument("--label-column", default="label")
    parser.add_argument(
        "--slice-column",
        action="append",
        default=[],
        help="Categorical column to evaluate separately; repeat as needed",
    )
    parser.add_argument(
        "--no-length-slice",
        action="store_true",
        help="Disable the default whitespace-word-count slice",
    )
    parser.add_argument(
        "--length-boundaries",
        type=parse_boundaries,
        default=(0, 100, 250, 500, 1000),
        metavar="0,100,250,...",
        help="Word-count bucket starts (default: 0,100,250,500,1000)",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Fixed AI probability threshold (default: 0.5)",
    )
    parser.add_argument(
        "--threshold-data",
        type=Path,
        help=(
            "Separate development CSV/JSONL used to select the threshold; "
            "never tune on the evaluation input"
        ),
    )
    parser.add_argument(
        "--threshold-objective",
        choices=("balanced_accuracy", "f1_ai", "youden_j"),
        default="balanced_accuracy",
        help="Objective used with --threshold-data",
    )
    parser.add_argument(
        "--calibration-bins",
        type=int,
        default=10,
        help="Number of equal-width reliability bins (default: 10)",
    )
    parser.add_argument(
        "--min-slice-size",
        type=int,
        default=20,
        help="Minimum rows required to calculate slice metrics (default: 20)",
    )
    parser.add_argument(
        "--artifact",
        type=Path,
        help="Override the selected model's exported artifact path",
    )
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda", "mps"),
        default="auto",
        help="Torch inference device for transformer models (default: auto)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Texts per model inference batch (default: 8)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Write the JSON report here instead of standard output",
    )
    parser.add_argument(
        "--predictions-output",
        type=Path,
        help="Optionally write row-level predictions as CSV",
    )
    args = parser.parse_args(argv)

    if args.threshold is not None and not 0.0 <= args.threshold <= 1.0:
        parser.error("--threshold must be between 0 and 1")
    if args.calibration_bins < 1:
        parser.error("--calibration-bins must be at least 1")
    if args.min_slice_size < 1:
        parser.error("--min-slice-size must be at least 1")
    if args.batch_size < 1:
        parser.error("--batch-size must be at least 1")
    if args.score_column is not None and args.artifact is not None:
        parser.error("--artifact can only be used with --model")
    if args.score_column is not None and args.device != "auto":
        parser.error("--device can only be used with --model")
    if args.model is not None and args.score_scale != "probability":
        parser.error("--score-scale can only be used with --score-column")
    if args.threshold_data is not None and args.threshold is not None:
        parser.error("--threshold cannot be combined with --threshold-data")
    if (
        args.threshold_data is not None
        and args.threshold_data.resolve() == args.input.resolve()
    ):
        parser.error("--threshold-data must be separate from the evaluation input")
    output_paths = [
        path for path in (args.output, args.predictions_output) if path is not None
    ]
    protected_paths = [args.input]
    if args.threshold_data is not None:
        protected_paths.append(args.threshold_data)
    if any(
        output.resolve() == protected.resolve()
        for output in output_paths
        for protected in protected_paths
    ):
        parser.error("output paths cannot overwrite input data")
    if (
        args.output is not None
        and args.predictions_output is not None
        and args.output.resolve() == args.predictions_output.resolve()
    ):
        parser.error("--output and --predictions-output must be different files")
    if args.output is not None and args.output.suffix.casefold() != ".json":
        parser.error("--output must use a .json suffix")
    if (
        args.predictions_output is not None
        and args.predictions_output.suffix.casefold() != ".csv"
    ):
        parser.error("--predictions-output must use a .csv suffix")
    duplicate_slices = sorted(
        {column for column in args.slice_column if args.slice_column.count(column) > 1}
    )
    if duplicate_slices:
        parser.error("duplicate --slice-column values: " + ", ".join(duplicate_slices))
    if not args.no_length_slice and "length_words" in args.slice_column:
        parser.error(
            "length_words is reserved for the default length slice; use "
            "--no-length-slice to evaluate an input column with that name"
        )
    if args.threshold is None:
        args.threshold = 0.5
    return args


def read_records(path):
    if not path.is_file():
        raise FileNotFoundError(f"Input data not found: {path}")
    suffix = path.suffix.casefold()
    if suffix == ".csv":
        with path.open(encoding="utf-8-sig", newline="") as file:
            reader = csv.DictReader(file)
            if not reader.fieldnames:
                raise ValueError(f"Input CSV has no header: {path}")
            records = list(reader)
            fieldnames = list(reader.fieldnames)
    elif suffix in {".jsonl", ".ndjson"}:
        records = []
        fieldnames = []
        with path.open(encoding="utf-8") as file:
            for line_number, line in enumerate(file, start=1):
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as error:
                    raise ValueError(
                        f"Invalid JSON on line {line_number} of {path}: {error.msg}"
                    ) from error
                if not isinstance(record, dict):
                    raise TypeError(
                        f"Line {line_number} of {path} is not a JSON object"
                    )
                records.append(record)
                for fieldname in record:
                    if fieldname not in fieldnames:
                        fieldnames.append(fieldname)
    else:
        raise ValueError("Input data must use a .csv, .jsonl, or .ndjson suffix")
    if not records:
        raise ValueError(f"Input data contains no rows: {path}")
    return records, fieldnames


def file_sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _require_columns(records, columns, *, path):
    available = set().union(*(record.keys() for record in records))
    missing = [column for column in columns if column not in available]
    if missing:
        raise ValueError(f"{path} is missing required columns: {', '.join(missing)}")


def _parse_probability(value, *, scale, row_number, column):
    try:
        probability = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(
            f"Row {row_number}: {column!r} is not numeric: {value!r}"
        ) from error
    if scale == "percent":
        probability /= 100.0
    if not math.isfinite(probability) or not 0.0 <= probability <= 1.0:
        expected = "0 to 100" if scale == "percent" else "0 to 1"
        raise ValueError(f"Row {row_number}: {column!r} must be between {expected}")
    return probability


def prepare_data(records, *, path, args, include_slices=True):
    required = [args.text_column, args.label_column]
    if include_slices:
        required.extend(args.slice_column)
    if args.score_column is not None:
        required.append(args.score_column)
    _require_columns(records, required, path=path)

    texts = []
    labels = []
    precomputed_probabilities = []
    for row_number, record in enumerate(records, start=2):
        try:
            text = validate_text(record.get(args.text_column))
        except (TypeError, ValueError) as error:
            raise ValueError(
                f"Row {row_number}: invalid {args.text_column!r}: {error}"
            ) from error
        try:
            label = normalize_binary_label(record.get(args.label_column))
        except ValueError as error:
            raise ValueError(f"Row {row_number}: {error}") from error
        texts.append(text)
        labels.append(label)
        if args.score_column is not None:
            precomputed_probabilities.append(
                _parse_probability(
                    record.get(args.score_column),
                    scale=args.score_scale,
                    row_number=row_number,
                    column=args.score_column,
                )
            )
    return texts, labels, precomputed_probabilities or None


def score_data(texts, *, args, classifier=None, dataset_name="data"):
    if args.score_column is not None:
        raise RuntimeError("score_data cannot score precomputed probabilities")
    if classifier is None:
        classifier = load_classifier(
            args.model,
            artifact_path=args.artifact,
            device=args.device,
        )
    print(
        f"Scoring {len(texts)} rows from {dataset_name} with {args.model}...",
        file=sys.stderr,
        flush=True,
    )
    probabilities = classifier.score_many(texts, batch_size=args.batch_size)
    if len(probabilities) != len(texts):
        raise RuntimeError(
            f"{args.model} returned {len(probabilities)} scores for {len(texts)} rows"
        )
    return [float(probability) for probability in probabilities], classifier


def write_json(path, payload):
    serialized = json.dumps(payload, indent=2, ensure_ascii=False) + "\n"
    if path is None:
        sys.stdout.write(serialized)
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_name(path.name + ".tmp")
    temporary_path.write_text(serialized, encoding="utf-8")
    temporary_path.replace(path)


def write_predictions(
    path,
    records,
    fieldnames,
    texts,
    labels,
    probabilities,
    *,
    threshold,
):
    conflicting = sorted(set(fieldnames) & set(DERIVED_COLUMNS))
    if conflicting:
        raise ValueError(
            "Prediction output columns already exist in the input: "
            + ", ".join(conflicting)
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_name(path.name + ".tmp")
    with temporary_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=[*fieldnames, *DERIVED_COLUMNS])
        writer.writeheader()
        for record, text, label, probability in zip(
            records, texts, labels, probabilities
        ):
            prediction = int(probability >= threshold)
            row = {fieldname: record.get(fieldname, "") for fieldname in fieldnames}
            row.update(
                {
                    "evaluation_ai_probability": probability,
                    "evaluation_predicted_label": "ai" if prediction else "human",
                    "evaluation_is_correct": prediction == label,
                    "evaluation_word_count": len(text.split()),
                }
            )
            writer.writerow(row)
    temporary_path.replace(path)


def build_report(
    records,
    texts,
    labels,
    probabilities,
    *,
    args,
    threshold,
    threshold_selection,
):
    slices = {}
    if not args.no_length_slice:
        length_values = [
            word_count_bucket(text, boundaries=args.length_boundaries) for text in texts
        ]
        slices["length_words"] = evaluate_slices(
            labels,
            probabilities,
            length_values,
            threshold=threshold,
            n_bins=args.calibration_bins,
            min_slice_size=args.min_slice_size,
        )
    for column in args.slice_column:
        slices[column] = evaluate_slices(
            labels,
            probabilities,
            [record.get(column) for record in records],
            threshold=threshold,
            n_bins=args.calibration_bins,
            min_slice_size=args.min_slice_size,
        )

    return {
        "schema_version": 1,
        "generated_at": datetime.now(UTC).isoformat(),
        "data": {
            "path": str(args.input),
            "sha256": file_sha256(args.input),
            "row_count": len(records),
            "label_column": args.label_column,
            "text_column": args.text_column,
        },
        "scoring": (
            {
                "method": "precomputed",
                "score_column": args.score_column,
                "score_scale": args.score_scale,
            }
            if args.score_column is not None
            else {
                "method": "local_model",
                "model": args.model,
                "artifact": str(args.artifact) if args.artifact else None,
                "device": args.device,
            }
        ),
        "threshold_selection": threshold_selection,
        "overall": evaluate_predictions(
            labels,
            probabilities,
            threshold=threshold,
            n_bins=args.calibration_bins,
        ),
        "threshold_curve": threshold_curve(labels, probabilities),
        "slices": slices,
        "interpretation_notes": [
            "Scores estimate P(AI | text) only under the model's training distribution.",
            "Slice results below the configured minimum sample size are not evaluated.",
            "A threshold selected on development data must be frozen before final testing.",
        ],
    }


def main(argv=None):
    args = parse_args(argv)
    classifier = None
    try:
        records, fieldnames = read_records(args.input)
        texts, labels, probabilities = prepare_data(records, path=args.input, args=args)
        if probabilities is None:
            probabilities, classifier = score_data(
                texts, args=args, dataset_name=str(args.input)
            )

        threshold = args.threshold
        threshold_selection = {
            "method": "fixed",
            "threshold": threshold,
        }
        if args.threshold_data is not None:
            development_records, _ = read_records(args.threshold_data)
            development_texts, development_labels, development_probabilities = (
                prepare_data(
                    development_records,
                    path=args.threshold_data,
                    args=args,
                    include_slices=False,
                )
            )
            if development_probabilities is None:
                development_probabilities, classifier = score_data(
                    development_texts,
                    args=args,
                    classifier=classifier,
                    dataset_name=str(args.threshold_data),
                )
            threshold_selection = select_threshold(
                development_labels,
                development_probabilities,
                objective=args.threshold_objective,
            )
            threshold_selection.update(
                {
                    "method": "development_data",
                    "data_path": str(args.threshold_data),
                    "data_sha256": file_sha256(args.threshold_data),
                    "row_count": len(development_records),
                }
            )
            threshold = threshold_selection["threshold"]

        report = build_report(
            records,
            texts,
            labels,
            probabilities,
            args=args,
            threshold=threshold,
            threshold_selection=threshold_selection,
        )
        if args.predictions_output is not None:
            write_predictions(
                args.predictions_output,
                records,
                fieldnames,
                texts,
                labels,
                probabilities,
                threshold=threshold,
            )
        write_json(args.output, report)
    except (
        ArtifactNotReadyError,
        FileNotFoundError,
        RuntimeError,
        TypeError,
        ValueError,
    ) as error:
        raise SystemExit(f"error: {error}") from error


if __name__ == "__main__":
    main()
