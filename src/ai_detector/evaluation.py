"""Metrics and slice analysis for binary AI-text detector predictions."""

import math
import numbers
from itertools import pairwise

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    log_loss,
    roc_auc_score,
)

LABEL_ALIASES = {
    "0": 0,
    "human": 0,
    "human-written": 0,
    "human_written": 0,
    "1": 1,
    "ai": 1,
    "ai-generated": 1,
    "ai_generated": 1,
}


def normalize_binary_label(value):
    """Return a human=0/AI=1 label from a common serialized form."""
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, numbers.Real):
        numeric_value = float(value)
        if math.isfinite(numeric_value) and numeric_value in {0.0, 1.0}:
            return int(numeric_value)
    if isinstance(value, str):
        normalized = value.strip().casefold()
        if normalized in LABEL_ALIASES:
            return LABEL_ALIASES[normalized]
    raise ValueError(f"Unsupported label {value!r}; use human/AI or 0/1")


def _validate_inputs(labels, probabilities):
    normalized_labels = np.asarray(
        [normalize_binary_label(label) for label in labels], dtype=np.int64
    )
    normalized_probabilities = np.asarray(probabilities, dtype=np.float64)
    if normalized_labels.ndim != 1 or normalized_probabilities.ndim != 1:
        raise ValueError("labels and probabilities must be one-dimensional")
    if normalized_labels.size == 0:
        raise ValueError("at least one prediction is required")
    if normalized_labels.size != normalized_probabilities.size:
        raise ValueError("labels and probabilities must have equal lengths")
    if not np.isfinite(normalized_probabilities).all():
        raise ValueError("probabilities must contain only finite values")
    if (normalized_probabilities < 0.0).any() or (normalized_probabilities > 1.0).any():
        raise ValueError("probabilities must be between 0 and 1")
    return normalized_labels, normalized_probabilities


def _validate_threshold(threshold):
    threshold = float(threshold)
    if not math.isfinite(threshold) or not 0.0 <= threshold <= 1.0:
        raise ValueError("threshold must be between 0 and 1")
    return threshold


def _divide(numerator, denominator, *, default=None):
    if denominator == 0:
        return default
    return float(numerator / denominator)


def _threshold_metrics_from_counts(tp, fp, tn, fn):
    sample_count = tp + fp + tn + fn
    precision = _divide(tp, tp + fp, default=0.0)
    recall = _divide(tp, tp + fn)
    specificity = _divide(tn, tn + fp)
    if precision is None or recall is None or precision + recall == 0.0:
        f1 = 0.0
    else:
        f1 = float(2.0 * precision * recall / (precision + recall))
    if recall is None or specificity is None:
        balanced_accuracy = None
        youden_j = None
    else:
        balanced_accuracy = float((recall + specificity) / 2.0)
        youden_j = float(recall + specificity - 1.0)
    return {
        "sample_count": int(sample_count),
        "class_counts": {"human": int(tn + fp), "ai": int(tp + fn)},
        "confusion_matrix": {
            "true_human_pred_human": int(tn),
            "true_human_pred_ai": int(fp),
            "true_ai_pred_human": int(fn),
            "true_ai_pred_ai": int(tp),
        },
        "accuracy": _divide(tp + tn, sample_count),
        "balanced_accuracy": balanced_accuracy,
        "precision_ai": precision,
        "recall_ai": recall,
        "specificity_human": specificity,
        "f1_ai": f1,
        "youden_j": youden_j,
    }


def calibration_summary(labels, probabilities, *, n_bins=10):
    """Compute equal-width reliability bins, ECE, and maximum gap."""
    labels, probabilities = _validate_inputs(labels, probabilities)
    if not isinstance(n_bins, numbers.Integral) or n_bins < 1:
        raise ValueError("n_bins must be a positive integer")

    bin_indices = np.minimum(
        (probabilities * int(n_bins)).astype(np.int64), int(n_bins) - 1
    )
    bins = []
    weighted_gap = 0.0
    maximum_gap = 0.0
    for index in range(int(n_bins)):
        mask = bin_indices == index
        count = int(mask.sum())
        lower = index / n_bins
        upper = (index + 1) / n_bins
        if count:
            mean_probability = float(probabilities[mask].mean())
            observed_ai_rate = float(labels[mask].mean())
            gap = abs(mean_probability - observed_ai_rate)
            weighted_gap += count * gap
            maximum_gap = max(maximum_gap, gap)
        else:
            mean_probability = None
            observed_ai_rate = None
            gap = None
        bins.append(
            {
                "index": index,
                "lower": float(lower),
                "upper": float(upper),
                "upper_inclusive": index == n_bins - 1,
                "sample_count": count,
                "mean_ai_probability": mean_probability,
                "observed_ai_rate": observed_ai_rate,
                "absolute_gap": gap,
            }
        )
    return {
        "method": "equal_width",
        "n_bins": int(n_bins),
        "expected_calibration_error": float(weighted_gap / labels.size),
        "maximum_calibration_error": float(maximum_gap),
        "bins": bins,
    }


def evaluate_predictions(labels, probabilities, *, threshold=0.5, n_bins=10):
    """Evaluate discrimination, classification, and calibration metrics."""
    labels, probabilities = _validate_inputs(labels, probabilities)
    threshold = _validate_threshold(threshold)
    predictions = probabilities >= threshold
    positive = labels == 1
    negative = ~positive
    tp = int((predictions & positive).sum())
    fp = int((predictions & negative).sum())
    tn = int((~predictions & negative).sum())
    fn = int((~predictions & positive).sum())

    metrics = _threshold_metrics_from_counts(tp, fp, tn, fn)
    metrics["threshold"] = threshold
    if positive.any() and negative.any():
        metrics["roc_auc"] = float(roc_auc_score(labels, probabilities))
        metrics["average_precision"] = float(
            average_precision_score(labels, probabilities)
        )
    else:
        metrics["roc_auc"] = None
        metrics["average_precision"] = None
    metrics["brier_score"] = float(brier_score_loss(labels, probabilities))
    metrics["log_loss"] = float(log_loss(labels, probabilities, labels=[0, 1]))
    metrics["calibration"] = calibration_summary(labels, probabilities, n_bins=n_bins)
    return metrics


def threshold_curve(labels, probabilities, *, thresholds=None):
    """Return threshold-dependent metrics over a fixed, comparable grid."""
    labels, probabilities = _validate_inputs(labels, probabilities)
    if thresholds is None:
        thresholds = [index / 20 for index in range(21)]
    rows = []
    for threshold in thresholds:
        threshold = _validate_threshold(threshold)
        predictions = probabilities >= threshold
        positive = labels == 1
        negative = ~positive
        row = _threshold_metrics_from_counts(
            int((predictions & positive).sum()),
            int((predictions & negative).sum()),
            int((~predictions & negative).sum()),
            int((~predictions & positive).sum()),
        )
        row["threshold"] = threshold
        rows.append(row)
    return rows


def select_threshold(labels, probabilities, *, objective="balanced_accuracy"):
    """Select a threshold on development data using a named objective."""
    labels, probabilities = _validate_inputs(labels, probabilities)
    supported = {"balanced_accuracy", "f1_ai", "youden_j"}
    if objective not in supported:
        choices = ", ".join(sorted(supported))
        raise ValueError(f"objective must be one of: {choices}")
    if np.unique(labels).size != 2:
        raise ValueError("threshold selection requires both human and AI rows")

    order = np.argsort(-probabilities, kind="stable")
    sorted_probabilities = probabilities[order]
    sorted_labels = labels[order]
    positives = int(labels.sum())
    negatives = int(labels.size - positives)
    tp = 0
    fp = 0
    best = None
    candidates = []

    if sorted_probabilities[0] < 1.0:
        candidates.append((1.0, tp, fp, negatives, positives))

    start = 0
    while start < sorted_probabilities.size:
        score = float(sorted_probabilities[start])
        stop = start + 1
        while (
            stop < sorted_probabilities.size
            and sorted_probabilities[stop] == sorted_probabilities[start]
        ):
            stop += 1
        group = sorted_labels[start:stop]
        tp += int(group.sum())
        fp += int(group.size - group.sum())
        candidates.append((score, tp, fp, negatives - fp, positives - tp))
        start = stop

    if sorted_probabilities[-1] > 0.0:
        candidates.append((0.0, positives, negatives, 0, 0))

    for threshold, tp, fp, tn, fn in candidates:
        metrics = _threshold_metrics_from_counts(tp, fp, tn, fn)
        objective_value = metrics[objective]
        if objective_value is None:
            continue
        rank = (
            objective_value,
            -abs(threshold - 0.5),
            -threshold,
        )
        if best is None or rank > best[0]:
            best = (rank, threshold, metrics)

    _, threshold, metrics = best
    return {
        "objective": objective,
        "objective_value": float(metrics[objective]),
        "threshold": float(threshold),
        "candidate_count": len(candidates),
        "tie_break": "closest_to_0.5_then_lower",
    }


def evaluate_slices(
    labels,
    probabilities,
    slice_values,
    *,
    threshold=0.5,
    n_bins=10,
    min_slice_size=1,
):
    """Evaluate the detector independently for each value in a slice."""
    labels, probabilities = _validate_inputs(labels, probabilities)
    values = np.asarray(
        [
            "(missing)" if value is None or value == "" else str(value)
            for value in slice_values
        ],
        dtype=object,
    )
    if values.ndim != 1 or values.size != labels.size:
        raise ValueError("slice_values must have one value per prediction")
    if not isinstance(min_slice_size, numbers.Integral) or min_slice_size < 1:
        raise ValueError("min_slice_size must be a positive integer")

    groups = []
    for value in sorted(set(values.tolist()), key=str.casefold):
        mask = values == value
        sample_count = int(mask.sum())
        group = {"value": value, "sample_count": sample_count}
        if sample_count < min_slice_size:
            group["metrics"] = None
            group["status"] = f"skipped: fewer than {min_slice_size} rows"
        else:
            group["metrics"] = evaluate_predictions(
                labels[mask],
                probabilities[mask],
                threshold=threshold,
                n_bins=n_bins,
            )
            group["status"] = "evaluated"
        groups.append(group)
    return groups


def word_count_bucket(text, *, boundaries=(0, 100, 250, 500, 1000)):
    """Return a stable text-length bucket label based on whitespace words."""
    if not isinstance(text, str):
        raise TypeError("text must be a string")
    normalized_boundaries = tuple(int(value) for value in boundaries)
    if (
        not normalized_boundaries
        or normalized_boundaries[0] != 0
        or any(value < 0 for value in normalized_boundaries)
        or any(right <= left for left, right in pairwise(normalized_boundaries))
    ):
        raise ValueError("boundaries must start at 0 and be strictly increasing")
    word_count = len(text.split())
    for lower, upper in pairwise(normalized_boundaries):
        if lower <= word_count < upper:
            return f"{lower}-{upper - 1} words"
    return f"{normalized_boundaries[-1]}+ words"
