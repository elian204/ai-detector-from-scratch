"""Reusable inference helpers for the AI text detector."""

from .classifiers import (
    CausalTextClassifier,
    EncoderTextClassifier,
    SklearnTextClassifier,
    build_causal_batch,
    load_classifier,
    score_text,
)
from .evaluation import (
    calibration_summary,
    evaluate_predictions,
    evaluate_slices,
    normalize_binary_label,
    select_threshold,
    threshold_curve,
    word_count_bucket,
)
from .registry import (
    PROJECT_DIR,
    ArtifactNotReadyError,
    ModelSpec,
    artifact_status,
    ensure_artifact_ready,
    model_registry,
)
from .scoring import score_payload, validate_text

__all__ = [
    "PROJECT_DIR",
    "ArtifactNotReadyError",
    "CausalTextClassifier",
    "EncoderTextClassifier",
    "ModelSpec",
    "SklearnTextClassifier",
    "artifact_status",
    "build_causal_batch",
    "calibration_summary",
    "ensure_artifact_ready",
    "evaluate_predictions",
    "evaluate_slices",
    "load_classifier",
    "model_registry",
    "normalize_binary_label",
    "score_payload",
    "score_text",
    "select_threshold",
    "threshold_curve",
    "validate_text",
    "word_count_bucket",
]
