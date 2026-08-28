import csv
import importlib.util
import json
import sys
from pathlib import Path

import pytest

import ai_detector

PROJECT_DIR = Path(__file__).resolve().parents[1]
SCRIPT_PATH = (
    PROJECT_DIR / "scripts" / "19_robustness-evaluation" / "evaluate_detector.py"
)


def load_cli_module():
    spec = importlib.util.spec_from_file_location(
        "robustness_evaluation_cli", SCRIPT_PATH
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("human", 0),
        ("AI", 1),
        ("human-written", 0),
        ("ai_generated", 1),
        (0, 0),
        (1.0, 1),
        (False, 0),
    ],
)
def test_normalize_binary_label_accepts_common_forms(value, expected):
    assert ai_detector.normalize_binary_label(value) == expected


def test_normalize_binary_label_rejects_ambiguous_values():
    with pytest.raises(ValueError, match="human/AI or 0/1"):
        ai_detector.normalize_binary_label("mostly ai")


def test_evaluate_predictions_reports_classification_and_calibration():
    result = ai_detector.evaluate_predictions(
        [0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9], threshold=0.5, n_bins=5
    )

    assert result["accuracy"] == 1.0
    assert result["balanced_accuracy"] == 1.0
    assert result["roc_auc"] == 1.0
    assert result["average_precision"] == 1.0
    assert result["brier_score"] == pytest.approx(0.025)
    assert result["calibration"]["expected_calibration_error"] == pytest.approx(0.15)
    assert result["confusion_matrix"] == {
        "true_human_pred_human": 2,
        "true_human_pred_ai": 0,
        "true_ai_pred_human": 0,
        "true_ai_pred_ai": 2,
    }


def test_threshold_is_inclusive_and_single_class_auc_is_not_reported():
    result = ai_detector.evaluate_predictions([0, 0], [0.2, 0.5], threshold=0.5)

    assert result["confusion_matrix"]["true_human_pred_ai"] == 1
    assert result["roc_auc"] is None
    assert result["average_precision"] is None
    assert result["balanced_accuracy"] is None


@pytest.mark.parametrize(
    ("labels", "probabilities", "message"),
    [
        ([], [], "at least one"),
        ([0], [0.2, 0.3], "equal lengths"),
        ([0], [1.1], "between 0 and 1"),
        ([0], [float("nan")], "finite"),
    ],
)
def test_evaluate_predictions_rejects_invalid_inputs(labels, probabilities, message):
    with pytest.raises(ValueError, match=message):
        ai_detector.evaluate_predictions(labels, probabilities)


def test_select_threshold_uses_development_objective():
    selection = ai_detector.select_threshold(
        [0, 0, 1, 1],
        [0.1, 0.4, 0.45, 0.6],
        objective="balanced_accuracy",
    )

    assert selection["threshold"] == 0.45
    assert selection["objective_value"] == 1.0
    assert selection["candidate_count"] >= 4


def test_slice_evaluation_retains_small_groups_as_skipped():
    slices = ai_detector.evaluate_slices(
        [0, 0, 1],
        [0.1, 0.2, 0.9],
        ["news", "news", "essay"],
        min_slice_size=2,
    )

    by_value = {item["value"]: item for item in slices}
    assert by_value["news"]["status"] == "evaluated"
    assert by_value["essay"]["metrics"] is None
    assert by_value["essay"]["status"] == "skipped: fewer than 2 rows"


@pytest.mark.parametrize(
    ("word_count", "expected"),
    [
        (0, "0-99 words"),
        (99, "0-99 words"),
        (100, "100-249 words"),
        (1000, "1000+ words"),
    ],
)
def test_word_count_bucket_boundaries(word_count, expected):
    assert ai_detector.word_count_bucket("word " * word_count) == expected


def test_cli_evaluates_precomputed_scores_and_writes_predictions(tmp_path):
    module = load_cli_module()
    input_path = tmp_path / "evaluation.csv"
    output_path = tmp_path / "report.json"
    predictions_path = tmp_path / "predictions.csv"
    input_path.write_text(
        "text,label,domain,probability\n"
        '"one human sentence",human,a,0.1\n'
        '"another human sentence",human,b,0.2\n'
        '"one ai sentence",ai,a,0.8\n'
        '"another ai sentence",ai,b,0.9\n',
        encoding="utf-8",
    )

    module.main(
        [
            str(input_path),
            "--score-column",
            "probability",
            "--slice-column",
            "domain",
            "--min-slice-size",
            "1",
            "--output",
            str(output_path),
            "--predictions-output",
            str(predictions_path),
        ]
    )

    report = json.loads(output_path.read_text(encoding="utf-8"))
    assert report["schema_version"] == 1
    assert report["overall"]["accuracy"] == 1.0
    assert report["threshold_selection"] == {
        "method": "fixed",
        "threshold": 0.5,
    }
    assert set(report["slices"]) == {"length_words", "domain"}

    with predictions_path.open(encoding="utf-8", newline="") as file:
        predictions = list(csv.DictReader(file))
    assert predictions[0]["evaluation_predicted_label"] == "human"
    assert predictions[0]["evaluation_word_count"] == "3"
    assert predictions[-1]["evaluation_is_correct"] == "True"


def test_cli_selects_threshold_only_from_separate_data(tmp_path):
    module = load_cli_module()
    test_path = tmp_path / "test.jsonl"
    development_path = tmp_path / "development.jsonl"
    output_path = tmp_path / "report.json"
    test_rows = [
        {"text": "human test", "label": "human", "p": 0.4},
        {"text": "ai test", "label": "ai", "p": 0.44},
    ]
    development_rows = [
        {"text": "human one", "label": 0, "p": 0.1},
        {"text": "human two", "label": 0, "p": 0.4},
        {"text": "ai one", "label": 1, "p": 0.45},
        {"text": "ai two", "label": 1, "p": 0.6},
    ]
    test_path.write_text(
        "".join(json.dumps(row) + "\n" for row in test_rows),
        encoding="utf-8",
    )
    development_path.write_text(
        "".join(json.dumps(row) + "\n" for row in development_rows),
        encoding="utf-8",
    )

    module.main(
        [
            str(test_path),
            "--score-column",
            "p",
            "--threshold-data",
            str(development_path),
            "--no-length-slice",
            "--output",
            str(output_path),
        ]
    )

    report = json.loads(output_path.read_text(encoding="utf-8"))
    assert report["threshold_selection"]["method"] == "development_data"
    assert report["threshold_selection"]["threshold"] == 0.45
    assert report["overall"]["threshold"] == 0.45
    assert report["overall"]["accuracy"] == 0.5


def test_cli_rejects_threshold_tuning_on_evaluation_input(tmp_path):
    module = load_cli_module()
    input_path = tmp_path / "evaluation.csv"

    with pytest.raises(SystemExit):
        module.parse_args(
            [
                str(input_path),
                "--score-column",
                "probability",
                "--threshold-data",
                str(input_path),
            ]
        )


def test_cli_rejects_outputs_that_overwrite_inputs(tmp_path):
    module = load_cli_module()
    input_path = tmp_path / "evaluation.csv"

    with pytest.raises(SystemExit):
        module.parse_args(
            [
                str(input_path),
                "--score-column",
                "probability",
                "--output",
                str(input_path),
            ]
        )
