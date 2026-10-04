"""Integration checks for the scientific evaluation suite and its regression gate."""

import json

import pytest

from evaluation.baseline import check_against_baseline, load_baseline, save_baseline
from evaluation.benchmark import compare_metric_dicts, metrics_to_dict
from evaluation.runner import run_suite


pytestmark = pytest.mark.integration


def test_stored_baseline_is_current():
    """The committed baseline must match a fresh run: no silent drift."""
    report = run_suite()
    regression = check_against_baseline(report)
    assert regression.passed, regression.summary()


def test_baseline_round_trip(tmp_path):
    report = run_suite()
    path = tmp_path / "baseline.json"
    save_baseline(report, path)

    stored = load_baseline(path)
    assert stored["metrics"] == metrics_to_dict(report.metrics)
    assert json.loads(path.read_text(encoding="utf-8"))["version"] == 1


def test_regression_gate_blocks_a_degraded_version():
    baseline = metrics_to_dict(run_suite().metrics)
    degraded = dict(baseline)
    degraded["prediction_accuracy"] = baseline["prediction_accuracy"] - 0.2
    degraded["calibration_error"] = baseline["calibration_error"] + 0.2

    regression = compare_metric_dicts(baseline, degraded)
    assert not regression.passed
    regressed = {delta.name for delta in regression.regressions}
    assert {"prediction_accuracy", "calibration_error"} <= regressed


def test_regression_gate_allows_improvements():
    baseline = metrics_to_dict(run_suite().metrics)
    improved = dict(baseline)
    improved["overall"] = min(1.0, baseline["overall"] + 0.05)
    improved["brier_score"] = max(0.0, baseline["brier_score"] - 0.05)

    regression = compare_metric_dicts(baseline, improved)
    assert regression.passed
