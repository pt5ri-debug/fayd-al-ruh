import pytest

from evaluation.benchmark import (
    METRIC_SPECS,
    MetricSpec,
    compare_metric_dicts,
    compare_reports,
    metrics_to_dict,
    report_to_dict,
)
from evaluation.metrics import metric_names
from evaluation.runner import EvaluationMetrics, EvaluationReport, run_suite
from evaluation.scenarios import SCENARIOS, Domain, scenario_ids


def test_scenarios_cover_every_domain():
    domains = {scenario.domain for scenario in SCENARIOS}
    assert domains == set(Domain)
    assert len(scenario_ids()) == len(set(scenario_ids()))


def test_metric_specs_cover_every_metric():
    assert set(spec.name for spec in METRIC_SPECS) == set(metric_names(EvaluationMetrics))


def test_suite_is_deterministic():
    first = run_suite()
    second = run_suite()
    assert metrics_to_dict(first.metrics) == metrics_to_dict(second.metrics)


def test_suite_scenarios_all_pass_on_current_version():
    report = run_suite()
    failing = [result.scenario_id for result in report.scenarios if not result.passed]
    assert failing == []


def test_report_metrics_are_bounded():
    metrics = run_suite().metrics
    for name in metric_names(EvaluationMetrics):
        value = getattr(metrics, name)
        assert 0.0 <= value <= 1.0, name


def test_report_to_dict_is_json_ready():
    import json

    payload = json.dumps(report_to_dict(run_suite()))
    assert "metrics" in payload and "scenarios" in payload


def test_compare_reports_flags_no_regression_for_identical_runs():
    report = run_suite()
    regression = compare_reports(report, report)
    assert regression.passed
    assert regression.regressions == ()


def test_compare_reports_detects_lower_is_better_regression():
    baseline = {"brier_score": 0.1}
    candidate = {"brier_score": 0.2}
    regression = compare_metric_dicts(
        baseline, candidate, specs=(MetricSpec("brier_score", higher_is_better=False),)
    )
    assert not regression.passed


def test_compare_reports_tolerance_absorbs_noise():
    baseline = {"overall": 0.90}
    candidate = {"overall": 0.895}
    tolerant = compare_metric_dicts(
        baseline, candidate, specs=(MetricSpec("overall", higher_is_better=True, tolerance=0.01),)
    )
    assert tolerant.passed


def test_regression_summary_mentions_failures():
    regression = compare_metric_dicts(
        {"prediction_accuracy": 0.9},
        {"prediction_accuracy": 0.5},
        specs=(MetricSpec("prediction_accuracy", higher_is_better=True),),
    )
    assert "FAIL" in regression.summary()
    assert regression.regressions[0].name == "prediction_accuracy"


def test_metrics_are_frozen_and_comparable():
    report = run_suite()
    assert isinstance(report, EvaluationReport)
    assert isinstance(report.metrics, EvaluationMetrics)
    with pytest.raises(Exception):
        report.metrics.overall = 0.0  # type: ignore[misc]
