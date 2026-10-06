from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from evaluation.runner import EvaluationMetrics, EvaluationReport


@dataclass(frozen=True)
class MetricSpec:
    """Declares how a metric should be compared across versions.

    ``higher_is_better`` fixes the direction; ``tolerance`` absorbs harmless
    floating-point or sampling noise. When ``tolerance`` is zero the metric
    must not move at all in the wrong direction.
    """

    name: str
    higher_is_better: bool
    tolerance: float = 0.0
    label: str = ""


METRIC_SPECS: tuple[MetricSpec, ...] = (
    MetricSpec("prediction_accuracy", True, label="Prediction accuracy"),
    MetricSpec("brier_score", False, label="Brier score (lower is better)"),
    MetricSpec("calibration_error", False, label="Calibration error (ECE)"),
    MetricSpec("reliability", False, label="Reliability (lower is better)"),
    MetricSpec("resolution", True, label="Resolution"),
    MetricSpec("contradiction_rate", True, label="Contradiction detection"),
    MetricSpec("retrieval_accuracy", True, label="Retrieval accuracy"),
    MetricSpec("retrieval_mrr", True, label="Retrieval MRR"),
    MetricSpec("planning_success_rate", True, label="Planning success"),
    MetricSpec("decision_explanation_quality", True, label="Decision explanation quality"),
    MetricSpec("causal_chain_accuracy", True, label="Causal chain accuracy"),
    MetricSpec("causal_explanation_quality", True, label="Causal explanation quality"),
    MetricSpec("petroleum_causal_accuracy", True, label="Petroleum causal accuracy"),
    MetricSpec("petroleum_retrieval_accuracy", True, label="Petroleum retrieval accuracy"),
    MetricSpec("petroleum_ontology_accuracy", True, label="Petroleum ontology accuracy"),
    MetricSpec("petroleum_inference_accuracy", True, label="Petroleum ontology inference accuracy"),
    MetricSpec("petroleum_multihop_retrieval_accuracy", True, label="Petroleum ontology-guided retrieval accuracy"),
    MetricSpec("petroleum_ontology_learning_accuracy", True, label="Petroleum ontology learning accuracy"),
    MetricSpec("petroleum_ontology_reinforcement_accuracy", True, label="Petroleum ontology reinforcement accuracy"),
    MetricSpec("petroleum_ontology_feedback_accuracy", True, label="Petroleum ontology feedback accuracy"),
    MetricSpec("overall", True, label="Overall"),
)


@dataclass(frozen=True)
class MetricDelta:
    name: str
    baseline: float
    candidate: float
    delta: float
    regressed: bool


@dataclass(frozen=True)
class RegressionReport:
    deltas: tuple[MetricDelta, ...]

    @property
    def regressions(self) -> tuple[MetricDelta, ...]:
        return tuple(delta for delta in self.deltas if delta.regressed)

    @property
    def passed(self) -> bool:
        return not self.regressions

    def summary(self) -> str:
        lines = [
            f"{'REGRESSED' if delta.regressed else 'ok':>9}  {delta.name:<30} "
            f"{delta.baseline:.4f} -> {delta.candidate:.4f} ({delta.delta:+.4f})"
            for delta in self.deltas
        ]
        verdict = "PASS: no performance regression" if self.passed else "FAIL: performance regressed"
        return "\n".join([verdict, *lines])


def metrics_to_dict(metrics: EvaluationMetrics) -> dict[str, float]:
    return {
        spec.name: float(getattr(metrics, spec.name))
        for spec in METRIC_SPECS
    }


def compare_reports(
    baseline: EvaluationReport,
    candidate: EvaluationReport,
    specs: tuple[MetricSpec, ...] = METRIC_SPECS,
) -> RegressionReport:
    """Compare two reports metric-by-metric and flag any regression."""
    deltas: list[MetricDelta] = []
    for spec in specs:
        baseline_value = float(getattr(baseline.metrics, spec.name))
        candidate_value = float(getattr(candidate.metrics, spec.name))
        raw = candidate_value - baseline_value
        if spec.higher_is_better:
            regressed = candidate_value < baseline_value - spec.tolerance
        else:
            regressed = candidate_value > baseline_value + spec.tolerance
        deltas.append(
            MetricDelta(
                name=spec.name,
                baseline=baseline_value,
                candidate=candidate_value,
                delta=round(raw, 6),
                regressed=regressed,
            )
        )
    return RegressionReport(tuple(deltas))


def compare_metric_dicts(
    baseline: dict[str, float],
    candidate: dict[str, float],
    specs: tuple[MetricSpec, ...] = METRIC_SPECS,
) -> RegressionReport:
    deltas: list[MetricDelta] = []
    for spec in specs:
        if spec.name not in baseline or spec.name not in candidate:
            continue
        baseline_value = float(baseline[spec.name])
        candidate_value = float(candidate[spec.name])
        if spec.higher_is_better:
            regressed = candidate_value < baseline_value - spec.tolerance
        else:
            regressed = candidate_value > baseline_value + spec.tolerance
        deltas.append(
            MetricDelta(
                name=spec.name,
                baseline=baseline_value,
                candidate=candidate_value,
                delta=round(candidate_value - baseline_value, 6),
                regressed=regressed,
            )
        )
    return RegressionReport(tuple(deltas))


def report_to_dict(report: EvaluationReport) -> dict[str, Any]:
    return {
        "version": 1,
        "metrics": metrics_to_dict(report.metrics),
        "domain_scores": dict(report.domain_scores),
        "scenarios": [
            {
                "id": result.scenario_id,
                "domain": result.domain,
                "passed": result.passed,
                "checks": dict(result.checks),
            }
            for result in report.scenarios
        ],
        "calibration": [
            {
                "lower": bucket.lower,
                "upper": bucket.upper,
                "count": bucket.count,
                "mean_confidence": bucket.mean_confidence,
                "empirical_accuracy": bucket.empirical_accuracy,
            }
            for bucket in report.calibration
        ],
    }
