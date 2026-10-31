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
    MetricSpec("petroleum_learning_accuracy", True, label="Petroleum discipline learning accuracy"),
    MetricSpec("petroleum_generalisation_accuracy", True, label="Petroleum held-out generalisation accuracy"),
    MetricSpec("acceptance_gate_accuracy", True, label="Acceptance gate accuracy (held-out verdicts)"),
    MetricSpec("curriculum_coverage", True, label="Curriculum coverage"),
    MetricSpec("analogy_accuracy", True, label="Analogy accuracy (held-out judgements)"),
    MetricSpec("relation_extraction_accuracy", True, label="Relation extraction accuracy (held-out sentences)"),
    MetricSpec("polarity_accuracy", True, label="Relation polarity accuracy (held-out opposing pairs)"),
    MetricSpec("connection_accuracy", True, label="Connection explanation accuracy (held-out concept pairs)"),
    MetricSpec("consistency_accuracy", True, label="Contradiction detection accuracy (held-out graphs)"),
    MetricSpec("inference_accuracy", True, label="Graph inference accuracy (held-out entailments)"),
    MetricSpec("connection_reasoning_accuracy", True, label="Connection reasoning accuracy (held-out answer-path pairs)"),
    MetricSpec("causal_reasoning_accuracy", True, label="Causal reasoning accuracy (held-out answer-path causal pairs)"),
    MetricSpec("controller_causal_accuracy", True, label="Controller causal accuracy (held-out pairs walked on the controller's own causal graph)"),
    MetricSpec("answer_confidence_accuracy", True, label="Answer confidence accuracy (held-out term expectations)"),
    MetricSpec("grounding_accuracy", True, label="Grounding gate accuracy (held-out publish decisions)"),
    MetricSpec("reasoning_trace_accuracy", True, label="Reasoning trace accuracy (held-out trace shapes)"),
    MetricSpec("plan_accuracy", True, label="Plan accuracy (held-out plan shapes)"),
    MetricSpec("decision_recommendation_accuracy", True, label="Decision recommendation accuracy (held-out decision shapes)"),
    MetricSpec("belief_accuracy", True, label="Belief accuracy (held-out belief shapes)"),
    MetricSpec("lead_accuracy", True, label="Lead accuracy (held-out answer-body lead shapes)"),
    MetricSpec("temporal_accuracy", True, label="Temporal accuracy (held-out timeline shapes)"),
    MetricSpec("analogy_answer_accuracy", True, label="Analogy answer accuracy (held-out comparison shapes)"),
    MetricSpec("knowledge_carry_accuracy", True, label="Knowledge carry accuracy (held-out cross-session shapes)"),
    MetricSpec("answer_path_accuracy", True, label="Answer path accuracy (held-out connected-path shapes)"),
    MetricSpec("arabic_answer_path_accuracy", True, label="Arabic answer path accuracy (held-out connected-path shapes)"),
    MetricSpec("arabic_readable_body_accuracy", True, label="Arabic readable body accuracy (held-out: no identifier leaks into an Arabic body)"),
    MetricSpec("character_isolation_accuracy", True, label="Character isolation and sharing accuracy (held-out isolation/sharing judgements)"),
    MetricSpec("character_identity_accuracy", True, label="Character identity accuracy (held-out: a character's personality reaches its answer text, and the two windows share one memory and one name)"),
    MetricSpec("semantic_separation", True, label="Semantic separation (held-out relations)"),
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
