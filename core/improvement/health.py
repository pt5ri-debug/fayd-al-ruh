"""The system health analyzer: one honest reading of how the system is doing.

The monitor produces numbers; this module turns them into a *judgement* without
inventing one. A :class:`HealthReport` scores the components the metrics cover
(each from the metrics that belong to it, normalised by direction), names the
alerts that crossed a threshold, and reports a single bounded overall figure.

Three rules keep it honest:

* **no metric, no claim** -- a component with no samples is reported as
  ``unknown`` (a ``None`` score), never as a passing ``1.0``. An unmeasured
  system is not a healthy one;
* **direction is respected** -- a metric marked ``higher_is_better`` is scored
  as-is, its inverse as ``1 - value``, so a low failure rate reads as good; and
* **a score is not a diagnosis** -- the report says *what* looks weak, and the
  weakness detector (a separate module) says *why it might be*, so a single
  figure is never mistaken for a cause.

Dependency-free and Tk-free.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

from core.improvement.metrics import MetricKind, metric_spec
from core.improvement.monitor import MetricSummary, PerformanceMonitor

#: Below this component score the system is called degraded; below the second,
#: unhealthy. Both are deliberately conservative -- a warning is cheap, a false
#: "all is well" is not.
DEGRADED_THRESHOLD = 0.75
UNHEALTHY_THRESHOLD = 0.5

#: A metric with fewer samples than this contributes to the score but is noted,
#: so a figure from one observation is never presented with full confidence.
LOW_SAMPLE_WARNING = 3


class HealthStatus(str, Enum):
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"


def _component_score(summaries: tuple[MetricSummary, ...]) -> float | None:
    """The mean, direction-normalised score of a component's metrics.

    Only RATIO and GAUGE metrics carry a meaningful 0..1 reading; a counter or a
    duration has no intrinsic ceiling, so it is reported separately (as an alert
    threshold) rather than folded into a score that would be meaningless.
    """
    scored: list[float] = []
    for summary in summaries:
        if summary.kind not in {MetricKind.RATIO, MetricKind.GAUGE}:
            continue
        spec = metric_spec(summary.name)
        value = summary.mean
        if spec is not None and not spec.higher_is_better:
            value = 1.0 - value
        scored.append(max(0.0, min(1.0, value)))
    if not scored:
        return None
    return sum(scored) / len(scored)


@dataclass(frozen=True)
class HealthAlert:
    """One reading that crossed a threshold, with the metric that produced it."""

    component: str
    metric: str
    message: str
    severity: float

    def describe(self) -> str:
        return f"[{self.component}] {self.message}"


@dataclass(frozen=True)
class HealthReport:
    """The system's health: a score per component, alerts, and one overall figure."""

    status: HealthStatus
    overall: float | None
    component_scores: dict[str, float | None] = field(default_factory=dict)
    alerts: tuple[HealthAlert, ...] = ()
    notes: tuple[str, ...] = ()

    def score_for(self, component: str) -> float | None:
        return self.component_scores.get(component)

    def describe(self) -> str:
        lines = [f"System health: {self.status.value}"]
        if self.overall is None:
            lines.append("overall: unknown (no scorable metrics were recorded)")
        else:
            lines.append(f"overall: {self.overall:.4f}")
        for component, score in sorted(self.component_scores.items()):
            shown = "unknown" if score is None else f"{score:.4f}"
            lines.append(f"  {component}: {shown}")
        for alert in self.alerts:
            lines.append(f"  ! {alert.describe()}")
        for note in self.notes:
            lines.append(f"  note: {note}")
        return "\n".join(lines)


#: Alert thresholds: a metric, the comparison, and the value that trips it. Each
#: is written down here so a threshold is a documented decision, never a magic
#: number buried in a branch.
_ALERT_RULES: tuple[tuple[str, str, float, str, float], ...] = (
    ("task.failure_rate", ">", 0.2, "Task failure rate is high.", 0.8),
    ("retrieval.quality", "<", 0.6, "Retrieval quality is low.", 0.6),
    ("reasoning.accuracy", "<", 0.6, "Reasoning accuracy is low.", 0.7),
    ("planning.efficiency", "<", 0.6, "Planning is inefficient (wasted steps).", 0.5),
    ("tool.execution_success", "<", 0.8, "Tool execution is failing often.", 0.6),
    ("knowledge.integration_error", ">", 5.0, "Knowledge integration is erroring.", 0.5),
    ("knowledge.duplicate_rate", ">", 0.4, "Duplicate knowledge rate is high.", 0.4),
    ("user.correction", ">", 3.0, "Users are correcting answers often.", 0.5),
    ("resource.consumption", ">", 0.9, "The machine is nearly saturated.", 0.7),
)


def _trips(actual: float, comparison: str, threshold: float) -> bool:
    if comparison == ">":
        return actual > threshold
    if comparison == "<":
        return actual < threshold
    return False


def analyse_health(
    monitor: PerformanceMonitor | None = None,
    *,
    summaries: tuple[MetricSummary, ...] | None = None,
) -> HealthReport:
    """Read a monitor (or pre-computed summaries) into a health report.

    Passing ``summaries`` directly lets a caller judge a stored snapshot without
    rebuilding a monitor -- the window and the CLI both do that.
    """
    if summaries is None:
        summaries = monitor.summaries() if monitor is not None else ()

    by_component: dict[str, list[MetricSummary]] = {}
    for summary in summaries:
        by_component.setdefault(summary.component, []).append(summary)

    scores: dict[str, float | None] = {}
    for component in sorted(by_component):
        scores[component] = _component_score(tuple(by_component[component]))

    alerts: list[HealthAlert] = []
    for name, comparison, threshold, message, severity in _ALERT_RULES:
        for summary in summaries:
            if summary.name != name:
                continue
            actual = summary.total if summary.kind is MetricKind.COUNTER else summary.mean
            if _trips(actual, comparison, threshold):
                alerts.append(HealthAlert(summary.component, name, f"{message} ({name}={actual:g})", severity))

    notes: list[str] = []
    for summary in summaries:
        if summary.count < LOW_SAMPLE_WARNING:
            notes.append(f"{summary.name} has only {summary.count} sample(s) -- treat with care.")

    scorable = [score for score in scores.values() if score is not None]
    overall = (sum(scorable) / len(scorable)) if scorable else None
    if overall is None:
        status = HealthStatus.UNKNOWN
    elif overall < UNHEALTHY_THRESHOLD:
        status = HealthStatus.UNHEALTHY
    elif overall < DEGRADED_THRESHOLD or alerts:
        status = HealthStatus.DEGRADED
    else:
        status = HealthStatus.HEALTHY

    return HealthReport(
        status=status,
        overall=overall,
        component_scores=scores,
        alerts=tuple(alerts),
        notes=tuple(notes),
    )


__all__ = [
    "DEGRADED_THRESHOLD",
    "LOW_SAMPLE_WARNING",
    "UNHEALTHY_THRESHOLD",
    "HealthAlert",
    "HealthReport",
    "HealthStatus",
    "analyse_health",
]
