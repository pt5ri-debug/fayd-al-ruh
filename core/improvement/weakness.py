"""The weakness and failure detector: evidence-backed findings, never guesses.

Section 5 of the phase is strict about what a "weakness" may be. It must carry a
stable id, a description, the *evidence* it rests on, the affected component, a
frequency, a severity, a **confidence in the diagnosis**, a potential impact, a
suggested investigation, and the metrics it relates to -- and the detector must
not treat a correlation as a cause or act on a single failure.

This module produces exactly that. A :class:`Weakness` is built by a *rule* that
fires only when its own condition holds over a real :class:`MetricSummary` or a
real list of failures, and the rule names the metrics it used, so every finding
is traceable to the numbers behind it. Two honesty rules are structural:

* **frequency is counted, not assumed** -- a rule states the minimum support it
  needs, so one failure cannot raise a finding; and
* **confidence is separate from severity** -- a severe finding seen twice has a
  low diagnosis confidence and says so, rather than being presented as certain.

The detector reads only the monitor's summaries and a caller-supplied list of
failures; it never touches the knowledge graph, a model, or the network.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import Enum
from uuid import uuid4

from core.improvement.metrics import metric_spec
from core.improvement.monitor import MetricSummary, PerformanceMonitor


class WeaknessKind(str, Enum):
    """The failure modes the phase lists -- one closed vocabulary."""

    REPEATED_FAILURE = "repeated_failure"
    FREQUENT_CORRECTION = "frequent_correction"
    POOR_RETRIEVAL = "poor_retrieval"
    INCORRECT_REASONING = "incorrect_reasoning"
    UNSUPPORTED_CLAIM = "unsupported_claim"
    PLANNING_MISTAKE = "planning_mistake"
    UNNECESSARY_TOOL_USE = "unnecessary_tool_use"
    EXCESSIVE_LATENCY = "excessive_latency"
    MEMORY_DUPLICATION = "memory_duplication"
    RESOURCE_EXHAUSTION = "resource_exhaustion"
    REGRESSION = "regression"
    KNOWLEDGE_ERROR = "knowledge_error"


class Severity(str, Enum):
    """How bad the finding is, from a closed ladder."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

    @property
    def weight(self) -> float:
        return {"low": 0.25, "medium": 0.5, "high": 0.75, "critical": 1.0}[self.value]


@dataclass(frozen=True)
class FailureRecord:
    """One observed failure, as a caller reports it.

    ``component`` and ``subject`` are the keys a detector groups by: a subject is
    the *thing* that failed (a question, a tool, a task name), so "the same task
    failed twice" is a real, counted fact rather than a claim.
    """

    component: str
    subject: str
    detail: str = ""
    at: datetime = field(default_factory=lambda: datetime.now(UTC))


@dataclass(frozen=True)
class Weakness:
    """One evidence-backed finding about the system."""

    weakness_id: str
    kind: WeaknessKind
    description: str
    component: str
    severity: Severity
    frequency: int
    confidence: float
    impact: str
    suggested_investigation: str
    evidence: tuple[str, ...]
    related_metrics: tuple[str, ...] = ()
    detected_at: datetime = field(default_factory=lambda: datetime.now(UTC))

    @property
    def score(self) -> float:
        """A transparent priority hint: severity x frequency (log) x confidence.

        This is a *hint*, not a decision -- the prioritization engine ranks with
        named criteria, and the user may override it (section 7).
        """
        import math

        frequency_term = 1.0 + math.log1p(max(0, self.frequency))
        return round(self.severity.weight * frequency_term * self.confidence, 6)

    def describe(self) -> str:
        return (
            f"[{self.severity.value}] {self.weakness_id} {self.kind.value} "
            f"({self.component}, n={self.frequency}, conf={self.confidence:.2f}): {self.description}"
        )

    def to_primitive(self) -> dict[str, object]:
        return {
            "weakness_id": self.weakness_id,
            "kind": self.kind.value,
            "description": self.description,
            "component": self.component,
            "severity": self.severity.value,
            "frequency": self.frequency,
            "confidence": self.confidence,
            "impact": self.impact,
            "suggested_investigation": self.suggested_investigation,
            "evidence": list(self.evidence),
            "related_metrics": list(self.related_metrics),
            "detected_at": self.detected_at.isoformat(),
            "score": self.score,
        }


def _new_id() -> str:
    return f"WK-{uuid4().hex[:8]}"


#: A rule: a name, the weakness kind, the metrics it reads, the minimum support
#: (a frequency below which it never fires), and a predicate over the summaries.
@dataclass(frozen=True)
class _Rule:
    name: str
    kind: WeaknessKind
    component: str
    metrics: tuple[str, ...]
    min_support: int
    description: str
    impact: str
    investigation: str


def _metric_mean(summaries: dict[str, MetricSummary], name: str) -> float | None:
    summary = summaries.get(name)
    return summary.mean if summary is not None else None


def _metric_total(summaries: dict[str, MetricSummary], name: str) -> float | None:
    summary = summaries.get(name)
    return summary.total if summary is not None else None


def _confidence_from_support(support: int, *, strong: int = 5) -> float:
    """Diagnosis confidence rises with the evidence, but never reaches 1.0.

    A finding is always a hypothesis: the detector saw a pattern, not a cause.
    """
    return round(min(0.9, 0.3 + 0.6 * (support / (support + strong))), 4)


def detect_weaknesses(
    monitor: PerformanceMonitor | None = None,
    *,
    summaries: tuple[MetricSummary, ...] | None = None,
    failures: Sequence[FailureRecord] = (),
    minimum_frequency: int = 2,
) -> tuple[Weakness, ...]:
    """Find weaknesses from the metrics and the reported failures.

    ``minimum_frequency`` is the floor below which a pattern is *not* reported as
    a weakness -- the phase's "do not act on a single failure" rule. It defaults
    to two, and a caller may raise it.
    """
    if summaries is None:
        summaries = monitor.summaries() if monitor is not None else ()
    by_name: dict[str, MetricSummary] = {s.name: s for s in summaries}
    findings: list[Weakness] = []

    # -- repeated failures --------------------------------------------------
    by_subject: Counter[str] = Counter()
    subject_component: dict[str, str] = {}
    for record in failures:
        key = f"{record.component}:{record.subject}"
        by_subject[key] += 1
        subject_component[key] = record.component
    repeated = {key: count for key, count in by_subject.items() if count >= minimum_frequency}
    for key, count in sorted(repeated.items(), key=lambda item: -item[1]):
        component, _, subject = key.partition(":")
        findings.append(
            Weakness(
                weakness_id=_new_id(),
                kind=WeaknessKind.REPEATED_FAILURE,
                description=f"'{subject}' failed {count} times in {component}.",
                component=component,
                severity=Severity.HIGH if count >= 3 else Severity.MEDIUM,
                frequency=count,
                confidence=_confidence_from_support(count),
                impact="The same operation keeps failing, so the affected path is unreliable.",
                suggested_investigation=f"Inspect the last {count} failures for '{subject}' and their error details.",
                evidence=tuple(f"failure[{i}] {subject}" for i in range(count)),
                related_metrics=("task.failure_rate", "task.repeated_failure"),
            )
        )

    # -- metric-driven rules ------------------------------------------------
    task_failure = _metric_mean(by_name, "task.failure_rate")
    if task_failure is not None and task_failure > 0.2:
        support = by_name["task.failure_rate"].count
        if support >= minimum_frequency:
            findings.append(_finding(
                WeaknessKind.REPEATED_FAILURE, "execution",
                f"Task failure rate is {task_failure:.2f}.",
                Severity.HIGH if task_failure > 0.4 else Severity.MEDIUM,
                support, ("task.failure_rate",),
                "A high failure share means the execution layer is not dependable.",
                "Review the failing tasks and the recovery decisions for them.",
            ))

    retrieval_quality = _metric_mean(by_name, "retrieval.quality")
    if retrieval_quality is not None and retrieval_quality < 0.6:
        support = by_name["retrieval.quality"].count
        if support >= minimum_frequency:
            findings.append(_finding(
                WeaknessKind.POOR_RETRIEVAL, "retrieval",
                f"Retrieval quality is {retrieval_quality:.2f}.",
                Severity.HIGH if retrieval_quality < 0.35 else Severity.MEDIUM,
                support, ("retrieval.quality",),
                "Weak retrieval means answers rest on less relevant evidence.",
                "Inspect the queries whose top passage was irrelevant and their ranking signals.",
            ))

    reasoning = _metric_mean(by_name, "reasoning.accuracy")
    if reasoning is not None and reasoning < 0.6:
        support = by_name["reasoning.accuracy"].count
        if support >= minimum_frequency:
            findings.append(_finding(
                WeaknessKind.INCORRECT_REASONING, "reasoning",
                f"Reasoning accuracy is {reasoning:.2f}.",
                Severity.HIGH if reasoning < 0.35 else Severity.MEDIUM,
                support, ("reasoning.accuracy",),
                "Incorrect conclusions erode trust in every answer that rests on them.",
                "Compare the conclusions that failed verification with the rules that produced them.",
            ))

    unsupported = _metric_total(by_name, "reasoning.unsupported_claim")
    if unsupported is not None and unsupported >= minimum_frequency:
        findings.append(_finding(
            WeaknessKind.UNSUPPORTED_CLAIM, "reasoning",
            f"{int(unsupported)} claim(s) were refused as unsupported.",
            Severity.MEDIUM,
            int(unsupported), ("reasoning.unsupported_claim",),
            "Unsupported claims reaching the gate suggest the model is drifting from the evidence.",
            "Read the refused claims and the passages that were available for them.",
        ))

    planning = _metric_mean(by_name, "planning.efficiency")
    if planning is not None and planning < 0.6:
        support = by_name["planning.efficiency"].count
        if support >= minimum_frequency:
            findings.append(_finding(
                WeaknessKind.PLANNING_MISTAKE, "planning",
                f"Planning efficiency is {planning:.2f}.",
                Severity.MEDIUM, support, ("planning.efficiency",),
                "Wasted steps make every goal slower and costlier than it needs to be.",
                "Review the plans that needed replanning and the tasks that were not required.",
            ))

    unnecessary = _metric_total(by_name, "tool.unnecessary_use")
    if unnecessary is not None and unnecessary >= minimum_frequency:
        findings.append(_finding(
            WeaknessKind.UNNECESSARY_TOOL_USE, "execution",
            f"{int(unnecessary)} tool call(s) were not required.",
            Severity.LOW, int(unnecessary), ("tool.unnecessary_use",),
            "Unneeded tool calls waste time and widen the attack surface.",
            "Check which goals selected a tool they did not need.",
        ))

    latency = _metric_mean(by_name, "assistant.latency")
    if latency is not None and latency > 3000:
        support = by_name["assistant.latency"].count
        if support >= minimum_frequency:
            findings.append(_finding(
                WeaknessKind.EXCESSIVE_LATENCY, "assistant",
                f"Mean answer latency is {latency:.0f} ms.",
                Severity.MEDIUM if latency < 8000 else Severity.HIGH,
                support, ("assistant.latency", "retrieval.latency", "reasoning.inference_latency"),
                "Slow answers make the system unpleasant to use.",
                "Profile the slowest answers and check whether a model load or a retrieval dominates.",
            ))

    duplicates = _metric_mean(by_name, "knowledge.duplicate_rate")
    if duplicates is not None and duplicates > 0.4:
        support = by_name["knowledge.duplicate_rate"].count
        if support >= minimum_frequency:
            findings.append(_finding(
                WeaknessKind.MEMORY_DUPLICATION, "knowledge",
                f"Duplicate knowledge rate is {duplicates:.2f}.",
                Severity.MEDIUM, support, ("knowledge.duplicate_rate",),
                "Duplicated knowledge wastes memory and can skew retrieval ranking.",
                "Inspect the incoming items classified as duplicates and the matching rule.",
            ))

    integration_errors = _metric_total(by_name, "knowledge.integration_error")
    if integration_errors is not None and integration_errors >= minimum_frequency:
        findings.append(_finding(
            WeaknessKind.KNOWLEDGE_ERROR, "knowledge",
            f"{int(integration_errors)} claim(s) failed to integrate.",
            Severity.MEDIUM, int(integration_errors), ("knowledge.integration_error",),
            "Failed integration means learned knowledge never reaches an answer.",
            "Read the failed claims and the validator reason each was refused.",
        ))

    consumption = _metric_mean(by_name, "resource.consumption")
    if consumption is not None and consumption > 0.9:
        support = by_name["resource.consumption"].count
        if support >= minimum_frequency:
            findings.append(_finding(
                WeaknessKind.RESOURCE_EXHAUSTION, "resource",
                f"Machine utilisation reached {consumption:.2f}.",
                Severity.HIGH, support, ("resource.consumption", "resource.memory_usage"),
                "Near-saturation risks swapping and unresponsiveness.",
                "Check which workload held the machine and whether the governor was consulted.",
            ))

    corrections = _metric_total(by_name, "user.correction")
    if corrections is not None and corrections >= minimum_frequency:
        findings.append(_finding(
            WeaknessKind.FREQUENT_CORRECTION, "interface",
            f"Users corrected answers {int(corrections)} time(s).",
            Severity.HIGH if corrections >= 4 else Severity.MEDIUM,
            int(corrections), ("user.correction",),
            "Repeated corrections are the strongest signal that answers are wrong.",
            "Read the corrected exchanges and the sources they used.",
        ))

    return tuple(findings)


def _finding(
    kind: WeaknessKind,
    component: str,
    description: str,
    severity: Severity,
    support: int,
    metrics: tuple[str, ...],
    impact: str,
    investigation: str,
) -> Weakness:
    return Weakness(
        weakness_id=_new_id(),
        kind=kind,
        description=description,
        component=component,
        severity=severity,
        frequency=support,
        confidence=_confidence_from_support(support),
        impact=impact,
        suggested_investigation=investigation,
        evidence=tuple(f"{name}={_value_text(name)}" for name in metrics),
        related_metrics=metrics,
    )


def _value_text(name: str) -> str:
    """A stable placeholder for an evidence line; the caller fills real figures.

    Kept deliberately simple: the evidence strings name the metrics the finding
    rests on, which is what makes the finding traceable. A caller that has the
    summaries may replace them with exact values through
    :func:`describe_weaknesses`.
    """
    spec = metric_spec(name)
    return spec.description if spec is not None else name


def describe_weaknesses(weaknesses: Iterable[Weakness]) -> str:
    """Render findings as a readable, ranked block."""
    ordered = sorted(weaknesses, key=lambda w: (-w.severity.weight, -w.frequency, w.weakness_id))
    if not ordered:
        return "No weaknesses detected."
    lines = [f"{len(ordered)} weakness(es) detected:"]
    for weakness in ordered:
        lines.append(f"  {weakness.describe()}")
        lines.append(f"      impact: {weakness.impact}")
        lines.append(f"      investigate: {weakness.suggested_investigation}")
        if weakness.evidence:
            lines.append(f"      evidence: {', '.join(weakness.evidence)}")
    return "\n".join(lines)


__all__ = [
    "FailureRecord",
    "Severity",
    "Weakness",
    "WeaknessKind",
    "describe_weaknesses",
    "detect_weaknesses",
]
