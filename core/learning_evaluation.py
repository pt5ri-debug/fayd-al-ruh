"""Learning evaluation: measure whether learning actually improved anything.

Phase 3 requires learning evaluation, knowledge evaluation, learning metrics, and
learning history/version tracking -- and, above all, it requires the system to
**measure** improvement rather than claim it.

This module is that measurement. It snapshots a coordinator's state, runs a
learning action, snapshots again, and reports the **measured** delta: knowledge
integrated, lessons promoted, procedures trusted, feedback processed, and the
change in a small set of named metrics. Nothing here asserts an improvement --
a delta of zero is reported as zero, and a metric that fell is reported as a
regression.

**Honest by construction.** Every figure is read from the real stores; a metric
with no measurable value is ``None`` rather than a fabricated number, and the
report says plainly when there is nothing to compare.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from core.learning_coordinator import LearningCoordinator
from core.learning_events import KIND_KNOWLEDGE

#: The metrics a learning evaluation reports (all read from the real stores).
METRIC_KNOWLEDGE = "knowledge_events"
METRIC_EXPERIENCE = "experience_events"
METRIC_FEEDBACK = "feedback_events"
METRIC_PROCEDURES = "procedures"
METRIC_TRUSTED = "trusted_procedures"
METRIC_LESSONS = "lessons"

LEARNING_METRICS: tuple[str, ...] = (
    METRIC_KNOWLEDGE,
    METRIC_EXPERIENCE,
    METRIC_FEEDBACK,
    METRIC_PROCEDURES,
    METRIC_TRUSTED,
    METRIC_LESSONS,
)


@dataclass(frozen=True)
class LearningSnapshot:
    """One measured snapshot of a coordinator's learning state."""

    metrics: dict[str, float] = field(default_factory=dict)
    version: int = 0
    events: int = 0

    def get(self, name: str) -> float | None:
        """The metric's value, or ``None`` when it was not measurable."""
        return self.metrics.get(name)


@dataclass(frozen=True)
class LearningComparison:
    """The measured difference between two snapshots, per metric."""

    before: LearningSnapshot
    after: LearningSnapshot
    deltas: dict[str, float] = field(default_factory=dict)

    @property
    def improved(self) -> tuple[str, ...]:
        return tuple(name for name, delta in self.deltas.items() if delta > 0)

    @property
    def regressed(self) -> tuple[str, ...]:
        """Metrics that fell -- reported honestly, never hidden."""
        return tuple(name for name, delta in self.deltas.items() if delta < 0)

    @property
    def changed(self) -> bool:
        return any(delta != 0 for delta in self.deltas.values())

    def describe(self) -> str:
        if not self.deltas:
            return "no metrics to compare"
        parts = ", ".join(f"{name}={delta:+g}" for name, delta in sorted(self.deltas.items()))
        verdict = "improved" if self.improved and not self.regressed else (
            "regressed" if self.regressed else "unchanged"
        )
        return f"{verdict}: {parts}"


@dataclass
class LearningReport:
    """The outcome of one evaluated learning action."""

    action: str
    comparison: LearningComparison
    detail: str = ""

    @property
    def improved(self) -> bool:
        return bool(self.comparison.improved) and not self.comparison.regressed

    def describe(self) -> str:
        return f"{self.action}: {self.comparison.describe()}" + (f" ({self.detail})" if self.detail else "")


def snapshot(coordinator: LearningCoordinator) -> LearningSnapshot:
    """Read one honest snapshot of a coordinator's learning state."""
    status = coordinator.status()
    counts = status.counts
    return LearningSnapshot(
        metrics={
            METRIC_KNOWLEDGE: float(counts.get(KIND_KNOWLEDGE, 0)),
            METRIC_EXPERIENCE: float(counts.get("experience", 0)),
            METRIC_FEEDBACK: float(counts.get("feedback", 0)),
            METRIC_PROCEDURES: float(status.procedures),
            METRIC_TRUSTED: float(status.trusted_procedures),
            METRIC_LESSONS: float(status.lessons),
        },
        version=status.version,
        events=status.events,
    )


def compare(before: LearningSnapshot, after: LearningSnapshot) -> LearningComparison:
    """The measured per-metric difference between two snapshots."""
    names = set(before.metrics) | set(after.metrics)
    deltas: dict[str, float] = {}
    for name in names:
        first = before.get(name)
        second = after.get(name)
        if first is None or second is None:
            continue
        deltas[name] = round(second - first, 6)
    return LearningComparison(before=before, after=after, deltas=deltas)


def evaluate_learning(
    coordinator: LearningCoordinator,
    action: Callable[[], Any],
    *,
    name: str = "learning action",
) -> LearningReport:
    """Run ``action`` and report the **measured** learning it produced.

    The before/after snapshots come from the real stores, so an action that
    changed nothing is reported as unchanged rather than as an improvement.
    """
    before = snapshot(coordinator)
    result = action()
    after = snapshot(coordinator)
    comparison = compare(before, after)
    detail = ""
    if hasattr(result, "describe"):
        detail = str(result.describe())
    return LearningReport(action=name, comparison=comparison, detail=detail)


def learning_metrics(coordinator: LearningCoordinator) -> dict[str, float]:
    """The coordinator's current metrics as a plain dictionary."""
    return dict(snapshot(coordinator).metrics)


def history_summary(coordinator: LearningCoordinator, *, limit: int = 200) -> str:
    """A readable history summary: how many events of each kind, newest first."""
    events = coordinator.history(limit=limit)
    if not events:
        return "no learning events recorded yet"
    counts: dict[str, int] = {}
    for event in events:
        counts[event.kind] = counts.get(event.kind, 0) + 1
    parts = ", ".join(f"{kind}={count}" for kind, count in sorted(counts.items()))
    latest = events[0]
    return f"{len(events)} event(s): {parts}; latest: {latest.describe()}"


def version_history(coordinator: LearningCoordinator) -> tuple[int, ...]:
    """Every version the coordinator's store has stamped, newest first."""
    if coordinator.store is None:
        return ()
    versions: list[int] = []
    for event in coordinator.history(limit=1000):
        if not event.version:
            continue
        try:
            versions.append(int(event.version))
        except ValueError:
            continue
    return tuple(sorted(set(versions), reverse=True))


__all__ = [
    "LEARNING_METRICS",
    "METRIC_EXPERIENCE",
    "METRIC_FEEDBACK",
    "METRIC_KNOWLEDGE",
    "METRIC_LESSONS",
    "METRIC_PROCEDURES",
    "METRIC_TRUSTED",
    "LearningComparison",
    "LearningReport",
    "LearningSnapshot",
    "compare",
    "evaluate_learning",
    "history_summary",
    "learning_metrics",
    "snapshot",
    "version_history",
]
