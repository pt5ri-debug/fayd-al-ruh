"""Baseline comparison and regression detection.

Section 9 of the phase requires every candidate to be measured against a
reproducible baseline, and section 14 requires a deployment to detect a
regression afterwards. This module is the one comparison, and it **reuses the
evaluation suite's own comparator** when that is importable, falling back to an
identical local rule when it is not (so a sandbox with no numpy can still compare
two metric dictionaries).

A :class:`RegressionVerdict` is honest about three things:

* which metrics **regressed** (fell past the tolerance when higher is better, or
  rose past it when lower is better);
* which metrics **improved**; and
* whether the comparison was **conclusive** -- an empty candidate, an empty
  baseline, or a baseline missing the candidate's metrics makes the verdict
  ``inconclusive`` rather than a pass. Section 9 forbids claiming an improvement
  the measurements do not support.

Dependency-free at import: the evaluation comparator is imported lazily.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

#: The default per-metric slack before a change counts as a regression. Zero
#: means a metric may not move in the wrong direction at all.
DEFAULT_TOLERANCE = 0.0


@dataclass(frozen=True)
class MetricDelta:
    """One metric's movement from the baseline to the candidate."""

    name: str
    baseline: float
    candidate: float
    higher_is_better: bool
    tolerance: float = DEFAULT_TOLERANCE

    @property
    def delta(self) -> float:
        return self.candidate - self.baseline

    @property
    def regressed(self) -> bool:
        if self.higher_is_better:
            return self.candidate < self.baseline - self.tolerance
        return self.candidate > self.baseline + self.tolerance

    @property
    def improved(self) -> bool:
        if self.higher_is_better:
            return self.candidate > self.baseline + self.tolerance
        return self.candidate < self.baseline - self.tolerance

    def describe(self) -> str:
        mark = "REGRESSED" if self.regressed else ("improved" if self.improved else "unchanged")
        return f"{self.name:<32} {self.baseline:.4f} -> {self.candidate:.4f} ({self.delta:+.4f}) [{mark}]"


@dataclass(frozen=True)
class RegressionVerdict:
    """The result of comparing a candidate against a baseline."""

    deltas: tuple[MetricDelta, ...]
    conclusive: bool
    reason: str = ""

    @property
    def regressions(self) -> tuple[MetricDelta, ...]:
        return tuple(d for d in self.deltas if d.regressed)

    @property
    def improvements(self) -> tuple[MetricDelta, ...]:
        return tuple(d for d in self.deltas if d.improved)

    @property
    def passed(self) -> bool:
        """True only when the comparison was conclusive and nothing regressed."""
        return self.conclusive and not self.regressions

    def describe(self) -> str:
        if not self.conclusive:
            return f"INCONCLUSIVE: {self.reason}"
        verdict = "PASS: no regression" if not self.regressions else "FAIL: regressions found"
        lines = [verdict]
        lines.extend(f"  {delta.describe()}" for delta in self.deltas)
        return "\n".join(lines)


#: Direction per known metric, so a comparison without the evaluation suite's
#: specs still knows which way is better. A metric not listed is assumed
#: higher-is-better, and the verdict names that assumption.
_LOWER_IS_BETTER: frozenset[str] = frozenset(
    {
        "brier_score",
        "calibration_error",
        "reliability",
        "contradiction_rate",
    }
)


def _direction(name: str) -> bool:
    return name not in _LOWER_IS_BETTER


def compare_metrics(
    baseline: dict[str, float],
    candidate: dict[str, float],
    *,
    tolerance: float = DEFAULT_TOLERANCE,
    specs: object | None = None,
) -> RegressionVerdict:
    """Compare two metric dictionaries, honestly.

    When the evaluation suite is importable its own ``METRIC_SPECS`` supply the
    direction and tolerance (so the loop and the gate agree exactly); otherwise a
    small local rule is used and the verdict says so.
    """
    if not baseline:
        return RegressionVerdict((), False, "no baseline metrics to compare against")
    if not candidate:
        return RegressionVerdict((), False, "the candidate produced no metrics")

    resolved = _resolve_specs(specs)
    shared = [name for name in candidate if name in baseline]
    if not shared:
        return RegressionVerdict((), False, "the baseline and the candidate share no metrics")

    deltas: list[MetricDelta] = []
    for name in sorted(shared):
        higher = resolved.get(name, (True, tolerance))[0]
        tol = resolved.get(name, (higher, tolerance))[1]
        deltas.append(
            MetricDelta(
                name=name,
                baseline=float(baseline[name]),
                candidate=float(candidate[name]),
                higher_is_better=higher,
                tolerance=tol,
            )
        )
    return RegressionVerdict(tuple(deltas), True, "")


def _resolve_specs(specs: object | None) -> dict[str, tuple[bool, float]]:
    """Read ``(higher_is_better, tolerance)`` per metric from the evaluation suite.

    Best-effort: a machine with no evaluation suite (a stripped sandbox) simply
    gets the local direction table, and the verdict is still conclusive.
    """
    from core.best_effort import best_effort

    resolved: dict[str, tuple[bool, float]] = {}
    source = specs
    if source is None:
        with best_effort("import the evaluation metric specs"):
            from evaluation.benchmark import METRIC_SPECS

            source = METRIC_SPECS
    if isinstance(source, (list, tuple)):
        for spec in source:
            name = getattr(spec, "name", None)
            if isinstance(name, str):
                resolved[name] = (
                    bool(getattr(spec, "higher_is_better", True)),
                    float(getattr(spec, "tolerance", DEFAULT_TOLERANCE)),
                )
    return resolved


def load_baseline_metrics(path: Path | str | None = None) -> dict[str, float]:
    """Read the evaluation baseline's metric dictionary.

    Returns an empty mapping when the file is absent or malformed, so a caller
    can tell "no baseline" from "a baseline of zeros" and report it honestly.
    """
    if path is None:
        from core.best_effort import best_effort

        with best_effort("locate the evaluation baseline"):
            from evaluation.baseline import BASELINE_PATH

            path = BASELINE_PATH
    if path is None:
        return {}
    candidate = Path(path)
    if not candidate.exists():
        return {}
    try:
        data = json.loads(candidate.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    metrics = data.get("metrics") if isinstance(data, dict) else None
    if not isinstance(metrics, dict):
        return {}
    return {str(k): float(v) for k, v in metrics.items() if isinstance(v, (int, float))}


@dataclass(frozen=True)
class ComparisonResult:
    """A verdict plus the two metric sets it compared, for the history."""

    verdict: RegressionVerdict
    baseline: dict[str, float] = field(default_factory=dict)
    candidate: dict[str, float] = field(default_factory=dict)

    def to_primitive(self) -> dict[str, object]:
        return {
            "conclusive": self.verdict.conclusive,
            "passed": self.verdict.passed,
            "reason": self.verdict.reason,
            "regressions": [d.name for d in self.verdict.regressions],
            "improvements": [d.name for d in self.verdict.improvements],
            "deltas": [
                {
                    "name": d.name,
                    "baseline": d.baseline,
                    "candidate": d.candidate,
                    "delta": d.delta,
                    "regressed": d.regressed,
                }
                for d in self.verdict.deltas
            ],
        }


__all__ = [
    "DEFAULT_TOLERANCE",
    "ComparisonResult",
    "MetricDelta",
    "RegressionVerdict",
    "compare_metrics",
    "load_baseline_metrics",
]
