"""Candidate evaluation: the ten-step, reproducible comparison the phase requires.

Section 9 of the phase lays out a fixed process -- define the problem, establish
baseline metrics, define success criteria, prepare a workload, run the candidate
in isolation, measure, compare, detect regressions, assess resource impact, and
produce a report -- and forbids claiming an improvement the measurements do not
support. This module follows that order exactly and makes the last rule
structural: a candidate whose run produced no metrics, or whose comparison is
inconclusive, is marked :attr:`CandidateStatus.INCONCLUSIVE`, never ``improved``.

The candidate is executed through an injected ``run`` callable -- the seam a test
supplies -- so the whole engine is exercised with no subprocess and no model. The
default runner is the real thing: it runs ``python -m evaluation --json`` *inside*
the sandbox, with the sandbox's throwaway data root, and parses the metric
dictionary it prints. Nothing here loads a model, reaches the network, or writes
to production state.
"""

from __future__ import annotations

import json
import subprocess
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum

from core.improvement.regression import ComparisonResult, compare_metrics
from core.improvement.sandbox import SandboxError, SandboxSession


class CandidateStatus(str, Enum):
    """The honest outcome of one candidate evaluation."""

    IMPROVED = "improved"
    UNCHANGED = "unchanged"
    REGRESSED = "regressed"
    INCONCLUSIVE = "inconclusive"
    ERROR = "error"


@dataclass(frozen=True)
class CandidateRun:
    """What a candidate run produced: its metrics and its resource cost."""

    metrics: dict[str, float] = field(default_factory=dict)
    elapsed_seconds: float = 0.0
    peak_memory_bytes: int | None = None
    note: str = ""


#: The seam: a candidate runner reads the sandbox and returns its metrics.
CandidateRunner = Callable[[SandboxSession], CandidateRun]


@dataclass(frozen=True)
class EvaluationCriteria:
    """The success criteria a candidate must meet (section 9, step 3)."""

    #: A metric must improve by at least this much to count as a real gain.
    min_improvement: float = 0.0
    #: A metric may not fall by more than this before the candidate is rejected.
    max_regression: float = 0.0
    #: Metrics that must not regress at all, whatever the tolerance.
    required_metrics: tuple[str, ...] = ()
    #: A ceiling on the candidate's memory use, in bytes (a resource impact gate).
    max_memory_bytes: int | None = None

    def __post_init__(self) -> None:
        if self.min_improvement < 0 or self.max_regression < 0:
            raise ValueError("Evaluation thresholds cannot be negative.")


@dataclass(frozen=True)
class EvaluationReport:
    """The measured result of one candidate evaluation (section 9, step 10)."""

    proposal_id: str
    problem: str
    status: CandidateStatus
    comparison: ComparisonResult
    criteria: EvaluationCriteria
    run: CandidateRun
    reasons: tuple[str, ...] = ()
    baseline_source: str = ""

    @property
    def improved(self) -> bool:
        return self.status is CandidateStatus.IMPROVED

    @property
    def acceptable(self) -> bool:
        """True only for a measured improvement that met the criteria."""
        return self.status is CandidateStatus.IMPROVED and not self.reasons

    def version_evaluations(self) -> tuple[object, ...]:
        """The measured deltas as the model registry's own ``Evaluation`` records.

        This is what lets the loop reuse the Phase-6 acceptance rule
        (:meth:`cognition.versions.ModelRegistry.should_accept`) rather than
        inventing a second definition of "an improvement". The registry is
        imported lazily so a stripped sandbox with no cognition package still
        reports the deltas it measured.
        """
        from core.best_effort import best_effort

        records: list[object] = []
        with best_effort("build version evaluation records"):
            from cognition.versions import Evaluation

            for delta in self.comparison.verdict.deltas:
                records.append(
                    Evaluation(
                        metric=delta.name,
                        before=delta.baseline,
                        after=delta.candidate,
                        lower_is_better=not delta.higher_is_better,
                    )
                )
        return tuple(records)

    def accepted_by_registry(self) -> tuple[bool, str]:
        """Ask the model registry's acceptance rule whether this change passes.

        Returns ``(accepted, reason)``; a machine with no cognition package
        returns ``(False, reason)`` honestly rather than assuming acceptance.
        """
        records = self.version_evaluations()
        if not records:
            return False, "no measured deltas to hand the acceptance rule"
        from core.best_effort import best_effort

        outcome: tuple[bool, str] = (False, "the acceptance rule is unavailable")
        with best_effort("consult the model registry's acceptance rule"):
            from cognition.versions import ModelRegistry

            outcome = ModelRegistry().should_accept_improvement(
                records,  # type: ignore[arg-type]
                max_regression=self.criteria.max_regression,
                min_improvement=self.criteria.min_improvement,
            )
        return outcome

    def describe(self) -> str:
        lines = [
            f"Candidate evaluation for {self.proposal_id}: {self.status.value}",
            f"  problem: {self.problem}",
            f"  baseline: {self.baseline_source or 'unspecified'}",
            f"  candidate elapsed: {self.run.elapsed_seconds:.2f}s",
        ]
        if self.run.peak_memory_bytes is not None:
            lines.append(f"  peak memory: {self.run.peak_memory_bytes / (1024 * 1024):.1f} MB")
        if self.run.note:
            lines.append(f"  note: {self.run.note}")
        lines.append("  " + self.comparison.verdict.describe().replace("\n", "\n  "))
        for reason in self.reasons:
            lines.append(f"  ! {reason}")
        return "\n".join(lines)

    def to_primitive(self) -> dict[str, object]:
        return {
            "proposal_id": self.proposal_id,
            "problem": self.problem,
            "status": self.status.value,
            "acceptable": self.acceptable,
            "reasons": list(self.reasons),
            "baseline_source": self.baseline_source,
            "criteria": {
                "min_improvement": self.criteria.min_improvement,
                "max_regression": self.criteria.max_regression,
                "required_metrics": list(self.criteria.required_metrics),
                "max_memory_bytes": self.criteria.max_memory_bytes,
            },
            "run": {
                "elapsed_seconds": self.run.elapsed_seconds,
                "peak_memory_bytes": self.run.peak_memory_bytes,
                "note": self.run.note,
                "metrics": dict(self.run.metrics),
            },
            "comparison": self.comparison.to_primitive(),
        }


def evaluation_runner(session: SandboxSession, *, extra_args: tuple[str, ...] = ()) -> CandidateRun:
    """Run the project's own evaluation suite inside the sandbox and read its metrics.

    This is the real default runner. It starts ``python -m evaluation --json``
    with the sandbox's environment (so the candidate's data path is the throwaway
    root) and a wall-clock bound, and returns the metrics it measured. A run that
    fails, times out, or prints no metrics returns an empty :class:`CandidateRun`,
    which the evaluator turns into an honest ``INCONCLUSIVE`` rather than a pass.
    """
    started = time.monotonic()
    command = [sys.executable, "-m", "evaluation", "--json", "--no-check", *extra_args]
    try:
        completed = subprocess.run(
            command,
            cwd=str(session.root),
            env=session.environment(),
            capture_output=True,
            text=True,
            timeout=session.limits.timeout_seconds,
            check=False,
        )
    except (OSError, subprocess.SubprocessError) as error:
        return CandidateRun(note=f"the evaluation run could not start: {error}",
                            elapsed_seconds=time.monotonic() - started)
    elapsed = time.monotonic() - started
    metrics = _parse_metrics(completed.stdout)
    if not metrics:
        tail = (completed.stderr or completed.stdout or "").strip().splitlines()[-3:]
        return CandidateRun(note=f"no metrics were produced (exit {completed.returncode}): {' | '.join(tail)}",
                            elapsed_seconds=elapsed)
    return CandidateRun(metrics=metrics, elapsed_seconds=elapsed)


def _parse_metrics(stdout: str) -> dict[str, float]:
    """Read the ``metrics`` object out of the evaluation's JSON report."""
    text = stdout.strip()
    if not text:
        return {}
    # The report may be preceded by log lines; take the last JSON object.
    start = text.find("{")
    if start < 0:
        return {}
    try:
        data = json.loads(text[start:])
    except ValueError:
        return {}
    metrics = data.get("metrics") if isinstance(data, dict) else None
    if not isinstance(metrics, dict):
        return {}
    return {str(k): float(v) for k, v in metrics.items() if isinstance(v, (int, float))}


def evaluate_candidate(
    *,
    proposal_id: str,
    problem: str,
    session: SandboxSession,
    baseline_metrics: dict[str, float],
    run: CandidateRunner | None = None,
    criteria: EvaluationCriteria | None = None,
    baseline_source: str = "",
) -> EvaluationReport:
    """Run one candidate in isolation and compare it against the baseline.

    ``run`` is the seam; when it is absent the real evaluation runner is used.
    The function never raises for a failed candidate -- a failure is a reported
    ``ERROR`` status, so one bad experiment cannot break the loop.
    """
    criteria = criteria or EvaluationCriteria()
    runner = run or evaluation_runner
    try:
        candidate = runner(session)
    except SandboxError as error:
        return EvaluationReport(
            proposal_id=proposal_id,
            problem=problem,
            status=CandidateStatus.ERROR,
            comparison=ComparisonResult(compare_metrics(baseline_metrics, {})),
            criteria=criteria,
            run=CandidateRun(note=str(error)),
            reasons=(f"the candidate could not run: {error}",),
            baseline_source=baseline_source,
        )
    except Exception as error:
        return EvaluationReport(
            proposal_id=proposal_id,
            problem=problem,
            status=CandidateStatus.ERROR,
            comparison=ComparisonResult(compare_metrics(baseline_metrics, {})),
            criteria=criteria,
            run=CandidateRun(note=f"{type(error).__name__}: {error}"),
            reasons=(f"the candidate runner raised {type(error).__name__}: {error}",),
            baseline_source=baseline_source,
        )

    comparison = ComparisonResult(compare_metrics(baseline_metrics, candidate.metrics))
    reasons: list[str] = []

    if not comparison.verdict.conclusive:
        status = CandidateStatus.INCONCLUSIVE
        reasons.append(comparison.verdict.reason or "the comparison was inconclusive")
    else:
        verdict = comparison.verdict
        # Hard requirement: a metric that must not regress did.
        for name in criteria.required_metrics:
            delta = next((d for d in verdict.deltas if d.name == name), None)
            if delta is not None and delta.regressed:
                reasons.append(f"required metric '{name}' regressed by {delta.delta:+.4f}")
        # A regression past the tolerance rejects the candidate.
        for delta in verdict.regressions:
            if abs(delta.delta) > criteria.max_regression:
                reasons.append(f"'{delta.name}' regressed by {delta.delta:+.4f} (limit {criteria.max_regression})")
        # Resource impact: an over-budget candidate is refused.
        if (
            criteria.max_memory_bytes is not None
            and candidate.peak_memory_bytes is not None
            and candidate.peak_memory_bytes > criteria.max_memory_bytes
        ):
            reasons.append(
                f"peak memory {candidate.peak_memory_bytes} exceeds the {criteria.max_memory_bytes} budget"
            )
        improvements = [d for d in verdict.improvements if abs(d.delta) > criteria.min_improvement]
        if reasons:
            status = CandidateStatus.REGRESSED
        elif improvements:
            status = CandidateStatus.IMPROVED
        else:
            status = CandidateStatus.UNCHANGED
            reasons.append("no metric improved beyond the required threshold")

    return EvaluationReport(
        proposal_id=proposal_id,
        problem=problem,
        status=status,
        comparison=comparison,
        criteria=criteria,
        run=candidate,
        reasons=tuple(reasons),
        baseline_source=baseline_source,
    )


__all__ = [
    "CandidateRun",
    "CandidateRunner",
    "CandidateStatus",
    "EvaluationCriteria",
    "EvaluationReport",
    "evaluate_candidate",
    "evaluation_runner",
]
