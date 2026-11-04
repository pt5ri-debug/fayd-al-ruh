"""The experiment manager: one candidate change, run in isolation, measured.

An experiment ties a proposal to a sandbox, a candidate change, a baseline, and a
measured evaluation. It is the unit the phase's sections 8 and 9 describe, and it
composes the pieces already written -- the sandbox (isolation), the candidate
evaluator (measurement), and the history (the record) -- rather than adding a
mechanism of its own.

An :class:`Experiment` is a **record**, and :class:`ExperimentManager` is the one
thing that runs one: it creates a sandbox, applies the candidate's change inside
it, runs the checks, evaluates the candidate against the baseline, records the
result, and **always tears the sandbox down** -- keeping the artifacts when asked,
so a failed experiment is still inspectable. Nothing here merges, deploys, or
touches production; an experiment that passes is *offered* for approval, never
applied.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import Enum
from pathlib import Path
from uuid import uuid4

from core.improvement.evaluation import (
    CandidateRunner,
    CandidateStatus,
    EvaluationCriteria,
    EvaluationReport,
    evaluate_candidate,
)
from core.improvement.history import EntryKind, ImprovementHistory
from core.improvement.sandbox import SandboxError, SandboxLimits, SandboxManager, SandboxSession

#: A candidate change: a set of relative-path → new-content edits to apply inside
#: the sandbox before it is evaluated. Kept as plain data so a proposal's change
#: can be expressed without executing anything.
CandidateChange = dict[str, str]


class ExperimentStatus(str, Enum):
    """Where an experiment got to."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass(frozen=True)
class Experiment:
    """One isolated candidate experiment, with its measured outcome."""

    experiment_id: str
    proposal_id: str
    problem: str
    status: ExperimentStatus
    change: CandidateChange = field(default_factory=dict)
    checks_passed: bool = False
    checks_summary: str = ""
    evaluation: EvaluationReport | None = None
    sandbox_id: str = ""
    artifacts_dir: str = ""
    started_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    finished_at: datetime | None = None
    error: str = ""

    @property
    def passed(self) -> bool:
        """True only when the checks passed *and* the evaluation measured a gain."""
        return self.checks_passed and self.evaluation is not None and self.evaluation.acceptable

    def describe(self) -> str:
        lines = [
            f"Experiment {self.experiment_id} for {self.proposal_id}: {self.status.value}",
            f"  checks: {'passed' if self.checks_passed else 'failed'}",
        ]
        if self.checks_summary:
            lines.append(f"  {self.checks_summary}")
        if self.evaluation is not None:
            lines.append("  " + self.evaluation.describe().replace("\n", "\n  "))
        if self.error:
            lines.append(f"  error: {self.error}")
        return "\n".join(lines)

    def to_primitive(self) -> dict[str, object]:
        return {
            "experiment_id": self.experiment_id,
            "proposal_id": self.proposal_id,
            "problem": self.problem,
            "status": self.status.value,
            "change": dict(self.change),
            "checks_passed": self.checks_passed,
            "checks_summary": self.checks_summary,
            "sandbox_id": self.sandbox_id,
            "artifacts_dir": self.artifacts_dir,
            "error": self.error,
            "started_at": self.started_at.isoformat(),
            "finished_at": self.finished_at.isoformat() if self.finished_at else None,
            "evaluation": self.evaluation.to_primitive() if self.evaluation else None,
        }


class ExperimentManager:
    """Runs candidate experiments in isolated sandboxes and records the results."""

    def __init__(
        self,
        *,
        sandbox: SandboxManager,
        history: ImprovementHistory | None = None,
        baseline_metrics: dict[str, float] | None = None,
        criteria: EvaluationCriteria | None = None,
        artifacts_dir: Path | str | None = None,
    ) -> None:
        self.sandbox = sandbox
        self.history = history
        self.baseline_metrics = dict(baseline_metrics or {})
        self.criteria = criteria or EvaluationCriteria()
        self.artifacts_dir = Path(artifacts_dir) if artifacts_dir is not None else None
        self._experiments: dict[str, Experiment] = {}

    # -- reading ------------------------------------------------------------
    def get(self, experiment_id: str) -> Experiment | None:
        return self._experiments.get(experiment_id)

    def experiments(self) -> tuple[Experiment, ...]:
        return tuple(self._experiments.values())

    def for_proposal(self, proposal_id: str) -> tuple[Experiment, ...]:
        return tuple(e for e in self._experiments.values() if e.proposal_id == proposal_id)

    # -- running ------------------------------------------------------------
    def run(
        self,
        *,
        proposal_id: str,
        problem: str,
        change: CandidateChange,
        limits: SandboxLimits | None = None,
        runner: CandidateRunner | None = None,
        check_command: tuple[str, ...] | None = None,
        keep_artifacts: bool = True,
        on_progress: Callable[[str], None] | None = None,
    ) -> Experiment:
        """Run one candidate end to end in isolation and record the outcome.

        The sandbox is always torn down. ``keep_artifacts`` leaves the throwaway
        data root in place so a caller can inspect what the experiment wrote;
        that is the only thing that survives.
        """
        experiment_id = f"EXP-{uuid4().hex[:8]}"
        record = Experiment(
            experiment_id=experiment_id,
            proposal_id=proposal_id,
            problem=problem,
            status=ExperimentStatus.RUNNING,
            change=dict(change),
        )
        self._experiments[experiment_id] = record
        self._note(EntryKind.EXPERIMENT, f"Started experiment {experiment_id} for {proposal_id}.",
                   proposal_id=proposal_id, detail={"change": list(change)})

        session: SandboxSession | None = None
        try:
            session = self.sandbox.create(task=f"experiment for {proposal_id}", limits=limits)
            record = self._replace(record, sandbox_id=session.sandbox_id, artifacts_dir=str(session.data_root))
            self._experiments[experiment_id] = record

            for relative, content in change.items():
                self.sandbox.write(session, relative, content)
            if on_progress is not None:
                on_progress(f"applied {len(change)} file(s) in {session.sandbox_id}")

            checks_passed, checks_summary = self.sandbox.run_checks(
                session, command=check_command, progress=on_progress
            )
            record = self._replace(record, checks_passed=checks_passed, checks_summary=checks_summary)
            self._experiments[experiment_id] = record

            report = evaluate_candidate(
                proposal_id=proposal_id,
                problem=problem,
                session=session,
                baseline_metrics=self.baseline_metrics,
                run=runner,
                criteria=self.criteria,
                baseline_source="evaluation/baseline.json",
            )
            status = (
                ExperimentStatus.COMPLETED
                if report.status in {CandidateStatus.IMPROVED, CandidateStatus.UNCHANGED, CandidateStatus.REGRESSED}
                else ExperimentStatus.FAILED
            )
            record = self._replace(
                record,
                status=status,
                evaluation=report,
                finished_at=datetime.now(UTC),
            )
            self._experiments[experiment_id] = record
            self._note(
                EntryKind.EVALUATION,
                f"{experiment_id} evaluated: {report.status.value} (checks "
                f"{'passed' if checks_passed else 'failed'}).",
                proposal_id=proposal_id,
                detail=report.to_primitive(),
            )
            return record
        except SandboxError as error:
            record = self._replace(
                record,
                status=ExperimentStatus.FAILED,
                error=str(error),
                finished_at=datetime.now(UTC),
            )
            self._experiments[experiment_id] = record
            self._note(EntryKind.EXPERIMENT, f"{experiment_id} could not run: {error}", proposal_id=proposal_id)
            return record
        finally:
            if session is not None:
                self.sandbox.cleanup(session, keep_artifacts=keep_artifacts)

    def cancel(self, experiment_id: str) -> bool:
        """Ask a running experiment to stop (cooperative)."""
        record = self._experiments.get(experiment_id)
        if record is None or record.status is not ExperimentStatus.RUNNING:
            return False
        session = next((s for s in self.sandbox.sessions() if s.sandbox_id == record.sandbox_id), None)
        if session is not None:
            session.cancel()
        self._experiments[experiment_id] = self._replace(record, status=ExperimentStatus.CANCELLED)
        return True

    def describe(self) -> str:
        if not self._experiments:
            return "No experiments have run."
        lines = [f"{len(self._experiments)} experiment(s):"]
        for record in self._experiments.values():
            lines.append(f"  {record.experiment_id} [{record.status.value}] {record.proposal_id}")
        return "\n".join(lines)

    # -- helpers ------------------------------------------------------------
    @staticmethod
    def _replace(record: Experiment, **changes: object) -> Experiment:
        import dataclasses

        return dataclasses.replace(record, **changes)  # type: ignore[arg-type]

    def _note(
        self,
        kind: EntryKind,
        summary: str,
        *,
        proposal_id: str = "",
        detail: dict[str, object] | None = None,
    ) -> None:
        if self.history is not None:
            self.history.append(kind, summary, proposal_id=proposal_id, detail=detail or {})


__all__ = [
    "CandidateChange",
    "Experiment",
    "ExperimentManager",
    "ExperimentStatus",
]
