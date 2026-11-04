"""Outcome verification (Phase 8, section 12).

Section 12 is explicit: *a tool returning successfully is not the same as a goal
being achieved*. After execution the verifier compares the actual output against
the task's own success criteria, checks whether expected artifacts exist, and
decides one of five outcomes -- verified success, partial success, failed,
unverified, or cancelled. It also names side effects and whether a follow-up task
is needed.

The verifier never reads a tool result as a conclusion: it checks the *criteria*
the plan declared, and when it cannot check them (a criterion that names no
artifact and no output requirement) it says so honestly rather than claiming
success.

Dependency-free and Tk-free.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

from core.execution.engine import ExecutionReport, TaskExecution
from core.planning.goals import Goal
from core.planning.plan import TaskPlan
from core.planning.tasks import TaskState


class Outcome(str, Enum):
    """The five outcomes section 12 names."""

    VERIFIED_SUCCESS = "verified_success"
    PARTIAL_SUCCESS = "partial_success"
    FAILED = "failed"
    UNVERIFIED = "unverified"
    CANCELLED = "cancelled"


@dataclass(frozen=True)
class VerificationResult:
    """The verdict for one plan run, with every finding."""

    outcome: Outcome
    goal_achieved: bool
    verified_tasks: tuple[str, ...] = ()
    unverified_tasks: tuple[str, ...] = ()
    failed_tasks: tuple[str, ...] = ()
    missing_artifacts: tuple[str, ...] = ()
    side_effects: tuple[str, ...] = ()
    follow_up: tuple[str, ...] = ()
    reasons: tuple[str, ...] = ()

    @property
    def success(self) -> bool:
        return self.outcome is Outcome.VERIFIED_SUCCESS

    def describe(self) -> str:
        lines = [f"{self.outcome.value}: goal {'achieved' if self.goal_achieved else 'not achieved'}"]
        lines.extend(f"  ! {reason}" for reason in self.reasons)
        if self.missing_artifacts:
            lines.append("  missing: " + ", ".join(self.missing_artifacts))
        if self.side_effects:
            lines.append("  side effects: " + ", ".join(self.side_effects))
        if self.follow_up:
            lines.append("  follow-up: " + "; ".join(self.follow_up))
        return "\n".join(lines)


#: A checker for an artifact path -- injectable so a test needs no real file.
ArtifactCheck = Callable[[str], bool]


def _default_artifact_check(path: str) -> bool:
    return Path(path).exists()


def _criterion_met(criterion: str, execution: TaskExecution) -> bool | None:
    """Whether one success criterion is met, or ``None`` when unverifiable.

    A criterion the verifier can read deterministically is checked; anything it
    cannot check returns ``None`` so the overall verdict is *unverified* rather
    than a claimed success.
    """
    text = criterion.casefold()
    if "produced output" in text or "output" in text or "result" in text:
        return bool(execution.output.strip())
    if "wrote" in text or "written" in text or "file" in text:
        return bool(execution.output.strip())
    if "completed without error" in text or "succeeded" in text:
        return execution.ok
    return None


@dataclass
class OutcomeVerifier:
    """Check a run's outputs against the plan's declared criteria."""

    artifact_check: ArtifactCheck = _default_artifact_check
    #: The working directory artifacts are resolved against (the sandbox root).
    root: Path | None = None

    def verify(self, plan: TaskPlan, goal: Goal, report: ExecutionReport) -> VerificationResult:
        """Verify ``report`` against ``plan``'s tasks and ``goal``'s criteria."""
        if report.cancelled:
            return VerificationResult(
                Outcome.CANCELLED, False,
                reasons=("the run was cancelled before it finished.",),
            )

        by_id = {execution.task_id: execution for execution in report.executions}
        verified: list[str] = []
        unverified: list[str] = []
        failed: list[str] = []
        reasons: list[str] = []
        side_effects: list[str] = []

        for task in plan.tasks:
            execution = by_id.get(task.task_id)
            if execution is None:
                unverified.append(task.task_id)
                reasons.append(f"task {task.task_id} never ran.")
                continue
            if execution.state is TaskState.CANCELLED:
                return VerificationResult(
                    Outcome.CANCELLED, False,
                    reasons=(f"task {task.task_id} was cancelled.",),
                )
            if not execution.ok:
                failed.append(task.task_id)
                reasons.append(
                    f"task {task.task_id} failed: {execution.error or 'the tool reported failure'}"
                )
                continue

            # Check the task's own criteria.
            checks = [_criterion_met(criterion, execution) for criterion in task.success_criteria]
            if not task.success_criteria:
                checks = [True if execution.output.strip() else None]
            if any(check is False for check in checks):
                failed.append(task.task_id)
                reasons.append(f"task {task.task_id} ran but its criteria were not met.")
            elif any(check is None for check in checks):
                unverified.append(task.task_id)
                reasons.append(f"task {task.task_id} produced output the verifier cannot judge.")
            else:
                verified.append(task.task_id)
            if execution.output.strip():
                side_effects.append(f"{task.task_id} produced {len(execution.output)} chars of output")

        # Check the goal's declared artifacts exist.
        missing: list[str] = []
        for artifact in goal.verification.artifacts:
            path = self._resolve(artifact)
            if not self.artifact_check(str(path)):
                missing.append(artifact)
        if missing:
            reasons.append("expected artifacts are missing: " + ", ".join(missing))

        # The goal is achieved only when the minimum number of tasks verified and
        # no artifact is missing and nothing failed.
        min_tasks = max(1, goal.verification.min_successful_tasks)
        goal_achieved = (
            len(verified) >= min_tasks and not missing and not failed
        )

        # Decide the overall outcome.
        if failed and not verified:
            outcome = Outcome.FAILED
        elif failed or missing or unverified:
            outcome = Outcome.PARTIAL_SUCCESS if verified else Outcome.UNVERIFIED
        elif goal_achieved:
            outcome = Outcome.VERIFIED_SUCCESS
        else:
            outcome = Outcome.UNVERIFIED

        follow_up: list[str] = []
        if missing:
            follow_up.append("produce the missing artifacts: " + ", ".join(missing))
        for task_id in failed:
            follow_up.append(f"recover task {task_id}")
        for task_id in unverified:
            follow_up.append(f"review task {task_id} (its output could not be judged)")

        return VerificationResult(
            outcome=outcome,
            goal_achieved=goal_achieved,
            verified_tasks=tuple(verified),
            unverified_tasks=tuple(unverified),
            failed_tasks=tuple(failed),
            missing_artifacts=tuple(missing),
            side_effects=tuple(side_effects),
            follow_up=tuple(follow_up),
            reasons=tuple(reasons),
        )

    def _resolve(self, artifact: str) -> Path:
        if self.root is not None:
            return self.root / artifact
        return Path(artifact)


def outcome_from_report(report: ExecutionReport) -> Outcome:
    """A coarse outcome from a report alone (used when no goal is available)."""
    if report.cancelled:
        return Outcome.CANCELLED
    if report.all_completed:
        return Outcome.VERIFIED_SUCCESS
    if report.any_completed:
        return Outcome.PARTIAL_SUCCESS
    if report.failed:
        return Outcome.FAILED
    return Outcome.UNVERIFIED


__all__ = [
    "ArtifactCheck",
    "Outcome",
    "OutcomeVerifier",
    "VerificationResult",
    "outcome_from_report",
]
