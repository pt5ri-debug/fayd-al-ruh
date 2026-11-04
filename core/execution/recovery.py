"""Failure recovery (Phase 8, section 13).

Section 13 lists the recovery mechanisms -- retry when safe, alternative tool
selection, plan revision, dependency repair, cancellation, checkpoint
restoration, user escalation, and partial-result preservation -- and two hard
rules: **never automatically repeat an irreversible action**, and **never conceal
a failure or falsely report success**.

:class:`RecoveryManager` reads a :class:`core.execution.verification.VerificationResult`
and a :class:`core.execution.engine.ExecutionReport` and proposes the *least
drastic* recovery that is safe, in a fixed order. It only *proposes*: the caller
(the execution coordinator, or a human) decides. A proposal always names what it
would do and why, so a failure is visible rather than swallowed.

Dependency-free and Tk-free.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from core.execution.engine import ExecutionReport
from core.execution.verification import Outcome, VerificationResult
from core.planning.plan import TaskPlan
from core.planning.tasks import Task, TaskState


class RecoveryAction(str, Enum):
    """What a recovery decision proposes."""

    NONE = "none"
    RETRY = "retry"
    ALTERNATIVE_TOOL = "alternative_tool"
    REPLAN = "replan"
    REPAIR_DEPENDENCY = "repair_dependency"
    RESTORE_CHECKPOINT = "restore_checkpoint"
    CANCEL = "cancel"
    ESCALATE = "escalate"
    PRESERVE_PARTIAL = "preserve_partial"


@dataclass(frozen=True)
class RecoveryDecision:
    """One recovery proposal -- what to do, for which tasks, and why."""

    action: RecoveryAction
    tasks: tuple[str, ...] = ()
    reason: str = ""
    safe: bool = True

    @property
    def is_noop(self) -> bool:
        return self.action is RecoveryAction.NONE

    def describe(self) -> str:
        target = f" [{', '.join(self.tasks)}]" if self.tasks else ""
        return f"{self.action.value}{target}: {self.reason}"


@dataclass
class RecoveryManager:
    """Propose safe recovery for a failed or partial run (section 13)."""

    #: The most times a single task may be retried by recovery (bounded).
    max_recovery_retries: int = 2
    #: How many times recovery has retried each task (so the bound is real).
    retry_counts: dict[str, int] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.retry_counts is None:
            self.retry_counts = {}

    def propose(
        self,
        plan: TaskPlan,
        report: ExecutionReport,
        verification: VerificationResult,
    ) -> RecoveryDecision:
        """Propose the least-drastic safe recovery for a run.

        The order is deliberate, and each rule refuses when it would be unsafe:

        1. a **cancelled** run is not recovered -- it is reported as cancelled;
        2. a **verified success** needs no recovery;
        3. a **failed** retryable, reversible task is retried, within the bound;
        4. a task with **no permitted tool** is replanned (an alternative tool);
        5. a task **blocked by a failed dependency** is a dependency repair;
        6. an **irreversible** failure is escalated, never retried;
        7. anything else that partially succeeded preserves its partial result.
        """
        if verification.outcome is Outcome.CANCELLED:
            return RecoveryDecision(RecoveryAction.NONE, (), "the run was cancelled; nothing to recover.")
        if verification.outcome is Outcome.VERIFIED_SUCCESS:
            return RecoveryDecision(RecoveryAction.NONE, (), "the goal was verified; nothing to recover.")

        by_id = {task.task_id: task for task in plan.tasks}
        failed = list(verification.failed_tasks)
        unverified = list(verification.unverified_tasks)

        # 3. Retry a retryable, reversible failed task, within the bound.
        retryable: list[str] = []
        irreversible: list[str] = []
        for task_id in failed:
            task = by_id.get(task_id)
            if task is None:
                continue
            if not task.reversible:
                irreversible.append(task_id)
            elif task.retryable and self.retry_counts.get(task_id, 0) < self.max_recovery_retries:
                retryable.append(task_id)
        if retryable:
            return RecoveryDecision(
                RecoveryAction.RETRY, tuple(retryable),
                "the task is retryable and reversible, and its retry budget is not exhausted.",
            )

        # 4. A task with no tool is replanned with an alternative tool.
        tool_less = [t.task_id for t in plan.tasks if not t.tool]
        if tool_less:
            return RecoveryDecision(
                RecoveryAction.ALTERNATIVE_TOOL, tuple(tool_less),
                "no permitted tool was selected; a replan may find an alternative.",
            )

        # 5. A task blocked by a failed dependency is a dependency repair.
        blocked = [
            execution.task_id
            for execution in report.executions
            if execution.state is TaskState.BLOCKED
        ]
        if blocked:
            return RecoveryDecision(
                RecoveryAction.REPAIR_DEPENDENCY, tuple(blocked),
                "the task is blocked by a dependency that did not complete.",
            )

        # 6. An irreversible failure is escalated -- never retried automatically.
        if irreversible:
            return RecoveryDecision(
                RecoveryAction.ESCALATE, tuple(irreversible),
                "the action is irreversible; recovery must not repeat it automatically.",
                safe=False,
            )

        # 7. Preserve whatever partially succeeded.
        if verification.verified_tasks:
            return RecoveryDecision(
                RecoveryAction.PRESERVE_PARTIAL, verification.verified_tasks,
                "the run partially succeeded; preserve the verified tasks and report the rest.",
            )

        # Nothing safe remains: escalate to the user rather than guess.
        return RecoveryDecision(
            RecoveryAction.ESCALATE, tuple(failed + unverified),
            "no safe automatic recovery applies; the user must decide.",
            safe=False,
        )

    def record_retry(self, task_id: str) -> None:
        """Count one recovery retry of a task (so the bound is enforced)."""
        assert self.retry_counts is not None
        self.retry_counts[task_id] = self.retry_counts.get(task_id, 0) + 1

    def retries_used(self, task_id: str) -> int:
        assert self.retry_counts is not None
        return self.retry_counts.get(task_id, 0)


@dataclass(frozen=True)
class Checkpoint:
    """A point a run can return to (section 13's checkpoint restoration).

    A checkpoint records which tasks had completed and their outputs, so a
    restored run continues from the last good point rather than from the start.
    """

    plan_id: str
    completed: tuple[str, ...] = ()
    outputs: dict[str, str] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.outputs is None:
            object.__setattr__(self, "outputs", {})

    @classmethod
    def from_report(cls, report: ExecutionReport) -> Checkpoint:
        return cls(
            plan_id=report.plan_id,
            completed=tuple(e.task_id for e in report.completed),
            outputs={e.task_id: e.output for e in report.completed},
        )

    def restore(self, plan: TaskPlan) -> TaskPlan:
        """Return a copy of ``plan`` with the checkpoint's tasks marked complete."""
        restored: list[Task] = []
        for task in plan.tasks:
            if task.task_id in self.completed:
                restored.append(task.with_state(TaskState.COMPLETED, note="restored from checkpoint"))
            else:
                restored.append(task.with_state(TaskState.PENDING, note=""))
        return plan.with_tasks(tuple(restored))


__all__ = [
    "Checkpoint",
    "RecoveryAction",
    "RecoveryDecision",
    "RecoveryManager",
]
