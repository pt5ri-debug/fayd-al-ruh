"""The execution coordinator (Phase 8, sections 10, 14, 15).

This is the one place the planning and execution layers meet the rest of the
project. It runs a validated plan under the sandbox and the permission gates,
verifies the outcome, records the experience through the **Phase-3 learning
system**, and returns a single readable report. It also carries the two
integrations the phase requires:

* **Agent integration (section 14).** A plan's tasks may be assigned to a
  Phase-5 personality, whose capability ceiling is consulted through the same
  permission gate, so an agent never runs a tool its owner does not hold. Bounded
  delegation is done through the existing :class:`core.agents.collaboration.AgentCollaboration`.
* **Learning integration (section 15).** A finished run becomes an
  :class:`core.experience_learning.ExperienceRecord` and is recorded through the
  Phase-3 coordinator, so a lesson is learned the same way every other lesson is
  -- never by fine-tuning, never as a trusted fact.

It reuses rather than replaces: :class:`core.planning.coordinator.PlanningCoordinator`
plans, :class:`core.execution.engine.ExecutionEngine` executes,
:class:`core.execution.verification.OutcomeVerifier` verifies,
:class:`core.execution.recovery.RecoveryManager` proposes recovery, and the
project's own learning/memory/agent layers do the rest.

Dependency-free and Tk-free: a coordinator built with no arguments runs an
entirely local, offline plan.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from core.execution.engine import ExecutionEngine, ExecutionReport
from core.execution.history import ExecutionHistory
from core.execution.monitor import ExecutionMonitor
from core.execution.permissions import PermissionManager
from core.execution.recovery import (
    Checkpoint,
    RecoveryDecision,
    RecoveryManager,
)
from core.execution.verification import Outcome, OutcomeVerifier, VerificationResult
from core.planning.coordinator import PlanningCoordinator, default_planner
from core.planning.goals import Goal, GoalStatus
from core.planning.plan import PlanStatus, TaskPlan
from core.planning.tasks import Task, TaskState


@runtime_checkable
class LearningService(Protocol):
    """The Phase-3 learning seam: the one method the coordinator calls.

    A real :class:`core.learning_coordinator.LearningCoordinator` satisfies it;
    a test injects a stub, so the coordinator is exercised with no learning store.
    """

    def learn_experience(self, experience: Any) -> Any: ...


@dataclass
class ExecutionResult:
    """Everything one run produced -- the one thing a caller reads."""

    plan: TaskPlan
    goal: Goal
    execution: ExecutionReport
    verification: VerificationResult
    recovery: RecoveryDecision
    learning_event_id: str = ""
    notes: tuple[str, ...] = ()

    @property
    def success(self) -> bool:
        return self.verification.outcome is Outcome.VERIFIED_SUCCESS

    def describe(self) -> str:
        lines = [
            f"goal {self.goal.goal_id}: {self.goal.description}",
            f"plan {self.plan.plan_id} rev {self.plan.revision} — {self.execution.status.value}",
            self.verification.describe(),
        ]
        if not self.recovery.is_noop:
            lines.append("recovery: " + self.recovery.describe())
        if self.learning_event_id:
            lines.append(f"recorded as learning event {self.learning_event_id}")
        lines.extend(f"note: {note}" for note in self.notes)
        return "\n".join(lines)


@dataclass
class ExecutionCoordinator:
    """Plan, execute, verify, and learn -- one governed entry point."""

    planner: PlanningCoordinator = field(default_factory=default_planner)
    engine: ExecutionEngine | None = None
    verifier: OutcomeVerifier | None = None
    recovery: RecoveryManager = field(default_factory=RecoveryManager)
    monitor: ExecutionMonitor | None = None
    history: ExecutionHistory = field(default_factory=ExecutionHistory)
    #: Where the audit log is persisted (``None`` keeps it in memory only).
    data_root: str | None = None
    #: The Phase-3 learning coordinator (``None`` disables learning recording).
    learning: LearningService | None = None
    #: The owner whose learning scope the experience is recorded under.
    owner: str = ""
    #: The sandbox root artifacts are resolved against (for verification).
    root: Path | None = None

    def __post_init__(self) -> None:
        if self.engine is None:
            self.engine = self._default_engine()
        if self.verifier is None:
            self.verifier = OutcomeVerifier(root=self.root)
        if self.data_root and not self.history.data_root:
            self.history.data_root = self.data_root

    def _default_engine(self) -> ExecutionEngine:
        from core.tools import build_sandbox

        root = self.root or Path.cwd()
        sandbox = build_sandbox(root, read_only=False)
        catalogue = self.planner.catalogue
        catalogue.available_tools = frozenset(sandbox.policy.allowed_tools)
        permissions = PermissionManager(approvals=sandbox.approvals)
        return ExecutionEngine(
            sandbox=sandbox,
            catalogue=catalogue,
            permissions=permissions,
            actor=self.owner or "system",
        )

    # ---------------------------------------------------------------- running

    def run_goal(
        self,
        goal: Goal,
        *,
        token: object | None = None,
        max_parallel: int | None = None,
    ) -> ExecutionResult:
        """Plan, validate, execute, verify, and record one goal.

        An **invalid** plan is never executed: the run returns immediately with a
        report that says why. A plan that is valid runs under the sandbox and the
        permission gate, and its outcome is verified -- a tool's success is never
        taken as the goal's success.
        """
        plan = self.planner.plan(goal)
        self.history.log(
            "plan", f"planned {len(plan.tasks)} task(s) for {goal.goal_id}",
            plan_id=plan.plan_id, goal_id=goal.goal_id, ok=plan.status is PlanStatus.VALIDATED,
            detail=(plan.validation.describe() if plan.validation else ""),
        )
        if plan.status is not PlanStatus.VALIDATED:
            self.planner.goals.set_status(goal.goal_id, GoalStatus.BLOCKED)
            empty = ExecutionReport(plan.plan_id, goal.goal_id, status=PlanStatus.REJECTED)
            verification = VerificationResult(
                Outcome.FAILED, False,
                reasons=(plan.validation.describe() if plan.validation else "the plan is invalid.",),
            )
            return ExecutionResult(plan, goal, empty, verification, RecoveryDecision(
                self.recovery.propose(plan, empty, verification).action,
                (), "the plan was not executed because it is invalid.",
            ))

        return self.execute_plan(plan, goal, token=token, max_parallel=max_parallel)

    def execute_plan(
        self,
        plan: TaskPlan,
        goal: Goal,
        *,
        token: object | None = None,
        max_parallel: int | None = None,
    ) -> ExecutionResult:
        """Execute an already-validated plan, verify it, and record the experience."""
        assert self.engine is not None
        self.planner.goals.set_status(goal.goal_id, GoalStatus.ACTIVE)
        monitor = self.monitor or ExecutionMonitor()
        monitor.start(plan.tasks)

        def on_progress(completed: int, total: int, task: Task, execution: object) -> None:
            monitor.observe(task, execution)  # type: ignore[arg-type]
            state = getattr(execution, "state", TaskState.PENDING)
            state_value = getattr(state, "value", str(state))
            self.history.log(
                "task",
                f"{task.task_id} -> {state_value}",
                plan_id=plan.plan_id, goal_id=goal.goal_id, task_id=task.task_id,
                ok=bool(getattr(execution, "ok", False)),
                detail=str(getattr(execution, "error", "") or getattr(execution, "output", ""))[:200],
            )

        report = self.engine.execute(plan, token=token, on_progress=on_progress, max_parallel=max_parallel)
        assert self.verifier is not None
        verification = self.verifier.verify(plan, goal, report)

        # Update the goal's own status from the verified outcome.
        goal_status = {
            Outcome.VERIFIED_SUCCESS: GoalStatus.COMPLETED,
            Outcome.PARTIAL_SUCCESS: GoalStatus.PARTIAL,
            Outcome.FAILED: GoalStatus.FAILED,
            Outcome.CANCELLED: GoalStatus.CANCELLED,
            Outcome.UNVERIFIED: GoalStatus.BLOCKED,
        }[verification.outcome]
        self.planner.goals.set_status(goal.goal_id, goal_status)

        recovery = self.recovery.propose(plan, report, verification)
        self.history.log(
            "verify", verification.outcome.value,
            plan_id=plan.plan_id, goal_id=goal.goal_id, ok=verification.success,
            detail=verification.describe(),
        )
        if not recovery.is_noop:
            self.history.log(
                "recovery", recovery.describe(), plan_id=plan.plan_id, goal_id=goal.goal_id,
                ok=recovery.safe,
            )

        event_id = self._record_experience(plan, goal, report, verification)
        # Persist the goal's final status and the plan's *final* task states.
        final_plan = self._final_plan(plan, report)
        self.planner.persist(goal, final_plan)
        self.history.save()
        return ExecutionResult(final_plan, goal, report, verification, recovery, event_id)

    @staticmethod
    def _final_plan(plan: TaskPlan, report: ExecutionReport) -> TaskPlan:
        """The plan with every task's executed state recorded (for persistence)."""
        by_id = {execution.task_id: execution for execution in report.executions}
        tasks = tuple(
            task.with_state(by_id[task.task_id].state, note=by_id[task.task_id].note)
            if task.task_id in by_id
            else task
            for task in plan.tasks
        )
        return plan.with_tasks(tasks).with_status(report.status)

    # -------------------------------------------------------------- learning

    def _record_experience(
        self,
        plan: TaskPlan,
        goal: Goal,
        report: ExecutionReport,
        verification: VerificationResult,
    ) -> str:
        """Record the run through the Phase-3 learning system (never fine-tuning).

        The verdict is the plan evaluator's own reading of the verification (one
        place decides whether the goal was achieved), and a coordinator with no
        learning service, or a run the learning layer refuses, records nothing --
        the failure is swallowed *and reported in the history*, never hidden.
        """
        if self.learning is None:
            return ""
        try:
            from core.plan_evaluator import PlanEvaluator

            evaluator = PlanEvaluator()
            evaluation = evaluator.evaluate_outcome(goal, verification=verification)
            event_id = evaluator.record_experience(
                evaluation,
                goal,
                self.learning,
                approach="; ".join(task.description for task in plan.tasks),
                scope=self.owner,
            )
            if event_id:
                return event_id
            self.history.log(
                "learning", "the learning layer recorded no event",
                plan_id=plan.plan_id, goal_id=goal.goal_id, ok=False,
            )
            return ""
        except Exception as error:  # learning must never break a run
            self.history.log(
                "learning", f"could not record the experience: {type(error).__name__}",
                plan_id=plan.plan_id, goal_id=goal.goal_id, ok=False,
            )
            return ""

    # ------------------------------------------------------- checkpointing

    def checkpoint(self, report: ExecutionReport) -> Checkpoint:
        """A checkpoint a later run can resume from (section 13)."""
        return Checkpoint.from_report(report)

    def resume(
        self,
        plan: TaskPlan,
        goal: Goal,
        checkpoint: Checkpoint,
        *,
        token: object | None = None,
    ) -> ExecutionResult:
        """Resume a plan from a checkpoint (its completed tasks are kept)."""
        restored = checkpoint.restore(plan)
        return self.execute_plan(restored, goal, token=token)

    # -------------------------------------------------------------- reporting

    def status(self) -> str:
        return "\n".join(
            [
                self.planner.status(),
                self.history.describe(),
            ]
        )


def default_executor(
    *,
    root: Path | None = None,
    owner: str = "",
    learning: LearningService | None = None,
    data_root: str | None = None,
) -> ExecutionCoordinator:
    """Build the coordinator the CLI and the gate use."""
    planner = default_planner()
    return ExecutionCoordinator(
        planner=planner,
        root=root,
        owner=owner,
        learning=learning,
        history=ExecutionHistory(data_root=data_root),
    )


__all__ = ["ExecutionCoordinator", "ExecutionResult", "default_executor"]
