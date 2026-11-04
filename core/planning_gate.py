"""The held-out gate for autonomous planning and task execution (Phase 8).

The task's own claim: a goal can be decomposed into tasks, each bound to a tool
the sandbox **permits**, executed under a permission and approval gate, verified
against the goal's own criteria, and -- when something fails -- recovered from
safely, all without ever widening the system's authority or reporting a tool's
success as the goal's success.

This module is the catalogue: a held-out set of hand-written judgements about what
the planning/execution layer must do, scored as one number and wired into
``evaluation`` as ``task_execution_accuracy``, so a regression in autonomy is
a **gate failure**, not a passing suite.

**What is measured, and how it is honest.** Every judgement runs the **real**
code -- the goal manager, the decomposition engine, the tool catalogue, the plan
validator, the execution engine, the permission manager, the outcome verifier,
and the recovery manager -- over synthetic files written to a temporary directory.
Nothing here touches production knowledge or memory, loads a model, or reaches the
network, so it runs on a fresh clone with no display and no weights.

Several judgements are **negative controls**, the honesty rules the phase exists
for:

* a **suggested** goal must not authorize execution;
* an action outside the sandbox allow-list must be **refused**, never selected;
* a **high-risk** action must need the user's approval, and be refused without it;
* an **irreversible** action must never be retried automatically;
* a **tool's success** must not be reported as the goal's success when the goal's
  own criteria are unmet;
* an **invalid** plan must never be executed;
* a failure must never be **hidden** (the run says it failed).

The module is dependency-free, Tk-free, and loads no model.
"""

from __future__ import annotations

import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

from core.execution.coordinator import ExecutionCoordinator
from core.execution.engine import ExecutionEngine, ExecutionReport, TaskExecution
from core.execution.permissions import ActionCategory, PermissionManager
from core.execution.recovery import RecoveryAction, RecoveryManager
from core.execution.verification import Outcome, OutcomeVerifier
from core.planning.coordinator import PlanningCoordinator
from core.planning.goals import GoalKind, GoalStatus
from core.planning.plan import PlanStatus, TaskPlan, validate_plan
from core.planning.tasks import Task, TaskState
from core.planning.tools import RiskCategory, ToolCatalogue, select_tool
from core.tools import build_sandbox, build_sandbox_tools


@dataclass(frozen=True)
class PlanningJudgement:
    """One held-out claim about the planning/execution layer."""

    name: str
    kind: str
    note: str = ""


#: The held-out catalogue. The ``plan-*`` kinds are the planning claims; the
#: ``exec-*`` kinds the execution claims; the ``honesty-*`` kinds the rules that
#: must never be relaxed.
PLANNING_JUDGEMENTS: tuple[PlanningJudgement, ...] = (
    # -- goal structure ------------------------------------------------------
    PlanningJudgement("goal-persisted", "goal", "a goal and its status survive a restart"),
    PlanningJudgement("goal-subgoal-needs-parent", "goal", "a subgoal names its parent"),
    PlanningJudgement("goal-version-immutable", "goal", "a status change is a new version"),
    # -- decomposition -------------------------------------------------------
    PlanningJudgement("decompose-read-write", "decompose", "read then write, in order"),
    PlanningJudgement("decompose-bound", "decompose", "the task count stays bounded"),
    PlanningJudgement("decompose-depth-refused", "decompose", "over-deep decomposition is refused"),
    PlanningJudgement("decompose-unrecognised-honest", "decompose", "no fabricated tool"),
    # -- tool selection ------------------------------------------------------
    PlanningJudgement("select-respects-allow-list", "tools", "a tool outside the list is refused"),
    PlanningJudgement("select-pinned", "tools", "a pinned tool wins"),
    PlanningJudgement("select-honest-no-match", "tools", "no match yields no tool"),
    # -- plan validation -----------------------------------------------------
    PlanningJudgement("plan-valid", "plan", "a real plan validates"),
    PlanningJudgement("plan-unknown-tool-invalid", "plan", "an unknown tool is an issue"),
    PlanningJudgement("plan-suggested-refused", "plan", "a suggested goal is refused"),
    PlanningJudgement("plan-no-tool-invalid", "plan", "a task with no tool is an issue"),
    # -- execution -----------------------------------------------------------
    PlanningJudgement("exec-read-write", "exec", "a read then a write really runs"),
    PlanningJudgement("exec-compute", "exec", "the calculator really computes"),
    PlanningJudgement("exec-dependency-order", "exec", "a dependent task waits"),
    PlanningJudgement("exec-output-chained", "exec", "a dependency's output is chained in"),
    PlanningJudgement("exec-invalid-not-run", "exec", "an invalid plan is never executed"),
    PlanningJudgement("exec-cancelled", "exec", "a cancelled run stops and says so"),
    # -- permission and approval ---------------------------------------------
    PlanningJudgement("perm-low-allowed", "perm", "a low-risk action needs no approval"),
    PlanningJudgement("perm-high-needs-approval", "perm", "a high-risk action needs approval"),
    PlanningJudgement("perm-high-refused-without", "perm", "a high-risk action is refused without it"),
    PlanningJudgement("perm-approval-unblocks", "perm", "a live approval lets it run"),
    PlanningJudgement("perm-irreversible-escalates", "perm", "an irreversible action escalates"),
    PlanningJudgement("perm-agent-ceiling", "perm", "an agent cannot exceed its ceiling"),
    # -- verification --------------------------------------------------------
    PlanningJudgement("verify-success", "verify", "a met goal is verified"),
    PlanningJudgement("verify-tool-not-goal", "verify", "a tool's success is not the goal's"),
    PlanningJudgement("verify-missing-artifact", "verify", "a missing artifact fails the goal"),
    PlanningJudgement("verify-cancelled", "verify", "a cancelled run is not a success"),
    # -- recovery ------------------------------------------------------------
    PlanningJudgement("recover-retry-reversible", "recover", "a reversible failure is retried"),
    PlanningJudgement("recover-no-retry-irreversible", "recover", "an irreversible failure is not retried"),
    PlanningJudgement("recover-verified-noop", "recover", "a verified success needs no recovery"),
    # -- audit and history ---------------------------------------------------
    PlanningJudgement("history-records-run", "history", "a run is recorded"),
    PlanningJudgement("history-redacts-secret", "history", "a credential never enters the log"),
    # -- honest failure ------------------------------------------------------
    PlanningJudgement("honesty-failure-reported", "honesty", "a failure is never hidden"),
    PlanningJudgement("honesty-no-fabricated-tool", "honesty", "an unrecognised goal fabricates nothing"),
)


@contextmanager
def _workspace() -> Iterator[Path]:
    """A throwaway sandbox root with a source file, cleaned up afterwards."""
    with tempfile.TemporaryDirectory() as folder:
        root = Path(folder)
        (root / "report.txt").write_text("porosity is 20 percent; permeability is 100 mD.")
        yield root


def _catalogue(root: Path, *, write_risk: RiskCategory = RiskCategory.MEDIUM,
               write_reversible: bool = True) -> ToolCatalogue:
    """The real sandbox tool specs, with the write tool's risk overridable."""
    sandbox = build_sandbox(root, read_only=False)
    catalogue = ToolCatalogue()
    for spec in build_sandbox_tools(sandbox):
        if spec.name == "write_file":
            catalogue.register(spec, risk=write_risk, reversible=write_reversible)
        else:
            catalogue.register(spec)
    return catalogue


def _engine(root: Path, catalogue: ToolCatalogue) -> ExecutionEngine:
    sandbox = build_sandbox(root, read_only=False)
    catalogue.available_tools = frozenset(sandbox.policy.allowed_tools)
    return ExecutionEngine(sandbox=sandbox, catalogue=catalogue)


def _judgement_holds(judgement: PlanningJudgement) -> bool:
    """Run one judgement's real check, returning whether it holds."""
    try:
        return _CHECKERS[judgement.kind](judgement)
    except Exception:
        return False


# --------------------------------------------------------------------- goal


def _check_goal(judgement: PlanningJudgement) -> bool:
    from core.planning.goals import Goal, GoalManager, VerificationCriteria

    if judgement.name == "goal-persisted":
        with tempfile.TemporaryDirectory() as data:
            manager = GoalManager()
            goal = manager.create("read a.txt", desired_outcome="content")
            manager.set_status(goal.goal_id, GoalStatus.COMPLETED)
            from core.planning.storage import GoalStore

            assert GoalStore(data).save(manager)
            reloaded = GoalStore(data).load()
            back = reloaded.get(goal.goal_id)
            return back is not None and back.status is GoalStatus.COMPLETED

    if judgement.name == "goal-subgoal-needs-parent":
        try:
            Goal(description="x", kind=GoalKind.SUBGOAL)
        except ValueError:
            # A subgoal with no parent is refused; with one it is accepted.
            return Goal(
                description="x", kind=GoalKind.SUBGOAL, parent_goal="goal-parent"
            ).parent_goal == "goal-parent"
        return False

    if judgement.name == "goal-version-immutable":
        goal = Goal(description="x")
        updated = goal.with_status(GoalStatus.ACTIVE)
        return goal.status is GoalStatus.PENDING and updated.status is GoalStatus.ACTIVE

    # a VerificationCriteria default is honest
    return VerificationCriteria().min_successful_tasks == 1


# -------------------------------------------------------------- decomposition


def _check_decompose(judgement: PlanningJudgement) -> bool:
    from core.planning.decomposition import DecompositionEngine, DecompositionError

    planner = PlanningCoordinator()

    if judgement.name == "decompose-read-write":
        goal = planner.goals.create("Read report.txt and write summary.md")
        tasks = planner.decomposition.decompose(goal)  # type: ignore[union-attr]
        tools = [t.tool for t in tasks]
        return "read_file" in tools and "write_file" in tools and tools.index("read_file") < tools.index("write_file")

    if judgement.name == "decompose-bound":
        engine = DecompositionEngine(planner.catalogue, max_tasks=3)
        goal = planner.goals.create("read a.txt write b.txt compute 1+1")
        return len(engine.decompose(goal)) <= 3

    if judgement.name == "decompose-depth-refused":
        engine = DecompositionEngine(planner.catalogue, max_depth=1)
        goal = planner.goals.create("read a.txt")
        try:
            engine.decompose(goal, depth=5)
        except DecompositionError:
            return True
        return False

    if judgement.name == "decompose-unrecognised-honest":
        goal = planner.goals.create("Achieve world peace")
        tasks = planner.decomposition.decompose(goal)  # type: ignore[union-attr]
        return all(not t.tool for t in tasks)

    return False


# --------------------------------------------------------------------- tools


def _check_tools(judgement: PlanningJudgement) -> bool:
    with _workspace() as root:
        catalogue = _catalogue(root)

        if judgement.name == "select-respects-allow-list":
            catalogue.available_tools = frozenset({"calculator"})
            selection = select_tool(
                Task(task_id="t", description="write a file"), catalogue
            )
            return selection.descriptor is None or selection.descriptor.tool_id == "calculator"

        if judgement.name == "select-pinned":
            selection = select_tool(
                Task(task_id="t", description="compute", tool="calculator"), catalogue
            )
            return selection.descriptor is not None and selection.descriptor.tool_id == "calculator"

        if judgement.name == "select-honest-no-match":
            selection = select_tool(
                Task(task_id="t", description="achieve world peace"), catalogue
            )
            return selection.descriptor is None

    return False


# ---------------------------------------------------------------------- plan


def _check_plan(judgement: PlanningJudgement) -> bool:
    with _workspace() as root:
        catalogue = _catalogue(root)

        if judgement.name == "plan-valid":
            goal = PlanningCoordinator(catalogue=catalogue).goals.create(
                "Read report.txt and write summary.md"
            )
            plan = PlanningCoordinator(catalogue=catalogue).plan(goal)
            return plan.status is PlanStatus.VALIDATED

        if judgement.name == "plan-unknown-tool-invalid":
            goal = PlanningCoordinator(catalogue=catalogue).goals.create("read report.txt")
            plan = TaskPlan(
                goal_id=goal.goal_id,
                tasks=(Task(task_id="t", description="x", tool="not_a_tool"),),
            )
            return not validate_plan(plan, goal, catalogue).ok

        if judgement.name == "plan-suggested-refused":
            from core.planning.goals import Goal

            goal = Goal(description="do something", kind=GoalKind.SUGGESTED)
            plan = TaskPlan(
                goal_id=goal.goal_id,
                tasks=(Task(task_id="t", description="compute 1+1", tool="calculator"),),
            )
            validation = validate_plan(plan, goal, catalogue)
            return not validation.ok and any("suggested" in i for i in validation.issues)

        if judgement.name == "plan-no-tool-invalid":
            goal = PlanningCoordinator(catalogue=catalogue).goals.create("read report.txt")
            plan = TaskPlan(
                goal_id=goal.goal_id,
                tasks=(Task(task_id="t", description="x", tool=""),),
            )
            return not validate_plan(plan, goal, catalogue).ok

    return False


# ----------------------------------------------------------------- execution


def _check_exec(judgement: PlanningJudgement) -> bool:
    if judgement.name == "exec-invalid-not-run":
        with _workspace() as root:
            planner = PlanningCoordinator(catalogue=_catalogue(root))
            coordinator = ExecutionCoordinator(planner=planner, root=root)
            goal = planner.goals.create("Achieve world peace")  # no tool -> invalid
            result = coordinator.run_goal(goal)
            return (
                result.plan.status is PlanStatus.REJECTED
                and not result.execution.executions
                and not result.success
            )

    if judgement.name == "exec-cancelled":
        from core.agents.runtime import CancellationToken

        with _workspace() as root:
            planner = PlanningCoordinator(catalogue=_catalogue(root))
            coordinator = ExecutionCoordinator(planner=planner, root=root)
            goal = planner.goals.create("Read report.txt")
            token = CancellationToken()
            token.cancel()
            result = coordinator.run_goal(goal, token=token)
            return result.verification.outcome is Outcome.CANCELLED

    with _workspace() as root:
        planner = PlanningCoordinator(catalogue=_catalogue(root))
        coordinator = ExecutionCoordinator(planner=planner, root=root)

        if judgement.name == "exec-read-write":
            goal = planner.goals.create("Read report.txt and write summary.md")
            result = coordinator.run_goal(goal)
            return result.success and (root / "summary.md").exists()

        if judgement.name == "exec-compute":
            goal = planner.goals.create("Compute 12 * 4")
            result = coordinator.run_goal(goal)
            return result.success and any(
                e.output.strip() == "48" for e in result.execution.executions
            )

        if judgement.name == "exec-dependency-order":
            goal = planner.goals.create("Read report.txt and write summary.md")
            result = coordinator.run_goal(goal)
            order = [e.task_id for e in result.execution.executions]
            # the read task (t1) must appear before the write task (t2)
            read_id = next(t.task_id for t in result.plan.tasks if t.tool == "read_file")
            write_id = next(t.task_id for t in result.plan.tasks if t.tool == "write_file")
            return order.index(read_id) < order.index(write_id)

        if judgement.name == "exec-output-chained":
            goal = planner.goals.create("Read report.txt and write summary.md")
            result = coordinator.run_goal(goal)
            if not (root / "summary.md").exists():
                return False
            written = (root / "summary.md").read_text()
            return "porosity" in written

    return False


# ---------------------------------------------------------------- permission


def _check_perm(judgement: PlanningJudgement) -> bool:
    with _workspace() as root:
        sandbox = build_sandbox(root, read_only=False)

        if judgement.name == "perm-low-allowed":
            catalogue = _catalogue(root)
            manager = PermissionManager()
            record = manager.check(catalogue.get("read_file"), actor="system")  # type: ignore[arg-type]
            return record.allowed and record.category is ActionCategory.LOW

        if judgement.name == "perm-high-needs-approval":
            catalogue = _catalogue(root, write_risk=RiskCategory.HIGH)
            manager = PermissionManager()
            descriptor = catalogue.get("write_file")
            return manager.needs_approval(descriptor)  # type: ignore[arg-type]

        if judgement.name == "perm-high-refused-without":
            catalogue = _catalogue(root, write_risk=RiskCategory.HIGH)
            manager = PermissionManager()
            record = manager.check(catalogue.get("write_file"), actor="system")  # type: ignore[arg-type]
            return not record.allowed

        if judgement.name == "perm-approval-unblocks":
            catalogue = _catalogue(root, write_risk=RiskCategory.HIGH)
            manager = PermissionManager(approvals=sandbox.approvals)
            descriptor = catalogue.get("write_file")
            request = manager.request_approval(descriptor, actor="system", justification="test")  # type: ignore[arg-type]
            request.approve("user", ttl_seconds=60)  # type: ignore[attr-defined]
            return manager.check(descriptor, actor="system").allowed  # type: ignore[arg-type]

        if judgement.name == "perm-irreversible-escalates":
            catalogue = _catalogue(root, write_risk=RiskCategory.HIGH, write_reversible=False)
            manager = PermissionManager()
            record = manager.check(catalogue.get("write_file"), actor="system")  # type: ignore[arg-type]
            return record.category is ActionCategory.HIGH and not record.allowed

        if judgement.name == "perm-agent-ceiling":
            from core.agents.capabilities import CapabilityGateway

            catalogue = _catalogue(root, write_risk=RiskCategory.HIGH)
            manager = PermissionManager(gateway=CapabilityGateway())
            record = manager.check(catalogue.get("write_file"), actor="natasha")  # type: ignore[arg-type]
            return not record.allowed

    return False


# ---------------------------------------------------------------- verification


def _check_verify(judgement: PlanningJudgement) -> bool:
    if judgement.name == "verify-cancelled":
        report = ExecutionReport("p", "g", status=PlanStatus.CANCELLED, cancelled=True)
        from core.planning.goals import Goal

        goal = Goal(description="x")
        result = OutcomeVerifier().verify(TaskPlan(goal_id="g"), goal, report)
        return result.outcome is Outcome.CANCELLED and not result.success

    if judgement.name == "verify-tool-not-goal":
        # A tool that returns fine, but the goal's artifact is missing -> not success.
        from core.planning.goals import Goal, VerificationCriteria

        goal = Goal(
            description="write out.md",
            verification=VerificationCriteria(artifacts=("out.md",)),
        )
        task = Task(task_id="t", description="write", tool="write_file",
                    success_criteria=("produced output",))
        plan = TaskPlan(goal_id=goal.goal_id, tasks=(task,))
        from core.execution.engine import TaskExecution

        report = ExecutionReport(
            plan.plan_id, goal.goal_id,
            executions=(TaskExecution("t", "write_file", TaskState.COMPLETED, True, output="ok"),),
        )
        result = OutcomeVerifier(root=Path("/nonexistent-fayd-root")).verify(plan, goal, report)
        return not result.goal_achieved and not result.success

    if judgement.name == "verify-missing-artifact":
        from core.planning.goals import Goal, VerificationCriteria

        goal = Goal(description="x", verification=VerificationCriteria(artifacts=("never.md",)))
        task = Task(task_id="t", description="write", tool="write_file")
        plan = TaskPlan(goal_id=goal.goal_id, tasks=(task,))
        from core.execution.engine import TaskExecution

        report = ExecutionReport(
            plan.plan_id, goal.goal_id,
            executions=(TaskExecution("t", "write_file", TaskState.COMPLETED, True, output="ok"),),
        )
        result = OutcomeVerifier(root=Path("/nonexistent-fayd-root")).verify(plan, goal, report)
        return "never.md" in result.missing_artifacts and not result.success

    if judgement.name == "verify-success":
        with _workspace() as root:
            planner = PlanningCoordinator(catalogue=_catalogue(root))
            coordinator = ExecutionCoordinator(planner=planner, root=root)
            goal = planner.goals.create("Read report.txt and write summary.md")
            run = coordinator.run_goal(goal)
            return run.success and run.verification.goal_achieved

    return False


# -------------------------------------------------------------------- recovery


def _check_recover(judgement: PlanningJudgement) -> bool:
    if judgement.name == "recover-verified-noop":
        report = ExecutionReport("p", "g")
        from core.execution.verification import VerificationResult

        verification = VerificationResult(Outcome.VERIFIED_SUCCESS, True)
        decision = RecoveryManager().propose(TaskPlan(goal_id="g"), report, verification)
        return decision.is_noop

    if judgement.name == "recover-retry-reversible":
        task = Task(task_id="t", description="x", tool="read_file", retryable=True, reversible=True)
        plan = TaskPlan(goal_id="g", tasks=(task,))
        report = ExecutionReport(
            plan.plan_id, "g",
            executions=(_failed_execution("t"),),
        )
        from core.execution.verification import VerificationResult

        verification = VerificationResult(Outcome.FAILED, False, failed_tasks=("t",))
        decision = RecoveryManager().propose(plan, report, verification)
        return decision.action is RecoveryAction.RETRY and decision.tasks == ("t",)

    if judgement.name == "recover-no-retry-irreversible":
        task = Task(task_id="t", description="x", tool="write_file", retryable=True, reversible=False)
        plan = TaskPlan(goal_id="g", tasks=(task,))
        report = ExecutionReport(plan.plan_id, "g", executions=(_failed_execution("t"),))
        from core.execution.verification import VerificationResult

        verification = VerificationResult(Outcome.FAILED, False, failed_tasks=("t",))
        decision = RecoveryManager().propose(plan, report, verification)
        return decision.action is RecoveryAction.ESCALATE and not decision.safe

    return False


def _failed_execution(task_id: str) -> TaskExecution:
    from core.execution.engine import TaskExecution

    return TaskExecution(task_id, "read_file", TaskState.FAILED, False, error="boom")


# -------------------------------------------------------------------- history


def _check_history(judgement: PlanningJudgement) -> bool:
    from core.execution.history import ExecutionHistory

    if judgement.name == "history-records-run":
        with _workspace() as root:
            planner = PlanningCoordinator(catalogue=_catalogue(root))
            coordinator = ExecutionCoordinator(planner=planner, root=root)
            goal = planner.goals.create("Read report.txt")
            coordinator.run_goal(goal)
            return coordinator.history.count(kind="verify") >= 1

    if judgement.name == "history-redacts-secret":
        history = ExecutionHistory()
        entry = history.log("test", "Authorization: Bearer abc123 token=secret99")
        return "abc123" not in entry.summary and "secret99" not in entry.summary

    return False


# -------------------------------------------------------------------- honesty


def _check_honesty(judgement: PlanningJudgement) -> bool:
    if judgement.name == "honesty-failure-reported":
        with _workspace() as root:
            planner = PlanningCoordinator(catalogue=_catalogue(root))
            coordinator = ExecutionCoordinator(planner=planner, root=root)
            # A read of a file that does not exist must fail, visibly.
            goal = planner.goals.create("Read missing.txt")
            result = coordinator.run_goal(goal)
            return not result.success and bool(result.verification.reasons)

    if judgement.name == "honesty-no-fabricated-tool":
        with _workspace() as root:
            planner = PlanningCoordinator(catalogue=_catalogue(root))
            goal = planner.goals.create("Achieve world peace")
            plan = planner.plan(goal)
            return all(not t.tool for t in plan.tasks)

    return False


_CHECKERS = {
    "goal": _check_goal,
    "decompose": _check_decompose,
    "tools": _check_tools,
    "plan": _check_plan,
    "exec": _check_exec,
    "perm": _check_perm,
    "verify": _check_verify,
    "recover": _check_recover,
    "history": _check_history,
    "honesty": _check_honesty,
}


def task_execution_accuracy(
    judgements: tuple[PlanningJudgement, ...] = PLANNING_JUDGEMENTS,
) -> float:
    """The share of held-out planning/execution judgements the layer satisfies.

    A judgement counts only when the real code does exactly what it must -- a
    goal is persisted, a plan is validated, an action outside the allow-list is
    refused, a high-risk action needs approval, a tool's success is not taken as
    the goal's, a failure is recovered from safely and never hidden. So a
    regression in autonomy fails the relevant judgement. Returns ``0.0`` for an
    empty catalogue.
    """
    catalogue = tuple(judgements)
    if not catalogue:
        return 0.0
    correct = sum(1 for judgement in catalogue if _judgement_holds(judgement))
    return correct / len(catalogue)


#: The roadmap's name for the gate; kept as an alias so existing callers and
#: tests that predate the rename keep working.
planning_execution_accuracy = task_execution_accuracy


__all__ = [
    "PLANNING_JUDGEMENTS",
    "PlanningJudgement",
    "planning_execution_accuracy",
    "task_execution_accuracy",
]
