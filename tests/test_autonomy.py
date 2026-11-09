"""Phase 8: autonomous planning and task execution.

Covers the structured goal and its durability, the decomposition engine, the tool
catalogue and its honest selection, the plan validator, the execution engine (its
dependency scheduling, argument chaining, cancellation, and bounded retries), the
permission and approval manager, outcome verification, recovery, the audit log
with its redaction, the execution coordinator's learning integration, the CLI, and
the held-out gate.

Everything here runs over synthetic files in a temporary directory with an
injected runner where a real tool is unnecessary, so no model, display, or network
is needed -- and nothing touches production knowledge or memory.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest
from core.agents.runtime import CancellationToken
from core.execution.coordinator import ExecutionCoordinator, default_executor
from core.execution.engine import ExecutionEngine, ExecutionReport, TaskExecution
from core.execution.history import ExecutionHistory, redact
from core.execution.permissions import ActionCategory, PermissionManager
from core.execution.recovery import Checkpoint, RecoveryAction, RecoveryManager
from core.execution.verification import Outcome, OutcomeVerifier, VerificationResult
from core.planning.coordinator import PlanningCoordinator
from core.planning.decomposition import DecompositionEngine, DecompositionError
from core.planning.dependencies import DependencyError, DependencyGraph
from core.planning.goals import Goal, GoalKind, GoalManager, GoalStatus, VerificationCriteria
from core.planning.plan import PlanStatus, TaskPlan, validate_plan
from core.planning.storage import GoalStore, PlanStore
from core.planning.tasks import Task, TaskState
from core.planning.tools import RiskCategory, ToolCatalogue, select_tool
from core.planning_gate import (
    PLANNING_JUDGEMENTS,
    planning_execution_accuracy,
    task_execution_accuracy,
)
from core.tools import ToolResult, build_sandbox, build_sandbox_tools


@pytest.fixture()
def root(tmp_path: Path) -> Path:
    (tmp_path / "report.txt").write_text("porosity is 20 percent; permeability is 100 mD.")
    return tmp_path


def _catalogue(root: Path, **kwargs: object) -> ToolCatalogue:
    sandbox = build_sandbox(root, read_only=False)
    catalogue = ToolCatalogue()
    for spec in build_sandbox_tools(sandbox):
        if spec.name == "write_file" and kwargs:
            catalogue.register(spec, **kwargs)  # type: ignore[arg-type]
        else:
            catalogue.register(spec)
    return catalogue


def _engine(root: Path, catalogue: ToolCatalogue, **kwargs: object) -> ExecutionEngine:
    sandbox = build_sandbox(root, read_only=False)
    catalogue.available_tools = frozenset(sandbox.policy.allowed_tools)
    return ExecutionEngine(sandbox=sandbox, catalogue=catalogue, **kwargs)  # type: ignore[arg-type]


# ------------------------------------------------------------------- goals


def test_goal_requires_description():
    with pytest.raises(ValueError):
        Goal(description="  ")


def test_goal_priority_bounded():
    with pytest.raises(ValueError):
        Goal(description="x", priority=1.5)


def test_subgoal_needs_parent():
    with pytest.raises(ValueError):
        Goal(description="x", kind=GoalKind.SUBGOAL)


def test_goal_status_is_a_new_version():
    goal = Goal(description="x")
    updated = goal.with_status(GoalStatus.ACTIVE)
    assert goal.status is GoalStatus.PENDING
    assert updated.status is GoalStatus.ACTIVE


def test_goal_terminal_states():
    assert GoalStatus.COMPLETED.terminal
    assert GoalStatus.PARTIAL.terminal
    assert GoalStatus.FAILED.terminal
    assert GoalStatus.CANCELLED.terminal
    assert not GoalStatus.PENDING.terminal


def test_goal_manager_authorizes_only_user_goals():
    manager = GoalManager()
    user = manager.create("do it")
    subgoal = manager.create("part of it", kind=GoalKind.SUBGOAL, parent_goal=user.goal_id)
    suggested = manager.create("maybe", kind=GoalKind.SUGGESTED)
    assert manager.authorizes(user.goal_id)
    assert manager.authorizes(subgoal.goal_id)
    assert not manager.authorizes(suggested.goal_id)


def test_goal_manager_open_goals_excludes_terminal():
    manager = GoalManager()
    a = manager.create("a")
    manager.set_status(a.goal_id, GoalStatus.COMPLETED)
    manager.create("b")
    assert [g.description for g in manager.open_goals()] == ["b"]


def test_goal_manager_trim_never_drops_an_open_goal():
    manager = GoalManager(max_goals=2)
    first = manager.create("first")
    manager.set_status(first.goal_id, GoalStatus.COMPLETED)
    manager.create("second")
    manager.create("third")
    assert manager.get(first.goal_id) is None  # the oldest terminal goal was dropped
    assert len(manager.open_goals()) == 2


def test_goal_round_trips_through_primitive():
    goal = Goal(
        description="x", desired_outcome="y", priority=0.7,
        constraints=("a",), required_permissions=("read",),
        verification=VerificationCriteria(artifacts=("out.md",), success_criteria=("ok",)),
    )
    back = Goal.from_primitive(goal.to_primitive())
    assert back.description == goal.description
    assert back.desired_outcome == "y"
    assert back.priority == 0.7
    assert back.verification.artifacts == ("out.md",)
    assert back.verification.success_criteria == ("ok",)


# ------------------------------------------------------------ goal storage


def test_goal_store_round_trip(tmp_path: Path):
    manager = GoalManager()
    goal = manager.create("read a.txt", desired_outcome="content")
    manager.set_status(goal.goal_id, GoalStatus.COMPLETED)
    store = GoalStore(str(tmp_path))
    assert store.save(manager)
    reloaded = store.load()
    back = reloaded.get(goal.goal_id)
    assert back is not None and back.status is GoalStatus.COMPLETED


def test_goal_store_tolerates_a_corrupt_document(tmp_path: Path):
    from storage.json_store import JsonStore

    JsonStore(str(tmp_path)).save("planning", "goals", [{"bad": "record"}, 42])
    assert GoalStore(str(tmp_path)).load().all_goals() == ()


def test_goal_store_with_no_root_is_a_noop():
    assert GoalStore(None).save(GoalManager()) is False
    assert GoalStore(None).load().all_goals() == ()


def test_plan_store_round_trip(tmp_path: Path):
    goal = Goal(description="x")
    task = Task(task_id="t1", description="read", tool="read_file", arguments={"relative": "a.txt"})
    plan = TaskPlan(goal_id=goal.goal_id, tasks=(task,), notes=("a note",))
    store = PlanStore(str(tmp_path))
    store.remember(plan)
    assert store.save()
    fresh = PlanStore(str(tmp_path))
    assert fresh.load() == 1
    back = fresh.get(plan.plan_id)
    assert back is not None
    assert back.tasks[0].tool == "read_file"
    assert back.tasks[0].arguments == {"relative": "a.txt"}
    assert back.notes == ("a note",)


def test_symbolic_goal_store_round_trip(tmp_path: Path):
    from core.goal import Goal as SymbolicGoal
    from core.goal import GoalStore as SymbolicGoalStore

    store = SymbolicGoalStore(str(tmp_path))
    store.add(SymbolicGoal("finish", frozenset({"done"})))
    assert store.save()
    fresh = SymbolicGoalStore(str(tmp_path))
    assert fresh.load() == 1
    assert fresh.goals[0].description == "finish"


# --------------------------------------------------------- decomposition


def test_decomposition_read_write_in_order(root: Path):
    planner = PlanningCoordinator(catalogue=_catalogue(root))
    goal = planner.goals.create("Read report.txt and write summary.md")
    tasks = planner.decomposition.decompose(goal)  # type: ignore[union-attr]
    tools = [t.tool for t in tasks]
    assert "read_file" in tools and "write_file" in tools
    assert tools.index("read_file") < tools.index("write_file")


def test_decomposition_write_depends_on_read(root: Path):
    planner = PlanningCoordinator(catalogue=_catalogue(root))
    goal = planner.goals.create("Read report.txt and write summary.md")
    tasks = planner.decomposition.decompose(goal)  # type: ignore[union-attr]
    read_id = next(t.task_id for t in tasks if t.tool == "read_file")
    write = next(t for t in tasks if t.tool == "write_file")
    assert read_id in write.dependencies


def test_decomposition_is_bounded(root: Path):
    engine = DecompositionEngine(_catalogue(root), max_tasks=2)
    goal = PlanningCoordinator(catalogue=_catalogue(root)).goals.create(
        "read a.txt write b.txt compute 1+1"
    )
    assert len(engine.decompose(goal)) <= 2


def test_decomposition_a_listing_goal_uses_list_dir_not_read_file(root: Path):
    """A goal that asks to *list* is a directory listing, not a file read.

    "list the files" names no file, so a read task would be missing its input and
    could never satisfy the goal; it must select ``list_dir`` and run to success.
    """
    planner = PlanningCoordinator(catalogue=_catalogue(root))
    goal = planner.goals.create("List the files in the workspace")
    tasks = planner.decomposition.decompose(goal)  # type: ignore[union-attr]
    listing = next(t for t in tasks if t.tool == "list_dir")
    assert listing.arguments.get("relative") == "."
    assert all(t.tool != "read_file" for t in tasks)
    # The goal actually runs to success through the real coordinator, rather than
    # failing on a missing file input the way a read task would.
    result = default_executor(root=root).run_goal(goal)
    assert result.success, result.describe()


def test_decomposition_depth_over_bound_is_refused(root: Path):
    engine = DecompositionEngine(_catalogue(root), max_depth=1)
    goal = PlanningCoordinator(catalogue=_catalogue(root)).goals.create("read a.txt")
    with pytest.raises(DecompositionError):
        engine.decompose(goal, depth=5)


def test_decomposition_unrecognised_goal_fabricates_no_tool(root: Path):
    planner = PlanningCoordinator(catalogue=_catalogue(root))
    goal = planner.goals.create("Achieve world peace")
    tasks = planner.decomposition.decompose(goal)  # type: ignore[union-attr]
    assert tasks and all(not t.tool for t in tasks)


def test_decomposition_notes_record_what_it_cannot_do(root: Path):
    planner = PlanningCoordinator(catalogue=_catalogue(root))
    goal = planner.goals.create("Summarise and compare report.txt and write summary.md")
    notes = planner.decomposition.notes_for(goal)  # type: ignore[union-attr]
    assert any("summarise" in note for note in notes)
    assert any("compare" in note for note in notes)


def test_expression_hint_accepts_only_arithmetic(root: Path):
    engine = DecompositionEngine(_catalogue(root))
    assert engine._expression_hint("Compute 12 * 4") == "12 * 4"
    assert engine._expression_hint("compute the average") == ""


# ------------------------------------------------------------- dependencies


def test_dependency_graph_orders_and_detects_cycles():
    a = Task(task_id="a", description="a")
    b = Task(task_id="b", description="b", dependencies=("a",))
    graph = DependencyGraph.from_tasks((b, a))
    assert [t.task_id for t in graph.order()] == ["a", "b"]


def test_dependency_graph_refuses_a_missing_dependency():
    a = Task(task_id="a", description="a", dependencies=("ghost",))
    with pytest.raises(DependencyError):
        DependencyGraph.from_tasks((a,))


def test_dependency_graph_refuses_a_cycle():
    a = Task(task_id="a", description="a", dependencies=("b",))
    b = Task(task_id="b", description="b", dependencies=("a",))
    with pytest.raises(DependencyError):
        DependencyGraph.from_tasks((a, b)).order()


# ------------------------------------------------------------------ tools


def test_select_pinned_tool(root: Path):
    catalogue = _catalogue(root)
    selection = select_tool(Task(task_id="t", description="compute", tool="calculator"), catalogue)
    assert selection.descriptor is not None
    assert selection.descriptor.tool_id == "calculator"


def test_select_unknown_pinned_tool_yields_none(root: Path):
    catalogue = _catalogue(root)
    selection = select_tool(Task(task_id="t", description="x", tool="ghost"), catalogue)
    assert selection.descriptor is None
    assert "unknown" in selection.reason


def test_select_respects_the_allow_list(root: Path):
    catalogue = _catalogue(root)
    catalogue.available_tools = frozenset({"calculator"})
    selection = select_tool(Task(task_id="t", description="write a file"), catalogue)
    assert selection.descriptor is None or selection.descriptor.tool_id == "calculator"


def test_select_honest_when_nothing_matches(root: Path):
    catalogue = _catalogue(root)
    selection = select_tool(Task(task_id="t", description="achieve world peace"), catalogue)
    assert selection.descriptor is None


def test_select_rejects_a_tool_over_the_risk_ceiling(root: Path):
    catalogue = _catalogue(root, risk=RiskCategory.HIGH)
    # A high-risk write must not be chosen when the ceiling is low.
    selection = select_tool(
        Task(task_id="t", description="write the file", tool="write_file"),
        catalogue,
        max_risk=RiskCategory.LOW,
    )
    assert selection.descriptor is None
    assert any("risk" in reason for _, reason in selection.rejected)


def test_catalogue_conservative_default_for_an_undeclared_tool(root: Path):
    catalogue = _catalogue(root)
    assert catalogue.get("write_file") is not None
    assert catalogue.available("read_file")


# ------------------------------------------------------------------- plan


def test_plan_validates_a_real_goal(root: Path):
    planner = PlanningCoordinator(catalogue=_catalogue(root))
    goal = planner.goals.create("Read report.txt and write summary.md")
    plan = planner.plan(goal)
    assert plan.status is PlanStatus.VALIDATED


def test_plan_unknown_tool_is_an_issue(root: Path):
    catalogue = _catalogue(root)
    goal = Goal(description="x")
    plan = TaskPlan(goal_id=goal.goal_id, tasks=(Task(task_id="t", description="x", tool="ghost"),))
    validation = validate_plan(plan, goal, catalogue)
    assert not validation.ok
    assert any("unknown tool" in issue for issue in validation.issues)


def test_plan_suggested_goal_is_refused(root: Path):
    catalogue = _catalogue(root)
    goal = Goal(description="x", kind=GoalKind.SUGGESTED)
    plan = TaskPlan(
        goal_id=goal.goal_id,
        tasks=(Task(task_id="t", description="compute 1+1", tool="calculator"),),
    )
    validation = validate_plan(plan, goal, catalogue)
    assert not validation.ok
    assert any("suggested" in issue for issue in validation.issues)


def test_plan_task_without_a_tool_is_an_issue(root: Path):
    catalogue = _catalogue(root)
    goal = Goal(description="x")
    plan = TaskPlan(goal_id=goal.goal_id, tasks=(Task(task_id="t", description="x", tool=""),))
    assert not validate_plan(plan, goal, catalogue).ok


def test_plan_empty_is_an_issue(root: Path):
    catalogue = _catalogue(root)
    goal = Goal(description="x")
    assert not validate_plan(TaskPlan(goal_id=goal.goal_id), goal, catalogue).ok


def test_plan_rejects_duplicate_task_ids(root: Path):
    task = Task(task_id="t", description="x")
    with pytest.raises(ValueError):
        TaskPlan(goal_id="g", tasks=(task, task))


# -------------------------------------------------------------- execution


def test_engine_runs_a_read_and_a_write(root: Path):
    planner = PlanningCoordinator(catalogue=_catalogue(root))
    coordinator = ExecutionCoordinator(planner=planner, root=root)
    goal = planner.goals.create("Read report.txt and write summary.md")
    result = coordinator.run_goal(goal)
    assert result.success
    assert (root / "summary.md").exists()
    assert "porosity" in (root / "summary.md").read_text()


def test_engine_chains_a_dependency_output(root: Path):
    engine = _engine(root, _catalogue(root))
    read = Task(
        task_id="t1", description="read", tool="read_file", arguments={"relative": "report.txt"}
    )
    write = Task(
        task_id="t2", description="write", tool="write_file",
        arguments={"relative": "out.md", "content": "{t1}"}, dependencies=("t1",),
    )
    plan = TaskPlan(goal_id="g", tasks=(read, write))
    report = engine.execute(plan)
    assert report.all_completed
    assert "porosity" in (root / "out.md").read_text()


def test_engine_does_not_run_an_invalid_plan(root: Path):
    planner = PlanningCoordinator(catalogue=_catalogue(root))
    coordinator = ExecutionCoordinator(planner=planner, root=root)
    goal = planner.goals.create("Achieve world peace")
    result = coordinator.run_goal(goal)
    assert result.plan.status is PlanStatus.REJECTED
    assert result.execution.executions == ()
    assert not result.success


def test_engine_cancellation(root: Path):
    planner = PlanningCoordinator(catalogue=_catalogue(root))
    coordinator = ExecutionCoordinator(planner=planner, root=root)
    goal = planner.goals.create("Read report.txt")
    token = CancellationToken()
    token.cancel()
    result = coordinator.run_goal(goal, token=token)
    assert result.verification.outcome is Outcome.CANCELLED


def test_engine_validates_arguments_before_running(root: Path):
    engine = _engine(root, _catalogue(root))
    task = Task(task_id="t1", description="read", tool="read_file", arguments={})
    report = engine.execute(TaskPlan(goal_id="g", tasks=(task,)))
    assert report.failed
    assert "missing tool inputs" in report.executions[0].error


def test_engine_retries_a_retryable_task(root: Path):
    calls = {"n": 0}

    def runner(task, descriptor, arguments):
        calls["n"] += 1
        return ToolResult(task.tool, calls["n"] >= 2, output="ok" if calls["n"] >= 2 else "", error="flaky")

    engine = _engine(root, _catalogue(root), runner=runner, max_retries=2, retry_backoff=0.0)
    task = Task(task_id="t1", description="read", tool="read_file", arguments={"relative": "report.txt"})
    report = engine.execute(TaskPlan(goal_id="g", tasks=(task,)))
    assert report.all_completed
    assert report.executions[0].attempts == 2


def test_engine_does_not_retry_an_irreversible_task(root: Path):
    calls = {"n": 0}

    def runner(task, descriptor, arguments):
        calls["n"] += 1
        return ToolResult(task.tool, False, error="nope")

    engine = _engine(root, _catalogue(root), runner=runner, max_retries=3, retry_backoff=0.0)
    task = Task(
        task_id="t1", description="write", tool="write_file", reversible=False,
        arguments={"relative": "x.md", "content": "x"},
    )
    report = engine.execute(TaskPlan(goal_id="g", tasks=(task,)))
    assert report.failed
    assert calls["n"] == 1


def test_engine_blocks_a_task_with_a_failed_dependency(root: Path):
    def runner(task, descriptor, arguments):
        if task.task_id == "t1":
            return ToolResult(task.tool, False, error="boom")
        return ToolResult(task.tool, True, output="ok")

    engine = _engine(root, _catalogue(root), runner=runner)
    a = Task(task_id="t1", description="read", tool="read_file", arguments={"relative": "report.txt"})
    b = Task(task_id="t2", description="write", tool="write_file", dependencies=("t1",),
             arguments={"relative": "x.md", "content": "x"})
    report = engine.execute(TaskPlan(goal_id="g", tasks=(a, b)))
    by_id = {e.task_id: e for e in report.executions}
    assert by_id["t1"].state is TaskState.FAILED
    assert by_id["t2"].state is TaskState.BLOCKED


# ------------------------------------------------------------- permission


def test_low_risk_needs_no_approval(root: Path):
    catalogue = _catalogue(root)
    manager = PermissionManager()
    record = manager.check(catalogue.get("read_file"))  # type: ignore[arg-type]
    assert record.allowed
    assert record.category is ActionCategory.LOW


def test_high_risk_needs_approval(root: Path):
    catalogue = _catalogue(root, risk=RiskCategory.HIGH)
    manager = PermissionManager()
    descriptor = catalogue.get("write_file")
    assert manager.needs_approval(descriptor)  # type: ignore[arg-type]
    assert not manager.check(descriptor).allowed  # type: ignore[arg-type]


def test_irreversible_escalates_to_high(root: Path):
    catalogue = _catalogue(root, reversible=False)
    manager = PermissionManager()
    record = manager.check(catalogue.get("write_file"))  # type: ignore[arg-type]
    assert record.category is ActionCategory.HIGH


def test_an_approval_unblocks_a_high_risk_action(root: Path):
    sandbox = build_sandbox(root, read_only=False)
    catalogue = _catalogue(root, risk=RiskCategory.HIGH)
    manager = PermissionManager(approvals=sandbox.approvals)
    descriptor = catalogue.get("write_file")
    request = manager.request_approval(descriptor, actor="system", justification="test")  # type: ignore[arg-type]
    request.approve("user", ttl_seconds=60)  # type: ignore[attr-defined]
    assert manager.check(descriptor).allowed  # type: ignore[arg-type]


def test_an_agent_cannot_exceed_its_ceiling(root: Path):
    from core.agents.capabilities import CapabilityGateway

    catalogue = _catalogue(root, risk=RiskCategory.HIGH)
    manager = PermissionManager(gateway=CapabilityGateway())
    record = manager.check(catalogue.get("write_file"), actor="natasha")  # type: ignore[arg-type]
    assert not record.allowed


def test_permission_decisions_are_recorded(root: Path):
    catalogue = _catalogue(root)
    manager = PermissionManager()
    manager.check(catalogue.get("read_file"))  # type: ignore[arg-type]
    assert manager.recent()


# ------------------------------------------------------------ verification


def test_verifier_success(root: Path):
    planner = PlanningCoordinator(catalogue=_catalogue(root))
    coordinator = ExecutionCoordinator(planner=planner, root=root)
    goal = planner.goals.create("Read report.txt and write summary.md")
    result = coordinator.run_goal(goal)
    assert result.verification.outcome is Outcome.VERIFIED_SUCCESS
    assert result.verification.goal_achieved


def test_verifier_tool_success_is_not_goal_success(root: Path):
    goal = Goal(description="x", verification=VerificationCriteria(artifacts=("never.md",)))
    task = Task(task_id="t", description="write", tool="write_file")
    plan = TaskPlan(goal_id=goal.goal_id, tasks=(task,))
    report = ExecutionReport(
        plan.plan_id, goal.goal_id,
        executions=(TaskExecution("t", "write_file", TaskState.COMPLETED, True, output="ok"),),
    )
    result = OutcomeVerifier(root=root).verify(plan, goal, report)
    assert not result.goal_achieved
    assert not result.success


def test_verifier_reports_a_missing_artifact(root: Path):
    goal = Goal(description="x", verification=VerificationCriteria(artifacts=("never.md",)))
    task = Task(task_id="t", description="write", tool="write_file")
    plan = TaskPlan(goal_id=goal.goal_id, tasks=(task,))
    report = ExecutionReport(
        plan.plan_id, goal.goal_id,
        executions=(TaskExecution("t", "write_file", TaskState.COMPLETED, True, output="ok"),),
    )
    result = OutcomeVerifier(root=root).verify(plan, goal, report)
    assert "never.md" in result.missing_artifacts


def test_verifier_cancelled_run_is_not_success():
    goal = Goal(description="x")
    report = ExecutionReport("p", "g", status=PlanStatus.CANCELLED, cancelled=True)
    result = OutcomeVerifier().verify(TaskPlan(goal_id="g"), goal, report)
    assert result.outcome is Outcome.CANCELLED
    assert not result.success


def test_verifier_names_a_failed_task(root: Path):
    goal = Goal(description="x")
    task = Task(task_id="t", description="read", tool="read_file")
    plan = TaskPlan(goal_id=goal.goal_id, tasks=(task,))
    report = ExecutionReport(
        plan.plan_id, goal.goal_id,
        executions=(TaskExecution("t", "read_file", TaskState.FAILED, False, error="boom"),),
    )
    result = OutcomeVerifier(root=root).verify(plan, goal, report)
    assert any("boom" in reason for reason in result.reasons)


# --------------------------------------------------------------- recovery


def test_recovery_noop_on_verified_success():
    plan = TaskPlan(goal_id="g")
    decision = RecoveryManager().propose(plan, ExecutionReport("p", "g"), VerificationResult(Outcome.VERIFIED_SUCCESS, True))
    assert decision.is_noop


def test_recovery_noop_on_cancelled():
    plan = TaskPlan(goal_id="g")
    decision = RecoveryManager().propose(plan, ExecutionReport("p", "g", cancelled=True), VerificationResult(Outcome.CANCELLED, False))
    assert decision.is_noop


def test_recovery_retries_a_reversible_failure():
    task = Task(task_id="t", description="x", tool="read_file", retryable=True, reversible=True)
    plan = TaskPlan(goal_id="g", tasks=(task,))
    report = ExecutionReport(plan.plan_id, "g", executions=(TaskExecution("t", "read_file", TaskState.FAILED, False, error="boom"),))
    verification = VerificationResult(Outcome.FAILED, False, failed_tasks=("t",))
    decision = RecoveryManager().propose(plan, report, verification)
    assert decision.action is RecoveryAction.RETRY
    assert decision.tasks == ("t",)


def test_recovery_never_retries_an_irreversible_failure():
    task = Task(task_id="t", description="x", tool="write_file", retryable=True, reversible=False)
    plan = TaskPlan(goal_id="g", tasks=(task,))
    report = ExecutionReport(plan.plan_id, "g", executions=(TaskExecution("t", "write_file", TaskState.FAILED, False, error="boom"),))
    verification = VerificationResult(Outcome.FAILED, False, failed_tasks=("t",))
    decision = RecoveryManager().propose(plan, report, verification)
    assert decision.action is RecoveryAction.ESCALATE
    assert not decision.safe


def test_recovery_retry_bound_is_real():
    manager = RecoveryManager(max_recovery_retries=1)
    task = Task(task_id="t", description="x", tool="read_file")
    plan = TaskPlan(goal_id="g", tasks=(task,))
    report = ExecutionReport(plan.plan_id, "g", executions=(TaskExecution("t", "read_file", TaskState.FAILED, False),))
    verification = VerificationResult(Outcome.FAILED, False, failed_tasks=("t",))
    assert manager.propose(plan, report, verification).action is RecoveryAction.RETRY
    manager.record_retry("t")
    assert manager.propose(plan, report, verification).action is not RecoveryAction.RETRY


def test_checkpoint_restore_keeps_completed_tasks():
    a = Task(task_id="a", description="a")
    b = Task(task_id="b", description="b")
    plan = TaskPlan(goal_id="g", tasks=(a, b))
    report = ExecutionReport(plan.plan_id, "g", executions=(TaskExecution("a", "", TaskState.COMPLETED, True, output="ok"),))
    checkpoint = Checkpoint.from_report(report)
    restored = checkpoint.restore(plan)
    states = {t.task_id: t.state for t in restored.tasks}
    assert states["a"] is TaskState.COMPLETED
    assert states["b"] is TaskState.PENDING


# ---------------------------------------------------------------- history


def test_history_records_a_run(root: Path):
    planner = PlanningCoordinator(catalogue=_catalogue(root))
    coordinator = ExecutionCoordinator(planner=planner, root=root)
    coordinator.run_goal(planner.goals.create("Read report.txt"))
    assert coordinator.history.count(kind="verify") >= 1


def test_redact_removes_credentials():
    text = redact("Authorization: Bearer abc123 token=secret99 password=hunter2")
    assert "abc123" not in text
    assert "secret99" not in text
    assert "hunter2" not in text
    assert "<redacted>" in text


def test_history_entry_redacts_on_construction():
    entry = ExecutionHistory().log("test", "api_key=abcdef123456")
    assert "abcdef123456" not in entry.summary


def test_history_round_trips(tmp_path: Path):
    history = ExecutionHistory(data_root=str(tmp_path))
    history.log("plan", "planned a goal")
    assert history.save()
    fresh = ExecutionHistory(data_root=str(tmp_path))
    assert fresh.load() == 1


def test_history_bound_drops_the_oldest():
    history = ExecutionHistory(limit=2)
    history.log("a", "1")
    history.log("b", "2")
    history.log("c", "3")
    assert history.count() == 2
    assert history.recent(1)[0].summary == "3"


# -------------------------------------------------------------- learning


def test_coordinator_records_an_experience(root: Path):
    from core.learning_coordinator import LearningCoordinator

    data = tempfile.mkdtemp()
    learning = LearningCoordinator("fayd_al_ruh", data_root=data)
    planner = PlanningCoordinator(catalogue=_catalogue(root), data_root=data)
    coordinator = ExecutionCoordinator(
        planner=planner, root=root, learning=learning, owner="fayd_al_ruh", data_root=data
    )
    goal = planner.goals.create("Read report.txt and write summary.md")
    result = coordinator.run_goal(goal)
    assert result.success
    assert result.learning_event_id


def test_coordinator_without_learning_records_nothing(root: Path):
    planner = PlanningCoordinator(catalogue=_catalogue(root))
    coordinator = ExecutionCoordinator(planner=planner, root=root)
    result = coordinator.run_goal(planner.goals.create("Read report.txt"))
    assert result.learning_event_id == ""


def test_learning_failure_does_not_break_a_run(root: Path):
    class Broken:
        def learn_experience(self, experience: object) -> object:
            raise RuntimeError("no store")

    planner = PlanningCoordinator(catalogue=_catalogue(root))
    coordinator = ExecutionCoordinator(planner=planner, root=root, learning=Broken())
    result = coordinator.run_goal(planner.goals.create("Read report.txt"))
    assert result.success
    assert result.learning_event_id == ""
    assert coordinator.history.count(kind="learning") == 1


# ---------------------------------------------------------- persistence


def test_a_goal_and_plan_survive_a_restart(root: Path):
    data = tempfile.mkdtemp()
    planner = PlanningCoordinator(catalogue=_catalogue(root), data_root=data)
    coordinator = ExecutionCoordinator(planner=planner, root=root, data_root=data)
    goal = planner.goals.create("Read report.txt and write summary.md")
    result = coordinator.run_goal(goal)

    fresh = PlanningCoordinator(catalogue=_catalogue(root), data_root=data)
    back = fresh.goals.get(goal.goal_id)
    assert back is not None and back.status is GoalStatus.COMPLETED
    plan = fresh.saved_plan(result.plan.plan_id)
    assert plan is not None
    assert all(t.state is TaskState.COMPLETED for t in plan.tasks)


# --------------------------------------------------------------------- CLI


def test_cli_plan_and_run(root: Path, capsys: pytest.CaptureFixture[str]):
    from cli.autonomy import main

    assert main(["plan", "Read report.txt and write summary.md", "--root", str(root)]) == 0
    out = capsys.readouterr().out
    assert "read_file" in out and "write_file" in out

    assert main(["run", "Read report.txt and write summary.md", "--root", str(root)]) == 0
    assert "verified_success" in capsys.readouterr().out


def test_cli_tools_and_gate(capsys: pytest.CaptureFixture[str]):
    from cli.autonomy import main

    assert main(["tools"]) == 0
    assert "calculator" in capsys.readouterr().out
    assert main(["gate"]) == 0
    assert "1.0000" in capsys.readouterr().out


def test_cli_run_reports_failure(root: Path, capsys: pytest.CaptureFixture[str]):
    from cli.autonomy import main

    code = main(["run", "Read missing.txt", "--root", str(root)])
    assert code == 4
    assert "failed" in capsys.readouterr().out


def test_cli_json_output(root: Path, capsys: pytest.CaptureFixture[str]):
    import json

    from cli.autonomy import main

    assert main(["plan", "Read report.txt", "--root", str(root), "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["tasks"]


# ------------------------------------------------- roadmap extensions


def test_planner_persists_ordered_steps_across_a_restart(root: Path):
    from core.goal import Goal as SymbolicGoal
    from core.goal import PlanStep as SymbolicStep
    from core.planner import Planner

    data = tempfile.mkdtemp()
    planner = Planner(data_root=data)
    ordered = planner.order_steps(
        SymbolicGoal("write summary.md"),
        [SymbolicStep("read report.txt"), SymbolicStep("write summary.md", depends_on=("read report.txt",))],
    )
    assert [s.description for s in ordered] == ["read report.txt", "write summary.md"]

    # A fresh planner over the same data root reloads the recorded plan.
    reloaded = Planner(data_root=data)
    assert reloaded.plans
    assert [s.description for s in reloaded.plans[-1]] == ["read report.txt", "write summary.md"]


def test_planner_without_a_data_root_keeps_plans_in_memory(root: Path):
    from core.goal import Goal as SymbolicGoal
    from core.goal import PlanStep as SymbolicStep
    from core.planner import Planner

    planner = Planner()
    planner.order_steps(SymbolicGoal("x"), [SymbolicStep("a")])
    assert planner.plans
    assert planner.save_plans() is False


def test_tool_agent_replays_a_task_plan_through_the_sandbox(root: Path):
    from core.tool_agent import ToolAgent

    sandbox = build_sandbox(root, read_only=False)
    planner = PlanningCoordinator(catalogue=_catalogue(root))
    plan = planner.plan(planner.goals.create("Read report.txt and write summary.md"))
    assert plan is not None

    agent = ToolAgent(sandbox=sandbox)
    seen: list[object] = []
    run = agent.run_task_plan(plan, on_step=seen.append)
    assert run.steps
    assert len(seen) == len(run.steps)
    assert all(step.result.ok for step in run.steps)
    assert (root / "summary.md").exists()


def test_tool_agent_plan_replay_reports_a_refused_tool(root: Path):
    from core.tool_agent import ToolAgent

    sandbox = build_sandbox(root, read_only=True)  # a read-only sandbox refuses the write
    planner = PlanningCoordinator(catalogue=_catalogue(root))
    plan = planner.plan(planner.goals.create("Read report.txt and write summary.md"))
    agent = ToolAgent(sandbox=sandbox, stop_on_error=True)
    run = agent.run_task_plan(plan)
    assert any(not step.result.ok for step in run.steps)
    assert not (root / "summary.md").exists()


# ---------------------------------------------------------- plan evaluator


def test_plan_evaluator_reads_the_runs_verdict(root: Path):
    from core.plan_evaluator import PlanEvaluator

    planner = PlanningCoordinator(catalogue=_catalogue(root))
    coordinator = ExecutionCoordinator(planner=planner, root=root)
    result = coordinator.run_goal(planner.goals.create("Read report.txt and write summary.md"))
    evaluation = PlanEvaluator().evaluate_outcome(result.goal, result)
    assert evaluation.goal_achieved and evaluation.success
    assert evaluation.outcome == "verified_success"


def test_plan_evaluator_is_unverified_without_a_verification():
    from core.plan_evaluator import PlanEvaluator

    evaluation = PlanEvaluator().evaluate_outcome(object(), object())
    assert not evaluation.goal_achieved
    assert evaluation.outcome == "unverified"


def test_plan_evaluator_records_an_experience(root: Path):
    from core.learning_coordinator import LearningCoordinator
    from core.plan_evaluator import PlanEvaluator

    data = tempfile.mkdtemp()
    learning = LearningCoordinator("fayd_al_ruh", data_root=data)
    planner = PlanningCoordinator(catalogue=_catalogue(root))
    coordinator = ExecutionCoordinator(planner=planner, root=root)
    result = coordinator.run_goal(planner.goals.create("Read report.txt and write summary.md"))
    evaluator = PlanEvaluator()
    evaluation = evaluator.evaluate_outcome(result.goal, result)
    event_id = evaluator.record_experience(evaluation, result.goal, learning, scope="fayd_al_ruh")
    assert event_id


def test_plan_evaluator_without_learning_records_nothing(root: Path):
    from core.plan_evaluator import PlanEvaluator

    evaluation = PlanEvaluator().evaluate_outcome(object(), object())
    assert PlanEvaluator().record_experience(evaluation, object(), None) == ""


# ---------------------------------------------------------------- the gate


def test_gate_is_perfect_on_the_real_layer():
    assert task_execution_accuracy() == 1.0
    assert planning_execution_accuracy() == 1.0  # the roadmap alias


def test_gate_is_zero_for_an_empty_catalogue():
    assert task_execution_accuracy(()) == 0.0


def test_gate_discriminates_a_regression(monkeypatch: pytest.MonkeyPatch):
    import core.planning_gate as gate

    monkeypatch.setitem(gate._CHECKERS, "honesty", lambda judgement: False)
    broken = (gate.PlanningJudgement("x", "honesty"), gate.PlanningJudgement("y", "verify"))
    assert gate.planning_execution_accuracy(broken) < 1.0


def test_gate_catalogue_has_negative_controls():
    kinds = {j.kind for j in PLANNING_JUDGEMENTS}
    assert {"honesty", "perm", "verify", "recover"} <= kinds
    names = {j.name for j in PLANNING_JUDGEMENTS}
    assert "honesty-no-fabricated-tool" in names
    assert "recover-no-retry-irreversible" in names
    assert "verify-tool-not-goal" in names
