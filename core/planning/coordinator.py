"""The planning coordinator (Phase 8, sections 3 and 6).

The coordinator is the one entry point that turns a goal into a validated plan:
it decomposes the goal, selects a permitted tool for every task, builds and
checks the dependency graph, validates the whole plan, and returns it. It also
performs the **replanning** section 6 requires -- when a task fails, new
information arrives, a dependency changes, a tool becomes unavailable, or the
outcome is not achieved -- producing a new plan revision rather than mutating the
old one, so what was validated stays inspectable.

The coordinator plans; it does not execute. Execution is
:class:`core.execution.engine.ExecutionEngine`, and the bridge between them is
:class:`core.execution.coordinator.ExecutionCoordinator`.

Dependency-free and Tk-free.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace

from core.planning.decomposition import (
    DecompositionEngine,
    default_decomposition_engine,
)
from core.planning.dependencies import DependencyGraph
from core.planning.goals import Goal, GoalManager, GoalStatus
from core.planning.plan import PlanStatus, TaskPlan, validate_plan
from core.planning.storage import GoalStore, PlanStore
from core.planning.tasks import Task, TaskState
from core.planning.tools import RiskCategory, ToolCatalogue, build_tool_catalogue


def _default_catalogue() -> ToolCatalogue:
    """The catalogue of the project's built-in sandbox tools (real specs)."""
    import tempfile
    from pathlib import Path

    from core.tools import build_sandbox, build_sandbox_tools

    # A throwaway sandbox only to obtain the real specs; the coordinator never
    # runs anything through it.
    sandbox = build_sandbox(Path(tempfile.mkdtemp(prefix="fayd-planning-")))
    return build_tool_catalogue(build_sandbox_tools(sandbox))


@dataclass
class PlanningCoordinator:
    """Turn goals into validated plans, and replan when the world changes."""

    catalogue: ToolCatalogue = field(default_factory=_default_catalogue)
    decomposition: DecompositionEngine | None = None
    max_risk: RiskCategory = RiskCategory.MEDIUM
    goals: GoalManager = field(default_factory=GoalManager)
    #: Where goals and plans are persisted (``None`` keeps them in memory only).
    data_root: str | None = None
    goal_store: GoalStore | None = None
    plan_store: PlanStore | None = None

    def __post_init__(self) -> None:
        if self.decomposition is None:
            self.decomposition = default_decomposition_engine(self.catalogue)
        if self.goal_store is None and self.data_root:
            self.goal_store = GoalStore(self.data_root)
        if self.plan_store is None:
            self.plan_store = PlanStore(self.data_root)
        # Load any persisted goals and plans so both survive a restart.
        self.plan_store.load()
        if self.goal_store is not None:
            self.goals = self.goal_store.load(self.goals)

    # -------------------------------------------------------------- planning

    def plan(self, goal: Goal, *, known_facts: frozenset[str] = frozenset()) -> TaskPlan:
        """Decompose, select tools, and validate a goal into a plan.

        The goal is recorded and moved to ``PLANNING``; the returned plan carries
        its validation, so the caller sees exactly why it may or may not run. An
        invalid plan is returned with status ``REJECTED`` -- never silently
        executed.
        """
        if self.goals.get(goal.goal_id) is None:
            self.goals.add(goal)
        tasks = self.decomposition.decompose(goal, known_facts=known_facts)  # type: ignore[union-attr]
        notes = self.decomposition.notes_for(goal)  # type: ignore[union-attr]
        plan = TaskPlan(goal_id=goal.goal_id, tasks=tasks, notes=notes)
        validation = validate_plan(
            plan, goal, self.catalogue, max_risk=self.max_risk, known_facts=known_facts
        )
        plan = plan.with_validation(validation)
        self.goals.set_status(goal.goal_id, GoalStatus.PLANNING)
        self._persist(goal, plan)
        return plan

    def validate(
        self,
        plan: TaskPlan,
        goal: Goal,
        *,
        known_facts: frozenset[str] = frozenset(),
    ) -> TaskPlan:
        """Re-validate an (edited) plan, returning it with the new verdict."""
        validation = validate_plan(
            plan, goal, self.catalogue, max_risk=self.max_risk, known_facts=known_facts
        )
        return plan.with_validation(validation)

    # ------------------------------------------------------------- replanning

    def replan(
        self,
        plan: TaskPlan,
        goal: Goal,
        *,
        reason: str,
        failed_tasks: tuple[str, ...] = (),
        unavailable_tools: tuple[str, ...] = (),
        known_facts: frozenset[str] = frozenset(),
    ) -> TaskPlan:
        """Produce a new plan revision in response to a change (section 6).

        The old plan is marked ``SUPERSEDED`` (its history is kept); the new plan
        keeps every completed task's work, re-decomposes the rest, and re-validates.
        A failure or an unavailable tool is recorded in the plan's notes so the
        reason for the revision is visible.
        """
        notes = list(plan.notes)
        notes.append(f"replanned: {reason}")
        if failed_tasks:
            notes.append("failed tasks: " + ", ".join(failed_tasks))
        if unavailable_tools:
            notes.append("unavailable tools: " + ", ".join(unavailable_tools))

        # Re-decompose the goal, but keep tasks that already completed.
        completed = {
            task.description: task
            for task in plan.tasks
            if task.state is TaskState.COMPLETED
        }
        fresh = self.decomposition.decompose(goal, known_facts=known_facts)  # type: ignore[union-attr]
        merged: list[Task] = []
        for task in fresh:
            if task.description in completed:
                merged.append(completed[task.description])
            else:
                merged.append(task)
        new_plan = TaskPlan(
            goal_id=plan.goal_id,
            tasks=tuple(merged),
            revision=plan.revision + 1,
            notes=tuple(notes),
        )
        validation = validate_plan(
            new_plan, goal, self.catalogue, max_risk=self.max_risk, known_facts=known_facts
        )
        new_plan = new_plan.with_validation(validation)
        if validation.ok:
            new_plan = replace(new_plan, status=PlanStatus.VALIDATED)
        return new_plan

    def mark_superseded(self, plan: TaskPlan) -> TaskPlan:
        return plan.with_status(PlanStatus.SUPERSEDED)

    # -------------------------------------------------------------- persistence

    def _persist(self, goal: Goal, plan: TaskPlan) -> None:
        """Save the goal and plan (best-effort: a read-only root is not an error)."""
        if self.goal_store is not None:
            self.goal_store.save(self.goals)
        if self.plan_store is not None:
            self.plan_store.remember(plan)
            self.plan_store.save()

    def persist(self, goal: Goal | None = None, plan: TaskPlan | None = None) -> None:
        """Persist the current goals and (optionally) one plan."""
        if self.goal_store is not None:
            self.goal_store.save(self.goals)
        if plan is not None and self.plan_store is not None:
            self.plan_store.remember(plan)
        if self.plan_store is not None:
            self.plan_store.save()

    def remember_plan(self, plan: TaskPlan) -> None:
        if self.plan_store is not None:
            self.plan_store.remember(plan)

    def saved_plan(self, plan_id: str) -> TaskPlan | None:
        return self.plan_store.get(plan_id) if self.plan_store else None

    def save_goals(self) -> bool:
        return self.goal_store.save(self.goals) if self.goal_store else False

    def load_goals(self) -> int:
        if self.goal_store is None:
            return 0
        before = len(self.goals.all_goals())
        self.goals = self.goal_store.load(self.goals)
        return len(self.goals.all_goals()) - before

    # -------------------------------------------------------------- queries

    def graph(self, plan: TaskPlan) -> DependencyGraph:
        return plan.graph()

    def open_goals(self) -> tuple[Goal, ...]:
        return self.goals.open_goals()

    def status(self) -> str:
        return self.goals.describe()


def default_planner(
    *, max_risk: RiskCategory = RiskCategory.MEDIUM, data_root: str | None = None
) -> PlanningCoordinator:
    """The coordinator the execution layer uses by default."""
    return PlanningCoordinator(max_risk=max_risk, data_root=data_root)


__all__ = ["PlanningCoordinator", "default_planner"]
