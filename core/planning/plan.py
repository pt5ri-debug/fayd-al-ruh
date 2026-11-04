"""The plan representation and validator (Phase 8, section 6).

A :class:`TaskPlan` is the inspectable, editable artifact the planner produces:
the goal it serves, its tasks, the tool each task uses, the dependency graph, the
estimated resources and risks, and the verification criteria. :func:`validate_plan`
runs the ten checks section 6 lists -- goal validity, information availability,
dependencies, tool availability, permissions, resource estimates, risks, ordering,
verification criteria, and the plan's own readiness -- and returns a
:class:`PlanValidation` the coordinator refuses to execute a plan without.

The plan is a plain, frozen value: an edit produces a new plan (via
:meth:`TaskPlan.with_tasks`), so what was validated is what runs.

Dependency-free and Tk-free.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from enum import Enum
from typing import Any

from core.planning.dependencies import DependencyError, DependencyGraph
from core.planning.goals import Goal, GoalKind, GoalStatus
from core.planning.tasks import Task
from core.planning.tools import RiskCategory, ToolCatalogue


class PlanStatus(str, Enum):
    """Where a plan is in its life -- one closed vocabulary."""

    DRAFT = "draft"
    VALIDATED = "validated"
    REJECTED = "rejected"
    RUNNING = "running"
    COMPLETED = "completed"
    PARTIAL = "partial"
    FAILED = "failed"
    CANCELLED = "cancelled"
    SUPERSEDED = "superseded"


@dataclass(frozen=True)
class PlanValidation:
    """The result of validating a plan -- every finding, not just a boolean."""

    ok: bool
    issues: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()
    estimated_cost: int = 0
    highest_risk: RiskCategory = RiskCategory.LOW

    def describe(self) -> str:
        if self.ok and not self.warnings:
            return f"valid (cost {self.estimated_cost}, risk {self.highest_risk.value})"
        lines = ["valid" if self.ok else "invalid"]
        lines.extend(f"  ! {issue}" for issue in self.issues)
        lines.extend(f"  ~ {warning}" for warning in self.warnings)
        return "\n".join(lines)


@dataclass(frozen=True)
class TaskPlan:
    """An inspectable, editable execution plan."""

    goal_id: str
    tasks: tuple[Task, ...] = ()
    plan_id: str = ""
    status: PlanStatus = PlanStatus.DRAFT
    #: Free-text notes the planner recorded (why a tool was chosen, what it
    #: could not decide), so a reader sees the reasoning, not only the result.
    notes: tuple[str, ...] = ()
    #: The revision number -- incremented on every replan (section 6).
    revision: int = 0
    validation: PlanValidation | None = None

    def __post_init__(self) -> None:
        if not self.plan_id:
            import uuid

            object.__setattr__(self, "plan_id", f"plan-{uuid.uuid4().hex[:12]}")
        ids = [task.task_id for task in self.tasks]
        if len(ids) != len(set(ids)):
            raise ValueError("a plan cannot contain duplicate task ids.")

    # ------------------------------------------------------------- editing

    def with_tasks(self, tasks: tuple[Task, ...]) -> TaskPlan:
        return replace(self, tasks=tasks, revision=self.revision + 1)

    def with_status(self, status: PlanStatus) -> TaskPlan:
        return replace(self, status=status)

    def with_validation(self, validation: PlanValidation) -> TaskPlan:
        status = PlanStatus.VALIDATED if validation.ok else PlanStatus.REJECTED
        return replace(self, validation=validation, status=status)

    def task(self, task_id: str) -> Task | None:
        for task in self.tasks:
            if task.task_id == task_id:
                return task
        return None

    def replace_task(self, task: Task) -> TaskPlan:
        tasks = tuple(task if t.task_id == task.task_id else t for t in self.tasks)
        return self.with_tasks(tasks)

    def graph(self) -> DependencyGraph:
        return DependencyGraph.from_tasks(self.tasks)

    @property
    def estimated_cost(self) -> int:
        return sum(max(1, len(t.estimated_resources) or 1) for t in self.tasks)

    def describe(self) -> str:
        header = (
            f"{self.plan_id} rev {self.revision} [{self.status.value}] "
            f"for {self.goal_id}: {len(self.tasks)} task(s)"
        )
        body = [f"  {task.describe()}" for task in self.tasks]
        notes = [f"  note: {note}" for note in self.notes]
        return "\n".join([header, *body, *notes])

    def to_primitive(self) -> dict[str, object]:
        """A JSON-safe form of the plan (for durable storage)."""
        from core.planning.storage import task_to_primitive

        return {
            "plan_id": self.plan_id,
            "goal_id": self.goal_id,
            "status": self.status.value,
            "revision": self.revision,
            "notes": list(self.notes),
            "tasks": [task_to_primitive(task) for task in self.tasks],
        }

    @classmethod
    def from_primitive(cls, data: dict[str, Any]) -> TaskPlan:
        """Rebuild a plan from its primitive form."""
        from core.planning.storage import task_from_primitive

        raw_tasks = data.get("tasks") or []
        tasks = tuple(
            task_from_primitive(item) for item in raw_tasks if isinstance(item, dict)
        )
        return cls(
            goal_id=str(data.get("goal_id", "")),
            tasks=tasks,
            plan_id=str(data.get("plan_id", "")),
            status=PlanStatus(str(data.get("status", PlanStatus.DRAFT.value))),
            notes=tuple(str(n) for n in data.get("notes", ()) or ()),
            revision=int(data.get("revision", 0)),  # type: ignore[arg-type]
        )


def validate_plan(
    plan: TaskPlan,
    goal: Goal,
    catalogue: ToolCatalogue,
    *,
    max_risk: RiskCategory = RiskCategory.MEDIUM,
    known_facts: frozenset[str] = frozenset(),
) -> PlanValidation:
    """Run section 6's pre-execution checks over a plan.

    Returns a :class:`PlanValidation`; the plan is **never** executed when it is
    not ``ok``. An issue is a reason not to run; a warning is something the
    caller should know but that does not, by itself, block execution.
    """
    issues: list[str] = []
    warnings: list[str] = []

    # 1. The goal must be valid and authorizing (a suggestion authorizes nothing).
    if not goal.description.strip():
        issues.append("the goal has no description.")
    if goal.kind is GoalKind.SUGGESTED:
        issues.append("a suggested goal does not authorize execution; the user must ask for it.")
    if goal.status in {GoalStatus.COMPLETED, GoalStatus.CANCELLED, GoalStatus.FAILED}:
        warnings.append(f"the goal is already {goal.status.value}.")

    # 2. The plan must have tasks.
    if not plan.tasks:
        issues.append("the plan has no tasks.")

    # 3-4. Dependencies must be valid and every tool must be available.
    try:
        graph = plan.graph()
    except DependencyError as error:
        issues.append(f"invalid dependencies: {error}")
        graph = DependencyGraph()

    highest = RiskCategory.LOW
    cost = 0
    for task in plan.tasks:
        cost += max(1, len(task.estimated_resources) or 1)
        if not task.tool:
            issues.append(f"task {task.task_id} has no tool selected.")
            continue
        descriptor = catalogue.get(task.tool)
        if descriptor is None:
            issues.append(f"task {task.task_id} names unknown tool {task.tool!r}.")
            continue
        if not catalogue.available(task.tool):
            issues.append(f"tool {task.tool!r} (task {task.task_id}) is not available.")
        if descriptor.risk.rank > max_risk.rank:
            issues.append(
                f"tool {task.tool!r} risk {descriptor.risk.value} exceeds the {max_risk.value} ceiling."
            )
        if descriptor.risk.rank > highest.rank:
            highest = descriptor.risk
        if descriptor.requires_approval and not task.reversible:
            warnings.append(
                f"task {task.task_id} uses an irreversible tool that needs approval."
            )

    # 5. Permissions: a task's declared permissions must cover its tool.
    for task in plan.tasks:
        descriptor = catalogue.get(task.tool)
        if descriptor is None:
            continue
        required = set(task.required_permissions)
        if required and descriptor.capability.value not in required:
            warnings.append(
                f"task {task.task_id} declares {sorted(required)} but its tool needs "
                f"the {descriptor.capability.value} capability."
            )

    # 6. Resource estimate: warn when the plan is very large.
    if len(plan.tasks) > 32:
        warnings.append(f"the plan has {len(plan.tasks)} tasks; consider splitting it.")

    # 7. Risks: name the highest-risk action.
    if highest is RiskCategory.HIGH:
        warnings.append("the plan includes a high-risk action.")

    # 8. Ordering: every dependency must be declared and acyclic (from graph).
    if graph.tasks:
        for task in plan.tasks:
            for dependency in task.dependencies:
                if dependency not in graph.tasks:
                    issues.append(
                        f"task {task.task_id} depends on {dependency!r}, which is not in the plan."
                    )

    # 9. Verification: the goal must say how completion is judged, or the default
    #    ("outcome text present") applies -- reported so the reader knows.
    if not goal.verification.success_criteria and not goal.verification.artifacts:
        warnings.append("the goal declares no explicit success criteria; the default applies.")

    # 10. Readiness: a task with unsatisfied preconditions outside the plan is a
    #     warning (the coordinator may still satisfy them from known facts).
    for task in plan.tasks:
        unknown = set(task.preconditions) - known_facts
        if unknown and not task.dependencies:
            warnings.append(
                f"task {task.task_id} expects preconditions {sorted(unknown)} that the plan "
                "does not produce."
            )

    ok = not issues
    return PlanValidation(
        ok=ok,
        issues=tuple(issues),
        warnings=tuple(warnings),
        estimated_cost=cost,
        highest_risk=highest,
    )


__all__ = ["PlanStatus", "PlanValidation", "TaskPlan", "validate_plan"]
