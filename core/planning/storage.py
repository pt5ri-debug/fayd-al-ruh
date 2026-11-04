"""Durable planning state (Phase 8, sections 3 and 4).

Goals and plans survive a restart. This module is the one place they are
persisted: a :class:`GoalStore` for the structured goals and a :class:`PlanStore`
for the task plans, both over the project's own
:class:`storage.json_store.JsonStore` (no new dependency).

Honest by construction: a corrupt or unreadable document is skipped rather than
crashing the store, the store is bounded so a long run cannot grow without limit,
and nothing here loads a model or reaches the network. The store is
**best-effort**: a read-only data folder returns ``False`` from :meth:`save`
rather than raising, so planning never fails because it could not write.

Dependency-free and Tk-free.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from core.planning.goals import Goal, GoalManager
from core.planning.plan import TaskPlan
from core.planning.tasks import Task, TaskState

#: The namespace and key the goals and plans are stored under.
PLANNING_NAMESPACE = "planning"
GOALS_KEY = "goals"
PLANS_KEY = "plans"

#: The most goals and plans kept on disk.
DEFAULT_GOAL_LIMIT = 500
DEFAULT_PLAN_LIMIT = 500


def _store(data_root: str | None) -> Any:
    if not data_root:
        return None
    try:
        from storage.json_store import JsonStore

        return JsonStore(data_root)
    except Exception:
        return None


# --------------------------------------------------------------------- goals


@dataclass
class GoalStore:
    """Persist a :class:`GoalManager`'s goals and their statuses."""

    data_root: str | None = None
    limit: int = DEFAULT_GOAL_LIMIT

    def save(self, manager: GoalManager) -> bool:
        store = _store(self.data_root)
        if store is None:
            return False
        goals = manager.all_goals()[-self.limit :]
        try:
            store.save(PLANNING_NAMESPACE, GOALS_KEY, [g.to_primitive() for g in goals])
            return True
        except Exception:
            return False

    def load(self, manager: GoalManager | None = None) -> GoalManager:
        """Load the goals into ``manager`` (or a fresh one) and return it."""
        target = manager or GoalManager()
        store = _store(self.data_root)
        if store is None:
            return target
        try:
            data = store.load(PLANNING_NAMESPACE, GOALS_KEY)
        except Exception:
            return target
        if not isinstance(data, list):
            return target
        for item in data:
            if not isinstance(item, dict):
                continue
            try:
                target.add(Goal.from_primitive(item))
            except Exception:
                continue
        return target


# --------------------------------------------------------------------- plans


@dataclass
class PlanStore:
    """Persist task plans so a plan survives a restart."""

    data_root: str | None = None
    limit: int = DEFAULT_PLAN_LIMIT
    plans: dict[str, TaskPlan] = field(default_factory=dict)

    def remember(self, plan: TaskPlan) -> None:
        """Keep a plan in memory (the caller persists with :meth:`save`)."""
        self.plans[plan.plan_id] = plan

    def get(self, plan_id: str) -> TaskPlan | None:
        return self.plans.get(plan_id)

    def save(self) -> bool:
        store = _store(self.data_root)
        if store is None:
            return False
        plans = list(self.plans.values())[-self.limit :]
        try:
            store.save(PLANNING_NAMESPACE, PLANS_KEY, [p.to_primitive() for p in plans])
            return True
        except Exception:
            return False

    def load(self) -> int:
        store = _store(self.data_root)
        if store is None:
            return 0
        try:
            data = store.load(PLANNING_NAMESPACE, PLANS_KEY)
        except Exception:
            return 0
        if not isinstance(data, list):
            return 0
        loaded: dict[str, TaskPlan] = {}
        for item in data:
            if not isinstance(item, dict):
                continue
            try:
                plan = TaskPlan.from_primitive(item)
            except Exception:
                continue
            loaded[plan.plan_id] = plan
        self.plans = dict(list(loaded.items())[-self.limit :])
        return len(self.plans)


def task_to_primitive(task: Task) -> dict[str, Any]:
    """A JSON-safe form of a task (used by :meth:`TaskPlan.to_primitive`)."""
    return {
        "task_id": task.task_id,
        "description": task.description,
        "goal_id": task.goal_id,
        "parent_task": task.parent_task,
        "tool": task.tool,
        "arguments": dict(task.arguments),
        "dependencies": list(task.dependencies),
        "inputs": list(task.inputs),
        "expected_outputs": list(task.expected_outputs),
        "required_permissions": list(task.required_permissions),
        "success_criteria": list(task.success_criteria),
        "state": task.state.value,
        "reversible": task.reversible,
        "retryable": task.retryable,
        "note": task.note,
    }


def task_from_primitive(data: dict[str, Any]) -> Task:
    """Rebuild a task from its primitive form."""
    return Task(
        task_id=str(data["task_id"]),
        description=str(data["description"]),
        goal_id=str(data.get("goal_id", "")),
        parent_task=str(data.get("parent_task", "")),
        tool=str(data.get("tool", "")),
        arguments={str(k): str(v) for k, v in (data.get("arguments") or {}).items()},
        dependencies=tuple(str(d) for d in data.get("dependencies", ())),
        inputs=tuple(str(i) for i in data.get("inputs", ())),
        expected_outputs=tuple(str(o) for o in data.get("expected_outputs", ())),
        required_permissions=tuple(str(p) for p in data.get("required_permissions", ())),
        success_criteria=tuple(str(c) for c in data.get("success_criteria", ())),
        state=TaskState(data.get("state", TaskState.PENDING.value)),
        reversible=bool(data.get("reversible", True)),
        retryable=bool(data.get("retryable", True)),
        note=str(data.get("note", "")),
    )


__all__ = [
    "DEFAULT_GOAL_LIMIT",
    "DEFAULT_PLAN_LIMIT",
    "GOALS_KEY",
    "PLANNING_NAMESPACE",
    "PLANS_KEY",
    "GoalStore",
    "PlanStore",
    "task_from_primitive",
    "task_to_primitive",
]
