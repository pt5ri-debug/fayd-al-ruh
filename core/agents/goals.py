"""Goal and task management for each personality.

A personality works on **goals** (what it is trying to achieve) and **tasks** (the
concrete steps it takes). Both belong to one owner, both are recorded, and a task
carries the provenance the collaboration layer needs: who delegated it, its depth,
and the task it descends from -- so a task tree is inspectable and a delegation
loop is detectable.

The manager is in-memory and bounded: it is the *working* task state (durable task
history is the learning system's, Phase 3). It refuses a cross-owner read the same
way the memory layer does, so one personality cannot list another's tasks.

Dependency-free and Tk-free: it holds no model and touches no network.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import Enum
from typing import Any

from core.learning_isolation import is_owner, owner_for

#: How many tasks a manager keeps per owner before the oldest finished task is
#: dropped. Bounded so a long session cannot grow the working state without limit.
DEFAULT_TASK_HISTORY = 200


class TaskStatus(str, Enum):
    """The lifecycle of one task -- one closed vocabulary."""

    PENDING = "pending"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"
    TIMED_OUT = "timed_out"
    REFUSED = "refused"

    @property
    def terminal(self) -> bool:
        """True when the task will not run further."""
        return self in {
            TaskStatus.SUCCEEDED,
            TaskStatus.FAILED,
            TaskStatus.CANCELLED,
            TaskStatus.TIMED_OUT,
            TaskStatus.REFUSED,
        }


def _new_id(prefix: str) -> str:
    return f"{prefix}-{uuid.uuid4().hex[:12]}"


def _now() -> str:
    return datetime.now(UTC).isoformat()


@dataclass(frozen=True)
class Task:
    """One unit of work owned by one personality."""

    task_id: str
    owner: str
    description: str
    goal_id: str = ""
    status: TaskStatus = TaskStatus.PENDING
    result: str = ""
    error: str = ""
    #: The personality that asked this one to do the task (empty when it is the
    #: owner's own task), and how deep the delegation chain is.
    delegated_by: str = ""
    depth: int = 0
    #: The task this one descends from (the parent in the delegation tree).
    parent_task: str = ""
    created_at: str = field(default_factory=_now)
    started_at: str = ""
    finished_at: str = ""

    def with_status(self, status: TaskStatus, **changes: Any) -> Task:
        """A copy in a new status (the record is frozen, so history is immutable)."""
        from dataclasses import replace

        return replace(self, status=status, **changes)

    def describe(self) -> str:
        where = f" [goal {self.goal_id}]" if self.goal_id else ""
        extra = f" — {self.error}" if self.error else ""
        return f"{self.task_id} ({self.owner}){where}: {self.status.value} — {self.description}{extra}"


@dataclass(frozen=True)
class Goal:
    """One goal a personality is pursuing, with the tasks under it."""

    goal_id: str
    owner: str
    description: str
    created_at: str = field(default_factory=_now)
    closed: bool = False

    def describe(self) -> str:
        return f"{self.goal_id} ({self.owner}): {self.description}{' [closed]' if self.closed else ''}"


@dataclass
class GoalManager:
    """The goals and tasks of every personality, kept apart by owner.

    ``max_depth`` bounds a delegation chain: a task whose ``depth`` would exceed it
    is refused, which is one of the two guards against an uncontrolled
    agent-to-agent loop (the other is the delegation budget in
    :mod:`core.agents.collaboration`).
    """

    max_depth: int = 3
    history_limit: int = DEFAULT_TASK_HISTORY
    goals: dict[str, Goal] = field(default_factory=dict)
    tasks: dict[str, Task] = field(default_factory=dict)
    _order: list[str] = field(default_factory=list, repr=False)

    def _check_owner(self, owner: str) -> str:
        resolved = owner_for(owner)
        if not is_owner(resolved):
            raise ValueError(f"{owner!r} is not a known personality.")
        return resolved

    # ----------------------------------------------------------------- goals

    def create_goal(self, owner: str, description: str) -> Goal:
        """Open a goal for ``owner``."""
        resolved = self._check_owner(owner)
        goal = Goal(goal_id=_new_id("goal"), owner=resolved, description=description.strip())
        self.goals[goal.goal_id] = goal
        return goal

    def close_goal(self, goal_id: str) -> Goal:
        """Close a goal (its tasks are kept -- history is never deleted)."""
        goal = self.goals[goal_id]
        from dataclasses import replace

        closed = replace(goal, closed=True)
        self.goals[goal_id] = closed
        return closed

    def goals_for(self, owner: str) -> tuple[Goal, ...]:
        """The goals of ``owner`` (never another's)."""
        resolved = self._check_owner(owner)
        return tuple(goal for goal in self.goals.values() if goal.owner == resolved)

    # ----------------------------------------------------------------- tasks

    def create_task(
        self,
        owner: str,
        description: str,
        *,
        goal_id: str = "",
        delegated_by: str = "",
        depth: int = 0,
        parent_task: str = "",
    ) -> Task:
        """Record a task for ``owner``; refuse one whose depth exceeds the bound."""
        resolved = self._check_owner(owner)
        if depth > self.max_depth:
            raise ValueError(
                f"delegation depth {depth} exceeds the bound of {self.max_depth}."
            )
        task = Task(
            task_id=_new_id("task"),
            owner=resolved,
            description=description.strip(),
            goal_id=goal_id,
            delegated_by=owner_for(delegated_by) if delegated_by else "",
            depth=depth,
            parent_task=parent_task,
        )
        self.tasks[task.task_id] = task
        self._order.append(task.task_id)
        self._trim()
        return task

    def update_task(self, task_id: str, status: TaskStatus, *, result: str = "", error: str = "") -> Task:
        """Move a task to a new status, stamping the start/finish time."""
        task = self.tasks[task_id]
        changes: dict[str, str] = {}
        if status is TaskStatus.RUNNING and not task.started_at:
            changes["started_at"] = _now()
        if status.terminal:
            changes["finished_at"] = _now()
        if result:
            changes["result"] = result
        if error:
            changes["error"] = error
        updated = task.with_status(status, **changes)
        self.tasks[task_id] = updated
        return updated

    def get_task(self, task_id: str) -> Task:
        """One task by id (raises ``KeyError`` for an unknown id)."""
        return self.tasks[task_id]

    def tasks_for(self, owner: str, *, include_delegated: bool = True) -> tuple[Task, ...]:
        """The tasks owned by ``owner``, newest last.

        ``include_delegated`` keeps the tasks ``owner`` was asked to do; with it
        off, only the tasks ``owner`` itself opened are returned.
        """
        resolved = self._check_owner(owner)
        chosen = [
            self.tasks[task_id]
            for task_id in self._order
            if task_id in self.tasks and self.tasks[task_id].owner == resolved
        ]
        if not include_delegated:
            chosen = [task for task in chosen if not task.delegated_by]
        return tuple(chosen)

    def children_of(self, task_id: str) -> tuple[Task, ...]:
        """The tasks delegated *from* ``task_id`` (its children in the tree)."""
        return tuple(task for task in self.tasks.values() if task.parent_task == task_id)

    def history(self, owner: str, *, limit: int = 20) -> tuple[Task, ...]:
        """The most recent tasks of ``owner``."""
        tasks = self.tasks_for(owner)
        return tasks[-limit:]

    def _trim(self) -> None:
        """Drop the oldest finished tasks past the bound (never a running one)."""
        if len(self._order) <= self.history_limit:
            return
        removable = [
            task_id
            for task_id in self._order
            if task_id in self.tasks and self.tasks[task_id].status.terminal
        ]
        excess = len(self._order) - self.history_limit
        for task_id in removable[:excess]:
            self.tasks.pop(task_id, None)
            self._order.remove(task_id)

    def describe(self) -> str:
        return "\n".join(task.describe() for task in self.tasks.values())


__all__ = [
    "DEFAULT_TASK_HISTORY",
    "Goal",
    "GoalManager",
    "Task",
    "TaskStatus",
]
