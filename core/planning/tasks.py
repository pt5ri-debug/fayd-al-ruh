"""The structured task (Phase 8, section 5).

A task is one concrete unit of work under a goal: what it needs, what it
produces, what it depends on, which tool runs it, what permission it needs, how
its success is judged, and what to do if it fails. The record is frozen and its
life is a sequence of new versions, so a task's history is never mutated.

The state vocabulary is the one section 8 lists -- pending, ready, running,
waiting for approval, completed, failed, cancelled, blocked -- plus a single
``skipped`` state for a task a recovery decision deliberately passed over.

Dependency-free and Tk-free.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import Enum


class TaskState(str, Enum):
    """The explicit state of one task -- one closed vocabulary (section 8)."""

    PENDING = "pending"
    READY = "ready"
    RUNNING = "running"
    WAITING_APPROVAL = "waiting_approval"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    BLOCKED = "blocked"
    #: Deliberately passed over by a recovery decision (never a silent drop).
    SKIPPED = "skipped"

    @property
    def terminal(self) -> bool:
        return self in {
            TaskState.COMPLETED,
            TaskState.FAILED,
            TaskState.CANCELLED,
            TaskState.SKIPPED,
        }

    @property
    def active(self) -> bool:
        return self in {TaskState.READY, TaskState.RUNNING, TaskState.WAITING_APPROVAL}


def _new_id(prefix: str) -> str:
    return f"{prefix}-{uuid.uuid4().hex[:12]}"


def _now() -> str:
    return datetime.now(UTC).isoformat()


@dataclass(frozen=True)
class Task:
    """One unit of work in a plan.

    Every field section 5 lists is here. ``tool`` names a tool in the catalogue;
    ``handler`` is the seam an injected executor supplies in a test (the real
    execution path resolves the tool through the sandbox, never from this field,
    so a plan cannot smuggle in an unregistered action).
    """

    description: str
    task_id: str = field(default_factory=lambda: _new_id("task"))
    goal_id: str = ""
    parent_task: str = ""
    preconditions: tuple[str, ...] = ()
    inputs: tuple[str, ...] = ()
    expected_outputs: tuple[str, ...] = ()
    dependencies: tuple[str, ...] = ()
    tool: str = ""
    #: The concrete arguments handed to the tool (validated before execution).
    arguments: dict[str, str] = field(default_factory=dict)
    estimated_resources: tuple[str, ...] = ()
    required_permissions: tuple[str, ...] = ()
    success_criteria: tuple[str, ...] = ()
    failure_handling: str = ""
    state: TaskState = TaskState.PENDING
    #: Whether the tool this task uses requires the user's explicit approval.
    requires_approval: bool = False
    #: Whether this task may be retried after a failure (bounded by the engine).
    retryable: bool = True
    #: Whether the task's action is reversible -- an irreversible task is never
    #: retried automatically (section 13).
    reversible: bool = True
    #: A free-text note recorded when the task was blocked or skipped.
    note: str = ""
    created_at: str = field(default_factory=_now)
    updated_at: str = field(default_factory=_now)

    def __post_init__(self) -> None:
        if not self.description.strip():
            raise ValueError("A task needs a description.")
        if self.task_id in self.dependencies:
            raise ValueError(f"task {self.task_id} cannot depend on itself.")

    def with_state(self, state: TaskState, *, note: str = "") -> Task:
        """A copy of the task in a new state (history is immutable)."""
        from dataclasses import replace

        return replace(
            self,
            state=state,
            note=note or self.note,
            updated_at=_now(),
        )

    def describe(self) -> str:
        tool = f" via {self.tool}" if self.tool else ""
        deps = f" (after {', '.join(self.dependencies)})" if self.dependencies else ""
        return f"{self.task_id}: {self.state.value} — {self.description}{tool}{deps}"


__all__ = ["Task", "TaskState"]
