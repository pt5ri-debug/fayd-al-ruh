"""Execution monitoring (Phase 8, section 11).

Section 11 lists what to monitor -- the current task, progress, dependencies,
tool status, resource consumption, execution time, errors, retry count, approval
requirements, and expected-vs-actual output -- and what to support: progress
reporting, cancellation, pause/resume, deadlock detection, stalled-task
detection, and resource-limit enforcement. It also restates the honesty rule: a
task is not complete merely because a tool returned successfully.

:class:`ExecutionMonitor` is a passive observer the engine feeds: it tracks the
live state of a run, detects a stall (no progress within a bound) and a deadlock
(every remaining task blocked), and enforces a wall-clock budget. It **never**
decides success -- it reports what it sees.

Dependency-free and Tk-free.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass, field

from core.execution.engine import ExecutionReport, TaskExecution
from core.planning.dependencies import DependencyGraph
from core.planning.tasks import Task, TaskState


@dataclass(frozen=True)
class MonitorSnapshot:
    """A point-in-time view of a run (what a progress bar or a log reads)."""

    total: int
    completed: int
    running: int
    failed: int
    blocked: int
    waiting_approval: int
    retries: int
    elapsed_seconds: float
    current_task: str = ""

    @property
    def fraction(self) -> float:
        return round(self.completed / self.total, 6) if self.total else 0.0

    def describe(self) -> str:
        current = f" ({self.current_task})" if self.current_task else ""
        return (
            f"{self.completed}/{self.total} done, {self.running} running, "
            f"{self.failed} failed, {self.blocked} blocked in {self.elapsed_seconds:.2f}s{current}"
        )


@dataclass
class ExecutionMonitor:
    """Track a run's live state, detect stalls/deadlocks, enforce a budget."""

    tasks: dict[str, Task] = field(default_factory=dict)
    retries: int = 0
    started_at: float | None = None
    clock: Callable[[], float] = time.monotonic
    #: A run that makes no progress for this long is *stalled*.
    stall_seconds: float = 120.0
    #: A run over this wall-clock budget is stopped (0 disables the budget).
    budget_seconds: float = 0.0
    _last_progress_at: float = 0.0
    _last_completed: int = 0

    def start(self, tasks: tuple[Task, ...]) -> None:
        self.tasks = {task.task_id: task for task in tasks}
        self.started_at = self.clock()
        self._last_progress_at = self.started_at
        self._last_completed = 0

    def observe(self, task: Task, execution: TaskExecution) -> None:
        """Record one task's outcome, updating progress and retry counts."""
        self.tasks[task.task_id] = task
        self.retries += max(0, execution.attempts - 1)
        completed = sum(
            1 for t in self.tasks.values() if t.state is TaskState.COMPLETED
        )
        if completed > self._last_completed:
            self._last_completed = completed
            self._last_progress_at = self.clock()

    def elapsed(self) -> float:
        if self.started_at is None:
            return 0.0
        return round(self.clock() - self.started_at, 6)

    def snapshot(self, *, current_task: str = "") -> MonitorSnapshot:
        counts: dict[TaskState, int] = {}
        for task in self.tasks.values():
            counts[task.state] = counts.get(task.state, 0) + 1
        return MonitorSnapshot(
            total=len(self.tasks),
            completed=counts.get(TaskState.COMPLETED, 0),
            running=counts.get(TaskState.RUNNING, 0),
            failed=counts.get(TaskState.FAILED, 0),
            blocked=counts.get(TaskState.BLOCKED, 0),
            waiting_approval=counts.get(TaskState.WAITING_APPROVAL, 0),
            retries=self.retries,
            elapsed_seconds=self.elapsed(),
            current_task=current_task,
        )

    def stalled(self) -> bool:
        """True when the run has made no progress for ``stall_seconds``."""
        if self.started_at is None or self.stall_seconds <= 0:
            return False
        return (self.clock() - self._last_progress_at) > self.stall_seconds

    def deadlocked(self) -> bool:
        """True when every remaining task is blocked (no task can make progress)."""
        remaining = [t for t in self.tasks.values() if not t.state.terminal]
        if not remaining:
            return False
        graph = DependencyGraph.from_tasks(tuple(self.tasks.values()))
        if graph.ready():
            return False
        return all(
            task.state in {TaskState.BLOCKED, TaskState.WAITING_APPROVAL}
            or bool(graph.blocked())
            for task in remaining
        )

    def over_budget(self) -> bool:
        if self.budget_seconds <= 0 or self.started_at is None:
            return False
        return self.elapsed() > self.budget_seconds

    def report(self, execution: ExecutionReport) -> str:
        """A readable monitoring summary of a finished run."""
        snapshot = self.snapshot()
        lines = [snapshot.describe()]
        if self.stalled():
            lines.append("  ! stalled: no progress within the stall window.")
        if self.deadlocked():
            lines.append("  ! deadlocked: every remaining task is blocked.")
        if self.over_budget():
            lines.append("  ! over budget: the run exceeded its wall-clock budget.")
        lines.extend(f"  {e.describe()}" for e in execution.executions if not e.ok)
        return "\n".join(lines)


__all__ = ["ExecutionMonitor", "MonitorSnapshot"]
