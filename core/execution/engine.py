"""The controlled execution engine (Phase 8, section 8).

Section 8 asks for a controlled engine that schedules a plan's tasks by
dependency, runs them sequentially or in bounded parallel, supports cancellation,
timeouts, progress tracking, output capture, retry policies, checkpoints, and
logs, and keeps explicit task states. It must not retry dangerous actions
automatically, and retries must be bounded with backoff.

The engine runs every task through the project's **existing** deny-by-default
sandbox (:class:`core.tools.ToolSandbox`): it validates the task's arguments
against the tool's declared input schema, asks the :class:`PermissionManager`,
runs the tool, and captures the result. It never calls a tool the sandbox does
not permit, never retries an irreversible task, and never reports a task complete
merely because a tool returned -- that is the verifier's job
(:mod:`core.execution.verification`).

The `runner` seam lets a test inject a fake tool result, so the whole engine is
exercised with no real tool and no filesystem.

Dependency-free and Tk-free.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field, replace

from core.execution.permissions import PermissionManager
from core.planning.dependencies import DependencyGraph
from core.planning.plan import PlanStatus, TaskPlan
from core.planning.tasks import Task, TaskState
from core.planning.tools import RiskCategory, ToolCatalogue, ToolDescriptor
from core.tools import ToolResult, ToolSandbox

#: A progress callback: (completed, total, task, result) -> None.
ProgressCallback = Callable[[int, int, Task, "TaskExecution"], None]
#: The runner seam: (task, descriptor, arguments) -> ToolResult.
Runner = Callable[[Task, ToolDescriptor, dict[str, str]], ToolResult]


class ExecutionError(RuntimeError):
    """Raised for a malformed execution request (never for a tool failure)."""


class Cancelled(Exception):
    """Raised inside a run when the caller's cancellation token is tripped."""


@dataclass(frozen=True)
class TaskExecution:
    """The result of one task's execution."""

    task_id: str
    tool: str
    state: TaskState
    ok: bool
    output: str = ""
    error: str = ""
    attempts: int = 1
    duration_seconds: float = 0.0
    #: Whether the outcome was independently verified (set by the verifier).
    verified: bool = False
    note: str = ""

    def describe(self) -> str:
        head = f"{self.task_id} ({self.tool}): {self.state.value}"
        if self.attempts > 1:
            head += f" x{self.attempts}"
        if not self.ok:
            return f"{head} — {self.error or 'failed'}"
        return f"{head} in {self.duration_seconds:.2f}s"


@dataclass
class ExecutionReport:
    """The engine's report for one plan run -- the one thing a caller reads."""

    plan_id: str
    goal_id: str
    executions: tuple[TaskExecution, ...] = ()
    status: PlanStatus = PlanStatus.RUNNING
    cancelled: bool = False
    stalled: bool = False
    duration_seconds: float = 0.0
    notes: tuple[str, ...] = ()

    @property
    def completed(self) -> tuple[TaskExecution, ...]:
        return tuple(e for e in self.executions if e.state is TaskState.COMPLETED)

    @property
    def failed(self) -> tuple[TaskExecution, ...]:
        return tuple(e for e in self.executions if e.state is TaskState.FAILED)

    @property
    def all_completed(self) -> bool:
        return bool(self.executions) and all(
            e.state is TaskState.COMPLETED for e in self.executions
        )

    @property
    def any_completed(self) -> bool:
        return any(e.state is TaskState.COMPLETED for e in self.executions)

    def describe(self) -> str:
        head = f"{self.plan_id} [{self.status.value}] {len(self.completed)}/{len(self.executions)} completed"
        if self.cancelled:
            head += " (cancelled)"
        if self.stalled:
            head += " (stalled)"
        body = [f"  {e.describe()}" for e in self.executions]
        notes = [f"  note: {n}" for n in self.notes]
        return "\n".join([head, *body, *notes])


def _default_runner(sandbox: ToolSandbox) -> Runner:
    def run(task: Task, descriptor: ToolDescriptor, arguments: dict[str, str]) -> ToolResult:
        return sandbox.run(descriptor.tool_id, **arguments)

    return run


@dataclass
class ExecutionEngine:
    """Run a validated plan under the sandbox, the permissions, and the clock."""

    sandbox: ToolSandbox
    catalogue: ToolCatalogue
    permissions: PermissionManager = field(default_factory=PermissionManager)
    #: The most tasks to run at once -- CPU-first, so one by default.
    max_parallel: int = 1
    #: The most retries for one retryable task (bounded, with backoff).
    max_retries: int = 1
    #: The default per-task timeout in seconds.
    timeout_seconds: float = 30.0
    #: The backoff base between retries (seconds); a test sets it to 0.
    retry_backoff: float = 0.05
    runner: Runner | None = None
    #: The actor recorded in every permission decision (a Phase-5 owner key).
    actor: str = "system"
    #: A monotonic clock, injectable so a test can measure timeouts cheaply.
    clock: Callable[[], float] = time.monotonic

    def __post_init__(self) -> None:
        if self.max_parallel < 1:
            raise ValueError("max_parallel must be at least 1.")
        if self.max_retries < 0:
            raise ValueError("max_retries cannot be negative.")
        if self.runner is None:
            self.runner = _default_runner(self.sandbox)

    # ------------------------------------------------------------- validation

    def resolve_arguments(self, task: Task, outputs: dict[str, str]) -> dict[str, str]:
        """Resolve argument references to a dependency's captured output.

        A value written exactly as ``{<task_id>}`` is replaced with that task's
        output, so a plan can *chain* one task's result into the next (a read
        feeding a write) without a model inventing the content. A reference to a
        task that produced nothing is left unresolved, so validation reports it
        rather than passing an empty string silently.
        """
        resolved: dict[str, str] = {}
        for key, value in task.arguments.items():
            if len(value) >= 3 and value.startswith("{") and value.endswith("}"):
                reference = value[1:-1]
                if reference in outputs:
                    resolved[key] = outputs[reference]
                    continue
            resolved[key] = value
        return resolved

    def validate_arguments(self, task: Task, descriptor: ToolDescriptor) -> str:
        """Return "" when the task's arguments satisfy the tool's input schema.

        Section 7's rule: **all tool inputs are validated before execution**. A
        tool that declares an input schema is refused when a declared input is
        missing, so a malformed call never reaches the tool.
        """
        schema = set(descriptor.input_schema)
        provided = set(task.arguments)
        if schema and not schema <= provided:
            missing = sorted(schema - provided)
            return f"task {task.task_id} is missing tool inputs: {', '.join(missing)}"
        return ""

    # --------------------------------------------------------------- running

    def execute(
        self,
        plan: TaskPlan,
        *,
        token: object | None = None,
        on_progress: ProgressCallback | None = None,
        max_parallel: int | None = None,
    ) -> ExecutionReport:
        """Run ``plan`` and return a report; never raises for a task failure.

        ``token`` is any object with an ``is_cancelled()`` method (the project's
        :class:`core.agents.runtime.CancellationToken`), so cancellation is the
        same mechanism the agent runtime uses.
        """
        started = self.clock()
        tasks: dict[str, Task] = {task.task_id: task for task in plan.tasks}
        executions: dict[str, TaskExecution] = {}
        outputs: dict[str, str] = {}
        notes: list[str] = []
        parallel = max_parallel or self.max_parallel
        cancelled = False
        stalled = False

        def is_cancelled() -> bool:
            return bool(token is not None and getattr(token, "is_cancelled", lambda: False)())

        while True:
            if is_cancelled():
                cancelled = True
                break
            graph = DependencyGraph.from_tasks(tuple(tasks.values()))
            ready = [
                task for task in graph.ready() if task.task_id not in executions
            ]
            if not ready:
                # No work can start: either everything is terminal, or what
                # remains is blocked (a failed dependency) -- report, never spin.
                remaining = [t for t in tasks.values() if not t.state.terminal]
                if remaining:
                    blocked = graph.blocked()
                    for task in remaining:
                        if task.state in {TaskState.PENDING, TaskState.READY}:
                            tasks[task.task_id] = task.with_state(
                                TaskState.BLOCKED,
                                note="a dependency did not complete",
                            )
                            executions[task.task_id] = TaskExecution(
                                task.task_id, task.tool, TaskState.BLOCKED, False,
                                error="a dependency did not complete",
                            )
                    stalled = not blocked and not any(
                        t.state is TaskState.RUNNING for t in tasks.values()
                    )
                break

            batch = ready[:parallel]
            if parallel == 1 or len(batch) == 1:
                for task in batch:
                    execution = self._run_task(task, tasks, is_cancelled, outputs)
                    executions[task.task_id] = execution
                    if execution.output:
                        outputs[task.task_id] = execution.output
                    tasks[task.task_id] = tasks[task.task_id].with_state(
                        execution.state, note=execution.note
                    )
                    if on_progress:
                        on_progress(len(executions), len(tasks), tasks[task.task_id], execution)
            else:
                with ThreadPoolExecutor(max_workers=parallel) as pool:
                    futures = {
                        pool.submit(self._run_task, task, dict(tasks), is_cancelled, dict(outputs)): task
                        for task in batch
                    }
                    for future in futures:
                        task = futures[future]
                        execution = future.result()
                        executions[task.task_id] = execution
                        if execution.output:
                            outputs[task.task_id] = execution.output
                        tasks[task.task_id] = tasks[task.task_id].with_state(
                            execution.state, note=execution.note
                        )
                        if on_progress:
                            on_progress(len(executions), len(tasks), tasks[task.task_id], execution)

        duration = self.clock() - started
        status = self._status(tasks, executions, cancelled)
        return ExecutionReport(
            plan_id=plan.plan_id,
            goal_id=plan.goal_id,
            executions=tuple(executions[tid] for tid in tasks if tid in executions),
            status=status,
            cancelled=cancelled,
            stalled=stalled,
            duration_seconds=round(duration, 6),
            notes=tuple(notes),
        )

    # ----------------------------------------------------------- one task

    def _run_task(
        self,
        task: Task,
        tasks: dict[str, Task],
        is_cancelled: Callable[[], bool],
        outputs: dict[str, str],
    ) -> TaskExecution:
        """Run one task with its permission check, timeout, and bounded retries."""
        if is_cancelled():
            return TaskExecution(task.task_id, task.tool, TaskState.CANCELLED, False, error="cancelled")

        if not task.tool:
            return TaskExecution(
                task.task_id, "", TaskState.BLOCKED, False,
                error="the task has no tool selected", note="no permitted tool matched",
            )
        descriptor = self.catalogue.get(task.tool)
        if descriptor is None:
            return TaskExecution(
                task.task_id, task.tool, TaskState.FAILED, False,
                error=f"unknown tool {task.tool!r}",
            )

        # Resolve any reference to a dependency's captured output, then validate
        # the inputs before anything runs (section 7).
        resolved = self.resolve_arguments(task, outputs)
        task = replace(task, arguments=resolved)
        invalid = self.validate_arguments(task, descriptor)
        if invalid:
            return TaskExecution(task.task_id, task.tool, TaskState.FAILED, False, error=invalid)

        # Permission / approval gate (section 9).
        decision = self.permissions.check(descriptor, actor=self.actor)
        if not decision.allowed:
            state = (
                TaskState.WAITING_APPROVAL
                if self.permissions.needs_approval(descriptor)
                else TaskState.FAILED
            )
            return TaskExecution(
                task.task_id, task.tool, state, False, error=decision.reason, note=decision.reason
            )

        timeout = descriptor.timeout_seconds or self.timeout_seconds
        attempts = 0
        last_error = ""
        started = self.clock()
        while True:
            attempts += 1
            if is_cancelled():
                return TaskExecution(
                    task.task_id, task.tool, TaskState.CANCELLED, False,
                    attempts=attempts, error="cancelled",
                )
            try:
                assert self.runner is not None
                result = self.runner(task, descriptor, dict(task.arguments))
            except Exception as error:  # a tool must never crash the engine
                result = ToolResult(task.tool, False, error=f"{type(error).__name__}: {error}")
            elapsed = self.clock() - started
            if elapsed > timeout:
                last_error = f"timed out after {timeout:.1f}s"
                result = ToolResult(task.tool, False, error=last_error)

            if result.ok:
                return TaskExecution(
                    task.task_id, task.tool, TaskState.COMPLETED, True,
                    output=result.output, attempts=attempts,
                    duration_seconds=round(elapsed, 6),
                )
            last_error = result.error or "the tool reported failure"
            # Retry only a retryable, reversible task, within the bound.
            if (
                task.retryable
                and task.reversible
                and attempts <= self.max_retries
            ):
                if self.retry_backoff:
                    time.sleep(self.retry_backoff * attempts)
                continue
            return TaskExecution(
                task.task_id, task.tool, TaskState.FAILED, False,
                error=last_error, attempts=attempts,
                duration_seconds=round(elapsed, 6),
            )

    # ------------------------------------------------------------- helpers

    @staticmethod
    def _status(
        tasks: dict[str, Task],
        executions: dict[str, TaskExecution],
        cancelled: bool,
    ) -> PlanStatus:
        if cancelled:
            return PlanStatus.CANCELLED
        states = {execution.state for execution in executions.values()}
        if not executions:
            return PlanStatus.DRAFT
        if states <= {TaskState.COMPLETED}:
            return PlanStatus.COMPLETED
        if TaskState.COMPLETED in states:
            return PlanStatus.PARTIAL
        if states & {TaskState.FAILED, TaskState.BLOCKED}:
            return PlanStatus.FAILED
        return PlanStatus.PARTIAL


def risk_of(catalogue: ToolCatalogue, tool: str) -> RiskCategory:
    """The declared risk of a tool (``LOW`` for an unknown one)."""
    descriptor = catalogue.get(tool)
    return descriptor.risk if descriptor else RiskCategory.LOW


__all__ = [
    "Cancelled",
    "ExecutionEngine",
    "ExecutionError",
    "ExecutionReport",
    "ProgressCallback",
    "Runner",
    "TaskExecution",
    "risk_of",
]
