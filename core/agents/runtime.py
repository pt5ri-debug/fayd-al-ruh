"""The shared agent runtime: run one personality's task, bounded and audited.

The task requires one shared runtime capable of executing personality-specific
tasks -- initialization, context preparation, task execution, memory retrieval,
reasoning, tool selection through the existing permitted interfaces, result
validation, completion reporting, cancellation, timeouts, resource limits, and
error handling -- and it explicitly asks that the runtime **not** spin up a
separate process or model per personality.

So this runtime is a *coordinator*, not a model host. It holds no language model
of its own: the actual work is an injectable :class:`TaskExecutor` (the mouth, the
reasoning facade, a test's stub), and the runtime contributes what the task asks
for around it -- the capability gate, the resource governor, the working context,
the task record, the deadline and cancellation, the result validation, and the
audit. That keeps one process, reuses whatever model the mouth already has, and
makes the whole runtime testable with no model and no display.

Dependency-free and Tk-free: importing it loads no model and touches no network.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

from core.agents.capabilities import (
    AgentCapability,
    CapabilityGateway,
    CapabilitySet,
)
from core.agents.context import ContextManager, WorkingContext
from core.agents.goals import GoalManager, Task, TaskStatus
from core.agents.registry import PersonalityProfile, PersonalityRegistry
from core.learning_isolation import owner_for

#: The default wall-clock budget for one task, in seconds. Generous enough for a
#: slow CPU-bound local model, small enough that a hung executor is reported.
DEFAULT_TIMEOUT_SECONDS = 120.0


class TaskCancelled(Exception):
    """Raised inside an executor when the task's cancellation token is tripped."""


class TaskTimedOut(Exception):
    """Raised inside an executor when the task's deadline has passed."""


class ExecutionDeferred(Exception):
    """Raised when the resource governor declines to start the task now."""

    def __init__(self, reason: str, detail: str = "") -> None:
        super().__init__(detail or reason)
        self.reason = reason
        self.detail = detail


@dataclass
class CancellationToken:
    """A cooperative cancellation + deadline handle passed to an executor.

    Python cannot preempt a running call, so cancellation is **cooperative**: the
    executor polls :meth:`check` (or :meth:`is_cancelled`) and returns promptly. A
    deadline is the same mechanism with a clock behind it, so a hung executor that
    polls is stopped, and one that does not poll is still bounded by the runtime's
    own post-run check. The token is the *only* thing an executor needs to honour
    to be cancellable.
    """

    deadline: float = 0.0
    _cancelled: bool = False

    def cancel(self) -> None:
        """Ask the executor to stop at its next check."""
        self._cancelled = True

    def is_cancelled(self) -> bool:
        return self._cancelled

    def expired(self) -> bool:
        """True when a deadline was set and has passed."""
        return bool(self.deadline) and time.monotonic() >= self.deadline

    def remaining(self) -> float | None:
        """Seconds left before the deadline, or ``None`` when there is none."""
        if not self.deadline:
            return None
        return max(0.0, self.deadline - time.monotonic())

    def check(self) -> None:
        """Raise :class:`TaskCancelled`/:class:`TaskTimedOut` if either applies.

        An executor calls this between steps so a cancelled or over-budget task
        stops promptly instead of running to completion.
        """
        if self._cancelled:
            raise TaskCancelled("the task was cancelled.")
        if self.expired():
            raise TaskTimedOut("the task exceeded its time budget.")


@dataclass(frozen=True)
class AgentRequest:
    """What an executor is handed for one task.

    It carries everything the task needs and nothing it does not: the owner and
    its profile (who is acting), the task record, the working context, the
    cancellation token, the capability the runtime already cleared, and any
    caller-supplied payload. An executor therefore never has to resolve a
    personality or re-check a permission.
    """

    owner: str
    profile: PersonalityProfile
    task: Task
    context: WorkingContext
    token: CancellationToken
    question: str = ""
    capability: AgentCapability = AgentCapability.ANSWER
    payload: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class AgentResponse:
    """What an executor returns for one task -- validated by the runtime."""

    ok: bool = True
    text: str = ""
    data: dict[str, Any] = field(default_factory=dict)
    error: str = ""
    #: The capabilities the executor actually used (for the audit trail).
    capabilities_used: tuple[AgentCapability, ...] = ()

    def describe(self) -> str:
        if not self.ok:
            return f"failed: {self.error or 'no reason given'}"
        return self.text[:200] if self.text else "(no text)"


@runtime_checkable
class TaskExecutor(Protocol):
    """The seam that does the actual work for one task.

    The runtime calls ``executor(request) -> AgentResponse``; a real executor wraps
    a mouth (which owns the model), and a test injects a stub, so the runtime is
    exercised with no model. An executor should poll ``request.token.check()``
    between steps to be cancellable.
    """

    def __call__(self, request: AgentRequest) -> AgentResponse: ...


@dataclass(frozen=True)
class ValidationResult:
    """The outcome of validating an executor's response."""

    ok: bool
    reason: str = ""


def default_validator(response: AgentResponse, *, max_chars: int = 200_000) -> ValidationResult:
    """The runtime's default result validation.

    A failed response is reported as-is; a successful one must carry either text
    or structured data (an empty success is a bug, not an answer), and its text is
    bounded so a runaway executor cannot bloat the working context.
    """
    if not response.ok:
        return ValidationResult(False, response.error or "the executor reported failure.")
    if not response.text.strip() and not response.data:
        return ValidationResult(False, "the executor returned an empty result.")
    if len(response.text) > max_chars:
        return ValidationResult(False, f"the result exceeds {max_chars} characters.")
    return ValidationResult(True)


@dataclass(frozen=True)
class TaskOutcome:
    """The runtime's report for one task -- the one thing a caller reads."""

    task: Task
    status: TaskStatus
    response: AgentResponse
    duration_seconds: float
    deferred_reason: str = ""
    validated: bool = True
    validation_reason: str = ""

    @property
    def ok(self) -> bool:
        return self.status is TaskStatus.SUCCEEDED

    def describe(self) -> str:
        head = f"{self.task.task_id} ({self.task.owner}): {self.status.value}"
        if self.deferred_reason:
            return f"{head} — deferred: {self.deferred_reason}"
        if not self.validated:
            return f"{head} — invalid: {self.validation_reason}"
        return f"{head} in {self.duration_seconds:.2f}s — {self.response.describe()}"


@dataclass
class AgentRuntime:
    """The one runtime every personality's task goes through.

    Everything it depends on is injectable, so a test runs the whole thing with no
    model, no display, and no disk:

    * ``registry`` -- the personality registry (identity + capabilities);
    * ``gateway`` -- the capability gate (deny-by-default, audited);
    * ``governor`` -- the resource governor (built lazily, so importing is free);
    * ``contexts`` / ``goals`` -- the working state;
    * ``executor`` -- the work (a mouth wrapper, or a stub);
    * ``events`` -- an optional event bus, published to on every transition.
    """

    registry: PersonalityRegistry = field(default_factory=PersonalityRegistry)
    gateway: CapabilityGateway = field(default_factory=CapabilityGateway)
    contexts: ContextManager = field(default_factory=ContextManager)
    goals: GoalManager = field(default_factory=GoalManager)
    executor: TaskExecutor | None = None
    events: Any | None = None
    governor: Any | None = None
    validator: Any = default_validator
    default_timeout: float = DEFAULT_TIMEOUT_SECONDS

    # ------------------------------------------------------------ resources

    def _governor(self) -> Any:
        if self.governor is None:
            from core.resource_governor import ResourceGovernor

            self.governor = ResourceGovernor()
        return self.governor

    def _publish(self, name: str, data: dict[str, Any]) -> None:
        if self.events is not None:
            from core.best_effort import best_effort

            with best_effort(f"publish {name}"):
                self.events.publish(name, data, source="agents")

    # ------------------------------------------------------------ the run

    def run(
        self,
        owner: str,
        description: str,
        *,
        executor: TaskExecutor | None = None,
        capability: AgentCapability = AgentCapability.ANSWER,
        question: str = "",
        payload: dict[str, Any] | None = None,
        timeout: float | None = None,
        goal_id: str = "",
        delegated_by: str = "",
        depth: int = 0,
        parent_task: str = "",
        memory_bytes: int | None = None,
        token: CancellationToken | None = None,
    ) -> TaskOutcome:
        """Execute one task for ``owner`` and return the outcome.

        The order is deliberate and is the task's own list: initialize (resolve the
        profile) → check the capability → ask the governor → record the task →
        prepare the context → execute with a deadline → validate → report. Every
        failure mode has its own status, so a caller can tell a refusal from a
        timeout from a crash.
        """
        resolved = owner_for(owner)
        profile = self.registry.resolve(resolved)
        context = self.contexts.context_for(resolved)

        # 1. Capability gate -- a personality may not do what it is not granted.
        decision = self.gateway.check(resolved, capability)
        if not decision.allowed:
            task = self.goals.create_task(
                resolved, description, goal_id=goal_id, delegated_by=delegated_by,
                depth=depth, parent_task=parent_task,
            )
            updated = self.goals.update_task(task.task_id, TaskStatus.REFUSED, error=decision.reason)
            self._publish("agent.task.refused", {"owner": resolved, "reason": decision.reason})
            return TaskOutcome(
                task=updated,
                status=TaskStatus.REFUSED,
                response=AgentResponse(ok=False, error=decision.reason),
                duration_seconds=0.0,
                validation_reason=decision.reason,
            )

        # 2. Resource governor -- a task that cannot be judged is deferred, not run.
        try:
            self._admit(resolved, description, memory_bytes)
        except ExecutionDeferred as deferred:
            task = self.goals.create_task(
                resolved, description, goal_id=goal_id, delegated_by=delegated_by,
                depth=depth, parent_task=parent_task,
            )
            updated = self.goals.update_task(task.task_id, TaskStatus.PENDING, error=deferred.detail)
            self._publish(
                "agent.task.deferred", {"owner": resolved, "reason": deferred.reason}
            )
            return TaskOutcome(
                task=updated,
                status=TaskStatus.PENDING,
                response=AgentResponse(ok=False, error=deferred.detail),
                duration_seconds=0.0,
                deferred_reason=deferred.reason,
            )

        # 3. Record the task and prepare the working context.
        task = self.goals.create_task(
            resolved, description, goal_id=goal_id, delegated_by=delegated_by,
            depth=depth, parent_task=parent_task,
        )
        context.current_task = task.task_id
        context.add("task.started", description, task_id=task.task_id)
        self.goals.update_task(task.task_id, TaskStatus.RUNNING)
        self._publish("agent.task.started", {"owner": resolved, "task": task.task_id})

        # 4. Execute with a deadline and a cancellation token.
        active_executor = executor or self.executor
        run_token = token or CancellationToken()
        budget = self.default_timeout if timeout is None else timeout
        if budget and budget > 0:
            run_token.deadline = time.monotonic() + budget
        started = time.monotonic()

        if active_executor is None:
            outcome = self._finish(
                task,
                TaskStatus.FAILED,
                AgentResponse(ok=False, error="no executor was provided for this task."),
                started,
                context,
            )
            return outcome

        request = AgentRequest(
            owner=resolved,
            profile=profile,
            task=task,
            context=context,
            token=run_token,
            question=question or description,
            capability=capability,
            payload=dict(payload or {}),
        )
        try:
            response = active_executor(request)
        except TaskCancelled:
            return self._finish(
                task, TaskStatus.CANCELLED, AgentResponse(ok=False, error="cancelled"), started, context
            )
        except TaskTimedOut:
            return self._finish(
                task, TaskStatus.TIMED_OUT, AgentResponse(ok=False, error="timed out"), started, context
            )
        except Exception as error:  # an executor's crash is a failed task, never fatal
            from core.best_effort import best_effort

            with best_effort("record an executor failure"):
                self._publish(
                    "agent.task.error", {"owner": resolved, "task": task.task_id, "error": str(error)}
                )
            return self._finish(
                task, TaskStatus.FAILED, AgentResponse(ok=False, error=str(error)), started, context
            )

        # A deadline that passed while the executor ignored the token is still a timeout.
        if run_token.expired():
            return self._finish(
                task, TaskStatus.TIMED_OUT, AgentResponse(ok=False, error="timed out"), started, context
            )

        # 5. Validate the result before it is accepted.
        validation = self.validator(response)
        status = TaskStatus.SUCCEEDED if validation.ok else TaskStatus.FAILED
        return self._finish(task, status, response, started, context, validation=validation)

    # ------------------------------------------------------------- helpers

    def _admit(self, owner: str, description: str, memory_bytes: int | None) -> None:
        """Ask the resource governor; raise :class:`ExecutionDeferred` on a deferral."""
        decision = self._governor().request(
            self._resource_request(owner, description, memory_bytes)
        )
        if not decision.granted:
            raise ExecutionDeferred(decision.reason.value, decision.detail)

    @staticmethod
    def _resource_request(owner: str, description: str, memory_bytes: int | None) -> Any:
        from core.resource_governor import ResourceRequest

        return ResourceRequest(
            label=f"agent:{owner}",
            memory_bytes=memory_bytes,
            metadata={"owner": owner, "description": description[:120]},
        )

    def _finish(
        self,
        task: Task,
        status: TaskStatus,
        response: AgentResponse,
        started: float,
        context: WorkingContext,
        *,
        validation: ValidationResult | None = None,
    ) -> TaskOutcome:
        """Update the task, the context, and the audit; return the outcome."""
        duration = time.monotonic() - started
        error = "" if status is TaskStatus.SUCCEEDED else (response.error or status.value)
        updated = self.goals.update_task(
            task.task_id, status, result=response.text if response.ok else "", error=error
        )
        context.current_task = ""
        context.add(
            "task.finished",
            response.text[:500] if response.ok else error,
            task_id=task.task_id,
            status=status.value,
        )
        self._publish(
            "agent.task.finished",
            {"owner": task.owner, "task": task.task_id, "status": status.value},
        )
        return TaskOutcome(
            task=updated,
            status=status,
            response=response,
            duration_seconds=duration,
            validated=validation.ok if validation is not None else True,
            validation_reason=validation.reason if validation is not None else "",
        )

    def cancel(self, owner: str, task_id: str) -> bool:
        """Cancel a running task (cooperative; returns whether it was found)."""
        try:
            task = self.goals.get_task(task_id)
        except KeyError:
            return False
        if task.owner != owner_for(owner) or task.status.terminal:
            return False
        self.goals.update_task(task_id, TaskStatus.CANCELLED, error="cancelled by request")
        self._publish("agent.task.cancelled", {"owner": task.owner, "task": task_id})
        return True

    def capabilities_for(self, owner: str) -> CapabilitySet:
        """The effective capability set of ``owner`` (the registry's answer)."""
        return self.registry.resolve(owner).capabilities

    def describe(self) -> str:
        return "\n".join(
            [
                "Agent runtime",
                f"  personalities: {len(self.registry.ids())}",
                f"  executor:      {'set' if self.executor is not None else 'none'}",
                f"  default timeout: {self.default_timeout:.0f}s",
                f"  contexts:      {len(self.contexts.contexts)}",
            ]
        )


__all__ = [
    "DEFAULT_TIMEOUT_SECONDS",
    "AgentRequest",
    "AgentResponse",
    "AgentRuntime",
    "CancellationToken",
    "ExecutionDeferred",
    "TaskCancelled",
    "TaskExecutor",
    "TaskOutcome",
    "TaskTimedOut",
    "ValidationResult",
    "default_validator",
]
