"""Controlled collaboration between personalities: bounded, explicit, provable.

The task allows personalities to collaborate -- to delegate a task to another and
combine the results -- but only under strict control: an explicit delegation, a
structured request/response, a shared task id, a delegation **depth** limit, a
timeout, cancellation, result aggregation, conflict detection, provenance
tracking, clear ownership, and no uncontrolled agent-to-agent loops.

This module is that control. It is a thin coordinator over
:class:`core.agents.runtime.AgentRuntime`: a delegation *is* a task the runtime
runs, tagged with the delegator, the depth, and the parent task, so the task tree
is inspectable and a loop is detectable. The two guards the task asks for are
enforced here and in the goal manager: a **depth** bound and a **delegation
budget** per top-level task.

Delegated agents receive only the context their task needs (the question and the
caller's payload) -- never the delegator's private memory, which the memory access
layer would refuse anyway.

Dependency-free and Tk-free: it holds no model and touches no network.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from core.agents.runtime import AgentResponse, AgentRuntime, TaskOutcome
from core.learning_isolation import CHARACTER_OWNERS, owner_for

if TYPE_CHECKING:
    from core.agents.capabilities import AgentCapability


class DelegationRefused(Exception):
    """Raised when a delegation is refused (depth, budget, or permission)."""


@dataclass(frozen=True)
class DelegationRequest:
    """A structured request from one personality to another.

    ``depth`` is the delegator's own depth plus one; ``root_task`` is the
    top-level task the whole tree descends from, so the budget is counted per tree
    rather than per agent. ``shared_id`` names the collaboration so a result can be
    matched back to its request.
    """

    shared_id: str
    from_owner: str
    to_owner: str
    question: str
    depth: int = 1
    root_task: str = ""
    payload: dict[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class DelegationResult:
    """The structured response to a :class:`DelegationRequest`."""

    shared_id: str
    from_owner: str
    to_owner: str
    ok: bool
    text: str = ""
    error: str = ""
    task_id: str = ""
    depth: int = 0
    duration_seconds: float = 0.0

    def describe(self) -> str:
        verdict = "ok" if self.ok else f"failed: {self.error}"
        return f"{self.from_owner} -> {self.to_owner} [{self.shared_id}] {verdict}: {self.text[:120]}"


@dataclass(frozen=True)
class CollaborationReport:
    """The aggregated result of a set of delegations under one shared id."""

    shared_id: str
    results: tuple[DelegationResult, ...]

    @property
    def ok(self) -> bool:
        """True when at least one delegate succeeded."""
        return any(result.ok for result in self.results)

    @property
    def unanimous(self) -> bool:
        """True when every delegate that answered produced the same text."""
        texts = {result.text.strip() for result in self.results if result.ok}
        return len(texts) == 1

    def disagreements(self) -> tuple[tuple[str, str], ...]:
        """The distinct answers, so a caller can see where delegates differ."""
        seen: list[tuple[str, str]] = []
        for result in self.results:
            if result.ok and result.text.strip():
                if not any(text == result.text.strip() for _owner, text in seen):
                    seen.append((result.to_owner, result.text.strip()))
        return tuple(seen)

    def describe(self) -> str:
        head = f"collaboration {self.shared_id}: {len(self.results)} delegate(s), ok={self.ok}"
        if len(self.disagreements()) > 1:
            head += " (delegates disagree)"
        return "\n".join([head, *(result.describe() for result in self.results)])


@dataclass
class AgentCollaboration:
    """The one controlled path for a personality to delegate to another.

    ``max_depth`` bounds the delegation chain and ``max_delegations`` bounds how
    many delegations one top-level task may spawn (the loop guard). ``runtime``
    does the actual work, so a delegation is a real task with a real record. The
    collaboration is **optional and bounded**: nothing here runs unless a caller
    asks for it.
    """

    runtime: AgentRuntime = field(default_factory=AgentRuntime)
    max_depth: int = 3
    max_delegations: int = 8
    #: Per top-level task, how many delegations have been issued (the budget).
    _budget: dict[str, int] = field(default_factory=dict, repr=False)
    _counter: int = field(default=0, repr=False)

    def _next_id(self) -> str:
        self._counter += 1
        return f"collab-{self._counter:04d}"

    def delegate(self, request: DelegationRequest) -> DelegationResult:
        """Run one delegation, enforcing depth, budget, and the capability gate.

        Refuses before touching the runtime when the depth would exceed the bound
        or the tree's delegation budget is spent -- so an uncontrolled loop is
        stopped by construction, not by a timeout.
        """
        from_owner = owner_for(request.from_owner)
        to_owner = owner_for(request.to_owner)
        if to_owner not in CHARACTER_OWNERS:
            raise DelegationRefused(f"{request.to_owner!r} is not a delegate personality.")
        if request.depth > self.max_depth:
            raise DelegationRefused(
                f"delegation depth {request.depth} exceeds the bound of {self.max_depth}."
            )
        root = request.root_task or request.shared_id
        used = self._budget.get(root, 0)
        if used >= self.max_delegations:
            raise DelegationRefused(
                f"the delegation budget for {root} ({self.max_delegations}) is spent."
            )
        self._budget[root] = used + 1

        outcome: TaskOutcome = self.runtime.run(
            to_owner,
            request.question,
            capability=_answer_capability(),
            question=request.question,
            payload={"delegated_by": from_owner, **request.payload},
            delegated_by=from_owner,
            depth=request.depth,
            parent_task=root if request.depth > 1 else "",
        )
        response: AgentResponse = outcome.response
        return DelegationResult(
            shared_id=request.shared_id,
            from_owner=from_owner,
            to_owner=to_owner,
            ok=outcome.ok,
            text=response.text if response.ok else "",
            error=response.error,
            task_id=outcome.task.task_id,
            depth=request.depth,
            duration_seconds=outcome.duration_seconds,
        )

    def collaborate(
        self,
        from_owner: str,
        question: str,
        targets: tuple[str, ...],
        *,
        shared_id: str = "",
        payload: dict[str, object] | None = None,
    ) -> CollaborationReport:
        """Ask several personalities the same question and aggregate the answers.

        Every delegate gets the same structured request and the same shared id;
        the report keeps each answer separate (the conflict layer compares them),
        so aggregation never hides a disagreement.
        """
        resolved_from = owner_for(from_owner)
        collaboration_id = shared_id or self._next_id()
        results: list[DelegationResult] = []
        for target in targets:
            request = DelegationRequest(
                shared_id=collaboration_id,
                from_owner=resolved_from,
                to_owner=owner_for(target),
                question=question,
                depth=1,
                root_task=collaboration_id,
                payload=dict(payload or {}),
            )
            results.append(self.delegate(request))
        return CollaborationReport(shared_id=collaboration_id, results=tuple(results))

    def budget_used(self, root_task: str) -> int:
        """How many delegations the tree ``root_task`` has spent."""
        return self._budget.get(root_task, 0)

    def reset_budget(self, root_task: str) -> None:
        """Forget a tree's budget (a new top-level task)."""
        self._budget.pop(root_task, None)

    def describe(self) -> str:
        return "\n".join(
            [
                "Agent collaboration",
                f"  max depth:       {self.max_depth}",
                f"  delegation budget: {self.max_delegations} per tree",
                f"  active trees:    {len(self._budget)}",
            ]
        )


def _answer_capability() -> AgentCapability:
    """The capability a delegation needs (imported lazily to avoid a cycle)."""
    from core.agents.capabilities import AgentCapability

    return AgentCapability.ANSWER


__all__ = [
    "AgentCollaboration",
    "CollaborationReport",
    "DelegationRefused",
    "DelegationRequest",
    "DelegationResult",
]
