"""The agent system: the seven personalities composed into one multi-agent layer.

This module is the integration point. It owns one of each component the task lists
-- the personality registry, the capability gate, the runtime, the context
manager, the goal manager, the memory access layer, the learning scope, the
message bus, the router, the collaboration coordinator, the conflict resolver, and
the monitor -- and wires them to the project's **existing** services rather than
rebuilding any of them.

It is deliberately small: every piece of behaviour lives in its own module, and
this facade only *composes* them, so a test can drive the whole system through one
object and a caller (the CLI, a GUI, the cognitive controller) gets one entry
point.

The default executor wraps a **mouth** -- the assistant the project already builds
for a personality (``ai.mouths.build_mouth``) -- so the agent layer reuses the one
model runtime and never loads a second copy. A test injects its own executor, so
the whole system runs with no model and no display.

Dependency-free and Tk-free at import: no model is loaded until a caller asks for
one, and nothing here reaches the network.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from core.agents.capabilities import (
    AgentCapability,
    CapabilityGateway,
    CapabilitySet,
)
from core.agents.collaboration import AgentCollaboration
from core.agents.communication import MessageBus
from core.agents.conflict import ConflictResolver
from core.agents.context import ContextManager, WorkingContext
from core.agents.goals import GoalManager, Task
from core.agents.learning_scope import AgentLearningScope
from core.agents.memory_access import PersonalityMemoryAccess
from core.agents.monitoring import AgentMonitor, MonitoringReport
from core.agents.registry import PersonalityProfile, PersonalityRegistry
from core.agents.routing import RoutingDecision, TaskRouter, TaskType
from core.agents.runtime import (
    AgentRequest,
    AgentResponse,
    AgentRuntime,
    TaskExecutor,
    TaskOutcome,
)


def mouth_executor(mouth: str, *, model_key: str | None = None, max_new_tokens: int = 256) -> TaskExecutor:
    """A :class:`TaskExecutor` that answers through a personality's **mouth**.

    The mouth is the assistant the project already builds for the personality
    (``ai.mouths.build_mouth``), so the agent layer reuses the one model runtime
    -- no second model, no second process. The executor polls the cancellation
    token *before* the call (a local model call cannot be preempted mid-way) and
    returns the mouth's text as a structured response. Importing this function
    loads nothing; the mouth is built on first use.
    """
    state: dict[str, Any] = {"assistant": None}

    def _assistant() -> Any:
        if state["assistant"] is None:
            from ai.mouths import build_mouth

            state["assistant"] = build_mouth(
                mouth, model_key=model_key, max_new_tokens=max_new_tokens
            )
        return state["assistant"]

    def execute(request: AgentRequest) -> AgentResponse:
        request.token.check()
        assistant = _assistant()
        answer = assistant.ask(request.question)
        text = getattr(answer, "text", None) or str(answer)
        return AgentResponse(ok=True, text=text, capabilities_used=(AgentCapability.ANSWER,))

    return execute


@dataclass
class AgentSystem:
    """The composed multi-agent system: one of each component, wired together.

    ``data_root`` keeps every durable path (memory, learning, configuration) under
    one folder, so a test runs the whole system in a temp directory and a portable
    build keeps its state beside its data. ``executor`` is the work; when it is
    ``None`` and ``mouth`` is named, the runtime builds a mouth executor lazily.
    """

    data_root: str | Path | None = None
    registry: PersonalityRegistry = field(default_factory=PersonalityRegistry)
    executor: TaskExecutor | None = None
    #: The mouth to answer through when no executor is injected (``None`` means the
    #: runtime has no executor until one is set).
    mouth: str = ""
    events: Any | None = None

    gateway: CapabilityGateway = field(default_factory=CapabilityGateway)
    contexts: ContextManager = field(default_factory=ContextManager)
    goals: GoalManager = field(default_factory=GoalManager)
    bus: MessageBus = field(default_factory=MessageBus)
    router: TaskRouter = field(init=False)
    runtime: AgentRuntime = field(init=False)
    collaboration: AgentCollaboration = field(init=False)
    conflicts: ConflictResolver = field(default_factory=ConflictResolver)
    monitor: AgentMonitor = field(init=False)

    def __post_init__(self) -> None:
        self.router = TaskRouter(registry=self.registry)
        executor = self.executor
        if executor is None and self.mouth:
            executor = mouth_executor(self.mouth)
        self.runtime = AgentRuntime(
            registry=self.registry,
            gateway=self.gateway,
            contexts=self.contexts,
            goals=self.goals,
            executor=executor,
            events=self.events,
        )
        self.collaboration = AgentCollaboration(runtime=self.runtime)
        self.monitor = AgentMonitor(
            registry=self.registry, goals=self.goals, gateway=self.gateway
        )

    # ------------------------------------------------------------ identity

    def profiles(self) -> tuple[PersonalityProfile, ...]:
        """Every personality's profile."""
        return self.registry.all_profiles()

    def profile(self, key: str) -> PersonalityProfile:
        """One personality's profile."""
        return self.registry.resolve(key)

    def capabilities(self, key: str) -> CapabilitySet:
        """One personality's effective capability set."""
        return self.registry.resolve(key).capabilities

    # ------------------------------------------------------------ memory

    def memory_access(self, key: str) -> PersonalityMemoryAccess:
        """The memory access layer **as** ``key`` (its own reader identity)."""
        return PersonalityMemoryAccess(key, data_root=self.data_root)

    def learning_scope(self, key: str) -> AgentLearningScope:
        """The learning scope of ``key``."""
        return AgentLearningScope(key, data_root=self.data_root)

    # ------------------------------------------------------------- running

    def run(
        self,
        owner: str,
        question: str,
        *,
        capability: AgentCapability = AgentCapability.ANSWER,
        timeout: float | None = None,
        **kwargs: Any,
    ) -> TaskOutcome:
        """Run one question through ``owner`` (a thin pass-through to the runtime)."""
        return self.runtime.run(owner, question, capability=capability, question=question, timeout=timeout, **kwargs)

    def route(
        self,
        question: str,
        *,
        conversation: str = "",
        task_type: TaskType | None = None,
        override: str = "",
        user_choice: str = "",
    ) -> RoutingDecision:
        """Route ``question`` to a personality and explain the decision."""
        return self.router.route(
            question,
            conversation=conversation,
            task_type=task_type,
            override=override,
            user_choice=user_choice,
        )

    def context(self, owner: str) -> WorkingContext:
        """One personality's independent working context."""
        return self.contexts.context_for(owner)

    def task(self, task_id: str) -> Task:
        """One task record by id."""
        return self.goals.get_task(task_id)

    # ------------------------------------------------------------ reporting

    def monitoring(self) -> MonitoringReport:
        """Every personality's measured activity."""
        return self.monitor.report()

    def status(self) -> str:
        """A readable status of the whole system."""
        return "\n".join(
            [
                self.runtime.describe(),
                self.router.describe() if hasattr(self.router, "describe") else "",
                self.collaboration.describe(),
                self.bus.describe(),
                self.monitoring().describe(),
            ]
        ).strip()


def default_system(
    *,
    data_root: str | Path | None = None,
    executor: TaskExecutor | None = None,
    mouth: str = "",
) -> AgentSystem:
    """Build an :class:`AgentSystem` (the convenience constructor)."""
    return AgentSystem(data_root=data_root, executor=executor, mouth=mouth)


__all__ = [
    "AgentSystem",
    "default_system",
    "mouth_executor",
]
