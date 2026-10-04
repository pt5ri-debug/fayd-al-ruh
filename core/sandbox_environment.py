"""Task-scoped sandbox environment: the second, richer environment tier.

A ``SandboxTask`` declares an explicit allow-list of actions and a success
condition. The ``SandboxEnvironment`` only executes actions that belong to the
task, rejects anything else as a violation, and stops at a step budget. It is
still purely symbolic: no external tool is executed here.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from core.action import Action
from core.state import State
from core.state_transition import StateTransition
from core.symbolic_simulation import (
    ActionSpec,
    SimulationTrace,
    StepRecord,
)


@dataclass(frozen=True)
class SandboxTask:
    """A clearly-specified task: allowed actions plus a success predicate."""

    name: str
    description: str
    allowed_actions: tuple[ActionSpec, ...]
    initial_facts: frozenset[str] = frozenset()
    max_steps: int = 32

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("Sandbox task name cannot be empty.")
        if not self.description.strip():
            raise ValueError("Sandbox task description cannot be empty.")
        if not self.allowed_actions:
            raise ValueError("Sandbox task needs at least one allowed action.")
        if self.max_steps < 1:
            raise ValueError("Sandbox task max_steps must be at least 1.")
        names = [spec.name for spec in self.allowed_actions]
        if len(names) != len(set(names)):
            raise ValueError("Sandbox task action names must be unique.")

    def allowed_action(self, name: str) -> Action | None:
        for spec in self.allowed_actions:
            if spec.name == name:
                return spec.to_action()
        return None

    @property
    def allowed_action_names(self) -> frozenset[str]:
        return frozenset(spec.name for spec in self.allowed_actions)


@dataclass(frozen=True)
class SandboxStep:
    index: int
    action: ActionSpec
    accepted: bool
    resulting_facts: frozenset[str]
    reason: str = ""


@dataclass
class SandboxResult:
    task_name: str
    success: bool
    terminated: str
    steps: tuple[SandboxStep, ...]
    violations: tuple[str, ...]
    final_state: State
    reward: float = 0.0

    @property
    def accepted_steps(self) -> int:
        return sum(1 for step in self.steps if step.accepted)


class SandboxEnvironment:
    """Runs one explicit task under an allow-list and a step budget."""

    def __init__(self, task: SandboxTask, transition: StateTransition | None = None) -> None:
        self.task = task
        self.transition = transition or StateTransition()
        self.state = State(task.initial_facts)
        self._steps: list[SandboxStep] = []
        self._violations: list[str] = []
        self._terminated = "running"

    @property
    def terminated(self) -> str:
        return self._terminated

    @property
    def step_count(self) -> int:
        return len(self._steps)

    def _close(self, reason: str) -> None:
        if self._terminated == "running":
            self._terminated = reason

    def step(self, action: Action) -> SandboxStep:
        """Attempt one action; disallowed actions are recorded, not executed."""
        if self._terminated != "running":
            raise RuntimeError(f"Sandbox already terminated: {self._terminated}")
        if len(self._steps) >= self.task.max_steps:
            self._close("step_limit")
            raise RuntimeError("Sandbox reached its step limit.")

        allowed = self.task.allowed_action(action.name)
        if allowed is None:
            reason = f"Action '{action.name}' is not allowed by task '{self.task.name}'."
            self._violations.append(reason)
            step = SandboxStep(len(self._steps), ActionSpec.from_action(action), False, self.state.facts, reason)
            self._steps.append(step)
            return step

        try:
            self.state = self.transition.apply(self.state, allowed)
        except ValueError as error:
            reason = f"{action.name}: {error}"
            self._violations.append(reason)
            step = SandboxStep(len(self._steps), ActionSpec.from_action(action), False, self.state.facts, reason)
            self._steps.append(step)
            return step

        step = SandboxStep(len(self._steps), ActionSpec.from_action(allowed), True, self.state.facts)
        self._steps.append(step)
        return step

    def run(self, actions: list[Action], success: Callable[[State], bool]) -> SandboxResult:
        for action in actions:
            if self._terminated != "running":
                break
            try:
                step = self.step(action)
            except RuntimeError:
                break
            if step.accepted and success(self.state):
                self._close("success")
                break
        if self._terminated == "running":
            self._close("completed")
        return self.result(success)

    def result(self, success: Callable[[State], bool]) -> SandboxResult:
        achieved = bool(success(self.state))
        return SandboxResult(
            task_name=self.task.name,
            success=achieved,
            terminated=self._terminated,
            steps=tuple(self._steps),
            violations=tuple(self._violations),
            final_state=self.state,
            reward=1.0 if achieved else 0.0,
        )

    def to_trace(self) -> SimulationTrace:
        """Export accepted steps as a replayable symbolic trace."""
        accepted = [step for step in self._steps if step.accepted]
        records = tuple(
            StepRecord(step.index, step.action, step.resulting_facts) for step in accepted
        )
        return SimulationTrace(
            initial_facts=self.task.initial_facts,
            steps=records,
            max_steps=self.task.max_steps,
            final_facts=self.state.facts,
            terminated=self._terminated,
        )
