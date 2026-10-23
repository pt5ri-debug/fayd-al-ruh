"""Replayable symbolic simulation: the first, fully deterministic environment tier.

A simulation records the exact action sequence it executed. The trace can be
serialized and later replayed to reconstruct the final state bit-for-bit, which
is what makes this tier safe to run and easy to debug: no shell, no network, no
filesystem, only in-memory state transitions.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from core.action import Action
from core.state import State
from core.state_transition import StateTransition


@dataclass(frozen=True)
class ActionSpec:
    """A serializable action, used to rebuild actions during replay."""

    name: str
    preconditions: frozenset[str] = frozenset()
    add_effects: frozenset[str] = frozenset()
    remove_effects: frozenset[str] = frozenset()
    cost: float = 1.0

    @classmethod
    def from_action(cls, action: Action) -> ActionSpec:
        return cls(
            action.name,
            action.preconditions,
            action.add_effects,
            action.remove_effects,
            action.cost,
        )

    def to_action(self) -> Action:
        return Action(self.name, self.preconditions, self.add_effects, self.remove_effects, self.cost)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "preconditions": sorted(self.preconditions),
            "add_effects": sorted(self.add_effects),
            "remove_effects": sorted(self.remove_effects),
            "cost": self.cost,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ActionSpec:
        return cls(
            str(data["name"]),
            frozenset(data.get("preconditions", [])),
            frozenset(data.get("add_effects", [])),
            frozenset(data.get("remove_effects", [])),
            float(data.get("cost", 1.0)),
        )


@dataclass(frozen=True)
class StepRecord:
    """One executed action and the state it produced."""

    index: int
    action: ActionSpec
    resulting_facts: frozenset[str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "action": self.action.to_dict(),
            "resulting_facts": sorted(self.resulting_facts),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> StepRecord:
        return cls(
            int(data["index"]),
            ActionSpec.from_dict(data["action"]),
            frozenset(data.get("resulting_facts", [])),
        )


@dataclass(frozen=True)
class SimulationTrace:
    """An ordered, serializable record of a symbolic simulation."""

    initial_facts: frozenset[str]
    steps: tuple[StepRecord, ...]
    max_steps: int
    final_facts: frozenset[str]
    terminated: str

    @property
    def step_count(self) -> int:
        return len(self.steps)

    def to_dict(self) -> dict[str, Any]:
        return {
            "initial_facts": sorted(self.initial_facts),
            "steps": [step.to_dict() for step in self.steps],
            "max_steps": self.max_steps,
            "final_facts": sorted(self.final_facts),
            "terminated": self.terminated,
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, sort_keys=True)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SimulationTrace:
        return cls(
            initial_facts=frozenset(data.get("initial_facts", [])),
            steps=tuple(StepRecord.from_dict(step) for step in data.get("steps", [])),
            max_steps=int(data.get("max_steps", 0)),
            final_facts=frozenset(data.get("final_facts", [])),
            terminated=str(data.get("terminated", "unknown")),
        )

    @classmethod
    def from_json(cls, payload: str) -> SimulationTrace:
        return cls.from_dict(json.loads(payload))


@dataclass
class SimulationResult:
    trace: SimulationTrace
    final_state: State
    reward: float = 0.0
    violations: tuple[str, ...] = field(default_factory=tuple)


class SymbolicSimulation:
    """Deterministic, replayable symbolic environment.

    Actions are applied through the pure ``StateTransition`` function, so the
    same initial state and action sequence always yield the same trace.
    """

    def __init__(
        self,
        initial_facts: frozenset[str] | None = None,
        max_steps: int = 32,
        transition: StateTransition | None = None,
    ) -> None:
        if max_steps < 1:
            raise ValueError("max_steps must be at least 1.")
        self.initial_facts = initial_facts or frozenset()
        self.state = State(self.initial_facts)
        self.max_steps = max_steps
        self.transition = transition or StateTransition()
        self._steps: list[StepRecord] = []
        self._violations: list[str] = []
        self._terminated = "running"

    @property
    def step_count(self) -> int:
        return len(self._steps)

    @property
    def terminated(self) -> str:
        return self._terminated

    def apply(self, action: Action) -> State:
        """Apply one action, recording it. Raises if the environment is closed."""
        if self._terminated != "running":
            raise RuntimeError(f"Simulation already terminated: {self._terminated}")
        if len(self._steps) >= self.max_steps:
            self._terminated = "step_limit"
            raise RuntimeError("Simulation reached its step limit.")
        try:
            self.state = self.transition.apply(self.state, action)
        except ValueError as error:
            self._violations.append(f"{action.name}: {error}")
            raise
        self._steps.append(StepRecord(len(self._steps), ActionSpec.from_action(action), self.state.facts))
        return self.state

    def run(self, actions: list[Action]) -> SimulationResult:
        for action in actions:
            if self._terminated != "running":
                break
            try:
                self.apply(action)
            except (ValueError, RuntimeError):
                break
        if self._terminated == "running":
            self._terminated = "completed"
        return self.result()

    def result(self) -> SimulationResult:
        return SimulationResult(
            trace=SimulationTrace(
                initial_facts=self.initial_facts,
                steps=tuple(self._steps),
                max_steps=self.max_steps,
                final_facts=self.state.facts,
                terminated=self._terminated,
            ),
            final_state=self.state,
            violations=tuple(self._violations),
        )

    @staticmethod
    def replay(trace: SimulationTrace, transition: StateTransition | None = None) -> SimulationResult:
        """Rebuild the final state from a trace without re-deriving actions."""
        engine = transition or StateTransition()
        state = State(trace.initial_facts)
        for step in trace.steps:
            state = engine.apply(state, step.action.to_action())
        return SimulationResult(
            trace=SimulationTrace(
                initial_facts=trace.initial_facts,
                steps=trace.steps,
                max_steps=trace.max_steps,
                final_facts=state.facts,
                terminated=trace.terminated,
            ),
            final_state=state,
        )
