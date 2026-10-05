"""A symbolic world model learned from observed transitions.

The sandbox planner in :mod:`core.planner` uses :class:`core.action_model
.ActionModel`, which reads the transition declared by each action. That is a
*hand-written* model. This module learns the same transition function from
observed ``(state, action, next_state)`` transitions instead, so the agent can
predict the effect of an action it has seen rather than the one it was told.

An action's *preconditions* are known (the environment enforces them, so they
are part of how an action is invoked). Its *effects* are what must be learned.
Each observation is therefore keyed by the action name and the set of its
preconditions that held, which keeps the learned model contextual without
fragmenting it across irrelevant facts. Unchanged steps are recorded too, so the
model learns that an action does nothing where it does nothing, and it reports
an explicit ``unknown`` prediction for a pair it has never seen.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field

from core.action import Action
from core.sandbox_environment import SandboxEnvironment, SandboxTask
from core.state import State


@dataclass
class _EffectStats:
    add: dict[str, int] = field(default_factory=dict)
    remove: dict[str, int] = field(default_factory=dict)
    count: int = 0

    def add_effect(self, added: frozenset[str], removed: frozenset[str]) -> None:
        for fact in added:
            self.add[fact] = self.add.get(fact, 0) + 1
        for fact in removed:
            self.remove[fact] = self.remove.get(fact, 0) + 1
        self.count += 1

    def transition(self, threshold: float) -> tuple[frozenset[str], frozenset[str]]:
        added = frozenset(fact for fact, hits in self.add.items() if hits / self.count >= threshold)
        removed = frozenset(
            fact for fact, hits in self.remove.items() if hits / self.count >= threshold
        )
        return added, removed


@dataclass(frozen=True)
class LearnedTransition:
    """The learned effect of one action in one situation."""

    added: frozenset[str]
    removed: frozenset[str]
    samples: int
    confidence: float

    @property
    def known(self) -> bool:
        return self.samples > 0


def _condition(action: Action, state: State) -> frozenset[str]:
    """Which of the action's known preconditions held when it ran."""
    return frozenset(fact for fact in action.preconditions if fact in state.facts)


class SymbolicWorldModel:
    """Learns action effects from observed transitions.

    Each observation is stored twice: under the exact state it was seen in, and
    under a coarser ``(action, satisfied preconditions)`` key. Prediction tries
    the exact state first and falls back to the coarse key, then to an explicit
    unknown, so a planner never imagines an effect that was never seen.
    """

    def __init__(self, *, threshold: float = 0.5) -> None:
        if not 0.0 < threshold <= 1.0:
            raise ValueError("threshold must be in (0, 1].")
        self.threshold = threshold
        self._exact: dict[tuple[str, frozenset[str]], _EffectStats] = {}
        self._general: dict[tuple[str, frozenset[str]], _EffectStats] = {}
        self.transitions_observed = 0

    def observe(self, state: State, action: Action, next_state: State) -> None:
        """Record one observed transition, including no-ops."""
        added = next_state.facts - state.facts
        removed = state.facts - next_state.facts
        for table, key in (
            (self._exact, (action.name, frozenset(state.facts))),
            (self._general, (action.name, _condition(action, state))),
        ):
            table.setdefault(key, _EffectStats()).add_effect(added, removed)
        self.transitions_observed += 1

    def observe_environment(
        self, task: SandboxTask, *, attempts: int = 200, seed: int = 0
    ) -> int:
        """Explore a sandbox task and observe the transitions it produces.

        This is the agent's interaction with the environment: it tries actions,
        observes what happened, and stores it. Only the states before and after
        are read, never the action's declared effects.
        """
        rng = random.Random(seed)
        observed = 0
        for _ in range(attempts):
            env = SandboxEnvironment(task)
            for _ in range(task.max_steps):
                if env.terminated != "running":
                    break
                spec = rng.choice(task.allowed_actions)
                action = spec.to_action()
                before = env.state
                try:
                    env.step(action)
                except RuntimeError:
                    break
                self.observe(before, action, env.state)
                observed += 1
        return observed

    def predict(self, state: State, action: Action) -> LearnedTransition:
        """Predict the effect of ``action`` in ``state`` from observed transitions."""
        keys = [
            (self._exact, (action.name, frozenset(state.facts))),
            (self._general, (action.name, _condition(action, state))),
            (self._general, (action.name, frozenset())),
        ]
        for table, key in keys:
            stats = table.get(key)
            if stats is None or stats.count == 0:
                continue
            added, removed = stats.transition(self.threshold)
            confidence = max(
                (hits / stats.count for hits in stats.add.values()), default=1.0
            )
            return LearnedTransition(added, removed, stats.count, round(confidence, 6))
        return LearnedTransition(frozenset(), frozenset(), 0, 0.0)

    def apply(self, state: State, action: Action) -> State:
        """Apply the learned effect. Unknown actions leave the state unchanged."""
        learned = self.predict(state, action)
        return State((state.facts | learned.added) - learned.removed)

    def is_known(self, state: State, action: Action) -> bool:
        """Whether the model has seen this action do something in this situation."""
        return self.predict(state, action).known

    @property
    def coverage(self) -> int:
        """Number of distinct (situation, action) pairs the model has learned."""
        return len(self._exact) + len(self._general)


def explore_task(task: SandboxTask, *, attempts: int = 200, seed: int = 0) -> SymbolicWorldModel:
    """Build a world model by exploring a sandbox task and observing transitions.

    The sandbox is stepped for real; the model only records the states before and
    after each action, so it reconstructs the dynamics rather than copying them.
    """
    model = SymbolicWorldModel()
    model.observe_environment(task, attempts=attempts, seed=seed)
    return model


__all__ = [
    "LearnedTransition",
    "SymbolicWorldModel",
    "explore_task",
]
