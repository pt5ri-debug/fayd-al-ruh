"""Adapts a learned ML policy to the core :class:`ActionPolicy` protocol.

The learned agent speaks in feature vectors and action indices; the agent loop
speaks in facts and action names. This adapter is the only place that bridges
the two, so ``core`` never imports numpy and the ML stack never learns about
``core`` decision internals.
"""

from __future__ import annotations

from core.action_policy import ActionPolicy

from ml.dataset import FeatureHasher
from ml.learner import QLearningAgent


class LearnedActionPolicy(ActionPolicy):
    """Wraps a trained :class:`QLearningAgent` as a name-based policy."""

    def __init__(
        self,
        agent: QLearningAgent,
        action_names: tuple[str, ...],
        hasher: FeatureHasher | None = None,
    ) -> None:
        if len(action_names) != agent.num_actions:
            raise ValueError(
                "action_names must match the agent's action count "
                f"({len(action_names)} != {agent.num_actions})."
            )
        if hasher is not None and hasher.size != agent.num_features:
            raise ValueError("hasher size must match the agent's feature count.")
        self.agent = agent
        self.action_names = action_names
        self.hasher = hasher or FeatureHasher(agent.num_features)

    def select(self, facts: frozenset[str], actions: tuple[str, ...]) -> str | None:
        available = [name for name in self.action_names if name in actions]
        if not available:
            return None
        features = self.hasher.encode(tuple(sorted(facts)) or ("<empty>",))
        values = self.agent.q_values(features)
        best = max(available, key=lambda name: float(values[self.action_names.index(name)]))
        return best

    @classmethod
    def from_task(
        cls, task, agent: QLearningAgent, hasher: FeatureHasher | None = None
    ) -> LearnedActionPolicy:
        return cls(agent, tuple(spec.name for spec in task.allowed_actions), hasher)


def rank_actions(
    agent: QLearningAgent, hasher: FeatureHasher, facts: frozenset[str], action_names: tuple[str, ...]
) -> list[tuple[str, float]]:
    """Return actions ordered by learned value, highest first (for explanations)."""
    features = hasher.encode(tuple(sorted(facts)) or ("<empty>",))
    values = agent.q_values(features)
    return sorted(
        ((name, float(values[index])) for index, name in enumerate(action_names)),
        key=lambda item: (-item[1], item[0]),
    )


__all__ = ["LearnedActionPolicy", "rank_actions"]
