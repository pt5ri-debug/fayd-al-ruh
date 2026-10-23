"""Reinforcement learning driven by observed interaction reward.

The sandbox learner in :mod:`ml.learner` is rewarded by a hand-written success
predicate. Here the reward comes from :class:`core.interaction_reward
.InteractionRewardModel`, which is learned from recorded outcomes: which action
actually worked in which context, as observed and recorded, not as declared.

The environment exposed here has the same reset/step shape as
``SandboxPolicyEnv``, so the existing :class:`ml.learner.QLearningAgent` trains
unchanged. Only the reward source differs. That is the whole point: swap the
proxy for the observed outcome and keep the rest of the learning stack intact.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from core.action import Action
from core.interaction_reward import InteractionRewardModel, Outcome
from core.progress import ProgressCallback
from core.sandbox_environment import SandboxEnvironment, SandboxTask
from core.state import State
from core.state_transition import StateTransition

from ml.dataset import FeatureHasher
from ml.learner import QLearningAgent, state_tokens


def recorded_action_outcomes(task: SandboxTask, *, source: str = "recorded-interaction") -> list[Outcome]:
    """A recorded interaction log: whether each action actually changes the world.

    Every allowed action is observed in a state where its preconditions hold
    (its own applicable state, built from the task's initial facts). The recorded
    result is whether the action changed the state at all -- a direct observation
    that needs no goal predicate. An action that does nothing where it can run is
    recorded as ineffective, so the reward model learns which actions are
    productive without ever reading the task's success condition.
    """
    transition = StateTransition()
    context = task.name
    initial = State(task.initial_facts)
    outcomes: list[Outcome] = []
    for spec in task.allowed_actions:
        action = Action(
            spec.name, spec.preconditions, spec.add_effects, spec.remove_effects, spec.cost
        )
        applicable = initial if initial.satisfies(spec.preconditions) else State(
            initial.facts | spec.preconditions
        )
        try:
            resulting = transition.apply(applicable, action)
            observed = 1.0 if resulting.facts != applicable.facts else 0.0
        except ValueError:
            observed = 0.0
        outcomes.append(
            Outcome(
                context=context,
                action=spec.name,
                predicted_confidence=None,
                observed=observed,
                source=source,
            )
        )
    return outcomes


class InteractionRewardEnv:
    """Reset/step RL environment whose reward is learned from observed outcomes."""

    def __init__(
        self,
        task: SandboxTask,
        reward_model: InteractionRewardModel,
        *,
        hasher: FeatureHasher,
        goal: Callable[[State], bool],
        step_penalty: float = -0.02,
        success_bonus: float = 0.0,
    ) -> None:
        self.task = task
        self.reward_model = reward_model
        self.hasher = hasher
        self.goal = goal
        self.step_penalty = step_penalty
        self.success_bonus = success_bonus
        self.context = task.name
        self._env = SandboxEnvironment(task)

    @property
    def num_actions(self) -> int:
        return len(self.task.allowed_actions)

    def _features(self) -> np.ndarray:
        return self.hasher.encode(state_tokens(self._env.state))

    def reset(self) -> np.ndarray:
        self._env = SandboxEnvironment(self.task)
        return self._features()

    def step(self, action_index: int) -> tuple[np.ndarray, float, bool, bool]:
        if not 0 <= action_index < self.num_actions:
            raise ValueError(f"Action index out of range: {action_index}")
        spec = self.task.allowed_actions[action_index]
        action = Action(
            spec.name, spec.preconditions, spec.add_effects, spec.remove_effects, spec.cost
        )
        before = self._env.state.facts
        try:
            step = self._env.step(action)
        except RuntimeError:
            return self._features(), self.step_penalty, True, False

        changed = self._env.state.facts != before
        achieved = self.goal(self._env.state)
        done = achieved or self._env.terminated != "running" or self._env.step_count >= self.task.max_steps
        # The reward is the observed success rate of this action in this
        # context, learned from interaction outcomes -- not the task predicate.
        # It is credited only when the action actually changed the world, which
        # is itself a direct observation and needs no goal predicate.
        reward = self.reward_model.action_reward(self.context, spec.name) if changed else 0.0
        reward += self.step_penalty
        if not step.accepted:
            reward += self.step_penalty
        if achieved:
            reward += self.success_bonus
        return self._features(), reward, done, achieved


@dataclass(frozen=True)
class InteractionReport:
    task_name: str
    episodes: int
    success_rate: float
    final_epsilon: float
    learned_action_values: dict[str, float]


def train_interaction_policy(
    task: SandboxTask,
    reward_model: InteractionRewardModel,
    goal: Callable[[State], bool],
    *,
    episodes: int = 300,
    hasher: FeatureHasher | None = None,
    seed: int = 0,
    agent: QLearningAgent | None = None,
    progress: ProgressCallback | None = None,
) -> tuple[QLearningAgent, InteractionReport]:
    """Train a Q-learning agent whose reward comes from observed interaction."""
    hasher = hasher or FeatureHasher(64)
    env = InteractionRewardEnv(task, reward_model, hasher=hasher, goal=goal)
    learner = agent or QLearningAgent(hasher.size, env.num_actions, seed=seed)

    successes = 0
    window: list[bool] = []
    for episode in range(1, episodes + 1):
        features = env.reset()
        done = False
        while not done:
            action = learner.act(features)
            next_features, reward, done, achieved = env.step(action)
            learner.update(features, action, reward, next_features, done)
            features = next_features
        learner.decay_epsilon()
        successes += 1 if achieved else 0
        window.append(achieved)
        if progress is not None:
            progress(
                episode,
                episodes,
                detail=f"success {successes / episode:.2f}  eps {learner.epsilon:.3f}",
            )

    recent = window[-max(1, episodes // 5) :]
    success_rate = sum(recent) / len(recent)
    action_names = tuple(spec.name for spec in task.allowed_actions)
    values = learner.q_values(hasher.encode(state_tokens(State(task.initial_facts))))
    learned = {name: round(float(values[index]), 6) for index, name in enumerate(action_names)}
    return learner, InteractionReport(
        task_name=task.name,
        episodes=episodes,
        success_rate=round(success_rate, 6),
        final_epsilon=learner.epsilon,
        learned_action_values=learned,
    )


def evaluate_interaction_policy(
    task: SandboxTask,
    reward_model: InteractionRewardModel,
    goal: Callable[[State], bool],
    agent: QLearningAgent,
    *,
    episodes: int = 20,
    hasher: FeatureHasher | None = None,
) -> float:
    """Greedy success rate of an interaction-trained policy, on the real goal."""
    hasher = hasher or FeatureHasher(64)
    env = InteractionRewardEnv(task, reward_model, hasher=hasher, goal=goal)
    hits = 0
    for _ in range(episodes):
        features = env.reset()
        done = False
        achieved = False
        while not done:
            action = agent.act(features, greedy=True)
            features, _reward, done, achieved = env.step(action)
        hits += 1 if achieved else 0
    return hits / episodes if episodes else 0.0


__all__ = [
    "InteractionReport",
    "InteractionRewardEnv",
    "evaluate_interaction_policy",
    "recorded_action_outcomes",
    "train_interaction_policy",
]
