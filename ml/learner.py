"""Symbolic reinforcement learning over the sandbox tasks (tier 1 of the ladder).

A linear Q-function is trained with temporal-difference updates on top of the
task-scoped sandboxes. This is real reinforcement learning with function
approximation, but it stays entirely inside the safe, replayable symbolic
environments -- no external tool is touched.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np

from core.action import Action
from core.sandbox_environment import SandboxEnvironment, SandboxTask
from core.state import State
from ml.dataset import FeatureHasher


def state_tokens(state: State) -> tuple[str, ...]:
    return tuple(sorted(state.facts)) or ("<empty>",)


class SandboxPolicyEnv:
    """Adapts a SandboxTask to a reset/step RL interface over action indices."""

    def __init__(
        self,
        task: SandboxTask,
        success: Callable[[State], bool],
        hasher: FeatureHasher,
        *,
        success_reward: float = 1.0,
        step_penalty: float = -0.02,
    ) -> None:
        self.task = task
        self.success = success
        self.hasher = hasher
        self.success_reward = success_reward
        self.step_penalty = step_penalty
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
        try:
            step = self._env.step(action)
        except RuntimeError:
            return self._features(), self.step_penalty, True, False
        achieved = self.success(self._env.state)
        done = achieved or self._env.terminated != "running" or self._env.step_count >= self.task.max_steps
        reward = self.success_reward if achieved else self.step_penalty
        if not step.accepted:
            reward += self.step_penalty
        return self._features(), reward, done, achieved


@dataclass
class EpisodeStats:
    episode: int
    total_reward: float
    steps: int
    success: bool
    epsilon: float


@dataclass
class PolicyReport:
    episodes: tuple[EpisodeStats, ...]
    success_rate_window: float
    final_epsilon: float
    learned_actions: dict[str, str] = field(default_factory=dict)

    @property
    def success_rate(self) -> float:
        if not self.episodes:
            return 0.0
        return sum(1 for episode in self.episodes if episode.success) / len(self.episodes)

    @property
    def learning_curve(self) -> list[float]:
        """Rolling success rate, useful for plotting or regression checks."""
        curve: list[float] = []
        hits = 0
        for index, episode in enumerate(self.episodes, start=1):
            hits += 1 if episode.success else 0
            curve.append(hits / index)
        return curve


class QLearningAgent:
    """Epsilon-greedy linear Q-learning agent."""

    def __init__(
        self,
        num_features: int,
        num_actions: int,
        *,
        alpha: float = 0.2,
        gamma: float = 0.95,
        epsilon: float = 1.0,
        epsilon_decay: float = 0.97,
        epsilon_min: float = 0.05,
        seed: int = 0,
    ) -> None:
        if min(num_features, num_actions) < 1:
            raise ValueError("Q-learning needs positive feature and action counts.")
        self.num_features = num_features
        self.num_actions = num_actions
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.W = np.zeros((num_features, num_actions), dtype=np.float64)
        self._rng = np.random.default_rng(seed)

    def q_values(self, features: np.ndarray) -> np.ndarray:
        return features @ self.W

    def act(self, features: np.ndarray, *, greedy: bool = False) -> int:
        if not greedy and self._rng.random() < self.epsilon:
            return int(self._rng.integers(self.num_actions))
        return int(np.argmax(self.q_values(features)))

    def update(
        self,
        features: np.ndarray,
        action: int,
        reward: float,
        next_features: np.ndarray,
        done: bool,
    ) -> float:
        prediction = float(self.q_values(features)[action])
        bootstrap = 0.0 if done else float(np.max(self.q_values(next_features)))
        target = reward + self.gamma * bootstrap
        error = target - prediction
        self.W[:, action] += self.alpha * error * features
        return error

    def decay_epsilon(self) -> None:
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def save(self, path: str | Path) -> Path:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            target,
            W=self.W,
            epsilon=np.array(self.epsilon),
            alpha=np.array(self.alpha),
            gamma=np.array(self.gamma),
        )
        return target

    @classmethod
    def load(cls, path: str | Path) -> "QLearningAgent":
        data = np.load(Path(path), allow_pickle=False)
        weights = data["W"]
        agent = cls(
            int(weights.shape[0]),
            int(weights.shape[1]),
            alpha=float(data["alpha"]),
            gamma=float(data["gamma"]),
            epsilon=float(data["epsilon"]),
        )
        agent.W = weights
        return agent


def train_policy(
    task: SandboxTask,
    success: Callable[[State], bool],
    *,
    episodes: int = 300,
    hasher: FeatureHasher | None = None,
    agent: QLearningAgent | None = None,
    seed: int = 0,
) -> tuple[QLearningAgent, PolicyReport]:
    """Train an epsilon-greedy agent on one sandbox task."""
    if episodes < 1:
        raise ValueError("episodes must be at least 1.")
    hasher = hasher or FeatureHasher(64)
    env = SandboxPolicyEnv(task, success, hasher)
    agent = agent or QLearningAgent(hasher.size, env.num_actions, seed=seed)
    history: list[EpisodeStats] = []

    for episode_index in range(1, episodes + 1):
        features = env.reset()
        total_reward = 0.0
        steps = 0
        achieved = False
        while True:
            action = agent.act(features)
            next_features, reward, done, achieved = env.step(action)
            agent.update(features, action, reward, next_features, done)
            features = next_features
            total_reward += reward
            steps += 1
            if done:
                break
        agent.decay_epsilon()
        history.append(EpisodeStats(episode_index, total_reward, steps, achieved, agent.epsilon))

    window = history[-max(1, episodes // 10) :]
    report = PolicyReport(
        episodes=tuple(history),
        success_rate_window=sum(1 for e in window if e.success) / len(window),
        final_epsilon=agent.epsilon,
    )
    return agent, report


def evaluate_policy(
    task: SandboxTask,
    success: Callable[[State], bool],
    agent: QLearningAgent,
    *,
    episodes: int = 20,
    hasher: FeatureHasher | None = None,
) -> float:
    """Greedy success rate, used as the held-out generalisation check."""
    hasher = hasher or FeatureHasher(64)
    env = SandboxPolicyEnv(task, success, hasher)
    wins = 0
    for _ in range(episodes):
        features = env.reset()
        while True:
            action = agent.act(features, greedy=True)
            features, _reward, done, achieved = env.step(action)
            if done:
                wins += 1 if achieved else 0
                break
    return wins / episodes
