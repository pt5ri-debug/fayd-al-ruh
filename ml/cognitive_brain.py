"""A predictive-coding brain: perception, prediction, learning, and reflection.

This wires the ML stack into one loop that mirrors a few well-understood
properties of biological learning, without claiming to reproduce a mind:

- perception   : facts are encoded into a distributed representation
- prediction   : a learned world model predicts the next representation
- learning     : prediction errors drive online weight updates (predictive coding)
- memory       : experiences are consolidated into a replay buffer
- reflection   : a metacognitive loop tracks surprise and tunes its own
                 learning rate (a simple form of learning-to-learn)

It runs entirely on the symbolic/sandbox environments, so it is deterministic,
replayable, and safe.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

import numpy as np

from core.action import Action
from core.sandbox_environment import SandboxEnvironment, SandboxTask
from core.state import State
from ml.dataset import FeatureHasher, ReplayBuffer, Transition
from ml.learner import QLearningAgent, state_tokens
from ml.models import LinearRegressor


def _action_vector(action_index: int, num_actions: int) -> np.ndarray:
    vector = np.zeros(num_actions, dtype=np.float64)
    vector[action_index] = 1.0
    return vector


@dataclass
class MetacognitiveState:
    """Reflection summary: how surprised the brain has been, and its self-tuned rate."""

    average_error: float
    recent_error: float
    trend: float
    learning_rate: float
    confidence: float


@dataclass
class BrainReport:
    episodes: int
    initial_error: float
    final_error: float
    policy_success_rate: float
    memory_size: int
    learning_rate: float
    error_curve: list[float] = field(default_factory=list)

    @property
    def error_improvement(self) -> float:
        return self.initial_error - self.final_error


class CognitiveBrain:
    """A small predictive-coding agent with memory and metacognition."""

    def __init__(
        self,
        task: SandboxTask,
        success,
        *,
        feature_size: int = 64,
        learning_rate: float = 0.05,
        seed: int = 0,
        memory_capacity: int = 5_000,
    ) -> None:
        self.task = task
        self.success = success
        self.hasher = FeatureHasher(feature_size)
        self.num_actions = len(task.allowed_actions)
        self.memory = ReplayBuffer(memory_capacity)
        self.world_model = LinearRegressor(
            feature_size + self.num_actions, feature_size, seed=seed
        )
        self.policy = QLearningAgent(feature_size, self.num_actions, seed=seed)
        self.learning_rate = learning_rate
        self._errors: deque[float] = deque(maxlen=50)

    # --- perception -----------------------------------------------------
    def perceive(self, state: State) -> np.ndarray:
        return self.hasher.encode(state_tokens(state))

    def _input(self, features: np.ndarray, action_index: int) -> np.ndarray:
        return np.concatenate([features, _action_vector(action_index, self.num_actions)])

    # --- prediction & learning -----------------------------------------
    def predict(self, features: np.ndarray, action_index: int) -> np.ndarray:
        return self.world_model.forward(self._input(features, action_index))

    def observe(
        self, features: np.ndarray, action_index: int, next_features: np.ndarray
    ) -> float:
        """Learn from one transition; returns the prediction error."""
        inputs = self._input(features, action_index).reshape(1, -1)
        targets = next_features.reshape(1, -1)
        prediction = self.world_model.forward(inputs)[0]
        error = float(np.mean((prediction - next_features) ** 2))
        grads = self.world_model.backward(inputs, targets)
        for param, grad in zip(self.world_model.parameters(), grads):
            param -= self.learning_rate * grad
        self._errors.append(error)
        return error

    # --- reflection (metacognition) ------------------------------------
    def reflect(self) -> MetacognitiveState:
        if not self._errors:
            return MetacognitiveState(0.0, 0.0, 0.0, self.learning_rate, 1.0)
        errors = list(self._errors)
        half = max(1, len(errors) // 2)
        early, late = errors[:half], errors[half:] or errors[-1:]
        trend = float(np.mean(early) - np.mean(late))
        if trend < 0:
            # Surprise is rising: increase the rate to adapt faster.
            self.learning_rate = min(0.5, self.learning_rate * 1.1)
        else:
            # Learning is stable: settle down for finer adjustments.
            self.learning_rate = max(0.005, self.learning_rate * 0.95)
        recent = float(np.mean(errors[-5:]))
        confidence = 1.0 / (1.0 + recent)
        return MetacognitiveState(
            average_error=float(np.mean(errors)),
            recent_error=recent,
            trend=trend,
            learning_rate=self.learning_rate,
            confidence=confidence,
        )

    # --- action selection ----------------------------------------------
    def act(self, features: np.ndarray, *, greedy: bool = False) -> int:
        return self.policy.act(features, greedy=greedy)

    # --- one full episode ----------------------------------------------
    def run_episode(self, *, train_policy: bool = True, greedy: bool = False) -> tuple[float, bool, int]:
        environment = SandboxEnvironment(self.task)
        features = self.perceive(environment.state)
        total_reward = 0.0
        achieved = False
        steps = 0
        while True:
            current_tokens = state_tokens(environment.state)
            action_index = self.act(features, greedy=greedy)
            spec = self.task.allowed_actions[action_index]
            action = Action(
                spec.name, spec.preconditions, spec.add_effects, spec.remove_effects, spec.cost
            )
            try:
                environment.step(action)
            except RuntimeError:
                break
            next_features = self.perceive(environment.state)
            self.observe(features, action_index, next_features)
            achieved = self.success(environment.state)
            reward = 1.0 if achieved else -0.02
            if train_policy:
                self.policy.update(features, action_index, reward, next_features, achieved)
            self.memory.add(
                Transition(current_tokens, action_index, reward,
                           state_tokens(environment.state), achieved)
            )
            features = next_features
            total_reward += reward
            steps += 1
            if achieved or environment.terminated != "running" or steps >= self.task.max_steps:
                break
        if train_policy:
            self.policy.decay_epsilon()
        return total_reward, achieved, steps

    def learn(self, *, episodes: int = 200) -> BrainReport:
        """Run a full learning session and summarise the trajectory."""
        if episodes < 1:
            raise ValueError("episodes must be at least 1.")
        error_curve: list[float] = []
        for _ in range(episodes):
            self.run_episode()
            error_curve.append(self.reflect().recent_error)

        # Held-out greedy evaluation after learning.
        wins = 0
        for _ in range(20):
            _reward, achieved, _steps = self.run_episode(train_policy=False, greedy=True)
            wins += 1 if achieved else 0
        return BrainReport(
            episodes=episodes,
            initial_error=error_curve[0] if error_curve else 0.0,
            final_error=error_curve[-1] if error_curve else 0.0,
            policy_success_rate=wins / 20,
            memory_size=len(self.memory),
            learning_rate=self.learning_rate,
            error_curve=error_curve,
        )
