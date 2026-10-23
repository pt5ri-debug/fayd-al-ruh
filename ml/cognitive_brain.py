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

import json
import random
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import cast

import numpy as np
from core.action import Action
from core.action_policy import ActionPolicy
from core.progress import ProgressCallback
from core.sandbox_environment import SandboxEnvironment, SandboxTask
from core.state import State

from ml.dataset import FeatureHasher, ReplayBuffer, Transition
from ml.learner import QLearningAgent, state_tokens
from ml.models import LinearRegressor, load_model, save_model


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
        max_learning_rate: float = 0.1,
        min_learning_rate: float = 0.005,
        max_grad_norm: float = 1.0,
        replay_batch_size: int = 0,
        replay_updates: int = 1,
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
        self.max_learning_rate = max_learning_rate
        self.min_learning_rate = min_learning_rate
        self.max_grad_norm = max_grad_norm
        # When set, each episode replays this many random past transitions and
        # re-learns from them, so stored experience actually shapes the weights.
        self.replay_batch_size = replay_batch_size
        self.replay_updates = replay_updates
        self._rng = random.Random(seed)
        self._errors: deque[float] = deque(maxlen=50)
        self.episodes_trained = 0

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
        # Clip gradients so a single surprising transition cannot blow up the
        # world model (predictive coding must stay stable under outlier errors).
        total_norm = float(np.sqrt(sum(float(np.sum(g * g)) for g in grads)))
        scale = 1.0 if total_norm <= self.max_grad_norm else self.max_grad_norm / (total_norm + 1e-12)
        for param, grad in zip(self.world_model.parameters(), grads, strict=False):
            param -= self.learning_rate * scale * grad
        self._errors.append(error)
        return error

    def replay_batch(self, batch: list[Transition]) -> float:
        """Re-learn the world model from a batch of stored experiences.

        This is the step that makes the replay buffer load-bearing: the same
        transitions are revisited after the policy has improved, so early
        experience is consolidated rather than merely logged.
        """
        if not batch:
            return 0.0
        inputs = np.stack(
            [self._input(self.perceive(State(frozenset(t.state_tokens))), t.action) for t in batch]
        )
        targets = np.stack([self.perceive(State(frozenset(t.next_tokens))) for t in batch])
        errors = np.mean((self.world_model.forward(inputs) - targets) ** 2, axis=1)
        grads = self.world_model.backward(inputs, targets)
        total_norm = float(np.sqrt(sum(float(np.sum(g * g)) for g in grads)))
        scale = 1.0 if total_norm <= self.max_grad_norm else self.max_grad_norm / (total_norm + 1e-12)
        for param, grad in zip(self.world_model.parameters(), grads, strict=False):
            param -= self.learning_rate * scale * grad
        error = float(np.mean(errors))
        self._errors.append(error)
        return error

    def replay(self) -> float:
        """Replay a random batch from memory; a no-op when replay is disabled."""
        if self.replay_batch_size < 1 or not self.memory.items:
            return 0.0
        batch = self.memory.sample(self.replay_batch_size, rng=self._rng)
        error = 0.0
        for _ in range(max(1, self.replay_updates)):
            error = self.replay_batch(batch)
        return error

    # --- reflection (metacognition) ------------------------------------
    def reflect(self) -> MetacognitiveState:
        if not self._errors:
            return MetacognitiveState(0.0, 0.0, 0.0, self.learning_rate, 1.0)
        errors = list(self._errors)
        half = max(1, len(errors) // 2)
        early, late = errors[:half], errors[half:] or errors[-1:]
        trend = float(np.mean(early) - np.mean(late))
        if trend >= 0:
            # Error is falling (or flat): the model is settling, so slow down
            # for finer adjustments.
            self.learning_rate = max(self.min_learning_rate, self.learning_rate * 0.95)
        else:
            # Error is rising: the model is being surprised, so adapt a little
            # faster -- but stay bounded so it can never diverge.
            self.learning_rate = min(self.max_learning_rate, self.learning_rate * 1.1)
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
    def run_episode(
        self,
        *,
        train_policy: bool = True,
        greedy: bool = False,
        advisor: ActionPolicy | None = None,
    ) -> tuple[float, bool, int]:
        environment = SandboxEnvironment(self.task)
        features = self.perceive(environment.state)
        total_reward = 0.0
        achieved = False
        steps = 0
        action_names = tuple(spec.name for spec in self.task.allowed_actions)
        while True:
            current_tokens = state_tokens(environment.state)
            if advisor is not None:
                chosen = advisor.select(environment.state.facts, action_names)
                action_index = action_names.index(chosen) if chosen in action_names else self.act(features, greedy=greedy)
            else:
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
            self.replay()
        return total_reward, achieved, steps

    # --- persistence (continuous learning across sessions) -------------
    def save(self, directory: str | Path) -> Path:
        """Persist the world model, the policy, the memory, and the counters."""
        target = Path(directory)
        target.mkdir(parents=True, exist_ok=True)
        save_model(self.world_model, target / "world_model.npz")
        self.policy.save(target / "policy.npz")
        metadata = {
            "task": self.task.name,
            "episodes_trained": self.episodes_trained,
            "learning_rate": self.learning_rate,
            "epsilon": self.policy.epsilon,
            "feature_size": self.hasher.size,
            "num_actions": self.num_actions,
            "memory": self.memory.to_dicts(),
        }
        (target / "brain.json").write_text(
            json.dumps(metadata, ensure_ascii=False), encoding="utf-8"
        )
        return target

    @classmethod
    def load(cls, directory: str | Path, task: SandboxTask, success) -> CognitiveBrain:
        """Restore a saved brain so learning can continue where it stopped."""
        source = Path(directory)
        metadata = json.loads((source / "brain.json").read_text(encoding="utf-8"))
        brain = cls(
            task,
            success,
            feature_size=int(metadata["feature_size"]),
            learning_rate=float(metadata["learning_rate"]),
        )
        brain.world_model = cast(LinearRegressor, load_model(source / "world_model.npz"))
        brain.policy = QLearningAgent.load(source / "policy.npz")
        brain.learning_rate = float(metadata["learning_rate"])
        brain.episodes_trained = int(metadata["episodes_trained"])
        brain.memory.load_dicts(metadata.get("memory", []))
        return brain

    def learn(
        self, *, episodes: int = 200, progress: ProgressCallback | None = None
    ) -> BrainReport:
        if episodes < 1:
            raise ValueError("episodes must be at least 1.")
        error_curve: list[float] = []
        for episode in range(1, episodes + 1):
            self.run_episode()
            self.episodes_trained += 1
            recent_error = self.reflect().recent_error
            error_curve.append(recent_error)
            if progress is not None:
                progress(
                    episode,
                    episodes,
                    detail=f"error {recent_error:.4f}  memory {len(self.memory)}",
                )

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
