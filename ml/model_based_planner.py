"""Model-based planning and Dyna-Q over the learned world model.

Two things the project could not do before now become possible:

- **Plan by imagination.** :func:`plan_with_world_model` searches inside the
  *learned* world model (:mod:`ml.world_model`) instead of the hand-written one,
  so the agent plans with what it has come to believe.
- **Learn from imagination.** :class:`DynaQAgent` replays real transitions and
  simulated ones, so a single real step yields many updates. That is the sample
  efficiency a model buys you, and it is measurable.

Success is always scored by the environment's real goal. The world model is only
ever used to choose and to imagine; it never decides what success means.
"""

from __future__ import annotations

import random
from collections.abc import Callable
from dataclasses import dataclass

from core.action import Action
from core.progress import ProgressCallback
from core.sandbox_environment import SandboxEnvironment, SandboxTask
from core.state import State

from ml.dataset import FeatureHasher
from ml.learner import QLearningAgent, state_tokens
from ml.world_model import SymbolicWorldModel, explore_task


@dataclass(frozen=True)
class PlanResult:
    """A plan found by searching the learned world model."""

    actions: tuple[str, ...]
    solved: bool
    nodes_expanded: int


def plan_with_world_model(
    model: SymbolicWorldModel,
    task: SandboxTask,
    goal: Callable[[State], bool],
    *,
    initial_state: State | None = None,
    max_depth: int = 6,
) -> PlanResult:
    """Breadth-first search inside the *learned* model.

    Only actions the model has actually seen something for are expanded, so the
    plan contains no imagined effect that was never observed. If the model is
    incomplete, the search honestly reports that it could not solve the task.
    """
    start = initial_state or State(task.initial_facts)
    if goal(start):
        return PlanResult((), True, 0)

    actions = [spec.to_action() for spec in task.allowed_actions]
    frontier: list[tuple[State, tuple[str, ...]]] = [(start, ())]
    seen = {start.facts}
    expanded = 0
    while frontier:
        state, path = frontier.pop(0)
        if len(path) >= max_depth:
            continue
        expanded += 1
        for action in actions:
            learned = model.predict(state, action)
            if not learned.known:
                continue
            next_state = State((state.facts | learned.added) - learned.removed)
            if next_state.facts == state.facts or next_state.facts in seen:
                continue
            next_path = (*path, action.name)
            if goal(next_state):
                return PlanResult(next_path, True, expanded)
            seen.add(next_state.facts)
            frontier.append((next_state, next_path))
    return PlanResult((), False, expanded)


class DynaQAgent(QLearningAgent):
    """Q-learning that also learns from imagined transitions (Dyna-Q).

    ``imagine`` replays the learned world model from states in memory: it picks
    a stored state and action, predicts the next state with the model, scores the
    transition with the same reward the agent uses in reality, and updates Q. One
    real interaction therefore trains the policy many times over.
    """

    def __init__(
        self,
        num_features: int,
        num_actions: int,
        *,
        planning_steps: int = 10,
        alpha: float = 0.2,
        gamma: float = 0.95,
        epsilon: float = 1.0,
        epsilon_decay: float = 0.97,
        epsilon_min: float = 0.05,
        seed: int = 0,
    ) -> None:
        super().__init__(
            num_features,
            num_actions,
            alpha=alpha,
            gamma=gamma,
            epsilon=epsilon,
            epsilon_decay=epsilon_decay,
            epsilon_min=epsilon_min,
            seed=seed,
        )
        if planning_steps < 0:
            raise ValueError("planning_steps cannot be negative.")
        self.planning_steps = planning_steps
        # A dedicated Python RNG for imagination; the parent's numpy RNG keeps
        # driving action selection, so exploration behaviour is unchanged.
        self._plan_rng = random.Random(seed)

    def imagine(
        self,
        model: SymbolicWorldModel,
        task: SandboxTask,
        goal: Callable[[State], bool],
        hasher: FeatureHasher,
        *,
        reward_of: Callable[[State, Action, State, bool], float],
        real_states: list[State],
    ) -> int:
        """Run ``planning_steps`` imagined updates; return how many were applied."""
        if self.planning_steps == 0 or not real_states:
            return 0
        actions = [spec.to_action() for spec in task.allowed_actions]
        applied = 0
        for _ in range(self.planning_steps):
            state = self._plan_rng.choice(real_states)
            action_index = self._plan_rng.randrange(self.num_actions)
            action = actions[action_index]
            learned = model.predict(state, action)
            if not learned.known:
                continue
            next_state = State((state.facts | learned.added) - learned.removed)
            done = goal(next_state)
            reward = reward_of(state, action, next_state, done)
            self.update(
                hasher.encode(state_tokens(state)),
                action_index,
                reward,
                hasher.encode(state_tokens(next_state)),
                done,
            )
            applied += 1
        return applied


@dataclass(frozen=True)
class PlanningReport:
    task_name: str
    episodes: int
    planning_steps: int
    success_rate: float
    first_success_episode: int
    model_transitions: int
    planned_solution: tuple[str, ...]


def train_model_based_policy(
    task: SandboxTask,
    goal: Callable[[State], bool],
    *,
    episodes: int = 200,
    planning_steps: int = 10,
    explore_attempts: int = 200,
    hasher: FeatureHasher | None = None,
    seed: int = 0,
    progress: ProgressCallback | None = None,
) -> tuple[DynaQAgent, SymbolicWorldModel, PlanningReport]:
    """Explore, then train a Dyna-Q agent that learns from real and imagined steps."""
    hasher = hasher or FeatureHasher(64)
    model = explore_task(task, attempts=explore_attempts, seed=seed)
    agent = DynaQAgent(
        hasher.size,
        len(task.allowed_actions),
        planning_steps=planning_steps,
        seed=seed,
    )

    def reward_of(_state: State, _action: Action, next_state: State, done: bool) -> float:
        return 1.0 if done else -0.02

    first_success = 0
    window: list[bool] = []
    for episode in range(1, episodes + 1):
        env = SandboxEnvironment(task)
        states: list[State] = [env.state]
        achieved = False
        while env.terminated == "running" and env.step_count < task.max_steps:
            features = hasher.encode(state_tokens(env.state))
            action_index = agent.act(features)
            spec = task.allowed_actions[action_index]
            try:
                env.step(spec.to_action())
            except RuntimeError:
                break
            states.append(env.state)
            next_features = hasher.encode(state_tokens(env.state))
            achieved = goal(env.state)
            done = achieved or env.terminated != "running" or env.step_count >= task.max_steps
            agent.update(features, action_index, 1.0 if achieved else -0.02, next_features, done)
            if done:
                break
        # Consolidate: re-learn from imagined transitions drawn from this episode.
        agent.imagine(model, task, goal, hasher, reward_of=reward_of, real_states=states)
        agent.decay_epsilon()
        if achieved and first_success == 0:
            first_success = episode
        window.append(achieved)
        if progress is not None:
            recent = window[-max(1, episodes // 5) :]
            progress(
                episode,
                episodes,
                detail=f"success {sum(recent) / len(recent):.2f}  "
                f"transitions {model.transitions_observed}",
            )

    recent = window[-max(1, episodes // 5) :]
    plan = plan_with_world_model(model, task, goal)
    return agent, model, PlanningReport(
        task_name=task.name,
        episodes=episodes,
        planning_steps=planning_steps,
        success_rate=round(sum(recent) / len(recent), 6),
        first_success_episode=first_success,
        model_transitions=model.transitions_observed,
        planned_solution=plan.actions,
    )


def evaluate_model_based_policy(
    task: SandboxTask,
    goal: Callable[[State], bool],
    agent: DynaQAgent,
    *,
    episodes: int = 20,
    hasher: FeatureHasher | None = None,
) -> float:
    """Greedy success rate of a Dyna-Q policy, measured on the real goal."""
    hasher = hasher or FeatureHasher(64)
    hits = 0
    for _ in range(episodes):
        env = SandboxEnvironment(task)
        while env.terminated == "running" and env.step_count < task.max_steps:
            features = hasher.encode(state_tokens(env.state))
            action_index = agent.act(features, greedy=True)
            try:
                env.step(task.allowed_actions[action_index].to_action())
            except RuntimeError:
                break
            if goal(env.state):
                hits += 1
                break
    return hits / episodes if episodes else 0.0


__all__ = [
    "DynaQAgent",
    "PlanResult",
    "PlanningReport",
    "evaluate_model_based_policy",
    "plan_with_world_model",
    "train_model_based_policy",
]
