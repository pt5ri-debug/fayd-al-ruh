"""Tests for model-based planning and Dyna-Q over a learned world model."""

import numpy as np
import pytest

from core.sandbox_tasks import HELD_OUT_TASKS, TASKS, task_by_name
from core.state import State
from ml.dataset import FeatureHasher
from ml.learner import train_policy
from ml.model_based_planner import (
    DynaQAgent,
    evaluate_model_based_policy,
    plan_with_world_model,
    train_model_based_policy,
)
from ml.world_model import SymbolicWorldModel, explore_task

_SUCCESS_FACTS = {
    "wet-garden": "wet_soil",
    "stack-blocks": "tower_built",
    "plant-seed": "sprout",
    "boil-water": "tea_ready",
    "light-candle": "lit_candle",
    "recharge-lamp": "lamp_on",
}
_MULTI_STEP = ("stack-blocks", "plant-seed", "boil-water", "recharge-lamp")


def _goal(name: str):
    fact = _SUCCESS_FACTS[name]
    return lambda state: fact in state.facts


def test_world_model_learns_effects_from_observation_only():
    """The model reconstructs every declared effect without reading it."""
    for task in (*TASKS, *HELD_OUT_TASKS):
        model = explore_task(task, attempts=200, seed=0)
        for spec in task.allowed_actions:
            state = State(task.initial_facts)
            if not state.satisfies(spec.preconditions):
                state = State(state.facts | spec.preconditions)
            learned = model.predict(state, spec.to_action())
            assert learned.known
            assert learned.added == spec.add_effects
            assert learned.removed == spec.remove_effects


def test_world_model_is_unknown_before_any_observation():
    task = task_by_name("wet-garden")
    model = SymbolicWorldModel()
    spec = task.allowed_actions[0]
    assert not model.predict(State(task.initial_facts), spec.to_action()).known


@pytest.mark.parametrize("task_name", [task.name for task in (*TASKS, *HELD_OUT_TASKS)])
def test_planner_reconstructs_the_solution_inside_the_learned_model(task_name):
    task = task_by_name(task_name)
    model = explore_task(task, attempts=200, seed=0)

    result = plan_with_world_model(model, task, _goal(task_name), max_depth=6)

    assert result.solved
    # Replaying the plan in the real sandbox must reach the real goal.
    from core.sandbox_environment import SandboxEnvironment

    env = SandboxEnvironment(task)
    for action_name in result.actions:
        env.step(task.allowed_action(action_name))  # type: ignore[arg-type]
    assert _goal(task_name)(env.state)


def test_planner_reports_failure_when_the_model_is_empty():
    task = task_by_name("plant-seed")
    result = plan_with_world_model(SymbolicWorldModel(), task, _goal("plant-seed"))
    assert not result.solved
    assert result.actions == ()


def test_imagination_updates_the_policy_from_imagined_transitions():
    task = task_by_name("plant-seed")
    goal = _goal("plant-seed")
    hasher = FeatureHasher(64)
    model = explore_task(task, attempts=200, seed=0)
    agent = DynaQAgent(hasher.size, len(task.allowed_actions), planning_steps=20, seed=0)
    states = [State(task.initial_facts), State(task.initial_facts | {"tilled_soil"})]

    before = agent.W.copy()
    applied = agent.imagine(
        model,
        task,
        goal,
        hasher,
        reward_of=lambda _s, _a, _n, done: 1.0 if done else -0.02,
        real_states=states,
    )

    assert applied > 0
    assert not np.array_equal(before, agent.W)


def test_zero_planning_steps_means_no_imagination():
    task = task_by_name("plant-seed")
    hasher = FeatureHasher(64)
    model = explore_task(task, attempts=200, seed=0)
    agent = DynaQAgent(hasher.size, len(task.allowed_actions), planning_steps=0, seed=0)
    applied = agent.imagine(
        model,
        task,
        _goal("plant-seed"),
        hasher,
        reward_of=lambda _s, _a, _n, done: 1.0 if done else -0.02,
        real_states=[State(task.initial_facts)],
    )
    assert applied == 0


def test_dyna_q_improves_sample_efficiency_on_multi_step_tasks():
    """Averaged over seeds, Dyna-Q beats model-free at a small budget.

    This is the measured claim, not a guaranteed per-seed one: on easy tasks the
    model-free agent already saturates, so the gain is concentrated where the
    task has several steps. The aggregate is asserted to avoid cherry-picking.
    """
    hasher = FeatureHasher(64)
    seeds = 3
    free_total = 0.0
    dyna_total = 0.0
    for name in _MULTI_STEP:
        task = task_by_name(name)
        goal = _goal(name)
        for seed in range(seeds):
            _, free_report = train_policy(task, goal, episodes=40, hasher=hasher, seed=seed)
            _, _, dyna_report = train_model_based_policy(
                task, goal, episodes=40, planning_steps=10, seed=seed
            )
            free_total += free_report.success_rate_window
            dyna_total += dyna_report.success_rate
    assert dyna_total > free_total


@pytest.mark.parametrize("task_name", [task.name for task in (*TASKS, *HELD_OUT_TASKS)])
def test_dyna_q_policy_reaches_the_real_goal(task_name):
    task = task_by_name(task_name)
    goal = _goal(task_name)
    hasher = FeatureHasher(64)

    agent, _model, _report = train_model_based_policy(
        task, goal, episodes=120, planning_steps=10, seed=0
    )
    held_out = evaluate_model_based_policy(task, goal, agent, episodes=20, hasher=hasher)

    assert held_out >= 0.9
