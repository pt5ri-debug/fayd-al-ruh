"""Tests for reward learned from observed interaction outcomes."""

import pytest
from core.interaction_reward import (
    InteractionRewardModel,
    Outcome,
    outcomes_from_evaluation_corpus,
    outcomes_from_reviews,
)
from core.sandbox_tasks import HELD_OUT_TASKS, TASKS, task_by_name
from ml.dataset import FeatureHasher
from ml.interaction_learner import (
    evaluate_interaction_policy,
    recorded_action_outcomes,
    train_interaction_policy,
)

_SUCCESS_FACTS = {
    "wet-garden": "wet_soil",
    "stack-blocks": "tower_built",
    "plant-seed": "sprout",
    "boil-water": "tea_ready",
    "light-candle": "lit_candle",
    "recharge-lamp": "lamp_on",
}


def _goal_for(task_name: str):
    fact = _SUCCESS_FACTS[task_name]
    return lambda state: fact in state.facts


def _reward_model() -> InteractionRewardModel:
    model = InteractionRewardModel()
    model.record_many(outcomes_from_evaluation_corpus())
    for task in (*TASKS, *HELD_OUT_TASKS):
        model.record_many(recorded_action_outcomes(task))
    return model


def test_outcome_validates_its_ranges():
    with pytest.raises(ValueError):
        Outcome("ctx", "act", 1.5, 0.5)
    with pytest.raises(ValueError):
        Outcome("ctx", "act", 0.5, -0.1)
    with pytest.raises(ValueError):
        Outcome("", "act", 0.5, 0.5)


def test_action_outcomes_without_confidence_do_not_move_calibration():
    model = InteractionRewardModel()
    model.record(Outcome("ctx", "water", None, 1.0))
    model.record(Outcome("ctx", "wait", None, 0.0))

    assert model.forecasts_recorded == 0
    assert model.expected_calibration_error() == 0.0
    assert model.action_reward("ctx", "water") == 1.0


def test_calibration_reward_tracks_observed_outcomes():
    model = InteractionRewardModel(num_bins=10)
    model.record(Outcome("ctx", "a", 0.9, 1.0))
    model.record(Outcome("ctx", "b", 0.9, 1.0))
    model.record(Outcome("ctx", "c", 0.1, 0.0))

    # High-confidence forecasts that came true are rewarded, low ones are not.
    assert model.calibration_reward(0.9) == 1.0
    assert model.calibration_reward(0.1) == 0.0
    assert model.expected_calibration_error() < 0.2


def test_action_reward_ranks_effective_actions_above_ineffective_ones():
    model = InteractionRewardModel()
    model.record(Outcome("ctx", "works", 0.5, 1.0))
    model.record(Outcome("ctx", "works", 0.5, 1.0))
    model.record(Outcome("ctx", "fails", 0.5, 0.0))

    assert model.action_reward("ctx", "works") == 1.0
    assert model.action_reward("ctx", "fails") == 0.0
    assert model.rank_actions("ctx")[0][0] == "works"


def test_action_reward_falls_back_to_default_for_unknown_context():
    model = InteractionRewardModel(default_reward=0.5)
    assert model.action_reward("never-seen", "anything") == 0.5


def test_evaluation_corpus_outcomes_are_real_pairs():
    outcomes = outcomes_from_evaluation_corpus()

    assert outcomes
    assert all(0.0 <= item.predicted_confidence <= 1.0 for item in outcomes)
    assert all(0.0 <= item.observed <= 1.0 for item in outcomes)
    assert all(item.source.startswith("evaluation:") for item in outcomes)


def test_human_reviews_are_a_separate_interaction_channel():
    outcomes = outcomes_from_reviews([("wet-garden", "water", 0.8, 1.0)])

    assert outcomes[0].source == "human-review"
    assert outcomes[0].succeeded


def test_recorded_action_outcomes_mark_off_path_actions_ineffective():
    task = next(item for item in TASKS if item.name == "boil-water")
    outcomes = {outcome.action: outcome.observed for outcome in recorded_action_outcomes(task)}

    assert outcomes["fill"] == 1.0
    assert outcomes["stir"] == 0.0


def test_reward_model_survives_save_and_load(tmp_path):
    model = _reward_model()
    path = model.save(tmp_path / "reward.json")

    restored = InteractionRewardModel.load(path)

    assert restored.summary() == model.summary()
    assert restored.action_reward("wet-garden", "water") == model.action_reward(
        "wet-garden", "water"
    )


@pytest.mark.parametrize("task_name", [task.name for task in (*TASKS, *HELD_OUT_TASKS)])
def test_interaction_reward_trains_a_policy_that_reaches_the_real_goal(task_name):
    """The policy never sees the sandbox predicate; reward is learned, not given."""
    task = task_by_name(task_name)
    goal = _goal_for(task_name)
    hasher = FeatureHasher(64)
    model = _reward_model()

    agent, report = train_interaction_policy(task, model, goal, episodes=400, hasher=hasher)
    held_out = evaluate_interaction_policy(task, model, goal, agent, episodes=20, hasher=hasher)

    assert report.success_rate >= 0.9
    assert held_out >= 0.9
