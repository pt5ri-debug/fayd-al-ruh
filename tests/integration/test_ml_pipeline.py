"""Integration checks for the ML learning pipeline.

These assert the end-to-end learning contract: an agent starts knowing nothing
and measurably improves on the project's own safe, replayable tasks.
"""

import pytest
from core.sandbox_tasks import HELD_OUT_TASKS, TASKS, task_by_name
from ml.cognitive_brain import CognitiveBrain
from ml.dataset import FeatureHasher, build_classification_dataset, train_val_split
from ml.learner import QLearningAgent, evaluate_policy, train_policy
from ml.models import MLP
from ml.trainer import Trainer

pytestmark = pytest.mark.integration


def _success_for(task_name: str):
    facts = {
        "wet-garden": "wet_soil",
        "stack-blocks": "tower_built",
        "plant-seed": "sprout",
        "boil-water": "tea_ready",
        "light-candle": "lit_candle",
        "recharge-lamp": "lamp_on",
    }
    fact = facts[task_name]
    return lambda state: fact in state.facts


def test_learning_improves_policy_on_every_task():
    hasher = FeatureHasher(64)
    for task in TASKS:
        success = _success_for(task.name)
        agent, report = train_policy(task, success, episodes=200, hasher=hasher)

        assert report.episodes[0].epsilon > 0.9  # starts nearly fully exploratory
        assert report.success_rate_window == 1.0
        assert evaluate_policy(task, success, agent, episodes=20, hasher=hasher) == 1.0


def test_cognitive_brain_learns_and_consolidates_memory():
    task = task_by_name("wet-garden")
    success = _success_for("wet-garden")
    brain = CognitiveBrain(task, success, seed=0)

    report = brain.learn(episodes=150)

    assert report.policy_success_rate == 1.0
    assert report.memory_size > 0
    assert report.final_error <= report.initial_error


def test_supervised_classifier_generalises_on_experience():
    hasher = FeatureHasher(64)
    experiences = []
    for task in TASKS:
        success = _success_for(task.name)
        brain = CognitiveBrain(task, success, seed=0)
        brain.learn(episodes=80)
        experiences.extend(brain.memory.items)

    dataset = build_classification_dataset(experiences, hasher)
    train, val = train_val_split(dataset, val_fraction=0.25, seed=1)
    model = MLP(hasher.size, 32, max(2, dataset.num_classes), seed=3)
    report = Trainer(learning_rate=0.05, batch_size=8, max_epochs=150, patience=15, seed=2).fit(
        model, train, val
    )

    assert report.final.val_accuracy >= 0.7
    assert report.final.val_loss <= report.epochs[0].val_loss


def test_generalisation_on_novel_task_uses_the_same_machinery():
    """Brand-new tasks train with no task-specific tuning and still succeed."""
    hasher = FeatureHasher(64)

    for task in HELD_OUT_TASKS:
        success = _success_for(task.name)

        agent, report = train_policy(task, success, episodes=200, hasher=hasher)
        assert report.success_rate_window == 1.0
        assert evaluate_policy(task, success, agent, episodes=20, hasher=hasher) == 1.0

        brain = CognitiveBrain(task, success, seed=0)
        brain_report = brain.learn(episodes=150)
        assert brain_report.policy_success_rate == 1.0
        assert brain_report.final_error <= brain_report.initial_error


def test_policy_survives_save_and_load(tmp_path):
    task = task_by_name("wet-garden")
    success = _success_for("wet-garden")
    hasher = FeatureHasher(64)
    agent, _report = train_policy(task, success, episodes=150, hasher=hasher)

    path = agent.save(tmp_path / "policy.npz")
    restored = QLearningAgent.load(path)

    assert restored.num_actions == agent.num_actions
    assert restored.num_features == agent.num_features
    assert evaluate_policy(task, success, restored, episodes=20, hasher=hasher) == 1.0
