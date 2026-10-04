"""Integration checks for the ML learning pipeline.

These assert the end-to-end learning contract: an agent starts knowing nothing
and measurably improves on the project's own safe, replayable tasks.
"""

import pytest

from core.sandbox_tasks import TASKS, task_by_name
from ml.cognitive_brain import CognitiveBrain
from ml.dataset import FeatureHasher, build_classification_dataset, train_val_split
from ml.learner import evaluate_policy, train_policy
from ml.models import MLP
from ml.trainer import Trainer

pytestmark = pytest.mark.integration


def _success_for(task_name: str):
    if task_name == "wet-garden":
        return lambda state: "wet_soil" in state.facts
    return lambda state: "tower_built" in state.facts


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
