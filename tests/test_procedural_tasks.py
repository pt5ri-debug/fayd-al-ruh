"""Tests for procedural tasks derived from the corpus knowledge."""

from core.petroleum_corpus import entries
from ml.petroleum import petroleum_success, train_petroleum
from ml.procedural_tasks import (
    procedural_sandbox_tasks,
    procedural_success,
    procedural_tasks,
)


def test_one_task_per_entry_with_procedures():
    with_procedures = [entry for entry in entries() if any(s.strip() for s in entry.procedures)]
    assert len(procedural_tasks()) == len(with_procedures)


def test_derived_tasks_are_unique_and_ordered():
    tasks = procedural_tasks()
    names = [item.task.name for item in tasks]
    assert len(names) == len(set(names))
    assert all(item.steps >= 1 for item in tasks)


def test_procedural_success_predicate_matches_last_step():
    from core.state import State

    task = procedural_sandbox_tasks()[0]
    success = procedural_success(task.name)
    assert not success(State(facts=frozenset({"procedure_ready"})))
    # The last chain action is the third from the end (before ``inspect`` and
    # ``review``); its effects carry the shared completion fact.
    last_step = task.allowed_actions[-3]
    final = State(facts=frozenset({"procedure_ready"}) | set(last_step.add_effects))
    assert success(final)


def test_petroleum_success_dispatches_procedural_tasks():
    task = procedural_sandbox_tasks()[0]
    assert petroleum_success(task.name) is not None


def test_training_can_cover_procedural_tasks():
    report = train_petroleum(
        episodes=20, include_held_out=False, include_procedural=True, max_procedural=3
    )
    assert report.procedural_total == len(procedural_tasks())
    assert report.procedural_covered == 3
    assert 0.0 < report.procedural_coverage < 1.0
    assert all(outcome.held_out_success == 1.0 for outcome in report.outcomes[-3:])
