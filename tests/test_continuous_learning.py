"""Tests for continuous learning: knowledge updates, decay, and regression checks."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest
from core.continuous_learning import (
    ContinuousLearner,
    Feedback,
    RegressionMonitor,
)
from core.documents import DocumentStore, build_document_store
from core.knowledge_update import KnowledgeUpdate, KnowledgeUpdater
from core.provenance import ProvenanceStage


def _store() -> DocumentStore:
    return build_document_store()


# --- knowledge updates ----------------------------------------------------


def test_update_adds_a_citable_statement():
    store = _store()
    updater = KnowledgeUpdater(store)
    outcome = updater.apply(
        KnowledgeUpdate("Porosity measures the void volume in the rock.", source="book", strength=0.6)
    )
    assert outcome.action == "added"
    assert store.get(outcome.doc_id) is not None
    assert store.get(outcome.doc_id).text == "Porosity measures the void volume in the rock."


def test_update_reinforces_on_repeat():
    updater = KnowledgeUpdater(_store())
    first = updater.apply(KnowledgeUpdate("Porosity measures the voids.", source="a", strength=0.5))
    second = updater.apply(KnowledgeUpdate("Porosity measures the voids.", source="b", strength=0.5))
    assert first.action == "added"
    assert second.action == "reinforced"
    assert second.confidence > first.confidence


def test_doc_id_is_stable_across_instances():
    # A random per-process hash would give the same statement a different id on
    # every run, so persisted knowledge could never be reinforced or decayed.
    first = KnowledgeUpdater(_store()).apply(
        KnowledgeUpdate("Porosity measures the void volume in the rock.", source="a")
    )
    second = KnowledgeUpdater(_store()).apply(
        KnowledgeUpdate("Porosity measures the void volume in the rock.", source="b")
    )
    assert first.doc_id == second.doc_id


def test_update_rejects_off_topic_statement():
    updater = KnowledgeUpdater(_store())
    outcome = updater.apply(
        KnowledgeUpdate("zzz qqq www xxx", source="noise", strength=0.9)
    )
    assert outcome.action == "rejected"
    assert outcome.doc_id not in {doc.doc_id for doc in updater.store.documents()}


def test_update_records_provenance():
    updater = KnowledgeUpdater(_store())
    outcome = updater.apply(KnowledgeUpdate("Permeability measures how easily fluid flows.", source="paper"))
    record = updater.ledger.get(outcome.doc_id)
    assert record is not None
    assert record.stage is ProvenanceStage.LEARNING
    assert record.source == "paper"


def test_decay_stale_weakens_and_drops():
    store = _store()
    updater = KnowledgeUpdater(store, max_age=timedelta(days=30))
    outcome = updater.apply(
        KnowledgeUpdate("Permeability measures how easily fluid flows.", source="x", strength=0.2)
    )
    later = datetime.now(UTC) + timedelta(days=400)
    decayed = updater.decay_stale(now=later)
    assert any(item.doc_id == outcome.doc_id for item in decayed)
    assert updater.strength(outcome.doc_id) < 0.2


def test_decay_keeps_fresh_statements():
    updater = KnowledgeUpdater(_store(), max_age=timedelta(days=365))
    outcome = updater.apply(KnowledgeUpdate("Permeability measures how easily fluid flows.", source="x"))
    assert updater.decay_stale() == []
    assert updater.strength(outcome.doc_id) > 0


def test_supersede_replaces_and_links_provenance():
    updater = KnowledgeUpdater(_store())
    old = updater.apply(KnowledgeUpdate("Permeability measures how easily fluid flows.", source="old"))
    new = updater.supersede(
        old.doc_id, KnowledgeUpdate("Permeability measures how easily fluid flows through rock.", source="new")
    )
    assert new.action == "superseded"
    assert updater.store.get(old.doc_id) is None
    record = updater.ledger.get(new.doc_id)
    assert old.doc_id in record.parent_ids


def test_update_validates_input():
    with pytest.raises(ValueError):
        KnowledgeUpdate("", source="x")
    with pytest.raises(ValueError):
        KnowledgeUpdate("text", source="")
    with pytest.raises(ValueError):
        KnowledgeUpdate("text", source="x", strength=1.5)


# --- regression monitor ---------------------------------------------------


def test_regression_monitor_detects_drops():
    monitor = RegressionMonitor({"a": 1.0, "b": 0.8}, tolerance=0.01)
    report = monitor.check({"a": 0.9, "b": 0.8})
    assert not report.passed
    assert [check.name for check in report.regressions] == ["a"]


def test_regression_monitor_passes_within_tolerance():
    monitor = RegressionMonitor({"a": 1.0}, tolerance=0.05)
    assert monitor.check({"a": 0.96}).passed


def test_regression_monitor_reports_improvements():
    monitor = RegressionMonitor({"a": 0.5}, tolerance=0.0)
    report = monitor.check({"a": 0.9})
    assert report.passed
    assert [check.name for check in report.improved] == ["a"]


def test_regression_monitor_validates_tolerance():
    with pytest.raises(ValueError):
        RegressionMonitor({}, tolerance=-1)


# --- continuous learner ---------------------------------------------------


def test_learner_records_reward_and_grows_corpus_on_confirmation():
    updater = KnowledgeUpdater(_store())
    learner = ContinuousLearner(updater)
    step = learner.learn(
        Feedback(
            question="What is the relationship between porosity and permeability?",
            answer="Porosity measures the voids and relates to permeability.",
            confirmed=True,
            confidence=0.8,
        )
    )
    assert step.reward == 1.0
    assert step.changed_corpus


def test_learner_learns_from_a_correction():
    updater = KnowledgeUpdater(_store())
    learner = ContinuousLearner(updater)
    step = learner.learn(
        Feedback(
            question="What is permeability?",
            answer="A wrong answer.",
            confirmed=False,
            correction="Permeability measures how easily fluids flow.",
            confidence=0.4,
        )
    )
    assert step.reward == 0.0
    assert step.update is not None
    assert step.update.action == "added"


def test_learner_rewards_are_bounded_and_summarised():
    learner = ContinuousLearner(KnowledgeUpdater(_store()))
    for confirmed in (True, True, False):
        learner.learn(
            Feedback(
                question="permeability",
                answer="Permeability measures flow.",
                confirmed=confirmed,
                correction="" if confirmed else "Permeability measures how easily fluid flows.",
                confidence=0.7,
            )
        )
    reward = learner.reward_for("permeability")
    assert 0.0 <= reward <= 1.0
    summary = learner.summary()
    assert summary["learning_steps"] == 3.0
    assert summary["outcomes_recorded"] == 3.0


def test_feedback_requires_a_correction_when_rejected():
    with pytest.raises(ValueError):
        Feedback(question="q", answer="a", confirmed=False)
    with pytest.raises(ValueError):
        Feedback(question="", answer="a", confirmed=True)
    with pytest.raises(ValueError):
        Feedback(question="q", answer="a", confirmed=True, confidence=2.0)
