"""The proactive mind: the engines, the thought cycle, and the scheduler.

These tests exercise the whole autonomous loop over the project's *real*
ontology -- curiosity discovers gaps, motivation ranks them, the idea engine and
imagination propose, self-criticism reviews, and the cycle is recorded. Nothing
here loads a model or reaches the network, and the mind shares the one graph.
"""

from __future__ import annotations

import pytest
from cognition.engines.association import AssociationEngine
from cognition.engines.criticism import SelfCriticismEngine
from cognition.engines.curiosity import CuriosityEngine
from cognition.engines.ideas import IdeaEngine, NoveltyTracker
from cognition.engines.imagination import ImaginationEngine
from cognition.engines.questions import SelfQuestioningEngine
from cognition.proactive_mind import ProactiveMind
from cognition.records import Idea, IdeaKind
from cognition.scheduler import SchedulerSettings, ThoughtScheduler
from core.ontology import Ontology
from core.petroleum_corpus import seed_petroleum_corpus


@pytest.fixture(scope="module")
def ontology() -> Ontology:
    graph = Ontology()
    seed_petroleum_corpus(graph)
    return graph


# -- engines ----------------------------------------------------------------
def test_curiosity_finds_gaps_in_the_real_graph(ontology):
    gaps = CuriosityEngine(ontology).discover(())
    assert gaps
    assert all(gap.topic for gap in gaps)


def test_curiosity_is_bounded(ontology):
    gaps = CuriosityEngine(ontology).discover((), max_gaps=3)
    assert len(gaps) <= 3


def test_association_walks_the_real_graph(ontology):
    engine = AssociationEngine(ontology)
    neighbours = engine.neighbours("effective_porosity")
    assert "permeability" in neighbours


def test_association_ranks_pairs_deterministically(ontology):
    engine = AssociationEngine(ontology)
    labels = ("effective_porosity", "permeability", "density_log")
    first = engine.rank_pairs(labels, limit=3)
    second = engine.rank_pairs(labels, limit=3)
    assert first
    assert [a.describe() for a in first] == [a.describe() for a in second]


def test_ideas_carry_their_engine_and_rationale(ontology):
    engine = IdeaEngine(ontology)
    ideas = engine.generate(labels=("effective_porosity", "permeability"), limit=3)
    assert ideas
    assert all(idea.text and idea.engine for idea in ideas)


def test_criticism_reviews_and_never_inflates_confidence(ontology):
    engine = SelfCriticismEngine(ontology)
    idea = Idea(text="effective_porosity is caused by permeability", kind=IdeaKind.IDEA, confidence=0.4)
    reviewed = engine.review(idea)
    assert reviewed.confidence <= idea.confidence


def test_novelty_falls_for_a_repeat():
    tracker = NoveltyTracker()
    assert tracker.novelty("a fresh thought about flow") > 0.5
    tracker.remember("a fresh thought about flow")
    assert tracker.novelty("a fresh thought about flow") < 0.5


def test_imagination_produces_scenarios(ontology):
    scenarios = ImaginationEngine(ontology).generate(
        labels=("effective_porosity", "permeability"), limit=2
    )
    assert scenarios
    assert all(s.describe() for s in scenarios)


def test_question_engine_turns_gaps_into_questions(ontology):
    gaps = CuriosityEngine(ontology).discover(())
    questions = SelfQuestioningEngine(ontology).questions(gaps, limit=3)
    assert questions
    assert all(q.text.endswith("?") for q in questions)


# -- the thought cycle ------------------------------------------------------
def test_a_cycle_runs_and_is_recorded(ontology, tmp_path, monkeypatch):
    monkeypatch.setenv("FAYD_DATA_DIR", str(tmp_path))
    mind = ProactiveMind(ontology)
    cycle = mind.cycle(max_ideas=3)
    assert cycle.questions or cycle.ideas
    assert mind.last_cycle() is not None
    assert sum(mind.counts().values()) >= 1


def test_a_cycle_is_reproducible_for_the_same_inputs(ontology, tmp_path, monkeypatch):
    monkeypatch.setenv("FAYD_DATA_DIR", str(tmp_path))
    first = ProactiveMind(ontology).cycle(max_ideas=2)
    second = ProactiveMind(ontology).cycle(max_ideas=2)
    # Question ids are generated, but the *drives* a cycle records are deterministic.
    assert first.notes == second.notes
    assert len(first.questions) == len(second.questions)


def test_pending_ideas_are_newest_first(ontology, tmp_path, monkeypatch):
    monkeypatch.setenv("FAYD_DATA_DIR", str(tmp_path))
    mind = ProactiveMind(ontology)
    mind.cycle(max_ideas=2)
    pending = mind.pending_ideas(limit=5)
    assert pending
    assert pending[0].created_at >= pending[-1].created_at


def test_status_reports_honestly(ontology, tmp_path, monkeypatch):
    monkeypatch.setenv("FAYD_DATA_DIR", str(tmp_path))
    status = ProactiveMind(ontology).status()
    assert "ideas" in status
    assert "paused" in status


# -- the scheduler ----------------------------------------------------------
def test_scheduler_is_due_when_never_run(tmp_path, monkeypatch):
    monkeypatch.setenv("FAYD_DATA_DIR", str(tmp_path))
    scheduler = ThoughtScheduler()
    assert scheduler.due().allowed


def test_scheduler_defers_under_load(tmp_path, monkeypatch):
    monkeypatch.setenv("FAYD_DATA_DIR", str(tmp_path))
    scheduler = ThoughtScheduler(settings=SchedulerSettings(interval_seconds=3600))
    scheduler.record_cycle()
    decision = scheduler.due()
    assert not decision.allowed


def test_scheduler_pause_and_resume(tmp_path, monkeypatch):
    monkeypatch.setenv("FAYD_DATA_DIR", str(tmp_path))
    scheduler = ThoughtScheduler()
    scheduler.pause()
    assert not scheduler.due().allowed
    scheduler.resume()
    assert scheduler.due().allowed


def test_scheduler_refuses_a_negative_interval():
    with pytest.raises(ValueError):
        SchedulerSettings(interval_seconds=-1)
