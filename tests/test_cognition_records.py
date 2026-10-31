"""The Autonomous Cognitive System: records, reliability, and storage.

These tests pin the *foundation* the rest of the system is built on -- the record
objects and their round trips, the reliability rule that decides what may be
called a fact, and the persistent stores (including the fact store that makes
knowledge survive a restart).
"""

from __future__ import annotations

from cognition.records import (
    Experience,
    Idea,
    IdeaKind,
    Reliability,
    Source,
    SourceKind,
    ThoughtCycle,
    TrainingExample,
    TrainingExampleKind,
    promote,
)
from cognition.reliability import (
    BASE_RELIABILITY,
    TRUST_THRESHOLD,
    is_trusted,
    score,
)
from cognition.storage import (
    CycleStore,
    ExampleStore,
    ExperienceStore,
    FactStore,
    IdeaStore,
    meta_get,
    meta_set,
)


# -- records ----------------------------------------------------------------
def test_idea_round_trips_through_primitive():
    idea = Idea(text="porosity may relate to permeability", kind=IdeaKind.IDEA, engine="test")
    restored = Idea.from_primitive(idea.to_primitive())
    assert restored.id == idea.id
    assert restored.text == idea.text
    assert restored.kind is idea.kind
    assert restored.engine == idea.engine


def test_idea_reading_is_lenient_about_bad_user_data():
    restored = Idea.from_primitive({"text": 5, "confidence": "bad", "sources": "not-a-list"})
    assert restored.text == "5"
    assert restored.confidence == 0.3
    assert restored.sources == ()


def test_promote_moves_one_rung_and_refuses_a_skip():
    idea = Idea(text="a", kind=IdeaKind.IDEA)
    hypothesis = promote(idea, IdeaKind.HYPOTHESIS)
    assert hypothesis.kind is IdeaKind.HYPOTHESIS
    assert idea.kind is IdeaKind.IDEA  # the original is untouched
    try:
        promote(idea, IdeaKind.VERIFIED)
    except ValueError:
        pass
    else:  # pragma: no cover - the ladder must refuse a skip
        raise AssertionError("promote must refuse skipping a rung")


def test_experience_and_cycle_round_trip():
    experience = Experience(expectation="it rains", outcome="it did not", lesson="check the sky")
    assert Experience.from_primitive(experience.to_primitive()).lesson == "check the sky"
    cycle = ThoughtCycle(questions=("q1",), ideas=("i1",), notes=("n1",), duration_ms=12.5)
    restored = ThoughtCycle.from_primitive(cycle.to_primitive())
    assert restored.questions == ("q1",)
    assert restored.duration_ms == 12.5


def test_training_example_carries_its_verdict():
    example = TrainingExample(
        task="what is porosity",
        answer="a measure of void space",
        kind=TrainingExampleKind.ANSWER,
        qualified=True,
        reason="student was adequate",
    )
    restored = TrainingExample.from_primitive(example.to_primitive())
    assert restored.qualified is True
    assert restored.reason == "student was adequate"


# -- reliability ------------------------------------------------------------
def test_reliability_ranks_seeds_above_generated_thoughts():
    assert BASE_RELIABILITY[SourceKind.CORPUS] > BASE_RELIABILITY[SourceKind.SELF]


def test_a_confident_guess_is_not_a_fact_without_verification():
    guess = score(source=Source(SourceKind.SELF, "mind"), confidence=1.0, verified=False)
    assert not is_trusted(guess)
    checked = score(source=Source(SourceKind.CORPUS, "corpus"), confidence=0.9, verified=True)
    assert is_trusted(checked)


def test_corroboration_raises_a_score_but_stays_bounded():
    low = score(source=Source(SourceKind.INTERNET, "web"), confidence=0.5, corroborations=0)
    high = score(source=Source(SourceKind.INTERNET, "web"), confidence=0.5, corroborations=3)
    assert high.score > low.score
    assert score(
        source=Source(SourceKind.CORPUS, "c"), confidence=1.0, corroborations=99, verified=True
    ).score <= 1.0


def test_reliability_rejects_an_out_of_range_confidence():
    try:
        score(source=None, confidence=1.5)
    except ValueError:
        pass
    else:  # pragma: no cover
        raise AssertionError("confidence outside [0, 1] must raise")


def test_trust_threshold_is_the_documented_value():
    assert TRUST_THRESHOLD == 0.75


# -- storage ----------------------------------------------------------------
def test_idea_store_persists_and_is_bounded(tmp_path, monkeypatch):
    monkeypatch.setenv("FAYD_DATA_DIR", str(tmp_path))
    store = IdeaStore(max_items=3)
    for i in range(5):
        store.add(Idea(text=f"idea {i}", kind=IdeaKind.IDEA))
    reopened = IdeaStore(max_items=3)
    assert reopened.count() == 3
    assert reopened.recent(1)[0].text == "idea 4"


def test_fact_store_survives_a_restart(tmp_path, monkeypatch):
    monkeypatch.setenv("FAYD_DATA_DIR", str(tmp_path))
    store = FactStore()
    store.learn("f1", "waterflood raises pressure")
    reopened = FactStore()
    assert reopened.all() == (("f1", "waterflood raises pressure"),)


def test_fact_store_replaces_a_repeated_id(tmp_path, monkeypatch):
    monkeypatch.setenv("FAYD_DATA_DIR", str(tmp_path))
    store = FactStore()
    store.learn("f1", "first")
    store.learn("f1", "second")
    assert store.all() == (("f1", "second"),)


def test_example_store_separates_qualified_from_rejected(tmp_path, monkeypatch):
    monkeypatch.setenv("FAYD_DATA_DIR", str(tmp_path))
    store = ExampleStore()
    store.add(TrainingExample(task="a", answer="b", qualified=True))
    store.add(TrainingExample(task="c", answer="d", qualified=False))
    assert len(store.qualified()) == 1
    assert len(store.rejected()) == 1


def test_meta_round_trips_and_is_best_effort(tmp_path, monkeypatch):
    monkeypatch.setenv("FAYD_DATA_DIR", str(tmp_path))
    meta_set("resource_mode", "light")
    assert meta_get("resource_mode") == "light"
    assert meta_get("missing", "fallback") == "fallback"


def test_stores_tolerate_a_corrupt_document(tmp_path, monkeypatch):
    monkeypatch.setenv("FAYD_DATA_DIR", str(tmp_path))
    from storage.json_store import JsonStore

    JsonStore(tmp_path).save("cognition/ideas", "all", "not-a-list")
    assert IdeaStore().count() == 0
    JsonStore(tmp_path).save("cognition/facts", "all", [{"id": "x"}, "bad-row"])
    assert FactStore().all() == (("x", ""),)


def test_cycle_and_experience_stores_round_trip(tmp_path, monkeypatch):
    monkeypatch.setenv("FAYD_DATA_DIR", str(tmp_path))
    CycleStore().add(ThoughtCycle(notes=("n",)))
    ExperienceStore().add(Experience(expectation="a", outcome="b", lesson="c"))
    assert CycleStore().count() == 1
    assert ExperienceStore().count() == 1


def test_source_round_trips():
    source = Source(SourceKind.TEACHER, "local teacher")
    restored = Source.from_primitive(source.to_primitive())
    assert restored.kind is SourceKind.TEACHER
    assert restored.reference == "local teacher"


def test_reliability_record_round_trips():
    reliability = Reliability(score=0.8, verified=True)
    assert Reliability.from_primitive(reliability.to_primitive()).verified is True
