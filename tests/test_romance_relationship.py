"""Tests for the Romance & Companion Studio relationship engine (Phase 5).

Every test runs against a throwaway database. The engine must move the state only
from an explicit event, by a bounded amount; must require evidence for a move;
must be deterministic and idempotent; must never move randomly; and must explain
why the relationship changed.
"""

from __future__ import annotations

import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import pytest
from core.romance import CompanionStorageService
from core.romance.characters import CharacterDraft, CharacterManager
from core.romance.relationship import (
    DIMENSIONS,
    EVENT_KINDS,
    MAX_STEP,
    STAGES,
    RelationshipEngine,
    RelationshipError,
)


@contextmanager
def _storage() -> Iterator[CompanionStorageService]:
    with tempfile.TemporaryDirectory() as folder:
        yield CompanionStorageService(root=Path(folder))


def _character(storage: CompanionStorageService, name: str = "Layla") -> str:
    manager = CharacterManager(storage)
    bundle = manager.create(CharacterDraft(name=name, fictional_age=27))
    return bundle.character.record_id


# ------------------------------------------------------------------ state


def test_a_fresh_relationship_starts_at_the_baseline():
    with _storage() as storage:
        character_id = _character(storage)
        engine = RelationshipEngine(storage)
        snapshot = engine.snapshot(character_id)
        assert snapshot.stage == "stranger"
        assert snapshot.event_count == 0
        # Six dimensions start at 0; stability starts at 0.5.
        assert snapshot.dimensions["trust"] == 0.0
        assert snapshot.dimensions["stability"] == 0.5


def test_the_seven_dimensions_are_exactly_those_scored():
    assert DIMENSIONS == (
        "familiarity",
        "trust",
        "closeness",
        "comfort",
        "affection",
        "shared_experiences",
        "stability",
    )
    with _storage() as storage:
        engine = RelationshipEngine(storage)
        character_id = _character(storage)
        assert set(engine.snapshot(character_id).dimensions) == set(DIMENSIONS)


# ------------------------------------------------------------------ observe


def test_observe_moves_only_the_event_dimensions():
    with _storage() as storage:
        character_id = _character(storage)
        engine = RelationshipEngine(storage)
        change = engine.observe(
            character_id, "personal_share", message_id="MSG-1", conversation_id="CONV-1"
        )
        assert change.deltas  # something moved
        snapshot = engine.snapshot(character_id)
        assert snapshot.dimensions["trust"] > 0.0
        assert snapshot.dimensions["closeness"] > 0.0
        # A dimension the event does not name stays put.
        assert snapshot.dimensions["affection"] == 0.0


def test_a_single_event_never_exceeds_the_bound():
    with _storage() as storage:
        character_id = _character(storage)
        engine = RelationshipEngine(storage)
        change = engine.observe(character_id, "shared_experience", message_id="MSG-1")
        assert all(abs(delta) <= MAX_STEP for delta in change.deltas.values())


def test_an_unknown_event_is_refused():
    with _storage() as storage:
        character_id = _character(storage)
        engine = RelationshipEngine(storage)
        with pytest.raises(RelationshipError):
            engine.observe(character_id, "not-an-event", message_id="MSG-1")


def test_evidence_is_required_for_a_move():
    with _storage() as storage:
        character_id = _character(storage)
        engine = RelationshipEngine(storage)
        # A milestone the user marks, with no evidence: recorded, but moves nothing.
        change = engine.observe(character_id, "milestone", title="A date to remember")
        assert change.deltas == {}
        assert engine.snapshot(character_id).dimensions["closeness"] == 0.0
        # The event is still recorded.
        assert len(engine.events(character_id)) == 1


def test_observe_is_idempotent_for_the_same_message():
    with _storage() as storage:
        character_id = _character(storage)
        engine = RelationshipEngine(storage)
        engine.observe(character_id, "personal_share", message_id="MSG-1")
        first = engine.snapshot(character_id).dimensions["trust"]
        engine.observe(character_id, "personal_share", message_id="MSG-1")
        second = engine.snapshot(character_id).dimensions["trust"]
        assert first == second


def test_the_state_never_leaves_zero_to_one():
    with _storage() as storage:
        character_id = _character(storage)
        engine = RelationshipEngine(storage)
        for index in range(50):
            engine.observe(character_id, "affection_expressed", message_id=f"MSG-{index}")
        snapshot = engine.snapshot(character_id)
        assert all(0.0 <= value <= 1.0 for value in snapshot.dimensions.values())
        assert snapshot.dimensions["affection"] == 1.0


# ------------------------------------------------------------------ no-random


def test_the_state_never_changes_without_an_event():
    with _storage() as storage:
        character_id = _character(storage)
        engine = RelationshipEngine(storage)
        before = engine.snapshot(character_id).dimensions
        # Reading the state repeatedly changes nothing.
        for _ in range(5):
            engine.snapshot(character_id)
        after = engine.snapshot(character_id).dimensions
        assert before == after


def test_conflict_moves_the_relationship_down():
    with _storage() as storage:
        character_id = _character(storage)
        engine = RelationshipEngine(storage)
        engine.observe(character_id, "meaningful_conversation", message_id="MSG-1")
        engine.observe(character_id, "meaningful_conversation", message_id="MSG-2")
        before = engine.snapshot(character_id).dimensions["closeness"]
        engine.observe(character_id, "conflict", message_id="MSG-3")
        after = engine.snapshot(character_id).dimensions["closeness"]
        assert after < before


# ------------------------------------------------------------------ stages


def test_the_stage_rises_as_the_relationship_deepens():
    with _storage() as storage:
        character_id = _character(storage)
        engine = RelationshipEngine(storage)
        assert engine.snapshot(character_id).stage == "stranger"
        # Enough affection to reach a later stage.
        for index in range(20):
            engine.observe(character_id, "affection_expressed", message_id=f"MSG-{index}")
        assert engine.snapshot(character_id).stage != "stranger"


def test_stages_are_configurable():
    with _storage() as storage:
        character_id = _character(storage)
        custom = (("new", "New", 0.0), ("close", "Close", 0.1))
        engine = RelationshipEngine(storage, stages=custom)
        engine.observe(character_id, "affection_expressed", message_id="MSG-1")
        snapshot = engine.snapshot(character_id)
        assert snapshot.stage in {"new", "close"}


def test_default_stages_span_stranger_to_partner():
    assert STAGES[0][0] == "stranger"
    assert STAGES[-1][0] == "romantic_partner"
    thresholds = [entry[2] for entry in STAGES]
    assert thresholds == sorted(thresholds)


# ------------------------------------------------------------------ explain


def test_explain_names_the_events():
    with _storage() as storage:
        character_id = _character(storage)
        engine = RelationshipEngine(storage)
        engine.observe(character_id, "personal_share", message_id="MSG-1", title="Shared a secret")
        text = engine.explain(character_id)
        assert "Shared a secret" in text
        assert "event(s)" in text


def test_explain_on_a_fresh_relationship_is_honest():
    with _storage() as storage:
        character_id = _character(storage)
        engine = RelationshipEngine(storage)
        text = engine.explain(character_id)
        assert "No events yet" in text


# ------------------------------------------------------------------ reset / edit


def test_reset_returns_to_the_starting_point():
    with _storage() as storage:
        character_id = _character(storage)
        engine = RelationshipEngine(storage)
        engine.observe(character_id, "personal_share", message_id="MSG-1")
        engine.reset(character_id)
        snapshot = engine.snapshot(character_id)
        assert snapshot.dimensions["trust"] == 0.0
        assert snapshot.stage == "stranger"


def test_edit_sets_a_dimension_directly():
    with _storage() as storage:
        character_id = _character(storage)
        engine = RelationshipEngine(storage)
        snapshot = engine.edit(character_id, affection=0.8)
        assert snapshot.dimensions["affection"] == 0.8


def test_edit_clamps_and_refuses_unknown_dimensions():
    with _storage() as storage:
        character_id = _character(storage)
        engine = RelationshipEngine(storage)
        assert engine.edit(character_id, affection=5.0).dimensions["affection"] == 1.0
        with pytest.raises(RelationshipError):
            engine.edit(character_id, not_a_dimension=0.5)


# ------------------------------------------------------------------ isolation


def test_each_character_has_an_independent_relationship():
    with _storage() as storage:
        first = _character(storage, name="Layla")
        second = _character(storage, name="Karim")
        engine = RelationshipEngine(storage)
        engine.observe(first, "personal_share", message_id="MSG-1")
        assert engine.snapshot(first).dimensions["trust"] > 0.0
        assert engine.snapshot(second).dimensions["trust"] == 0.0


def test_event_kinds_are_the_declared_set():
    assert "first_conversation" in EVENT_KINDS
    assert "conflict" in EVENT_KINDS
    assert "reconciliation" in EVENT_KINDS
