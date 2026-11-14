"""Tests for the Romance & Companion Studio roleplay engine (Phase 7).

Every test runs against a throwaway database. Scenarios are fictional by default,
a cast is explicit and never duplicated, and a session can be paused, resumed,
branched, saved, and restarted -- with a branch being a *new* session that never
rewrites the one it forked from.
"""

from __future__ import annotations

import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import pytest
from core.romance import CompanionStorageService
from core.romance.characters import CharacterDraft, CharacterManager
from core.romance.roleplay import (
    CATEGORIES,
    CATEGORY_COFFEE_SHOP,
    CATEGORY_MULTI,
    CONTENT_GENERAL,
    SESSION_STATES,
    STATE_ACTIVE,
    STATE_ENDED,
    STATE_PAUSED,
    RoleplayEngine,
    RoleplayError,
)


@contextmanager
def _storage() -> Iterator[CompanionStorageService]:
    with tempfile.TemporaryDirectory() as folder:
        yield CompanionStorageService(root=Path(folder))


def _character(storage: CompanionStorageService, name: str = "Layla") -> str:
    manager = CharacterManager(storage)
    return manager.create(CharacterDraft(name=name, fictional_age=27)).character.record_id


# ------------------------------------------------------------------ scenarios


def test_there_are_ten_categories():
    assert len(CATEGORIES) == 10
    assert CATEGORY_COFFEE_SHOP in CATEGORIES
    assert CATEGORY_MULTI in CATEGORIES


def test_create_a_scenario_is_fictional_by_default():
    with _storage() as storage:
        engine = RoleplayEngine(storage)
        scenario = engine.create_scenario(
            "A coffee shop, late evening",
            category=CATEGORY_COFFEE_SHOP,
            location="A quiet café by the sea",
        )
        assert scenario.is_real_place is False
        assert scenario.content_level == CONTENT_GENERAL


def test_a_scenario_needs_a_title():
    with _storage() as storage:
        engine = RoleplayEngine(storage)
        with pytest.raises(RoleplayError):
            engine.create_scenario("   ")


def test_an_unknown_category_or_content_level_is_refused():
    with _storage() as storage:
        engine = RoleplayEngine(storage)
        with pytest.raises(RoleplayError):
            engine.create_scenario("x", category="not-a-category")
        with pytest.raises(RoleplayError):
            engine.create_scenario("x", content_level="not-a-level")


def test_scenario_summaries_carry_the_category_label():
    with _storage() as storage:
        engine = RoleplayEngine(storage)
        engine.create_scenario("Evening", category=CATEGORY_COFFEE_SHOP)
        summary = engine.summaries()[0]
        assert summary.category_label == "Coffee shop meeting"
        assert summary.participant_count == 0


def test_scenarios_can_be_filtered_by_category():
    with _storage() as storage:
        engine = RoleplayEngine(storage)
        engine.create_scenario("A", category=CATEGORY_COFFEE_SHOP)
        engine.create_scenario("B", category=CATEGORY_MULTI)
        assert len(engine.scenarios(category=CATEGORY_MULTI)) == 1


def test_get_scenario_refuses_an_unknown_id():
    with _storage() as storage:
        engine = RoleplayEngine(storage)
        with pytest.raises(RoleplayError):
            engine.get_scenario("SCN-does-not-exist")


# ------------------------------------------------------------------ cast


def test_add_participants_in_order_and_read_the_cast():
    with _storage() as storage:
        first = _character(storage, name="Layla")
        second = _character(storage, name="Karim")
        engine = RoleplayEngine(storage)
        scenario = engine.create_scenario("A journey", category=CATEGORY_MULTI)
        engine.add_participant(scenario.record_id, first, role="traveller")
        engine.add_participant(scenario.record_id, second, role="guide")
        assert engine.cast(scenario.record_id) == (first, second)


def test_a_character_cannot_join_a_scenario_twice():
    with _storage() as storage:
        character_id = _character(storage)
        engine = RoleplayEngine(storage)
        scenario = engine.create_scenario("Solo")
        engine.add_participant(scenario.record_id, character_id)
        with pytest.raises(RoleplayError):
            engine.add_participant(scenario.record_id, character_id)


def test_adding_an_unknown_character_is_refused():
    with _storage() as storage:
        engine = RoleplayEngine(storage)
        scenario = engine.create_scenario("x")
        with pytest.raises(RoleplayError):
            engine.add_participant(scenario.record_id, "CHR-does-not-exist")


# ------------------------------------------------------------------ sessions


def test_start_opens_a_conversation_for_the_session():
    with _storage() as storage:
        character_id = _character(storage)
        engine = RoleplayEngine(storage)
        scenario = engine.create_scenario("A date", category=CATEGORY_COFFEE_SHOP)
        engine.add_participant(scenario.record_id, character_id)
        session = engine.start(scenario.record_id)
        assert session.state == STATE_ACTIVE
        # A conversation was opened for the scene, in the dating mode.
        conversation = engine.conversations.get(session.conversation_id)
        assert conversation.scenario_id == scenario.record_id
        assert conversation.mode == "dating_simulation"


def test_a_scenario_with_no_character_cannot_start():
    with _storage() as storage:
        engine = RoleplayEngine(storage)
        scenario = engine.create_scenario("Empty")
        with pytest.raises(RoleplayError):
            engine.start(scenario.record_id)


def test_pause_resume_end():
    with _storage() as storage:
        character_id = _character(storage)
        engine = RoleplayEngine(storage)
        scenario = engine.create_scenario("Scene")
        engine.add_participant(scenario.record_id, character_id)
        session = engine.start(scenario.record_id)
        assert engine.pause(session.record_id).state == STATE_PAUSED
        assert engine.resume(session.record_id).state == STATE_ACTIVE
        assert engine.end(session.record_id).state == STATE_ENDED
        assert set(SESSION_STATES) == {STATE_ACTIVE, STATE_PAUSED, STATE_ENDED}


def test_branch_is_a_new_session_that_points_back():
    with _storage() as storage:
        character_id = _character(storage)
        engine = RoleplayEngine(storage)
        scenario = engine.create_scenario("Scene")
        engine.add_participant(scenario.record_id, character_id)
        original = engine.start(scenario.record_id)
        branch = engine.branch(original.record_id)
        # A new session; the original is untouched.
        assert branch.record_id != original.record_id
        assert branch.branch_of == original.record_id
        assert engine.get_session(original.record_id).branch_of == ""
        # Both sessions belong to the scenario.
        assert len(engine.sessions(scenario.record_id)) == 2


def test_restart_opens_a_fresh_session():
    with _storage() as storage:
        character_id = _character(storage)
        engine = RoleplayEngine(storage)
        scenario = engine.create_scenario("Scene")
        engine.add_participant(scenario.record_id, character_id)
        first = engine.start(scenario.record_id)
        engine.end(first.record_id)
        second = engine.restart(first.record_id)
        assert second.record_id != first.record_id
        assert second.state == STATE_ACTIVE
        assert second.scenario_id == scenario.record_id


def test_get_session_refuses_an_unknown_id():
    with _storage() as storage:
        engine = RoleplayEngine(storage)
        with pytest.raises(RoleplayError):
            engine.get_session("SES-does-not-exist")


def test_scenario_summary_counts_sessions():
    with _storage() as storage:
        character_id = _character(storage)
        engine = RoleplayEngine(storage)
        scenario = engine.create_scenario("Scene")
        engine.add_participant(scenario.record_id, character_id)
        engine.start(scenario.record_id)
        summary = engine.summaries()[0]
        assert summary.session_count == 1
        assert summary.participant_count == 1
