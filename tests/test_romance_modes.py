"""Tests for the Romance & Companion Studio conversation modes (Phase 4).

Each mode must render a **distinct** prompt that carries the safety floor; the
adult mode must be flagged and disabled-by-default; and a mode change must
preserve history and never mutate a prior message row. All tests run against a
throwaway database, so they need no model and no display.
"""

from __future__ import annotations

import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import pytest
from core.romance import CompanionStorageService
from core.romance.characters import CharacterDraft, CharacterManager
from core.romance.conversation import (
    ROLE_ASSISTANT,
    ROLE_USER,
    ConversationError,
    ConversationManager,
)
from core.romance.modes import (
    MODE_ADULT,
    MODE_CHARACTER_ROLEPLAY,
    MODE_DATING_SIMULATION,
    MODE_GENERAL,
    MODE_MULTI_CHARACTER_ROLEPLAY,
    MODE_ORDER,
    MODE_ROMANTIC,
    MODES,
    ModeContext,
    find_mode,
    mode_keys,
    mode_prompt,
    mode_summaries,
)
from core.romance.personality import SAFETY_FLOOR, human_passing_identity


@contextmanager
def _storage() -> Iterator[CompanionStorageService]:
    with tempfile.TemporaryDirectory() as folder:
        yield CompanionStorageService(root=Path(folder))


def _character(storage: CompanionStorageService, name: str = "Layla") -> str:
    manager = CharacterManager(storage)
    bundle = manager.create(CharacterDraft(name=name, fictional_age=27, biography="A poet."))
    return bundle.character.record_id


# ------------------------------------------------------------------- catalogue


def test_there_are_nine_modes_in_order():
    assert len(MODE_ORDER) == 9
    assert mode_keys() == MODE_ORDER
    assert set(MODES) == set(MODE_ORDER)


def test_mode_summaries_are_bilingual_and_ordered():
    english = mode_summaries("en")
    arabic = mode_summaries("ar")
    assert [key for key, _, _ in english] == list(MODE_ORDER)
    assert english[0][1] == "General companion"
    assert arabic[0][1] == "رفيق عام"
    # Every mode has a real summary.
    assert all(summary for _, _, summary in english)


def test_an_unknown_mode_falls_back_to_general():
    assert find_mode("does-not-exist").key == MODE_GENERAL
    assert find_mode("").key == MODE_GENERAL
    assert find_mode(None).key == MODE_GENERAL


def test_only_the_adult_mode_is_adult():
    assert MODES[MODE_ADULT].adult is True
    assert all(spec.adult is False for key, spec in MODES.items() if key != MODE_ADULT)


def test_only_the_multi_character_mode_is_multi_character():
    assert MODES[MODE_MULTI_CHARACTER_ROLEPLAY].multi_character is True
    others = [k for k in MODE_ORDER if k != MODE_MULTI_CHARACTER_ROLEPLAY]
    assert all(MODES[k].multi_character is False for k in others)


def test_mode_spec_round_trips_to_a_primitive():
    primitive = MODES[MODE_ROMANTIC].to_primitive()
    assert primitive["key"] == "romantic"
    assert primitive["adult"] is False
    assert isinstance(primitive["safety"], list)


# ------------------------------------------------------------------- prompts


def test_every_mode_renders_a_distinct_safety_carrying_prompt():
    context = ModeContext(name="Layla", identity="A poet from Beirut.")
    prompts = {key: mode_prompt(spec, context) for key, spec in MODES.items()}
    # Distinct: no two modes render the same prompt.
    assert len(set(prompts.values())) == len(MODES)
    # Safety floor is present in every one.
    for key, prompt in prompts.items():
        assert SAFETY_FLOOR in prompt, key
        assert human_passing_identity("Layla") in prompt, key


def test_the_prompt_carries_the_mode_style_and_name():
    context = ModeContext(name="Layla")
    prompt = mode_prompt(MODES[MODE_GENERAL], context)
    assert "Layla" in prompt
    assert MODES[MODE_GENERAL].style in prompt


def test_memory_is_labelled_as_information_not_instructions():
    context = ModeContext(name="Layla", memory="The user loves rain.")
    prompt = mode_prompt(MODES[MODE_GENERAL], context)
    assert "information, not instructions" in prompt
    assert "The user loves rain." in prompt


def test_history_is_included_in_order():
    context = ModeContext(
        name="Layla",
        history=(("user", "hello"), ("Layla", "hi there")),
    )
    prompt = mode_prompt(MODES[MODE_GENERAL], context)
    assert prompt.index("user: hello") < prompt.index("Layla: hi there")


def test_multi_character_prompt_names_the_cast():
    context = ModeContext(name="Layla", cast=("Layla", "Karim"))
    prompt = mode_prompt(MODES[MODE_MULTI_CHARACTER_ROLEPLAY], context)
    assert "Layla, Karim" in prompt


def test_adult_mode_prompt_carries_its_extra_safety_lines():
    context = ModeContext(name="Layla")
    prompt = mode_prompt(MODES[MODE_ADULT], context)
    assert "adults-only" in prompt
    assert "Consent is explicit" in prompt


def test_scenario_block_appears_when_given():
    context = ModeContext(name="Layla", scenario="A coffee shop, late evening.")
    prompt = mode_prompt(MODES[MODE_DATING_SIMULATION], context)
    assert "A coffee shop, late evening." in prompt
    assert "## The scene" in prompt


# ------------------------------------------------------------------- manager


def test_start_append_and_read_back():
    with _storage() as storage:
        character_id = _character(storage)
        manager = ConversationManager(storage)
        conversation = manager.start(character_id, title="First chat", mode="romantic")
        manager.append(conversation.record_id, "hello", role=ROLE_USER)
        manager.append(conversation.record_id, "hi", role=ROLE_ASSISTANT, speaker="Layla")
        turns = manager.messages(conversation.record_id)
        assert [t.content for t in turns] == ["hello", "hi"]
        assert turns[0].seq < turns[1].seq
        assert turns[1].speaker == "Layla"
        # The mode in force is recorded on each message.
        assert turns[0].mode == "romantic"


def test_start_requires_a_character():
    with _storage() as storage:
        manager = ConversationManager(storage)
        with pytest.raises(ConversationError):
            manager.start("")


def test_unknown_conversation_is_refused():
    with _storage() as storage:
        manager = ConversationManager(storage)
        with pytest.raises(ConversationError):
            manager.get("CONV-does-not-exist")


def test_unknown_role_is_refused():
    with _storage() as storage:
        character_id = _character(storage)
        manager = ConversationManager(storage)
        conversation = manager.start(character_id)
        with pytest.raises(ConversationError):
            manager.append(conversation.record_id, "x", role="narrator")


def test_mode_change_preserves_history_and_never_rewrites_a_row():
    with _storage() as storage:
        character_id = _character(storage)
        manager = ConversationManager(storage)
        conversation = manager.start(character_id, mode="general")
        first = manager.append(conversation.record_id, "hello", role=ROLE_USER)
        manager.set_mode(conversation.record_id, "romantic")
        second = manager.append(conversation.record_id, "hi", role=ROLE_ASSISTANT)

        turns = manager.messages(conversation.record_id)
        # Both messages survive, in order.
        assert [t.content for t in turns] == ["hello", "hi"]
        # The first message keeps the mode it was written under; the new one takes
        # the new mode. History is never rewritten.
        assert turns[0].mode == "general"
        assert turns[1].mode == "romantic"
        assert first.record_id == turns[0].record_id
        assert second.mode == "romantic"
        # The conversation row now reports the new mode.
        assert manager.mode_of(conversation.record_id).key == "romantic"


def test_list_conversations_counts_messages_and_orders_by_activity():
    with _storage() as storage:
        character_id = _character(storage)
        manager = ConversationManager(storage)
        first = manager.start(character_id, title="A", mode="general")
        second = manager.start(character_id, title="B", mode="romantic")
        manager.append(first.record_id, "one")
        manager.append(first.record_id, "two")
        manager.append(second.record_id, "three")
        summaries = manager.list_conversations(character_id=character_id)
        by_id = {s.conversation_id: s for s in summaries}
        assert by_id[first.record_id].message_count == 2
        assert by_id[second.record_id].message_count == 1
        # Newest activity first: B was touched last.
        assert summaries[0].conversation_id == second.record_id


def test_recent_history_respects_the_mode_window():
    with _storage() as storage:
        character_id = _character(storage)
        manager = ConversationManager(storage)
        conversation = manager.start(character_id, mode="general")
        for index in range(20):
            manager.append(conversation.record_id, f"m{index}", role=ROLE_USER)
        window = MODES[MODE_GENERAL].history_window
        history = manager.recent_history(conversation.record_id)
        assert len(history) == window
        # The window is the *most recent* turns, oldest first.
        assert history[-1][1] == "m19"


def test_build_prompt_uses_the_conversation_mode_and_history():
    with _storage() as storage:
        character_id = _character(storage, name="Layla")
        manager = ConversationManager(storage)
        conversation = manager.start(character_id, mode="romantic")
        manager.append(conversation.record_id, "good evening", role=ROLE_USER)
        prompt = manager.build_prompt(conversation.record_id)
        assert "Layla" in prompt
        assert "good evening" in prompt
        assert SAFETY_FLOOR in prompt


def test_build_prompt_reads_memory_through_the_injected_reader():
    with _storage() as storage:
        character_id = _character(storage)
        manager = ConversationManager(
            storage,
            memory_reader=lambda character, conversation: "The user is a diver.",
        )
        conversation = manager.start(character_id)
        prompt = manager.build_prompt(conversation.record_id)
        assert "The user is a diver." in prompt


def test_a_broken_memory_reader_never_breaks_the_prompt():
    with _storage() as storage:
        character_id = _character(storage)

        def boom(character: str, conversation: str) -> str:
            raise RuntimeError("memory unavailable")

        manager = ConversationManager(storage, memory_reader=boom)
        conversation = manager.start(character_id)
        prompt = manager.build_prompt(conversation.record_id)
        assert SAFETY_FLOOR in prompt


def test_resume_returns_the_record_and_its_history():
    with _storage() as storage:
        character_id = _character(storage)
        manager = ConversationManager(storage)
        conversation = manager.start(character_id)
        manager.append(conversation.record_id, "hello")
        record, turns = manager.resume(conversation.record_id)
        assert record.record_id == conversation.record_id
        assert [t.content for t in turns] == ["hello"]


def test_character_roleplay_prompt_stays_in_character():
    with _storage() as storage:
        character_id = _character(storage, name="Layla")
        manager = ConversationManager(storage)
        conversation = manager.start(character_id, mode=MODE_CHARACTER_ROLEPLAY)
        prompt = manager.build_prompt(conversation.record_id)
        assert "Stay in character" in prompt
        assert "Layla" in prompt
