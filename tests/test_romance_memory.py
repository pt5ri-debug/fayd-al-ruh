"""Tests for the Romance & Companion Studio memory system (Phase 6).

Every test runs against a throwaway database. The memory system must isolate
per character, de-duplicate, rank honestly (relevance + recency + importance),
never treat a memory as a certain fact, keep a correction as a *new* record, and
retain only when explicitly asked.
"""

from __future__ import annotations

import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import pytest
from core.romance import CompanionStorageService
from core.romance.characters import CharacterDraft, CharacterManager
from core.romance.memory import (
    KIND_EPISODIC,
    KIND_RELATIONSHIP,
    KIND_SEMANTIC,
    MEMORY_HEADER,
    MEMORY_KINDS,
    CompanionMemoryManager,
    MemoryError,
)


@contextmanager
def _storage() -> Iterator[CompanionStorageService]:
    with tempfile.TemporaryDirectory() as folder:
        yield CompanionStorageService(root=Path(folder))


def _character(storage: CompanionStorageService, name: str = "Layla") -> str:
    manager = CharacterManager(storage)
    return manager.create(CharacterDraft(name=name, fictional_age=27)).character.record_id


# ------------------------------------------------------------------ kinds


def test_the_four_durable_kinds_are_declared():
    assert MEMORY_KINDS == (
        "episodic",
        "semantic_personal",
        "relationship",
        "character",
    )


def test_an_unknown_kind_is_refused():
    with _storage() as storage:
        character_id = _character(storage)
        manager = CompanionMemoryManager(storage)
        with pytest.raises(MemoryError):
            manager.add(character_id, "not-a-kind", "x")


def test_a_memory_needs_content():
    with _storage() as storage:
        character_id = _character(storage)
        manager = CompanionMemoryManager(storage)
        with pytest.raises(MemoryError):
            manager.add(character_id, KIND_EPISODIC, "   ")


# ------------------------------------------------------------------ writing


def test_add_stores_a_memory_with_provenance():
    with _storage() as storage:
        character_id = _character(storage)
        manager = CompanionMemoryManager(storage)
        memory = manager.add(
            character_id,
            KIND_SEMANTIC,
            "The user is a petroleum engineer.",
            importance=0.8,
            source="user",
            source_message_id="MSG-1",
        )
        assert memory.content == "The user is a petroleum engineer."
        assert memory.importance == 0.8
        assert memory.source_message_id == "MSG-1"
        # A fresh memory is never assumed certain.
        assert memory.confidence == 0.5


def test_an_identical_memory_is_not_stored_twice():
    with _storage() as storage:
        character_id = _character(storage)
        manager = CompanionMemoryManager(storage)
        first = manager.add(character_id, KIND_SEMANTIC, "The user loves rain.")
        second = manager.add(character_id, KIND_SEMANTIC, "  the user loves rain.  ")
        assert first.record_id == second.record_id
        assert len(manager.memories(character_id)) == 1


def test_a_fiction_memory_is_marked_fictional():
    with _storage() as storage:
        character_id = _character(storage)
        manager = CompanionMemoryManager(storage)
        memory = manager.add(
            character_id, KIND_EPISODIC, "We danced in the rain.", fictional=True
        )
        assert memory.fictional is True


# ------------------------------------------------------------------ isolation


def test_each_character_has_its_own_memories():
    with _storage() as storage:
        first = _character(storage, name="Layla")
        second = _character(storage, name="Karim")
        manager = CompanionMemoryManager(storage)
        manager.add(first, KIND_SEMANTIC, "The user is a diver.")
        assert len(manager.memories(first)) == 1
        assert manager.memories(second) == ()
        assert manager.retrieve(second, "diver") == ()


# ------------------------------------------------------------------ retrieval


def test_retrieve_ranks_by_relevance():
    with _storage() as storage:
        character_id = _character(storage)
        manager = CompanionMemoryManager(storage)
        manager.add(character_id, KIND_SEMANTIC, "The user is a diver.", importance=0.5)
        manager.add(character_id, KIND_SEMANTIC, "The user loves rain.", importance=0.5)
        top = manager.retrieve(character_id, "diver")
        assert top
        assert top[0].content == "The user is a diver."


def test_retrieve_returns_nothing_for_a_blank_character():
    with _storage() as storage:
        character_id = _character(storage)
        manager = CompanionMemoryManager(storage)
        assert manager.retrieve(character_id, "anything") == ()


def test_retrieve_respects_the_limit():
    with _storage() as storage:
        character_id = _character(storage)
        manager = CompanionMemoryManager(storage)
        for index in range(10):
            manager.add(character_id, KIND_EPISODIC, f"event number {index}")
        assert len(manager.retrieve(character_id, "event", limit=3)) == 3
        assert manager.retrieve(character_id, "event", limit=0) == ()


def test_a_recalled_memory_carries_its_kind_and_score():
    with _storage() as storage:
        character_id = _character(storage)
        manager = CompanionMemoryManager(storage)
        manager.add(character_id, KIND_RELATIONSHIP, "We met at a coffee shop.")
        recalled = manager.retrieve(character_id, "coffee")[0]
        assert recalled.kind == KIND_RELATIONSHIP
        assert recalled.score > 0.0
        assert "something" in recalled.line() or "relationship" in recalled.line()


def test_reference_block_is_labelled_and_bounded():
    with _storage() as storage:
        character_id = _character(storage)
        manager = CompanionMemoryManager(storage)
        manager.add(character_id, KIND_SEMANTIC, "The user is a diver.")
        block = manager.reference_block(character_id, "diver", max_chars=200)
        assert MEMORY_HEADER in block
        assert "diver" in block


def test_reference_block_is_empty_when_there_is_nothing():
    with _storage() as storage:
        character_id = _character(storage)
        manager = CompanionMemoryManager(storage)
        assert manager.reference_block(character_id, "x") == ""


# ------------------------------------------------------------------ correction


def test_supersede_writes_a_new_record_and_keeps_the_original():
    with _storage() as storage:
        character_id = _character(storage)
        manager = CompanionMemoryManager(storage)
        original = manager.add(character_id, KIND_SEMANTIC, "The user is a teacher.")
        replacement = manager.supersede(original.record_id, "The user is an engineer.")
        # A new record; the original is untouched but points at the new one.
        assert replacement.record_id != original.record_id
        stored_original = manager.get(original.record_id)
        assert stored_original.content == "The user is a teacher."
        assert stored_original.superseded_by == replacement.record_id
        # The active set hides the superseded memory.
        active = [m.record_id for m in manager.memories(character_id)]
        assert replacement.record_id in active
        assert original.record_id not in active


def test_edit_and_forget():
    with _storage() as storage:
        character_id = _character(storage)
        manager = CompanionMemoryManager(storage)
        memory = manager.add(character_id, KIND_EPISODIC, "A walk in the park.")
        edited = manager.edit(memory.record_id, importance=0.9)
        assert edited.importance == 0.9
        manager.forget(memory.record_id)
        assert manager.memories(character_id) == ()


def test_get_refuses_an_unknown_memory():
    with _storage() as storage:
        manager = CompanionMemoryManager(storage)
        with pytest.raises(MemoryError):
            manager.get("MEM-does-not-exist")


# ------------------------------------------------------------------ export / retention


def test_export_is_plain_data():
    with _storage() as storage:
        character_id = _character(storage)
        manager = CompanionMemoryManager(storage)
        manager.add(character_id, KIND_SEMANTIC, "The user is a diver.")
        export = manager.export(character_id)
        assert export.character_id == character_id
        assert export.memories[0]["content"] == "The user is a diver."
        assert export.to_primitive()["character_id"] == character_id


def test_retention_is_opt_in_and_keeps_the_most_important():
    with _storage() as storage:
        character_id = _character(storage)
        manager = CompanionMemoryManager(storage)
        for index in range(6):
            manager.add(
                character_id, KIND_EPISODIC, f"event {index}", importance=index / 10
            )
        # Nothing is forgotten until the caller asks.
        assert len(manager.memories(character_id)) == 6
        forgotten = manager.apply_retention(character_id, keep=2)
        assert forgotten == 4
        kept = manager.memories(character_id)
        assert len(kept) == 2
        # The two most important survive.
        assert {m.content for m in kept} == {"event 5", "event 4"}


def test_retention_below_the_count_is_a_noop():
    with _storage() as storage:
        character_id = _character(storage)
        manager = CompanionMemoryManager(storage)
        manager.add(character_id, KIND_EPISODIC, "one")
        assert manager.apply_retention(character_id, keep=5) == 0


def test_retention_refuses_a_negative_keep():
    with _storage() as storage:
        character_id = _character(storage)
        manager = CompanionMemoryManager(storage)
        with pytest.raises(MemoryError):
            manager.apply_retention(character_id, keep=-1)
