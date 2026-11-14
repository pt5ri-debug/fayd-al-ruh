"""Tests for the Romance & Companion Studio orchestrator (Phase 10).

Every test runs against a throwaway database and an injected generation seam, so
the whole orchestrator is exercised with **no model and no GPU**. The rules it
pins: the safety floor is checked before anything is written; a user's message is
kept even when generation fails; a reply is written only when one was really
produced; a multi-character scene generates sequentially and never lets one
character impersonate another.
"""

from __future__ import annotations

import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import pytest
from core.romance import CompanionStorageService
from core.romance.characters import CharacterDraft, CharacterManager
from core.romance.conversation import ConversationManager
from core.romance.orchestrator import (
    CompanionOrchestrator,
    OrchestratorError,
)
from core.romance.safety import CompanionSafetyManager


@contextmanager
def _storage() -> Iterator[CompanionStorageService]:
    with tempfile.TemporaryDirectory() as folder:
        yield CompanionStorageService(root=Path(folder))


def _character(storage: CompanionStorageService, name: str = "Layla", *, age: int = 27) -> str:
    manager = CharacterManager(storage)
    return manager.create(CharacterDraft(name=name, fictional_age=age)).character.record_id


class _Seam:
    """A generation seam: a scripted list of replies, one per call."""

    def __init__(self, *replies: str) -> None:
        self.replies = list(replies) or ["Hello."]
        self.calls: list[str] = []

    def __call__(self, prompt: str, key, handle) -> tuple[str, bool, str]:
        self.calls.append(prompt)
        text = self.replies[min(len(self.calls) - 1, len(self.replies) - 1)]
        return text, True, ""


def _orchestrator(storage: CompanionStorageService, seam=None, **kwargs):
    return CompanionOrchestrator(storage, generator=seam or _Seam(), **kwargs)


# -- one turn -------------------------------------------------------------


def test_a_turn_appends_the_message_and_the_reply():
    with _storage() as storage:
        character_id = _character(storage)
        conversations = ConversationManager(storage)
        conversation = conversations.start(character_id)
        orch = _orchestrator(storage)
        result = orch.send(conversation.record_id, "Hello there.")
        assert result.ok and result.reply == "Hello."
        turns = conversations.messages(conversation.record_id)
        assert [t.role for t in turns] == ["user", "assistant"]
        assert turns[0].content == "Hello there."
        assert turns[1].speaker == "Layla"


def test_the_prompt_names_the_character_as_a_person():
    with _storage() as storage:
        character_id = _character(storage, "Noor")
        conversations = ConversationManager(storage)
        conversation = conversations.start(character_id)
        seam = _Seam("Hi.")
        orch = _orchestrator(storage, seam)
        orch.send(conversation.record_id, "Who are you?")
        assert seam.calls, "the seam must have been called"
        prompt = seam.calls[0]
        assert "Noor" in prompt
        # The human-passing identity is in every prompt: the character believes it
        # is human and treats the user as a fellow human being with feelings.
        assert "never describe yourself as a program" in prompt.lower()
        assert "You are a human being" in prompt
        assert "a fellow human being" in prompt


def test_a_blank_message_is_refused():
    with _storage() as storage:
        character_id = _character(storage)
        conversation = ConversationManager(storage).start(character_id)
        orch = _orchestrator(storage)
        with pytest.raises(OrchestratorError):
            orch.send(conversation.record_id, "   ")


def test_a_generation_failure_keeps_the_users_message_and_writes_no_reply():
    with _storage() as storage:
        character_id = _character(storage)
        conversations = ConversationManager(storage)
        conversation = conversations.start(character_id)

        def failing(prompt, key, handle):
            return "", False, "no model is installed"

        orch = CompanionOrchestrator(storage, generator=failing)
        result = orch.send(conversation.record_id, "Are you there?")
        assert result.ok is False and "no model" in result.error
        turns = conversations.messages(conversation.record_id)
        assert [t.role for t in turns] == ["user"]
        assert turns[0].content == "Are you there?"


def test_no_router_and_no_seam_reports_honestly():
    with _storage() as storage:
        character_id = _character(storage)
        conversation = ConversationManager(storage).start(character_id)
        orch = CompanionOrchestrator(storage)  # no generator, no router
        result = orch.send(conversation.record_id, "Hello.")
        assert result.ok is False and "no language model" in result.error


def test_the_relationship_moves_with_the_conversation():
    with _storage() as storage:
        character_id = _character(storage)
        conversation = ConversationManager(storage).start(character_id)
        orch = _orchestrator(storage)
        first = orch.send(conversation.record_id, "Hi, I am glad to meet you.")
        assert first.relationship is not None
        # The first turn is the first conversation: familiarity moved.
        assert first.relationship.dimensions["familiarity"] > 0.0
        second = orch.send(conversation.record_id, "Tell me about your day.")
        assert second.relationship.dimensions["familiarity"] >= first.relationship.dimensions["familiarity"]


def test_the_relationship_event_carries_the_message_as_evidence():
    with _storage() as storage:
        character_id = _character(storage)
        conversation = ConversationManager(storage).start(character_id)
        orch = _orchestrator(storage)
        orch.send(conversation.record_id, "Hello.")
        events = orch.relationships.events(character_id)
        assert events and events[0].message_id
        assert events[0].kind == "first_conversation"


# -- multi-character ------------------------------------------------------


def test_a_multi_character_turn_generates_sequentially_with_distinct_speakers():
    with _storage() as storage:
        first = _character(storage, "Layla")
        second = _character(storage, "Karim")
        conversations = ConversationManager(storage)
        conversation = conversations.start(first, mode="multi_character_roleplay")
        seam = _Seam("Layla speaks.", "Karim speaks.")
        orch = _orchestrator(storage, seam)
        result = orch.send_multi(conversation.record_id, "Good evening.", cast=(first, second))
        assert result.ok
        assert result.speakers == ("Layla", "Karim")
        assert len(seam.calls) == 2  # one generation per character, sequentially
        assert result.replies[0].content == "Layla speaks."
        assert result.replies[1].content == "Karim speaks."


def test_a_multi_character_turn_never_lets_a_character_impersonate_another():
    with _storage() as storage:
        first = _character(storage, "Layla")
        second = _character(storage, "Karim")
        conversations = ConversationManager(storage)
        conversation = conversations.start(first, mode="multi_character_roleplay")
        orch = _orchestrator(storage, _Seam("One.", "Two."))
        orch.send_multi(conversation.record_id, "Hello.", cast=(first, second))
        turns = conversations.messages(conversation.record_id)
        speakers = [t.speaker for t in turns if t.role == "assistant"]
        assert speakers == ["Layla", "Karim"]


def test_a_multi_character_turn_deduplicates_the_cast():
    with _storage() as storage:
        first = _character(storage, "Layla")
        conversations = ConversationManager(storage)
        conversation = conversations.start(first, mode="multi_character_roleplay")
        seam = _Seam("Alone.")
        orch = _orchestrator(storage, seam)
        result = orch.send_multi(conversation.record_id, "Hi.", cast=(first, first))
        assert result.speakers == ("Layla",)
        assert len(seam.calls) == 1


def test_a_multi_character_turn_stops_at_the_first_failed_generation():
    with _storage() as storage:
        first = _character(storage, "Layla")
        second = _character(storage, "Karim")
        conversations = ConversationManager(storage)
        conversation = conversations.start(first, mode="multi_character_roleplay")

        calls = {"n": 0}

        def flaky(prompt, key, handle):
            calls["n"] += 1
            if calls["n"] == 1:
                return "Layla speaks.", True, ""
            return "", False, "out of memory"

        orch = CompanionOrchestrator(storage, generator=flaky)
        result = orch.send_multi(conversation.record_id, "Hi.", cast=(first, second))
        assert result.ok is False and "out of memory" in result.error
        # The one reply that succeeded is kept; the failed one is not invented.
        assert len(result.replies) == 1


# -- the safety floor -----------------------------------------------------


def test_an_adult_turn_is_refused_when_adult_mode_is_off():
    with _storage() as storage:
        character_id = _character(storage, age=27)
        conversation = ConversationManager(storage).start(character_id, mode="adult")
        safety = CompanionSafetyManager(storage)
        orch = CompanionOrchestrator(storage, generator=_Seam("x"), safety=safety)
        result = orch.send(conversation.record_id, "Let us be close.", mode="adult")
        assert result.refused is True
        # A refusal is not written as a reply, and the message is not kept either.
        assert ConversationManager(storage).messages(conversation.record_id) == ()


def test_a_minor_character_cannot_reach_an_adult_turn_even_with_the_mode_on():
    with _storage() as storage:
        character_id = _character(storage, age=16)
        conversation = ConversationManager(storage).start(character_id, mode="adult")
        safety = CompanionSafetyManager(storage)
        safety.set_adult_mode(True, confirmed=True)
        orch = CompanionOrchestrator(storage, generator=_Seam("x"), safety=safety)
        result = orch.send(conversation.record_id, "Hello.", mode="adult")
        assert result.refused is True


def test_the_safety_floor_fails_closed_and_never_crashes_a_turn():
    with _storage() as storage:
        character_id = _character(storage)
        conversation = ConversationManager(storage).start(character_id)

        class _Boom:
            def check_content(self, *args, **kwargs):
                raise RuntimeError("safety exploded")

        orch = CompanionOrchestrator(storage, generator=_Seam("Hi."), safety=_Boom())
        # A broken safety check must not raise; it fails **closed** (a refusal),
        # which is the honest behaviour for a floor.
        result = orch.send(conversation.record_id, "Hello.")
        assert result.refused is True
        assert "could not be evaluated" in result.reason


# -- resume ---------------------------------------------------------------


def test_a_conversation_resumes_with_its_history():
    with _storage() as storage:
        character_id = _character(storage)
        conversations = ConversationManager(storage)
        conversation = conversations.start(character_id)
        orch = _orchestrator(storage)
        orch.send(conversation.record_id, "First.")
        orch.send(conversation.record_id, "Second.")
        record, turns = conversations.resume(conversation.record_id)
        assert record.record_id == conversation.record_id
        assert len(turns) == 4
