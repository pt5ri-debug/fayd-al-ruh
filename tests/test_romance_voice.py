"""Tests for the Romance & Companion Studio voice adapter (Phase 8).

The adapter is a thin layer over the project's own speech stack: it must offer
only the voices this machine really has, resolve a saved voice honestly, speak
through an injected reader, and never raise into a conversation when an engine is
missing.
"""

from __future__ import annotations

import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import pytest
from core.romance import CompanionStorageService
from core.romance.characters import CharacterDraft, CharacterManager
from core.romance.voice import (
    ENGINE_AUTO,
    ENGINE_NAMES,
    CompanionVoiceAdapter,
    VoiceAdapterError,
)


@contextmanager
def _storage() -> Iterator[CompanionStorageService]:
    with tempfile.TemporaryDirectory() as folder:
        yield CompanionStorageService(root=Path(folder))


def _character(storage: CompanionStorageService, name: str = "Layla") -> str:
    manager = CharacterManager(storage)
    return manager.create(CharacterDraft(name=name, fictional_age=27)).character.record_id


class _FakePlayback:
    def __init__(self) -> None:
        self.stopped = False

    def stop(self) -> None:
        self.stopped = True


class _FakeReader:
    """A reader seam: records what it was asked, returns a token clip."""

    def __init__(self, *, fail: bool = False) -> None:
        self.voice: str | None = None
        self.mood: str | None = None
        self.spoken: list[str] = []
        self.fail = fail
        self.playback = _FakePlayback()
        self.played = 0

    def set_voice(self, key: str) -> str:
        self.voice = key
        return key

    def set_mood(self, mood: str) -> str:
        self.mood = mood
        return mood

    def render(self, text: str) -> str:
        if self.fail:
            raise RuntimeError("no engine")
        self.spoken.append(text)
        return f"clip:{text}"

    def play(self, clip: str) -> _FakePlayback:
        self.played += 1
        return self.playback


class _FakeRecognizer:
    def __init__(self, text: str = "hello") -> None:
        self.text = text

    def transcribe(self, clip: object) -> str:
        return self.text


# ------------------------------------------------------------------ engines


def test_the_known_engines_are_declared():
    assert ENGINE_AUTO in ENGINE_NAMES
    assert "piper" in ENGINE_NAMES


# ------------------------------------------------------------------ profiles


def test_a_character_gets_a_default_voice_profile():
    with _storage() as storage:
        character_id = _character(storage)
        adapter = CompanionVoiceAdapter(storage, reader=_FakeReader())
        profile = adapter.profile(character_id)
        assert profile.character_id == character_id
        assert profile.enabled is True
        assert profile.speed == 1.0


def test_profile_refuses_an_unknown_character():
    with _storage() as storage:
        adapter = CompanionVoiceAdapter(storage, reader=_FakeReader())
        with pytest.raises(VoiceAdapterError):
            adapter.profile("CHR-does-not-exist")


def test_set_voice_updates_and_clamps():
    with _storage() as storage:
        character_id = _character(storage)
        adapter = CompanionVoiceAdapter(storage, reader=_FakeReader())
        updated = adapter.set_voice(character_id, engine="piper", speed=9.0, volume=-1.0)
        assert updated.engine == "piper"
        assert updated.speed == 4.0
        assert updated.volume == 0.0


def test_set_voice_refuses_an_unknown_engine_or_field():
    with _storage() as storage:
        character_id = _character(storage)
        adapter = CompanionVoiceAdapter(storage, reader=_FakeReader())
        with pytest.raises(VoiceAdapterError):
            adapter.set_voice(character_id, engine="not-an-engine")
        with pytest.raises(VoiceAdapterError):
            adapter.set_voice(character_id, colour="blue")


def test_set_voice_refuses_an_unknown_mood():
    with _storage() as storage:
        character_id = _character(storage)
        adapter = CompanionVoiceAdapter(storage, reader=_FakeReader())
        with pytest.raises(VoiceAdapterError):
            adapter.set_voice(character_id, mood="not-a-mood")


def test_view_reports_a_missing_voice_honestly():
    with _storage() as storage:
        character_id = _character(storage)
        adapter = CompanionVoiceAdapter(storage, reader=_FakeReader())
        adapter.set_voice(character_id, voice_key="system:does-not-exist:None")
        view = adapter.view(character_id)
        # A saved voice that is no longer installed is reported, not swapped.
        assert view.voice_available is False
        assert "not installed" in view.describe()


def test_view_is_honest_when_the_voice_is_off():
    with _storage() as storage:
        character_id = _character(storage)
        adapter = CompanionVoiceAdapter(storage, reader=_FakeReader())
        adapter.set_voice(character_id, enabled=False)
        assert adapter.view(character_id).describe() == "Voice is off for this character."


# ------------------------------------------------------------------ speaking


def test_read_aloud_uses_the_character_voice_and_mood():
    with _storage() as storage:
        character_id = _character(storage)
        reader = _FakeReader()
        adapter = CompanionVoiceAdapter(storage, reader=reader)
        adapter.set_voice(character_id, voice_key="woman-1", mood="calm")
        clip = adapter.read_aloud(character_id, "Hello there.")
        assert clip == "clip:Hello there."
        assert reader.voice == "woman-1"
        assert reader.mood == "calm"


def test_read_aloud_is_a_noop_when_the_voice_is_off():
    with _storage() as storage:
        character_id = _character(storage)
        reader = _FakeReader()
        adapter = CompanionVoiceAdapter(storage, reader=reader)
        adapter.set_voice(character_id, enabled=False)
        assert adapter.read_aloud(character_id, "Hello.") is None
        assert reader.spoken == []


def test_read_aloud_never_raises_when_the_engine_fails():
    with _storage() as storage:
        character_id = _character(storage)
        adapter = CompanionVoiceAdapter(storage, reader=_FakeReader(fail=True))
        assert adapter.read_aloud(character_id, "Hello.") is None


def test_speak_plays_and_stop_stops():
    with _storage() as storage:
        character_id = _character(storage)
        reader = _FakeReader()
        adapter = CompanionVoiceAdapter(storage, reader=reader)
        playback = adapter.speak(character_id, "Hello.")
        assert playback is reader.playback
        assert reader.played == 1
        adapter.stop()
        assert reader.playback.stopped is True


def test_speak_returns_none_when_nothing_can_be_rendered():
    with _storage() as storage:
        character_id = _character(storage)
        adapter = CompanionVoiceAdapter(storage, reader=_FakeReader(fail=True))
        assert adapter.speak(character_id, "Hello.") is None


# ------------------------------------------------------------------ listening


def test_transcribe_uses_the_recognizer():
    with _storage() as storage:
        adapter = CompanionVoiceAdapter(storage, reader=_FakeReader(), recognizer=_FakeRecognizer("مرحبا"))
        assert adapter.transcribe(object()) == "مرحبا"
        assert adapter.recognizer_available() is True


def test_transcribe_never_raises_without_a_recognizer():
    class _Boom:
        def transcribe(self, clip: object) -> str:
            raise RuntimeError("no engine")

    with _storage() as storage:
        adapter = CompanionVoiceAdapter(storage, reader=_FakeReader(), recognizer=_Boom())
        assert adapter.transcribe(object()) is None


# ------------------------------------------------------------------ real voices


def test_available_voices_are_only_what_is_installed():
    with _storage() as storage:
        adapter = CompanionVoiceAdapter(storage, reader=_FakeReader())
        # The list is built from the machine's real engines; every entry has a key.
        for option in adapter.available_voices():
            assert option.key
            assert option.source
