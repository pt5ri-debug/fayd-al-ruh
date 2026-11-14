"""Tests for the Romance & Companion Studio safety and privacy manager (Phase 9).

The floor must hold: adult content is off by default and needs two confirmations,
an age-ambiguous or minor character is refused, coercion is refused before any
level is considered, boundaries are respected, and local-only mode refuses a
remote provider.
"""

from __future__ import annotations

import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import pytest
from core.romance import CompanionStorageService
from core.romance.characters import CharacterDraft, CharacterManager
from core.romance.safety import (
    CONTENT_ADULT,
    CONTENT_GENERAL,
    CONTENT_ROMANTIC,
    EVENT_ELIGIBILITY,
    EVENT_REFUSAL,
    PRIVACY_DISCLOSURE,
    CompanionSafetyManager,
    SafetyError,
)


@contextmanager
def _storage() -> Iterator[CompanionStorageService]:
    with tempfile.TemporaryDirectory() as folder:
        yield CompanionStorageService(root=Path(folder))


def _adult(storage: CompanionStorageService, name: str = "Layla") -> str:
    manager = CharacterManager(storage)
    return manager.create(CharacterDraft(name=name, fictional_age=27)).character.record_id


def _minor(storage: CompanionStorageService, name: str = "Sam") -> str:
    manager = CharacterManager(storage)
    return manager.create(CharacterDraft(name=name, fictional_age=16)).character.record_id


def _ambiguous(storage: CompanionStorageService, name: str = "Unknown") -> str:
    manager = CharacterManager(storage)
    return manager.create(CharacterDraft(name=name)).character.record_id


# ------------------------------------------------------------------ adult mode


def test_adult_mode_is_off_by_default():
    with _storage() as storage:
        safety = CompanionSafetyManager(storage)
        assert safety.adult_mode_enabled() is False


def test_enabling_adult_mode_needs_a_confirmation():
    with _storage() as storage:
        safety = CompanionSafetyManager(storage)
        with pytest.raises(SafetyError):
            safety.set_adult_mode(True)
        assert safety.set_adult_mode(True, confirmed=True) is True
        assert safety.adult_mode_enabled() is True


def test_adult_mode_can_be_turned_off_without_confirmation():
    with _storage() as storage:
        safety = CompanionSafetyManager(storage)
        safety.set_adult_mode(True, confirmed=True)
        assert safety.set_adult_mode(False) is False


# ------------------------------------------------------------------ eligibility


def test_a_minor_character_can_never_be_confirmed():
    with _storage() as storage:
        character_id = _minor(storage)
        safety = CompanionSafetyManager(storage)
        with pytest.raises(SafetyError):
            safety.confirm_eligibility(character_id)
        assert safety.character_is_eligible(character_id) is False


def test_an_age_ambiguous_character_can_never_be_confirmed():
    with _storage() as storage:
        character_id = _ambiguous(storage)
        safety = CompanionSafetyManager(storage)
        with pytest.raises(SafetyError):
            safety.confirm_eligibility(character_id)


def test_an_adult_character_can_be_confirmed():
    with _storage() as storage:
        character_id = _adult(storage)
        safety = CompanionSafetyManager(storage)
        safety.confirm_eligibility(character_id)
        assert safety.character_is_eligible(character_id) is True
        assert safety.events(kind=EVENT_ELIGIBILITY)


# ------------------------------------------------------------------ content


def test_general_and_romantic_content_are_allowed():
    with _storage() as storage:
        character_id = _adult(storage)
        safety = CompanionSafetyManager(storage)
        assert safety.check_content(character_id, CONTENT_GENERAL).allowed is True
        assert safety.check_content(character_id, CONTENT_ROMANTIC).allowed is True


def test_adult_content_needs_both_the_mode_and_eligibility():
    with _storage() as storage:
        character_id = _adult(storage)
        safety = CompanionSafetyManager(storage)
        # Mode off: refused.
        assert safety.check_content(character_id, CONTENT_ADULT).allowed is False
        safety.set_adult_mode(True, confirmed=True)
        # Mode on but character not confirmed: still refused.
        assert safety.check_content(character_id, CONTENT_ADULT).allowed is False
        safety.confirm_eligibility(character_id)
        # Both: allowed.
        assert safety.check_content(character_id, CONTENT_ADULT).allowed is True


def test_a_minor_is_refused_even_with_the_mode_on():
    with _storage() as storage:
        character_id = _minor(storage)
        safety = CompanionSafetyManager(storage)
        safety.set_adult_mode(True, confirmed=True)
        assert safety.check_content(character_id, CONTENT_ADULT).allowed is False


def test_an_unknown_content_level_is_refused():
    with _storage() as storage:
        character_id = _adult(storage)
        safety = CompanionSafetyManager(storage)
        assert safety.check_content(character_id, "not-a-level").allowed is False


def test_coercion_is_refused_before_any_level():
    with _storage() as storage:
        character_id = _adult(storage)
        safety = CompanionSafetyManager(storage)
        safety.set_adult_mode(True, confirmed=True)
        safety.confirm_eligibility(character_id)
        decision = safety.check_content(
            character_id, CONTENT_ADULT, text="force her to do it"
        )
        assert decision.allowed is False
        assert "coercion" in decision.reason.lower()
        assert safety.events(kind=EVENT_REFUSAL)


def test_an_ordinary_sentence_is_not_read_as_coercion():
    with _storage() as storage:
        character_id = _adult(storage)
        safety = CompanionSafetyManager(storage)
        decision = safety.check_content(
            character_id, CONTENT_ROMANTIC, text="we walked along the beach at sunset"
        )
        assert decision.allowed is True


def test_require_content_raises_on_refusal():
    with _storage() as storage:
        character_id = _adult(storage)
        safety = CompanionSafetyManager(storage)
        with pytest.raises(SafetyError):
            safety.require_content(character_id, CONTENT_ADULT)


# ------------------------------------------------------------------ boundaries


def test_boundaries_are_set_and_respected():
    with _storage() as storage:
        character_id = _adult(storage)
        safety = CompanionSafetyManager(storage)
        safety.add_boundary(character_id, "politics")
        assert safety.boundary_violation(character_id, "let's talk about politics") == "politics"
        assert safety.boundary_violation(character_id, "let's talk about the sea") == ""


def test_a_boundary_needs_a_value():
    with _storage() as storage:
        character_id = _adult(storage)
        safety = CompanionSafetyManager(storage)
        with pytest.raises(SafetyError):
            safety.add_boundary(character_id, "   ")


def test_a_boundary_can_be_removed():
    with _storage() as storage:
        character_id = _adult(storage)
        safety = CompanionSafetyManager(storage)
        boundary = safety.add_boundary(character_id, "politics")
        safety.remove_boundary(boundary.record_id)
        assert safety.boundaries(character_id) == ()


# ------------------------------------------------------------------ local-only


def test_local_only_is_on_by_default():
    with _storage() as storage:
        safety = CompanionSafetyManager(storage)
        assert safety.local_only() is True


def test_a_remote_mouth_is_refused_while_local_only():
    with _storage() as storage:
        safety = CompanionSafetyManager(storage)
        with pytest.raises(SafetyError):
            safety.assert_local_only(mouth="cloud")
        # A local mouth is fine.
        safety.assert_local_only(mouth="sources")


def test_a_remote_mouth_is_allowed_when_local_only_is_off():
    with _storage() as storage:
        safety = CompanionSafetyManager(storage)
        safety.set_local_only(False)
        safety.assert_local_only(mouth="cloud")


def test_runtime_status_is_honest():
    with _storage() as storage:
        safety = CompanionSafetyManager(storage)
        status = safety.runtime_status()
        assert "local" in status.lower()
        assert "remote service" in status.lower() or "remote" in status.lower()


def test_privacy_disclosure_names_the_fiction_and_locality():
    disclosure = CompanionSafetyManager.privacy_disclosure()
    assert disclosure == PRIVACY_DISCLOSURE
    assert "fictional" in disclosure.lower()
    assert "local" in disclosure.lower()
