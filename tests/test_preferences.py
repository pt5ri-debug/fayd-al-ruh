"""Tests for persistent user preferences."""

from __future__ import annotations

import pytest

from core.preferences import PreferenceRecord, UserPreferences


def test_defaults_are_conservative(tmp_path):
    record = UserPreferences(tmp_path).record
    assert record.user_name == ""
    assert record.language == "en"
    assert record.speak is True
    assert record.wake_word == "jarvis"


def test_update_persists_across_instances(tmp_path):
    preferences = UserPreferences(tmp_path)
    preferences.update(user_name="Sam", language="ar", speak=False, wake_word="computer")
    reloaded = UserPreferences(tmp_path).record
    assert reloaded.user_name == "Sam"
    assert reloaded.language == "ar"
    assert reloaded.speak is False
    assert reloaded.wake_word == "computer"


def test_unknown_preference_is_rejected(tmp_path):
    with pytest.raises(ValueError):
        UserPreferences(tmp_path).update(colour="blue")


def test_unsupported_language_is_rejected(tmp_path):
    with pytest.raises(ValueError):
        UserPreferences(tmp_path).update(language="fr")


def test_corrupt_store_falls_back_to_defaults(tmp_path):
    store = UserPreferences(tmp_path)
    store.store.save("jarvis", "preferences", {"language": "klingon"})
    assert UserPreferences(tmp_path).record == PreferenceRecord()


def test_record_round_trips_through_primitive():
    record = PreferenceRecord(user_name="Noor", language="ar", speak=False, wake_word="jarvis")
    assert PreferenceRecord.from_primitive(record.to_primitive()) == record
