"""The centralized settings model (Phase 3).

Pins the two rules the phase exists to make checkable: every **global** setting
has exactly one authoritative reader/writer/store, and the **context-dependent**
settings are deliberately *not* centralized. Also pins that the one internet
preference is the same value every surface reads.
"""

from __future__ import annotations

from pathlib import Path

from gui import settings_hub

# ------------------------------------------------------- one authoritative home

def test_every_entry_names_a_real_reader_and_writer() -> None:
    for entry in settings_hub.SETTINGS_ENTRIES:
        assert settings_hub.resolve(entry.reader) is not None, entry.reader
        assert settings_hub.resolve(entry.writer) is not None, entry.writer


def test_writers_are_unique_per_setting() -> None:
    """No two *distinct* settings may claim the same writer (one home each).

    ``internet`` and ``tashkeel`` share ``UserPreferences.update`` on purpose --
    they are two fields of one preferences record -- so the rule is: a writer is
    shared only by settings that live in the *same* store.
    """
    by_writer: dict[str, list[settings_hub.SettingsEntry]] = {}
    for entry in settings_hub.SETTINGS_ENTRIES:
        by_writer.setdefault(entry.writer, []).append(entry)
    for writer, entries in by_writer.items():
        stores = {entry.store for entry in entries}
        assert len(stores) == 1, f"{writer} serves settings in different stores: {stores}"


def test_entry_keys_are_unique_and_lookupable() -> None:
    keys = [entry.key for entry in settings_hub.SETTINGS_ENTRIES]
    assert len(keys) == len(set(keys))
    for key in keys:
        assert settings_hub.entry_by_key(key).key == key


def test_an_unknown_key_raises() -> None:
    import pytest

    with pytest.raises(KeyError):
        settings_hub.entry_by_key("nope")


def test_settings_group_by_their_owning_section() -> None:
    grouped = settings_hub.settings_by_link()
    assert "themes" in grouped and "fonts" in grouped and "voices" in grouped
    assert any(e.key == "theme" for e in grouped["themes"])
    assert any(e.key == "typography" for e in grouped["fonts"])
    assert {e.key for e in grouped["voices"]} >= {"reading_voice", "auto_speak"}


def test_context_dependent_settings_are_named_as_separate() -> None:
    """The boundary is written down: these are *not* global and stay distinct."""
    names = {name for name, _note in settings_hub.CONTEXTUAL_SETTINGS}
    assert {"mouth_profile", "character_config", "agent_capabilities", "task_limits"} <= names
    global_keys = {entry.key for entry in settings_hub.SETTINGS_ENTRIES}
    # None of the contextual settings leaked into the global catalogue.
    assert not (names & global_keys)


def test_the_stores_are_recorded() -> None:
    stores = settings_hub.stores()
    assert stores["theme"] == ("gui", "appearance")
    assert stores["typography"] == ("gui", "typography")
    assert stores["cloud_provider"] == ("cloud", "provider")
    assert stores["internet"] == ("jarvis", "preferences")


# ------------------------------------------------------- the one internet switch


def test_internet_defaults_off_and_round_trips(tmp_path: Path) -> None:
    assert settings_hub.read_internet(tmp_path) is False
    assert settings_hub.write_internet(tmp_path, True) is True
    assert settings_hub.read_internet(tmp_path) is True
    assert settings_hub.write_internet(tmp_path, False) is False
    assert settings_hub.read_internet(tmp_path) is False


def test_internet_status_names_the_paths_it_governs(tmp_path: Path) -> None:
    off = settings_hub.internet_status(tmp_path)
    assert "off" in off
    settings_hub.write_internet(tmp_path, True)
    on = settings_hub.internet_status(tmp_path)
    assert "on" in on


def test_the_training_center_reads_the_one_shared_switch(tmp_path: Path) -> None:
    """Turning the internet on in the hub turns it on for the Training Center."""
    from gui.training_center import TrainingCenterService

    service = TrainingCenterService(root=tmp_path)
    assert service.web_enabled() is False
    settings_hub.write_internet(tmp_path, True)
    assert service.web_enabled() is True
    # And the reverse: the service writes the same shared value.
    service.set_web(False)
    assert settings_hub.read_internet(tmp_path) is False


def test_the_chat_session_reads_the_one_shared_switch(tmp_path: Path, monkeypatch) -> None:
    """A chat session built after the switch is on really searches."""
    from ai.jarvis import build_session
    from core.jarvis import JarvisMode

    monkeypatch.setenv("FAYD_DATA_DIR", str(tmp_path))
    settings_hub.write_internet(tmp_path, True)
    mode = build_session(root=tmp_path, text_only=True)
    assert isinstance(mode, JarvisMode)
    assert mode.web_enabled is True


# ------------------------------------------------------------- the live values


def test_current_values_reads_every_global_setting(tmp_path: Path) -> None:
    values = settings_hub.current_values(tmp_path)
    for entry in settings_hub.SETTINGS_ENTRIES:
        assert entry.key in values
        assert values[entry.key] != ""  # never blank; "unknown" is the honest value


def test_describe_lists_every_setting_and_its_writer(tmp_path: Path) -> None:
    text = settings_hub.describe(tmp_path)
    for entry in settings_hub.SETTINGS_ENTRIES:
        assert entry.title in text
        assert entry.writer in text
    assert "Context-dependent" in text


def test_a_broken_store_still_reports_unknown_never_raises(tmp_path: Path) -> None:
    """A corrupt preferences file must not stop the Settings page from opening."""
    (tmp_path / "jarvis").mkdir(parents=True, exist_ok=True)
    (tmp_path / "jarvis" / "preferences.json").write_text("{not json", encoding="utf-8")
    assert settings_hub.read_internet(tmp_path) is False  # tolerant default
    values = settings_hub.current_values(tmp_path)
    assert values["internet"] == "off"
