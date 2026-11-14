"""Window-level tests for the Romance & Companion Studio (skipped with no display).

These drive the *real* Tk window against the *real*
:class:`gui.romance.RomanceStudioService`, so the page is exercised the way a user
drives it: a character is created, selected, and chatted with (through an injected
generation seam, so no model is needed), and the profile, relationship, memories,
and voice panels are drawn -- all with no model and no network.
"""

from __future__ import annotations

import gc
import time
from pathlib import Path

import pytest
from gui.romance import RomanceStudioService


def _display_available() -> bool:
    try:
        import tkinter as tk

        root = tk.Tk()
    except Exception:
        return False
    root.destroy()
    return True


def _wait_until(root, predicate, *, timeout: float = 20.0) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        if predicate():
            return True
        root.update()
        time.sleep(0.01)
    return predicate()


class _Seam:
    def __init__(self, *replies: str) -> None:
        self.replies = list(replies) or ["Hello there."]

    def __call__(self, prompt: str, key, handle) -> tuple[str, bool, str]:
        return self.replies[0], True, ""


def _close_window(root, dialog) -> None:
    _wait_until(root, lambda: not dialog._busy, timeout=5.0)
    root.destroy()
    gc.collect()


def _dialog(tmp_path: Path):
    import tkinter as tk

    from gui.app import DARK_THEME
    from gui.romance_dialog import RomanceStudioDialog

    service = RomanceStudioService(root=tmp_path, generator=_Seam("Nice to meet you."))
    root = tk.Tk()
    root.withdraw()
    dialog = RomanceStudioDialog(
        root, service, theme=DARK_THEME, font="TkDefaultFont", mono="TkFixedFont"
    )
    root.update()
    return root, dialog, service


@pytest.mark.skipif(not _display_available(), reason="no display available")
def test_the_window_builds_with_the_studio_style(tmp_path: Path) -> None:
    root, dialog, _service = _dialog(tmp_path)
    try:
        assert dialog.window.title() == "Romance & Companion Studio"
        assert dialog._sidebar is not None
        assert dialog._right_panel is not None
        assert dialog._spinner is not None
        assert dialog._send_button is not None
    finally:
        _close_window(root, dialog)


@pytest.mark.skipif(not _display_available(), reason="no display available")
def test_the_disclosure_is_shown_outside_the_fiction(tmp_path: Path) -> None:
    root, dialog, _service = _dialog(tmp_path)
    try:
        transcript = "\n".join(w.get("1.0", "end") for _row, w in dialog._row_frames)
        assert "fictional AI companions" in transcript
        assert "off by default" in transcript
    finally:
        _close_window(root, dialog)


@pytest.mark.skipif(not _display_available(), reason="no display available")
def test_a_character_is_created_and_drawn(tmp_path: Path) -> None:
    root, dialog, service = _dialog(tmp_path)
    try:
        service.create_character("Layla", fictional_age=27, interests=("coffee",))
        dialog._refresh_all()
        root.update()
        assert dialog._char_list.size() == 1
        assert "Layla" in dialog._char_list.get(0)
        # Selecting it fills the right-hand profile panel.
        dialog._on_character_selected(0)
        root.update()
        assert dialog._char_title.cget("text") == "Layla"
        assert "coffee" in dialog._traits_text.get("1.0", "end")
    finally:
        _close_window(root, dialog)


@pytest.mark.skipif(not _display_available(), reason="no display available")
def test_a_turn_is_sent_and_the_reply_is_drawn(tmp_path: Path) -> None:
    root, dialog, service = _dialog(tmp_path)
    try:
        service.create_character("Layla", fictional_age=27)
        dialog._refresh_all()
        dialog._on_character_selected(0)
        root.update()
        dialog._entry.insert("1.0", "Hello, I am glad to meet you.")
        dialog._send()
        assert _wait_until(root, lambda: not dialog._busy)
        transcript = "\n".join(w.get("1.0", "end") for _row, w in dialog._row_frames)
        assert "glad to meet you" in transcript
        assert "Nice to meet you." in transcript
        # The relationship indicator reflects the turn.
        assert dialog._stage_label.cget("text")
    finally:
        _close_window(root, dialog)


@pytest.mark.skipif(not _display_available(), reason="no display available")
def test_the_right_panel_collapses(tmp_path: Path) -> None:
    root, dialog, _service = _dialog(tmp_path)
    try:
        assert dialog._right_open is True
        dialog._toggle_right()
        root.update()
        assert dialog._right_open is False
        assert not dialog._right_panel.winfo_ismapped()
        dialog._toggle_right()
        root.update()
        assert dialog._right_open is True
    finally:
        _close_window(root, dialog)


@pytest.mark.skipif(not _display_available(), reason="no display available")
def test_the_safety_panel_reports_local_and_adult_state(tmp_path: Path) -> None:
    root, dialog, _service = _dialog(tmp_path)
    try:
        text = dialog._safety_label.cget("text")
        assert "Adult mode: off" in text
        assert "local only" in text
    finally:
        _close_window(root, dialog)


@pytest.mark.skipif(not _display_available(), reason="no display available")
def test_the_discover_section_adds_one_example(tmp_path: Path) -> None:
    root, dialog, service = _dialog(tmp_path)
    try:
        # The Discover list shows the shipped examples without creating any.
        assert dialog._example_list.size() >= 1
        assert list(service.character_rows()) == []
        dialog._example_list.selection_set(0)
        dialog._add_example()
        root.update()
        assert len(service.character_rows()) == 1
        assert dialog._char_list.size() == 1
        # The added one is now marked present in the Discover list.
        assert dialog._example_list.get(0).startswith("✓")
    finally:
        _close_window(root, dialog)


@pytest.mark.skipif(not _display_available(), reason="no display available")
def test_the_model_picker_lists_the_project_models(tmp_path: Path) -> None:
    root, dialog, service = _dialog(tmp_path)
    try:
        # The picker offers every model the project knows, each labelled with
        # its real availability -- it is not a decorative dropdown.
        choices = service.model_choices()
        assert len(dialog._model_keys) == len(choices)
        assert dialog._model_menu["menu"].index("end") + 1 == len(choices)
        assert "ready" in dialog._model_note.cget("text")
    finally:
        _close_window(root, dialog)


@pytest.mark.skipif(not _display_available(), reason="no display available")
def test_assigning_a_model_persists_on_the_character(tmp_path: Path) -> None:
    root, dialog, service = _dialog(tmp_path)
    try:
        bundle = service.create_character("Layla", fictional_age=27)
        dialog._refresh_all()
        dialog._on_character_selected(0)
        root.update()
        key = dialog._model_keys[0]
        dialog._model_var.set(dialog._model_labels[key])
        dialog._assign_model()
        root.update()
        assert service.character(bundle.id).character.model_key == key
        assert "Model:" in dialog._profile_text.get("1.0", "end")
    finally:
        _close_window(root, dialog)


@pytest.mark.skipif(not _display_available(), reason="no display available")
def test_the_settings_window_writes_a_real_setting(tmp_path: Path) -> None:
    root, dialog, service = _dialog(tmp_path)
    try:
        dialog._open_settings()
        settings = dialog._settings_dialog
        assert settings is not None
        settings._store("temperature", "1.5")
        assert service.settings()["temperature"] == "1.5"
        # A hand-typed value beyond the documented range is clamped.
        settings._store("temperature", "99")
        assert service.settings()["temperature"] == "2"
        settings._store("max_tokens", "abc")
        assert service.settings()["max_tokens"] == "512"
    finally:
        _close_window(root, dialog)


@pytest.mark.skipif(not _display_available(), reason="no display available")
def test_the_editor_creates_and_edits_a_character(tmp_path: Path) -> None:
    root, dialog, service = _dialog(tmp_path)
    try:
        dialog._open_editor("")
        editor = dialog._editors[-1]
        editor.window.update()
        editor._vars["name"].set("Mona")
        editor._vars["age"].set("29")
        editor._vars["nationality"].set("Egyptian")
        editor._combo_set("default_mode", "romantic")
        editor._dial_scales["warmth"].set(0.9)
        editor._save()
        root.update()
        rows = service.character_rows()
        assert [r.name for r in rows] == ["Mona"]
        bundle = service.character(rows[0].character_id)
        assert bundle.character.nationality == "Egyptian"
        assert bundle.character.default_mode == "romantic"
        assert round(service.personality(rows[0].character_id)["warmth"], 2) == 0.90
        # The Studio reflects the new character at once.
        assert dialog._char_list.size() == 1
        assert dialog._selected_character == rows[0].character_id
    finally:
        _close_window(root, dialog)


@pytest.mark.skipif(not _display_available(), reason="no display available")
def test_the_editor_refuses_a_character_with_no_name(tmp_path: Path) -> None:
    root, dialog, service = _dialog(tmp_path)
    try:
        dialog._open_editor("")
        editor = dialog._editors[-1]
        editor.window.update()
        editor._vars["name"].set("   ")
        editor._save()
        root.update()
        assert list(service.character_rows()) == []
        assert "name" in editor._status_label.cget("text").lower()
    finally:
        _close_window(root, dialog)


@pytest.mark.skipif(not _display_available(), reason="no display available")
def test_the_editor_adds_and_removes_a_boundary(tmp_path: Path) -> None:
    root, dialog, service = _dialog(tmp_path)
    try:
        bundle = service.create_character("Layla", fictional_age=27)
        dialog._open_editor(bundle.id)
        editor = dialog._editors[-1]
        editor.window.update()
        editor._boundary_entry.insert(0, "no politics")
        editor._add_boundary()
        root.update()
        assert [b.value for b in service.boundaries(bundle.id)] == ["no politics"]
        assert editor._boundary_list.size() == 1
        editor._boundary_list.selection_set(0)
        editor._remove_boundary()
        root.update()
        assert service.boundaries(bundle.id) == ()
    finally:
        _close_window(root, dialog)


@pytest.mark.skipif(not _display_available(), reason="no display available")
def test_the_relationship_can_be_edited_and_reset(tmp_path: Path) -> None:
    root, dialog, service = _dialog(tmp_path)
    try:
        bundle = service.create_character("Layla", fictional_age=27)
        dialog._refresh_all()
        dialog._on_character_selected(0)
        root.update()
        service.set_relationship(bundle.id, trust=0.9)
        dialog._refresh_character_panels()
        assert "trust 0.90" in dialog._relationship_text.get("1.0", "end")
        service.reset_relationship(bundle.id)
        dialog._refresh_character_panels()
        assert "trust 0.90" not in dialog._relationship_text.get("1.0", "end")
    finally:
        _close_window(root, dialog)


@pytest.mark.skipif(not _display_available(), reason="no display available")
def test_a_memory_is_added_listed_and_forgotten(tmp_path: Path) -> None:
    root, dialog, service = _dialog(tmp_path)
    try:
        bundle = service.create_character("Layla", fictional_age=27)
        dialog._refresh_all()
        dialog._on_character_selected(0)
        root.update()
        service.add_memory(bundle.id, "semantic_personal", "Layla loves jasmine tea")
        dialog._refresh_character_panels()
        assert dialog._memory_list.size() == 1
        assert "jasmine tea" in dialog._memory_list.get(0)
        dialog._memory_list.selection_set(0)
        dialog._forget_memory()
        root.update()
        assert dialog._memory_list.size() == 0
    finally:
        _close_window(root, dialog)


@pytest.mark.skipif(not _display_available(), reason="no display available")
def test_a_scenario_is_created_and_a_participant_added(tmp_path: Path) -> None:
    root, dialog, service = _dialog(tmp_path)
    try:
        bundle = service.create_character("Layla", fictional_age=27)
        dialog._refresh_all()
        dialog._on_character_selected(0)
        root.update()
        scenario = service.create_scenario("Coffee shop", category="coffee_shop", location="Cairo")
        service.add_participant(scenario.record_id, bundle.id)
        dialog._refresh_scenarios()
        assert dialog._scenario_list.size() == 1
        dialog._scenario_list.selection_set(0)
        dialog._on_scenario_selected(0)
        panel = dialog._scenario_text.get("1.0", "end")
        assert "Coffee shop" in panel
        assert "Layla" in panel
    finally:
        _close_window(root, dialog)


@pytest.mark.skipif(not _display_available(), reason="no display available")
def test_changing_the_mode_updates_its_note(tmp_path: Path) -> None:
    root, dialog, service = _dialog(tmp_path)
    try:
        service.create_character("Layla", fictional_age=27)
        dialog._refresh_all()
        dialog._on_character_selected(0)
        root.update()
        dialog._new_conversation()
        dialog._mode_var.set("flirting_playful")
        root.update()
        assert "playful" in dialog._mode_note.cget("text").lower()
        assert service.modes()  # the menu was filled from the real modes
        assert dialog._mode_menu["menu"].index("end") + 1 == len(service.modes())
    finally:
        _close_window(root, dialog)
