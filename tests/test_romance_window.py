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
from gui.romance import RomanceStudioService, _default_setting


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
        # A non-numeric value falls back to the setting's documented default.
        settings._store("max_tokens", "abc")
        assert service.settings()["max_tokens"] == _default_setting("max_tokens")
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


@pytest.mark.skipif(not _display_available(), reason="no display available")
def test_the_library_browser_lists_archetypes_and_previews_a_character(tmp_path: Path) -> None:
    root, dialog, service = _dialog(tmp_path)
    try:
        # The archetype menu is filled from the service's real categories.
        categories = service.library_categories()
        assert dialog._library_category_menu["menu"].index("end") + 1 == len(categories) + 1
        # The list shows the whole ready-made library without creating anything.
        assert dialog._example_list.size() == len(service.library_rows())
        assert service.character_rows() == []
        dialog._example_list.selection_set(0)
        dialog._on_library_selected(0)
        root.update()
        # The preview is filled, and choosing a library companion brings it into the
        # Studio and selects it, so the chat area shows its name.
        assert dialog._library_preview.cget("text").strip()
        assert dialog._selected_character
        assert dialog._char_title.cget("text") != "No companion selected"
    finally:
        _close_window(root, dialog)


@pytest.mark.skipif(not _display_available(), reason="no display available")
def test_choosing_a_library_companion_selects_it_for_the_chat(tmp_path: Path) -> None:
    root, dialog, service = _dialog(tmp_path)
    try:
        # Nothing is selected, and the chat area says so.
        assert dialog._selected_character == ""
        assert dialog._char_title.cget("text") == "No companion selected"
        # Choosing the first library companion (as a click on the list does) makes it
        # the selected companion the chat area names -- the reported behaviour.
        dialog._example_list.selection_set(0)
        dialog._on_library_selected(0)
        root.update()
        rows = service.library_rows()
        chosen = rows[0]
        assert dialog._selected_character
        assert dialog._char_title.cget("text") == chosen.name
        # It was added exactly once, and is now marked present in the library list.
        assert len(service.character_rows()) == 1
        # A second choice of the same companion re-selects it, adding no duplicate.
        dialog._on_library_selected(0)
        root.update()
        assert len(service.character_rows()) == 1
        assert dialog._char_title.cget("text") == chosen.name
    finally:
        _close_window(root, dialog)


@pytest.mark.skipif(not _display_available(), reason="no display available")
def test_the_composer_sends_to_a_companion_chosen_from_the_library(tmp_path: Path) -> None:
    root, dialog, service = _dialog(tmp_path)
    try:
        # Choose a library companion, then talk to it straight away -- the turn must
        # reach that companion (not "select a character first").
        dialog._example_list.selection_set(0)
        dialog._on_library_selected(0)
        root.update()
        character_id = dialog._selected_character
        assert character_id
        dialog._entry.insert("1.0", "Hello, how are you?")
        dialog._send()
        _wait_until(root, lambda: not dialog._busy, timeout=20.0)
        root.update()
        conversations = service.conversations_for(character_id)
        assert conversations, "the turn did not reach the chosen companion"
        messages = service.messages(conversations[0].conversation_id)
        assert any(m.role == "user" and "Hello" in m.content for m in messages)
    finally:
        _close_window(root, dialog)


@pytest.mark.skipif(not _display_available(), reason="no display available")
def test_filtering_by_archetype_narrows_the_library_and_adds_the_group(tmp_path: Path) -> None:
    root, dialog, service = _dialog(tmp_path)
    try:
        dialog._library_category_var.set("family")
        root.update()
        family = service.library_rows(category="family")
        assert dialog._example_list.size() == len(family)
        # "Add the whole group" creates exactly that archetype's characters.
        dialog._add_library_category()
        root.update()
        assert len(service.character_rows()) == len(family)
        assert dialog._char_list.size() == len(family)
    finally:
        _close_window(root, dialog)


@pytest.mark.skipif(not _display_available(), reason="no display available")
def test_the_header_stats_and_avatar_follow_the_real_state(tmp_path: Path) -> None:
    root, dialog, service = _dialog(tmp_path)
    try:
        # With nothing yet, the stat strip reads zero and the avatar is neutral.
        assert dialog._hero_stats["characters"].cget("text") == "0"
        service.create_character("Layla", fictional_age=27)
        dialog._refresh_all()
        dialog._on_character_selected(0)
        root.update()
        assert dialog._hero_stats["characters"].cget("text") == "1"
        assert dialog._char_title.cget("text") == "Layla"
        # The avatar canvas carries the character's initial.
        texts = [
            dialog._avatar.itemcget(i, "text")
            for i in dialog._avatar.find_all()
            if dialog._avatar.type(i) == "text"
        ]
        assert "L" in texts
        # The subtitle summarises the character from its real fields.
        assert "27" in dialog._char_subtitle.cget("text")
    finally:
        _close_window(root, dialog)


@pytest.mark.skipif(not _display_available(), reason="no display available")
def test_the_own_files_panel_reports_and_adds_knowledge_and_facts(tmp_path: Path) -> None:
    root, dialog, service = _dialog(tmp_path)
    try:
        bundle = service.create_character("Huda", fictional_age=38)
        dialog._refresh_all()
        dialog._on_character_selected(0)
        root.update()
        # The panel reads the companion's own folder: the profile is written.
        text = dialog._own_text.get("1.0", "end")
        assert "Profile written: yes" in text
        assert "Facts she was told: 0" in text
        # Adding knowledge and a fact is reflected in the panel (through the
        # service, the same path the window's own dialogs call).
        assert service.add_character_knowledge(bundle.id, "Work", "She is a marine biologist.") >= 1
        assert service.remember_character_fact(bundle.id, "She loves the sea")
        dialog._refresh_own_panel()
        text = dialog._own_text.get("1.0", "end")
        assert "Knowledge bank: 1 chunk(s)" in text
        assert "Facts she was told: 1" in text
        assert "She loves the sea" in text
    finally:
        _close_window(root, dialog)


@pytest.mark.skipif(not _display_available(), reason="no display available")
def test_the_window_offers_the_own_files_controls(tmp_path: Path) -> None:
    root, dialog, _service = _dialog(tmp_path)
    try:
        # The section is present and its four real controls are wired to methods.
        assert dialog._own_text is not None
        for name in ("_add_own_knowledge", "_tell_own_fact", "_show_own_recall", "_open_own_folder"):
            assert callable(getattr(dialog, name))
    finally:
        _close_window(root, dialog)


@pytest.mark.skipif(not _display_available(), reason="no display available")
def test_the_window_offers_the_knowledge_pack_controls(tmp_path: Path) -> None:
    root, dialog, _service = _dialog(tmp_path)
    try:
        # The knowledge-base controls are real methods, and the panel reports the
        # pack's state (not built yet, until one is).
        for name in ("_build_own_knowledge_pack", "_show_own_knowledge_status"):
            assert callable(getattr(dialog, name))
        bundle = _service.create_character("Layla", fictional_age=27)
        dialog._refresh_all()
        dialog._on_character_selected(0)
        root.update()
        assert "Her knowledge base: not built yet" in dialog._own_text.get("1.0", "end")
        # Building through the service is reflected in the panel.
        _service.build_character_knowledge_pack(bundle.id, target_mb=1.0)
        dialog._refresh_own_panel()
        text = dialog._own_text.get("1.0", "end")
        assert "Her knowledge base:" in text
        assert "not built yet" not in text
    finally:
        _close_window(root, dialog)


class _StreamingSeam:
    """A seam that yields the answer in pieces, so the window can show it live."""

    def __init__(self, *pieces: str) -> None:
        self.pieces = list(pieces) or ["Hello ", "there."]

    def __call__(self, prompt: str, key, handle):
        yield from self.pieces


@pytest.mark.skipif(not _display_available(), reason="no display available")
def test_a_streamed_reply_is_drawn_piece_by_piece(tmp_path: Path) -> None:
    import tkinter as tk

    from gui.app import DARK_THEME
    from gui.romance_dialog import RomanceStudioDialog

    service = RomanceStudioService(
        root=tmp_path, generator=_Seam("unused"), stream_generator=_StreamingSeam()
    )
    root = tk.Tk()
    root.withdraw()
    dialog = RomanceStudioDialog(
        root, service, theme=DARK_THEME, font="TkDefaultFont", mono="TkFixedFont"
    )
    try:
        service.create_character("Layla", fictional_age=27)
        dialog._refresh_all()
        dialog._on_character_selected(0)
        root.update()
        dialog._entry.insert("1.0", "Say hello.")
        dialog._send()
        assert _wait_until(root, lambda: not dialog._busy)
        transcript = "\n".join(w.get("1.0", "end") for _row, w in dialog._row_frames)
        # The streamed pieces arrived and were joined into one reply.
        assert "Hello " in transcript
        assert "Hello there." in transcript
    finally:
        _close_window(root, dialog)


@pytest.mark.skipif(not _display_available(), reason="no display available")
def test_a_streamed_reply_never_shows_the_reasoning_trace(tmp_path: Path) -> None:
    import tkinter as tk

    from gui.app import DARK_THEME
    from gui.romance_dialog import RomanceStudioDialog

    opener = chr(60) + "thinking" + chr(62)
    closer = chr(60) + "/thinking" + chr(62)
    seam = _StreamingSeam("Answer. ", opener + "hidden reasoning", closer + " Done.")
    service = RomanceStudioService(root=tmp_path, generator=_Seam("unused"), stream_generator=seam)
    root = tk.Tk()
    root.withdraw()
    dialog = RomanceStudioDialog(
        root, service, theme=DARK_THEME, font="TkDefaultFont", mono="TkFixedFont"
    )
    try:
        service.create_character("Layla", fictional_age=27)
        dialog._refresh_all()
        dialog._on_character_selected(0)
        root.update()
        dialog._entry.insert("1.0", "Say hello.")
        dialog._send()
        assert _wait_until(root, lambda: not dialog._busy)
        transcript = "\n".join(w.get("1.0", "end") for _row, w in dialog._row_frames)
        assert "hidden reasoning" not in transcript
        assert "Answer.  Done." in transcript
    finally:
        _close_window(root, dialog)


@pytest.mark.skipif(not _display_available(), reason="no display available")
def test_a_streamed_reply_appears_one_finished_sentence_at_a_time(tmp_path: Path) -> None:
    """Each finished sentence is drawn as it lands, not only when the reply ends."""
    import time as _time
    import tkinter as tk

    from gui.app import DARK_THEME
    from gui.romance_dialog import RomanceStudioDialog

    class _SlowSeam:
        def __call__(self, prompt: str, key, handle):
            # A fragment first (no sentence yet), then a finished sentence, a pause
            # long enough for the window to paint, then the rest.
            yield "Hello there"
            yield ". How are "
            _time.sleep(0.8)
            yield "you today?"

    service = RomanceStudioService(
        root=tmp_path, generator=_Seam("unused"), stream_generator=_SlowSeam()
    )
    root = tk.Tk()
    root.withdraw()
    dialog = RomanceStudioDialog(
        root, service, theme=DARK_THEME, font="TkDefaultFont", mono="TkFixedFont"
    )
    try:
        service.create_character("Layla", fictional_age=27)
        dialog._refresh_all()
        dialog._on_character_selected(0)
        root.update()
        dialog._entry.insert("1.0", "Say hello.")
        dialog._send()

        def _shown() -> str:
            return "\n".join(w.get("1.0", "end") for _row, w in dialog._row_frames)

        # The first sentence is on screen while the model is still writing.
        assert _wait_until(root, lambda: "Hello there." in _shown(), timeout=6.0)
        assert "you today?" not in _shown()
        # And the whole answer is there once the stream ends.
        assert _wait_until(root, lambda: not dialog._busy)
        assert "Hello there. How are you today?" in _shown()
    finally:
        _close_window(root, dialog)


@pytest.mark.skipif(not _display_available(), reason="no display available")
def test_the_reply_bubble_shows_a_thinking_placeholder_first(tmp_path: Path) -> None:
    import tkinter as tk

    from gui.app import DARK_THEME
    from gui.romance_dialog import _THINKING_PLACEHOLDER, RomanceStudioDialog

    service = RomanceStudioService(
        root=tmp_path, generator=_Seam("unused"), stream_generator=_StreamingSeam("Hi there.")
    )
    root = tk.Tk()
    root.withdraw()
    dialog = RomanceStudioDialog(
        root, service, theme=DARK_THEME, font="TkDefaultFont", mono="TkFixedFont"
    )
    try:
        service.create_character("Layla", fictional_age=27)
        dialog._refresh_all()
        dialog._on_character_selected(0)
        root.update()
        # A reply bubble created for a stream starts with the placeholder, so the
        # wait is visible rather than a blank row under the character's name.
        bubble = dialog._add_message("assistant", _THINKING_PLACEHOLDER, title="Layla")
        assert bubble.get("1.0", "end").strip() == _THINKING_PLACEHOLDER
    finally:
        _close_window(root, dialog)


@pytest.mark.skipif(not _display_available(), reason="no display available")
def test_the_window_offers_sound_and_microphone_toggles(tmp_path: Path) -> None:
    root, dialog, service = _dialog(tmp_path)
    try:
        assert dialog._sound_button is not None
        assert dialog._mic_button is not None
        # Sound starts off (the Studio's own default) and the button reflects it.
        assert service.voice_enabled() is False
        assert dialog._sound_button.cget("text") == "🔇"
        dialog._toggle_sound()
        assert service.voice_enabled() is True
        assert dialog._sound_button.cget("text") == "🔊"
        dialog._toggle_sound()
        assert service.voice_enabled() is False
        assert dialog._sound_button.cget("text") == "🔇"
        # The microphone toggle reports honestly (it never claims a recorder that
        # this machine does not have).
        dialog._toggle_microphone()
        assert "icrophone" in dialog._status_line.cget("text")
    finally:
        _close_window(root, dialog)


@pytest.mark.skipif(not _display_available(), reason="no display available")
def test_saving_a_persona_does_not_hijack_the_selected_companion(tmp_path: Path) -> None:
    """A persona is the *user*: saving one must not change who the chat talks to."""
    root, dialog, service = _dialog(tmp_path)
    try:
        companion = service.create_character("Layla", fictional_age=27)
        dialog._refresh_all()
        dialog._select_character(companion.id)
        root.update()
        assert dialog._selected_character == companion.id

        dialog._create_persona()
        root.update()
        editor = dialog._editors[-1]
        editor._vars["name"].set("Ahmad")
        editor._save()
        root.update()

        # The chat still talks to Layla; the persona did not become the companion.
        assert dialog._selected_character == companion.id
        assert dialog._is_companion(dialog._selected_character)
    finally:
        _close_window(root, dialog)


@pytest.mark.skipif(not _display_available(), reason="no display available")
def test_selecting_a_persona_never_makes_it_the_chat_companion(tmp_path: Path) -> None:
    root, dialog, service = _dialog(tmp_path)
    try:
        companion = service.create_character("Layla", fictional_age=27)
        dialog._refresh_all()
        dialog._select_character(companion.id)
        root.update()
        persona = service.create_persona("Ahmad")
        dialog._refresh_all()
        dialog._select_character(persona.id)
        root.update()
        # The companion is unchanged and the window says why.
        assert dialog._selected_character == companion.id
        assert dialog._selected_persona == persona.id
        assert "persona" in dialog._status_line.cget("text").lower()
    finally:
        _close_window(root, dialog)


@pytest.mark.skipif(not _display_available(), reason="no display available")
def test_the_persona_panel_lists_personas_and_marks_the_active_one(tmp_path: Path) -> None:
    root, dialog, service = _dialog(tmp_path)
    try:
        # With no persona the label says so honestly.
        dialog._refresh_personas()
        root.update()
        assert "No persona yet" in dialog._persona_label.cget("text")
        assert dialog._persona_list.size() == 0
        # Creating one fills the list and the active line.
        service.create_persona("Omar", fictional_age=31)
        dialog._refresh_personas()
        root.update()
        assert dialog._persona_list.size() == 1
        assert "Omar" in dialog._persona_list.get(0)
        assert "Omar" in dialog._persona_label.cget("text")
    finally:
        _close_window(root, dialog)


@pytest.mark.skipif(not _display_available(), reason="no display available")
def test_selecting_and_activating_a_persona_from_the_window(tmp_path: Path) -> None:
    root, dialog, service = _dialog(tmp_path)
    try:
        first = service.create_persona("Omar", fictional_age=31)
        second = service.create_persona("Sara", fictional_age=29)
        dialog._refresh_personas()
        root.update()
        # Rows are newest first, so select the row for the persona we want.
        index = dialog._persona_ids.index(second.id)
        dialog._on_persona_selected(index)
        assert dialog._selected_persona == second.id
        dialog._activate_persona()
        root.update()
        assert service.active_persona_name() == "Sara"
        assert "Sara" in dialog._persona_label.cget("text")
        assert service.characters.get(first.id).character.is_active is False
    finally:
        _close_window(root, dialog)


@pytest.mark.skipif(not _display_available(), reason="no display available")
def test_the_window_offers_the_persona_controls(tmp_path: Path) -> None:
    root, dialog, _service = _dialog(tmp_path)
    try:
        assert dialog._persona_list is not None
        assert dialog._persona_label is not None
        for name in (
            "_create_persona", "_edit_persona", "_activate_persona",
            "_duplicate_persona", "_export_persona", "_import_persona", "_delete_persona",
        ):
            assert callable(getattr(dialog, name))
    finally:
        _close_window(root, dialog)


@pytest.mark.skipif(not _display_available(), reason="no display available")
def test_the_transcript_labels_the_users_own_turns_with_the_persona_name(tmp_path: Path) -> None:
    root, dialog, service = _dialog(tmp_path)
    try:
        service.create_character("Layla", fictional_age=27)
        service.create_persona("Omar", fictional_age=31)
        dialog._refresh_all()
        dialog._on_character_selected(0)
        root.update()
        # The user's own turns are labelled with the persona's name.
        assert dialog._user_label() == "Omar"
        dialog._entry.insert("1.0", "Hello there.")
        dialog._send()
        assert _wait_until(root, lambda: not dialog._busy)
        # The label is drawn on the user's turn, not in the body text.
        labels = [
            child.cget("text")
            for row, _widget in dialog._row_frames
            for child in row.winfo_children()
            for child in child.winfo_children()
            if child.winfo_class() == "Label"
        ]
        assert "Omar" in labels
    finally:
        _close_window(root, dialog)


@pytest.mark.skipif(not _display_available(), reason="no display available")
def test_the_editor_offers_the_avatar_and_the_role_aware_wording(tmp_path: Path) -> None:
    import tkinter as tk

    from core.romance.model import ROLE_USER
    from gui.app import DARK_THEME
    from gui.romance_editor import CharacterEditorDialog

    service = RomanceStudioService(root=tmp_path, generator=_Seam("Hi."))
    root = tk.Tk()
    root.withdraw()
    try:
        # A companion's editor: no persona switch, the companion wording.
        companion = CharacterEditorDialog(
            root, service, character_id="", role="companion",
            theme=DARK_THEME, font="TkDefaultFont", mono="TkFixedFont",
        )
        root.update()
        assert companion.window.title() == "New companion"
        assert getattr(companion, "_active_var", None) is None
        assert companion._avatar_canvas is not None
        for name in ("_upload_avatar", "_generate_avatar", "_clear_avatar", "_refresh_avatar"):
            assert callable(getattr(companion, name))

        # A persona's editor: the persona wording and the "speak as this" switch.
        persona = CharacterEditorDialog(
            root, service, character_id="", role=ROLE_USER,
            theme=DARK_THEME, font="TkDefaultFont", mono="TkFixedFont",
        )
        root.update()
        assert "persona" in persona.window.title().lower()
        assert persona._active_var is not None
    finally:
        root.destroy()
        gc.collect()


@pytest.mark.skipif(not _display_available(), reason="no display available")
def test_the_editor_saves_a_persona_and_activates_it(tmp_path: Path) -> None:
    import tkinter as tk

    from core.romance.model import ROLE_USER
    from gui.app import DARK_THEME
    from gui.romance_editor import CharacterEditorDialog

    service = RomanceStudioService(root=tmp_path, generator=_Seam("Hi."))
    root = tk.Tk()
    root.withdraw()
    try:
        persona = CharacterEditorDialog(
            root, service, character_id="", role=ROLE_USER,
            theme=DARK_THEME, font="TkDefaultFont", mono="TkFixedFont",
        )
        root.update()
        persona._vars["name"].set("Omar")
        persona._vars["age"].set("31")
        persona._active_var.set(True)
        persona._save()
        root.update()
        assert persona.character_id
        assert service.active_persona_name() == "Omar"
        assert service.character(persona.character_id).character.role == ROLE_USER
    finally:
        root.destroy()
        gc.collect()
