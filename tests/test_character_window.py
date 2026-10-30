"""The Characters window: pick a character, edit it, APPLY -- and hot reload.

The feature this pins: the sidebar's **Characters** entry opens one window that
lists the seven characters and the project's own model, shows the chosen one's
complete editable configuration (general, personality, language model, voice,
learning, and its own learning management), and an **APPLY** button that validates,
saves, activates what it can now, and **rolls back** on failure.

The window parts are skipped unless a display (or Xvfb) is available, exactly like
``tests/test_gui.py``. The engine the window drives is exercised headless in
``tests/test_character_system.py``; here the *view* is what is checked.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from tests.test_gui_design import _display_available, _service


@pytest.fixture
def data_root(tmp_path, monkeypatch):
    root = tmp_path / "data"
    root.mkdir()
    monkeypatch.setenv("FAYD_DATA_DIR", str(root))
    return root


def _manager(data_root: Path):
    from gui.character_manager import CharacterManagerService

    return CharacterManagerService(data_root=data_root)


def _dialog(root, service):
    from gui.app import DARK_THEME
    from gui.character_dialog import CharacterManagerDialog

    return CharacterManagerDialog(
        root, service, theme=DARK_THEME, font="DejaVu Sans", mono="DejaVu Sans Mono"
    )


@pytest.mark.skipif(not _display_available(), reason="no display available")
def test_window_lists_every_character_and_opens_a_settings_page(data_root) -> None:
    import tkinter as tk

    root = tk.Tk()
    try:
        service = _manager(data_root)
        dialog = _dialog(root, service)
        root.update()
        # All seven characters plus the project's own model are offered.
        assert set(dialog._character_buttons) == set(service.keys())
        # The first one is selected and its page is drawn.
        assert dialog._current == service.keys()[0]
        assert dialog._title.cget("text") == service.config(dialog._current).general.name
        # Every section is present on the page.
        assert set(dialog._vars)  # the editor built real variables
        for key in ("general.name", "personality.traits", "model.key", "voice.voice",
                    "learning.enabled"):
            assert key in dialog._vars
    finally:
        dialog._close()
        root.destroy()


@pytest.mark.skipif(not _display_available(), reason="no display available")
def test_choosing_a_character_shows_that_characters_page(data_root) -> None:
    import tkinter as tk

    root = tk.Tk()
    try:
        service = _manager(data_root)
        dialog = _dialog(root, service)
        root.update()
        dialog._select("vision")
        root.update()
        assert dialog._current == "vision"
        assert dialog._title.cget("text") == service.config("vision").general.name
        assert dialog._vars["general.name"].get() == service.config("vision").general.name
    finally:
        dialog._close()
        root.destroy()


@pytest.mark.skipif(not _display_available(), reason="no display available")
def test_apply_saves_and_activates_a_valid_change(data_root) -> None:
    import tkinter as tk

    root = tk.Tk()
    try:
        service = _manager(data_root)
        dialog = _dialog(root, service)
        root.update()
        dialog._vars["general.name"].set("Jarvis Prime")
        dialog._on_apply()
        root.update()
        assert service.config("jarvis").general.name == "Jarvis Prime"
        # The status line names what was activated and what needs a restart.
        status = dialog._status.cget("text")
        assert "Activated now" in status
        assert "restart" in status
    finally:
        dialog._close()
        root.destroy()


@pytest.mark.skipif(not _display_available(), reason="no display available")
def test_an_invalid_change_does_not_corrupt_the_configuration(data_root) -> None:
    import tkinter as tk

    root = tk.Tk()
    try:
        service = _manager(data_root)
        dialog = _dialog(root, service)
        root.update()
        before = service.config("jarvis")
        dialog._vars["personality.formality"].set("not-a-number")
        dialog._on_apply()
        root.update()
        # The status names the problem and nothing was saved.
        assert "Not applied" in dialog._status.cget("text")
        assert service.config("jarvis").to_primitive() == before.to_primitive()
    finally:
        dialog._close()
        root.destroy()


@pytest.mark.skipif(not _display_available(), reason="no display available")
def test_natasha_shows_the_private_note_not_a_share_checkbox(data_root) -> None:
    import tkinter as tk

    root = tk.Tk()
    try:
        service = _manager(data_root)
        dialog = _dialog(root, service)
        dialog._select("natasha")
        root.update()
        # Natasha has no sharing checkbox -- the backend refuses it, so the page
        # says so rather than offering a control that could not work.
        assert "learning.share_with_main_model" not in dialog._vars
        assert "natasha" in service.share_reason("natasha").lower()
    finally:
        dialog._close()
        root.destroy()


@pytest.mark.skipif(not _display_available(), reason="no display available")
def test_the_learning_management_section_shows_and_searches_records(data_root) -> None:
    import tkinter as tk

    from core.character_learning_store import CharacterLearningStore
    from core.learning_pipeline import LearningPipeline

    root = tk.Tk()
    try:
        store = CharacterLearningStore("banner", root=data_root)
        LearningPipeline("banner", store=store).learn_from_user(
            question="q", statement="Banner learned a durable fact."
        )
        service = _manager(data_root)
        dialog = _dialog(root, service)
        dialog._select("banner")
        root.update()
        # The statistics line names the record count.
        assert "1" in service.stats("banner").summary()
        # A search narrows the list without raising.
        dialog._search_var.set("durable")
        dialog._on_search()
        root.update()
        assert len(service.search("banner", "durable")) == 1
    finally:
        dialog._close()
        root.destroy()


@pytest.mark.skipif(not _display_available(), reason="no display available")
def test_the_main_window_characters_action_opens_the_window(tmp_path) -> None:
    """The sidebar's Characters entry routes to the management window."""
    import tkinter as tk

    from gui.actions import action_by_key
    from gui.app import FaydGUI

    root = tk.Tk()
    try:
        app = FaydGUI(root, service=_service(tmp_path))
        root.update()
        opened: list[str] = []
        app._open_characters = lambda: opened.append("characters")  # type: ignore[assignment]
        app._on_action(action_by_key("characters"))
        assert opened == ["characters"]
    finally:
        root.destroy()


@pytest.mark.skipif(not _display_available(), reason="no display available")
def test_the_window_opens_once_and_is_reused(tmp_path, monkeypatch) -> None:
    """Opening the Characters window twice raises the same one."""
    import tkinter as tk

    from gui.app import FaydGUI

    monkeypatch.setenv("FAYD_DATA_DIR", str(tmp_path / "data"))
    (tmp_path / "data").mkdir(exist_ok=True)
    root = tk.Tk()
    try:
        app = FaydGUI(root, service=_service(tmp_path))
        root.update()
        app._open_characters()
        first = app._characters_dialog
        assert first is not None
        app._open_characters()
        assert app._characters_dialog is first
    finally:
        dialog = getattr(app, "_characters_dialog", None)
        if dialog is not None:
            dialog._close()
        root.destroy()
