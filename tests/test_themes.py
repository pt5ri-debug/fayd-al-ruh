"""The project's appearance themes, and the sidebar section that chooses them.

Two halves. The **headless** half proves the catalogue itself: every shipped
theme is one complete, valid palette with readable body text, the names and
summaries agree, and the choice round-trips through the store. The **window**
half proves the feature the user asked for: a *Themes* entry of its own in the
sidebar lists every theme, and choosing one restyles the main window *and* every
dialog already open -- through the shared :class:`gui.dialog_base.ThemedDialog`
-- without reopening it.

The window parts are skipped unless a display (or Xvfb) is available, exactly
like ``tests/test_gui.py``.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, cast

import pytest
from gui.themes import (
    DARK_THEME,
    HIGH_CONTRAST_THEME,
    LIGHT_THEME,
    MIDNIGHT_THEME,
    NORD_THEME,
    OCEAN_THEME,
    SEPIA_THEME,
    SOLARIZED_THEME,
    THEMES,
    Theme,
    contrast_ratio,
    default_theme,
    find_theme,
    load_theme_name,
    save_theme_name,
    theme_names,
    theme_summaries,
    validate_theme,
)

ALL_THEMES = (
    DARK_THEME,
    LIGHT_THEME,
    MIDNIGHT_THEME,
    NORD_THEME,
    SOLARIZED_THEME,
    SEPIA_THEME,
    OCEAN_THEME,
    HIGH_CONTRAST_THEME,
)


def _display_available() -> bool:
    try:
        import tkinter as tk

        root = tk.Tk()
    except Exception:
        return False
    root.destroy()
    return True


@pytest.fixture
def data_root(tmp_path, monkeypatch):
    """Point the whole program's data folder at a fresh temp directory."""
    root = tmp_path / "data"
    root.mkdir()
    monkeypatch.setenv("FAYD_DATA_DIR", str(root))
    return root


# ------------------------------------------------------------- the catalogue


def test_the_catalogue_ships_every_documented_theme() -> None:
    """The eight themes the module documents are the eight it ships."""
    assert theme_names() == (
        "dark",
        "light",
        "midnight",
        "nord",
        "solarized",
        "sepia",
        "ocean",
        "high-contrast",
    )
    assert set(THEMES) == set(theme_names())
    # The two OpenHands appearances are the first two, so the menu order is stable.
    assert theme_names()[:2] == ("dark", "light")


def test_every_theme_is_a_complete_palette() -> None:
    """A theme with a missing token would leave a widget painted with a stale colour."""
    for theme in ALL_THEMES:
        assert validate_theme(theme) == (), theme.name
        assert isinstance(theme, Theme)


def test_validate_theme_reports_a_missing_or_malformed_token() -> None:
    """A palette with a bad colour is reported, not shipped (Tk would reject it)."""
    import dataclasses

    broken = dataclasses.replace(DARK_THEME, base="not-a-colour")
    problems = validate_theme(broken)
    assert any("base" in problem for problem in problems)
    empty = dataclasses.replace(DARK_THEME, foreground="")
    assert any("foreground" in problem for problem in validate_theme(empty))


def test_every_theme_keeps_its_body_text_readable() -> None:
    """Each palette is legible: body text against the base clears WCAG AA (4.5:1)."""
    for theme in ALL_THEMES:
        ratio = contrast_ratio(theme.foreground, theme.base)
        assert ratio is not None, theme.name
        assert ratio >= 4.5, f"{theme.name} body contrast is only {ratio:.2f}:1"


def test_primary_text_is_readable_on_the_primary_fill() -> None:
    """A button's label is readable on the accent it sits on, in every theme."""
    for theme in ALL_THEMES:
        ratio = contrast_ratio(theme.primary_foreground, theme.primary)
        assert ratio is not None, theme.name
        assert ratio >= 4.5, f"{theme.name} primary contrast is only {ratio:.2f}:1"


def test_each_theme_has_a_summary_and_a_unique_name() -> None:
    """The sidebar shows a description per theme, and no two names collide."""
    names = [name for name, _summary in theme_summaries()]
    assert names == list(theme_names())
    assert len(set(names)) == len(names)
    for name, summary in theme_summaries():
        assert summary.strip(), name
        assert len(summary) <= 120, name


def test_find_theme_returns_none_for_an_unknown_key() -> None:
    """An unknown key resolves to nothing rather than raising."""
    assert find_theme("does-not-exist") is None
    assert find_theme("nord") is NORD_THEME
    assert default_theme() is DARK_THEME


def test_the_choice_round_trips_through_the_store(data_root) -> None:
    """A saved theme is the one the next launch opens with; a bad value is ignored."""
    assert load_theme_name() == "dark"
    save_theme_name("ocean")
    assert load_theme_name() == "ocean"
    # A key that is not in the catalogue falls back to the default, never raises.
    save_theme_name("not-a-theme")
    assert load_theme_name() == "ocean"


def test_a_broken_store_reads_as_the_default(monkeypatch) -> None:
    """A store that cannot be read must not stop the window opening."""
    import storage.json_store as json_store

    class _Boom:
        def __init__(self, *_a: Any, **_k: Any) -> None:
            pass

        def load(self, *_a: Any, **_k: Any) -> Any:
            raise RuntimeError("no store")

    monkeypatch.setattr(json_store, "JsonStore", _Boom)
    assert load_theme_name() == "dark"


# -------------------------------------------------------- the live re-theme


class _RecordingDialog:
    """A stub dialog: enough of the surface the main window's dispatch touches."""

    def __init__(self) -> None:
        self.applied: list[str] = []
        self.window = SimpleNamespace()

    def retheme(self, theme: Any) -> None:
        self.applied.append(theme.name)


@pytest.mark.skipif(not _display_available(), reason="no display available")
def test_the_main_window_rethemes_every_open_dialog(tmp_path, monkeypatch) -> None:
    """One switch restyles the window and every dialog it has open."""
    import tkinter as tk

    from gui.app import FaydGUI
    from gui.service import ChatService

    monkeypatch.setenv("FAYD_DATA_DIR", str(tmp_path))
    root = tk.Tk()
    try:
        app = FaydGUI(root, service=ChatService(data_root=tmp_path))
        root.update()
        # Two live dialogs: one single-window, one in a per-mouth mapping.
        single = _RecordingDialog()
        app._docs_studio_dialog = cast(Any, single)
        chat = _RecordingDialog()
        app._model_chat_dialogs = cast(Any, {"ultron": chat})

        app._apply_theme("nord")
        root.update()

        assert app._theme is NORD_THEME
        assert single.applied == ["nord"]
        assert chat.applied == ["nord"]
        assert load_theme_name() == "nord"
    finally:
        root.destroy()


@pytest.mark.skipif(not _display_available(), reason="no display available")
def test_an_unknown_or_current_theme_is_a_no_op(tmp_path, monkeypatch) -> None:
    """Choosing the theme in force, or an unknown one, changes nothing."""
    import tkinter as tk

    from gui.app import FaydGUI
    from gui.service import ChatService

    monkeypatch.setenv("FAYD_DATA_DIR", str(tmp_path))
    root = tk.Tk()
    try:
        app = FaydGUI(root, service=ChatService(data_root=tmp_path))
        root.update()
        dialog = _RecordingDialog()
        app._docs_studio_dialog = cast(Any, dialog)
        app._apply_theme("dark")  # already in force
        app._apply_theme("no-such-theme")
        root.update()
        assert app._theme is DARK_THEME
        assert dialog.applied == []
    finally:
        root.destroy()


@pytest.mark.skipif(not _display_available(), reason="no display available")
def test_the_header_control_still_toggles_dark_and_light(tmp_path, monkeypatch) -> None:
    """The ☾/☀ control keeps its dark ⇄ light behaviour beside the full menu."""
    import tkinter as tk

    from gui.app import FaydGUI
    from gui.service import ChatService

    monkeypatch.setenv("FAYD_DATA_DIR", str(tmp_path))
    root = tk.Tk()
    try:
        app = FaydGUI(root, service=ChatService(data_root=tmp_path))
        root.update()
        app._toggle_theme()
        root.update()
        assert app._theme is LIGHT_THEME
        app._toggle_theme()
        root.update()
        assert app._theme is DARK_THEME
    finally:
        root.destroy()


# ------------------------------------------------------------ sidebar entry


@pytest.mark.skipif(not _display_available(), reason="no display available")
def test_the_sidebar_has_a_themes_section_of_its_own(tmp_path, monkeypatch) -> None:
    """Themes are a separate collapsible entry, not buried in Settings."""
    import tkinter as tk

    from gui.app import FaydGUI
    from gui.service import ChatService

    monkeypatch.setenv("FAYD_DATA_DIR", str(tmp_path))
    root = tk.Tk()
    try:
        app = FaydGUI(root, service=ChatService(data_root=tmp_path))
        root.update()
        assert "themes" in app._sections
        # Settings no longer repeats the appearance switch.
        assert "themes" not in app._sections["settings"].title.lower()
    finally:
        root.destroy()


@pytest.mark.skipif(not _display_available(), reason="no display available")
def test_opening_the_themes_section_lists_every_theme(tmp_path, monkeypatch) -> None:
    """The section opens to a row per theme, with the current one marked."""
    import tkinter as tk

    from gui.app import FaydGUI
    from gui.service import ChatService

    monkeypatch.setenv("FAYD_DATA_DIR", str(tmp_path))
    root = tk.Tk()
    try:
        app = FaydGUI(root, service=ChatService(data_root=tmp_path))
        root.update()
        app._toggle_themes()
        root.update()
        panel = app._themes_panel
        assert panel.winfo_children(), "the section must not open empty"
        labels = [
            str(widget.cget("text"))
            for row in panel.winfo_children()
            for widget in row.winfo_children()
            if widget.winfo_class() == "Label"
        ]
        for name in theme_names():
            assert any(name in label for label in labels), name
        # The theme in force is the one marked with the filled dot.
        marked = [label for label in labels if label.startswith("●")]
        assert len(marked) == 1
        assert "dark" in marked[0]
    finally:
        root.destroy()


@pytest.mark.skipif(not _display_available(), reason="no display available")
def test_clicking_a_theme_row_applies_it(tmp_path, monkeypatch) -> None:
    """A row's click switches the window to that theme."""
    import tkinter as tk

    from gui.app import FaydGUI
    from gui.service import ChatService

    monkeypatch.setenv("FAYD_DATA_DIR", str(tmp_path))
    root = tk.Tk()
    try:
        app = FaydGUI(root, service=ChatService(data_root=tmp_path))
        root.update()
        app._toggle_themes()
        root.update()
        # Find the row labelled "sepia" and click its label.
        target = None
        for row in app._themes_panel.winfo_children():
            for widget in row.winfo_children():
                if widget.winfo_class() == "Label" and "sepia" in str(widget.cget("text")):
                    target = widget
        assert target is not None
        target.event_generate("<Button-1>")
        root.update()
        assert app._theme is SEPIA_THEME
        assert load_theme_name() == "sepia"
    finally:
        root.destroy()


@pytest.mark.skipif(not _display_available(), reason="no display available")
def test_a_real_dialog_rethemes_in_place(tmp_path, monkeypatch) -> None:
    """A real ``ThemedDialog`` rebuilds in the new palette, keeping its messages."""
    import tkinter as tk

    from gui.app import DARK_THEME as APP_DARK
    from gui.image_gen_dialog import ImageGenDialog

    monkeypatch.setenv("FAYD_DATA_DIR", str(tmp_path))

    class _Service:
        backend = "procedural"

        def backend_status(self):
            return (("procedural", True, "built in"),)

        def for_backend(self, *_a: Any, **_k: Any) -> Any:
            return SimpleNamespace(available=lambda: True)

        def available(self) -> bool:
            return True

        def generate(self, *_a: Any, **_k: Any) -> Any:
            raise AssertionError("not called")

        def download_model(self, *_a: Any, **_k: Any) -> Any:
            raise AssertionError("not called")

    root = tk.Tk()
    try:
        dialog = ImageGenDialog(
            root, _Service(), theme=APP_DARK, font="DejaVu Sans", mono="DejaVu Sans Mono"
        )
        root.update()
        assert dialog.theme.name == "dark"

        dialog.retheme(HIGH_CONTRAST_THEME)
        root.update()

        assert dialog.theme.name == "high-contrast"
        assert dialog.window.cget("bg") == HIGH_CONTRAST_THEME.base
        # The window is still usable after the rebuild.
        assert dialog._composer is not None
    finally:
        root.destroy()


@pytest.mark.skipif(not _display_available(), reason="no display available")
def test_themed_dialog_preserves_messages_across_a_retheme(tmp_path, monkeypatch) -> None:
    """The conversation a user was reading survives the rebuild."""
    import tkinter as tk

    from gui.dialog_base import ThemedDialog
    from gui.typography import resolve

    monkeypatch.setenv("FAYD_DATA_DIR", str(tmp_path))

    class _Dialog(ThemedDialog):
        def __init__(self, parent: Any, theme: Any) -> None:
            self.theme = theme
            self._type = resolve(parent, ui_family="DejaVu Sans", mono_family="DejaVu Sans Mono")
            self._rows: list[Any] = []
            self.window = tk.Toplevel(parent)
            self.window.configure(bg=theme.base)
            self._themed_build()

        def _build(self) -> None:
            self._body = tk.Frame(self.window, bg=self.theme.base)
            self._body.pack(fill="both", expand=True)

        def _after_build(self) -> None:
            self._add_message("note", "opening note")

        def _add_message(self, role: str, text: str) -> Any:
            label = tk.Label(self._body, text=text, bg=self.theme.base, fg=self.theme.foreground)
            label.pack()
            self._rows.append(label)
            return label

    root = tk.Tk()
    try:
        dialog = _Dialog(root, DARK_THEME)
        root.update()
        dialog._add_message("note", "a user message")
        assert len(dialog._rows) == 2  # the opening note plus the user's own

        dialog.retheme(NORD_THEME)
        root.update()

        assert dialog.theme is NORD_THEME
        assert dialog.window.cget("bg") == NORD_THEME.base
        # The opening note is redrawn by the build; the user's own message is replayed.
        texts = [str(widget.cget("text")) for widget in dialog._rows]
        assert "a user message" in texts
    finally:
        root.destroy()
