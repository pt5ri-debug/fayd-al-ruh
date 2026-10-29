"""The sidebar's grouped, collapsible shape.

The headless parts (the action groups, the grouping rule, the persistence) run
without a display; the widget parts (the collapsible section and the window that
builds them) are skipped unless a display or Xvfb is available, exactly like
``tests/test_gui.py``.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest
from core.jarvis import JarvisMode
from gui.actions import (
    ACTION_GROUPS,
    SIDEBAR_LABELS,
    action_group_by_key,
    grouped_visible_actions,
    visible_actions,
)
from gui.service import ChatService


class FakeSession:
    """A minimal stand-in for ``JarvisMode`` (enough for the window to build)."""

    def __init__(self) -> None:
        self.router: Any = None
        self.assistant: Any = SimpleNamespace(name="grounded")
        self.voice = SimpleNamespace(assistant=self.assistant)
        self.help_text = "help"
        self.memory: Any = None
        self._conversation = SimpleNamespace(id="CNV-fake")

    @property
    def conversation(self) -> SimpleNamespace:
        return self._conversation

    def handle(self, message: str, **_: Any):
        from core.jarvis import JarvisTurn

        return JarvisTurn(heard=message, response="ok", woke=True, routed="answer")


def _service(tmp_path: Path) -> ChatService:
    return ChatService(mode=cast(JarvisMode, FakeSession()), data_root=tmp_path)


def _display_available() -> bool:
    try:
        import tkinter as tk

        root = tk.Tk()
    except Exception:
        return False
    root.destroy()
    return True


# --------------------------------------------------------------- the grouping


def test_every_visible_action_lands_in_exactly_one_group() -> None:
    """No action is dropped, and none is listed twice."""
    grouped = grouped_visible_actions()
    listed = [action.key for _group, members in grouped for action in members]
    assert sorted(listed) == sorted(action.key for action in visible_actions())
    assert len(listed) == len(set(listed))


def test_groups_are_in_the_declared_order_and_only_visible_keys() -> None:
    """Sections keep their order, and a hidden action never reappears in one."""
    visible = {action.key for action in visible_actions()}
    grouped = grouped_visible_actions()
    titles = [group.title for group, _members in grouped]
    declared = [group.title for group in ACTION_GROUPS]
    assert titles == [title for title in declared if title in titles]
    for _group, members in grouped:
        assert members, "an empty group must not be drawn"
        assert all(action.key in visible for action in members)
        assert all(not action.hidden for action in members)


def test_a_new_unnamed_action_is_still_shown(monkeypatch) -> None:
    """An action no group names is appended to the last section, not lost."""
    from gui import actions as actions_module

    extra = actions_module.GuiAction(key="brand-new", label="Brand new", command=lambda _a: 0)
    monkeypatch.setattr(
        actions_module, "visible_actions", lambda: (*actions_module.ACTIONS, extra)
    )
    grouped = grouped_visible_actions()
    listed = [action.key for _group, members in grouped for action in members]
    assert "brand-new" in listed
    assert listed[-1] == "brand-new"


def test_action_group_by_key_reports_an_unknown_key() -> None:
    assert action_group_by_key("knowledge").title == "Knowledge & Web"
    with pytest.raises(KeyError):
        action_group_by_key("does-not-exist")


def test_the_reading_voice_is_not_repeated_in_the_settings_group() -> None:
    """The duplication the redesign removed: no group offers the voice picker."""
    for group in ACTION_GROUPS:
        assert "voices" not in group.keys
        assert "settings" not in group.keys


def test_sidebar_labels_are_short_and_keep_the_full_name() -> None:
    """A drawn label fits the narrow column; the full name is still available."""
    for action in visible_actions():
        short = action.sidebar_label
        assert short.strip()
        # The full label is never lost: it stays on the action (the tooltip).
        assert action.label
        if action.key in SIDEBAR_LABELS:
            assert short == SIDEBAR_LABELS[action.key]
            assert len(short) <= 40


# ------------------------------------------------------------- the persistence


def test_sidebar_sections_round_trip_through_the_store(tmp_path) -> None:
    from gui.app import load_sidebar_sections, save_sidebar_sections

    assert load_sidebar_sections(tmp_path) == {}
    save_sidebar_sections({"chats": True, "settings": False}, tmp_path)
    assert load_sidebar_sections(tmp_path) == {"chats": True, "settings": False}


def test_a_broken_sections_record_reads_as_empty(tmp_path) -> None:
    from gui.app import load_sidebar_sections

    # A value that is not a real bool is dropped, so a corrupt store cannot leave
    # a section in a state the window cannot draw.
    from storage.json_store import JsonStore

    JsonStore(tmp_path).save("gui", "sidebar_sections", {"sections": {"chats": "yes", "x": 1}})
    assert load_sidebar_sections(tmp_path) == {}


# --------------------------------------------------------------- the section


@pytest.mark.skipif(not _display_available(), reason="no display available")
def test_sidebar_section_toggles_and_reports_the_change() -> None:
    import tkinter as tk

    from gui.sidebar import SidebarSection
    from gui.typography import for_root

    root = tk.Tk()
    try:
        from gui.app import DARK_THEME

        seen: list[tuple[str, bool]] = []
        section = SidebarSection(
            root,
            key="demo",
            title="Demo",
            icon="★",
            theme=DARK_THEME,
            typography=for_root(root),
            open=False,
            on_toggle=lambda key, is_open: seen.append((key, is_open)),
        )
        root.update()
        assert section.is_open is False
        assert not section.body.winfo_ismapped()
        section.toggle()
        root.update()
        assert section.is_open is True
        assert section.body.winfo_ismapped()
        assert seen == [("demo", True)]
        section.set_open(False)
        root.update()
        assert not section.body.winfo_ismapped()
        # set_open to the same value is a no-op (no second callback).
        section.set_open(False)
        assert seen == [("demo", True), ("demo", False)]
    finally:
        root.destroy()


@pytest.mark.skipif(not _display_available(), reason="no display available")
def test_window_builds_the_grouped_collapsible_sidebar(tmp_path) -> None:
    import tkinter as tk

    from gui.app import FaydGUI

    root = tk.Tk()
    try:
        app = FaydGUI(root, service=_service(tmp_path))
        root.update()
        # The section keys are present: the chats, the voice, one per group, the settings.
        assert "chats" in app._sections
        assert "voices" in app._sections
        assert "settings" in app._sections
        for group in ACTION_GROUPS:
            assert f"group:{group.key}" in app._sections
        # Every visible action is drawn exactly once.
        assert len(app._action_buttons) == len(visible_actions())
        # The chats are open by default, the action groups are closed.
        assert app._sections["chats"].is_open is True
        assert app._sections[f"group:{ACTION_GROUPS[1].key}"].is_open is False
    finally:
        root.destroy()


@pytest.mark.skipif(not _display_available(), reason="no display available")
def test_window_settings_section_does_not_repeat_the_voice_list(tmp_path) -> None:
    """Opening Settings builds the Cloud pickers but no voice radios (one home)."""
    import tkinter as tk

    from gui.app import FaydGUI

    root = tk.Tk()
    try:
        app = FaydGUI(root, service=_service(tmp_path))
        root.update()
        app._toggle_settings()
        root.update()
        assert app._settings_built is True
        assert app._sections["settings"].is_open is True

        def walk(widget: Any, out: list[Any]) -> None:
            for child in widget.winfo_children():
                out.append(child)
                walk(child, out)

        widgets: list[Any] = []
        walk(app._settings_panel, widgets)
        assert not any(isinstance(w, tk.Radiobutton) for w in widgets)
        # The voice radios live in the Reading Voice section instead. A fixed
        # source set is used so the assertion does not depend on what this machine
        # has installed.
        from unittest import mock

        from core import voices as voices_module
        from core.voices import VOICE_CATALOG

        installed = (voices_module.SystemVoice("Microsoft Hoda", "ar-SA", "Female"),)
        sections = voices_module.voice_sections(system=installed, model=(), catalog=True)
        with mock.patch.object(app.service, "voice_sections", lambda: sections):
            app._toggle_voices()
            root.update()
            voice_widgets: list[Any] = []
            walk(app._voices_panel, voice_widgets)
        assert any(isinstance(w, tk.Radiobutton) for w in voice_widgets)
        assert len(VOICE_CATALOG) > 0
    finally:
        root.destroy()


@pytest.mark.skipif(not _display_available(), reason="no display available")
def test_window_settings_toggle_is_idempotent(tmp_path) -> None:
    """A second "show the settings" call does not rebuild the pickers."""
    import tkinter as tk

    from gui.app import FaydGUI

    root = tk.Tk()
    try:
        app = FaydGUI(root, service=_service(tmp_path))
        root.update()
        app._toggle_settings()
        root.update()
        first = list(app._settings_panel.winfo_children())
        app._toggle_settings()
        root.update()
        assert app._sections["settings"].is_open is True
        assert list(app._settings_panel.winfo_children()) == first
    finally:
        root.destroy()


@pytest.mark.skipif(not _display_available(), reason="no display available")
def test_opening_the_reading_voice_section_by_click_fills_it(tmp_path) -> None:
    """Clicking the Reading Voice *header* populates the list, not just opens it.

    The list used to be built only by the programmatic ``_toggle_voices`` entry
    point, so a plain header click (how a user actually reaches it) opened an empty
    panel. The toggle handler now builds it too.
    """
    import tkinter as tk

    from gui.app import FaydGUI

    root = tk.Tk()
    try:
        app = FaydGUI(root, service=_service(tmp_path))
        root.update()
        section = app._sections["voices"]
        if section.is_open:  # a stored state may have left it open
            section.set_open(False)
            root.update()
        assert not app._voices_panel.winfo_children()
        section.toggle()  # the header's own click
        root.update()
        assert section.is_open is True
        assert app._voices_panel.winfo_children()
    finally:
        root.destroy()


@pytest.mark.skipif(not _display_available(), reason="no display available")
def test_opening_the_settings_section_by_click_builds_it(tmp_path) -> None:
    """Clicking the Settings *header* builds its contents (the cloud pickers)."""
    import tkinter as tk

    from gui.app import FaydGUI

    root = tk.Tk()
    try:
        app = FaydGUI(root, service=_service(tmp_path))
        root.update()
        section = app._sections["settings"]
        if section.is_open:
            section.set_open(False)
            root.update()
        assert app._settings_built is False
        section.toggle()
        root.update()
        assert app._settings_built is True
        assert app._settings_panel.winfo_children()
    finally:
        root.destroy()


# --------------------------------------------------------- the shared column


def test_column_section_states_round_trip_and_ignore_a_corrupt_record(tmp_path) -> None:
    from gui.sidebar import load_section_states, save_section_states

    assert load_section_states("demo", tmp_path) == {}
    save_section_states("demo", {"one": True, "two": False}, tmp_path)
    assert load_section_states("demo", tmp_path) == {"one": True, "two": False}
    # A different scope is a separate record.
    assert load_section_states("other", tmp_path) == {}
    # A non-bool value is dropped rather than handed to a widget.
    from storage.json_store import JsonStore

    JsonStore(tmp_path).save(
        "gui", "sidebar_sections:bad", {"sections": {"x": "yes", "y": 1}}
    )
    assert load_section_states("bad", tmp_path) == {}


@pytest.mark.skipif(not _display_available(), reason="no display available")
def test_sidebar_column_builds_collapsible_sections_and_persists(tmp_path) -> None:
    import tkinter as tk

    from gui.app import DARK_THEME
    from gui.sidebar import SidebarColumn, load_section_states
    from gui.typography import for_root

    root = tk.Tk()
    try:
        column = SidebarColumn(
            root, theme=DARK_THEME, typography=for_root(root), width=240,
            scope="demo-column", state_root=tmp_path,
        )
        body = column.add("one", "One", icon="★")
        column.add("two", "Two", icon="☆", open=False)
        root.update()
        assert set(column.sections) == {"one", "two"}
        assert column.sections["one"].is_open is True
        assert column.sections["two"].is_open is False
        # The scrollbar is always visible, not only on overflow.
        assert column.scrollbar.winfo_ismapped()
        # A user's own toggle persists the choice under the scope.
        column.sections["two"].toggle()
        root.update()
        assert load_section_states("demo-column", tmp_path)["two"] is True
        # A programmatic open does not persist.
        column.set_open("two", False)
        assert column.sections["two"].is_open is False
        assert load_section_states("demo-column", tmp_path)["two"] is True
        # The section body is the frame the caller fills.
        assert isinstance(body, tk.Frame)
    finally:
        root.destroy()
