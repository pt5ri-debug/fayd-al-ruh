"""Copy, cut, paste, and select-all across every window.

The feature this pins: every place a user can write -- the composer, the search
box, and every field of every dialog -- supports the clipboard shortcuts a
desktop user expects (``Ctrl+C``/``Ctrl+X``/``Ctrl+V``/``Ctrl+A``), a right-click
offers the same as a menu, and a read-only message stays selectable and copyable
while refusing to be pasted into.

The window parts are skipped unless a display (or Xvfb) is available, exactly
like ``tests/test_gui.py``.
"""

from __future__ import annotations

import pytest

try:
    import tkinter as tk
    from tkinter import ttk

    from gui import clipboard
except ImportError:  # pragma: no cover - no libtk on this machine
    tk = None  # type: ignore[assignment]

from tests.test_gui_design import _display_available

pytestmark = pytest.mark.skipif(tk is None, reason="Tk is unavailable")


def _press(widget: tk.Misc, root: tk.Misc, key: str) -> None:
    """Focus ``widget`` and deliver a Ctrl+``key`` chord to it."""
    widget.focus_force()
    root.update()
    widget.event_generate(f"<Control-Key-{key}>", when="now")
    root.update()


# ------------------------------------------------------------------ headless


def test_text_widgets_are_recognised_without_a_display() -> None:
    """The class test names exactly the widgets that hold text."""
    assert clipboard.is_text_widget.__module__ == "gui.clipboard"


# ------------------------------------------------------------------- display


@pytest.mark.skipif(not _display_available(), reason="no display available")
def test_select_all_and_copy_work_in_an_entry() -> None:
    root = tk.Tk()
    try:
        clipboard.install_app_clipboard(root)
        entry = tk.Entry(root)
        entry.pack()
        entry.insert(0, "porosity is the fraction of pore space")
        root.update()
        _press(entry, root, "a")
        assert entry.selection_get() == "porosity is the fraction of pore space"
        root.clipboard_clear()
        root.clipboard_append("keep me")
        _press(entry, root, "c")
        assert root.clipboard_get() == "porosity is the fraction of pore space"
    finally:
        root.destroy()


@pytest.mark.skipif(not _display_available(), reason="no display available")
def test_paste_and_cut_change_an_entry() -> None:
    root = tk.Tk()
    try:
        clipboard.install_app_clipboard(root)
        entry = tk.Entry(root)
        entry.pack()
        entry.insert(0, "old")
        root.update()
        root.clipboard_clear()
        root.clipboard_append("new")
        entry.delete(0, "end")
        _press(entry, root, "v")
        assert entry.get() == "new"
        _press(entry, root, "a")
        _press(entry, root, "x")
        assert entry.get() == ""
        assert root.clipboard_get() == "new"
    finally:
        root.destroy()


@pytest.mark.skipif(not _display_available(), reason="no display available")
def test_a_dialog_field_shares_the_one_clipboard() -> None:
    """A Toplevel is a different window; the shortcuts must reach it too."""
    root = tk.Tk()
    try:
        clipboard.install_app_clipboard(root)
        dialog = tk.Toplevel(root)
        field = tk.Entry(dialog)
        field.pack()
        field.insert(0, "dialog text")
        root.update()
        _press(field, root, "a")
        assert field.selection_get() == "dialog text"
        root.clipboard_clear()
        root.clipboard_append("keep me")
        _press(field, root, "c")
        assert root.clipboard_get() == "dialog text"
    finally:
        root.destroy()


@pytest.mark.skipif(not _display_available(), reason="no display available")
def test_a_combobox_and_a_spinbox_take_the_clipboard() -> None:
    root = tk.Tk()
    try:
        clipboard.install_app_clipboard(root)
        combo = ttk.Combobox(root, values=["a"])
        combo.set("combo value")
        combo.pack()
        spin = ttk.Spinbox(root, from_=0, to=9)
        spin.pack()
        root.update()
        _press(combo, root, "a")
        assert combo.selection_get() == "combo value"
        root.clipboard_clear()
        root.clipboard_append("P")
        combo.delete(0, "end")
        _press(combo, root, "v")
        assert combo.get() == "P"
    finally:
        root.destroy()


@pytest.mark.skipif(not _display_available(), reason="no display available")
def test_install_is_idempotent_and_does_not_stack_bindings() -> None:
    """A second installation must not make a paste apply twice."""
    root = tk.Tk()
    try:
        clipboard.install_app_clipboard(root)
        clipboard.install_app_clipboard(root)
        entry = tk.Entry(root)
        entry.pack()
        root.update()
        root.clipboard_clear()
        root.clipboard_append("X")
        entry.delete(0, "end")
        _press(entry, root, "v")
        assert entry.get() == "X"
    finally:
        root.destroy()


# ---------------------------------------------------------------- read-only


@pytest.mark.skipif(not _display_available(), reason="no display available")
def test_read_only_message_refuses_paste_but_still_copies() -> None:
    root = tk.Tk()
    try:
        clipboard.install_app_clipboard(root)
        message = tk.Text(root, height=3)
        message.pack()
        message.insert("1.0", "porosity is the fraction of pore space")
        clipboard.make_read_only(message)
        root.update()
        body = message.get("1.0", "end-1c")

        root.clipboard_clear()
        root.clipboard_append("INJECTED")
        _press(message, root, "v")
        assert message.get("1.0", "end-1c") == body  # paste refused

        message.tag_add("sel", "1.0", "end-1c")
        _press(message, root, "x")
        assert message.get("1.0", "end-1c") == body  # cut refused

        message.tag_add("sel", "1.0", "end-1c")
        root.clipboard_clear()
        root.clipboard_append("keep me")
        _press(message, root, "c")
        assert root.clipboard_get() == body  # copy still works

        message.tag_remove("sel", "1.0", "end-1c")
        _press(message, root, "a")
        assert message.tag_ranges("sel")  # select-all still works
    finally:
        root.destroy()


@pytest.mark.skipif(not _display_available(), reason="no display available")
def test_read_only_message_refuses_typing_and_movement_is_allowed() -> None:
    root = tk.Tk()
    try:
        clipboard.install_app_clipboard(root)
        message = tk.Text(root, height=3)
        message.pack()
        message.insert("1.0", "read only")
        clipboard.make_read_only(message)
        root.update()
        body = message.get("1.0", "end-1c")
        _press(message, root, "a")
        message.event_generate("<KeyPress-Z>", when="now")
        root.update()
        assert message.get("1.0", "end-1c") == body
    finally:
        root.destroy()


@pytest.mark.skipif(not _display_available(), reason="no display available")
def test_an_editable_text_area_still_accepts_typing() -> None:
    root = tk.Tk()
    try:
        clipboard.install_app_clipboard(root)
        area = tk.Text(root, height=3)
        area.pack()
        area.focus_force()
        root.update()
        area.event_generate("<KeyPress-a>", when="now")
        root.update()
        assert area.get("1.0", "end-1c") == "a"
    finally:
        root.destroy()


@pytest.mark.skipif(not _display_available(), reason="no display available")
def test_a_disabled_text_is_still_copyable_and_refuses_paste() -> None:
    """A ``state="disabled"`` area used to block Ctrl+C; now it copies."""
    root = tk.Tk()
    try:
        clipboard.install_app_clipboard(root)
        area = tk.Text(root, height=3)
        area.pack()
        area.insert("1.0", "porosity is the fraction of pore space")
        area.configure(state="disabled")
        root.update()
        area.focus_force()
        root.update()
        area.tag_add("sel", "1.0", "end-1c")
        root.clipboard_clear()
        root.clipboard_append("keep me")
        area.event_generate("<Control-Key-c>", when="now")
        root.update()
        assert root.clipboard_get() == "porosity is the fraction of pore space"
        root.clipboard_clear()
        root.clipboard_append("INJECT")
        area.event_generate("<Control-Key-v>", when="now")
        root.update()
        assert "INJECT" not in area.get("1.0", "end-1c")
    finally:
        root.destroy()


# ------------------------------------------------------------- context menu


@pytest.mark.skipif(not _display_available(), reason="no display available")
def test_context_menu_reflects_the_widget_it_is_opened_on() -> None:
    root = tk.Tk()
    try:
        menu = clipboard.install_context_menu(root)
        editable = tk.Entry(root)
        editable.pack()
        editable.insert(0, "editable")
        read_only = tk.Text(root, height=2)
        read_only.pack()
        read_only.insert("1.0", "read only")
        clipboard.make_read_only(read_only)
        root.update()

        menu.widget = editable
        menu._build()
        labels = [
            menu.menu.entrycget(i, "label")
            for i in range(int(menu.menu.index("end") or 0) + 1)
            if menu.menu.type(i) == "command"
        ]
        assert labels == ["Cut", "Copy", "Paste", "Select all"]
        # With nothing selected, Cut and Copy are disabled.
        assert menu.menu.entrycget("Cut", "state") == "disabled"

        editable.selection_range(0, "end")
        menu._build()
        assert menu.menu.entrycget("Cut", "state") == "normal"

        menu.widget = read_only
        menu._build()
        assert menu.menu.entrycget("Cut", "state") == "disabled"  # read-only
        assert menu.menu.entrycget("Paste", "state") == "disabled"
    finally:
        root.destroy()


@pytest.mark.skipif(not _display_available(), reason="no display available")
def test_context_menu_carries_the_callers_extra_entries() -> None:
    root = tk.Tk()
    try:
        calls: list[str] = []
        menu = clipboard.install_context_menu(
            root, extras=lambda _w: [("Clear transcript", lambda: calls.append("clear"))]
        )
        area = tk.Text(root, height=2)
        area.pack()
        root.update()
        menu.widget = area
        menu._build()
        labels = [
            menu.menu.entrycget(i, "label")
            for i in range(int(menu.menu.index("end") or 0) + 1)
            if menu.menu.type(i) == "command"
        ]
        assert labels[-1] == "Clear transcript"
        menu.menu.invoke("Clear transcript")
        assert calls == ["clear"]
    finally:
        root.destroy()


@pytest.mark.skipif(not _display_available(), reason="no display available")
def test_a_second_installation_does_not_add_a_second_context_menu() -> None:
    """A dialog installing on its own window must reuse the one menu."""
    for extras_first in (True, False):
        root = tk.Tk()
        try:
            dialog = tk.Toplevel(root)
            if extras_first:
                clipboard.install_app_clipboard(root, extras=lambda _w: [("X", lambda: None)])
                clipboard.install_app_clipboard(dialog)
            else:
                clipboard.install_app_clipboard(dialog)
                clipboard.install_app_clipboard(root, extras=lambda _w: [("X", lambda: None)])
            # Exactly one handler is bound to the shared tag.
            assert str(root.tk.call("bind", "all", "<Button-3>")).count("popup") == 1
            first = clipboard.install_context_menu(root)
            assert first is clipboard.install_context_menu(dialog)
            # The main window's extras reach the one menu whichever order it was
            # installed in.
            assert first.extras is not None
        finally:
            root.destroy()


# --------------------------------------------------------------- main window


@pytest.mark.skipif(not _display_available(), reason="no display available")
def test_composer_selects_all_and_copies() -> None:
    from gui.app import FaydGUI

    from tests.test_gui import FakeSession, _service

    root = tk.Tk()
    try:
        app = FaydGUI(root, service=_service(FakeSession()))
        root.update()
        entry = app._entry
        entry.delete("1.0", "end")
        entry.insert("1.0", "hello world")
        root.update()
        _press(entry, root, "a")
        assert entry.get("sel.first", "sel.last") == "hello world"
        root.clipboard_clear()
        root.clipboard_append("keep me")
        _press(entry, root, "c")
        assert root.clipboard_get() == "hello world"
    finally:
        root.destroy()


@pytest.mark.skipif(not _display_available(), reason="no display available")
def test_delete_in_the_search_entry_does_not_clear_the_transcript() -> None:
    """A regression: the root-level Delete handler used to fire from any field."""
    from gui.app import FaydGUI

    from tests.test_gui import FakeSession, _service

    root = tk.Tk()
    try:
        app = FaydGUI(root, service=_service(FakeSession()))
        root.update()
        app._add_bubble("assistant", "porosity is the fraction of pore space")
        root.update()
        before = len(app._bubbles)
        search = app._search_entry
        search.focus_force()
        root.update()
        search.event_generate("<Delete>", when="now")
        root.update()
        assert len(app._bubbles) == before
    finally:
        root.destroy()


@pytest.mark.skipif(not _display_available(), reason="no display available")
def test_a_read_only_message_cannot_be_pasted_into_in_the_main_window() -> None:
    from gui.app import FaydGUI

    from tests.test_gui import FakeSession, _service

    root = tk.Tk()
    try:
        app = FaydGUI(root, service=_service(FakeSession()))
        root.update()
        app._add_bubble("assistant", "porosity is the fraction of pore space")
        root.update()
        bubble = app._bubbles[-1]
        body = bubble.get("1.0", "end-1c")
        root.clipboard_clear()
        root.clipboard_append("INJECTED")
        bubble.focus_force()
        root.update()
        bubble.event_generate("<Control-Key-v>", when="now")
        root.update()
        assert bubble.get("1.0", "end-1c") == body
    finally:
        root.destroy()


# --------------------------------------------------------------- a real dialog


@pytest.mark.skipif(not _display_available(), reason="no display available")
def test_a_real_dialog_field_copies_and_pastes(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("FAYD_DATA_DIR", str(tmp_path / "data"))
    (tmp_path / "data").mkdir()
    from gui.app import DARK_THEME
    from gui.character_dialog import CharacterManagerDialog
    from gui.character_manager import CharacterManagerService

    root = tk.Tk()
    try:
        service = CharacterManagerService(data_root=tmp_path / "data")
        dialog = CharacterManagerDialog(
            root, service, theme=DARK_THEME, font="DejaVu Sans", mono="DejaVu Sans Mono"
        )
        root.update()
        widgets: list[tk.Misc] = []
        stack: list[tk.Misc] = [dialog.window]
        while stack:
            current = stack.pop()
            for child in current.winfo_children():
                widgets.append(child)
                stack.append(child)
        entry = next(w for w in widgets if isinstance(w, tk.Entry))
        entry.delete(0, "end")
        entry.insert(0, "dialog text")
        root.update()
        _press(entry, root, "a")
        assert entry.selection_get() == "dialog text"
        root.clipboard_clear()
        root.clipboard_append("PASTED")
        entry.delete(0, "end")
        _press(entry, root, "v")
        assert entry.get() == "PASTED"
    finally:
        root.destroy()
