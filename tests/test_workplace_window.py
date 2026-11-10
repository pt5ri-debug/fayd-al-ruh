"""Window-level tests for the Workplace (skipped with no display).

These drive the *real* Tk window against the *real*
:class:`gui.workplace.WorkplaceService`, so the page is exercised the way a user
drives it: the dashboard reads real figures, the institution tree and the ten
divisions are drawn, a division is added and deactivated from the sidebar, and the
taxonomy is listed -- all with no model and no network.
"""

from __future__ import annotations

import gc
import time
from pathlib import Path

import pytest
from gui.workplace import WorkplaceService


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


def _all_text_widgets(parent) -> list:
    import tkinter as tk

    found = []
    for child in parent.winfo_children():
        if isinstance(child, tk.Text):
            found.append(child)
        found.extend(_all_text_widgets(child))
    return found


def _close_window(root, dialog) -> None:
    _wait_until(root, lambda: not dialog._busy, timeout=5.0)
    root.destroy()
    gc.collect()


def _dialog(tmp_path: Path):
    import tkinter as tk

    from gui.app import DARK_THEME
    from gui.workplace_dialog import WorkplaceDialog

    service = WorkplaceService(root=tmp_path)
    root = tk.Tk()
    root.withdraw()
    dialog = WorkplaceDialog(
        root, service, theme=DARK_THEME, font="TkDefaultFont", mono="TkFixedFont"
    )
    root.update()
    return root, dialog, service


@pytest.mark.skipif(not _display_available(), reason="no display available")
def test_the_window_builds_with_the_chat_window_style(tmp_path: Path) -> None:
    root, dialog, _service = _dialog(tmp_path)
    try:
        assert dialog.window.title() == "Workplace"
        assert dialog._sidebar is not None
        assert dialog._spinner is not None
    finally:
        _close_window(root, dialog)


@pytest.mark.skipif(not _display_available(), reason="no display available")
def test_the_dashboard_reads_real_figures(tmp_path: Path) -> None:
    root, dialog, _service = _dialog(tmp_path)
    try:
        assert dialog._dash_labels["divisions"].cget("text") == "10"
        assert dialog._dash_labels["institutions"].cget("text") == "1"
        assert "ready" in dialog._status_line.cget("text").lower()
    finally:
        _close_window(root, dialog)


@pytest.mark.skipif(not _display_available(), reason="no display available")
def test_the_tree_and_divisions_are_drawn(tmp_path: Path) -> None:
    root, dialog, _service = _dialog(tmp_path)
    try:
        tree = dialog._tree.get("1.0", "end")
        assert "Egyptian Petroleum Research Institute" in tree
        assert "Technical Support & Technology Center" in tree
        assert dialog._division_list.size() == 10
        assert "Engineering Services" in dialog._division_list.get(0)
    finally:
        _close_window(root, dialog)


@pytest.mark.skipif(not _display_available(), reason="no display available")
def test_the_taxonomy_is_listed(tmp_path: Path) -> None:
    root, dialog, _service = _dialog(tmp_path)
    try:
        text = dialog._taxonomy_text.get("1.0", "end")
        assert "Welding Engineering" in text
        assert "WPS / PQR / WPQ" in text
    finally:
        _close_window(root, dialog)


@pytest.mark.skipif(not _display_available(), reason="no display available")
def test_deactivating_a_division_updates_the_list(tmp_path: Path) -> None:
    root, dialog, _service = _dialog(tmp_path)
    try:
        dialog._division_list.selection_clear(0, "end")
        dialog._division_list.selection_set(0)
        dialog._on_division_selected()
        assert dialog._toggle_active_button.cget("text") == "Deactivate"
        dialog._on_toggle_active()
        assert dialog._division_list.size() == 9
        dialog._show_inactive.set(True)
        dialog._refresh_divisions()
        assert dialog._division_list.size() == 10
    finally:
        _close_window(root, dialog)


@pytest.mark.skipif(not _display_available(), reason="no display available")
def test_the_honesty_note_is_shown(tmp_path: Path) -> None:
    root, dialog, _service = _dialog(tmp_path)
    try:
        text = "\n".join(w.get("1.0", "end") for w in _all_text_widgets(dialog._transcript))
        assert "proposed, configurable defaults" in text
    finally:
        _close_window(root, dialog)
