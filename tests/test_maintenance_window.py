"""The Maintenance window: backup/restore, diagnostics, and shortcuts.

The window parts are skipped unless a display (or Xvfb) is available, exactly
like ``tests/test_agents_window.py``. The engine the window drives is exercised
headless in ``tests/test_maintenance.py``; here the *view* is what is checked, so
the window is opened with a stand-in service and no real store is touched.
"""

from __future__ import annotations

import time

import pytest

from tests.test_gui_design import _display_available


class StubService:
    """A stand-in service: fixed stores, diagnostics, and shortcuts."""

    def __init__(self) -> None:
        self.backed_up: list[str] = []

    def store_statuses(self):
        from gui.maintenance import STORE_MEMORY, STORE_TITLES, StoreStatus

        return (
            StoreStatus(STORE_MEMORY, STORE_TITLES[STORE_MEMORY], True, "1 database"),
            StoreStatus("knowledge", STORE_TITLES["knowledge"], False, "no knowledge base"),
        )

    def shortcuts(self):
        from gui.maintenance import Shortcut

        return (Shortcut("Ctrl+N", "New chat"), Shortcut("Ctrl+S", "Save the conversation"))

    def backup(self, store: str):
        from gui.maintenance import BackupResult

        self.backed_up.append(store)
        return BackupResult(store, True, "/tmp/backup", 3, "copied 3 file(s)")

    def restore(self, store: str, source: str):
        from gui.maintenance import BackupResult

        return BackupResult(store, True, source, 3, "restored 3 file(s)")

    def diagnostics(self):
        from gui.maintenance import DiagnosticLine

        return (DiagnosticLine("system", "Linux (x86_64)"), DiagnosticLine("cpu cores", "4"))

    def recent_log(self, limit: int = 50):
        return ("loaded qwen3:4b",)

    def memory_summary(self, owner: str) -> str:
        return f"{owner}: 1 fact, 0 exchanges"

    def memory_lines(self, owner: str, *, limit: int = 20):
        from gui.maintenance import MemoryLine

        return (MemoryLine(owner, "fact", "waterflood supports pressure"),)

    def knowledge_search(self, query: str, *, limit: int = 5):
        from gui.maintenance import KnowledgeHit

        return (KnowledgeHit("wikipedia", f"porosity is the fraction of void space ({query})"),)


def _dialog(root, service):
    from gui.app import DARK_THEME
    from gui.maintenance_dialog import MaintenanceDialog

    return MaintenanceDialog(
        root, service, theme=DARK_THEME, font="DejaVu Sans", mono="DejaVu Sans Mono"  # type: ignore[arg-type]
    )


def _drain(root, dialog, seconds: float = 10.0) -> None:
    end = time.time() + seconds
    while time.time() < end:
        root.update()
        if not dialog._busy:
            return
        time.sleep(0.02)


@pytest.mark.skipif(not _display_available(), reason="no display available")
def test_window_builds_with_every_sidebar_section() -> None:
    import tkinter as tk

    root = tk.Tk()
    try:
        dialog = _dialog(root, StubService())
        for key in ("stores", "diagnostics", "shortcuts", "memory", "about"):
            assert key in dialog._sidebar_column.sections
    finally:
        root.destroy()


@pytest.mark.skipif(not _display_available(), reason="no display available")
def test_the_stores_and_shortcuts_are_shown() -> None:
    import tkinter as tk

    root = tk.Tk()
    try:
        dialog = _dialog(root, StubService())
        assert "present" in dialog._stores_label.cget("text")
        assert "Ctrl+N" in dialog._shortcuts_label.cget("text")
    finally:
        root.destroy()


@pytest.mark.skipif(not _display_available(), reason="no display available")
def test_backup_runs_and_reports() -> None:
    import tkinter as tk

    root = tk.Tk()
    try:
        service = StubService()
        dialog = _dialog(root, service)
        dialog._store_var.set("memory")
        dialog._on_backup()
        _drain(root, dialog)
        assert service.backed_up == ["memory"]
        texts = [card.get("1.0", "end-1c") for card in dialog._cards]
        assert any("copied 3 file(s)" in text for text in texts)
    finally:
        root.destroy()


@pytest.mark.skipif(not _display_available(), reason="no display available")
def test_diagnostics_run_and_report() -> None:
    import tkinter as tk

    root = tk.Tk()
    try:
        dialog = _dialog(root, StubService())
        dialog._on_diagnostics()
        _drain(root, dialog)
        texts = [card.get("1.0", "end-1c") for card in dialog._cards]
        assert any("Linux (x86_64)" in text for text in texts)
    finally:
        root.destroy()


@pytest.mark.skipif(not _display_available(), reason="no display available")
def test_the_operation_log_is_shown() -> None:
    import tkinter as tk

    root = tk.Tk()
    try:
        dialog = _dialog(root, StubService())
        dialog._on_log()
        _drain(root, dialog)
        texts = [card.get("1.0", "end-1c") for card in dialog._cards]
        assert any("loaded qwen3:4b" in text for text in texts)
    finally:
        root.destroy()


@pytest.mark.skipif(not _display_available(), reason="no display available")
def test_closing_releases_the_pump_and_clears_the_callback() -> None:
    import tkinter as tk

    root = tk.Tk()
    closed: list[bool] = []
    try:
        from gui.app import DARK_THEME
        from gui.maintenance_dialog import MaintenanceDialog

        dialog = MaintenanceDialog(
            root, StubService(), theme=DARK_THEME, font="DejaVu Sans",  # type: ignore[arg-type]
            mono="DejaVu Sans Mono", on_close=lambda: closed.append(True),
        )
        dialog._close()
        assert closed == [True]
        assert dialog._closed is True
    finally:
        root.destroy()


@pytest.mark.skipif(not _display_available(), reason="no display available")
def test_memory_and_knowledge_run_and_report() -> None:
    import tkinter as tk

    root = tk.Tk()
    try:
        dialog = _dialog(root, StubService())
        dialog._on_memory()
        _drain(root, dialog)
        texts = [card.get("1.0", "end-1c") for card in dialog._cards]
        assert any("waterflood supports pressure" in text for text in texts)

        dialog._kb_query.insert(0, "porosity")
        dialog._on_kb_search()
        _drain(root, dialog)
        texts = [card.get("1.0", "end-1c") for card in dialog._cards]
        assert any("porosity is the fraction of void space" in text for text in texts)
    finally:
        root.destroy()


def test_the_maintenance_action_is_grouped_and_visible() -> None:
    """The action exists, is visible, and lives in the System group."""
    from gui.actions import action_group_by_key, visible_actions

    keys = {action.key for action in visible_actions()}
    assert "maintenance" in keys
    group = action_group_by_key("system")
    assert "maintenance" in group.keys


def test_the_maintenance_action_opens_its_own_window() -> None:
    """The main window routes the action to its own dialog, not the headless run."""
    source = __import__("pathlib").Path("gui/app.py").read_text(encoding="utf-8")
    assert 'action.key == "maintenance"' in source
    assert "self._open_maintenance()" in source
    assert "_maintenance_dialog" in source
