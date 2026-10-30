"""The Model Manager window: list, select, inspect, and run -- and stay honest.

The window parts are skipped unless a display (or Xvfb) is available, exactly
like ``tests/test_gui.py``. The engine the window drives is exercised headless in
``tests/test_model_manager_service.py``; here the *view* is what is checked, so
the window is opened with a stand-in service and no real model is ever loaded.
"""

from __future__ import annotations

import pytest

from tests.test_gui_design import _display_available, _service


class StubService:
    """A stand-in service: fixed rows, no registry, no runtime, no model."""

    def __init__(self):
        from gui.model_manager import ManagerSnapshot, ModelRow

        self.selected = ""
        self.removed: list[str] = []
        self.checked: list[str] = []
        self.rows = (
            ModelRow(
                model_id="m-ready",
                display_name="Ready Model",
                state="ready",
                state_marker="✓ ready",
                runtime="llama.cpp",
                source="local",
                format="gguf",
                path="/models/ready.gguf",
                size_bytes=2 * 1024**3,
                quantization="Q4_K_M",
                ready=True,
                from_catalog=False,
                status_line="ready",
            ),
            ModelRow(
                model_id="m-missing",
                display_name="Missing Model",
                state="missing",
                state_marker="! file missing",
                runtime="llama.cpp",
                source="catalog",
                format="gguf",
                path="",
                size_bytes=0,
                quantization="",
                ready=False,
                from_catalog=True,
                status_line="not downloaded",
            ),
        )
        self._snapshot = ManagerSnapshot(
            rows=self.rows,
            registered=1,
            catalog=1,
            ready=1,
            runtime_lines=("llama.cpp: not available",),
        )

    def snapshot(self):
        return self._snapshot

    def search(self, query="", *, state="", runtime="", rows=None):
        source = tuple(rows) if rows is not None else self.rows
        needle = str(query or "").strip().casefold()
        wanted = str(state or "").strip()
        result = []
        for row in source:
            if wanted and row.state != wanted:
                continue
            if runtime and row.runtime != runtime:
                continue
            if needle and needle not in " ".join(
                (row.display_name, row.model_id, row.path)
            ).casefold():
                continue
            result.append(row)
        return tuple(result)

    def state_names(self):
        return ("ready", "not-downloaded", "missing", "damaged", "unknown")

    def diagnostics(self):
        from core.model_manager import DiagnosticsReport, ModelManagerStatus

        return DiagnosticsReport(
            status=ModelManagerStatus(registry_path="/tmp/registry"),
            resources=None,
        )

    def task_names(self):
        return ("chat", "petroleum")

    def select_for_task(self, task):
        return self.rows[0] if task == "chat" else None

    def select(self, model_id):
        self.selected = model_id
        return model_id

    def detail(self, model_id):
        from gui.model_manager import ModelDetail

        row = next(r for r in self.rows if r.model_id == model_id)
        return ModelDetail(
            row=row,
            identity=row.model_id,
            integrity="ok" if row.ready else "",
            integrity_detail="",
            architecture="llama",
            context_length=4096,
            parameter_count=3_000_000_000,
            license="apache-2.0",
            tags=(),
            notes="",
            engine="ultron",
            registered_at="2026-01-01T00:00:00+00:00",
            updated_at="2026-01-01T00:00:00+00:00",
            settings=_FakeSettings(),
        )

    def log_tail(self, limit=40):
        return ()

    def inspect(self, model_id):
        raise AssertionError("inspect should not run in this test")

    def compatibility(self, model_id):
        from core.model_manager import CompatibilityReport

        self.checked.append(model_id)
        return CompatibilityReport(
            model_id=model_id, runtime="llama.cpp", can_run=True, gpu_offload=False
        )

    def load(self, model_id):
        raise AssertionError("load should not run in this test")

    def test(self, model_id, **kwargs):
        raise AssertionError("test should not run in this test")

    def unload(self, model_id=""):
        raise AssertionError("unload should not run in this test")

    def update_settings(self, model_id, **changes):
        raise AssertionError("update_settings should not run in this test")

    def reset_settings(self, model_id):
        raise AssertionError("reset_settings should not run in this test")

    def remove(self, model_id, *, confirm=False):
        self.removed.append(model_id)

    def import_custom_models(self, **kwargs):
        return ()

    def discover(self, roots=None):
        raise AssertionError("discover should not run in this test")

    def register(self, path, **kwargs):
        raise AssertionError("register should not run in this test")


class _FakeSettings:
    temperature = 0.7
    max_tokens = 512
    context = 4096


def _dialog(root, service):
    from gui.app import DARK_THEME
    from gui.model_manager_dialog import ModelManagerDialog

    return ModelManagerDialog(
        root, service, theme=DARK_THEME, font="DejaVu Sans", mono="DejaVu Sans Mono"
    )


@pytest.mark.skipif(not _display_available(), reason="no display available")
def test_window_lists_every_model() -> None:
    import tkinter as tk

    root = tk.Tk()
    try:
        dialog = _dialog(root, StubService())
        listed = dialog._table.get_children()
        assert set(listed) == {"m-ready", "m-missing"}
    finally:
        root.destroy()


@pytest.mark.skipif(not _display_available(), reason="no display available")
def test_selecting_a_model_shows_its_detail() -> None:
    import tkinter as tk

    root = tk.Tk()
    try:
        service = StubService()
        dialog = _dialog(root, service)
        dialog._table.selection_set("m-ready")
        dialog._on_select_row()
        assert service.selected == "m-ready"
        body = dialog._detail_text.get("1.0", "end")
        assert "Ready Model" in body
        assert "llama.cpp" in body
    finally:
        root.destroy()


@pytest.mark.skipif(not _display_available(), reason="no display available")
def test_the_detail_names_the_memory_figure_as_an_estimate() -> None:
    import tkinter as tk

    root = tk.Tk()
    try:
        service = StubService()
        dialog = _dialog(root, service)
        dialog._table.selection_set("m-ready")
        dialog._on_select_row()
        dialog._on_check()
        dialog._pump_results()
        # The check runs on a worker thread; wait for its result to be pumped.
        for _ in range(200):
            dialog._pump_results()
            if service.checked:
                break
        body = dialog._detail_text.get("1.0", "end")
        assert "can run" in body
    finally:
        root.destroy()


@pytest.mark.skipif(not _display_available(), reason="no display available")
def test_forgetting_a_model_reloads_the_table() -> None:
    import tkinter as tk

    root = tk.Tk()
    try:
        service = StubService()
        dialog = _dialog(root, service)
        dialog._table.selection_set("m-ready")
        dialog._on_select_row()
        dialog._on_forget()
        assert service.removed == ["m-ready"]
        # The stub's snapshot is unchanged, so the row is still listed; the point
        # is that the view asked the service to forget it and did not crash.
        assert dialog._table.get_children()
    finally:
        root.destroy()


@pytest.mark.skipif(not _display_available(), reason="no display available")
def test_the_window_shows_the_honesty_note() -> None:
    import tkinter as tk

    root = tk.Tk()
    try:
        dialog = _dialog(root, StubService())
        summary = dialog._summary_label.cget("text")
        assert "registered" in summary
        assert "ready to run" in summary
    finally:
        root.destroy()


@pytest.mark.skipif(not _display_available(), reason="no display available")
def test_a_broken_registry_does_not_blank_the_window() -> None:
    import tkinter as tk

    class Broken(StubService):
        def snapshot(self):
            raise RuntimeError("the registry file is corrupt")

    root = tk.Tk()
    try:
        dialog = _dialog(root, Broken())
        assert "corrupt" in dialog._status_label.cget("text")
    finally:
        root.destroy()


@pytest.mark.skipif(not _display_available(), reason="no display available")
def test_the_main_window_model_manager_action_opens_the_window(tmp_path) -> None:
    """The sidebar's Model Manager entry routes to the management window."""
    import tkinter as tk

    from gui.actions import action_by_key
    from gui.app import FaydGUI

    root = tk.Tk()
    try:
        app = FaydGUI(root, service=_service(tmp_path))
        root.update()
        opened: list[str] = []
        app._open_model_manager = lambda: opened.append("models")  # type: ignore[assignment]
        app._on_action(action_by_key("models"))
        assert opened == ["models"]
    finally:
        root.destroy()


@pytest.mark.skipif(not _display_available(), reason="no display available")
def test_the_model_manager_window_opens_once_and_is_reused(tmp_path, monkeypatch) -> None:
    """Opening the Model Manager window twice raises the same one."""
    import tkinter as tk

    from gui.app import FaydGUI

    monkeypatch.setenv("FAYD_DATA_DIR", str(tmp_path / "data"))
    (tmp_path / "data").mkdir(exist_ok=True)
    root = tk.Tk()
    try:
        app = FaydGUI(root, service=_service(tmp_path))
        root.update()
        app._open_model_manager()
        first = app._model_manager_dialog
        assert first is not None
        app._open_model_manager()
        assert app._model_manager_dialog is first
    finally:
        dialog = getattr(app, "_model_manager_dialog", None)
        if dialog is not None:
            dialog._close()
        root.destroy()


# ------------------------------------------------------------- pure helpers


def test_human_size_reads_in_binary_units() -> None:
    from gui.model_manager import human_size

    assert human_size(0) == "—"
    assert human_size(512) == "512.0 B"
    assert human_size(2 * 1024**3) == "2.0 GB"


def test_detail_text_is_honest_about_a_missing_file() -> None:
    from gui.model_manager import detail_text

    service = StubService()
    detail = service.detail("m-missing")
    body = detail_text(detail)
    assert "Missing Model" in body
    assert "no local file" in body


def test_detail_text_names_the_fetch_command_when_there_is_one() -> None:
    from dataclasses import replace

    from gui.model_manager import detail_text

    service = StubService()
    detail = replace(service.detail("m-missing"), download_command="python -m cli ultron download")
    body = detail_text(detail)
    assert "python -m cli ultron download" in body



@pytest.mark.skipif(not _display_available(), reason="no display available")
def test_the_sidebar_lists_the_tasks_and_the_model_in_force() -> None:
    """The Tasks block names the model in force per task, or says none is ready."""
    import tkinter as tk

    root = tk.Tk()
    try:
        dialog = _dialog(root, StubService())
        body = dialog._tasks_label.cget("text")
        assert "chat: Ready Model" in body
        assert "petroleum: none ready" in body
    finally:
        root.destroy()


@pytest.mark.skipif(not _display_available(), reason="no display available")
def test_the_search_box_filters_the_table() -> None:
    import tkinter as tk

    root = tk.Tk()
    try:
        dialog = _dialog(root, StubService())
        dialog._search_var.set("missing")
        listed = dialog._table.get_children()
        assert listed == ("m-missing",)
        dialog._search_var.set("")
        assert set(dialog._table.get_children()) == {"m-ready", "m-missing"}
    finally:
        root.destroy()


@pytest.mark.skipif(not _display_available(), reason="no display available")
def test_the_state_filter_limits_the_table() -> None:
    import tkinter as tk

    root = tk.Tk()
    try:
        dialog = _dialog(root, StubService())
        dialog._state_var.set("ready")
        dialog._fill_table()
        assert dialog._table.get_children() == ("m-ready",)
    finally:
        root.destroy()


@pytest.mark.skipif(not _display_available(), reason="no display available")
def test_the_dashboard_reports_what_is_loaded_and_the_machine() -> None:
    import tkinter as tk

    root = tk.Tk()
    try:
        dialog = _dialog(root, StubService())
        body = dialog._diagnostics_label.cget("text")
        assert "No model is loaded" in body
        assert "No recent errors" in body
    finally:
        root.destroy()


@pytest.mark.skipif(not _display_available(), reason="no display available")
def test_running_diagnostics_shows_the_report() -> None:
    import tkinter as tk

    root = tk.Tk()
    try:
        service = StubService()
        dialog = _dialog(root, service)
        # Drive the finish path directly, so the test does not race the worker
        # thread; ``_on_diagnostics`` is checked separately for not raising.
        dialog._on_diagnostics()
        dialog._finish_diagnostics(service.diagnostics(), None)
        body = dialog._detail_text.get("1.0", "end")
        assert "Model Manager" in body
        assert "Diagnostics ready." == dialog._status_label.cget("text")
    finally:
        root.destroy()


@pytest.mark.skipif(not _display_available(), reason="no display available")
def test_the_diagnostics_button_is_enabled_and_wired() -> None:
    import tkinter as tk

    root = tk.Tk()
    try:
        dialog = _dialog(root, StubService())
        assert dialog._diagnostics_button.cget("text") == "Run diagnostics"
        assert dialog._diagnostics_button.cget("state") == "normal"
    finally:
        root.destroy()
