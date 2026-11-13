"""The Training Center window: the sidebar, the sections, and the wiring.

The window parts are skipped unless a display (or Xvfb) is available, exactly like
``tests/test_model_manager_window.py``. The engine the window drives is exercised
headless in ``tests/test_training_center.py``; here the *view* is what is checked,
so the window is opened with a stand-in service and no model, network, or training
run ever happens.
"""

from __future__ import annotations

import time

import pytest

try:
    import tkinter as tk
except ImportError:  # pragma: no cover - no libtk on this machine
    tk = None  # type: ignore[assignment]

from tests.test_gui_design import _display_available

pytestmark = pytest.mark.skipif(tk is None, reason="Tk is unavailable")


class StubService:
    """A stand-in service: fixed reports, no store, no model, no network."""

    def __init__(self) -> None:
        from core.training_center.dashboard import ChartSeries, DashboardSnapshot
        from core.training_center.qa_learning import ReviewSummary
        from core.training_center.records import LearningExample

        self.settings_set: list[tuple[str, object]] = []
        self.asked: list[tuple[str, str]] = []
        self._web = False
        self.dashboard_snapshot = DashboardSnapshot(
            learning_sources=2,
            documents=10,
            chunks=42,
            datasets=1,
            dataset_versions=1,
            training_examples=6,
            learning_examples=3,
            approved_examples=2,
            pending_examples=1,
            jobs=1,
            completed_jobs=1,
            charts=(ChartSeries("Training loss", (("run-1", 0.5),)),),
        )
        self._example = LearningExample(
            question="What is porosity?", answer="Void space.", provider="manual",
        )
        self._summary = ReviewSummary(counts={"pending": 1})

    # -- dashboard ---------------------------------------------------------
    def dashboard(self):
        return self.dashboard_snapshot

    # -- catalog / settings ------------------------------------------------
    def catalog_text(self) -> str:
        return "Training Center — catalog\n  dashboard.refresh  Refresh"

    def capabilities(self):
        from core.training_center.catalog import CAPABILITIES

        return CAPABILITIES

    def settings(self) -> dict:
        return {"chunk_chars": 1200, "review_required": True, "web": self._web}

    def setting(self, key: str):
        return self.settings().get(key)

    def set_setting(self, key: str, value) -> dict:
        self.settings_set.append((key, value))
        if key == "web":
            self._web = bool(value)
        return self.settings()

    def settings_text(self) -> str:
        return "Training Center — settings\n  chunk_chars: 1200"

    # -- the internet switch and the dropdown choices ----------------------
    def web_enabled(self) -> bool:
        return self._web

    def set_web(self, enabled: bool) -> bool:
        self._web = bool(enabled)
        return self._web

    def toggle_web(self) -> bool:
        self._web = not self._web
        return self._web

    def web_status(self) -> str:
        return "Internet: on" if self._web else "Internet: off"

    def provider_choices(self):
        return (("manual", "A hand-written answer"), ("document", "The sources"))

    def provider_values(self):
        return tuple(value for value, _ in self.provider_choices())

    def cloud_choices(self):
        return (("gemini", "Gemini (cloud)"),)

    def cloud_values(self):
        return tuple(value for value, _ in self.cloud_choices())

    def mouth_choices(self):
        return (("ultron", "Ultron"), ("vision", "Vision"))

    def mouth_values(self):
        return tuple(value for value, _ in self.mouth_choices())

    def character_choices(self):
        return (("sources", "Jarvis"), ("hakeem", "Hakeem"))

    def character_values(self):
        return tuple(value for value, _ in self.character_choices())

    def dataset_choices(self):
        return ("learned",)

    def run_choices(self):
        return ()

    def pipeline_choices(self):
        return ("demo",)

    def setting_keys(self):
        return ("chunk_chars", "review_required", "web")

    def setting_value_choices(self, key: str):
        return ("true", "false") if key == "web" else ("900",)

    def topic_choices(self):
        return ("Reservoir Engineering", "Drilling Engineering")

    # -- knowledge ---------------------------------------------------------
    def knowledge_stats(self) -> dict:
        return {"chunks": 42, "articles": 10, "size_bytes": 1024}

    def knowledge_search(self, query: str, *, limit: int = 4):
        return ()

    def reindex(self, *, progress=None) -> dict:
        return {"chunks": 42}

    def knowledge_store_dir(self):
        from pathlib import Path

        return Path("/tmp/fayd-tc-window")

    def export_knowledge(self, destination) -> str:
        return f"Copied 0 file(s) to {destination}."

    # -- sources / imports -------------------------------------------------
    def import_files(self, paths):
        raise AssertionError("import_files should not run in this test")

    def import_folder(self, folder, *, recursive: bool = True):
        raise AssertionError("import_folder should not run in this test")

    def import_urls(self, urls, *, approved: bool = False):
        raise AssertionError("import_urls should not run in this test")

    def provider_statuses(self):
        return ()

    # -- Q&A ---------------------------------------------------------------
    def ask(self, question: str, *, provider: str = "", topic: str = "", system: str = ""):
        self.asked.append((question, provider))
        return self._example

    def generate_questions(self, topic: str, *, count: int = 5, provider: str = ""):
        return ()

    def ask_many(self, questions, *, provider: str = "", topic: str = ""):
        return ()

    def examples(self, *, status: str = "", query: str = ""):
        return (self._example,)

    def review_summary(self):
        return self._summary

    def approve(self, example_id: str, *, note: str = ""):
        return self._example

    def reject(self, example_id: str, *, note: str = ""):
        return self._example

    def correct(self, example_id: str, answer: str, *, note: str = ""):
        return self._example

    # -- datasets ----------------------------------------------------------
    def dataset_rows(self):
        return ()

    def build_dataset_from_approved(self, dataset_id: str):
        raise AssertionError("build_dataset_from_approved should not run in this test")

    def build_dataset_from_texts(self, dataset_id: str, texts):
        raise AssertionError("build_dataset_from_texts should not run in this test")

    def validate_dataset(self, dataset_id: str, version=None) -> str:
        return f"{dataset_id}: ok"

    def export_dataset(self, dataset_id: str, destination, version=None):
        raise AssertionError("export_dataset should not run in this test")

    # -- jobs --------------------------------------------------------------
    def job_rows(self):
        return ()

    def build_config(self, **kwargs):
        return object()

    def feasibility(self, model, dataset_id, config, *, version=None):
        return type("Report", (), {"describe": lambda self: "feasibility: ok"})()

    def start_job(self, *args, **kwargs):
        raise AssertionError("start_job should not run in this test")

    def cancel_job(self, run_id: str) -> bool:
        return False

    def evaluate_run(self, run_id: str):
        raise AssertionError("evaluate_run should not run in this test")

    # -- pipelines ---------------------------------------------------------
    def pipeline_list(self):
        return ()

    def create_pipeline(self, pipeline_id: str):
        raise AssertionError("create_pipeline should not run in this test")

    def run_pipeline(self, pipeline_id: str, *, approved: bool = False):
        raise AssertionError("run_pipeline should not run in this test")

    # -- history -----------------------------------------------------------
    def history_lines(self, *, kind: str = "", query: str = "", limit: int = 100):
        return ()

    # -- resources ---------------------------------------------------------
    def resources(self):
        from core.training_center.resources import read_resources

        return read_resources()

    def estimate(self, **kwargs):
        return type("Estimate", (), {"describe": lambda self: "estimate: unknown"})()

    def model_choices(self):
        return ()


def _dialog(root, service):
    from gui.app import DARK_THEME
    from gui.training_center_dialog import TrainingCenterDialog

    return TrainingCenterDialog(
        root, service, theme=DARK_THEME, font="DejaVu Sans", mono="DejaVu Sans Mono"
    )


def _button_texts(widget: tk.Misc) -> set[str]:
    """Every button label in a widget tree, so a drawn button can be checked."""
    texts: set[str] = set()
    for child in widget.winfo_children():
        if isinstance(child, tk.Button):
            texts.add(str(child.cget("text")))
        texts |= _button_texts(child)
    return texts


@pytest.mark.skipif(not _display_available(), reason="no display available")
def test_window_builds_every_section() -> None:
    from core.training_center.catalog import SECTIONS

    root = tk.Tk()
    try:
        dialog = _dialog(root, StubService())
        keys = set(dialog._sidebar_column.sections)
        assert {"dashboard", "sources", "knowledge", "qa", "datasets"} <= keys
        assert {"jobs", "finetune", "pretrain", "language_model", "evaluation"} <= keys
        assert {"kb", "pipelines", "history", "resources", "settings"} <= keys
        assert len(SECTIONS) == 15
    finally:
        root.destroy()


@pytest.mark.skipif(not _display_available(), reason="no display available")
def test_every_button_comes_from_the_catalogue() -> None:
    from core.training_center.catalog import CAPABILITIES

    root = tk.Tk()
    try:
        dialog = _dialog(root, StubService())
        titles = {cap.title for cap in CAPABILITIES}
        drawn = _button_texts(dialog.window) & titles
        # Every catalogue capability is drawn as a button with the catalogue's own
        # title, so the window and the catalogue can never drift apart.
        assert len(drawn) == len(CAPABILITIES)
    finally:
        root.destroy()


@pytest.mark.skipif(not _display_available(), reason="no display available")
def test_opening_shows_the_dashboard_and_its_charts() -> None:
    root = tk.Tk()
    try:
        dialog = _dialog(root, StubService())
        text = "\n".join(card.get("1.0", "end") for card in dialog._cards)
        assert "dashboard" in text.lower()
        assert "Training loss" in text  # a chart line is drawn
    finally:
        root.destroy()


@pytest.mark.skipif(not _display_available(), reason="no display available")
def test_setting_a_preference_reaches_the_service(monkeypatch) -> None:
    import gui.training_center_dialog as dialog_module

    root = tk.Tk()
    try:
        service = StubService()
        dialog = _dialog(root, service)
        answers = iter(["chunk_chars", "900"])
        monkeypatch.setattr(dialog_module, "ask_choice", lambda *a, **k: next(answers))
        dialog._on_set_setting()
        assert service.settings_set == [("chunk_chars", 900)]
    finally:
        root.destroy()


@pytest.mark.skipif(not _display_available(), reason="no display available")
def test_window_closes_without_leaving_a_pending_callback() -> None:
    root = tk.Tk()
    try:
        dialog = _dialog(root, StubService())
        dialog._close()
        assert dialog._closed
        assert dialog._pump_after is None
    finally:
        root.destroy()


# -- live streaming into the chat area ------------------------------------


class _FakeResult:
    def describe(self) -> str:
        return "run finished: 4 steps"


class _FakeHandle:
    def __init__(self, run_id: str = "run-1") -> None:
        self.run_id = run_id
        self.result = _FakeResult()
        self.error: Exception | None = None

    def wait(self, timeout: float | None = None) -> bool:
        return True


class _StreamingStubService(StubService):
    """A stub that drives the training callback, so the live stream can be seen."""

    def __init__(self, steps: int = 4) -> None:
        super().__init__()
        self._steps = steps
        self.started: list[dict] = []

    def start_job(self, model, dataset_id, config, **kwargs):
        self.started.append({"dataset_id": dataset_id, "config": config, **kwargs})
        on_step = kwargs.get("on_step")
        if on_step is not None:
            for step in range(1, self._steps + 1):
                on_step(step, 1.0 / step)
        return _FakeHandle()

    def build_config(self, **kwargs):
        return type("Config", (), {"max_steps": self._steps, "run_id": ""})()


class _Config:
    def __init__(self, max_steps: int = 4) -> None:
        self.max_steps = max_steps
        self.run_id = ""


@pytest.mark.skipif(not _display_available(), reason="no display available")
def test_step_stream_reports_the_first_last_and_about_ten_between() -> None:
    root = tk.Tk()
    try:
        dialog = _dialog(root, StubService())
        lines: list[str] = []
        dialog._stream_line = lines.append  # type: ignore[method-assign]
        for step in range(1, 21):
            dialog._step_stream("run-1", step, 0.5, 20, started=time.monotonic())
        # The first and last step are always reported, and only about ten in
        # between -- never all twenty.
        assert any("step 1/20" in line for line in lines)
        assert any("step 20/20" in line for line in lines)
        assert 10 <= len(lines) <= 12
    finally:
        root.destroy()


@pytest.mark.skipif(not _display_available(), reason="no display available")
def test_streamed_training_wires_on_step_and_posts_the_result() -> None:
    root = tk.Tk()
    try:
        service = _StreamingStubService(steps=4)
        dialog = _dialog(root, service)
        handle = dialog._start_streamed_job(object(), "learned", _Config(max_steps=4))
        assert handle.run_id == "run-1"
        # The service received a real on_step callback...
        assert callable(service.started[0]["on_step"])
        # ...and the live lines are waiting to be drawn by the pump.
        assert dialog._lines.qsize() >= 2
        assert dialog._describe_handle(handle) == "run finished: 4 steps"
    finally:
        root.destroy()


@pytest.mark.skipif(not _display_available(), reason="no display available")
def test_flush_stream_draws_one_growing_message() -> None:
    root = tk.Tk()
    try:
        dialog = _dialog(root, StubService())
        dialog._stream_line("first line")
        dialog._drain_lines(cap=None)
        dialog._flush_stream(force=True)
        assert dialog._stream_body is not None
        first = dialog._stream_body.get("1.0", "end")
        assert "first line" in first
        dialog._stream_line("second line")
        dialog._drain_lines(cap=None)
        dialog._flush_stream(force=True)
        combined = dialog._stream_body.get("1.0", "end")
        assert "first line" in combined and "second line" in combined
    finally:
        root.destroy()
