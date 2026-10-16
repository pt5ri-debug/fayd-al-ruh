"""Window-level tests for the Code Studio dialog (skipped with no display).

These drive the real Tk window against a real throwaway git repository and a
scripted coding model, so the page is exercised the way a user drives it: create a
workspace, let the model edit a file, run the checks, merge, and discard.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from core.code_studio import CodeStudio
from gui.code_studio import CodeStudioService


def _display_available() -> bool:
    try:
        import tkinter as tk

        root = tk.Tk()
    except Exception:
        return False
    root.destroy()
    return True


def _git(repo: Path, *args: str) -> None:
    subprocess.run(
        ("git", *args),
        cwd=str(repo),
        capture_output=True,
        text=True,
        check=True,
        env={
            **__import__("os").environ,
            "GIT_AUTHOR_NAME": "Test",
            "GIT_AUTHOR_EMAIL": "test@example.com",
            "GIT_COMMITTER_NAME": "Test",
            "GIT_COMMITTER_EMAIL": "test@example.com",
        },
    )


@pytest.fixture()
def project(tmp_path: Path) -> Path:
    repo = tmp_path / "project"
    repo.mkdir()
    _git(repo, "init", "-q", "-b", "main")
    (repo / "core").mkdir()
    (repo / "core" / "app.py").write_text("x = 1\n", encoding="utf-8")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-qm", "initial")
    return repo


class FakeCoder:
    """A scripted coding model, so no Ollama server is needed."""

    def __init__(self, script: list[str]) -> None:
        self.model = "qwen2.5-coder:7b"
        self.script = list(script)

    def list_models(self) -> tuple[str, ...]:
        return ("qwen2.5-coder:7b",)

    def chat(self, messages, *, temperature: float = 0.2) -> str:
        if not self.script:
            return '{"action": "done", "summary": "nothing left to do"}'
        return self.script.pop(0)


class SlowCoder(FakeCoder):
    """A scripted model that pauses before each turn, so live steps are observable."""

    def __init__(self, script: list[str], *, delay: float = 0.5) -> None:
        super().__init__(script)
        self.delay = delay

    def chat(self, messages, *, temperature: float = 0.2) -> str:
        import time

        time.sleep(self.delay)
        return super().chat(messages, temperature=temperature)


def _service(project: Path, tmp_path: Path, script: list[str] | None = None) -> CodeStudioService:
    return CodeStudioService(
        project,
        studio=CodeStudio(project, workspaces_dir=tmp_path / "ws"),
        client=FakeCoder(script or []),
        check_command=("python", "-c", "pass"),
    )


def _wait_until(root, predicate, *, timeout: float = 10.0) -> bool:
    """Run the real Tk main loop until ``predicate`` holds (or time runs out).

    Tk's ``after`` callbacks only fire inside ``mainloop``, and a worker thread's
    result is drained by one, so the tests drive the loop the way a user's window
    does rather than calling ``update`` in a spin.
    """
    import time

    deadline = time.time() + timeout
    while time.time() < deadline:
        if predicate():
            return True
        root.update()
        time.sleep(0.01)
    return predicate()


def _all_text_widgets(parent) -> list:
    """Every ``tk.Text`` under ``parent``, so a test can search the whole transcript."""
    import tkinter as tk

    found = []
    for child in parent.winfo_children():
        if isinstance(child, tk.Text):
            found.append(child)
        found.extend(_all_text_widgets(child))
    return found


@pytest.mark.skipif(not _display_available(), reason="no display available")
def test_dialog_builds_and_opens_a_workspace_for_you(project: Path, tmp_path: Path) -> None:
    import tkinter as tk

    from gui.app import DARK_THEME
    from gui.code_studio_dialog import TITLE, CodeStudioDialog

    root = tk.Tk()
    try:
        dialog = CodeStudioDialog(
            root, _service(project, tmp_path), theme=DARK_THEME, font="TkDefaultFont", mono="TkFixedFont"
        )
        root.update()
        assert dialog.window.title() == TITLE
        assert "ready" in dialog._status_line.cget("text")
        # Opening the Studio opens the first workspace for the user, named PR1.
        assert _wait_until(root, lambda: dialog._workspace_list.size() == 1)
        assert dialog._workspace_list.get(0) == "PR1"
        assert dialog._current_slug == "PR1"
        assert dialog._merge_button.cget("state") == "disabled"
        assert dialog._apply_button.cget("state") == "disabled"
    finally:
        root.destroy()


@pytest.mark.skipif(not _display_available(), reason="no display available")
def test_dialog_lists_the_projects_files_by_default(project: Path, tmp_path: Path) -> None:
    """The project's own files are present without the user adding anything."""
    import tkinter as tk

    from gui.app import DARK_THEME
    from gui.code_studio_dialog import CodeStudioDialog

    root = tk.Tk()
    try:
        dialog = CodeStudioDialog(
            root, _service(project, tmp_path), theme=DARK_THEME, font="TkDefaultFont", mono="TkFixedFont"
        )
        root.update()
        assert _wait_until(root, lambda: dialog._file_list.size() >= 1)
        names = [dialog._file_list.get(i) for i in range(dialog._file_list.size())]
        assert "core/app.py" in names
    finally:
        root.destroy()


@pytest.mark.skipif(not _display_available(), reason="no display available")
def test_dialog_starts_a_new_change_named_in_order(project: Path, tmp_path: Path) -> None:
    import tkinter as tk

    from gui.app import DARK_THEME
    from gui.code_studio_dialog import CodeStudioDialog

    root = tk.Tk()
    try:
        dialog = CodeStudioDialog(
            root, _service(project, tmp_path), theme=DARK_THEME, font="TkDefaultFont", mono="TkFixedFont"
        )
        root.update()
        assert _wait_until(root, lambda: dialog._workspace_list.size() == 1)
        dialog._on_new_change()
        assert _wait_until(root, lambda: dialog._workspace_list.size() == 2)
        labels = [dialog._workspace_list.get(i) for i in range(dialog._workspace_list.size())]
        # Newest first, so PR2 is at the top and both numbers exist.
        assert "PR2" in labels and "PR1" in labels
        assert dialog._current_slug == "PR2"
    finally:
        root.destroy()


@pytest.mark.skipif(not _display_available(), reason="no display available")
def test_dialog_opens_a_file_and_saves_an_edit(project: Path, tmp_path: Path) -> None:
    import tkinter as tk

    from gui.app import DARK_THEME
    from gui.code_studio_dialog import CodeStudioDialog

    root = tk.Tk()
    try:
        dialog = CodeStudioDialog(
            root, _service(project, tmp_path), theme=DARK_THEME, font="TkDefaultFont", mono="TkFixedFont"
        )
        root.update()
        assert _wait_until(root, lambda: dialog._workspace_list.size() == 1)
        assert _wait_until(root, lambda: dialog._file_list.size() >= 1)

        index = [dialog._file_list.get(i) for i in range(dialog._file_list.size())].index("core/app.py")
        dialog._file_list.selection_set(index)
        dialog._on_file_select()
        root.update()
        assert dialog._editor.get("1.0", "end-1c") == "x = 1\n"

        dialog._editor.delete("1.0", "end")
        dialog._editor.insert("1.0", "x = 3\n")
        dialog._on_editor_modified()
        assert dialog._dirty
        dialog._on_save_file()
        root.update()
        assert "Saved core/app.py" in dialog._merge_note.cget("text")
        assert (dialog.service.workspace("PR1").path / "core" / "app.py").read_text(
            encoding="utf-8"
        ) == "x = 3\n"
    finally:
        root.destroy()


@pytest.mark.skipif(not _display_available(), reason="no display available")
def test_dialog_asks_the_model_and_shows_the_trace(project: Path, tmp_path: Path) -> None:
    import tkinter as tk

    from gui.app import DARK_THEME
    from gui.code_studio_dialog import CodeStudioDialog

    script = [
        '{"action": "write_file", "path": "core/app.py", "content": "x = 6\\n"}',
        '{"action": "done", "summary": "set x to 6"}',
    ]
    root = tk.Tk()
    try:
        dialog = CodeStudioDialog(
            root,
            _service(project, tmp_path, script),
            theme=DARK_THEME,
            font="TkDefaultFont",
            mono="TkFixedFont",
        )
        root.update()
        assert _wait_until(root, lambda: dialog._workspace_list.size() == 1)

        dialog._request_var.set("set x to 6")
        dialog._on_ask_model()
        assert _wait_until(
            root,
            lambda: dialog._trace is not None and "Done:" in dialog._trace.get("1.0", "end"),
        )
        trace_text = dialog._trace.get("1.0", "end")
        assert "write_file" in trace_text
        assert "set x to 6" in trace_text
        assert (dialog.service.workspace("PR1").path / "core" / "app.py").read_text(
            encoding="utf-8"
        ) == "x = 6\n"
    finally:
        root.destroy()


@pytest.mark.skipif(not _display_available(), reason="no display available")
def test_dialog_shows_the_question_and_streams_steps_live(project: Path, tmp_path: Path) -> None:
    """The request appears as a user bubble, and the model's steps appear as it works."""
    import tkinter as tk

    from gui.app import DARK_THEME
    from gui.code_studio_dialog import CodeStudioDialog

    script = [
        '{"action": "read_file", "path": "core/app.py"}',
        '{"action": "write_file", "path": "core/app.py", "content": "x = 5\\n"}',
        '{"action": "done", "summary": "set x to 5"}',
    ]
    root = tk.Tk()
    try:
        service = CodeStudioService(
            project,
            studio=CodeStudio(project, workspaces_dir=tmp_path / "ws"),
            client=SlowCoder(script),
            check_command=("python", "-c", "pass"),
        )
        dialog = CodeStudioDialog(
            root,
            service,
            theme=DARK_THEME,
            font="TkDefaultFont",
            mono="TkFixedFont",
        )
        root.update()
        assert _wait_until(root, lambda: dialog._workspace_list.size() == 1)

        dialog._request_var.set("make x five")
        dialog._on_ask_model()
        # The request is echoed as the user's own message and the composer clears.
        assert dialog._request_var.get() == ""
        assert _wait_until(
            root,
            lambda: any(
                "make x five" in child.get("1.0", "end")
                for child in _all_text_widgets(dialog._transcript)
            ),
        )

        # The live bubble fills in with the model's steps as the run proceeds: the
        # read the model did, then the write, both before the run finishes.
        assert _wait_until(
            root,
            lambda: dialog._live is not None
            and "read_file" in dialog._live.get("1.0", "end")
            and "write_file" in dialog._live.get("1.0", "end"),
        )
        assert _wait_until(
            root,
            lambda: dialog._trace is not None and "Done:" in dialog._trace.get("1.0", "end"),
        )
        # Once the run ends, the final trace supersedes the live bubble.
        assert dialog._live is None
    finally:
        root.destroy()


@pytest.mark.skipif(not _display_available(), reason="no display available")
def test_dialog_runs_the_checks_and_offers_to_apply(project: Path, tmp_path: Path) -> None:
    """The whole flow: describe the change, the model edits, checks pass, then approve."""
    import tkinter as tk

    from gui.app import DARK_THEME
    from gui.code_studio_dialog import CodeStudioDialog

    script = [
        '{"action": "write_file", "path": "core/app.py", "content": "x = 8\\n"}',
        '{"action": "done", "summary": "set x to 8"}',
    ]
    root = tk.Tk()
    try:
        dialog = CodeStudioDialog(
            root,
            _service(project, tmp_path, script),
            theme=DARK_THEME,
            font="TkDefaultFont",
            mono="TkFixedFont",
        )
        root.update()
        assert _wait_until(root, lambda: dialog._workspace_list.size() == 1)

        dialog._request_var.set("set x to 8")
        dialog._on_ask_model()
        # The checks run automatically after the model, and passing enables Apply.
        assert _wait_until(root, lambda: dialog._apply_button.cget("state") == "normal")
        # The change is only in the workspace, not in the project yet.
        assert (project / "core" / "app.py").read_text(encoding="utf-8") == "x = 1\n"

        dialog._on_merge()
        assert _wait_until(root, lambda: "Applied" in dialog._merge_note.cget("text"))
        assert (project / "core" / "app.py").read_text(encoding="utf-8") == "x = 8\n"
    finally:
        root.destroy()


@pytest.mark.skipif(not _display_available(), reason="no display available")
def test_dialog_does_not_offer_to_apply_a_failing_change(project: Path, tmp_path: Path) -> None:
    """A change that fails the checks is never offered for applying."""
    import tkinter as tk

    from gui.app import DARK_THEME
    from gui.code_studio_dialog import CodeStudioDialog

    script = [
        '{"action": "write_file", "path": "core/app.py", "content": "x = 9\\n"}',
        '{"action": "done", "summary": "set x to 9"}',
    ]
    root = tk.Tk()
    try:
        service = CodeStudioService(
            project,
            studio=CodeStudio(project, workspaces_dir=tmp_path / "ws"),
            client=FakeCoder(script),
            # A command that fails, standing in for a failing test suite.
            check_command=("python", "-c", "raise SystemExit(1)"),
        )
        dialog = CodeStudioDialog(
            root, service, theme=DARK_THEME, font="TkDefaultFont", mono="TkFixedFont"
        )
        root.update()
        assert _wait_until(root, lambda: dialog._workspace_list.size() == 1)

        dialog._request_var.set("set x to 9")
        dialog._on_ask_model()
        assert _wait_until(
            root,
            lambda: any(
                "checks failed" in child.get("1.0", "end")
                for child in _all_text_widgets(dialog._transcript)
            ),
        )
        assert dialog._apply_button.cget("state") == "disabled"
        # Nothing reached the project.
        assert (project / "core" / "app.py").read_text(encoding="utf-8") == "x = 1\n"
    finally:
        root.destroy()


@pytest.mark.skipif(not _display_available(), reason="no display available")
def test_dialog_shows_the_diff_and_gates_merge_on_the_checks(project: Path, tmp_path: Path) -> None:
    import tkinter as tk

    from gui.app import DARK_THEME
    from gui.code_studio_dialog import CodeStudioDialog

    root = tk.Tk()
    try:
        dialog = CodeStudioDialog(
            root, _service(project, tmp_path), theme=DARK_THEME, font="TkDefaultFont", mono="TkFixedFont"
        )
        root.update()
        assert _wait_until(root, lambda: dialog._workspace_list.size() == 1)
        dialog.service.save_file("PR1", "core/app.py", "x = 12\n")

        dialog._on_show_diff()
        root.update()
        assert "+x = 12" in dialog._review.get("1.0", "end")
        # Apply is still refused: the checks have not run.
        dialog._refresh_merge_state()
        assert dialog._merge_button.cget("state") == "disabled"

        dialog._on_run_checks()
        assert _wait_until(root, lambda: "passed" in dialog._checks_label.cget("text"))
        assert _wait_until(root, lambda: dialog._merge_button.cget("state") == "normal")

        dialog._on_merge()
        assert _wait_until(root, lambda: "Applied" in dialog._merge_note.cget("text"))
        assert (project / "core" / "app.py").read_text(encoding="utf-8") == "x = 12\n"
        # The applied workspace is gone and a fresh one is opened, so Apply is off.
        assert _wait_until(root, lambda: dialog._apply_button.cget("state") == "disabled")
    finally:
        root.destroy()


@pytest.mark.skipif(not _display_available(), reason="no display available")
def test_dialog_discards_a_workspace(project: Path, tmp_path: Path) -> None:
    import tkinter as tk

    from gui.app import DARK_THEME
    from gui.code_studio_dialog import CodeStudioDialog

    root = tk.Tk()
    try:
        dialog = CodeStudioDialog(
            root, _service(project, tmp_path), theme=DARK_THEME, font="TkDefaultFont", mono="TkFixedFont"
        )
        root.update()
        assert _wait_until(root, lambda: dialog._workspace_list.size() == 1)

        dialog._on_discard()
        root.update()
        assert "Discarded" in dialog._merge_note.cget("text")
        # A fresh workspace is opened automatically for the next change.
        assert _wait_until(root, lambda: dialog._current_slug == "PR2")
        assert (project / "core" / "app.py").read_text(encoding="utf-8") == "x = 1\n"
    finally:
        root.destroy()


@pytest.mark.skipif(not _display_available(), reason="no display available")
def test_dialog_reports_an_unreachable_model_honestly(project: Path, tmp_path: Path) -> None:
    import tkinter as tk

    from core.coder_model import CoderUnavailable
    from gui.app import DARK_THEME
    from gui.code_studio_dialog import CodeStudioDialog

    class Down(FakeCoder):
        def list_models(self):
            raise CoderUnavailable("connection refused")

    root = tk.Tk()
    try:
        service = CodeStudioService(
            project,
            studio=CodeStudio(project, workspaces_dir=tmp_path / "ws"),
            client=Down([]),
            check_command=("python", "-c", "pass"),
        )
        dialog = CodeStudioDialog(
            root, service, theme=DARK_THEME, font="TkDefaultFont", mono="TkFixedFont"
        )
        root.update()
        assert "not running" in dialog._status_line.cget("text")
        assert _wait_until(root, lambda: dialog._workspace_list.size() == 1)

        dialog._request_var.set("do something")
        dialog._on_ask_model()
        assert _wait_until(root, lambda: "not running" in dialog._merge_note.cget("text"))
    finally:
        root.destroy()


@pytest.mark.skipif(not _display_available(), reason="no display available")
def test_sidebar_action_opens_the_studio_window(project: Path, tmp_path: Path, monkeypatch) -> None:
    """The sidebar entry opens the Studio window (rather than running the CLI)."""
    import tkinter as tk

    import gui.code_studio as studio_module
    from gui.actions import action_by_key
    from gui.app import FaydGUI
    from gui.code_studio_dialog import CodeStudioDialog

    # The window points the Studio at the running project; in a test that is the
    # throwaway fixture repo, so nothing is written beside the real checkout.
    monkeypatch.setattr(
        studio_module,
        "CodeStudioService",
        lambda root, **kwargs: _service(project, tmp_path),
    )
    root = tk.Tk()
    try:
        app = FaydGUI(root, service=_gui_service())
        root.update()
        app._on_action(action_by_key("code-studio"))
        root.update()
        assert isinstance(getattr(app, "_code_studio_dialog", None), CodeStudioDialog)
        # Opening it again lifts the existing window instead of making a second one.
        first = app._code_studio_dialog
        app._on_action(action_by_key("code-studio"))
        root.update()
        assert app._code_studio_dialog is first
    finally:
        root.destroy()


def _gui_service():
    """A minimal GUI service, so the window can be built without a real session."""
    from types import SimpleNamespace
    from typing import Any, cast

    from gui.service import ChatService

    class FakeSession:
        """A stand-in for ``JarvisMode``; the chat switcher only reads a few fields."""

        def __init__(self) -> None:
            self.session = None
            self.assistant: Any = SimpleNamespace(name="grounded")
            self.help_text = "help"
            self.voice = SimpleNamespace(assistant=self.assistant)

        def ask(self, message, **kwargs):
            return SimpleNamespace(text="ok", citations=(), mode="answer")

        def new_conversation(self):
            return "s1"

        def save(self):
            return "s1"

    return ChatService(mode=cast(Any, FakeSession()), data_root=Path.cwd(), ultron_model=None)
