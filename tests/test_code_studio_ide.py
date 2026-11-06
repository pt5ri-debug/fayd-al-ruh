"""Window tests for the Code Studio's IDE panels (skipped with no display).

The universal Studio puts a developer's whole day in the window: the explorer, the
editor, find & replace, source control, the project's own tasks, and a notes scan.
These tests drive the real Tk window against a real throwaway git repository, so a
regression in a panel fails here rather than only in the headless service tests.
"""

from __future__ import annotations

import subprocess
import sys
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
    (repo / "core" / "app.py").write_text(
        "import os\n\n\ndef baz(a, b):\n    # TODO: document\n    return a + b\n",
        encoding="utf-8",
    )
    (repo / "README.md").write_text("# demo\n", encoding="utf-8")
    (repo / "pyproject.toml").write_text("[project]\nname='demo'\n", encoding="utf-8")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-qm", "initial")
    return repo


class FakeCoder:
    """A scripted coding model, so no Ollama server is needed."""

    def __init__(self, script: list[str] | None = None) -> None:
        self.model = "qwen2.5-coder:7b"
        self.script = list(script or [])

    def list_models(self) -> tuple[str, ...]:
        return ("qwen2.5-coder:7b",)

    def chat(self, messages, *, temperature: float = 0.2) -> str:
        if not self.script:
            return '{"action": "done", "summary": "nothing left to do"}'
        return self.script.pop(0)


def _service(project: Path, tmp_path: Path) -> CodeStudioService:
    return CodeStudioService(
        project,
        studio=CodeStudio(project, workspaces_dir=tmp_path / "ws"),
        client=FakeCoder(),
        check_command=(sys.executable, "-c", "print('ok')"),
    )


def _wait_until(root, predicate, *, timeout: float = 10.0) -> bool:
    import time

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


def _transcript_text(dialog) -> str:
    return "\n".join(
        child.get("1.0", "end") for child in _all_text_widgets(dialog._transcript)
    )


@pytest.fixture()
def dialog(project: Path, tmp_path: Path):
    import tkinter as tk

    from gui.app import DARK_THEME
    from gui.code_studio_dialog import CodeStudioDialog

    root = tk.Tk()
    window = CodeStudioDialog(
        root, _service(project, tmp_path), theme=DARK_THEME, font="TkDefaultFont", mono="TkFixedFont"
    )
    root.update()
    assert _wait_until(root, lambda: window._workspace_list.size() == 1)
    try:
        yield window, root
    finally:
        root.destroy()


# --- the sidebar groups the tools ------------------------------------------


@pytest.mark.skipif(not _display_available(), reason="no display available")
def test_sidebar_has_one_group_per_part_of_a_developers_day(dialog) -> None:
    window, _root = dialog
    keys = set(window._sidebar_column.sections)
    for expected in (
        "workspaces", "explorer", "editor", "find", "scm", "tasks", "notes", "review",
    ):
        assert expected in keys


@pytest.mark.skipif(not _display_available(), reason="no display available")
def test_the_left_column_scrolls_like_every_other_window(dialog) -> None:
    """The tool column keeps an always-visible scrollbar, so it is never a dead end."""
    window, _root = dialog
    assert window._sidebar_column.scrollbar.winfo_ismapped()
    assert window._sidebar_column.canvas.winfo_ismapped()


# --- explorer --------------------------------------------------------------


@pytest.mark.skipif(not _display_available(), reason="no display available")
def test_explorer_creates_a_new_file_and_folder(dialog, monkeypatch) -> None:
    import gui.code_studio_dialog as module

    window, root = dialog
    monkeypatch.setattr(module, "_ask_string", lambda *a, **k: "core/new.py")
    window._on_new_file()
    root.update()
    assert (window.service.workspace("PR1").path / "core" / "new.py").is_file()
    assert "core/new.py" in [window._file_list.get(i) for i in range(window._file_list.size())]

    monkeypatch.setattr(module, "_ask_string", lambda *a, **k: "core/pkg")
    window._on_new_folder()
    root.update()
    assert (window.service.workspace("PR1").path / "core" / "pkg").is_dir()


@pytest.mark.skipif(not _display_available(), reason="no display available")
def test_explorer_rename_copy_and_delete(dialog, monkeypatch) -> None:
    import gui.code_studio_dialog as module

    window, root = dialog
    workspace = window.service.workspace("PR1")
    index = [window._file_list.get(i) for i in range(window._file_list.size())].index("README.md")
    window._file_list.selection_set(index)
    window._on_file_select()
    root.update()

    monkeypatch.setattr(module, "_ask_string", lambda *a, **k: "README.rst")
    window._on_rename_file()
    root.update()
    assert (workspace.path / "README.rst").is_file()

    monkeypatch.setattr(module, "_ask_string", lambda *a, **k: "docs/README.rst")
    index = [window._file_list.get(i) for i in range(window._file_list.size())].index("README.rst")
    window._file_list.selection_set(index)
    window._on_copy_file()
    root.update()
    assert (workspace.path / "docs" / "README.rst").is_file()

    monkeypatch.setattr(module, "_confirm", lambda *a, **k: True)
    index = [window._file_list.get(i) for i in range(window._file_list.size())].index("README.rst")
    window._file_list.selection_set(index)
    window._on_delete_file()
    root.update()
    assert not (workspace.path / "README.rst").exists()


# --- editor ----------------------------------------------------------------


@pytest.mark.skipif(not _display_available(), reason="no display available")
def test_editor_outline_and_blame(dialog) -> None:
    window, root = dialog
    index = [window._file_list.get(i) for i in range(window._file_list.size())].index("core/app.py")
    window._file_list.selection_set(index)
    window._on_file_select()
    root.update()
    assert window._editor_panel.winfo_ismapped()

    window._on_outline()
    root.update()
    assert "baz" in window._review.get("1.0", "end")

    window._on_blame()
    root.update()
    assert "app.py" in window._review.get("1.0", "end") or "baz" in window._review.get("1.0", "end")


@pytest.mark.skipif(not _display_available(), reason="no display available")
def test_editor_discards_a_files_changes(dialog, monkeypatch) -> None:
    import gui.code_studio_dialog as module

    window, root = dialog
    window.service.save_file("PR1", "core/app.py", "x = 99\n")
    index = [window._file_list.get(i) for i in range(window._file_list.size())].index("core/app.py")
    window._file_list.selection_set(index)
    monkeypatch.setattr(module, "_confirm", lambda *a, **k: True)
    window._on_discard_file()
    root.update()
    assert (
        window.service.workspace("PR1").path / "core" / "app.py"
    ).read_text(encoding="utf-8") == "import os\n\n\ndef baz(a, b):\n    # TODO: document\n    return a + b\n"


# --- find & replace --------------------------------------------------------


@pytest.mark.skipif(not _display_available(), reason="no display available")
def test_find_and_replace_previews_then_applies(dialog, monkeypatch) -> None:
    import gui.code_studio_dialog as module

    window, root = dialog
    window._find_var.set("import os")
    window._on_search()
    root.update()
    assert "app.py" in window._review.get("1.0", "end")

    window._replace_var.set("import sys")
    window._on_replace_preview()
    root.update()
    assert "Would replace" in window._merge_note.cget("text")
    workspace = window.service.workspace("PR1")
    assert "import os" in (workspace.path / "core" / "app.py").read_text(encoding="utf-8")

    monkeypatch.setattr(module, "_confirm", lambda *a, **k: True)
    window._on_replace_apply()
    root.update()
    assert "import sys" in (workspace.path / "core" / "app.py").read_text(encoding="utf-8")


# --- source control --------------------------------------------------------


@pytest.mark.skipif(not _display_available(), reason="no display available")
def test_source_control_lists_then_stages_and_commits(dialog) -> None:
    window, root = dialog
    window.service.save_file("PR1", "core/app.py", "x = 7\n")
    window._refresh_source_control()
    root.update()
    assert window._git_status_list.size() >= 1
    assert "core/app.py" in window._git_status_list.get(0)

    window._git_status_list.selection_set(0)
    window._on_stage()
    root.update()
    window._commit_var.set("set x to 7")
    window._on_commit()
    root.update()
    assert window.service.git_status("PR1") == ()
    assert window.service.git_log("PR1", limit=1)[0].subject == "set x to 7"


@pytest.mark.skipif(not _display_available(), reason="no display available")
def test_source_control_history_and_branches_render(dialog) -> None:
    window, root = dialog
    window._on_history()
    root.update()
    assert "initial" in window._review.get("1.0", "end")
    window._on_branches()
    root.update()
    assert "studio/PR1" in window._review.get("1.0", "end")


@pytest.mark.skipif(not _display_available(), reason="no display available")
def test_source_control_creates_a_branch(dialog) -> None:
    window, root = dialog
    window._branch_var.set("feature/x")
    window._on_new_branch()
    root.update()
    assert "feature/x" in window._branch_var.get() or "Created" in window._merge_note.cget("text")
    names = {branch.name for branch in window.service.git_branches("PR1")}
    assert "feature/x" in names


@pytest.mark.skipif(not _display_available(), reason="no display available")
def test_source_control_stash_round_trip(dialog) -> None:
    window, root = dialog
    window.service.save_file("PR1", "core/app.py", "x = 11\n")
    window._on_stash()
    root.update()
    assert window.service.git_status("PR1") == ()
    window._on_stash_pop()
    root.update()
    assert window.service.git_status("PR1") != ()


@pytest.mark.skipif(not _display_available(), reason="no display available")
def test_remote_action_asks_before_touching_the_network(dialog, monkeypatch) -> None:
    """A fetch never runs until the user confirms it in the window."""
    import gui.code_studio_dialog as module

    window, root = dialog
    asked: list[str] = []

    def _refuse(parent, title, message):
        asked.append(title)
        return False

    monkeypatch.setattr(module, "_confirm", _refuse)
    window._on_fetch()
    root.update()
    assert asked == ["Fetch remote"]
    # A refusal is silent: nothing is reported as done.
    assert "Fetch" not in window._merge_note.cget("text")


@pytest.mark.skipif(not _display_available(), reason="no display available")
def test_export_patch_offers_a_save(dialog, monkeypatch) -> None:
    import gui.code_studio_dialog as module

    window, root = dialog
    window.service.save_file("PR1", "core/app.py", "x = 13\n")
    saved: dict[str, str] = {}
    monkeypatch.setattr(
        module, "_ask_save", lambda parent, text, default_name: saved.update(
            {"text": text, "name": default_name}
        )
    )
    window._on_export_patch()
    root.update()
    assert "diff --git" in saved["text"]
    assert saved["name"] == "PR1.patch"


# --- tasks -----------------------------------------------------------------


@pytest.mark.skipif(not _display_available(), reason="no display available")
def test_task_menu_lists_the_projects_commands(dialog) -> None:
    window, _root = dialog
    assert window._task_list.size() >= 1
    labels = [window._task_list.get(i) for i in range(window._task_list.size())]
    assert any("tests" in label.lower() or "Tests" in label for label in labels)


@pytest.mark.skipif(not _display_available(), reason="no display available")
def test_running_a_task_streams_its_output_into_the_transcript(dialog) -> None:
    window, root = dialog
    window._on_run_task()
    assert _wait_until(root, lambda: "passed" in window._merge_note.cget("text").lower())
    # The output is a transcript row, so it is selectable and copyable like a message.
    assert "ok" in _transcript_text(window)


@pytest.mark.skipif(not _display_available(), reason="no display available")
def test_running_a_custom_command(dialog) -> None:
    window, root = dialog
    window._custom_task_var.set(f'{sys.executable} -c "print(\'custom-output\')"')
    window._on_run_custom_task()
    assert _wait_until(root, lambda: "passed" in window._merge_note.cget("text").lower())
    assert "custom-output" in _transcript_text(window)


# --- notes -----------------------------------------------------------------


@pytest.mark.skipif(not _display_available(), reason="no display available")
def test_notes_scan_finds_the_todo(dialog) -> None:
    window, root = dialog
    window._on_scan_notes()
    root.update()
    assert "TODO" in window._review.get("1.0", "end")
    assert "TODO" in window._notes_label.cget("text")


# --- review and apply ------------------------------------------------------


@pytest.mark.skipif(not _display_available(), reason="no display available")
def test_review_panel_runs_the_checks_and_unlocks_apply(dialog) -> None:
    window, root = dialog
    window.service.save_file("PR1", "core/app.py", "x = 21\n")
    window._refresh_merge_state()
    assert window._merge_button.cget("state") == "disabled"
    window._on_run_checks()
    assert _wait_until(root, lambda: window._merge_button.cget("state") == "normal")
    assert "passed" in window._checks_label.cget("text")


@pytest.mark.skipif(not _display_available(), reason="no display available")
def test_the_window_opens_a_workspace_and_leaves_the_project_untouched(dialog) -> None:
    """The Studio only ever edits a workspace, never the running checkout."""
    window, _root = dialog
    assert window._current_slug == "PR1"
    window.service.save_file("PR1", "core/app.py", "x = 30\n")
    assert window.service.workspace("PR1").path != window.service.root
    assert (window.service.root / "core" / "app.py").read_text(encoding="utf-8").startswith(
        "import os"
    )


@pytest.mark.skipif(not _display_available(), reason="no display available")
def test_the_apply_button_is_the_users_approval(dialog) -> None:
    """A passing change still needs the user's own click to reach the project."""
    window, root = dialog
    window.service.save_file("PR1", "core/app.py", "x = 31\n")
    window._on_run_checks()
    assert _wait_until(root, lambda: window._apply_button.cget("state") == "normal")
    # Nothing has reached the project yet.
    assert "x = 31" not in (window.service.root / "core" / "app.py").read_text(encoding="utf-8")
    window._on_merge()
    assert _wait_until(root, lambda: "Applied" in window._merge_note.cget("text"))
    assert "x = 31" in (window.service.root / "core" / "app.py").read_text(encoding="utf-8")
