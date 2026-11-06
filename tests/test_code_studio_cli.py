"""Tests for the Code Studio CLI -- the headless twin of the Studio window.

The CLI drives the *same* :class:`gui.code_studio.CodeStudioService` the window
does, so these tests run the real commands against a real throwaway git repository
and check the exit codes and the JSON the machine callers read. No model is
needed: only the git, file, and task commands are exercised here.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest
from cli import studio as studio_cli


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


def _run(project: Path, *argv: str) -> tuple[int, str, str]:
    """Run one CLI command and return ``(exit_code, stdout, stderr)``."""
    import contextlib
    import io

    out, err = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        code = studio_cli.main(["--root", str(project), *argv])
    return code, out.getvalue(), err.getvalue()


def _ok(project: Path, *argv: str) -> str:
    """Run a command that must succeed, and return its stdout."""
    code, out, err = _run(project, *argv)
    assert code == 0, f"{argv} failed ({code}): {out}{err}"
    return out


# --- status and workspaces -------------------------------------------------


def test_status_reports_the_project(project: Path) -> None:
    code, out, _err = _run(project, "status")
    assert code == 0
    assert "Checkout : git" in out


def test_new_names_the_workspace_and_list_shows_it(project: Path) -> None:
    assert "PR1" in _ok(project, "new", "first change")
    assert "PR1" in _ok(project, "list")


def test_new_json_reports_the_workspace(project: Path) -> None:
    payload = json.loads(_ok(project, "new", "first change", "--json"))
    assert payload["ok"] is True
    assert "PR1" in payload["message"]


def test_shared_flags_work_after_the_subcommand(project: Path) -> None:
    """``studio list --json`` must behave like ``studio --json list``."""
    after = json.loads(_ok(project, "list", "--json"))
    before = json.loads(_ok(project, "--json", "list"))
    assert after == before == []


# --- files -----------------------------------------------------------------


def test_tree_lists_the_projects_files(project: Path) -> None:
    _ok(project, "new", "x")
    assert "core" in _ok(project, "tree", "PR1")


def test_tree_json_is_machine_readable(project: Path) -> None:
    _ok(project, "new", "x")
    payload = json.loads(_ok(project, "tree", "PR1", "--json"))
    assert any(entry["name"] == "core" for entry in payload)


def test_touch_mkdir_rename_cp_rm_round_trip(project: Path) -> None:
    _ok(project, "new", "x")
    _ok(project, "touch", "PR1", "core/new.py")
    _ok(project, "mkdir", "PR1", "pkg")
    _ok(project, "rename", "PR1", "core/new.py", "core/renamed.py")
    _ok(project, "cp", "PR1", "core/renamed.py", "core/copy.py")
    _ok(project, "rm", "PR1", "core/copy.py")
    workspace = project / "data" / "studio" / "workspaces" / "PR1"
    assert (workspace / "core" / "renamed.py").is_file()
    assert (workspace / "pkg").is_dir()
    assert not (workspace / "core" / "copy.py").exists()


def test_read_and_write(project: Path) -> None:
    _ok(project, "new", "x")
    assert "import os" in _ok(project, "read", "PR1", "core/app.py")
    local = project / "content.txt"
    local.write_text("x = 5\n", encoding="utf-8")
    _ok(project, "write", "PR1", "core/new.py", "--file", str(local))
    assert "x = 5" in _ok(project, "read", "PR1", "core/new.py")


def test_outline_and_notes(project: Path) -> None:
    _ok(project, "new", "x")
    assert "baz" in _ok(project, "outline", "PR1", "core/app.py")
    assert "TODO" in _ok(project, "notes", "PR1")


def test_search_and_replace(project: Path) -> None:
    _ok(project, "new", "x")
    assert "app.py" in _ok(project, "search", "PR1", "import os")
    # Preview by default: nothing is written without --apply.
    assert "Would replace" in _ok(project, "replace", "PR1", "--old", "import os", "--new", "import sys")
    workspace = project / "data" / "studio" / "workspaces" / "PR1"
    assert "import os" in (workspace / "core" / "app.py").read_text(encoding="utf-8")
    _ok(project, "replace", "PR1", "--old", "import os", "--new", "import sys", "--apply")
    assert "import sys" in (workspace / "core" / "app.py").read_text(encoding="utf-8")


# --- git -------------------------------------------------------------------


def test_git_status_log_and_branches(project: Path) -> None:
    _ok(project, "new", "x")
    _ok(project, "git", "PR1", "status")
    assert "initial" in _ok(project, "git", "PR1", "log")
    assert "studio/PR1" in _ok(project, "git", "PR1", "branches")


def _content_file(project: Path) -> str:
    path = project / "content.txt"
    path.write_text("x = 5\n", encoding="utf-8")
    return str(path)


def test_git_stage_and_commit(project: Path) -> None:
    _ok(project, "new", "x")
    _ok(project, "write", "PR1", "core/app.py", "--file", _content_file(project))
    _ok(project, "git", "PR1", "stage", "--paths", "core/app.py")
    _ok(project, "git", "PR1", "commit", "--message", "set x")
    payload = json.loads(_ok(project, "git", "PR1", "log", "--json"))
    assert payload[0]["subject"] == "set x"


def test_git_show_and_blame(project: Path) -> None:
    _ok(project, "new", "x")
    assert _ok(project, "git", "PR1", "show")
    assert "demo" in _ok(project, "git", "PR1", "blame", "--path", "README.md")


def test_git_fetch_is_gated_behind_approval(project: Path) -> None:
    """A network action never runs without --approve, and says so."""
    _ok(project, "new", "x")
    code, out, err = _run(project, "git", "PR1", "fetch")
    assert code == 2
    assert "approve" in (out + err).lower()


def test_git_stash_and_pop(project: Path) -> None:
    _ok(project, "new", "x")
    _ok(project, "write", "PR1", "core/app.py", "--file", _content_file(project))
    _ok(project, "git", "PR1", "stash-save", "--message", "wip", "--untracked")
    assert "(clean)" in _ok(project, "git", "PR1", "status")
    _ok(project, "git", "PR1", "stash-pop")
    assert "(clean)" not in _ok(project, "git", "PR1", "status")


# --- tasks -----------------------------------------------------------------


def test_tasks_lists_the_project_commands(project: Path) -> None:
    _ok(project, "new", "x")
    assert "tests" in _ok(project, "tasks", "PR1")


def test_tasks_json(project: Path) -> None:
    _ok(project, "new", "x")
    payload = json.loads(_ok(project, "tasks", "PR1", "--json"))
    assert payload["project"] == "python"
    assert any(spec["key"] == "tests" for spec in payload["tasks"])


def test_run_a_task_streams_output(project: Path) -> None:
    _ok(project, "new", "x")
    out = _ok(
        project, "run", "PR1", "custom",
        "--command", f'{sys.executable} -c "print(\'ran\')"',
    )
    assert "ran" in out


def test_run_an_unknown_task_is_an_argument_error(project: Path) -> None:
    _ok(project, "new", "x")
    code, _out, err = _run(project, "run", "PR1", "not-a-task")
    assert code == 2
    assert "not-a-task" in err or "unknown" in err.lower()


# --- diff, checks, merge ---------------------------------------------------


def test_diff_and_check_and_merge(project: Path) -> None:
    _ok(project, "new", "x")
    _ok(project, "write", "PR1", "core/app.py", "--file", _content_file(project))
    assert "+x = 5" in _ok(project, "diff", "PR1")

    # The checks are the project's own; a passing command unlocks the merge gate.
    code, _out, _err = _run(
        project, "check", "PR1", "--check-command", f'{sys.executable} -c "print(\'ok\')"'
    )
    assert code == 0
    assert _run(project, "merge", "PR1")[0] == 0
    assert "x = 5" in (project / "core" / "app.py").read_text(encoding="utf-8")


def test_discard_removes_the_workspace(project: Path) -> None:
    _ok(project, "new", "x")
    _ok(project, "discard", "PR1")
    assert "No workspaces yet" in _ok(project, "list")


def test_an_unknown_slug_is_a_gate_refusal(project: Path) -> None:
    code, _out, err = _run(project, "read", "PRX", "core/app.py")
    assert code == 4
    assert err


def test_help_lists_every_command() -> None:
    import contextlib
    import io

    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        code = studio_cli.main([])
    assert code == 0
    for command in ("status", "tree", "replace", "git", "tasks", "run"):
        assert command in out.getvalue()


def test_base_branch_flag_reaches_the_engine(project: Path) -> None:
    code, out, _err = _run(project, "--base-branch", "main", "status")
    assert code == 0
    assert "main" in out
