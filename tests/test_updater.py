"""Program-update checks against real, local git repositories.

These tests build small git repos on disk (a bare origin plus clones) so the
updater exercises real ``git`` behaviour -- fetching, comparing, and
fast-forwarding -- rather than a mock.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest
from core.updater import (
    UpdateStatus,
    apply_update,
    check_for_updates,
    format_status,
    is_source_checkout,
)

pytestmark = pytest.mark.skipif(shutil.which("git") is None, reason="git is not installed")


def _env() -> dict[str, str]:
    import os

    env = dict(os.environ)
    env.update(
        {
            "GIT_TERMINAL_PROMPT": "0",
            "GIT_AUTHOR_NAME": "Test",
            "GIT_AUTHOR_EMAIL": "test@example.com",
            "GIT_COMMITTER_NAME": "Test",
            "GIT_COMMITTER_EMAIL": "test@example.com",
        }
    )
    return env


def _git(cwd: Path, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ("git", *args), cwd=str(cwd), env=_env(), capture_output=True, text=True, check=True
    )


def _commit(cwd: Path, message: str) -> None:
    _git(cwd, "add", "-A")
    _git(cwd, "commit", "-m", message)


def _repo_pair(tmp_path: Path) -> tuple[Path, Path]:
    """A bare ``origin`` plus a clone that is one commit behind. Returns (origin, installed)."""
    origin = tmp_path / "origin.git"
    _git(tmp_path, "init", "--bare", "-b", "main", str(origin))

    seed = tmp_path / "seed"
    _git(tmp_path, "clone", str(origin), str(seed))
    (seed / ".gitignore").write_text("data/\n", encoding="utf-8")
    (seed / "app.txt").write_text("v1\n", encoding="utf-8")
    _commit(seed, "initial")
    _git(seed, "push", "-u", "origin", "main")

    installed = tmp_path / "installed"
    _git(tmp_path, "clone", str(origin), str(installed))

    (seed / "app.txt").write_text("v2\n", encoding="utf-8")
    (seed / "new_module.py").write_text("x = 1\n", encoding="utf-8")
    _commit(seed, "second commit")
    _git(seed, "push", "origin", "main")
    return origin, installed


def test_is_source_checkout_true_for_a_git_tree(tmp_path: Path) -> None:
    installed = tmp_path / "repo"
    _git(tmp_path, "init", "-b", "main", str(installed))
    assert is_source_checkout(installed)


def test_is_source_checkout_false_without_git(tmp_path: Path) -> None:
    assert not is_source_checkout(tmp_path)


def test_check_reports_an_available_update(tmp_path: Path) -> None:
    _origin, installed = _repo_pair(tmp_path)

    status = check_for_updates(str(installed), root=installed)

    assert isinstance(status, UpdateStatus)
    assert status.available is True
    assert status.behind == 1
    assert status.ahead == 0
    assert status.branch == "main"
    assert "new_module.py" in status.changed_files
    assert status.dirty is False
    assert "Update available" in status.summary()
    # The check must not modify the working tree.
    assert not (installed / "new_module.py").exists()


def test_apply_fast_forwards_the_checkout(tmp_path: Path) -> None:
    _origin, installed = _repo_pair(tmp_path)

    status = apply_update(str(installed), root=installed)

    assert status.applied is True
    assert (installed / "new_module.py").read_text(encoding="utf-8") == "x = 1\n"
    assert (installed / "app.txt").read_text(encoding="utf-8") == "v2\n"
    assert "Updated" in status.summary()


def test_apply_refuses_when_there_are_local_changes(tmp_path: Path) -> None:
    _origin, installed = _repo_pair(tmp_path)
    (installed / "app.txt").write_text("local edit\n", encoding="utf-8")

    status = apply_update(str(installed), root=installed)

    assert status.applied is False
    assert status.dirty is True
    assert (installed / "app.txt").read_text(encoding="utf-8") == "local edit\n"
    assert "local changes" in status.summary().lower()
    # The refused file is named, so the user knows exactly what to commit or stash.
    assert status.conflicting_files == ("app.txt",)
    assert "app.txt" in status.reason


def test_local_change_to_an_untouched_file_does_not_block_an_update(tmp_path: Path) -> None:
    """A scratch edit to a file the update does not change must not refuse it.

    git's own ``--ff-only`` leaves such a file exactly as it is, so refusing was a
    dead end for a harmless edit.
    """
    _origin, installed = _repo_pair(tmp_path)
    (installed / ".gitignore").write_text("data/\n# my scratch note\n", encoding="utf-8")

    status = check_for_updates(str(installed), root=installed)
    assert status.dirty is True
    assert status.conflicting_files == ()

    applied = apply_update(str(installed), root=installed)

    assert applied.applied is True
    assert (installed / "new_module.py").read_text(encoding="utf-8") == "x = 1\n"
    # The user's own edit is left untouched by the fast-forward.
    assert "# my scratch note" in (installed / ".gitignore").read_text(encoding="utf-8")


def test_conflicting_files_names_only_the_overlapping_ones(tmp_path: Path) -> None:
    _origin, installed = _repo_pair(tmp_path)
    (installed / "app.txt").write_text("local edit\n", encoding="utf-8")  # overlaps
    (installed / ".gitignore").write_text("data/\n# scratch\n", encoding="utf-8")  # does not

    status = check_for_updates(str(installed), root=installed)

    assert status.conflicting_files == ("app.txt",)
    assert status.dirty is True


def test_untracked_files_do_not_block_an_update(tmp_path: Path) -> None:
    origin, installed = _repo_pair(tmp_path)
    (installed / "scratch notes.txt").write_text("mine\n", encoding="utf-8")

    status = apply_update(str(origin), root=installed)

    assert status.applied is True
    assert status.dirty is False
    # The untracked file is left alone by the fast-forward.
    assert (installed / "scratch notes.txt").read_text(encoding="utf-8") == "mine\n"


def test_apply_refuses_when_the_branch_is_ahead(tmp_path: Path) -> None:
    _origin, installed = _repo_pair(tmp_path)
    (installed / "local_only.py").write_text("stay\n", encoding="utf-8")
    _commit(installed, "local work")

    status = apply_update(str(installed), root=installed)

    assert status.applied is False
    assert status.ahead >= 1
    assert (installed / "local_only.py").exists()


def test_check_is_quiet_when_up_to_date(tmp_path: Path) -> None:
    origin, installed = _repo_pair(tmp_path)
    apply_update(str(origin), root=installed)  # bring the clone fully up to date

    status = check_for_updates(str(origin), root=installed)

    assert status.available is False
    assert status.behind == 0
    assert "Already up to date" in status.summary()


def test_check_on_a_non_git_directory_is_unavailable(tmp_path: Path) -> None:
    status = check_for_updates("https://example.invalid/repo.git", root=tmp_path)

    assert status.available is False
    assert "portable build" in status.reason


def test_format_status_includes_a_command_trail(tmp_path: Path) -> None:
    _origin, installed = _repo_pair(tmp_path)
    status = check_for_updates(str(installed), root=installed)

    report = format_status(status, verbose=True)

    assert "Program update check" in report
    assert "git commands" in report
    assert "rev-list" in report


def test_check_falls_back_to_the_remote_default_branch(tmp_path: Path) -> None:
    origin, installed = _repo_pair(tmp_path)
    _git(installed, "checkout", "-b", "local-only")

    status = check_for_updates(str(origin), root=installed)

    assert status.available is True
    assert status.branch == "main"
    assert status.head_branch == "local-only"
    assert status.on_tracking_branch is False
    assert "checkout is on local-only" in status.summary()


def test_apply_refuses_when_not_on_the_compared_branch(tmp_path: Path) -> None:
    origin, installed = _repo_pair(tmp_path)
    _git(installed, "checkout", "-b", "local-only")

    status = apply_update(str(origin), root=installed)

    assert status.applied is False
    assert not (installed / "new_module.py").exists()


def test_check_flags_changes_to_committed_training_files(tmp_path: Path) -> None:
    origin, installed = _repo_pair(tmp_path)
    seed = tmp_path / "seed"
    (seed / "ml" / "artifacts").mkdir(parents=True)
    (seed / "ml" / "artifacts" / "release_manifest.json").write_text("{}", encoding="utf-8")
    _commit(seed, "retrain")
    _git(seed, "push", "origin", "main")

    status = check_for_updates(str(origin), root=installed)

    assert "ml/artifacts/release_manifest.json" in status.learned_files_changed
    assert "training files" in format_status(status)


def test_apply_leaves_the_data_folder_untouched(tmp_path: Path) -> None:
    """The update never copies or moves the user's learned state in data/."""
    origin, installed = _repo_pair(tmp_path)
    (installed / "data" / "ml" / "artifacts").mkdir(parents=True)
    (installed / "data" / "ml" / "artifacts" / "policy.npz").write_bytes(b"learned")
    (installed / "data" / "learned_cursor.json").write_text("{}", encoding="utf-8")

    status = apply_update(str(origin), root=installed)

    assert status.applied is True
    # The live data folder is left exactly as it was.
    assert (installed / "data" / "ml" / "artifacts" / "policy.npz").read_bytes() == b"learned"
    assert (installed / "data" / "learned_cursor.json").exists()
    # No data-backup-* folder is created beside the checkout.
    assert not list(tmp_path.glob("data-backup-*"))
    assert not list(installed.glob("data-backup-*"))


def test_apply_succeeds_without_a_data_folder(tmp_path: Path) -> None:
    origin, installed = _repo_pair(tmp_path)

    status = apply_update(str(origin), root=installed)

    assert status.applied is True
    assert not list(installed.glob("data-backup-*"))


def test_cli_update_command_uses_the_local_remote(tmp_path, capsys, monkeypatch) -> None:
    """`python -m cli update --repo <local> --json` reaches the real updater."""
    from cli import menu

    origin, installed = _repo_pair(tmp_path)
    monkeypatch.setattr("core.updater._repo_root", lambda: installed)

    code = menu.run_update(["--repo", str(origin), "--json"])
    out = capsys.readouterr().out

    assert code == 0
    assert '"available": true' in out
    assert '"behind": 1' in out


def test_portable_entry_lists_the_update_command() -> None:
    from cli.__main__ import _COMMANDS

    assert _COMMANDS["update"].__module__ == "cli.menu"


def test_sanitize_remote_url_strips_credentials() -> None:
    from core.updater import sanitize_remote_url

    assert (
        sanitize_remote_url("https://user:ghp_secret@github.com/org/repo.git")
        == "https://github.com/org/repo.git"
    )
    assert sanitize_remote_url("https://github.com/org/repo.git") == "https://github.com/org/repo.git"
    assert sanitize_remote_url("git@github.com:org/repo.git") == "git@github.com:org/repo.git"


def test_check_report_never_prints_a_token() -> None:
    from core.updater import GitCommand, _fetch_failure

    url = "https://user:ghp_SECRET@example.com/org/repo.git"
    message = _fetch_failure(
        GitCommand(argv=("git", "fetch"), returncode=128, stderr=f"fatal: unable to access '{url}/'"),
        url,
    )
    assert "ghp_SECRET" not in message
    assert "example.com" not in message
