"""Tests for the universal Code Studio engines: git, files, and tasks.

The Studio is the project's own code host and IDE in a window, so these tests take
it as seriously as the workflow it implements: they build a real, throwaway git
repository, make a real worktree, run real git commands, really edit files, and
really run commands. Nothing is mocked except the coding model (which is never
needed here), so a regression in the engines fails a real check rather than a stub.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest
from core.code_studio import CodeStudio, StudioError
from core.studio_files import (
    FileEngine,
    NoteScan,
    OutlineReport,
    TreeEntry,
    iter_tree,
)
from core.studio_git import GitEngine, GitError, StatusEntry
from core.studio_tasks import (
    CATEGORY_BUILD,
    CATEGORY_CUSTOM,
    CATEGORY_QUALITY,
    CATEGORY_TEST,
    CommandOutcome,
    CommandRunner,
    CommandSpec,
    TaskError,
    catalog_for,
    custom_spec,
    detect_project,
    find_spec,
)
from gui.code_studio import CodeStudioService

# --- fixtures --------------------------------------------------------------


def _git(repo: Path, *args: str) -> str:
    completed = subprocess.run(
        ("git", *args),
        cwd=str(repo),
        capture_output=True,
        text=True,
        check=True,
        env={
            **os.environ,
            "GIT_AUTHOR_NAME": "Test",
            "GIT_AUTHOR_EMAIL": "test@example.com",
            "GIT_COMMITTER_NAME": "Test",
            "GIT_COMMITTER_EMAIL": "test@example.com",
        },
    )
    return completed.stdout


@pytest.fixture()
def project(tmp_path: Path) -> Path:
    """A real git repository with a couple of files and one commit."""
    repo = tmp_path / "project"
    repo.mkdir()
    _git(repo, "init", "-q", "-b", "main")
    (repo / "core").mkdir()
    (repo / "core" / "app.py").write_text(
        "import os\n\n\nclass Foo:\n    def bar(self, x):\n        # TODO: rename\n        return x\n\n\ndef baz(a, b):\n    return a + b\n",
        encoding="utf-8",
    )
    (repo / "README.md").write_text("# demo\n", encoding="utf-8")
    (repo / "pyproject.toml").write_text("[project]\nname='demo'\n", encoding="utf-8")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-qm", "initial")
    return repo


@pytest.fixture()
def studio(project: Path, tmp_path: Path) -> CodeStudio:
    return CodeStudio(project, workspaces_dir=tmp_path / "workspaces", base_branch="main")


@pytest.fixture()
def service(studio: CodeStudio, project: Path) -> CodeStudioService:
    return CodeStudioService(project, studio=studio, check_command=(sys.executable, "-c", "print('ok')"))


# --- git engine ------------------------------------------------------------


def test_git_engine_reports_status_for_a_new_workspace(studio: CodeStudio) -> None:
    workspace = studio.create("demo")
    git = GitEngine(workspace.path)
    assert git.is_repository()
    assert git.current_branch() == "studio/PR1"
    assert git.status() == ()
    assert git.is_clean()


def test_git_engine_parses_an_untracked_and_a_modified_file(studio: CodeStudio) -> None:
    workspace = studio.create("demo")
    (workspace.path / "new.py").write_text("y = 2\n", encoding="utf-8")
    (workspace.path / "core" / "app.py").write_text("import os\n\nx = 5\n", encoding="utf-8")
    git = GitEngine(workspace.path)
    entries = {entry.path: entry for entry in git.status()}
    assert entries["new.py"].untracked
    assert entries["core/app.py"].unstaged
    assert not entries["core/app.py"].untracked


def test_git_engine_stage_and_commit_records_the_change(studio: CodeStudio) -> None:
    workspace = studio.create("demo")
    (workspace.path / "new.py").write_text("y = 2\n", encoding="utf-8")
    git = GitEngine(workspace.path)
    assert git.stage("new.py").ok
    assert any(entry.staged for entry in git.status())
    result = git.commit("add new.py")
    assert result.ok
    assert git.is_clean()
    commits = git.log(limit=5)
    assert commits[0].subject == "add new.py"
    assert not commits[0].merge


def test_git_engine_commit_uses_a_default_message_when_none_is_given(studio: CodeStudio) -> None:
    """A blank message falls back to a Studio message rather than failing git."""
    workspace = studio.create("demo")
    (workspace.path / "new.py").write_text("y = 2\n", encoding="utf-8")
    git = GitEngine(workspace.path)
    git.stage("new.py")
    assert git.commit("   ").ok
    assert git.log(limit=1)[0].subject.startswith("Studio:")


def test_git_engine_restore_discards_an_unstaged_change(studio: CodeStudio) -> None:
    workspace = studio.create("demo")
    (workspace.path / "core" / "app.py").write_text("changed\n", encoding="utf-8")
    git = GitEngine(workspace.path)
    assert git.restore("core/app.py").ok
    assert git.is_clean()


def test_git_engine_stash_round_trip(studio: CodeStudio) -> None:
    workspace = studio.create("demo")
    (workspace.path / "core" / "app.py").write_text("changed\n", encoding="utf-8")
    git = GitEngine(workspace.path)
    assert git.stash("wip", include_untracked=True).ok
    assert git.is_clean()
    assert len(git.stash_list()) == 1
    assert git.stash_pop().ok
    assert not git.is_clean()


def test_git_engine_branches_list_the_studio_branch_and_main(studio: CodeStudio) -> None:
    workspace = studio.create("demo")
    git = GitEngine(workspace.path)
    names = {branch.name for branch in git.branches()}
    assert "main" in names
    assert "studio/PR1" in names
    current = [branch for branch in git.branches() if branch.current]
    assert current and current[0].name == "studio/PR1"


def test_git_engine_diff_shows_the_change(studio: CodeStudio) -> None:
    workspace = studio.create("demo")
    (workspace.path / "core" / "app.py").write_text("changed\n", encoding="utf-8")
    git = GitEngine(workspace.path)
    text = git.diff()
    assert "core/app.py" in text
    assert "-import os" in text or "-import os" in text.replace("+", "-", 1)


def test_git_engine_blame_names_a_line(studio: CodeStudio) -> None:
    workspace = studio.create("demo")
    git = GitEngine(workspace.path)
    text = git.blame("README.md")
    assert "demo" in text


def test_git_engine_apply_patch_round_trips(studio: CodeStudio) -> None:
    workspace = studio.create("demo")
    git = GitEngine(workspace.path)
    patch = (
        "diff --git a/added.txt b/added.txt\n"
        "new file mode 100644\n"
        "--- /dev/null\n"
        "+++ b/added.txt\n"
        "@@ -0,0 +1 @@\n"
        "+hello\n"
    )
    check = git.apply_patch(patch, check=True)
    assert check.ok
    assert not (workspace.path / "added.txt").exists()
    assert git.apply_patch(patch).ok
    assert (workspace.path / "added.txt").read_text(encoding="utf-8") == "hello\n"


def test_git_engine_refuses_a_path_that_escapes(studio: CodeStudio) -> None:
    workspace = studio.create("demo")
    git = GitEngine(workspace.path)
    with pytest.raises(GitError):
        git.blame("../outside.txt")


def test_git_engine_remote_actions_never_run_without_being_asked(studio: CodeStudio) -> None:
    """fetch/pull/push are only ever the explicit calls; the engine's own readers
    (status/log/branches) never touch the network."""
    workspace = studio.create("demo")
    git = GitEngine(workspace.path)
    assert git.remotes() == ()
    assert git.status() == ()


# --- file engine -----------------------------------------------------------


def test_file_engine_tree_lists_nested_files(studio: CodeStudio) -> None:
    workspace = studio.create("demo")
    engine = FileEngine(studio)
    tree = engine.tree(workspace)
    names = {entry.name for entry in tree}
    assert "core" in names
    core = next(entry for entry in tree if entry.name == "core")
    assert core.directory
    assert {child.name for child in core.children} == {"app.py"}


def test_file_engine_tree_skips_ignored_folders(studio: CodeStudio) -> None:
    """The tree never shows a folder the Studio cannot act on (``.git``, caches)."""
    workspace = studio.create("demo")
    (workspace.path / "__pycache__").mkdir(exist_ok=True)
    (workspace.path / "__pycache__" / "junk.pyc").write_text("x", encoding="utf-8")
    (workspace.path / "node_modules").mkdir(exist_ok=True)
    (workspace.path / "node_modules" / "dep.js").write_text("x", encoding="utf-8")
    engine = FileEngine(studio)
    paths = [entry.path for entry in engine.tree(workspace)]
    assert not any("__pycache__" in path for path in paths)
    assert not any("node_modules" in path for path in paths)
    assert not any(path.startswith(".git") for path in paths)


def test_iter_tree_renders_the_tree(studio: CodeStudio) -> None:
    workspace = studio.create("demo")
    engine = FileEngine(studio)
    lines = list(iter_tree(engine.tree(workspace)))
    assert "core/" in lines
    assert any(line.strip() == "app.py" for line in lines)


def test_file_engine_creates_a_file_and_a_folder(studio: CodeStudio) -> None:
    workspace = studio.create("demo")
    engine = FileEngine(studio)
    engine.create_directory(workspace, "core/sub")
    engine.create_file(workspace, "core/sub/new.py", "print(1)\n")
    assert (workspace.path / "core" / "sub" / "new.py").read_text(encoding="utf-8") == "print(1)\n"
    with pytest.raises(StudioError):
        engine.create_file(workspace, "core/sub/new.py", "again")


def test_file_engine_refuses_a_path_that_escapes(studio: CodeStudio) -> None:
    workspace = studio.create("demo")
    engine = FileEngine(studio)
    for bad in ("../escape.py", "/etc/passwd", "core/../../escape.py"):
        with pytest.raises(StudioError):
            engine.create_file(workspace, bad, "")


def test_file_engine_rename_and_copy(studio: CodeStudio) -> None:
    workspace = studio.create("demo")
    engine = FileEngine(studio)
    engine.rename(workspace, "README.md", "README.rst")
    assert not (workspace.path / "README.md").exists()
    assert (workspace.path / "README.rst").is_file()
    engine.copy(workspace, "README.rst", "docs/README.rst")
    assert (workspace.path / "docs" / "README.rst").is_file()


def test_file_engine_delete_file_and_directory(studio: CodeStudio) -> None:
    workspace = studio.create("demo")
    engine = FileEngine(studio)
    engine.create_directory(workspace, "pkg")
    engine.create_file(workspace, "pkg/a.py", "a\n")
    engine.create_file(workspace, "pkg/b.py", "b\n")
    assert engine.delete_directory(workspace, "pkg") == 2
    assert not (workspace.path / "pkg").exists()
    engine.delete(workspace, "README.md")
    assert not (workspace.path / "README.md").exists()


def test_file_engine_replace_across_previews_then_applies(studio: CodeStudio) -> None:
    workspace = studio.create("demo")
    engine = FileEngine(studio)
    preview = engine.replace_across(workspace, "import os", "import sys", apply=False)
    assert not preview.applied
    assert preview.changes and preview.changes[0].path == "core/app.py"
    assert "import os" in (workspace.path / "core" / "app.py").read_text(encoding="utf-8")
    applied = engine.replace_across(workspace, "import os", "import sys", apply=True)
    assert applied.applied
    assert "import sys" in (workspace.path / "core" / "app.py").read_text(encoding="utf-8")


def test_file_engine_replace_across_refuses_too_many_matches(studio: CodeStudio) -> None:
    """A careless pattern is refused rather than rewriting thousands of matches."""
    workspace = studio.create("demo")
    engine = FileEngine(studio)
    (workspace.path / "core" / "app.py").write_text("x " * 100, encoding="utf-8")
    report = engine.replace_across(workspace, "x", "y", apply=True, max_replacements=10)
    assert not report.ok
    assert "Refusing" in report.message
    assert (workspace.path / "core" / "app.py").read_text(encoding="utf-8") == "x " * 100


def test_file_engine_outline_reads_python_symbols(studio: CodeStudio) -> None:
    workspace = studio.create("demo")
    engine = FileEngine(studio)
    report = engine.outline(workspace, "core/app.py")
    assert isinstance(report, OutlineReport)
    kinds = {(symbol.kind, symbol.name) for symbol in report.symbols}
    assert ("class", "Foo") in kinds
    assert ("def", "baz") in kinds
    assert ("def", "Foo.bar") in kinds


def test_file_engine_outline_reports_a_broken_python_file(studio: CodeStudio) -> None:
    workspace = studio.create("demo")
    (workspace.path / "broken.py").write_text("def (:\n", encoding="utf-8")
    engine = FileEngine(studio)
    report = engine.outline(workspace, "broken.py")
    assert report.error
    assert "syntax" in report.error.lower()


def test_file_engine_outline_falls_back_to_a_generic_scan(studio: CodeStudio) -> None:
    workspace = studio.create("demo")
    (workspace.path / "notes.md").write_text("# Title\n## Section\n", encoding="utf-8")
    engine = FileEngine(studio)
    report = engine.outline(workspace, "notes.md")
    assert report.language and report.symbols


def test_file_engine_scan_notes_finds_todo(studio: CodeStudio) -> None:
    workspace = studio.create("demo")
    engine = FileEngine(studio)
    scan = engine.scan_notes(workspace)
    assert isinstance(scan, NoteScan)
    assert scan.counts.get("TODO", 0) >= 1
    assert any("app.py" in note.path for note in scan.notes)


# --- task engine -----------------------------------------------------------


def test_detect_project_reads_the_python_markers(project: Path) -> None:
    profile = detect_project(project)
    assert profile.kind == "python"
    assert "pyproject.toml" in profile.markers


def test_detect_project_of_an_unknown_checkout(tmp_path: Path) -> None:
    empty = tmp_path / "empty"
    empty.mkdir()
    profile = detect_project(empty)
    assert profile.kind == "unknown"


def test_catalog_for_python_offers_the_project_tasks(project: Path) -> None:
    specs = catalog_for(project, check_command=(sys.executable, "-m", "pytest", "-q"))
    keys = {spec.key for spec in specs}
    assert "tests" in keys
    assert "lint" in keys
    assert "types" in keys
    assert "evaluation" in keys
    categories = {spec.category for spec in specs}
    assert CATEGORY_TEST in categories
    assert CATEGORY_QUALITY in categories


def test_catalog_for_node_offers_npm(project: Path) -> None:
    (project / "package.json").write_text('{"name": "demo"}', encoding="utf-8")
    specs = catalog_for(project)
    keys = {spec.key for spec in specs}
    assert "npm-test" in keys or "tests" in keys


def test_custom_spec_splits_a_command_line() -> None:
    spec = custom_spec("python -c 'print(1)'")
    assert spec.key == "custom"
    assert spec.category == CATEGORY_CUSTOM
    assert spec.command_line() == "python -c 'print(1)'"


def test_custom_spec_refuses_an_empty_command() -> None:
    with pytest.raises(TaskError):
        custom_spec("   ")


def test_find_spec_returns_none_for_an_unknown_key(project: Path) -> None:
    assert find_spec(project, "not-a-task") is None


def _spec(*argv: str, key: str = "custom") -> CommandSpec:
    """A minimal :class:`CommandSpec` around an argv, for the runner's own tests."""
    return CommandSpec(
        key=key, title=key, argv=tuple(argv), summary="test", category=CATEGORY_CUSTOM,
    )


def test_command_runner_runs_a_real_command_and_captures_output(tmp_path: Path) -> None:
    runner = CommandRunner()
    lines: list[str] = []
    outcome = runner.run(
        _spec(sys.executable, "-c", "print('hello'); print('world')"),
        cwd=tmp_path,
        on_line=lines.append,
    )
    assert outcome.ok
    assert outcome.returncode == 0
    assert "hello" in outcome.output
    assert "hello" in lines
    assert outcome.seconds >= 0


def test_command_runner_reports_a_failing_command(tmp_path: Path) -> None:
    runner = CommandRunner()
    outcome = runner.run(_spec(sys.executable, "-c", "import sys; sys.exit(3)"), cwd=tmp_path)
    assert not outcome.ok
    assert outcome.returncode == 3
    assert "3" in outcome.summary()


def test_command_runner_can_be_cancelled(tmp_path: Path) -> None:
    import threading
    import time

    runner = CommandRunner()
    script = "import time; print('start', flush=True); time.sleep(30)"
    result: dict[str, CommandOutcome] = {}

    def job() -> None:
        result["outcome"] = runner.run(_spec(sys.executable, "-c", script), cwd=tmp_path, timeout=60)

    thread = threading.Thread(target=job, daemon=True)
    thread.start()
    for _ in range(100):
        if runner.running:
            break
        time.sleep(0.05)
    runner.cancel()
    thread.join(timeout=20)
    assert not thread.is_alive()
    outcome = result["outcome"]
    assert outcome.cancelled or not outcome.ok


def test_command_runner_reports_a_missing_program_honestly(tmp_path: Path) -> None:
    """A missing program is an honest result, not an exception: the caller shows it."""
    runner = CommandRunner()
    outcome = runner.run(_spec("definitely-not-a-program-xyz"), cwd=tmp_path)
    assert not outcome.ok
    assert outcome.returncode == 127
    assert "not installed" in outcome.error or "not on the PATH" in outcome.error


def test_command_runner_never_uses_a_shell(tmp_path: Path) -> None:
    """A command is an argv, so shell metacharacters are inert (never interpreted)."""
    runner = CommandRunner()
    outcome = runner.run(_spec(sys.executable, "-c", "print(';rm -rf /')"), cwd=tmp_path)
    assert outcome.ok
    assert ";rm -rf /" in outcome.output


# --- service wiring --------------------------------------------------------


def test_service_exposes_the_git_status_of_a_workspace(service: CodeStudioService) -> None:
    service.create_workspace("demo")
    workspace = service.workspace("PR1")
    (workspace.path / "new.py").write_text("y = 2\n", encoding="utf-8")
    entries = service.git_status("PR1")
    assert isinstance(entries[0], StatusEntry)
    assert entries[0].path == "new.py"
    assert entries[0].untracked


def test_service_stage_commit_and_history(service: CodeStudioService) -> None:
    service.create_workspace("demo")
    workspace = service.workspace("PR1")
    (workspace.path / "new.py").write_text("y = 2\n", encoding="utf-8")
    assert service.stage("PR1", "new.py").ok
    assert service.commit("PR1", "add new.py").ok
    assert service.git_status("PR1") == ()
    commits = service.git_log("PR1", limit=5)
    assert commits[0].subject == "add new.py"


def test_service_file_tree_and_editing(service: CodeStudioService) -> None:
    service.create_workspace("demo")
    tree = service.file_tree("PR1")
    assert any(entry.name == "core" for entry in tree)
    assert service.create_file("PR1", "core/new.py", "print(1)\n").ok
    assert service.read_file("PR1", "core/new.py").detail == "print(1)\n"
    assert service.rename_file("PR1", "core/new.py", "core/renamed.py").ok
    assert service.copy_file("PR1", "core/renamed.py", "core/copy.py").ok
    assert service.delete_file("PR1", "core/copy.py").ok
    assert service.create_directory("PR1", "pkg").ok
    assert service.delete_file("PR1", "pkg", directory=True).ok


def test_service_replace_across_preview_then_apply(service: CodeStudioService) -> None:
    service.create_workspace("demo")
    preview = service.replace_across("PR1", "import os", "import sys")
    assert not preview.applied
    assert preview.summary_lines()
    applied = service.replace_across("PR1", "import os", "import sys", apply=True)
    assert applied.applied


def test_service_outline_and_notes(service: CodeStudioService) -> None:
    service.create_workspace("demo")
    report = service.outline("PR1", "core/app.py")
    assert any(symbol.name == "baz" for symbol in report.symbols)
    scan = service.scan_notes("PR1")
    assert scan.counts.get("TODO", 0) >= 1


def test_service_export_patch_contains_the_diff(service: CodeStudioService) -> None:
    service.create_workspace("demo")
    workspace = service.workspace("PR1")
    (workspace.path / "core" / "app.py").write_text("changed\n", encoding="utf-8")
    report = service.export_patch("PR1")
    assert report.ok
    assert "core/app.py" in report.detail
    assert report.detail.startswith("diff --git") or "diff --git" in report.detail


def test_service_remote_action_needs_an_explicit_call(service: CodeStudioService) -> None:
    service.create_workspace("demo")
    report = service.remote_action("PR1", "fetch")
    # No remote is configured, so the honest answer is a refusal, never a hang.
    assert not report.ok
    assert report.message


def test_service_run_command_runs_the_configured_check(service: CodeStudioService) -> None:
    service.create_workspace("demo")
    outcome = service.run_command("PR1", "tests")
    assert outcome.ok
    assert "ok" in outcome.output


def test_service_run_command_custom_line(service: CodeStudioService) -> None:
    service.create_workspace("demo")
    outcome = service.run_command(
        "PR1", "custom", command_line=f"{sys.executable} -c \"print('custom')\""
    )
    assert outcome.ok
    assert "custom" in outcome.output


def test_service_run_command_refuses_an_unknown_task(service: CodeStudioService) -> None:
    service.create_workspace("demo")
    with pytest.raises(TaskError):
        service.run_command("PR1", "not-a-task")


def test_service_command_catalog_matches_the_project(service: CodeStudioService) -> None:
    keys = {spec.key for spec in service.command_catalog()}
    assert "tests" in keys
    profile = service.project_profile()
    assert profile.kind == "python"


def test_service_git_conflicts_is_empty_on_a_clean_workspace(service: CodeStudioService) -> None:
    service.create_workspace("demo")
    report = service.git_conflicts("PR1")
    assert not report.in_progress
    assert report.files == ()


def test_service_stash_round_trip(service: CodeStudioService) -> None:
    service.create_workspace("demo")
    workspace = service.workspace("PR1")
    (workspace.path / "core" / "app.py").write_text("changed\n", encoding="utf-8")
    assert service.stash("PR1", "wip", include_untracked=True).ok
    assert service.git_status("PR1") == ()
    assert len(service.git_stash_list("PR1")) == 1
    assert service.stash_pop("PR1").ok
    assert service.git_status("PR1") != ()


def test_service_branches_and_create_branch(service: CodeStudioService) -> None:
    service.create_workspace("demo")
    assert service.create_branch("PR1", "feature/x").ok
    names = {branch.name for branch in service.git_branches("PR1")}
    assert "feature/x" in names


def test_service_blame_and_show(service: CodeStudioService) -> None:
    service.create_workspace("demo")
    blame = service.git_blame("PR1", "README.md")
    assert blame.ok
    show = service.git_show("PR1")
    assert show.ok
    assert show.detail


def test_service_state_reports_the_project_kind(service: CodeStudioService) -> None:
    state = service.state()
    assert state.repository
    assert state.project is not None
    assert state.project.kind == "python"
    assert state.current_branch == "main"


def test_service_can_merge_after_a_passing_check(service: CodeStudioService) -> None:
    service.create_workspace("demo")
    workspace = service.workspace("PR1")
    (workspace.path / "core" / "app.py").write_text("changed\n", encoding="utf-8")
    report, _check = service.run_checks("PR1")
    assert report.ok
    gate = service.can_merge("PR1")
    assert gate.ok
    assert service.merge("PR1").ok
    assert (service.root / "core" / "app.py").read_text(encoding="utf-8") == "changed\n"


def test_file_engine_summary_of_a_file(studio: CodeStudio) -> None:
    workspace = studio.create("demo")
    engine = FileEngine(studio)
    summary = engine.file_summary(workspace, "core/app.py")
    assert int(summary["lines"]) > 1
    assert int(summary["non_blank"]) > 1
    assert int(summary["bytes"]) > 0


def test_tree_entry_file_count_counts_a_subtree(studio: CodeStudio) -> None:
    workspace = studio.create("demo")
    engine = FileEngine(studio)
    tree = engine.tree(workspace)
    core = next(entry for entry in tree if entry.name == "core")
    assert isinstance(core, TreeEntry)
    assert core.file_count >= 1


def test_build_category_exists_in_a_node_project(project: Path) -> None:
    (project / "package.json").write_text('{"scripts": {"build": "true"}}', encoding="utf-8")
    specs = catalog_for(project)
    assert any(spec.category == CATEGORY_BUILD for spec in specs)
