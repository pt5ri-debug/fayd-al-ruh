"""Tests for the Code Studio: isolated workspaces, the coding agent, and merge.

The Studio edits a project's *own* code, so the tests take it seriously: they
build a real, throwaway git repository, make a real worktree, run a real diff and
a real merge, and check that the safety gates hold. The coding model is always an
injected fake, so no Ollama server and no network are needed.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from core.code_agent import AgentError, AgentStep, CodingAgent, parse_action
from core.code_studio import CodeStudio, StudioError
from core.coder_model import (
    CoderMessage,
    CoderUnavailable,
    OllamaCoderClient,
    coder_status,
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
            **__import__("os").environ,
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
    (repo / "core" / "app.py").write_text("x = 1\n", encoding="utf-8")
    (repo / "README.md").write_text("# demo\n", encoding="utf-8")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-qm", "initial")
    return repo


@pytest.fixture()
def studio(project: Path, tmp_path: Path) -> CodeStudio:
    return CodeStudio(project, workspaces_dir=tmp_path / "workspaces", base_branch="main")


class FakeCoder:
    """A scripted coding model: replies with the next canned message."""

    def __init__(self, script: list[str], *, models: tuple[str, ...] = ("qwen2.5-coder:7b",)) -> None:
        self.model = "qwen2.5-coder:7b"
        self.script = list(script)
        self.models = models
        self.calls: list[list[CoderMessage]] = []

    def list_models(self) -> tuple[str, ...]:
        return self.models

    def chat(self, messages, *, temperature: float = 0.2) -> str:
        self.calls.append(list(messages))
        if not self.script:
            return '{"action": "done", "summary": "no more scripted replies"}'
        return self.script.pop(0)


class BrokenCoder(FakeCoder):
    """A model whose server cannot be reached."""

    def list_models(self) -> tuple[str, ...]:
        raise CoderUnavailable("connection refused")

    def chat(self, messages, *, temperature: float = 0.2) -> str:
        raise CoderUnavailable("connection refused")


# --- workspace lifecycle ---------------------------------------------------


def test_default_workspaces_dir_belongs_to_the_checkout(project: Path, tmp_path: Path) -> None:
    """A checkout must never share another checkout's worktree folder.

    A shared folder registered a foreign repository's worktree in this project's
    state file, so ``--root /elsewhere`` listed a workspace whose files lived in
    the other checkout. The project's own checkout keeps its worktrees in the
    bundled data directory; any other checkout keeps them under its own root.
    """
    from core.paths import studio_workspaces_dir

    other = CodeStudio(tmp_path / "elsewhere")
    assert other.workspaces_dir == tmp_path / "elsewhere" / "data" / "studio" / "workspaces"
    assert other.workspaces_dir != studio_workspaces_dir()


def test_create_makes_a_worktree_on_its_own_branch(studio: CodeStudio) -> None:
    workspace = studio.create("Add a --json flag")
    assert workspace.slug == "add-a-json-flag"
    assert workspace.branch == "studio/add-a-json-flag"
    assert workspace.path.is_dir()
    assert (workspace.path / "core" / "app.py").is_file()
    assert studio.base_revision() == workspace.base_revision


def test_create_refuses_a_duplicate_workspace(studio: CodeStudio) -> None:
    studio.create("First change")
    with pytest.raises(StudioError, match="already exists"):
        studio.create("First change")


def test_list_and_get_round_trip_through_the_state_file(studio: CodeStudio) -> None:
    created = studio.create("A change")
    listed = studio.list_workspaces()
    assert [item.slug for item in listed] == [created.slug]
    fetched = studio.get(created.slug)
    assert fetched.branch == created.branch
    assert fetched.task == "A change"


def test_remove_deletes_the_worktree_and_the_branch(studio: CodeStudio) -> None:
    workspace = studio.create("Throwaway")
    studio.remove(workspace.slug)
    assert studio.list_workspaces() == ()
    assert not workspace.path.exists()
    branches = _git(studio.root, "branch", "--list", "studio/*")
    assert branches.strip() == ""


def test_get_unknown_slug_raises(studio: CodeStudio) -> None:
    with pytest.raises(StudioError, match="No workspace named"):
        studio.get("nope")


# --- confinement -----------------------------------------------------------


@pytest.mark.parametrize("bad", ["../escape.py", "core/../../escape.py", "/etc/passwd"])
def test_paths_cannot_escape_the_workspace(studio: CodeStudio, bad: str) -> None:
    workspace = studio.create("Confinement")
    with pytest.raises(StudioError, match="escapes the workspace"):
        studio.write_file(workspace, bad, "nope")


def test_non_text_files_are_refused(studio: CodeStudio) -> None:
    workspace = studio.create("Suffixes")
    with pytest.raises(StudioError, match="not a text file"):
        studio.write_file(workspace, "payload.bin", "nope")


def test_empty_path_is_refused(studio: CodeStudio) -> None:
    workspace = studio.create("Empty path")
    with pytest.raises(StudioError, match="Path cannot be empty"):
        studio.read_file(workspace, "   ")


def test_reading_a_missing_file_is_an_error(studio: CodeStudio) -> None:
    workspace = studio.create("Missing file")
    with pytest.raises(StudioError, match="Not a file"):
        studio.read_file(workspace, "core/absent.py")


def test_oversized_content_is_refused(project: Path, tmp_path: Path) -> None:
    studio = CodeStudio(project, workspaces_dir=tmp_path / "ws", max_file_bytes=64)
    workspace = studio.create("Too big")
    with pytest.raises(StudioError, match="larger than the Studio's size limit"):
        studio.write_file(workspace, "core/app.py", "x" * 1000)


def test_write_and_read_round_trip(studio: CodeStudio) -> None:
    workspace = studio.create("Round trip")
    written = studio.write_file(workspace, "core/new.py", "value = 3\n")
    assert written == len("value = 3\n")
    assert studio.read_file(workspace, "core/new.py") == "value = 3\n"


def test_list_files_skips_run_state_and_binaries(studio: CodeStudio) -> None:
    workspace = studio.create("Listing")
    studio.write_file(workspace, "notes.md", "hi\n")
    (workspace.path / "blob.bin").write_bytes(b"\x00\x01")
    (workspace.path / "__pycache__").mkdir()
    (workspace.path / "__pycache__" / "cached.py").write_text("x = 1\n", encoding="utf-8")
    listed = {entry.path for entry in studio.list_files(workspace)}
    assert "core/app.py" in listed
    assert "notes.md" in listed
    assert "blob.bin" not in listed
    assert "__pycache__/cached.py" not in listed


# --- surgical edits and search ---------------------------------------------


def test_edit_file_replaces_one_snippet_and_keeps_the_rest(studio: CodeStudio) -> None:
    workspace = studio.create("Edit")
    studio.write_file(workspace, "core/app.py", "a = 1\nb = 2\nc = 3\n")
    made = studio.edit_file(workspace, "core/app.py", "b = 2", "b = 20")
    assert made == 1
    assert studio.read_file(workspace, "core/app.py") == "a = 1\nb = 20\nc = 3\n"


def test_edit_file_refuses_an_ambiguous_match_and_changes_nothing(studio: CodeStudio) -> None:
    workspace = studio.create("Ambiguous")
    original = "x = 1\ny = 2\nx = 1\n"
    studio.write_file(workspace, "core/app.py", original)
    with pytest.raises(StudioError, match="2 times"):
        studio.edit_file(workspace, "core/app.py", "x = 1", "x = 9")
    assert studio.read_file(workspace, "core/app.py") == original


def test_edit_file_replace_all_changes_every_occurrence(studio: CodeStudio) -> None:
    workspace = studio.create("Replace all")
    studio.write_file(workspace, "core/app.py", "x = 1\ny = 2\nx = 1\n")
    made = studio.edit_file(workspace, "core/app.py", "x = 1", "x = 9", replace_all=True)
    assert made == 2
    assert studio.read_file(workspace, "core/app.py") == "x = 9\ny = 2\nx = 9\n"


def test_edit_file_reports_a_missing_snippet(studio: CodeStudio) -> None:
    workspace = studio.create("Missing")
    studio.write_file(workspace, "core/app.py", "a = 1\n")
    with pytest.raises(StudioError, match="was not found"):
        studio.edit_file(workspace, "core/app.py", "zzz", "b")


def test_edit_file_refuses_an_empty_old(studio: CodeStudio) -> None:
    workspace = studio.create("Empty old")
    with pytest.raises(StudioError, match="cannot be empty"):
        studio.edit_file(workspace, "core/app.py", "", "b")


def test_edit_file_cannot_escape_the_workspace(studio: CodeStudio) -> None:
    workspace = studio.create("Escape")
    with pytest.raises(StudioError, match="escapes"):
        studio.edit_file(workspace, "../outside.py", "a", "b")


def test_search_finds_lines_with_context(studio: CodeStudio) -> None:
    workspace = studio.create("Search")
    studio.write_file(workspace, "core/app.py", "alpha\nbeta target\ngamma\n")
    hits = studio.search(workspace, "target", context_lines=1)
    assert len(hits) == 1
    assert hits[0].path == "core/app.py"
    assert hits[0].line == 2
    assert "beta target" in hits[0].text
    assert hits[0].before == "alpha"
    assert hits[0].after == "gamma"
    assert "core/app.py:2:" in hits[0].render()


def test_search_is_case_insensitive_and_literal(studio: CodeStudio) -> None:
    workspace = studio.create("Literal")
    studio.write_file(workspace, "core/app.py", "def main():\n    pass\n")
    assert len(studio.search(workspace, "DEF MAIN")) == 1
    assert studio.search(workspace, "def.*(") == ()  # not a regex


def test_search_can_narrow_to_one_file(studio: CodeStudio) -> None:
    workspace = studio.create("Narrow")
    studio.write_file(workspace, "core/app.py", "needle\n")
    studio.write_file(workspace, "core/other.py", "needle\n")
    hits = studio.search(workspace, "needle", relative="core/app.py")
    assert [hit.path for hit in hits] == ["core/app.py"]


def test_search_refuses_an_empty_query_and_an_escape(studio: CodeStudio) -> None:
    workspace = studio.create("Bad search")
    with pytest.raises(StudioError, match="cannot be empty"):
        studio.search(workspace, "   ")
    with pytest.raises(StudioError, match="escapes"):
        studio.search(workspace, "x", relative="../")


def test_search_skips_binaries_and_run_state(studio: CodeStudio) -> None:
    workspace = studio.create("Skip")
    studio.write_file(workspace, "notes.md", "needle\n")
    (workspace.path / "blob.bin").write_bytes(b"needle")
    (workspace.path / "__pycache__").mkdir()
    (workspace.path / "__pycache__" / "c.py").write_text("needle\n", encoding="utf-8")
    hits = studio.search(workspace, "needle")
    assert [hit.path for hit in hits] == ["notes.md"]


# --- change inspection -----------------------------------------------------


def test_changed_files_includes_untracked_and_modified(studio: CodeStudio) -> None:
    workspace = studio.create("Changes")
    studio.write_file(workspace, "core/app.py", "x = 2\n")
    studio.write_file(workspace, "core/added.py", "y = 1\n")
    changed = set(studio.changed_files(workspace))
    assert changed == {"core/app.py", "core/added.py"}


def test_diff_shows_the_change_and_new_files(studio: CodeStudio) -> None:
    workspace = studio.create("Diff")
    studio.write_file(workspace, "core/app.py", "x = 2\n")
    studio.write_file(workspace, "core/added.py", "y = 1\n")
    diff = studio.diff(workspace)
    assert "-x = 1" in diff
    assert "+x = 2" in diff
    assert "core/added.py" in diff
    assert "new file" in diff


def test_diff_on_a_clean_workspace_says_so(studio: CodeStudio) -> None:
    workspace = studio.create("Clean")
    assert studio.diff(workspace) == "(no changes yet)"


# --- checks gate -----------------------------------------------------------


def test_checks_pass_and_are_recorded(studio: CodeStudio) -> None:
    workspace = studio.create("Checks")
    result = studio.run_checks(workspace, ["python", "-c", "print('fine')"])
    assert result.ok
    assert "passed" in result.summary()
    assert studio.get(workspace.slug).checks_passed is True


def test_failing_checks_block_the_merge(studio: CodeStudio) -> None:
    workspace = studio.create("Failing")
    studio.write_file(workspace, "core/app.py", "x = 2\n")
    studio.run_checks(workspace, ["python", "-c", "raise SystemExit(1)"])
    allowed, reason = studio.can_merge(workspace)
    assert not allowed
    assert "checks" in reason


def test_editing_after_a_pass_invalidates_the_check(studio: CodeStudio) -> None:
    """A pass describes one revision; changing a file must not keep it valid."""
    workspace = studio.create("Stale pass")
    studio.write_file(workspace, "core/app.py", "x = 2\n")
    studio.run_checks(workspace, ["python", "-c", "print('fine')"])
    assert studio.get(workspace.slug).checks_passed is True

    studio.write_file(workspace, "core/app.py", "x = 3\n")
    assert studio.get(workspace.slug).checks_passed is None
    allowed, reason = studio.can_merge(workspace)
    assert not allowed
    assert "checks" in reason


def test_edit_file_after_a_pass_invalidates_the_check(studio: CodeStudio) -> None:
    workspace = studio.create("Stale pass via edit")
    studio.write_file(workspace, "core/app.py", "x = 2\n")
    studio.run_checks(workspace, ["python", "-c", "print('fine')"])
    studio.edit_file(workspace, "core/app.py", "x = 2", "x = 3")
    assert studio.get(workspace.slug).checks_passed is None


def test_merge_is_refused_before_checks_run(studio: CodeStudio) -> None:
    workspace = studio.create("No checks")
    studio.write_file(workspace, "core/app.py", "x = 2\n")
    allowed, reason = studio.can_merge(workspace)
    assert not allowed
    assert "checks" in reason


def test_merge_is_refused_when_the_workspace_is_empty(studio: CodeStudio) -> None:
    workspace = studio.create("Empty")
    allowed, reason = studio.can_merge(workspace)
    assert not allowed
    assert "no changes" in reason


def test_a_missing_check_command_is_reported_not_raised(studio: CodeStudio) -> None:
    workspace = studio.create("Missing tool")
    result = studio.run_checks(workspace, ["definitely-not-a-real-binary-xyz"])
    assert not result.ok
    assert "Could not run checks" in result.output


# --- merge -----------------------------------------------------------------


def test_merge_fast_forwards_the_base_branch(studio: CodeStudio) -> None:
    workspace = studio.create("Merge me")
    studio.write_file(workspace, "core/app.py", "x = 42\n")
    studio.run_checks(workspace, ["python", "-c", "pass"])
    result = studio.merge(workspace)
    assert result.ok, result.message
    assert result.new_revision
    assert (studio.root / "core" / "app.py").read_text(encoding="utf-8") == "x = 42\n"


def test_merge_is_refused_when_the_base_branch_moved_on(studio: CodeStudio) -> None:
    workspace = studio.create("Stale")
    studio.write_file(workspace, "core/app.py", "x = 2\n")
    studio.run_checks(workspace, ["python", "-c", "pass"])
    # Someone else lands a commit on main while the workspace is open.
    (studio.root / "other.txt").write_text("moved\n", encoding="utf-8")
    _git(studio.root, "add", "-A")
    _git(studio.root, "commit", "-qm", "another change")
    allowed, reason = studio.can_merge(workspace)
    assert not allowed
    assert "has moved to" in reason


def test_merge_is_refused_when_the_checkout_is_on_another_branch(studio: CodeStudio) -> None:
    workspace = studio.create("Wrong branch")
    studio.write_file(workspace, "core/app.py", "x = 2\n")
    studio.run_checks(workspace, ["python", "-c", "pass"])
    _git(studio.root, "checkout", "-q", "-b", "elsewhere")
    result = studio.merge(workspace)
    assert not result.ok
    assert "not `main`" in result.message


def test_merge_is_refused_with_uncommitted_changes(studio: CodeStudio) -> None:
    workspace = studio.create("Dirty tree")
    studio.write_file(workspace, "core/app.py", "x = 2\n")
    studio.run_checks(workspace, ["python", "-c", "pass"])
    (studio.root / "core" / "app.py").write_text("x = 99\n", encoding="utf-8")
    result = studio.merge(workspace)
    assert not result.ok
    assert "uncommitted changes" in result.message


def test_a_non_repository_is_reported_clearly(tmp_path: Path) -> None:
    plain = tmp_path / "plain"
    plain.mkdir()
    studio = CodeStudio(plain, workspaces_dir=tmp_path / "ws")
    assert not studio.is_repository()
    with pytest.raises(StudioError, match="not a git checkout"):
        studio.create("Anything")


# --- the agent -------------------------------------------------------------


def test_parse_action_accepts_the_four_actions() -> None:
    assert parse_action('{"action": "list_files"}').name == "list_files"
    read = parse_action('{"action": "read_file", "path": "a.py"}')
    assert read.name == "read_file" and read.path == "a.py"
    write = parse_action('{"action": "write_file", "path": "a.py", "content": "x = 1\\n"}')
    assert write.name == "write_file" and write.content == "x = 1\n"
    done = parse_action('{"action": "done", "summary": "did it"}')
    assert done.name == "done" and done.summary == "did it"


def test_parse_action_accepts_the_new_actions() -> None:
    search = parse_action('{"action": "search", "query": "def main"}')
    assert search.name == "search" and search.query == "def main"
    found = parse_action('{"action": "find_files", "pattern": "**/*.py"}')
    assert found.name == "find_files" and found.pattern == "**/*.py"
    edit = parse_action(
        '{"action": "edit_file", "path": "a.py", "old": "x = 1", "new": "x = 2"}'
    )
    assert edit.name == "edit_file" and edit.old == "x = 1" and edit.new == "x = 2"
    assert edit.replace_all is False
    every = parse_action(
        '{"action": "edit_file", "path": "a.py", "old": "a", "new": "b", "replace_all": true}'
    )
    assert every.replace_all is True
    assert parse_action('{"action": "run_checks"}').name == "run_checks"
    assert parse_action('{"action": "diff"}').name == "diff"


@pytest.mark.parametrize(
    "reply",
    [
        '{"action": "search"}',  # no query
        '{"action": "find_files"}',  # no pattern
        '{"action": "edit_file", "path": "a.py", "new": "b"}',  # no old
        '{"action": "edit_file", "path": "a.py", "old": "", "new": "b"}',  # empty old
        '{"action": "edit_file", "old": "a", "new": "b"}',  # no path
    ],
)
def test_parse_action_rejects_incomplete_new_actions(reply: str) -> None:
    with pytest.raises(AgentError):
        parse_action(reply)


def test_parse_action_tolerates_a_fenced_reply() -> None:
    reply = '```json\n{"action": "done", "summary": "ok"}\n```'
    assert parse_action(reply).name == "done"


def test_parse_action_tolerates_surrounding_prose() -> None:
    reply = 'Sure. {"action": "list_files"} That is my step.'
    assert parse_action(reply).name == "list_files"


@pytest.mark.parametrize(
    "reply",
    [
        "",
        "no json at all",
        '{"action": "delete_everything"}',
        '{"action": "read_file"}',
        '{"action": "write_file", "path": "a.py"}',
        '{"action": "write_file", "path": "a.py", "content": 5}',
        "[1, 2, 3]",
    ],
)
def test_parse_action_rejects_bad_replies(reply: str) -> None:
    with pytest.raises(AgentError):
        parse_action(reply)


def test_agent_reads_then_writes_then_finishes(studio: CodeStudio) -> None:
    workspace = studio.create("Agent change")
    coder = FakeCoder(
        [
            '{"action": "read_file", "path": "core/app.py"}',
            '{"action": "write_file", "path": "core/app.py", "content": "x = 7\\n"}',
            '{"action": "done", "summary": "set x to 7"}',
        ]
    )
    trace = CodingAgent(studio, coder).run(workspace, "set x to 7")
    assert trace.ok
    assert trace.summary == "set x to 7"
    assert trace.writes == ["core/app.py"]
    assert studio.read_file(workspace, "core/app.py") == "x = 7\n"
    # The model was told what it read, so it could act on it.
    assert any("x = 1" in message.content for message in coder.calls[1])


def test_agent_reports_each_step_live(studio: CodeStudio) -> None:
    """``on_step`` fires as the work happens, in order, and matches the trace."""
    workspace = studio.create("Live steps")
    coder = FakeCoder(
        [
            '{"action": "read_file", "path": "core/app.py"}',
            '{"action": "write_file", "path": "core/app.py", "content": "x = 9\\n"}',
            '{"action": "done", "summary": "set x to 9"}',
        ]
    )
    live: list[AgentStep] = []
    trace = CodingAgent(studio, coder).run(workspace, "set x to 9", on_step=live.append)
    # Every step reached the callback, in the same order as the trace.
    assert [step.action for step in live] == ["read_file", "write_file", "done"]
    assert live == trace.steps


def test_agent_step_live_lines_show_action_and_observation() -> None:
    step = AgentStep(1, "write_file", detail="core/app.py", result="Wrote 6 bytes to core/app.py.")
    lines = step.live_lines()
    assert lines[0] == "→ write_file core/app.py"
    assert "Wrote 6 bytes" in lines[1]


def test_agent_step_live_lines_mark_a_failure_and_trim_a_long_read() -> None:
    failed = AgentStep(2, "read_file", detail="missing.py", result="ERROR: not found", ok=False)
    assert failed.live_lines()[0].startswith("✗ read_file missing.py")
    big = AgentStep(3, "read_file", detail="big.py", result="\n".join(str(i) for i in range(50)))
    lines = big.live_lines()
    # The head plus the cap, then one honest "more lines" note.
    assert len(lines) == 1 + 12 + 1
    assert lines[-1] == "   … [38 more line(s)]"


def test_agent_step_live_lines_render_done_and_invalid() -> None:
    assert AgentStep(4, "done", detail="all good").live_lines() == ["✓ Done: all good"]
    invalid = AgentStep(5, "invalid", detail="not JSON", ok=False)
    assert "not JSON" in invalid.live_lines()[0]


def test_agent_reports_a_bad_reply_and_keeps_going(studio: CodeStudio) -> None:
    workspace = studio.create("Bad reply")
    coder = FakeCoder(
        [
            "I will now edit the file.",  # not JSON: the loop should push back
            '{"action": "write_file", "path": "core/app.py", "content": "x = 8\\n"}',
            '{"action": "done", "summary": "recovered"}',
        ]
    )
    trace = CodingAgent(studio, coder).run(workspace, "set x to 8")
    assert trace.ok
    assert any(step.action == "invalid" for step in trace.steps)
    assert studio.read_file(workspace, "core/app.py") == "x = 8\n"


def test_agent_can_search_then_edit_one_snippet(studio: CodeStudio) -> None:
    """The OpenHands-style loop: locate, then change one exact snippet."""
    workspace = studio.create("Surgical")
    studio.write_file(workspace, "core/app.py", "a = 1\nb = 2\nc = 3\n")
    coder = FakeCoder(
        [
            '{"action": "search", "query": "b = 2"}',
            '{"action": "edit_file", "path": "core/app.py", "old": "b = 2", "new": "b = 20"}',
            '{"action": "done", "summary": "changed b"}',
        ]
    )
    trace = CodingAgent(studio, coder).run(workspace, "change b")
    assert trace.ok
    assert trace.writes == ["core/app.py"]
    # The search result reached the model, so it could edit the right place.
    assert any("core/app.py:2:" in message.content for message in coder.calls[2])
    # And only that one line changed.
    assert studio.read_file(workspace, "core/app.py") == "a = 1\nb = 20\nc = 3\n"


def test_agent_can_find_files(studio: CodeStudio) -> None:
    workspace = studio.create("Find")
    coder = FakeCoder(
        [
            '{"action": "find_files", "pattern": "**/*.py"}',
            '{"action": "done", "summary": "listed"}',
        ]
    )
    trace = CodingAgent(studio, coder).run(workspace, "list python files")
    assert trace.ok
    assert any("core/app.py" in step.result for step in trace.steps)


def test_agent_find_files_refuses_an_escape(studio: CodeStudio) -> None:
    workspace = studio.create("Find escape")
    coder = FakeCoder(
        [
            '{"action": "find_files", "pattern": "../*.py"}',
            '{"action": "done", "summary": "tried"}',
        ]
    )
    trace = CodingAgent(studio, coder).run(workspace, "escape")
    assert any(step.result.startswith("ERROR:") for step in trace.steps)


def test_agent_surfaces_an_edit_error_and_can_retry(studio: CodeStudio) -> None:
    workspace = studio.create("Retry")
    studio.write_file(workspace, "core/app.py", "a = 1\n")
    coder = FakeCoder(
        [
            '{"action": "edit_file", "path": "core/app.py", "old": "NOPE", "new": "x"}',
            '{"action": "edit_file", "path": "core/app.py", "old": "a = 1", "new": "a = 2"}',
            '{"action": "done", "summary": "fixed"}',
        ]
    )
    trace = CodingAgent(studio, coder).run(workspace, "fix a")
    assert any(step.result.startswith("ERROR:") for step in trace.steps)
    assert studio.read_file(workspace, "core/app.py") == "a = 2\n"


def test_agent_runs_checks_and_records_the_outcome(studio: CodeStudio) -> None:
    workspace = studio.create("Checks")
    coder = FakeCoder(
        [
            '{"action": "run_checks"}',
            '{"action": "done", "summary": "checked"}',
        ]
    )
    agent = CodingAgent(studio, coder, check_command=("python", "-c", "print('ok')"))
    trace = agent.run(workspace, "run the checks")
    assert trace.checked is True
    assert trace.checks_ok is True
    assert "Checks: passed" in trace.render()


def test_agent_can_be_denied_the_checks(studio: CodeStudio) -> None:
    workspace = studio.create("No checks")
    coder = FakeCoder(
        [
            '{"action": "run_checks"}',
            '{"action": "done", "summary": "done"}',
        ]
    )
    agent = CodingAgent(studio, coder, allow_checks=False)
    trace = agent.run(workspace, "run the checks")
    assert trace.checked is False
    assert any(step.result.startswith("ERROR:") for step in trace.steps)


def test_agent_edit_file_does_not_duplicate_a_write(studio: CodeStudio) -> None:
    workspace = studio.create("One write")
    studio.write_file(workspace, "core/app.py", "a = 1\n")
    coder = FakeCoder(
        [
            '{"action": "edit_file", "path": "core/app.py", "old": "a = 1", "new": "a = 2"}',
            '{"action": "edit_file", "path": "core/app.py", "old": "a = 2", "new": "a = 3"}',
            '{"action": "done", "summary": "twice"}',
        ]
    )
    trace = CodingAgent(studio, coder).run(workspace, "twice")
    assert trace.writes == ["core/app.py"]


def test_agent_can_review_its_own_diff(studio: CodeStudio) -> None:
    """The OpenHands habit: look at the change before calling it done."""
    workspace = studio.create("Review")
    studio.write_file(workspace, "core/app.py", "a = 1\n")
    coder = FakeCoder(
        [
            '{"action": "edit_file", "path": "core/app.py", "old": "a = 1", "new": "a = 2"}',
            '{"action": "diff"}',
            '{"action": "done", "summary": "reviewed"}',
        ]
    )
    trace = CodingAgent(studio, coder).run(workspace, "change a")
    assert trace.ok
    diff_step = next(step for step in trace.steps if step.action == "diff")
    assert "a = 2" in diff_step.result


def test_agent_records_a_failing_check_honestly(studio: CodeStudio) -> None:
    """A failing test run is a normal observation, not an ``ERROR:`` line."""
    workspace = studio.create("Failing checks")
    coder = FakeCoder(
        [
            '{"action": "run_checks"}',
            '{"action": "done", "summary": "done anyway"}',
        ]
    )
    agent = CodingAgent(studio, coder, check_command=("python", "-c", "raise SystemExit(1)"))
    trace = agent.run(workspace, "run the checks")
    assert trace.checked is True
    assert trace.checks_ok is False
    assert "Checks: failed" in trace.render()
    assert not any(step.result.startswith("ERROR:") for step in trace.steps)


def test_agent_refuses_a_write_outside_the_workspace(studio: CodeStudio) -> None:
    workspace = studio.create("Escape attempt")
    coder = FakeCoder(
        [
            '{"action": "write_file", "path": "../outside.py", "content": "bad\\n"}',
            '{"action": "done", "summary": "tried"}',
        ]
    )
    trace = CodingAgent(studio, coder).run(workspace, "write outside")
    failed = [step for step in trace.steps if not step.ok]
    assert failed and "escapes the workspace" in failed[0].result
    assert not (studio.root.parent / "outside.py").exists()


def test_agent_stops_at_the_step_limit(studio: CodeStudio) -> None:
    workspace = studio.create("Loop forever")
    coder = FakeCoder(['{"action": "list_files"}'] * 20)
    trace = CodingAgent(studio, coder, max_steps=3).run(workspace, "spin")
    assert not trace.ok
    assert "step limit" in trace.stopped_reason
    assert len(trace.steps) == 3


def test_agent_reports_an_unreachable_model(studio: CodeStudio) -> None:
    workspace = studio.create("No model")
    trace = CodingAgent(studio, BrokenCoder([])).run(workspace, "anything")
    assert not trace.ok
    assert "could not answer" in trace.stopped_reason


def test_agent_rejects_an_empty_task(studio: CodeStudio) -> None:
    workspace = studio.create("Empty task")
    with pytest.raises(AgentError, match="cannot be empty"):
        CodingAgent(studio, FakeCoder([])).run(workspace, "   ")


def test_agent_truncates_a_large_read(studio: CodeStudio) -> None:
    workspace = studio.create("Big file")
    studio.write_file(workspace, "core/big.py", "line = 1\n" * 500)
    coder = FakeCoder(
        [
            '{"action": "read_file", "path": "core/big.py"}',
            '{"action": "done", "summary": "read it"}',
        ]
    )
    trace = CodingAgent(studio, coder, read_chars=200).run(workspace, "read the big file")
    assert any("[file truncated]" in step.result for step in trace.steps)


# --- the coder client ------------------------------------------------------


def test_model_present_accepts_the_implicit_latest_tag() -> None:
    from core.coder_model import _model_present

    assert _model_present("qwen2.5-coder:7b", ("qwen2.5-coder:7b",))
    assert _model_present("llama3", ("llama3:latest",))
    assert not _model_present("qwen2.5-coder:7b", ("qwen2.5-coder:1.5b",))


def test_coder_status_reports_a_missing_server() -> None:
    status = coder_status(BrokenCoder([]))
    assert not status.available
    assert not status.server_reachable
    assert "not running" in status.summary()


def test_coder_status_reports_a_missing_model() -> None:
    client = FakeCoder([], models=("llama3:latest",))
    status = coder_status(client)
    assert status.server_reachable
    assert not status.installed
    assert "not pulled" in status.summary()


def test_coder_status_is_available_when_the_model_is_pulled() -> None:
    status = coder_status(FakeCoder([]))
    assert status.available
    assert "ready" in status.summary()


def test_message_rejects_a_bad_role() -> None:
    with pytest.raises(ValueError, match="Unsupported role"):
        CoderMessage("robot", "hello")


def test_message_rejects_empty_content() -> None:
    with pytest.raises(ValueError, match="cannot be empty"):
        CoderMessage("user", "   ")


def test_ollama_client_reports_an_unreachable_server() -> None:
    client = OllamaCoderClient(base_url="http://127.0.0.1:1", timeout=0.5)
    with pytest.raises(CoderUnavailable, match="Could not reach Ollama"):
        client.list_models()


def test_ollama_client_needs_a_model_name() -> None:
    with pytest.raises(ValueError, match="model name is required"):
        OllamaCoderClient("   ")


# --- the service -----------------------------------------------------------


def test_service_status_is_honest_without_a_model(project: Path, tmp_path: Path) -> None:
    service = CodeStudioService(
        project,
        studio=CodeStudio(project, workspaces_dir=tmp_path / "ws"),
        client=BrokenCoder([]),
    )
    assert not service.status().available
    assert service.state().repository


def test_service_status_offers_the_in_process_model_first(
    project: Path, tmp_path: Path, monkeypatch
) -> None:
    """A machine with no Ollama is told about the in-process model, not a server.

    The in-process model is the preferred runtime and needs no daemon, so the
    status must not present "start `ollama serve`" as the only way forward.
    """
    monkeypatch.setenv("FAYD_DATA_DIR", str(tmp_path / "data"))
    service = CodeStudioService(project, studio=CodeStudio(project, workspaces_dir=tmp_path / "ws"))
    summary = service.status().summary()
    assert service.status().backend == "in-process"
    assert "requirements-coder.txt" in summary
    assert not summary.startswith("Ollama is not running")


def test_service_status_reports_ollama_when_it_is_the_runtime(
    project: Path, tmp_path: Path, monkeypatch
) -> None:
    """An Ollama-only machine still gets the Ollama wording it needs."""
    from core.coder_llama import BackendStatus

    monkeypatch.setenv("FAYD_DATA_DIR", str(tmp_path / "data"))
    service = CodeStudioService(project, studio=CodeStudio(project, workspaces_dir=tmp_path / "ws"))
    monkeypatch.setattr(
        service,
        "backend",
        lambda: BackendStatus("ollama", True, detail="Coding model ready (ollama)."),
    )
    status = service.status()
    assert status.available
    assert status.server_reachable
    assert status.backend == "ollama"
    assert "ready" in status.summary()


def test_service_full_flow_create_ask_check_merge(project: Path, tmp_path: Path) -> None:
    coder = FakeCoder(
        [
            '{"action": "write_file", "path": "core/app.py", "content": "x = 5\\n"}',
            '{"action": "done", "summary": "set x to 5"}',
        ]
    )
    service = CodeStudioService(
        project,
        studio=CodeStudio(project, workspaces_dir=tmp_path / "ws"),
        client=coder,
        check_command=("python", "-c", "pass"),
    )
    created = service.create_workspace("Set x to 5")
    assert created.ok
    slug = service.state().workspaces[0].slug

    report, trace = service.run_agent(slug, "set x to 5")
    assert report.ok and trace is not None
    assert "core/app.py" in service.changed_files(slug)

    assert not service.can_merge(slug).ok
    checks = service.run_checks(slug)[0]
    assert checks.ok
    assert service.can_merge(slug).ok
    merged = service.merge(slug)
    assert merged.ok
    assert (project / "core" / "app.py").read_text(encoding="utf-8") == "x = 5\n"


def test_service_reports_a_missing_model_instead_of_failing(project: Path, tmp_path: Path) -> None:
    service = CodeStudioService(
        project,
        studio=CodeStudio(project, workspaces_dir=tmp_path / "ws"),
        client=BrokenCoder([]),
    )
    service.create_workspace("A change")
    slug = service.state().workspaces[0].slug
    report, trace = service.run_agent(slug, "do something")
    assert not report.ok
    assert trace is None
    assert "not running" in report.message


def test_service_editor_round_trip(project: Path, tmp_path: Path) -> None:
    service = CodeStudioService(
        project,
        studio=CodeStudio(project, workspaces_dir=tmp_path / "ws"),
        client=FakeCoder([]),
    )
    service.create_workspace("Editor")
    slug = service.state().workspaces[0].slug
    assert service.read_file(slug, "core/app.py").detail == "x = 1\n"
    assert service.save_file(slug, "core/app.py", "x = 3\n").ok
    assert service.read_file(slug, "core/app.py").detail == "x = 3\n"
    assert service.diff(slug).ok
    assert service.remove_workspace(slug).ok


def test_service_reports_unknown_workspaces(project: Path, tmp_path: Path) -> None:
    service = CodeStudioService(
        project,
        studio=CodeStudio(project, workspaces_dir=tmp_path / "ws"),
        client=FakeCoder([]),
    )
    assert not service.read_file("ghost", "core/app.py").ok
    assert not service.remove_workspace("ghost").ok
    assert not service.merge("ghost").ok
    report, trace = service.run_agent("ghost", "do it")
    assert not report.ok and trace is None


def test_service_state_without_a_repository(tmp_path: Path) -> None:
    plain = tmp_path / "plain"
    plain.mkdir()
    service = CodeStudioService(
        plain,
        studio=CodeStudio(plain, workspaces_dir=tmp_path / "ws"),
        client=FakeCoder([]),
    )
    state = service.state()
    assert not state.repository
    assert state.workspaces == ()


def test_service_builds_the_default_client_lazily(project: Path, tmp_path: Path) -> None:
    service = CodeStudioService(
        project, studio=CodeStudio(project, workspaces_dir=tmp_path / "ws")
    )
    # Constructing the service must not build (or reach) the model; it is built on
    # first use, and the default is the Ollama client for the code model.
    assert isinstance(service.client, OllamaCoderClient)
    assert service.client.model == "qwen2.5-coder:7b"


# --- the CLI ---------------------------------------------------------------


def test_cli_status_and_list(project: Path, tmp_path: Path, capsys) -> None:
    from cli import studio as studio_cli

    root = ["--root", str(project), "--base-branch", "main"]
    assert studio_cli.main([*root, "status"]) == 0
    assert "Workspaces: none yet." in capsys.readouterr().out
    assert studio_cli.main([*root, "list"]) == 0
    assert "No workspaces yet" in capsys.readouterr().out


def test_cli_coder_exit_code_when_unreachable(project: Path, monkeypatch, capsys) -> None:
    from cli import studio as studio_cli

    monkeypatch.setattr(
        "gui.code_studio.CodeStudioService.status",
        lambda self: coder_status(BrokenCoder([])),
    )
    assert studio_cli.main(["--root", str(project), "coder"]) == 3
    assert "ollama" in capsys.readouterr().out.lower()


def test_cli_new_files_read_write_diff_discard(project: Path, capsys, tmp_path: Path, monkeypatch) -> None:
    from cli import studio as studio_cli

    monkeypatch.setenv("FAYD_DATA_DIR", str(tmp_path / "data"))
    root = ["--root", str(project), "--base-branch", "main"]
    assert studio_cli.main([*root, "new", "CLI change", "--slug", "cli-change"]) == 0
    assert "cli-change" in capsys.readouterr().out

    assert studio_cli.main([*root, "files", "cli-change"]) == 0
    assert "core/app.py" in capsys.readouterr().out

    assert studio_cli.main([*root, "read", "cli-change", "core/app.py"]) == 0
    assert "x = 1" in capsys.readouterr().out

    assert studio_cli.main([*root, "write", "cli-change", "core/app.py", "--content", "x = 9\n"]) == 0
    capsys.readouterr()

    assert studio_cli.main([*root, "diff", "cli-change"]) == 0
    assert "+x = 9" in capsys.readouterr().out

    assert studio_cli.main([*root, "discard", "cli-change"]) == 0
    capsys.readouterr()
    assert studio_cli.main([*root, "list"]) == 0
    assert "No workspaces yet" in capsys.readouterr().out


def test_cli_merge_refuses_without_passing_checks(project: Path, tmp_path: Path, monkeypatch, capsys) -> None:
    from cli import studio as studio_cli

    monkeypatch.setenv("FAYD_DATA_DIR", str(tmp_path / "data"))
    root = ["--root", str(project), "--base-branch", "main"]
    studio_cli.main([*root, "new", "Merge gate", "--slug", "gate"])
    capsys.readouterr()
    studio_cli.main([*root, "write", "gate", "core/app.py", "--content", "x = 4\n"])
    capsys.readouterr()
    assert studio_cli.main([*root, "merge", "gate"]) == 4
    assert "checks" in capsys.readouterr().out


def test_cli_merge_after_passing_checks(project: Path, tmp_path: Path, monkeypatch, capsys) -> None:
    from cli import studio as studio_cli

    monkeypatch.setenv("FAYD_DATA_DIR", str(tmp_path / "data"))
    # The check command is narrowed to a passing process: the tiny fixture repo has
    # no test suite, so a real `pytest` run there would fail for the wrong reason.
    root = [
        "--root", str(project), "--base-branch", "main",
        "--check-command", "python -c pass",
    ]
    assert studio_cli.main([*root, "new", "Merge me", "--slug", "merge-me"]) == 0
    capsys.readouterr()
    assert studio_cli.main([*root, "write", "merge-me", "core/app.py", "--content", "x = 11\n"]) == 0
    capsys.readouterr()

    assert studio_cli.main([*root, "check", "merge-me"]) == 0
    capsys.readouterr()
    assert studio_cli.main([*root, "merge", "merge-me"]) == 0
    capsys.readouterr()
    assert (project / "core" / "app.py").read_text(encoding="utf-8") == "x = 11\n"


def test_cli_json_output(project: Path, tmp_path: Path, monkeypatch, capsys) -> None:
    import json

    from cli import studio as studio_cli

    monkeypatch.setenv("FAYD_DATA_DIR", str(tmp_path / "data"))
    root = ["--root", str(project), "--base-branch", "main", "--json"]
    assert studio_cli.main([*root, "status"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["base_branch"] == "main"
    assert payload["repository"] is True
    assert payload["workspaces"] == []


def test_cli_edit_and_search(project: Path, tmp_path: Path, monkeypatch, capsys) -> None:
    from cli import studio as studio_cli

    monkeypatch.setenv("FAYD_DATA_DIR", str(tmp_path / "data"))
    root = ["--root", str(project), "--base-branch", "main"]
    assert studio_cli.main([*root, "new", "edit me"]) == 0
    capsys.readouterr()

    assert studio_cli.main([*root, "search", "edit-me", "x = 1"]) == 0
    assert "core/app.py:1:" in capsys.readouterr().out

    assert studio_cli.main(
        [*root, "edit", "edit-me", "core/app.py", "--old", "x = 1", "--new", "x = 5"]
    ) == 0
    assert "Replaced 1 occurrence" in capsys.readouterr().out
    assert (project / "core" / "app.py").read_text(encoding="utf-8") == "x = 1\n"  # worktree only

    # An ambiguous edit is refused with a non-zero exit, not silently applied.
    assert studio_cli.main(
        [*root, "write", "edit-me", "core/app.py", "--content", "a = 1\nb = 2\na = 1\n"]
    ) == 0
    capsys.readouterr()
    assert studio_cli.main(
        [*root, "edit", "edit-me", "core/app.py", "--old", "a = 1", "--new", "a = 9"]
    ) == 4
    capsys.readouterr()

    # --all opts into replacing every occurrence.
    assert studio_cli.main(
        [*root, "edit", "edit-me", "core/app.py", "--old", "a = 1", "--new", "a = 9", "--all"]
    ) == 0
    assert "Replaced 2 occurrence" in capsys.readouterr().out


def test_cli_models_and_download_gate(project: Path, tmp_path: Path, monkeypatch, capsys) -> None:
    from cli import studio as studio_cli

    monkeypatch.setenv("FAYD_DATA_DIR", str(tmp_path / "data"))
    root = ["--root", str(project), "--base-branch", "main"]
    assert studio_cli.main([*root, "models"]) == 0
    out = capsys.readouterr().out
    assert "qwen2.5-coder-7b" in out

    # Without --approve the download is refused before it can leave the machine.
    assert studio_cli.main([*root, "download"]) == 2
    assert "--approve" in capsys.readouterr().err


def test_cli_backend_reports_the_runtime(project: Path, tmp_path: Path, monkeypatch, capsys) -> None:
    import json

    from cli import studio as studio_cli

    monkeypatch.setenv("FAYD_DATA_DIR", str(tmp_path / "data"))
    root = ["--root", str(project), "--base-branch", "main", "--json"]
    # No model and no server in the test environment: honest, non-zero, actionable.
    code = studio_cli.main([*root, "backend"])
    payload = json.loads(capsys.readouterr().out)
    assert code == 3
    assert payload["available"] is False
    assert payload["backend"] in {"in-process", "ollama", "none"}
    assert payload["hint"]


def test_cli_without_a_subcommand_prints_help(capsys) -> None:
    from cli import studio as studio_cli

    assert studio_cli.main([]) == 0
    assert "usage: studio" in capsys.readouterr().out
