"""Code Studio: edit the project through isolated git worktrees, then merge.

The Studio gives the window (and the CLI) a safe way to *change the project's own
code* without ever editing the running checkout directly. It borrows the model
that a code host uses -- a branch per change, reviewed and merged when it is
good -- and implements it with the ``git`` the project already depends on:

* A **workspace** is a ``git worktree`` under ``data/studio/workspaces/<slug>``
  on its own branch ``studio/<slug>``. It is a full copy of the project the model
  and the user can edit, test, and diff in isolation. The running checkout is
  untouched, so a half-finished or broken change can never stop the program.
* Every workspace is created from one **base revision** (``main`` by default), so
  a merge is a fast-forward of ``main`` to the workspace branch -- exactly what a
  pull request does. A merge is refused unless the checks pass and ``main`` has
  not moved on, so a merge can never quietly overwrite newer work.
* All paths are confined to the workspace root: a request to read or write
  ``../../etc/passwd`` is refused, not resolved. Only text files with an allowed
  suffix are written, and an over-large file is refused.

Everything here is dependency-free (stdlib + the project's own ``git``), so it
imports and runs on a fresh clone. Nothing is executed by this module: the
``run_checks`` step runs the project's own test command, and that is the only
process it starts.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

from core.paths import studio_workspaces_dir

#: The folder the Studio keeps inside a checkout's own data directory.
_WORKSPACES_SUBDIR = Path("studio") / "workspaces"

#: The branch every workspace is created from, and merged back into.
DEFAULT_BASE_BRANCH = "main"

#: The branch-name prefix that marks a Studio workspace. Kept distinctive so a
#: workspace branch is never mistaken for a release or feature branch.
BRANCH_PREFIX = "studio/"

#: Workspaces are named automatically, in order: ``PR1``, ``PR2``, ... The user
#: never has to invent a name; :meth:`CodeStudio.create` assigns the next number.
SERIAL_PREFIX = "PR"

#: Text files the Studio will write. Anything else is refused, so the editor and
#: the agent cannot drop a binary or an executable into the project by accident.
WRITABLE_SUFFIXES = frozenset(
    {
        ".py", ".md", ".txt", ".rst", ".toml", ".cfg", ".ini", ".json", ".yaml", ".yml",
        ".html", ".css", ".js", ".ts", ".sh", ".bat", ".ps1", ".spec", ".gitignore",
    }
)

#: Directories never walked when listing files: run state, caches, and learned
#: data the user does not hand-edit.
SKIP_DIRS = frozenset(
    {".git", "__pycache__", ".pytest_cache", ".mypy_cache", ".ruff_cache", "data", "node_modules"}
)

#: A single file the Studio will read or write, in bytes.
DEFAULT_MAX_FILE_BYTES = 512_000

#: A hard ceiling on how much of the tree one listing will return.
DEFAULT_MAX_LISTED_FILES = 4000

#: A ceiling on the files one search walks, so a query cannot stall on a huge tree.
DEFAULT_MAX_SEARCHED_FILES = 4000

#: A ceiling on the hits one search returns, so the model's prompt stays bounded.
DEFAULT_MAX_SEARCH_HITS = 200

_GIT_TIMEOUT = 120.0


class StudioError(RuntimeError):
    """A Studio operation could not be completed as asked."""


@dataclass(frozen=True)
class GitCommand:
    """One git invocation and its outcome, kept for transparent reporting."""

    argv: tuple[str, ...]
    returncode: int
    stdout: str = ""
    stderr: str = ""

    @property
    def ok(self) -> bool:
        return self.returncode == 0


@dataclass
class Workspace:
    """One isolated change: a worktree, its branch, and its base revision."""

    slug: str
    branch: str
    path: Path
    base_branch: str
    base_revision: str
    task: str = ""
    checks_passed: bool | None = None
    checks_summary: str = ""
    created_at: str = ""

    def as_dict(self) -> dict[str, object]:
        return {
            "slug": self.slug,
            "branch": self.branch,
            "path": str(self.path),
            "base_branch": self.base_branch,
            "base_revision": self.base_revision,
            "task": self.task,
            "checks_passed": self.checks_passed,
            "checks_summary": self.checks_summary,
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, data: dict[str, object]) -> "Workspace":
        passed = data.get("checks_passed")
        return cls(
            slug=str(data["slug"]),
            branch=str(data["branch"]),
            path=Path(str(data["path"])),
            base_branch=str(data.get("base_branch", DEFAULT_BASE_BRANCH)),
            base_revision=str(data.get("base_revision", "")),
            task=str(data.get("task", "")),
            checks_passed=bool(passed) if passed is not None else None,
            checks_summary=str(data.get("checks_summary", "")),
            created_at=str(data.get("created_at", "")),
        )


@dataclass(frozen=True)
class FileEntry:
    """One file in a workspace, with its size for the editor's file list."""

    path: str
    size: int


@dataclass(frozen=True)
class SearchHit:
    """One matching line found by :meth:`CodeStudio.search`."""

    path: str
    line: int
    text: str
    before: str = ""
    after: str = ""

    def render(self) -> str:
        """``path:line: text``, with optional context lines, for the model."""
        lines = []
        if self.before:
            lines.append(self.before)
        lines.append(f"{self.path}:{self.line}: {self.text}")
        if self.after:
            lines.append(self.after)
        return "\n".join(lines)


@dataclass(frozen=True)
class CheckResult:
    """The outcome of running the project's checks inside a workspace."""

    command: tuple[str, ...]
    ok: bool
    returncode: int
    output: str = ""
    timed_out: bool = False

    def summary(self) -> str:
        if self.timed_out:
            return f"Checks timed out after running `{' '.join(self.command)}`."
        state = "passed" if self.ok else "failed"
        return f"Checks {state} (exit {self.returncode}) via `{' '.join(self.command)}`."


@dataclass
class MergeResult:
    """The outcome of merging a workspace branch back into the base branch."""

    ok: bool
    message: str
    base_revision: str = ""
    new_revision: str = ""
    commands: list[GitCommand] = field(default_factory=list)


#: The reserved key under which the monotonic workspace counter is kept in the
#: state file. It is not a workspace, so every reader skips it.
_SERIAL_KEY = "__next_serial__"


def _newest_first(workspace: "Workspace") -> tuple[str, int, str]:
    """A sort key that orders workspaces newest first, deterministically.

    The timestamp has second resolution, so two workspaces made in the same second
    would otherwise tie. The auto-assigned serial number (``PR12``) is the honest
    tie-breaker, and the slug breaks any remaining tie.
    """
    tail = workspace.slug[len(SERIAL_PREFIX):]
    serial = int(tail) if tail.isdigit() else 0
    return (workspace.created_at, serial, workspace.slug)


def _noninteractive_env(*, identity: bool = False) -> dict[str, str]:
    env = dict(os.environ)
    env.setdefault("GIT_TERMINAL_PROMPT", "0")
    env.setdefault("GIT_SSH_COMMAND", "ssh -o BatchMode=yes")
    if identity:
        # A commit needs an author. A machine with no global git identity (a fresh
        # container, a new laptop) would otherwise fail with "Author identity
        # unknown", so the Studio supplies its own only for the commits it makes.
        env.setdefault("GIT_AUTHOR_NAME", "Fayd Al-Ruh Studio")
        env.setdefault("GIT_AUTHOR_EMAIL", "studio@fayd-al-ruh.local")
        env.setdefault("GIT_COMMITTER_NAME", env["GIT_AUTHOR_NAME"])
        env.setdefault("GIT_COMMITTER_EMAIL", env["GIT_AUTHOR_EMAIL"])
    return env


class CodeStudio:
    """Manages Studio workspaces for one project checkout.

    ``root`` is the project's own checkout (the one holding ``.git``). It is only
    ever *read* by this class -- work happens in the worktrees under
    :func:`core.paths.studio_workspaces_dir`.
    """

    def __init__(
        self,
        root: Path | str,
        *,
        workspaces_dir: Path | str | None = None,
        base_branch: str = DEFAULT_BASE_BRANCH,
        max_file_bytes: int = DEFAULT_MAX_FILE_BYTES,
    ) -> None:
        self.root = Path(root).resolve()
        self.workspaces_dir = (
            Path(workspaces_dir) if workspaces_dir is not None else self._default_workspaces_dir()
        )
        self.base_branch = base_branch
        self.max_file_bytes = max_file_bytes

    def _default_workspaces_dir(self) -> Path:
        """Where a checkout's own worktrees live when no folder is given.

        The folder must belong to the checkout the Studio is editing: a shared
        folder would register another repository's worktree in this one's state.
        So the project's own checkout keeps its worktrees in the bundled data
        directory (git-ignored, beside the code), and any *other* checkout keeps
        them under its own root.
        """
        try:
            if self.root == Path(studio_workspaces_dir()).resolve().parents[2]:
                return studio_workspaces_dir()
        except (OSError, IndexError):  # pragma: no cover - defensive only
            pass
        return self.root / "data" / _WORKSPACES_SUBDIR

    # --- git plumbing ---------------------------------------------------
    def _git(
        self,
        working_dir: Path,
        *args: str,
        timeout: float = _GIT_TIMEOUT,
        strip_output: bool = True,
        identity: bool = False,
    ) -> GitCommand:
        argv = ("git", *args)
        try:
            completed = subprocess.run(
                argv,
                cwd=str(working_dir),
                capture_output=True,
                text=True,
                timeout=timeout,
                env=_noninteractive_env(identity=identity),
            )
        except FileNotFoundError as error:
            raise StudioError("git is not installed or not on the PATH.") from error
        except subprocess.TimeoutExpired as error:
            raise StudioError(f"`{' '.join(argv)}` timed out.") from error
        stdout = completed.stdout
        stderr = completed.stderr
        if strip_output:
            stdout, stderr = stdout.strip(), stderr.strip()
        return GitCommand(
            argv=argv,
            returncode=completed.returncode,
            stdout=stdout,
            stderr=stderr,
        )

    def is_repository(self) -> bool:
        """True when ``root`` is a git working tree the Studio can branch from."""
        return (self.root / ".git").exists()

    def _require_repository(self) -> None:
        if not self.is_repository():
            raise StudioError(
                f"{self.root} is not a git checkout, so the Studio cannot make "
                "isolated workspaces here."
            )

    def base_revision(self, branch: str | None = None) -> str:
        """The commit a new workspace would start from (short hash)."""
        branch = branch or self.base_branch
        result = self._git(self.root, "rev-parse", "--short", branch)
        if not result.ok:
            raise StudioError(f"Base branch `{branch}` does not exist in this checkout.")
        return result.stdout

    def _state_path(self) -> Path:
        return self.workspaces_dir / "workspaces.json"

    def _load_state(self) -> dict[str, dict[str, object]]:
        path = self._state_path()
        if not path.is_file():
            return {}
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return {}
        return data if isinstance(data, dict) else {}

    def _save_state(self, state: dict[str, dict[str, object]]) -> None:
        path = self._state_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(state, indent=2, sort_keys=True), encoding="utf-8")

    def _record(self, workspace: Workspace) -> None:
        state = self._load_state()
        state[workspace.slug] = workspace.as_dict()
        self._save_state(state)

    @staticmethod
    def _workspaces_only(state: dict[str, dict[str, object]]) -> dict[str, dict[str, object]]:
        """The state without the reserved counter entry, which is not a workspace."""
        return {slug: entry for slug, entry in state.items() if slug != _SERIAL_KEY}

    def _next_slug(self) -> str:
        """The next auto-assigned workspace name: ``PR1``, ``PR2``, ...

        The counter only ever moves forward, so a discarded workspace's number is
        never handed out again -- the numbering stays stable and predictable.
        """
        state = self._load_state()
        stored = state.get(_SERIAL_KEY)
        value = stored.get("value") if isinstance(stored, dict) else None
        current = value if isinstance(value, int) else 0
        serial = max(current, self._highest_existing_serial()) + 1
        state[_SERIAL_KEY] = {"value": serial}
        self._save_state(state)
        return f"{SERIAL_PREFIX}{serial}"

    def _highest_existing_serial(self) -> int:
        """The largest ``prN`` number any recorded workspace already uses."""
        highest = 0
        for slug in self._workspaces_only(self._load_state()):
            if slug.startswith(SERIAL_PREFIX) and slug[len(SERIAL_PREFIX):].isdigit():
                highest = max(highest, int(slug[len(SERIAL_PREFIX):]))
        return highest

    # --- lifecycle ------------------------------------------------------
    def create(self, task: str = "", *, slug: str | None = None) -> Workspace:
        """Create a new isolated workspace on its own branch.

        The branch is cut from :attr:`base_branch` at its current commit, so the
        workspace starts from exactly what is published. The workspace is named
        automatically in order -- ``PR1``, ``PR2``, ... -- unless ``slug`` is
        given, so the user never has to invent a name. ``task`` is the plain
        description of the change, kept so the window can list it.
        """
        self._require_repository()
        slug = slug or self._next_slug()
        branch = f"{BRANCH_PREFIX}{slug}"
        path = self.workspaces_dir / slug
        if path.exists():
            raise StudioError(f"A workspace named `{slug}` already exists.")
        if self._git(self.root, "rev-parse", "--verify", "--quiet", f"refs/heads/{branch}").ok:
            raise StudioError(f"Branch `{branch}` already exists; choose another name.")

        revision = self.base_revision()
        self.workspaces_dir.mkdir(parents=True, exist_ok=True)
        result = self._git(
            self.root, "worktree", "add", "-b", branch, str(path), self.base_branch
        )
        if not result.ok:
            raise StudioError(f"Could not create the workspace: {result.stderr or result.stdout}")

        workspace = Workspace(
            slug=slug,
            branch=branch,
            path=path,
            base_branch=self.base_branch,
            base_revision=revision,
            task=task.strip(),
            created_at=_now_iso(),
        )
        self._record(workspace)
        return workspace

    def list_workspaces(self) -> tuple[Workspace, ...]:
        """Every workspace the Studio knows about, newest first."""
        state = self._workspaces_only(self._load_state())
        items = [Workspace.from_dict(entry) for entry in state.values()]
        return tuple(sorted(items, key=_newest_first, reverse=True))

    def get(self, slug: str) -> Workspace:
        """One workspace by slug, or raise :class:`StudioError`."""
        state = self._workspaces_only(self._load_state())
        if slug not in state:
            raise StudioError(f"No workspace named `{slug}`.")
        return Workspace.from_dict(state[slug])

    def forget(self, slug: str) -> bool:
        """Drop a workspace from the Studio's list without touching the worktree."""
        state = self._load_state()
        if slug not in state:
            return False
        del state[slug]
        self._save_state(state)
        return True

    def remove(self, slug: str, *, force: bool = True) -> None:
        """Delete a workspace's worktree and branch, discarding its changes."""
        workspace = self.get(slug)
        args = ["worktree", "remove"]
        if force:
            args.append("--force")
        args.append(str(workspace.path))
        result = self._git(self.root, *args)
        if not result.ok:
            # A stale worktree registration should not block a discard.
            self._git(self.root, "worktree", "prune")
            if workspace.path.exists():
                shutil.rmtree(workspace.path, ignore_errors=True)
        self._git(self.root, "branch", "-D", workspace.branch)
        self.forget(slug)

    # --- confinement ----------------------------------------------------
    def resolve(self, workspace: Workspace, relative: str) -> Path:
        """Resolve ``relative`` inside the workspace, refusing any escape."""
        if not relative.strip():
            raise StudioError("Path cannot be empty.")
        candidate = (workspace.path / relative).resolve()
        root = workspace.path.resolve()
        if candidate != root and root not in candidate.parents:
            raise StudioError(f"Path `{relative}` escapes the workspace.")
        return candidate

    def _check_suffix(self, path: Path) -> None:
        suffix = path.suffix.casefold()
        if path.name.casefold() == ".gitignore":
            return
        if suffix not in WRITABLE_SUFFIXES:
            raise StudioError(
                f"`{path.name}` is not a text file the Studio edits "
                f"({suffix or 'no extension'} is not in the allowed list)."
            )

    def list_files(self, workspace: Workspace, *, limit: int = DEFAULT_MAX_LISTED_FILES) -> tuple[FileEntry, ...]:
        """Every editable text file in the workspace, sorted by path."""
        entries: list[FileEntry] = []
        for path in sorted(workspace.path.rglob("*")):
            if len(entries) >= limit:
                break
            if not path.is_file():
                continue
            relative = path.relative_to(workspace.path)
            if any(part in SKIP_DIRS for part in relative.parts):
                continue
            try:
                self._check_suffix(path)
            except StudioError:
                continue
            try:
                size = path.stat().st_size
            except OSError:
                continue
            entries.append(FileEntry(path=relative.as_posix(), size=size))
        return tuple(entries)

    def read_file(self, workspace: Workspace, relative: str) -> str:
        """Read one text file from the workspace."""
        path = self.resolve(workspace, relative)
        self._check_suffix(path)
        if not path.is_file():
            raise StudioError(f"Not a file: {relative}")
        if path.stat().st_size > self.max_file_bytes:
            raise StudioError(f"`{relative}` is larger than the editor's size limit.")
        return path.read_text(encoding="utf-8", errors="replace")

    def write_file(self, workspace: Workspace, relative: str, content: str) -> int:
        """Write one text file into the workspace; returns the bytes written."""
        path = self.resolve(workspace, relative)
        self._check_suffix(path)
        encoded = content.encode("utf-8")
        if len(encoded) > self.max_file_bytes:
            raise StudioError("The content is larger than the Studio's size limit.")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        self._invalidate_checks(workspace)
        return len(encoded)

    def _invalidate_checks(self, workspace: Workspace) -> None:
        """Forget a recorded check result, because the files have changed since.

        A check result is evidence about one revision. Editing a file afterwards
        makes it stale, and a stale *pass* must never authorise a merge, so the
        result is cleared and Merge waits for a fresh run.
        """
        if workspace.checks_passed is None and not workspace.checks_summary:
            return
        workspace.checks_passed = None
        workspace.checks_summary = ""
        self._record(workspace)

    def edit_file(
        self,
        workspace: Workspace,
        relative: str,
        old: str,
        new: str,
        *,
        replace_all: bool = False,
    ) -> int:
        """Replace ``old`` with ``new`` in one file; returns the replacements made.

        This is the *surgical* edit a whole-file rewrite cannot express: a large
        file is changed at one site without the model having to reproduce every
        other line. The match is exact and literal (never a regular expression),
        and the file is left untouched unless the match is unambiguous:

        * ``old`` must be non-empty, and must occur exactly once -- unless
          ``replace_all`` is set, in which case every occurrence is replaced.
        * No match, or an ambiguous one, raises :class:`StudioError` and changes
          nothing, so a mistaken edit cannot silently corrupt the file.
        """
        if not old:
            raise StudioError("The `old` text to replace cannot be empty.")
        path = self.resolve(workspace, relative)
        self._check_suffix(path)
        if not path.is_file():
            raise StudioError(f"Not a file: {relative}")
        if path.stat().st_size > self.max_file_bytes:
            raise StudioError(f"`{relative}` is larger than the editor's size limit.")
        text = path.read_text(encoding="utf-8", errors="replace")
        count = text.count(old)
        if count == 0:
            raise StudioError(f"The `old` text was not found in {relative}.")
        if count > 1 and not replace_all:
            raise StudioError(
                f"The `old` text appears {count} times in {relative}, so the edit is "
                "ambiguous. Include more surrounding context to make it unique, or "
                "set `replace_all` to true."
            )
        updated = text.replace(old, new) if replace_all else text.replace(old, new, 1)
        encoded = updated.encode("utf-8")
        if len(encoded) > self.max_file_bytes:
            raise StudioError("The edited file would be larger than the Studio's size limit.")
        path.write_text(updated, encoding="utf-8")
        self._invalidate_checks(workspace)
        return count if replace_all else 1

    def search(
        self,
        workspace: Workspace,
        query: str,
        *,
        relative: str = "",
        limit: int = DEFAULT_MAX_SEARCH_HITS,
        context_lines: int = 0,
    ) -> tuple[SearchHit, ...]:
        """Find a literal string in the workspace's text files.

        Case-insensitive and literal (never a regular expression), so a search for
        ``def foo(`` finds exactly that. ``relative`` narrows the walk to one file
        or subfolder. Returns at most ``limit`` hits, each with its line number and
        an optional line of context, so the model can locate code before editing it.
        """
        needle = query.strip()
        if not needle:
            raise StudioError("The search query cannot be empty.")
        root = workspace.path.resolve()
        start = self.resolve(workspace, relative) if relative.strip() else root
        if start.is_file():
            candidates = [start]
        else:
            candidates = []
            for path in sorted(start.rglob("*")):
                if len(candidates) >= DEFAULT_MAX_SEARCHED_FILES:
                    break
                if not path.is_file():
                    continue
                rel = path.relative_to(root)
                if any(part in SKIP_DIRS for part in rel.parts):
                    continue
                try:
                    self._check_suffix(path)
                except StudioError:
                    continue
                candidates.append(path)

        lowered = needle.casefold()
        hits: list[SearchHit] = []
        for path in candidates:
            if len(hits) >= limit:
                break
            try:
                if path.stat().st_size > self.max_file_bytes:
                    continue
                lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
            except OSError:
                continue
            relative_path = path.relative_to(root).as_posix()
            for number, line in enumerate(lines, start=1):
                if lowered in line.casefold():
                    before, after = self._context(lines, number, context_lines)
                    hits.append(
                        SearchHit(
                            path=relative_path,
                            line=number,
                            text=line.strip()[:240],
                            before=before,
                            after=after,
                        )
                    )
                    if len(hits) >= limit:
                        break
        return tuple(hits)

    @staticmethod
    def _context(lines: Sequence[str], number: int, context_lines: int) -> tuple[str, str]:
        """The lines just before and after ``number`` (1-based), if asked for."""
        if context_lines <= 0:
            return "", ""
        start = max(0, number - 1 - context_lines)
        end = min(len(lines), number + context_lines)
        before = "\n".join(line.rstrip() for line in lines[start : number - 1])
        after = "\n".join(line.rstrip() for line in lines[number:end])
        return before, after

    # --- change inspection ----------------------------------------------
    def changed_files(self, workspace: Workspace) -> tuple[str, ...]:
        """Files added or modified in the workspace, including untracked ones."""
        result = self._git(workspace.path, "status", "--porcelain", strip_output=False)
        if not result.ok:
            return ()
        names: list[str] = []
        for line in result.stdout.splitlines():
            if len(line) < 4:
                continue
            # Porcelain format is a fixed 2-char status, a space, then the path.
            name = line[3:]
            if " -> " in name:  # a rename: keep the destination
                name = name.split(" -> ", 1)[1]
            name = name.strip().strip('"')
            if name:
                names.append(name)
        return tuple(names)

    def diff(self, workspace: Workspace) -> str:
        """The workspace's change as a unified diff against its base revision.

        Untracked files are included (``git add -N`` marks their intent to add)
        so a new file the model wrote is visible in the review, exactly like a
        pull request shows new files.
        """
        self._git(workspace.path, "add", "-A", "-N")
        result = self._git(workspace.path, "diff", "--no-color", "--stat", "-p")
        if not result.ok:
            return result.stderr or "(no diff available)"
        return result.stdout or "(no changes yet)"

    def commit(self, workspace: Workspace, message: str = "") -> GitCommand:
        """Commit the workspace's changes on its branch (needed before a merge)."""
        self._git(workspace.path, "add", "-A")
        text = message.strip() or f"Studio change: {workspace.task or workspace.slug}"
        return self._git(workspace.path, "commit", "-m", text, identity=True)

    # --- checks ---------------------------------------------------------
    def run_checks(
        self,
        workspace: Workspace,
        command: Sequence[str] | None = None,
        *,
        timeout: float = 900.0,
    ) -> CheckResult:
        """Run the project's checks inside the workspace.

        The default is the project's own test command, run *in the workspace* so a
        failing change is caught before it can reach ``main``. The result is
        recorded on the workspace, so the window can gate Merge on it.
        """
        argv = tuple(command) if command else ("python", "-m", "pytest", "-q")
        try:
            completed = subprocess.run(
                argv,
                cwd=str(workspace.path),
                capture_output=True,
                text=True,
                timeout=timeout,
                env=_noninteractive_env(),
            )
            result = CheckResult(
                command=argv,
                ok=completed.returncode == 0,
                returncode=completed.returncode,
                output=_tail(completed.stdout + completed.stderr),
            )
        except FileNotFoundError as error:
            result = CheckResult(
                command=argv, ok=False, returncode=127, output=f"Could not run checks: {error}"
            )
        except subprocess.TimeoutExpired:
            result = CheckResult(
                command=argv,
                ok=False,
                returncode=124,
                output=f"Timed out after {timeout:.0f}s.",
                timed_out=True,
            )
        workspace.checks_passed = result.ok
        workspace.checks_summary = result.summary()
        self._record(workspace)
        return result

    # --- merge ----------------------------------------------------------
    def can_merge(self, workspace: Workspace) -> tuple[bool, str]:
        """Whether this workspace may be merged now, with the reason either way."""
        self._require_repository()
        if not self.changed_files(workspace):
            return False, "This workspace has no changes to merge."
        current = self.base_revision()
        if current != workspace.base_revision:
            return (
                False,
                f"`{workspace.base_branch}` has moved to {current} since this workspace "
                f"started from {workspace.base_revision}. Re-create or rebase the workspace "
                "so the merge cannot overwrite newer work.",
            )
        if workspace.checks_passed is not True:
            return False, "Run the checks and let them pass before merging."
        return True, "Ready to merge."

    def merge(self, workspace: Workspace, *, message: str = "") -> MergeResult:
        """Fast-forward ``base_branch`` to the workspace branch, if it is safe.

        This is what the window's **Merge** button calls. It refuses when the
        checks have not passed or ``base_branch`` has moved on, so a merge can
        never discard newer work. The running checkout is only advanced when it is
        clean and sitting on ``base_branch``.
        """
        allowed, reason = self.can_merge(workspace)
        if not allowed:
            return MergeResult(ok=False, message=reason)

        head = self._git(self.root, "rev-parse", "--abbrev-ref", "HEAD").stdout
        if head != workspace.base_branch:
            return MergeResult(
                ok=False,
                message=(
                    f"This checkout is on `{head}`, not `{workspace.base_branch}`; "
                    f"check out `{workspace.base_branch}` and merge again."
                ),
            )
        dirty = self._git(self.root, "status", "--porcelain", "--untracked-files=no").stdout
        if dirty:
            return MergeResult(
                ok=False,
                message="The main checkout has uncommitted changes; commit them before merging.",
            )

        commands: list[GitCommand] = []
        commit = self.commit(workspace, message or f"Studio: {workspace.task or workspace.slug}")
        commands.append(commit)
        if not commit.ok and "nothing to commit" not in (commit.stdout + commit.stderr):
            return MergeResult(
                ok=False,
                message=f"Could not commit the workspace: {commit.stderr or commit.stdout}",
                commands=commands,
            )

        before = self.base_revision()
        merge = self._git(self.root, "merge", "--ff-only", workspace.branch)
        commands.append(merge)
        if not merge.ok:
            return MergeResult(
                ok=False,
                message=(
                    "The fast-forward merge failed (the branches may have diverged): "
                    f"{merge.stderr or merge.stdout}"
                ),
                base_revision=before,
                commands=commands,
            )

        after = self._git(self.root, "rev-parse", "--short", "HEAD").stdout
        return MergeResult(
            ok=True,
            message=f"Merged `{workspace.branch}` into `{workspace.base_branch}` at {after}.",
            base_revision=before,
            new_revision=after,
            commands=commands,
        )


def _now_iso() -> str:
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _tail(text: str, *, limit: int = 6000) -> str:
    """The end of a long process log, where the failure usually is."""
    stripped = text.strip()
    if len(stripped) <= limit:
        return stripped
    return "…\n" + stripped[-limit:]


def default_check_command(root: Path | str | None = None) -> tuple[str, ...]:
    """The project's own test command, used to gate a merge by default."""
    return ("python", "-m", "pytest", "-q")


__all__ = [
    "BRANCH_PREFIX",
    "DEFAULT_BASE_BRANCH",
    "DEFAULT_MAX_FILE_BYTES",
    "DEFAULT_MAX_SEARCHED_FILES",
    "DEFAULT_MAX_SEARCH_HITS",
    "SKIP_DIRS",
    "WRITABLE_SUFFIXES",
    "CheckResult",
    "CodeStudio",
    "FileEntry",
    "GitCommand",
    "MergeResult",
    "SearchHit",
    "StudioError",
    "Workspace",
    "default_check_command",
]
