"""The Code Studio's git engine: everything a developer does with git, in one place.

The Studio already made a ``git worktree`` per change and merged it back
(:mod:`core.code_studio`). This module is the rest of git: what a developer
reaches for *while* the change is being made -- see the status, review the log,
stage a file, commit, branch, stash, blame, discard a hunk, and (when the user
explicitly allows it) talk to a remote.

It is deliberately a **thin, honest wrapper** over the project's own ``git``:

* Every call is one real ``git`` invocation, recorded as a
  :class:`GitResult` (argv, exit code, stdout, stderr), so nothing is guessed
  and a failure says exactly what git said.
* Output is **parsed into records** (:class:`StatusEntry`, :class:`CommitInfo`,
  :class:`BranchInfo`, ...), so the window renders a table rather than a wall of
  text, and the CLI can emit JSON.
* A **remote URL is always sanitized** before it is reported, so an embedded
  token in ``https://user:secret@host/repo`` is never printed or logged.
* Network commands (``fetch``/``pull``/``push``) exist but are **never** run by
  this module on its own: the caller gates them (the service's permission and
  human-approval gates), which is the project's rule for anything that leaves
  the machine.

Nothing here executes a repository's own hooks beyond what git itself runs, and
nothing here evaluates a file's contents.
"""

from __future__ import annotations

import re
import subprocess
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

from core.updater import sanitize_remote_url

#: A git call that takes longer than this is treated as a failure rather than
#: hanging a window. A fetch over a slow link is the one slow operation, and it
#: is gated by an explicit approval anyway.
DEFAULT_TIMEOUT = 120.0

#: How much output one git call keeps. A ``git log`` on a huge history, or a
#: ``blame`` of a long file, is trimmed rather than handed to the window whole.
DEFAULT_MAX_OUTPUT = 200_000

#: A hard ceiling on the entries one listing returns, so a repository with
#: thousands of branches cannot stall a menu.
DEFAULT_LIST_LIMIT = 200

#: The porcelain-v2 status codes that mean a path is conflicted. Kept explicit so
#: a conflict is never confused with an ordinary modification.
_CONFLICT_CODES = frozenset({"DD", "AU", "UD", "UA", "DU", "AA", "UU"})


class GitError(RuntimeError):
    """A git operation could not be completed as asked."""


@dataclass(frozen=True)
class GitResult:
    """One git invocation and its outcome, kept so reporting is transparent."""

    argv: tuple[str, ...]
    returncode: int
    stdout: str = ""
    stderr: str = ""

    @property
    def ok(self) -> bool:
        return self.returncode == 0

    @property
    def output(self) -> str:
        """stdout plus stderr, trimmed -- what a caller shows on failure."""
        return (self.stdout + ("\n" if self.stdout and self.stderr else "") + self.stderr).strip()

    def summary(self) -> str:
        if self.ok:
            return f"`{' '.join(self.argv)}` succeeded."
        return f"`{' '.join(self.argv)}` failed (exit {self.returncode}): {self.stderr or self.stdout}"


@dataclass(frozen=True)
class StatusEntry:
    """One path in ``git status``, with its index and worktree state split out.

    Porcelain format is ``XY path``: ``X`` is the index (staged) state and ``Y``
    the worktree (unstaged) state, so the window can show a file as *staged*,
    *modified*, *untracked*, or *conflicted* rather than as one opaque line.
    """

    path: str
    index_status: str
    worktree_status: str
    original_path: str = ""

    @property
    def code(self) -> str:
        return f"{self.index_status}{self.worktree_status}"

    @property
    def conflicted(self) -> bool:
        return self.code in _CONFLICT_CODES

    @property
    def untracked(self) -> bool:
        return self.code == "??"

    @property
    def ignored(self) -> bool:
        return self.code == "!!"

    @property
    def staged(self) -> bool:
        """True when the index holds a change (staged add/modify/delete/rename)."""
        return self.index_status not in {" ", "?", "!"}

    @property
    def unstaged(self) -> bool:
        """True when the worktree differs from the index."""
        return self.worktree_status not in {" ", "?", "!"}

    @property
    def renamed(self) -> bool:
        return "R" in self.code

    @property
    def deleted(self) -> bool:
        return "D" in self.code

    def describe(self) -> str:
        """``M  path`` -- the code and the path, the way ``git status -s`` shows it."""
        label = f"{self.code} {self.path}"
        if self.original_path:
            label += f"  (from {self.original_path})"
        return label


@dataclass(frozen=True)
class CommitInfo:
    """One commit from ``git log``."""

    sha: str
    short_sha: str
    author: str
    date: str
    subject: str
    parents: tuple[str, ...] = ()

    @property
    def merge(self) -> bool:
        return len(self.parents) > 1

    def describe(self) -> str:
        mark = " (merge)" if self.merge else ""
        return f"{self.short_sha}  {self.date}  {self.author}: {self.subject}{mark}"


@dataclass(frozen=True)
class BranchInfo:
    """One local or remote branch, with its upstream and ahead/behind counts."""

    name: str
    current: bool = False
    upstream: str = ""
    ahead: int = 0
    behind: int = 0
    subject: str = ""

    def describe(self) -> str:
        mark = "* " if self.current else "  "
        text = f"{mark}{self.name}"
        if self.upstream:
            track = f" -> {self.upstream}"
            if self.ahead or self.behind:
                track += f" [{self.ahead} ahead, {self.behind} behind]"
            text += track
        if self.subject:
            text += f"  {self.subject}"
        return text


@dataclass(frozen=True)
class RemoteInfo:
    """One configured remote. Its URL is always sanitized of any credential."""

    name: str
    url: str

    def describe(self) -> str:
        return f"{self.name}\t{self.url}"


@dataclass(frozen=True)
class StashEntry:
    """One entry in the stash list."""

    index: int
    ref: str
    subject: str

    def describe(self) -> str:
        return f"stash@{{{self.index}}}  {self.subject}"


@dataclass(frozen=True)
class ConflictReport:
    """Which files are mid-merge, and whether an operation is in progress."""

    files: tuple[str, ...] = ()
    operation: str = ""

    @property
    def in_progress(self) -> bool:
        return bool(self.files) or bool(self.operation)

    def describe(self) -> str:
        if not self.in_progress:
            return "No merge or rebase in progress."
        head = f"{self.operation or 'merge'} in progress" if self.operation else "conflicts"
        return f"{head}: " + ", ".join(self.files)


@dataclass
class PatchResult:
    """The outcome of applying a patch, with git's own words kept."""

    ok: bool
    message: str
    detail: str = ""
    commands: list[GitResult] = field(default_factory=list)


def _noninteractive_env(*, identity: bool = False) -> dict[str, str]:
    """A git environment that never prompts, so a call can never block a window.

    ``identity=True`` also supplies an author for the commits the Studio makes
    itself, so a machine with no global git identity still works.
    """
    import os

    env = dict(os.environ)
    env.setdefault("GIT_TERMINAL_PROMPT", "0")
    env.setdefault("GIT_SSH_COMMAND", "ssh -o BatchMode=yes")
    if identity:
        env.setdefault("GIT_AUTHOR_NAME", "Fayd Al-Ruh Studio")
        env.setdefault("GIT_AUTHOR_EMAIL", "studio@fayd-al-ruh.local")
        env.setdefault("GIT_COMMITTER_NAME", "Fayd Al-Ruh Studio")
        env.setdefault("GIT_COMMITTER_EMAIL", "studio@fayd-al-ruh.local")
    return env


def _sanitize(text: str) -> str:
    """Strip any ``scheme://user:secret@host`` credential from free text.

    A remote URL with an embedded token is the classic leak, and git echoes the
    remote in several messages (a fetch failure, a push rejection). Every message
    this module returns therefore passes through here.
    """
    return re.sub(r"(?i)\b([a-z][a-z0-9+.\-]*://)[^/\s@]+@", r"\1<credentials>@", text)


class GitEngine:
    """Runs git for one directory (a workspace, or the project checkout)."""

    def __init__(
        self,
        root: Path | str,
        *,
        timeout: float = DEFAULT_TIMEOUT,
        max_output: int = DEFAULT_MAX_OUTPUT,
    ) -> None:
        self.root = Path(root)
        self.timeout = timeout
        self.max_output = max_output

    # --- plumbing -------------------------------------------------------
    def run(
        self,
        *args: str,
        check: bool = False,
        identity: bool = False,
        timeout: float | None = None,
    ) -> GitResult:
        """Run one git command in :attr:`root`, returning its recorded outcome.

        Never raises for a *failing* git command (a non-zero exit is reported in
        the result); it raises :class:`GitError` only when git itself cannot be
        started or the call times out.
        """
        argv = ("git", *args)
        try:
            completed = subprocess.run(
                argv,
                cwd=str(self.root),
                capture_output=True,
                text=True,
                timeout=timeout if timeout is not None else self.timeout,
                env=_noninteractive_env(identity=identity),
            )
        except FileNotFoundError as error:
            raise GitError("git is not installed or not on the PATH.") from error
        except subprocess.TimeoutExpired as error:
            raise GitError(f"`{' '.join(argv)}` timed out.") from error
        stdout = completed.stdout[: self.max_output]
        stderr = _sanitize(completed.stderr[: self.max_output])
        result = GitResult(
            argv=argv,
            returncode=completed.returncode,
            stdout=_sanitize(stdout),
            stderr=stderr,
        )
        if check and not result.ok:
            raise GitError(result.summary())
        return result

    def is_repository(self) -> bool:
        return (self.root / ".git").exists()

    def _require_repository(self) -> None:
        if not self.is_repository():
            raise GitError(f"{self.root} is not a git checkout.")

    # --- read-only inspection -------------------------------------------
    def current_branch(self) -> str:
        result = self.run("rev-parse", "--abbrev-ref", "HEAD")
        return result.stdout.strip() if result.ok else ""

    def head_revision(self, ref: str = "HEAD") -> str:
        result = self.run("rev-parse", "--short", ref)
        return result.stdout.strip() if result.ok else ""

    def full_revision(self, ref: str = "HEAD") -> str:
        result = self.run("rev-parse", ref)
        return result.stdout.strip() if result.ok else ""

    def has_commits(self) -> bool:
        return self.run("rev-parse", "--verify", "--quiet", "HEAD").ok

    def status(self, *, include_untracked: bool = True) -> tuple[StatusEntry, ...]:
        """Parse ``git status --porcelain`` into structured entries."""
        args = ["status", "--porcelain", "--untracked-files=" + ("all" if include_untracked else "no")]
        result = self.run(*args)
        if not result.ok:
            return ()
        entries: list[StatusEntry] = []
        for line in result.stdout.splitlines():
            if len(line) < 4:
                continue
            index_status, worktree_status = line[0], line[1]
            name = line[3:]
            original = ""
            if " -> " in name:  # a rename: "R  old -> new"
                original, name = name.split(" -> ", 1)
            entries.append(
                StatusEntry(
                    path=name.strip().strip('"'),
                    index_status=index_status,
                    worktree_status=worktree_status,
                    original_path=original.strip().strip('"'),
                )
            )
        return tuple(entries)

    def is_clean(self, *, include_untracked: bool = True) -> bool:
        return not self.status(include_untracked=include_untracked)

    def conflicts(self) -> ConflictReport:
        """Which files are unmerged, and whether a merge/rebase is under way."""
        unmerged = tuple(
            entry.path for entry in self.status(include_untracked=False) if entry.conflicted
        )
        operation = ""
        git_dir = self.root / ".git"
        if (git_dir / "MERGE_HEAD").exists():
            operation = "merge"
        elif (git_dir / "rebase-merge").exists() or (git_dir / "rebase-apply").exists():
            operation = "rebase"
        elif (git_dir / "CHERRY_PICK_HEAD").exists():
            operation = "cherry-pick"
        elif (git_dir / "REVERT_HEAD").exists():
            operation = "revert"
        return ConflictReport(files=unmerged, operation=operation)

    def log(
        self,
        *,
        limit: int = 30,
        path: str = "",
        all_branches: bool = False,
    ) -> tuple[CommitInfo, ...]:
        """The most recent commits, newest first, as structured records.

        The format uses a rare record separator so a multi-line commit message
        (or a subject with a pipe in it) cannot be mis-parsed.
        """
        sep = "\x1f"  # unit separator: never in a commit message in practice
        fmt = sep.join(["%H", "%h", "%an", "%ad", "%s", "%P"])
        args = ["log", f"--max-count={max(1, limit)}", f"--pretty=format:{fmt}", "--date=short"]
        if all_branches:
            args.append("--all")
        if path:
            args.extend(["--", path])
        result = self.run(*args)
        if not result.ok:
            return ()
        commits: list[CommitInfo] = []
        for line in result.stdout.splitlines():
            if line.count(sep) < 5:
                continue
            sha, short, author, date, subject, parents = line.split(sep, 5)
            commits.append(
                CommitInfo(
                    sha=sha,
                    short_sha=short,
                    author=author,
                    date=date,
                    subject=subject,
                    parents=tuple(parents.split()) if parents.strip() else (),
                )
            )
        return tuple(commits)

    def branches(self, *, include_remote: bool = False, limit: int = DEFAULT_LIST_LIMIT) -> tuple[BranchInfo, ...]:
        """Local branches (and, on request, remote-tracking ones) with tracking state."""
        sep = "\x1f"
        fmt = sep.join(
            [
                "%(refname:short)",
                "%(HEAD)",
                "%(upstream:short)",
                "%(upstream:track)",
                "%(contents:subject)",
            ]
        )
        args = ["for-each-ref", f"--count={max(1, limit)}", f"--format={fmt}", "refs/heads"]
        if include_remote:
            args.append("refs/remotes")
        result = self.run(*args)
        if not result.ok:
            return ()
        branches: list[BranchInfo] = []
        for line in result.stdout.splitlines():
            parts = line.split(sep)
            if len(parts) < 5:
                continue
            name, head, upstream, track, subject = parts[:5]
            ahead, behind = _parse_track(track)
            branches.append(
                BranchInfo(
                    name=name,
                    current=head.strip() == "*",
                    upstream=upstream,
                    ahead=ahead,
                    behind=behind,
                    subject=subject,
                )
            )
        return tuple(branches)

    def remotes(self) -> tuple[RemoteInfo, ...]:
        """Configured remotes, each with its URL sanitized of any credential."""
        result = self.run("remote", "-v")
        if not result.ok:
            return ()
        seen: dict[str, str] = {}
        for line in result.stdout.splitlines():
            parts = line.split()
            if len(parts) >= 2 and parts[0] not in seen:
                seen[parts[0]] = sanitize_remote_url(parts[1])
        return tuple(RemoteInfo(name, url) for name, url in seen.items())

    def tags(self, *, limit: int = DEFAULT_LIST_LIMIT) -> tuple[str, ...]:
        result = self.run("tag", "--sort=-creatordate")
        if not result.ok:
            return ()
        return tuple(result.stdout.splitlines()[:limit])

    def stash_list(self) -> tuple[StashEntry, ...]:
        result = self.run("stash", "list", "--pretty=format:%gd\x1f%s")
        if not result.ok:
            return ()
        entries: list[StashEntry] = []
        for index, line in enumerate(result.stdout.splitlines()):
            if "\x1f" in line:
                ref, subject = line.split("\x1f", 1)
            else:
                ref, subject = f"stash@{{{index}}}", line
            entries.append(StashEntry(index=index, ref=ref, subject=subject))
        return tuple(entries)

    def diff(
        self,
        *,
        staged: bool = False,
        path: str = "",
        against: str = "",
        stat: bool = False,
        context_lines: int = 3,
    ) -> str:
        """A unified diff, either of the worktree, the index, or against a ref."""
        args = ["diff", "--no-color", f"--unified={max(0, context_lines)}"]
        if stat:
            args.append("--stat")
        if staged:
            args.append("--cached")
        if against:
            args.append(against)
        if path:
            args.extend(["--", path])
        result = self.run(*args)
        if not result.ok:
            raise GitError(result.stderr or "Could not produce a diff.")
        return result.stdout or "(no changes)"

    def show(self, ref: str = "HEAD", *, stat: bool = True) -> str:
        """One commit's own diff (or any object git can show)."""
        args = ["show", "--no-color"]
        args.append("--stat" if stat else "-p")
        args.append(ref)
        result = self.run(*args)
        if not result.ok:
            raise GitError(result.stderr or f"Could not show {ref}.")
        return result.stdout

    def blame(self, path: str, *, limit: int = 400) -> str:
        """``git blame`` for one file, trimmed to a bounded number of lines."""
        result = self.run("blame", "--date=short", "--", path)
        if not result.ok:
            raise GitError(result.stderr or f"Could not blame {path}.")
        lines = result.stdout.splitlines()
        if len(lines) > limit:
            lines = [*lines[:limit], f"… [{len(lines) - limit} more line(s)]"]
        return "\n".join(lines) or "(nothing to blame)"

    def config_user(self) -> tuple[str, str]:
        """The configured ``user.name`` and ``user.email`` (empty when unset)."""
        name = self.run("config", "user.name").stdout.strip()
        email = self.run("config", "user.email").stdout.strip()
        return name, email

    # --- mutating, local -------------------------------------------------
    def stage(self, *paths: str) -> GitResult:
        """Stage the given paths (``git add``); with no path, stage everything."""
        if not paths:
            return self.run("add", "-A")
        return self.run("add", "--", *paths)

    def unstage(self, *paths: str) -> GitResult:
        """Unstage paths, keeping their changes in the worktree."""
        if not paths:
            return self.run("reset", "--quiet", "HEAD")
        return self.run("reset", "--quiet", "HEAD", "--", *paths)

    def commit(self, message: str, *, paths: Sequence[str] = (), all: bool = False) -> GitResult:
        """Create a commit with ``message`` (author supplied if none is configured)."""
        text = message.strip() or "Studio: update"
        args = ["commit", "-m", text]
        if all:
            args.insert(1, "-a")
        if paths:
            args.extend(["--", *paths])
        return self.run(*args, identity=True)

    def amend(self, message: str = "") -> GitResult:
        """Amend the last commit, optionally rewriting its message."""
        args = ["commit", "--amend"]
        if message.strip():
            args.extend(["-m", message.strip()])
        else:
            args.append("--no-edit")
        return self.run(*args, identity=True)

    def restore(self, *paths: str) -> GitResult:
        """Discard worktree changes for the given paths (``git restore``)."""
        if not paths:
            return self.run("restore", ".")
        return self.run("restore", "--", *paths)

    def stash(self, message: str = "", *, include_untracked: bool = False) -> GitResult:
        args = ["stash", "push"]
        if include_untracked:
            args.append("--include-untracked")
        if message.strip():
            args.extend(["-m", message.strip()])
        return self.run(*args)

    def stash_pop(self, index: int = 0) -> GitResult:
        return self.run("stash", "pop", f"stash@{{{max(0, index)}}}")

    def stash_drop(self, index: int = 0) -> GitResult:
        return self.run("stash", "drop", f"stash@{{{max(0, index)}}}")

    def create_branch(self, name: str, *, checkout: bool = True, start_point: str = "") -> GitResult:
        text = name.strip()
        if not text:
            raise GitError("A branch name is required.")
        if checkout:
            args = ["switch", "-c", text]
            if start_point:
                args.append(start_point)
            return self.run(*args)
        args = ["branch", text]
        if start_point:
            args.append(start_point)
        return self.run(*args)

    def switch_branch(self, name: str) -> GitResult:
        return self.run("switch", name.strip())

    def delete_branch(self, name: str, *, force: bool = False) -> GitResult:
        flag = "-D" if force else "-d"
        return self.run("branch", flag, name.strip())

    def apply_patch(self, patch_text: str, *, check: bool = False, reverse: bool = False) -> PatchResult:
        """Apply a unified diff to the worktree, optionally as a dry run.

        A patch is written to a temporary file and applied with ``git apply``; the
        call is never a shell, so nothing in the patch text can be executed. A
        ``check`` run reports whether the patch *would* apply without touching a
        file, which is how the window previews a change before applying it.
        """
        if not patch_text.strip():
            return PatchResult(False, "The patch is empty.")
        import tempfile

        with tempfile.NamedTemporaryFile("w", suffix=".patch", delete=False, encoding="utf-8") as handle:
            handle.write(patch_text)
            temp = Path(handle.name)
        try:
            args = ["apply"]
            if check:
                args.append("--check")
            if reverse:
                args.append("--reverse")
            args.append(str(temp))
            result = self.run(*args)
        finally:
            temp.unlink(missing_ok=True)
        if result.ok:
            verb = "would apply" if check else "applied"
            return PatchResult(True, f"The patch {verb} cleanly.", detail=result.output, commands=[result])
        return PatchResult(False, "The patch did not apply.", detail=result.output, commands=[result])

    def revert_commit(self, ref: str = "HEAD", *, no_commit: bool = False) -> GitResult:
        args = ["revert", "--no-edit"]
        if no_commit:
            args.append("--no-commit")
        args.append(ref)
        return self.run(*args)

    # --- network (never called by this module on its own) ---------------
    def fetch(self, *, remote: str = "origin", prune: bool = True) -> GitResult:
        args = ["fetch", remote]
        if prune:
            args.append("--prune")
        return self.run(*args)

    def pull(self, *, remote: str = "origin", rebase: bool = False) -> GitResult:
        args = ["pull"]
        if rebase:
            args.append("--rebase")
        args.append(remote)
        return self.run(*args)

    def push(self, *, remote: str = "origin", branch: str = "", set_upstream: bool = False) -> GitResult:
        args = ["push"]
        if set_upstream:
            args.append("--set-upstream")
        args.append(remote)
        if branch:
            args.append(branch)
        return self.run(*args)


def _parse_track(text: str) -> tuple[int, int]:
    """Read ``[ahead 2, behind 1]`` (or ``[gone]``) into ``(ahead, behind)``."""
    if not text.strip():
        return 0, 0
    ahead = re.search(r"ahead (\d+)", text)
    behind = re.search(r"behind (\d+)", text)
    return (int(ahead.group(1)) if ahead else 0, int(behind.group(1)) if behind else 0)


__all__ = [
    "DEFAULT_LIST_LIMIT",
    "DEFAULT_TIMEOUT",
    "BranchInfo",
    "CommitInfo",
    "ConflictReport",
    "GitEngine",
    "GitError",
    "GitResult",
    "PatchResult",
    "RemoteInfo",
    "StashEntry",
    "StatusEntry",
]
