"""Check a source checkout for program updates and apply them in place.

Instead of re-downloading the whole repository whenever something changes on
GitHub, this module asks the remote what has changed and lets an installed
checkout pull only the new commits. It is a thin, inspectable wrapper around the
``git`` the project is already developed and cloned with, so it carries no extra
dependency and never touches the network by itself unless a check is requested.

The design follows the project's existing governance: reading the remote is a
network action, and it is the caller (the CLI flag or the GUI button) that
decides to allow it. A check only ever fetches metadata; an update is a
fast-forward of the current branch, and it is refused outright whenever the
destination has local changes, is behind the current branch, or the remote is
unreachable -- a failed check must never look like a successful update.

Only a *source checkout* can self-update: a frozen portable build has no ``.git``
and no ``git`` on the path, so it gets a clear "reinstall the portable bundle"
message instead of a stack trace.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from core.paths import executable_dir, is_frozen
from core.version import __version__

# The repository the project is published from. Kept here so the CLI, the GUI,
# and the tests all agree on one address; the git remote, when present, wins.
DEFAULT_REPOSITORY_URL = "https://github.com/pt5ri-debug/fayd-al-ruh.git"

#: Paths the updater treats as learned state. Commits that touch these change
#: the *bundled* starting copy, not the user's own data (which lives in the
#: git-ignored data folder), so a change here is reported as a note.
LEARNED_PATH_PREFIXES = ("ml/artifacts/", "ml/llm/checkpoints/")

_GIT_TIMEOUT = 60.0


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


class UpdateError(RuntimeError):
    """A check or update could not be completed as requested."""


@dataclass
class UpdateStatus:
    """Everything a caller needs to describe an update check or its result."""

    available: bool
    current_version: str
    current_revision: str
    latest_revision: str
    branch: str
    behind: int
    ahead: int
    dirty: bool
    repository_url: str
    reason: str
    head_branch: str = ""
    applied: bool = False
    changed_files: tuple[str, ...] = ()
    commands: list[GitCommand] | None = None
    backup_path: str = ""
    backup_copied: int = 0
    backup_error: str = ""
    blocked: bool = False

    @property
    def learned_files_changed(self) -> tuple[str, ...]:
        """Changed files that belong to the committed training set."""
        return tuple(
            name
            for name in self.changed_files
            if name.startswith(LEARNED_PATH_PREFIXES)
        )

    @property
    def on_tracking_branch(self) -> bool:
        """True when the checked-out branch is the one being compared."""
        return not self.head_branch or self.head_branch == self.branch

    def summary(self) -> str:
        """A short, plain-language line for the terminal and the GUI bubble."""
        if self.applied:
            return (
                f"Updated to {self.latest_revision} "
                f"({len(self.changed_files)} file(s) changed)."
            )
        if self.blocked:
            return "Cannot check for updates here. See the note below."
        if self.latest_revision and not self.available:
            return f"Already up to date at {self.current_revision} ({self.current_version})."
        if not self.available:
            return "No update information is available right now."
        if not self.on_tracking_branch:
            return (
                f"Update available on {self.branch} ({self.behind} commit(s) behind), "
                f"but this checkout is on {self.head_branch}."
            )
        if self.dirty:
            return (
                "An update is available but this checkout has local changes. "
                "Commit or stash them first."
            )
        return f"Update available: {self.behind} commit(s) behind {self.repository_url}."


def _git_available() -> bool:
    return shutil.which("git") is not None


def _run_git(working_dir: Path, *args: str) -> GitCommand:
    """Run one git command in ``working_dir`` and capture its output."""
    argv = ("git", *args)
    try:
        completed = subprocess.run(
            argv,
            cwd=str(working_dir),
            capture_output=True,
            text=True,
            timeout=_GIT_TIMEOUT,
            env=_noninteractive_env(),
        )
    except FileNotFoundError as error:
        raise UpdateError("git is not installed or not on the PATH.") from error
    except subprocess.TimeoutExpired as error:
        raise UpdateError(f"`{' '.join(argv)}` timed out.") from error
    return GitCommand(
        argv=argv,
        returncode=completed.returncode,
        stdout=completed.stdout.strip(),
        stderr=completed.stderr.strip(),
    )


def _noninteractive_env() -> dict[str, str]:
    """An environment where git never blocks on a password or host-key prompt."""
    env = dict(os.environ)
    env.setdefault("GIT_TERMINAL_PROMPT", "0")
    env.setdefault("GIT_SSH_COMMAND", "ssh -o BatchMode=yes")
    return env


def sanitize_remote_url(url: str) -> str:
    """Strip any embedded credentials so a token is never printed or logged.

    A remote such as ``https://user:token@github.com/org/repo.git`` is reported
    as ``https://github.com/org/repo.git``.
    """
    from urllib.parse import urlsplit, urlunsplit

    try:
        parts = urlsplit(url)
    except ValueError:
        return url
    if "@" not in parts.netloc:
        return url
    host = parts.netloc.rsplit("@", 1)[-1]
    return urlunsplit((parts.scheme, host, parts.path, parts.query, parts.fragment))


def _repo_root() -> Path:
    """The directory that holds the ``.git`` folder, or the project root."""
    if is_frozen():
        return executable_dir()
    return Path(__file__).resolve().parents[1]


def is_source_checkout(root: Path | None = None) -> bool:
    """True when ``root`` is a git working tree we could update in place."""
    root = root or _repo_root()
    return (root / ".git").exists() and not is_frozen()


#: Plain-language explanation for the portable / extracted-zip case.
PORTABLE_REASON = (
    "This folder is not a git clone, so it cannot self-update. It was probably "
    "downloaded as a ZIP and extracted. Install Git and run `git clone "
    "https://github.com/pt5ri-debug/fayd-al-ruh.git` once (the repository is "
    "private, so git will ask you to sign in), copy your data/ folder into the "
    "clone, and update from there."
)

#: Plain-language explanation for a source folder where git is missing.
NO_GIT_REASON = (
    "git was not found on the PATH, so the update cannot be checked. Install Git "
    "from https://git-scm.com/download/win and reopen the program."
)


def _parse_ahead_behind(counts: str) -> tuple[int, int]:
    """Parse ``git rev-list --left-right --count`` into (behind, ahead)."""
    parts = counts.split()
    if len(parts) != 2:
        return 0, 0
    ahead, behind = (int(value) for value in parts)
    return behind, ahead


def check_for_updates(repository_url: str = DEFAULT_REPOSITORY_URL, *, root: Path | None = None) -> UpdateStatus:
    """Fetch remote metadata and report whether an update is available.

    This never modifies the working tree. It fetches the tracked branch from
    ``origin`` (falling back to ``repository_url`` when there is no remote) and
    compares commits, so the answer is exact rather than a version-string guess.
    """
    root = root or _repo_root()
    commands: list[GitCommand] = []

    # Explain exactly why a check is impossible, rather than a vague "portable".
    if is_frozen() or not (root / ".git").exists():
        return UpdateStatus(
            available=False,
            current_version=__version__,
            current_revision="",
            latest_revision="",
            branch="",
            behind=0,
            ahead=0,
            dirty=False,
            repository_url=repository_url,
            reason=PORTABLE_REASON,
            commands=commands,
            blocked=True,
        )
    if not _git_available():
        return UpdateStatus(
            available=False,
            current_version=__version__,
            current_revision="",
            latest_revision="",
            branch="",
            behind=0,
            ahead=0,
            dirty=False,
            repository_url=repository_url,
            reason=NO_GIT_REASON,
            commands=commands,
            blocked=True,
        )

    head = _run_git(root, "rev-parse", "--abbrev-ref", "HEAD")
    commands.append(head)
    if not head.ok:
        raise UpdateError("Could not read the current branch.")
    head_branch = head.stdout

    remote = _run_git(root, "remote", "get-url", "origin")
    commands.append(remote)
    origin_url = sanitize_remote_url(remote.stdout) if remote.ok else repository_url

    dirty = bool(_run_git(root, "status", "--porcelain").stdout)

    fetch = _run_git(root, "fetch", "--prune", "origin")
    commands.append(fetch)
    if not fetch.ok:
        raise UpdateError(_fetch_failure(fetch, origin_url))

    # Prefer the checked-out branch so an update fast-forwards what the user is
    # running; fall back to the remote's default branch when the local branch has
    # no counterpart on the remote yet.
    branch, branch_error = _tracking_branch(root, head_branch, commands)
    if branch_error is not None:
        raise UpdateError(branch_error)

    current = _run_git(root, "rev-parse", "--short", f"origin/{branch}")
    commands.append(current)
    if not current.ok:
        raise UpdateError(f"`origin/{branch}` does not exist on the remote.")

    counts = _run_git(root, "rev-list", "--left-right", "--count", f"HEAD...origin/{branch}")
    commands.append(counts)
    behind, ahead = _parse_ahead_behind(counts.stdout) if counts.ok else (0, 0)

    changed = _run_git(root, "diff", "--name-only", f"HEAD..origin/{branch}")
    commands.append(changed)
    changed_files = tuple(line for line in changed.stdout.splitlines() if line.strip())

    return UpdateStatus(
        available=behind > 0,
        current_version=__version__,
        current_revision=_run_git(root, "rev-parse", "--short", "HEAD").stdout,
        latest_revision=current.stdout,
        branch=branch,
        behind=behind,
        ahead=ahead,
        dirty=dirty,
        repository_url=origin_url,
        reason="" if behind > 0 else "The checkout already matches the remote branch.",
        head_branch=head_branch,
        changed_files=changed_files,
        commands=commands,
    )


def _tracking_branch(
    root: Path, head_branch: str, commands: list[GitCommand]
) -> tuple[str, str | None]:
    """The branch to compare against ``origin``, plus an error message if none.

    Uses the current branch when the remote tracks it; otherwise the remote's
    default branch, so a checkout on a local-only branch can still learn what the
    published default contains instead of failing outright.
    """
    exists = _run_git(root, "rev-parse", "--verify", "--quiet", f"refs/remotes/origin/{head_branch}")
    commands.append(exists)
    if exists.ok:
        return head_branch, None

    symbolic = _run_git(root, "symbolic-ref", "-q", "--short", "refs/remotes/origin/HEAD")
    commands.append(symbolic)
    if symbolic.ok and symbolic.stdout:
        return symbolic.stdout.rsplit("/", 1)[-1], None

    for candidate in ("main", "master"):
        remote_ref = _run_git(root, "rev-parse", "--verify", "--quiet", f"refs/remotes/origin/{candidate}")
        commands.append(remote_ref)
        if remote_ref.ok:
            return candidate, None
    return head_branch, (
        "No matching branch on the remote; set an upstream with "
        "`git branch --set-upstream-to=origin/main`."
    )


def _scrub_credentials(text: str) -> str:
    """Remove any ``scheme://user:secret@host`` credentials from free text."""
    import re

    return re.sub(r"(?P<scheme>[a-zA-Z][a-zA-Z0-9+.\-]*://)[^/@\s]+@", r"\g<scheme>", text)


def _fetch_failure(fetch: GitCommand, url: str) -> str:
    """Turn a failed fetch into an actionable message without leaking secrets."""
    detail = fetch.stderr or fetch.stdout or "unknown error"
    detail = _scrub_credentials(detail.replace(url, "<remote>"))
    if "Authentication" in detail or "could not read Username" in detail or "403" in detail:
        return (
            "Could not reach the project on GitHub: authentication failed. "
            "Set a valid token or use an HTTPS remote with credentials."
        )
    if "Could not resolve host" in detail or "unable to access" in detail.lower():
        return f"Could not reach the project on GitHub: {detail}"
    return f"Could not fetch updates from the project on GitHub: {detail}"


def backup_data_dir(root: Path) -> tuple[str, int, str]:
    """Copy the git-ignored data folder (trained state) to a timestamped backup.

    Returns ``(path, file_count, error)``. A missing data folder is not an
    error: it simply means there is no learned state to protect yet.
    """
    data = root / "data"
    if not data.is_dir():
        return "", 0, ""
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    target = root / f"data-backup-{stamp}"
    try:
        shutil.copytree(data, target)
    except OSError as exc:
        return "", 0, _scrub_credentials(str(exc))
    copied = sum(1 for path in target.rglob("*") if path.is_file())
    return str(target), copied, ""


def apply_update(
    repository_url: str = DEFAULT_REPOSITORY_URL,
    *,
    root: Path | None = None,
    status: UpdateStatus | None = None,
    backup: bool = True,
) -> UpdateStatus:
    """Fast-forward the current branch to the remote. Refuses unsafe merges.

    The update is applied only when the tree is clean and the branch can move
    forward without a merge (behind but not ahead). Anything else is reported
    rather than forced, so local work is never destroyed. Unless ``backup`` is
    false, the trained data folder is copied to a timestamped ``data-backup-*``
    beside the checkout first.
    """
    root = root or _repo_root()
    status = status or check_for_updates(repository_url, root=root)
    commands = list(status.commands or [])

    if is_frozen() or not is_source_checkout(root):
        status.reason = PORTABLE_REASON
        status.blocked = True
        return status
    if not status.available:
        return status
    if not status.on_tracking_branch:
        status.reason = (
            f"This checkout is on {status.head_branch}; check out {status.branch} to update it."
        )
        return status
    if status.dirty:
        status.reason = "Local changes would be overwritten; commit or stash them first."
        return status
    if status.ahead > 0:
        status.reason = (
            f"The branch has {status.ahead} local commit(s) not on the remote; "
            "refusing a non-fast-forward update."
        )
        return status

    # Protect the user's learned state before touching anything.
    if backup:
        status.backup_path, status.backup_copied, status.backup_error = backup_data_dir(root)

    merge = _run_git(root, "merge", "--ff-only", f"origin/{status.branch}")
    commands.append(merge)
    status.commands = commands
    if not merge.ok:
        status.reason = merge.stderr or "The fast-forward merge failed."
        return status

    status.applied = True
    status.available = False
    status.current_revision = _run_git(root, "rev-parse", "--short", "HEAD").stdout
    status.changed_files = tuple(status.changed_files)
    status.reason = "Fast-forwarded to the latest commit."
    return status


def _wrap(text: str, width: int = 76, indent: str = "    ") -> str:
    """Wrap a long note into indented lines, preserving any existing newlines."""
    import textwrap

    lines: list[str] = []
    for paragraph in text.splitlines() or [""]:
        wrapped = textwrap.wrap(paragraph, width=width - len(indent)) or [""]
        lines.extend(f"{indent}{line}" for line in wrapped)
    return "\n".join(lines)


def format_status(status: UpdateStatus, *, verbose: bool = False) -> str:
    """A multi-line human report, optionally including the git commands run."""
    if status.blocked:
        lines = [
            "Program update check",
            f"  repository     : {sanitize_remote_url(status.repository_url)}",
            f"  installed      : {status.current_version}",
            "  state          : cannot check here",
            "  reason         :",
            _wrap(status.reason),
        ]
        return "\n".join(lines)

    lines = [
        "Program update check",
        f"  repository     : {sanitize_remote_url(status.repository_url)}",
        f"  installed      : {status.current_version} ({status.current_revision or 'unknown'})",
        f"  latest         : {status.latest_revision or 'unknown'}"
        + (f" on {status.branch}" if status.branch else ""),
        f"  state          : {'update available' if status.available else 'up to date'}",
    ]
    if status.head_branch and status.head_branch != status.branch:
        lines.append(f"  checked out    : {status.head_branch} (not the compared branch)")
    if status.behind:
        lines.append(f"  commits behind : {status.behind}")
    if status.ahead:
        lines.append(f"  commits ahead  : {status.ahead}")
    if status.changed_files:
        lines.append(f"  files changed  : {len(status.changed_files)}")
        preview = status.changed_files[:10]
        lines.extend(f"    - {name}" for name in preview)
        if len(status.changed_files) > len(preview):
            lines.append(f"    … and {len(status.changed_files) - len(preview)} more")
    if status.dirty:
        lines.append("  local changes  : present (an update would refuse to overwrite them)")
    learned = status.learned_files_changed
    if learned:
        lines.append(
            f"  training files : {len(learned)} bundled training file(s) change in the repo copy"
        )
        lines.append("                   (your own data lives in data/ and is untouched)")
    if status.backup_path:
        lines.append(f"  data backup    : {status.backup_path} ({status.backup_copied} file(s))")
    elif status.backup_error:
        lines.append(f"  data backup    : not created - {status.backup_error}")
    if status.reason:
        lines.append(f"  note           : {status.reason}")
    if verbose and status.commands:
        lines.append("  git commands   :")
        for command in status.commands:
            lines.append(f"    $ {' '.join(command.argv)}  (exit {command.returncode})")
    return "\n".join(lines)


__all__ = [
    "DEFAULT_REPOSITORY_URL",
    "GitCommand",
    "UpdateError",
    "UpdateStatus",
    "apply_update",
    "backup_data_dir",
    "check_for_updates",
    "format_status",
    "is_source_checkout",
    "sanitize_remote_url",
]
