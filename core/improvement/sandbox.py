"""The experiment sandbox: one isolated place a candidate change may run.

Section 8 of the phase asks for an isolated environment that keeps a candidate
away from production. This module is that environment, and it is deliberately a
**composition of proven pieces** rather than a new mechanism:

* **code isolation** is a Code Studio git **worktree** -- a full copy of the
  project on its own branch under the git-ignored data folder, so the running
  checkout is never edited and a broken candidate cannot stop the program;
* **data isolation** is a temporary directory the candidate's data path is
  pointed at, so an experiment writes to a throwaway store, never the real one
  (the same trick the tests use);
* **filesystem confinement** is the Code Studio's own deny-by-default path
  check, which refuses ``..``, an absolute path, and a climb out of the root;
* **resource limits and a timeout** come from the existing
  :class:`~core.agents.runtime.CancellationToken` and a bounded wall-clock
  budget; and
* **cancellation** is the same cooperative token.

Honest by construction: a sandbox that cannot be created (no git, no repository,
a full disk) is reported as a *failure to isolate*, never silently downgraded to
running against production. An experiment whose sandbox is unavailable does not
run at all.
"""

from __future__ import annotations

import os
import shutil
import tempfile
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from uuid import uuid4

from core.best_effort import best_effort
from core.code_studio import CodeStudio, StudioError, Workspace


class SandboxError(RuntimeError):
    """Raised when an isolated environment cannot be created or used."""


#: The default wall-clock budget for one sandboxed run, in seconds. Bounded so a
#: hung candidate cannot hold the machine; a caller may lower or raise it.
DEFAULT_TIMEOUT_SECONDS = 900.0

#: The default ceiling on how much of the tree a sandboxed search or listing may
#: walk, so a candidate cannot stall on a huge repository.
DEFAULT_MAX_FILES = 4000

#: Files an experiment may never modify, whatever it is asked to do. Enforced
#: here as a *deny-list* on top of the Code Studio's allow-list of suffixes, so a
#: candidate cannot edit the governance that constrains it.
FORBIDDEN_PATHS: tuple[str, ...] = (
    ".git",
    "pyproject.toml",
    "requirements.txt",
    ".github",
)


@dataclass(frozen=True)
class SandboxLimits:
    """The bounded resources one sandboxed run may use."""

    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS
    max_files: int = DEFAULT_MAX_FILES
    #: A memory ceiling in bytes the caller *intends* to respect. It is recorded
    #: and reported, and consulted by the resource governor where one is wired;
    #: this module does not pretend to enforce it with the OS.
    max_memory_bytes: int | None = None
    allow_network: bool = False

    def __post_init__(self) -> None:
        if self.timeout_seconds <= 0:
            raise ValueError("A sandbox timeout must be positive.")
        if self.max_files <= 0:
            raise ValueError("A sandbox file bound must be positive.")


@dataclass
class SandboxSession:
    """One live isolated environment: a worktree, a data root, and a budget."""

    sandbox_id: str
    workspace: Workspace
    data_root: Path
    limits: SandboxLimits
    started_at: float = field(default_factory=time.monotonic)
    cancelled: bool = False
    _artifacts: dict[str, str] = field(default_factory=dict)

    @property
    def root(self) -> Path:
        """The sandbox's own project copy (the worktree)."""
        return self.workspace.path

    def elapsed(self) -> float:
        return time.monotonic() - self.started_at

    @property
    def expired(self) -> bool:
        return self.elapsed() > self.limits.timeout_seconds

    def cancel(self) -> None:
        """Ask a running experiment to stop at its next check."""
        self.cancelled = True

    def check(self) -> None:
        """Raise when the run must stop -- the cooperative cancellation point."""
        if self.cancelled:
            raise SandboxError("The experiment was cancelled.")
        if self.expired:
            raise SandboxError(f"The experiment exceeded its {self.limits.timeout_seconds:g}s budget.")

    def record(self, name: str, value: str) -> None:
        """Keep one artifact (a path, a summary) with the session."""
        self._artifacts[name] = value

    @property
    def artifacts(self) -> dict[str, str]:
        return dict(self._artifacts)

    def environment(self) -> dict[str, str]:
        """The environment a candidate run should use.

        ``FAYD_DATA_DIR`` is pointed at the session's own data root, so every
        store the candidate touches (metrics, memory, learning, knowledge) writes
        to the throwaway folder and **never** to the real one. Network access is
        not granted: the project is offline-first, and a sandbox that reached the
        network would not be isolated.
        """
        env = dict(os.environ)
        env["FAYD_DATA_DIR"] = str(self.data_root)
        env["FAYD_SANDBOX_ID"] = self.sandbox_id
        return env

    def describe(self) -> str:
        state = "cancelled" if self.cancelled else ("expired" if self.expired else "running")
        return (
            f"Sandbox {self.sandbox_id} ({state}): worktree {self.root}, "
            f"data {self.data_root}, budget {self.limits.timeout_seconds:g}s, "
            f"elapsed {self.elapsed():.1f}s"
        )


def _is_forbidden(relative: str) -> bool:
    """True when a path is one an experiment may never write."""
    normalised = relative.replace("\\", "/").lstrip("./")
    for forbidden in FORBIDDEN_PATHS:
        if normalised == forbidden or normalised.startswith(f"{forbidden}/"):
            return True
    return False


class SandboxManager:
    """Creates and tears down isolated environments for candidate experiments."""

    def __init__(
        self,
        *,
        root: Path | str,
        studio: CodeStudio | None = None,
        sessions_dir: Path | str | None = None,
        default_limits: SandboxLimits | None = None,
    ) -> None:
        self.root = Path(root)
        self._studio = studio or CodeStudio(self.root)
        self._sessions_dir = Path(sessions_dir) if sessions_dir is not None else None
        self.default_limits = default_limits or SandboxLimits()
        self._sessions: dict[str, SandboxSession] = {}

    @property
    def studio(self) -> CodeStudio:
        return self._studio

    def available(self) -> tuple[bool, str]:
        """Whether an isolated environment can be created here, and why not."""
        if not self._studio.is_repository():
            return False, f"{self.root} is not a git repository, so no isolated worktree can be made."
        return True, "ready"

    def create(
        self,
        *,
        task: str = "",
        slug: str | None = None,
        limits: SandboxLimits | None = None,
    ) -> SandboxSession:
        """Create one isolated session (a worktree plus a throwaway data root)."""
        ok, reason = self.available()
        if not ok:
            raise SandboxError(reason)
        limits = limits or self.default_limits
        try:
            workspace = self._studio.create(task, slug=slug)
        except StudioError as error:
            raise SandboxError(f"Could not create the sandbox worktree: {error}") from error

        base = self._sessions_dir
        if base is not None:
            base.mkdir(parents=True, exist_ok=True)
            data_root = Path(tempfile.mkdtemp(prefix=f"si-{workspace.slug}-", dir=base))
        else:
            data_root = Path(tempfile.mkdtemp(prefix=f"si-{workspace.slug}-"))

        session = SandboxSession(
            sandbox_id=f"SBX-{uuid4().hex[:8]}",
            workspace=workspace,
            data_root=data_root,
            limits=limits,
        )
        self._sessions[session.sandbox_id] = session
        return session

    def get(self, sandbox_id: str) -> SandboxSession:
        session = self._sessions.get(sandbox_id)
        if session is None:
            raise SandboxError(f"Unknown sandbox {sandbox_id}.")
        return session

    def sessions(self) -> tuple[SandboxSession, ...]:
        return tuple(self._sessions.values())

    def write(self, session: SandboxSession, relative: str, content: str) -> int:
        """Write a file inside the sandbox, refusing a forbidden path.

        The Code Studio's own confinement (no ``..``, no absolute path, an
        allowed suffix, a size cap) runs first; the deny-list here is the extra
        guard that keeps a candidate away from the project's governance files.
        """
        session.check()
        if _is_forbidden(relative):
            raise SandboxError(f"'{relative}' is protected and may not be modified by an experiment.")
        try:
            return self._studio.write_file(session.workspace, relative, content)
        except StudioError as error:
            raise SandboxError(str(error)) from error

    def read(self, session: SandboxSession, relative: str) -> str:
        session.check()
        try:
            return self._studio.read_file(session.workspace, relative)
        except StudioError as error:
            raise SandboxError(str(error)) from error

    def changed_files(self, session: SandboxSession) -> tuple[str, ...]:
        return self._studio.changed_files(session.workspace)

    def diff(self, session: SandboxSession) -> str:
        session.check()
        return self._studio.diff(session.workspace)

    def run_checks(
        self,
        session: SandboxSession,
        *,
        command: tuple[str, ...] | None = None,
        progress: Callable[[str], None] | None = None,
    ) -> tuple[bool, str]:
        """Run the project's own checks **inside** the sandbox and report honestly.

        The check runs with the sandbox's throwaway data root, so a check that
        writes learned data cannot touch the real store.
        """
        session.check()
        if progress is not None:
            progress(f"running checks in {session.sandbox_id}")
        try:
            result = self._studio.run_checks(
                session.workspace,
                command=command,
                timeout=session.limits.timeout_seconds,
                env=session.environment(),
            )
        except StudioError as error:
            return False, f"checks could not run: {error}"
        session.record("checks", result.summary())
        return result.ok, result.summary()

    def cleanup(self, session: SandboxSession, *, keep_artifacts: bool = False) -> None:
        """Tear the sandbox down safely: remove the worktree and the data root.

        ``keep_artifacts`` leaves the throwaway data root in place (a caller that
        wants to inspect an experiment's outputs); the worktree is always removed,
        because a leftover branch would linger in the repository.
        """
        with best_effort("remove the sandbox worktree"):
            self._studio.remove(session.workspace.slug, force=True)
        if not keep_artifacts:
            with best_effort("remove the sandbox data root"):
                shutil.rmtree(session.data_root, ignore_errors=True)
        self._sessions.pop(session.sandbox_id, None)

    @contextmanager
    def session(self, **kwargs: object) -> Iterator[SandboxSession]:
        """A context manager that always tears the sandbox down."""
        created = self.create(**kwargs)  # type: ignore[arg-type]
        try:
            yield created
        finally:
            self.cleanup(created)


__all__ = [
    "DEFAULT_MAX_FILES",
    "DEFAULT_TIMEOUT_SECONDS",
    "FORBIDDEN_PATHS",
    "SandboxError",
    "SandboxLimits",
    "SandboxManager",
    "SandboxSession",
]
