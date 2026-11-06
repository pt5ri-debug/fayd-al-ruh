"""The Code Studio's command engine: run the project's own development tasks.

A developer working in the Studio needs more than "run the tests": lint, a type
check, the evaluation gate, a formatter check, a build, or a one-off command of
their own. This module turns each of those into a named, described
:class:`CommandSpec` and runs it with **live output**, so a long run shows its
progress instead of freezing the window.

Two honesty rules shape it:

* **A command is only offered when it makes sense here.** :func:`detect_project`
  reads the checkout (``pyproject.toml``, ``package.json``, ``Cargo.toml``,
  ``go.mod``, ...) and :func:`catalog_for` returns the commands whose tool the
  project actually declares -- the project's own ``pytest``/``ruff``/``mypy``/
  ``evaluation`` for this repository, and the tool's own command for another kind
  of checkout. A command whose tool is absent is not silently dropped from the
  catalog; it is reported with the next step when it fails to start.
* **Nothing here is a shell.** A spec is an argv tuple, run with
  ``subprocess.Popen`` and **no** ``shell=True``, so a task can never be turned
  into an injection by a value from the UI. A user's own command is split with
  :func:`shlex.split` and validated before it runs.

Output is streamed line by line to a callback (the window appends it to a live
pane; the CLI prints it), and a run can be cancelled, which terminates the
process group so a child (a pytest worker, a node process) cannot be left behind.
"""

from __future__ import annotations

import os
import shlex
import signal
import subprocess
import threading
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path

#: How long a task may run before it is stopped. The evaluation gate and a full
#: test suite are the long ones; a lint is seconds.
DEFAULT_TIMEOUT = 900.0

#: How much of a task's output is kept for the record. The window shows the live
#: stream; this is what is stored on the outcome for a later review.
DEFAULT_MAX_OUTPUT = 400_000

#: Categories, in the order a menu shows them.
CATEGORY_TEST = "Test"
CATEGORY_QUALITY = "Quality"
CATEGORY_BUILD = "Build"
CATEGORY_RUN = "Run"
CATEGORY_CUSTOM = "Custom"

CATEGORY_ORDER: tuple[str, ...] = (
    CATEGORY_TEST,
    CATEGORY_QUALITY,
    CATEGORY_BUILD,
    CATEGORY_RUN,
    CATEGORY_CUSTOM,
)


class TaskError(RuntimeError):
    """A command could not be run as asked (a bad spec, or a tool that is missing)."""


@dataclass(frozen=True)
class CommandSpec:
    """One named development command the Studio can run.

    ``streams`` marks a command whose output is useful live (a test run, a build);
    ``cancellable`` marks one that may run long enough to need a stop button.
    """

    key: str
    title: str
    argv: tuple[str, ...]
    summary: str
    category: str = CATEGORY_QUALITY
    timeout: float = DEFAULT_TIMEOUT
    streams: bool = True
    cancellable: bool = True
    long_running: bool = False

    def command_line(self) -> str:
        return " ".join(shlex.quote(part) for part in self.argv)

    def describe(self) -> str:
        return f"{self.title}  —  {self.summary}\n    $ {self.command_line()}"


@dataclass
class CommandOutcome:
    """The result of one task run, with its own output and timing."""

    spec: CommandSpec
    ok: bool
    returncode: int
    output: str = ""
    started_at: float = 0.0
    finished_at: float = 0.0
    timed_out: bool = False
    cancelled: bool = False
    error: str = ""

    @property
    def seconds(self) -> float:
        if not self.started_at:
            return 0.0
        end = self.finished_at or time.time()
        return max(0.0, end - self.started_at)

    def summary(self) -> str:
        if self.error:
            return f"{self.spec.title}: {self.error}"
        if self.cancelled:
            return f"{self.spec.title} was stopped."
        if self.timed_out:
            return f"{self.spec.title} timed out after {self.spec.timeout:.0f}s."
        state = "passed" if self.ok else f"failed (exit {self.returncode})"
        return f"{self.spec.title} {state} in {self.seconds:.1f}s."

    def tail(self, *, limit: int = 6000) -> str:
        """The end of the output, where a failure usually is."""
        text = self.output.strip()
        if len(text) <= limit:
            return text
        return "…\n" + text[-limit:]


@dataclass
class ProjectProfile:
    """What kind of checkout this is, and the tool commands that fit it."""

    kind: str
    markers: tuple[str, ...] = ()
    notes: tuple[str, ...] = field(default_factory=tuple)


#: Markers checked, in order, to name a project's kind.
_MARKERS: tuple[tuple[str, str], ...] = (
    ("pyproject.toml", "python"),
    ("setup.py", "python"),
    ("package.json", "node"),
    ("Cargo.toml", "rust"),
    ("go.mod", "go"),
    ("pom.xml", "java"),
    ("build.gradle", "java"),
    ("Gemfile", "ruby"),
    ("composer.json", "php"),
    ("Makefile", "make"),
)


def detect_project(root: Path | str) -> ProjectProfile:
    """Name the checkout's kind from the files it declares.

    The order matters: a repository that holds both a ``pyproject.toml`` and a
    ``package.json`` (a Python project with a JS front end) is reported as Python
    first, because that is the language its own tests are written in.
    """
    base = Path(root)
    found: list[str] = []
    kind = "unknown"
    for marker, name in _MARKERS:
        if (base / marker).exists():
            found.append(marker)
            if kind == "unknown":
                kind = name
    return ProjectProfile(kind=kind, markers=tuple(found))


def catalog_for(root: Path | str, *, check_command: Sequence[str] | None = None) -> tuple[CommandSpec, ...]:
    """The task catalog that fits this checkout, in the order a menu shows it.

    A Python checkout gets the project's own quality gates (tests, lint, type
    check, evaluation). Another kind of checkout gets its own tool's commands.
    ``check_command`` is the same command the Studio's merge gate uses, so the
    catalog's *Tests* entry and the merge gate can never disagree.
    """
    profile = detect_project(root)
    specs: list[CommandSpec] = []

    tests = tuple(check_command) if check_command else ("python", "-m", "pytest", "-q")
    specs.append(
        CommandSpec(
            key="tests",
            title="Run the tests",
            argv=tests,
            summary="The project's own test command (the same one the merge gate runs).",
            category=CATEGORY_TEST,
            long_running=True,
        )
    )

    if profile.kind == "python":
        specs.extend(_python_specs())
    elif profile.kind == "node":
        specs.extend(_node_specs())
    elif profile.kind == "rust":
        specs.extend(_rust_specs())
    elif profile.kind == "go":
        specs.extend(_go_specs())

    if "Makefile" in profile.markers:
        specs.append(
            CommandSpec(
                key="make",
                title="Run `make`",
                argv=("make",),
                summary="The project's own Makefile default target.",
                category=CATEGORY_BUILD,
                long_running=True,
            )
        )
    return tuple(specs)


def _python_specs() -> list[CommandSpec]:
    return [
        CommandSpec(
            key="lint",
            title="Lint (ruff)",
            argv=("python", "-m", "ruff", "check", "."),
            summary="Static checks with Ruff, exactly as CI runs them.",
            category=CATEGORY_QUALITY,
        ),
        CommandSpec(
            key="format-check",
            title="Check formatting (ruff format --check)",
            argv=("python", "-m", "ruff", "format", "--check", "."),
            summary="Report files that would be reformatted, without changing them.",
            category=CATEGORY_QUALITY,
        ),
        CommandSpec(
            key="format",
            title="Format the code (ruff format)",
            argv=("python", "-m", "ruff", "format", "."),
            summary="Reformat the code in place (a real change to the workspace).",
            category=CATEGORY_QUALITY,
        ),
        CommandSpec(
            key="types",
            title="Type check (mypy)",
            argv=("python", "-m", "mypy"),
            summary="The project's type check, configured in pyproject.toml.",
            category=CATEGORY_QUALITY,
            long_running=True,
        ),
        CommandSpec(
            key="evaluation",
            title="Run the evaluation gate",
            argv=("python", "-m", "evaluation"),
            summary="The project's regression gate; it must print PASS.",
            category=CATEGORY_TEST,
            long_running=True,
        ),
        CommandSpec(
            key="build",
            title="Build the package",
            argv=("python", "-m", "build"),
            summary="Build the wheel and sdist (needs the `build` package).",
            category=CATEGORY_BUILD,
            long_running=True,
        ),
    ]


def _node_specs() -> list[CommandSpec]:
    return [
        CommandSpec(
            key="install",
            title="Install dependencies",
            argv=("npm", "install"),
            summary="Install the project's npm dependencies (writes node_modules).",
            category=CATEGORY_BUILD,
            long_running=True,
        ),
        CommandSpec(
            key="test",
            title="Run npm test",
            argv=("npm", "test"),
            summary="The package's own test script.",
            category=CATEGORY_TEST,
            long_running=True,
        ),
        CommandSpec(
            key="lint",
            title="Run npm run lint",
            argv=("npm", "run", "lint"),
            summary="The package's own lint script (when it declares one).",
            category=CATEGORY_QUALITY,
        ),
        CommandSpec(
            key="build",
            title="Run npm run build",
            argv=("npm", "run", "build"),
            summary="The package's own build script (when it declares one).",
            category=CATEGORY_BUILD,
            long_running=True,
        ),
    ]


def _rust_specs() -> list[CommandSpec]:
    return [
        CommandSpec("test", "cargo test", ("cargo", "test"), "Rust's own test runner.", CATEGORY_TEST, long_running=True),
        CommandSpec("check", "cargo check", ("cargo", "check"), "Compile without producing a binary.", CATEGORY_QUALITY),
        CommandSpec("clippy", "cargo clippy", ("cargo", "clippy"), "Rust's linter.", CATEGORY_QUALITY),
        CommandSpec("fmt", "cargo fmt --check", ("cargo", "fmt", "--check"), "Report formatting drift.", CATEGORY_QUALITY),
        CommandSpec("build", "cargo build", ("cargo", "build"), "Build the crate.", CATEGORY_BUILD, long_running=True),
    ]


def _go_specs() -> list[CommandSpec]:
    return [
        CommandSpec("test", "go test ./...", ("go", "test", "./..."), "Go's own test runner.", CATEGORY_TEST, long_running=True),
        CommandSpec("vet", "go vet ./...", ("go", "vet", "./..."), "Go's own static check.", CATEGORY_QUALITY),
        CommandSpec("build", "go build ./...", ("go", "build", "./..."), "Compile every package.", CATEGORY_BUILD, long_running=True),
    ]


def find_spec(root: Path | str, key: str, *, check_command: Sequence[str] | None = None) -> CommandSpec | None:
    """The catalog entry with this key, or ``None``."""
    for spec in catalog_for(root, check_command=check_command):
        if spec.key == key:
            return spec
    return None


def custom_spec(command_line: str, *, title: str = "") -> CommandSpec:
    """Turn a user's own command line into a spec, or raise on an empty one.

    The string is split with :func:`shlex.split` (never handed to a shell), so a
    value with ``;`` or ``&&`` is one argument rather than a second command.
    """
    parts = shlex.split(command_line.strip())
    if not parts:
        raise TaskError("The command cannot be empty.")
    return CommandSpec(
        key="custom",
        title=title.strip() or "Custom command",
        argv=tuple(parts),
        summary=f"A command of your own: {command_line.strip()}",
        category=CATEGORY_CUSTOM,
        long_running=True,
    )


class CommandRunner:
    """Runs :class:`CommandSpec` commands with live output and cancellation."""

    def __init__(self, *, max_output: int = DEFAULT_MAX_OUTPUT) -> None:
        self.max_output = max_output
        self._process: subprocess.Popen[str] | None = None
        self._lock = threading.Lock()
        self._cancelled = False

    def cancel(self) -> None:
        """Ask the running command to stop (terminates the whole process group)."""
        with self._lock:
            self._cancelled = True
            process = self._process
        if process is not None and process.poll() is None:
            _terminate_group(process)

    @property
    def running(self) -> bool:
        with self._lock:
            return self._process is not None and self._process.poll() is None

    def run(
        self,
        spec: CommandSpec,
        *,
        cwd: Path | str,
        on_line: Callable[[str], None] | None = None,
        env: dict[str, str] | None = None,
        timeout: float | None = None,
    ) -> CommandOutcome:
        """Run ``spec`` in ``cwd``, streaming each output line to ``on_line``."""
        with self._lock:
            self._cancelled = False
        started = time.time()
        limit = timeout if timeout is not None else spec.timeout
        try:
            process = subprocess.Popen(
                spec.argv,
                cwd=str(cwd),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                env={**os.environ, **(env or {})},
                start_new_session=True,  # so the whole group can be terminated
            )
        except FileNotFoundError:
            return CommandOutcome(
                spec=spec,
                ok=False,
                returncode=127,
                error=f"`{spec.argv[0]}` is not installed or not on the PATH.",
                started_at=started,
                finished_at=time.time(),
            )
        except OSError as error:
            return CommandOutcome(
                spec=spec,
                ok=False,
                returncode=126,
                error=str(error),
                started_at=started,
                finished_at=time.time(),
            )

        with self._lock:
            self._process = process
        chunks: list[str] = []
        timed_out = False
        try:
            assert process.stdout is not None
            for line in process.stdout:
                chunks.append(line)
                if on_line is not None:
                    on_line(line.rstrip("\n"))
                if sum(len(c) for c in chunks) > self.max_output:
                    chunks.append("… [output truncated]\n")
                    break
                if time.time() - started > limit:
                    timed_out = True
                    _terminate_group(process)
                    break
            process.wait(timeout=max(1.0, limit))
        except subprocess.TimeoutExpired:
            timed_out = True
            _terminate_group(process)
        finally:
            with self._lock:
                cancelled = self._cancelled
                self._process = None
            try:
                if process.stdout is not None:
                    process.stdout.close()
            except OSError:
                pass

        returncode = process.returncode if process.returncode is not None else -1
        cancelled = cancelled and not timed_out
        return CommandOutcome(
            spec=spec,
            ok=returncode == 0 and not timed_out and not cancelled,
            returncode=returncode,
            output="".join(chunks),
            started_at=started,
            finished_at=time.time(),
            timed_out=timed_out,
            cancelled=cancelled,
        )


def _terminate_group(process: subprocess.Popen[str]) -> None:
    """Stop a process and its children, without leaving a worker behind."""
    if process.poll() is not None:
        return
    try:
        os.killpg(os.getpgid(process.pid), signal.SIGTERM)
    except (ProcessLookupError, PermissionError, OSError):
        try:
            process.terminate()
        except OSError:
            return
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(os.getpgid(process.pid), signal.SIGKILL)
        except (ProcessLookupError, PermissionError, OSError):
            try:
                process.kill()
            except OSError:
                pass


__all__ = [
    "CATEGORY_BUILD",
    "CATEGORY_CUSTOM",
    "CATEGORY_ORDER",
    "CATEGORY_QUALITY",
    "CATEGORY_RUN",
    "CATEGORY_TEST",
    "DEFAULT_TIMEOUT",
    "CommandOutcome",
    "CommandRunner",
    "CommandSpec",
    "ProjectProfile",
    "TaskError",
    "catalog_for",
    "custom_spec",
    "detect_project",
    "find_spec",
]
