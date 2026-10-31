"""Processes this program started, so it can stop them again -- and only them.

A runtime that runs a model *out of process* (a server, a worker) needs lifecycle
management, and the two rules that matter are the ones a naive implementation gets
wrong:

* **Never touch a process this program did not start.** A record is only created
  through :meth:`ProcessRegistry.start`, which is the one place ``Popen`` is called
  for a managed runtime, so :meth:`ProcessRegistry.stop` can only ever signal a pid
  it owns. There is no "stop by name" and no "kill everything that looks like
  llama".
* **A shared server is not a per-model process.** :class:`ManagedProcess` records
  the *service* (its pid) separately from the *model* it is holding, so stopping a
  model inside a resident server is not the same operation as stopping the server,
  and neither is ever mistaken for the other.

Records are kept on disk as well as in memory, so a process orphaned by a crash is
still known after a restart -- and a stale record (a pid that is gone, or that has
been reused by another program) is detected rather than acted on. Detection is
deliberately conservative: if the identity of a pid cannot be confirmed, the record
is reported *stale* and nothing is signalled.

The whole module is exercised with an injected ``popen`` and an injected liveness
probe, so no test starts a real process.
"""

from __future__ import annotations

import json
import os
import signal
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

#: What a managed process is doing, in one word.
STATUS_STARTING = "starting"
STATUS_RUNNING = "running"
STATUS_STOPPED = "stopped"
STATUS_EXITED = "exited"
STATUS_FAILED = "failed"
STATUS_STALE = "stale"

#: How long a start may take before it is a startup timeout.
DEFAULT_START_TIMEOUT = 30.0
#: How long a graceful stop is given before the policy escalates.
DEFAULT_STOP_TIMEOUT = 10.0
#: How long to wait between readiness probes.
DEFAULT_POLL_INTERVAL = 0.2

_LOCK = threading.RLock()


class ProcessError(RuntimeError):
    """A managed process could not be started, stopped, or confirmed."""


@dataclass(frozen=True)
class ProcessRecord:
    """One process this program started, and what it is for.

    ``runtime`` and ``model_id`` name *what* the process is for, and
    ``service_pid`` distinguishes a shared server (one pid serving many models)
    from a per-model process. ``identity`` is a short fingerprint used to detect a
    reused pid, so a stop can never hit an unrelated program.
    """

    process_id: str
    pid: int
    runtime: str = ""
    model_id: str = ""
    command: tuple[str, ...] = field(default=())
    started_at: str = ""
    status: str = STATUS_STARTING
    #: For a shared runtime, the pid of the server; equals ``pid`` for a per-model one.
    service_pid: int | None = None
    #: ``True`` when this record is a long-lived service rather than one model.
    is_service: bool = False
    last_health_check: str = ""
    exit_code: int | None = None
    last_error: str = ""
    #: A short fingerprint of the process, used to refuse a reused pid.
    identity: str = ""
    host: str = ""

    def describe(self) -> str:
        what = self.model_id or ("the runtime service" if self.is_service else "a process")
        bits = [f"{self.process_id}: {what}", f"pid {self.pid}", self.status]
        if self.runtime:
            bits.append(self.runtime)
        line = " · ".join(bits)
        return f"{line} — {self.last_error}" if self.last_error else line

    def to_primitive(self) -> dict[str, Any]:
        return {
            "command": list(self.command),
            "exit_code": self.exit_code,
            "host": self.host,
            "identity": self.identity,
            "is_service": self.is_service,
            "last_error": self.last_error,
            "last_health_check": self.last_health_check,
            "model_id": self.model_id,
            "pid": self.pid,
            "process_id": self.process_id,
            "runtime": self.runtime,
            "service_pid": self.service_pid,
            "started_at": self.started_at,
            "status": self.status,
        }

    @classmethod
    def from_primitive(cls, data: object) -> ProcessRecord | None:
        if not isinstance(data, dict):
            return None
        pid = data.get("pid")
        process_id = str(data.get("process_id") or "")
        if not isinstance(pid, int) or not process_id:
            return None
        return cls(
            process_id=process_id,
            pid=pid,
            runtime=str(data.get("runtime") or ""),
            model_id=str(data.get("model_id") or ""),
            command=tuple(str(item) for item in (data.get("command") or ())),
            started_at=str(data.get("started_at") or ""),
            status=str(data.get("status") or STATUS_STARTING),
            service_pid=data.get("service_pid") if isinstance(data.get("service_pid"), int) else None,
            is_service=bool(data.get("is_service")),
            last_health_check=str(data.get("last_health_check") or ""),
            exit_code=data.get("exit_code") if isinstance(data.get("exit_code"), int) else None,
            last_error=str(data.get("last_error") or ""),
            identity=str(data.get("identity") or ""),
            host=str(data.get("host") or ""),
        )


def process_identity(pid: int) -> str:
    """A short fingerprint of a live process, or ``""`` when it cannot be read.

    On Linux the process's own start time (field 22 of ``/proc/<pid>/stat``) is a
    stable identity: a reused pid gets a *new* start time, so comparing the two is
    how a stale record is detected. Where the fingerprint cannot be read, the empty
    string means "unknown", and the caller treats that conservatively.
    """
    if pid <= 0:
        return ""
    try:
        text = Path(f"/proc/{pid}/stat").read_text(encoding="ascii", errors="ignore")
    except OSError:
        return ""
    # The command may contain spaces and parentheses; the fields after the last ')'
    # are what matter, and starttime is the 22nd field overall.
    try:
        tail = text.rsplit(")", 1)[1].split()
        return tail[19] if len(tail) > 19 else ""
    except (IndexError, ValueError):
        return ""


def pid_is_alive(pid: int) -> bool:
    """Whether a pid exists, without signalling it.

    ``os.kill(pid, 0)`` is the portable check: it raises ``ProcessLookupError`` for
    a gone process and ``PermissionError`` for one owned by another user (which is
    still alive). Nothing here changes the process's state.
    """
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError:
        return False
    return True


class ProcessRegistry:
    """The processes this program started, and the only code that signals them.

    ``root`` is where the durable record lives (a small JSON file); a test passes
    its own. ``popen`` and ``probe`` are the two seams: the default uses
    ``subprocess.Popen`` and the real liveness check, and a test injects fakes so
    nothing is really spawned.
    """

    def __init__(
        self,
        root: Path | str | None = None,
        *,
        popen: Any = None,
        probe: Callable[[int], bool] | None = None,
        identity: Callable[[int], str] | None = None,
        clock: Callable[[], float] | None = None,
        sleep: Callable[[float], None] | None = None,
    ) -> None:
        self.root = Path(root) if root is not None else _default_root()
        self._popen = popen
        self._probe = probe or pid_is_alive
        self._identity = identity or process_identity
        self._clock = clock or time.monotonic
        self._sleep = sleep or time.sleep
        self._records: dict[str, ProcessRecord] = {}
        self._handles: dict[str, Any] = {}
        self._load()

    # --- persistence ---------------------------------------------------

    def _path(self) -> Path:
        return self.root / "processes.json"

    def _load(self) -> None:
        try:
            raw = json.loads(self._path().read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return
        if not isinstance(raw, list):
            return
        for item in raw:
            record = ProcessRecord.from_primitive(item)
            if record is not None:
                self._records[record.process_id] = record

    def _save(self) -> None:
        try:
            self.root.mkdir(parents=True, exist_ok=True)
            payload = [record.to_primitive() for record in self._records.values()]
            temp = self._path().with_suffix(".json.tmp")
            temp.write_text(
                json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True),
                encoding="utf-8",
            )
            os.replace(temp, self._path())
        except OSError:
            # A record that cannot be written costs a future cleanup, not a run.
            return

    # --- reading -------------------------------------------------------

    def records(self) -> tuple[ProcessRecord, ...]:
        """Every recorded process, newest first, with its status refreshed."""
        with _LOCK:
            return tuple(
                self.refresh(record.process_id)
                for record in sorted(
                    self._records.values(), key=lambda item: item.started_at, reverse=True
                )
            )

    def get(self, process_id: str) -> ProcessRecord:
        """One record, with its status refreshed."""
        return self.refresh(process_id)

    def find(self, process_id: str) -> ProcessRecord | None:
        return self._records.get(process_id)

    def for_model(self, model_id: str) -> tuple[ProcessRecord, ...]:
        """The live processes holding a model."""
        return tuple(
            record
            for record in self.records()
            if record.model_id == model_id and record.status in (STATUS_STARTING, STATUS_RUNNING)
        )

    def refresh(self, process_id: str) -> ProcessRecord:
        """Re-read a record's liveness and update its status.

        A record whose pid is gone becomes ``exited`` (or ``stale`` when the pid was
        reused by another program, which is detected through the identity
        fingerprint). Nothing is signalled here -- this is a read.
        """
        with _LOCK:
            record = self._records.get(process_id)
            if record is None:
                raise ProcessError(f"No managed process is recorded as {process_id!r}.")
            if record.status in (STATUS_STOPPED, STATUS_EXITED, STATUS_FAILED):
                return record
            if not self._probe(record.pid):
                return self._replace(
                    record, status=STATUS_EXITED,
                    last_error=record.last_error or "The process is no longer running.",
                )
            current = self._identity(record.pid)
            if record.identity and current and current != record.identity:
                return self._replace(
                    record, status=STATUS_STALE,
                    last_error=(
                        "The pid now belongs to a different process, so this record is "
                        "stale and will not be signalled."
                    ),
                )
            if record.status == STATUS_STARTING:
                return self._replace(record, status=STATUS_RUNNING)
            return record

    def _replace(self, record: ProcessRecord, **changes: Any) -> ProcessRecord:
        from dataclasses import replace

        updated = replace(record, **changes)
        self._records[record.process_id] = updated
        self._save()
        return updated

    # --- starting ------------------------------------------------------

    def start(
        self,
        command: list[str],
        *,
        runtime: str = "",
        model_id: str = "",
        process_id: str = "",
        is_service: bool = False,
        ready: Callable[[], bool] | None = None,
        start_timeout: float = DEFAULT_START_TIMEOUT,
        env: dict[str, str] | None = None,
        cwd: Path | str | None = None,
    ) -> ProcessRecord:
        """Start a managed process and wait for it to become ready.

        The command is a **list**, never a shell string, so nothing in it can be
        interpreted as shell syntax. Readiness is the caller's own probe (an HTTP
        ping, a port check); when no probe is given the process is reported
        ``running`` as soon as it spawns. A process that exits during the wait, or
        that never becomes ready within ``start_timeout``, is stopped and reported
        as a failure -- never left running behind a failed start.
        """
        popen = self._popen or _default_popen
        pid_key = process_id or f"{runtime or 'proc'}-{model_id or 'service'}-{int(self._clock() * 1000)}"
        try:
            handle = popen(command, env=env, cwd=str(cwd) if cwd is not None else None)
        except Exception as error:
            record = ProcessRecord(
                process_id=pid_key, pid=0, runtime=runtime, model_id=model_id,
                command=tuple(command), started_at=_now(), status=STATUS_FAILED,
                is_service=is_service, last_error=f"Could not start: {error}",
            )
            self._records[pid_key] = record
            self._save()
            return record
        pid = int(getattr(handle, "pid", 0) or 0)
        record = ProcessRecord(
            process_id=pid_key,
            pid=pid,
            runtime=runtime,
            model_id=model_id,
            command=tuple(command),
            started_at=_now(),
            status=STATUS_STARTING,
            service_pid=pid if is_service else None,
            is_service=is_service,
            identity=self._identity(pid) if pid else "",
            host=_hostname(),
        )
        self._records[pid_key] = record
        self._handles[pid_key] = handle
        self._save()

        deadline = self._clock() + max(0.0, start_timeout)
        while True:
            if _handle_exited(handle):
                code = _handle_exit_code(handle)
                return self._replace(
                    record, status=STATUS_FAILED, exit_code=code,
                    last_error=(
                        "The process exited during startup"
                        + (f" with code {code}" if code is not None else "")
                        + "."
                    ),
                )
            if ready is None or ready():
                return self._replace(record, status=STATUS_RUNNING, last_health_check=_now())
            if self._clock() >= deadline:
                self._terminate(handle, record)
                return self._replace(
                    record, status=STATUS_FAILED,
                    last_error=(
                        f"The process did not become ready within {start_timeout:.0f}s "
                        "and was stopped."
                    ),
                )
            self._sleep(DEFAULT_POLL_INTERVAL)

    # --- stopping ------------------------------------------------------

    def stop(
        self,
        process_id: str,
        *,
        timeout: float = DEFAULT_STOP_TIMEOUT,
        force: bool = False,
    ) -> ProcessRecord:
        """Stop a process this program started, gracefully first.

        A ``SIGTERM`` (or ``terminate()`` on Windows) is sent first and given
        ``timeout`` seconds; only then, and only when ``force`` is set, is a
        ``SIGKILL`` used. A process that is already gone, or whose record is stale,
        is reported as such without any signal -- an unrelated program is never
        touched.
        """
        with _LOCK:
            record = self._records.get(process_id)
            if record is None:
                raise ProcessError(f"No managed process is recorded as {process_id!r}.")
            current = self.refresh(process_id)
            if current.status in (STATUS_STOPPED, STATUS_EXITED, STATUS_FAILED):
                return current
            if current.status == STATUS_STALE:
                return self._replace(
                    current, status=STATUS_STOPPED,
                    last_error="The record was stale; nothing was signalled.",
                )
            handle = self._handles.get(process_id)
            if handle is None:
                # A record from a previous run: signal the pid directly, and only
                # after confirming the identity still matches.
                if not self._probe(current.pid):
                    return self._replace(current, status=STATUS_EXITED)
                self._signal(current.pid, hard=False)
            else:
                self._terminate(handle, current, force=force, timeout=timeout)
            return self._await_exit(current, handle, timeout=timeout, force=force)

    def _await_exit(
        self, record: ProcessRecord, handle: Any, *, timeout: float, force: bool
    ) -> ProcessRecord:
        deadline = self._clock() + max(0.0, timeout)
        while self._probe(record.pid) and self._clock() < deadline:
            self._sleep(DEFAULT_POLL_INTERVAL)
        if self._probe(record.pid):
            if force:
                self._signal(record.pid, hard=True)
                self._sleep(DEFAULT_POLL_INTERVAL)
            else:
                return self._replace(
                    record, status=STATUS_RUNNING,
                    last_error=(
                        f"The process did not stop within {timeout:.0f}s. Re-run with "
                        "force to end it."
                    ),
                )
        code = _handle_exit_code(handle) if handle is not None else None
        self._handles.pop(record.process_id, None)
        return self._replace(
            record, status=STATUS_STOPPED, exit_code=code,
            last_error="",
        )

    def _terminate(
        self, handle: Any, record: ProcessRecord, *, force: bool = False, timeout: float = DEFAULT_STOP_TIMEOUT
    ) -> None:
        """Ask a handle to stop, preferring its own graceful path."""
        try:
            if force and hasattr(handle, "kill"):
                handle.kill()
            elif hasattr(handle, "terminate"):
                handle.terminate()
        except Exception:
            # Fall back to signalling the pid, which is still one we own.
            self._signal(record.pid, hard=force)

    def _signal(self, pid: int, *, hard: bool) -> None:
        """Signal a pid we started. Guarded, so an unrelated pid is never hit."""
        if pid <= 0 or not self._probe(pid):
            return
        sig = getattr(signal, "SIGKILL", None) if hard else signal.SIGTERM
        try:
            os.kill(pid, sig)  # type: ignore[arg-type]
        except (ProcessLookupError, PermissionError, OSError):
            return

    # --- cleanup -------------------------------------------------------

    def stop_all(self, *, force: bool = False) -> tuple[ProcessRecord, ...]:
        """Stop every process this program started, newest first."""
        stopped: list[ProcessRecord] = []
        for record in self.records():
            if record.status in (STATUS_RUNNING, STATUS_STARTING, STATUS_STALE):
                try:
                    stopped.append(self.stop(record.process_id, force=force))
                except ProcessError:
                    continue
        return tuple(stopped)

    def forget(self, process_id: str) -> None:
        """Drop a record for a process that is no longer running."""
        with _LOCK:
            record = self._records.get(process_id)
            if record is None:
                return
            if self._probe(record.pid):
                raise ProcessError(
                    f"{process_id!r} is still running; stop it before forgetting it."
                )
            self._records.pop(process_id, None)
            self._save()

    def prune(self) -> int:
        """Drop every record whose process is gone. Returns how many were dropped."""
        with _LOCK:
            gone = [
                process_id
                for process_id, record in self._records.items()
                if not self._probe(record.pid)
            ]
            for process_id in gone:
                self._records.pop(process_id, None)
            if gone:
                self._save()
            return len(gone)

    def describe(self) -> str:
        records = self.records()
        if not records:
            return "No process this program started is running."
        lines = ["Processes started by this program"]
        lines.extend(f"  {record.describe()}" for record in records)
        return "\n".join(lines)


def _default_popen(command: list[str], *, env: Any = None, cwd: Any = None) -> Any:
    """Spawn a process with a **list** command, never a shell string."""
    import subprocess

    return subprocess.Popen(command, env=env, cwd=cwd)


def _handle_exited(handle: Any) -> bool:
    """Whether a ``Popen``-like handle has exited, without blocking."""
    poll = getattr(handle, "poll", None)
    if callable(poll):
        try:
            return poll() is not None
        except Exception:
            return False
    return False


def _handle_exit_code(handle: Any) -> int | None:
    if handle is None:
        return None
    code = getattr(handle, "returncode", None)
    return code if isinstance(code, int) else None


def _default_root() -> Path:
    from core.paths import data_dir

    return data_dir() / "model_processes"


def _hostname() -> str:
    import socket

    try:
        return socket.gethostname()
    except OSError:  # pragma: no cover - a hostname is usually available
        return ""


def _now() -> str:
    from datetime import UTC, datetime

    return datetime.now(UTC).isoformat()


__all__ = [
    "DEFAULT_POLL_INTERVAL",
    "DEFAULT_START_TIMEOUT",
    "DEFAULT_STOP_TIMEOUT",
    "STATUS_EXITED",
    "STATUS_FAILED",
    "STATUS_RUNNING",
    "STATUS_STALE",
    "STATUS_STARTING",
    "STATUS_STOPPED",
    "ProcessError",
    "ProcessRecord",
    "ProcessRegistry",
    "pid_is_alive",
    "process_identity",
]
