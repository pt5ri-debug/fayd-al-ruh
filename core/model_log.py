"""A durable, bounded record of what the Model Manager actually did.

The project had no operation log for models at all: ``logging`` is imported zero
times in the whole repository, :mod:`core.progress` writes transient lines to
stderr, and :class:`core.audit_log.AuditLog` is an in-memory record of *permission
decisions*. So when a model failed to load, the reason existed only in the
exception a caller happened to catch, and was gone the moment the window closed.

This module is the missing half: a small append-only history of model operations
(discover, register, inspect, load, unload, test, download, remove, failure), each
carrying when it happened, which model it concerned, whether it succeeded, and the
real message. It is deliberately:

* **bounded** -- a ring of at most :data:`MAX_EVENTS`, so a long session cannot
  grow the file without limit (:mod:`core.best_effort` learned the same lesson);
* **redacted** -- a message is scrubbed before it is stored, so a base URL or a
  token that a transport error echoed back can never reach a log a user pastes
  into an issue;
* **forgiving** -- a corrupt line is skipped and counted, never raised at a caller
  that only wanted to read the history;
* **best-effort on write** -- a read-only data folder must not crash a window, but
  the failure is *recorded* rather than swallowed.

Storage is the same :class:`storage.json_store.JsonStore` the registry uses, so
the log is atomic and git-ignored like everything else under ``data/``.
"""

from __future__ import annotations

import re
import threading
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from core.best_effort import best_effort
from core.paths import model_manager_dir

#: The JsonStore namespace the log lives under.
NAMESPACE = "model_manager"
#: The single key holding the ring.
KEY = "operations"
#: How many events are kept. Old enough to debug a session, small enough to read.
MAX_EVENTS = 500

#: What an operation is. A closed set, so a report can group by it.
OP_DISCOVER = "discover"
OP_REGISTER = "register"
OP_UPDATE = "update"
OP_REMOVE = "remove"
OP_INSPECT = "inspect"
OP_LOAD = "load"
OP_UNLOAD = "unload"
OP_TEST = "test"
OP_DOWNLOAD = "download"
OP_FAILURE = "failure"
KNOWN_OPERATIONS: tuple[str, ...] = (
    OP_DISCOVER, OP_REGISTER, OP_UPDATE, OP_REMOVE, OP_INSPECT,
    OP_LOAD, OP_UNLOAD, OP_TEST, OP_DOWNLOAD, OP_FAILURE,
)

#: The shapes a secret takes. Scrubbed from every stored message.
_SECRET_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"sk-[A-Za-z0-9_\-]{8,}"),
    re.compile(r"hf_[A-Za-z0-9]{8,}"),
    re.compile(r"gh[pousr]_[A-Za-z0-9]{8,}"),
    re.compile(r"(?i)bearer\s+[A-Za-z0-9._\-]{8,}"),
    re.compile(r"(?i)(api[_-]?key|token|password|secret)\s*[=:]\s*\S+"),
    # A credential embedded in a URL: ``https://user:pass@host``.
    re.compile(r"(?i)(https?://)[^/\s:@]+:[^/\s@]+@"),
)

#: A message is capped, so one runaway error cannot fill the ring.
MAX_MESSAGE = 1000

_LOCK = threading.RLock()


def redact(text: str) -> str:
    """Remove anything secret-shaped from ``text``.

    Belt and braces with :func:`core.cloud_model.redact_secret` (which needs to be
    told the secret): this one scrubs a value it was never told about, which is
    what a log needs, because a transport error can echo a credential the caller
    did not pass in.
    """
    cleaned = str(text or "")
    for pattern in _SECRET_PATTERNS:
        cleaned = pattern.sub("***", cleaned)
    return cleaned[:MAX_MESSAGE]


def _now() -> str:
    """A UTC timestamp, ISO-8601, seconds precision."""
    return datetime.now(UTC).replace(microsecond=0).isoformat()


@dataclass(frozen=True)
class ModelEvent:
    """One thing the manager did, and what happened."""

    operation: str
    model_id: str = ""
    display_name: str = ""
    runtime: str = ""
    ok: bool = True
    message: str = ""
    at: str = ""

    def to_primitive(self) -> dict[str, Any]:
        return {
            "at": self.at,
            "display_name": self.display_name,
            "message": self.message,
            "model_id": self.model_id,
            "ok": bool(self.ok),
            "operation": self.operation,
            "runtime": self.runtime,
        }

    @classmethod
    def from_primitive(cls, data: object) -> ModelEvent | None:
        """Read one stored event, or ``None`` when the row is unusable."""
        if not isinstance(data, dict):
            return None
        operation = str(data.get("operation") or "").strip()
        if not operation:
            return None
        return cls(
            operation=operation,
            model_id=str(data.get("model_id") or "")[:256],
            display_name=str(data.get("display_name") or "")[:256],
            runtime=str(data.get("runtime") or "")[:64],
            ok=bool(data.get("ok", True)),
            message=redact(str(data.get("message") or "")),
            at=str(data.get("at") or ""),
        )

    def describe(self) -> str:
        """A one-line, readable rendering for a log surface."""
        who = self.display_name or self.model_id or "the manager"
        outcome = "ok" if self.ok else "failed"
        when = self.at or "unknown time"
        line = f"{when}  {self.operation:<8} {outcome:<6} {who}"
        return f"{line} — {self.message}" if self.message else line


class ModelLog:
    """The bounded, redacted operation history.

    ``root`` defaults to :func:`core.paths.model_manager_dir`; a test passes its
    own. Reads and writes share one lock, so a worker thread appending while the
    UI thread reads cannot interleave.
    """

    def __init__(self, root: Path | str | None = None) -> None:
        self.root = Path(root) if root is not None else model_manager_dir()

    def _store(self) -> Any:
        from storage.json_store import JsonStore

        return JsonStore(self.root)

    def append(
        self,
        operation: str,
        *,
        model_id: str = "",
        display_name: str = "",
        runtime: str = "",
        ok: bool = True,
        message: str = "",
    ) -> ModelEvent:
        """Record one event and return it. Never raises for a write failure.

        An unknown operation is still recorded (as itself): refusing to log
        something that happened would be worse than an unfamiliar label.
        """
        event = ModelEvent(
            operation=str(operation or OP_FAILURE).strip()[:64],
            model_id=str(model_id or "")[:256],
            display_name=str(display_name or "")[:256],
            runtime=str(runtime or "")[:64],
            ok=bool(ok),
            message=redact(message),
            at=_now(),
        )
        with _LOCK:
            events = list(self._read())
            events.append(event)
            trimmed = events[-MAX_EVENTS:]
            with best_effort("write the Model Manager operation log"):
                self._store().save(NAMESPACE, KEY, {
                    "events": [item.to_primitive() for item in trimmed],
                    "version": 1,
                })
        return event

    def entries(self) -> tuple[ModelEvent, ...]:
        """Every recorded event, oldest first."""
        with _LOCK:
            return tuple(self._read())

    def tail(self, limit: int = 50) -> tuple[ModelEvent, ...]:
        """The most recent ``limit`` events, newest first (what a window shows)."""
        if limit <= 0:
            return ()
        return tuple(reversed(self.entries()[-limit:]))

    def for_model(self, model_id: str) -> tuple[ModelEvent, ...]:
        """Every event that concerned one model, oldest first."""
        wanted = str(model_id or "").strip()
        if not wanted:
            return ()
        return tuple(event for event in self.entries() if event.model_id == wanted)

    def failures(self) -> tuple[ModelEvent, ...]:
        """Every event that reported a failure, oldest first."""
        return tuple(event for event in self.entries() if not event.ok)

    def _read(self) -> list[ModelEvent]:
        try:
            raw = self._store().load(NAMESPACE, KEY)
        except Exception:
            return []
        if not isinstance(raw, dict):
            return []
        rows = raw.get("events")
        if not isinstance(rows, list):
            return []
        events: list[ModelEvent] = []
        for row in rows:
            event = ModelEvent.from_primitive(row)
            if event is not None:
                events.append(event)
        return events

    def corrupt_rows(self) -> int:
        """How many stored rows could not be read back, for an honest report."""
        try:
            raw = self._store().load(NAMESPACE, KEY)
        except Exception:
            return 0
        if not isinstance(raw, dict) or not isinstance(raw.get("events"), list):
            return 0
        rows = raw["events"]
        return sum(1 for row in rows if ModelEvent.from_primitive(row) is None)

    def clear(self, *, confirm: bool = False) -> int:
        """Forget the history. Destructive, so it must be confirmed explicitly."""
        if not confirm:
            from core.model_registry import ConfirmationRequired

            raise ConfirmationRequired(
                "Clearing the operation log is destructive and must be confirmed "
                "explicitly (`confirm=True`)."
            )
        with _LOCK:
            count = len(self._read())
            with best_effort("clear the Model Manager operation log"):
                self._store().save(NAMESPACE, KEY, {"events": [], "version": 1})
        return count

    def describe(self, limit: int = 20) -> str:
        """A readable tail of the log, for a CLI or a diagnostics surface."""
        events = self.tail(limit)
        if not events:
            return "No model operations have been recorded yet."
        return "\n".join(event.describe() for event in events)


def model_log(root: Path | str | None = None) -> ModelLog:
    """The log at ``root`` (the default data folder when omitted)."""
    return ModelLog(root)


__all__ = [
    "KEY",
    "KNOWN_OPERATIONS",
    "MAX_EVENTS",
    "NAMESPACE",
    "OP_DISCOVER",
    "OP_DOWNLOAD",
    "OP_FAILURE",
    "OP_INSPECT",
    "OP_LOAD",
    "OP_REGISTER",
    "OP_REMOVE",
    "OP_TEST",
    "OP_UNLOAD",
    "OP_UPDATE",
    "ModelEvent",
    "ModelLog",
    "model_log",
    "redact",
]
