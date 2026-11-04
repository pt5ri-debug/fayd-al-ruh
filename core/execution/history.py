"""Execution history and audit log (Phase 8, section 3).

Section 3 asks for an execution history and audit log; section 9 asks that
approval decisions be recorded; section 10 asks that credentials never appear in
logs. This module is the one append-only record of what the planning/execution
layer did: every plan, every task run, every permission decision, and every
recovery proposal -- bounded, redacted, and durable through the project's own
:class:`storage.json_store.JsonStore` when a data root is given.

It is deliberately *not* a second learning store: the learning integration
(:mod:`core.execution.coordinator`) records outcomes through the Phase-3 learning
coordinator. This log is the audit trail a reader inspects.

Dependency-free and Tk-free.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

#: Namespaces under which the log is persisted (a JsonStore document each).
HISTORY_NAMESPACE = "planning"
HISTORY_KEY = "execution_history"

#: How many entries the in-memory log keeps before the oldest is dropped.
DEFAULT_HISTORY_LIMIT = 500

#: A conservative secret scrubber: a bearer token, an API key, or a password in
#: a ``key=value`` pair is replaced, so a log line never carries a credential.
_SECRET_PATTERNS = (
    re.compile(r"(?i)\b(bearer\s+)[A-Za-z0-9._\-]+"),
    re.compile(r"(?i)\b(api[_-]?key\s*[=:]\s*)[^\s,;]+"),
    re.compile(r"(?i)\b(token\s*[=:]\s*)[^\s,;]+"),
    re.compile(r"(?i)\b(password\s*[=:]\s*)[^\s,;]+"),
    re.compile(r"(?i)\b(secret\s*[=:]\s*)[^\s,;]+"),
)


def redact(text: str) -> str:
    """Replace anything that looks like a credential in ``text``."""
    result = text
    for pattern in _SECRET_PATTERNS:
        result = pattern.sub(lambda m: m.group(1) + "<redacted>", result)
    return result


@dataclass(frozen=True)
class HistoryEntry:
    """One audited event -- what happened, when, and with what outcome."""

    kind: str
    summary: str
    detail: str = ""
    ok: bool = True
    plan_id: str = ""
    goal_id: str = ""
    task_id: str = ""
    actor: str = "system"
    at: str = field(default_factory=lambda: datetime.now(UTC).isoformat())

    def __post_init__(self) -> None:
        # Redact on construction, so a secret cannot enter the log at all.
        object.__setattr__(self, "summary", redact(self.summary))
        object.__setattr__(self, "detail", redact(self.detail))

    def to_primitive(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "summary": self.summary,
            "detail": self.detail,
            "ok": self.ok,
            "plan_id": self.plan_id,
            "goal_id": self.goal_id,
            "task_id": self.task_id,
            "actor": self.actor,
            "at": self.at,
        }

    @classmethod
    def from_primitive(cls, data: dict[str, Any]) -> HistoryEntry:
        return cls(
            kind=str(data.get("kind", "")),
            summary=str(data.get("summary", "")),
            detail=str(data.get("detail", "")),
            ok=bool(data.get("ok", True)),
            plan_id=str(data.get("plan_id", "")),
            goal_id=str(data.get("goal_id", "")),
            task_id=str(data.get("task_id", "")),
            actor=str(data.get("actor", "system")),
            at=str(data.get("at", "")),
        )

    def describe(self) -> str:
        mark = "ok" if self.ok else "!"
        where = f" [{self.plan_id}]" if self.plan_id else ""
        return f"{mark} {self.kind}{where}: {self.summary}"


@dataclass
class ExecutionHistory:
    """The append-only audit log, in memory and (optionally) on disk."""

    limit: int = DEFAULT_HISTORY_LIMIT
    data_root: str | None = None
    entries: list[HistoryEntry] = field(default_factory=list)

    def record(self, entry: HistoryEntry) -> HistoryEntry:
        self.entries.append(entry)
        if len(self.entries) > self.limit:
            self.entries = self.entries[-self.limit :]
        return entry

    def log(self, kind: str, summary: str, **kwargs: Any) -> HistoryEntry:
        return self.record(HistoryEntry(kind=kind, summary=summary, **kwargs))

    def recent(self, limit: int = 20, *, kind: str | None = None) -> tuple[HistoryEntry, ...]:
        chosen = [e for e in self.entries if kind is None or e.kind == kind]
        return tuple(chosen[-limit:])

    def count(self, *, kind: str | None = None) -> int:
        return sum(1 for e in self.entries if kind is None or e.kind == kind)

    def failures(self) -> tuple[HistoryEntry, ...]:
        return tuple(e for e in self.entries if not e.ok)

    # ------------------------------------------------------------- durability

    def _store(self) -> Any:
        if not self.data_root:
            return None
        try:
            from storage.json_store import JsonStore

            return JsonStore(self.data_root)
        except Exception:  # a store is a convenience, never a hard dependency
            return None

    def save(self) -> bool:
        """Persist the log (best-effort: a read-only root is not an error)."""
        store = self._store()
        if store is None:
            return False
        try:
            store.save(HISTORY_NAMESPACE, HISTORY_KEY, [e.to_primitive() for e in self.entries])
            return True
        except Exception:
            return False

    def load(self) -> int:
        """Load the log from disk, returning how many entries were read."""
        store = self._store()
        if store is None:
            return 0
        try:
            data = store.load(HISTORY_NAMESPACE, HISTORY_KEY)
        except Exception:
            return 0
        if not isinstance(data, list):
            return 0
        loaded: list[HistoryEntry] = []
        for item in data:
            if isinstance(item, dict):
                try:
                    loaded.append(HistoryEntry.from_primitive(item))
                except Exception:
                    continue
        self.entries = loaded[-self.limit :]
        return len(self.entries)

    def describe(self) -> str:
        if not self.entries:
            return "no execution history"
        return "\n".join(entry.describe() for entry in self.entries[-10:])


__all__ = [
    "DEFAULT_HISTORY_LIMIT",
    "HISTORY_KEY",
    "HISTORY_NAMESPACE",
    "ExecutionHistory",
    "HistoryEntry",
    "redact",
]
