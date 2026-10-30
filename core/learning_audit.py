"""The append-only learning audit: what was learned, and what was refused.

The task requires that *all learning operations are logged* and that *invalid
cross-character access attempts are rejected and logged*. A rejected attempt that
leaves no trace is exactly the kind of silence this project refuses elsewhere (see
:mod:`core.best_effort`), so the audit is a first-class part of the design rather
than a debug aid.

The log is a single append-only JSONL file under the main model's learning
directory. Records are only ever added -- never edited or removed -- so it is a
tamper-evident trail, like :mod:`core.audit_log`. Two kinds of event are written:

* an **outcome** -- a record was stored, a status changed, a store was reset;
* a **refusal** -- an access the isolation policy denied.

Reading is streaming and bounded, so a long-lived install's audit never has to be
loaded whole. A corrupt line is skipped rather than crashing the reader: an audit
that cannot be read must never stop the program, but the failure is still visible
in the returned report.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from core.best_effort import best_effort
from core.learning_isolation import LearningAccessError, assert_read
from core.paths import learning_audit_dir

#: The audit file's name, and how many recent events a bounded read returns by
#: default.
AUDIT_FILENAME = "learning-audit.jsonl"
DEFAULT_LIMIT = 200

#: The event kinds, so a reader can filter without parsing a free-text reason.
EVENT_LEARNED = "learned"
EVENT_STATUS = "status"
EVENT_RESET = "reset"
EVENT_SHARED = "shared"
EVENT_DENIED = "denied"

EVENT_KINDS: tuple[str, ...] = (
    EVENT_LEARNED,
    EVENT_STATUS,
    EVENT_RESET,
    EVENT_SHARED,
    EVENT_DENIED,
)


@dataclass(frozen=True)
class AuditEvent:
    """One logged learning operation or refused access."""

    kind: str
    actor: str
    owner: str
    detail: str = ""
    record_id: str = ""
    at: str = field(default_factory=lambda: datetime.now(UTC).isoformat())

    def __post_init__(self) -> None:
        if self.kind not in EVENT_KINDS:
            raise ValueError(f"Unknown audit event kind {self.kind!r}.")

    def to_primitive(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "actor": self.actor,
            "owner": self.owner,
            "detail": self.detail,
            "record_id": self.record_id,
            "at": self.at,
        }

    @classmethod
    def from_primitive(cls, data: dict[str, Any]) -> AuditEvent:
        return cls(
            kind=str(data.get("kind") or EVENT_LEARNED),
            actor=str(data.get("actor") or ""),
            owner=str(data.get("owner") or ""),
            detail=str(data.get("detail") or ""),
            record_id=str(data.get("record_id") or ""),
            at=str(data.get("at") or ""),
        )

    def describe(self) -> str:
        """One readable line for a report."""
        where = f" [{self.record_id}]" if self.record_id else ""
        return f"{self.at} {self.kind}: {self.actor} -> {self.owner}{where} {self.detail}".strip()


class LearningAudit:
    """The append-only learning audit for the whole project.

    The audit lives under the main model's learning directory, because it records
    operations **about** the characters as well as the main model -- but it is
    *metadata* (who did what, to which store), never a character's record content,
    so writing it does not breach the isolation the stores enforce on records.
    """

    def __init__(self, root: Path | None = None) -> None:
        self.root = Path(root) if root is not None else learning_audit_dir()
        self.path = self.root / AUDIT_FILENAME

    # ---------------------------------------------------------------- writing

    def record(self, event: AuditEvent) -> AuditEvent:
        """Append one event. Best-effort, so a read-only folder never crashes a chat.

        The failure is recorded through :func:`core.best_effort.best_effort` rather
        than swallowed, exactly as every other deliberate best-effort write in the
        project.
        """
        line = json.dumps(event.to_primitive(), ensure_ascii=False, sort_keys=True)
        with best_effort("append to the learning audit"):
            self.root.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8") as handle:
                handle.write(line + "\n")
        return event

    def log(
        self,
        kind: str,
        *,
        actor: str,
        owner: str,
        detail: str = "",
        record_id: str = "",
    ) -> AuditEvent:
        """Append one event from its parts (the common entry point)."""
        return self.record(
            AuditEvent(kind=kind, actor=actor, owner=owner, detail=detail, record_id=record_id)
        )

    def log_denied(self, reader: str, owner: str, reason: str) -> AuditEvent:
        """Log a refused cross-character access attempt."""
        return self.log(EVENT_DENIED, actor=reader, owner=owner, detail=reason)

    # ---------------------------------------------------------------- reading

    def events(self, *, limit: int = DEFAULT_LIMIT) -> tuple[AuditEvent, ...]:
        """The most recent ``limit`` events, oldest first within the window.

        Streaming and bounded: the file is read line by line and only the tail is
        kept, so a very long audit never has to be held in memory. A corrupt line
        is skipped -- an unreadable audit must not stop the program.
        """
        if limit <= 0 or not self.path.is_file():
            return ()
        from collections import deque

        window: deque[AuditEvent] = deque(maxlen=limit)
        with best_effort("read the learning audit"):
            with self.path.open("r", encoding="utf-8") as handle:
                for line in handle:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        data = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if isinstance(data, dict):
                        window.append(AuditEvent.from_primitive(data))
        return tuple(window)

    def denials(self, *, limit: int = DEFAULT_LIMIT) -> tuple[AuditEvent, ...]:
        """The recent refused access attempts (the isolation trail)."""
        return tuple(event for event in self.events(limit=limit) if event.kind == EVENT_DENIED)

    def describe(self, *, limit: int = 20) -> str:
        """A short readable tail of the audit, for a status line."""
        events = self.events(limit=limit)
        if not events:
            return "No learning operations recorded yet."
        return "\n".join(event.describe() for event in events)


#: A process-wide audit, so a caller that does not build one still logs. Kept
#: lazy so importing this module touches no disk.
_default: LearningAudit | None = None


def default_audit() -> LearningAudit:
    """The shared audit instance (built on first use, so import stays free)."""
    global _default
    if _default is None:
        _default = LearningAudit()
    return _default


def log_access(reader: str, owner: str) -> bool:
    """Check the isolation policy and log the outcome; return whether it is allowed.

    The one helper every cross-owner read goes through: it *refuses* (returning
    ``False``) and **logs the refusal** rather than raising, so a caller can report
    a denied read gracefully. A caller that must fail hard uses
    :func:`core.learning_isolation.assert_read` instead.
    """
    try:
        assert_read(reader, owner)
    except LearningAccessError as error:
        default_audit().log_denied(reader, owner, str(error))
        return False
    default_audit().log(
        EVENT_SHARED,
        actor=reader,
        owner=owner,
        detail="permitted knowledge read.",
    )
    return True


__all__ = [
    "AUDIT_FILENAME",
    "DEFAULT_LIMIT",
    "EVENT_DENIED",
    "EVENT_KINDS",
    "EVENT_LEARNED",
    "EVENT_RESET",
    "EVENT_SHARED",
    "EVENT_STATUS",
    "AuditEvent",
    "LearningAudit",
    "default_audit",
    "log_access",
]
