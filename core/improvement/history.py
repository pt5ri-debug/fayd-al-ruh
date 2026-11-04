"""The improvement history and audit log: an append-only record of every step.

Section 15 of the phase requires an auditable history of weaknesses, proposals,
experiments, evaluations, approvals, deployments, rejections, failures,
rollbacks, and outcomes, with every improvement traceable to the problem it
addresses -- and explicitly forbids erasing a failed experiment merely because it
failed. Section 17 forbids modifying the audit log to hide a change.

This module is that history, built on the project's own durable
:class:`storage.json_store.JsonStore`. Three properties make it the audit the
phase asks for:

* **append-only** -- :class:`ImprovementHistory` only ever *adds* an entry. There
  is no edit and no delete; a correction is a new entry, exactly like the
  learning records elsewhere in the project;
* **provenance** -- every entry names the proposal and, where there is one, the
  weakness it answers, so the chain problem → proposal → experiment → outcome is
  reconstructable; and
* **durable and bounded** -- the store keeps the newest entries and reports how
  many older ones it trimmed, so the record is honest about its own bound rather
  than silently dropping history.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import Enum
from typing import Any
from uuid import uuid4


class EntryKind(str, Enum):
    """What an audit entry records -- one closed vocabulary."""

    WEAKNESS = "weakness"
    PROPOSAL = "proposal"
    EXPERIMENT = "experiment"
    EVALUATION = "evaluation"
    APPROVAL = "approval"
    REJECTION = "rejection"
    DEPLOYMENT = "deployment"
    ROLLBACK = "rollback"
    OUTCOME = "outcome"
    NOTE = "note"


@dataclass(frozen=True)
class HistoryEntry:
    """One append-only record of something the loop did or found."""

    entry_id: str
    kind: EntryKind
    summary: str
    proposal_id: str = ""
    weakness_id: str = ""
    detail: dict[str, Any] = field(default_factory=dict)
    actor: str = "loop"
    at: datetime = field(default_factory=lambda: datetime.now(UTC))

    def to_primitive(self) -> dict[str, object]:
        return {
            "entry_id": self.entry_id,
            "kind": self.kind.value,
            "summary": self.summary,
            "proposal_id": self.proposal_id,
            "weakness_id": self.weakness_id,
            "detail": self.detail,
            "actor": self.actor,
            "at": self.at.isoformat(),
        }

    @classmethod
    def from_primitive(cls, data: dict[str, object]) -> HistoryEntry:
        raw_at = data.get("at")
        at = datetime.now(UTC)
        if isinstance(raw_at, str):
            try:
                at = datetime.fromisoformat(raw_at)
            except ValueError:
                at = datetime.now(UTC)
        detail = data.get("detail")
        return cls(
            entry_id=str(data.get("entry_id", f"HIS-{uuid4().hex[:8]}")),
            kind=_kind(data.get("kind")),
            summary=str(data.get("summary", "")),
            proposal_id=str(data.get("proposal_id", "")),
            weakness_id=str(data.get("weakness_id", "")),
            detail=detail if isinstance(detail, dict) else {},
            actor=str(data.get("actor", "loop")),
            at=at,
        )


def _kind(value: object) -> EntryKind:
    if isinstance(value, str):
        try:
            return EntryKind(value)
        except ValueError:
            return EntryKind.NOTE
    return EntryKind.NOTE


#: How many entries the history keeps by default.
DEFAULT_MAX_ENTRIES = 2000

_NAMESPACE = "self_improvement/history"
_KEY = "entries"


class ImprovementHistory:
    """An append-only, durable audit trail over a JSON store (or memory alone)."""

    def __init__(self, *, store: Any | None = None, max_entries: int = DEFAULT_MAX_ENTRIES) -> None:
        self._store = store
        self.max_entries = max(1, max_entries)
        self._entries: list[HistoryEntry] = []
        self._trimmed = 0
        if store is not None:
            self.load()

    # -- writing ------------------------------------------------------------
    def append(
        self,
        kind: EntryKind,
        summary: str,
        *,
        proposal_id: str = "",
        weakness_id: str = "",
        detail: dict[str, Any] | None = None,
        actor: str = "loop",
    ) -> HistoryEntry:
        """Add one entry. There is deliberately no edit and no delete."""
        entry = HistoryEntry(
            entry_id=f"HIS-{uuid4().hex[:8]}",
            kind=kind,
            summary=summary,
            proposal_id=proposal_id,
            weakness_id=weakness_id,
            detail=detail or {},
            actor=actor,
        )
        self._entries.append(entry)
        if len(self._entries) > self.max_entries:
            overflow = len(self._entries) - self.max_entries
            del self._entries[:overflow]
            self._trimmed += overflow
        self._persist()
        return entry

    # -- reading ------------------------------------------------------------
    def entries(
        self,
        *,
        kind: EntryKind | None = None,
        proposal_id: str | None = None,
        limit: int | None = None,
    ) -> tuple[HistoryEntry, ...]:
        """Entries oldest-first, optionally filtered, newest-last."""
        chosen = [
            entry
            for entry in self._entries
            if (kind is None or entry.kind is kind) and (proposal_id is None or entry.proposal_id == proposal_id)
        ]
        if limit is not None and limit >= 0:
            chosen = chosen[-limit:]
        return tuple(chosen)

    def for_proposal(self, proposal_id: str) -> tuple[HistoryEntry, ...]:
        """Every entry about one proposal, so its whole story is in one place."""
        return self.entries(proposal_id=proposal_id)

    def trace(self, proposal_id: str) -> str:
        """A readable trace of one proposal's life, from its finding to its outcome."""
        related = self.for_proposal(proposal_id)
        if not related:
            return f"No history for {proposal_id}."
        lines = [f"History for {proposal_id}:"]
        for entry in related:
            lines.append(f"  {entry.at.isoformat()} [{entry.kind.value}] {entry.summary}")
        return "\n".join(lines)

    @property
    def trimmed(self) -> int:
        """How many older entries the bound dropped, so the bound is visible."""
        return self._trimmed

    def describe(self) -> str:
        return f"Improvement history: {len(self._entries)} entry(ies), {self._trimmed} trimmed."

    # -- persistence --------------------------------------------------------
    def _persist(self) -> None:
        if self._store is None:
            return
        from core.best_effort import best_effort

        with best_effort("persist the improvement history"):
            self._store.save(_NAMESPACE, _KEY, {"entries": [e.to_primitive() for e in self._entries]})

    def load(self) -> int:
        """Read the stored entries; a corrupt store reads as empty, never crashes."""
        if self._store is None:
            return 0
        from core.best_effort import best_effort

        data: object = None
        with best_effort("load the improvement history"):
            data = self._store.load(_NAMESPACE, _KEY)
        if not isinstance(data, dict):
            return 0
        raw = data.get("entries")
        if not isinstance(raw, list):
            return 0
        restored: list[HistoryEntry] = []
        for item in raw:
            if isinstance(item, dict):
                restored.append(HistoryEntry.from_primitive(item))
        self._entries = restored[-self.max_entries:]
        return len(self._entries)


def describe_entries(entries: Iterable[HistoryEntry]) -> str:
    """Render entries as a readable block."""
    ordered = list(entries)
    if not ordered:
        return "No history."
    return "\n".join(
        f"{entry.at.isoformat()} [{entry.kind.value}] {entry.summary}" for entry in ordered
    )


__all__ = [
    "DEFAULT_MAX_ENTRIES",
    "EntryKind",
    "HistoryEntry",
    "ImprovementHistory",
    "describe_entries",
]
