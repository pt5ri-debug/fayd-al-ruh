"""One character's independent learning store, and the main model's.

Every owner -- each of the seven characters and the project's own model -- has its
**own** store, resolved from :func:`core.learning_isolation.store_dir` and nowhere
else. A store is built for exactly one owner and has no method that takes another
owner's key, so a character *cannot* read or write another character's records even
if a caller asks it to: the isolation is structural rather than a policy flag the
caller could skip.

What a store holds, on disk:

* ``records.jsonl`` -- an **append-only** log of :class:`~core.learning_records.LearningRecord`
  lines. Nothing is ever rewritten: revising a record appends a new version that
  points back at the one it supersedes, so a conclusion's history -- and the reason
  it changed -- is preserved exactly as the task requires.
* ``stats.json`` -- a small, atomically-replaced count of records by status, so a
  statistics view is one read rather than a full scan.

Reading is streaming and bounded: :meth:`CharacterLearningStore.recent` reads the
file line by line and keeps only the tail, so a long-lived store is never loaded
whole into RAM. A corrupt line is skipped and counted, never crashing the reader.

Nothing here imports Tk or loads a model, so a store is cheap to build and testable
with no display.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from core.best_effort import best_effort
from core.learning_audit import (
    EVENT_LEARNED,
    EVENT_RESET,
    EVENT_STATUS,
    LearningAudit,
    default_audit,
)
from core.learning_isolation import (
    MAIN_MODEL_OWNER,
    assert_owner,
    is_main_model,
    owner_for,
    store_dir_for,
)
from core.learning_records import (
    LEARNING_STATUSES,
    LearningRecord,
)

#: The store's files.
RECORDS_FILENAME = "records.jsonl"
STATS_FILENAME = "stats.json"

#: How many records a bounded read returns by default.
DEFAULT_LIMIT = 200


@dataclass(frozen=True)
class LearningStats:
    """How many records a store holds, by status, with the newest timestamp."""

    owner: str
    total: int = 0
    by_status: dict[str, int] = field(default_factory=dict)
    by_source: dict[str, int] = field(default_factory=dict)
    latest: str = ""
    corrupt_lines: int = 0

    @property
    def verified(self) -> int:
        return self.by_status.get("verified", 0)

    @property
    def provisional(self) -> int:
        return self.by_status.get("provisional", 0)

    @property
    def pending(self) -> int:
        return self.by_status.get("pending", 0)

    @property
    def rejected(self) -> int:
        return self.by_status.get("rejected", 0)

    @property
    def contradicted(self) -> int:
        return self.by_status.get("contradicted", 0)

    @property
    def superseded(self) -> int:
        return self.by_status.get("superseded", 0)

    def to_primitive(self) -> dict[str, Any]:
        return {
            "owner": self.owner,
            "total": self.total,
            "by_status": dict(self.by_status),
            "by_source": dict(self.by_source),
            "latest": self.latest,
            "corrupt_lines": self.corrupt_lines,
        }

    @classmethod
    def from_primitive(cls, owner: str, data: dict[str, Any] | None) -> LearningStats:
        if not isinstance(data, dict):
            return cls(owner=owner)
        return cls(
            owner=owner,
            total=int(data.get("total") or 0),
            by_status={str(k): int(v) for k, v in (data.get("by_status") or {}).items()},
            by_source={str(k): int(v) for k, v in (data.get("by_source") or {}).items()},
            latest=str(data.get("latest") or ""),
            corrupt_lines=int(data.get("corrupt_lines") or 0),
        )

    def summary(self) -> str:
        """One honest line: how much this store holds and how settled it is."""
        return (
            f"{self.owner}: {self.total} records "
            f"({self.verified} verified, {self.provisional} provisional, "
            f"{self.pending} pending, {self.rejected} rejected, "
            f"{self.contradicted} contradicted, {self.superseded} superseded)"
        )


class CharacterLearningStore:
    """One owner's independent learning store.

    ``owner`` is a character key or :data:`core.learning_isolation.MAIN_MODEL_OWNER`.
    The directory is resolved once, from the isolation policy, and the store never
    accepts a path from a caller -- which is what makes "a character writes only to
    its own store" a property of the code rather than a promise.
    """

    def __init__(
        self,
        owner: str,
        *,
        root: Path | None = None,
        audit: LearningAudit | None = None,
    ) -> None:
        assert_owner(owner)
        # The store's identity is the *owner* name (``jarvis``), whatever vocabulary
        # the caller used (``sources`` or ``jarvis``): the directory and every record
        # written here then agree, and an isolation check never has to guess which
        # vocabulary it was handed.
        self.owner = owner_for(owner)
        # ``root`` overrides the *base* the isolation policy resolves from, which a
        # test uses to keep everything under a temp folder. The final path is still
        # the policy's own layout, so the isolation property is preserved.
        self.root = Path(store_dir_for(self.owner, root))
        self._audit = audit

    # ------------------------------------------------------------- locations

    @property
    def records_path(self) -> Path:
        return self.root / RECORDS_FILENAME

    @property
    def stats_path(self) -> Path:
        return self.root / STATS_FILENAME

    @property
    def audit(self) -> LearningAudit:
        """The audit to log to (the caller's, or the process-wide one)."""
        return self._audit if self._audit is not None else default_audit()

    @property
    def is_main_model(self) -> bool:
        """True when this store belongs to the project's own model."""
        return is_main_model(self.owner)

    # --------------------------------------------------------------- writing

    def append(self, record: LearningRecord) -> LearningRecord:
        """Append one record. The owner of the record must be this store's owner.

        A record whose ``owner`` is another character is **refused**, not
        silently rewritten: writing someone else's record into this store is
        exactly the cross-character write the task forbids, and it is logged as a
        refusal.
        """
        if record.owner != self.owner:
            from core.learning_isolation import LearningAccessError

            error = LearningAccessError(
                f"{self.owner} cannot write a record owned by {record.owner}."
            )
            self.audit.log_denied(self.owner, record.owner, str(error))
            raise error
        self._append_line(record.to_primitive())
        self.audit.log(
            EVENT_LEARNED,
            actor=self.owner,
            owner=self.owner,
            detail=record.describe(),
            record_id=record.record_id,
        )
        self._refresh_stats()
        return record

    def append_many(self, records: tuple[LearningRecord, ...]) -> int:
        """Append several records in one write. Returns how many were stored."""
        stored = 0
        payload = []
        for record in records:
            if record.owner != self.owner:
                from core.learning_isolation import LearningAccessError

                error = LearningAccessError(
                    f"{self.owner} cannot write a record owned by {record.owner}."
                )
                self.audit.log_denied(self.owner, record.owner, str(error))
                raise error
            payload.append(record.to_primitive())
            stored += 1
        for item in payload:
            self._append_line(item)
        for item in payload:
            self.audit.log(
                EVENT_LEARNED,
                actor=self.owner,
                owner=self.owner,
                detail=str(item.get("knowledge") or item.get("answer") or ""),
                record_id=str(item.get("record_id") or ""),
            )
        if stored:
            self._refresh_stats()
        return stored

    def revise(
        self,
        record_id: str,
        status: str,
        *,
        note: str = "",
        confidence: float | None = None,
    ) -> LearningRecord | None:
        """Supersede ``record_id`` with a new version in ``status``.

        The original record is **not** edited: a new line is appended carrying the
        new status, the reason (``note``), and ``supersedes=<record_id>``, so the
        history of a conclusion survives. Returns the new record, or ``None`` when
        the id is unknown (an honest "nothing to revise" rather than a guess).
        """
        original = self.find(record_id)
        if original is None:
            return None
        revised = original.with_status(status, note=note, confidence=confidence)
        from dataclasses import replace

        revised = replace(revised, supersedes=record_id, record_id=_new_id())
        self._append_line(revised.to_primitive())
        self.audit.log(
            EVENT_STATUS,
            actor=self.owner,
            owner=self.owner,
            detail=f"{record_id} -> {status} ({note})" if note else f"{record_id} -> {status}",
            record_id=revised.record_id,
        )
        self._refresh_stats()
        return revised

    def clear(self) -> None:
        """Forget everything in **this** store (the task's explicit reset).

        The caller must have already asked the user for confirmation; the store
        itself only performs the reset. The reset is logged, and the stats file is
        rewritten so a statistics view is immediately honest.
        """
        with best_effort("reset a learning store"):
            if self.records_path.exists():
                self.records_path.unlink()
        self.audit.log(EVENT_RESET, actor=self.owner, owner=self.owner, detail="store reset.")
        self._write_stats(LearningStats(owner=self.owner))

    # --------------------------------------------------------------- reading

    def recent(self, *, limit: int = DEFAULT_LIMIT) -> tuple[LearningRecord, ...]:
        """The most recent ``limit`` records, oldest first within the window.

        Streaming and bounded: the file is read line by line and only the tail is
        kept, so a long store is never held whole in memory.
        """
        if limit <= 0:
            return ()
        from collections import deque

        window: deque[LearningRecord] = deque(maxlen=limit)
        for record in self._iter():
            window.append(record)
        return tuple(window)

    def all(self) -> tuple[LearningRecord, ...]:
        """Every record, oldest first. For a small store, or an export.

        A store is bounded by the number of things a character actually learned, so
        this is safe for an export or a search; a *statistics* view uses
        :meth:`stats` instead, which is one small read.
        """
        return tuple(self._iter())

    def find(self, record_id: str) -> LearningRecord | None:
        """The most recent record with ``record_id`` (its latest revision)."""
        wanted = record_id.strip()
        if not wanted:
            return None
        found: LearningRecord | None = None
        for record in self._iter():
            if record.record_id == wanted:
                found = record
        return found

    def search(self, query: str, *, limit: int = 50) -> tuple[LearningRecord, ...]:
        """Records whose knowledge, answer, or question contains ``query``.

        A simple, honest substring search over the store's own records -- no index,
        no fuzzy matching -- so it can never return something that is not there.
        """
        needle = query.strip().casefold()
        if not needle:
            return ()
        hits: list[LearningRecord] = []
        for record in self._iter():
            haystack = " ".join(
                (record.knowledge, record.answer, record.question, record.source_reference)
            ).casefold()
            if needle in haystack:
                hits.append(record)
        return tuple(hits[-limit:])

    def trusted(self, *, limit: int = DEFAULT_LIMIT) -> tuple[LearningRecord, ...]:
        """Only the records that count as knowledge (provisional or verified)."""
        return tuple(record for record in self.recent(limit=limit) if record.trusted)

    def export(self) -> dict[str, Any]:
        """The whole store as plain data, for a backup or a transfer."""
        return {
            "owner": self.owner,
            "format": 1,
            "records": [record.to_primitive() for record in self._iter()],
        }

    def stats(self) -> LearningStats:
        """The store's statistics, read from the small stats file when present.

        When the file is missing (a store written by an older build, or one whose
        stats write failed), the counts are recomputed by streaming the records --
        the slow path is the fallback, never the default.
        """
        if self.stats_path.is_file():
            with best_effort("read learning stats"):
                data = json.loads(self.stats_path.read_text(encoding="utf-8"))
                return LearningStats.from_primitive(self.owner, data)
        return self._compute_stats()

    # -------------------------------------------------------------- internals

    def _iter(self) -> Iterator[LearningRecord]:
        """Stream the records, skipping and counting a corrupt line."""
        if not self.records_path.is_file():
            return
        with best_effort("read a learning store"):
            with self.records_path.open("r", encoding="utf-8") as handle:
                for line in handle:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        data = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if isinstance(data, dict):
                        yield LearningRecord.from_primitive(data)

    def _append_line(self, payload: dict[str, Any]) -> None:
        line = json.dumps(payload, ensure_ascii=False, sort_keys=True)
        with best_effort(f"append to {self.owner}'s learning store"):
            self.root.mkdir(parents=True, exist_ok=True)
            with self.records_path.open("a", encoding="utf-8") as handle:
                handle.write(line + "\n")

    def _compute_stats(self) -> LearningStats:
        by_status: dict[str, int] = dict.fromkeys(LEARNING_STATUSES, 0)
        by_source: dict[str, int] = {}
        total = 0
        latest = ""
        for record in self._iter():
            total += 1
            by_status[record.learning_status] = by_status.get(record.learning_status, 0) + 1
            by_source[record.source_type] = by_source.get(record.source_type, 0) + 1
            if record.created_at > latest:
                latest = record.created_at
        return LearningStats(
            owner=self.owner,
            total=total,
            by_status={k: v for k, v in by_status.items() if v},
            by_source=by_source,
            latest=latest,
        )

    def _refresh_stats(self) -> None:
        self._write_stats(self._compute_stats())

    def _write_stats(self, stats: LearningStats) -> None:
        with best_effort(f"write {self.owner}'s learning stats"):
            self.root.mkdir(parents=True, exist_ok=True)
            temporary = self.stats_path.with_suffix(".tmp")
            temporary.write_text(
                json.dumps(stats.to_primitive(), ensure_ascii=False, sort_keys=True),
                encoding="utf-8",
            )
            temporary.replace(self.stats_path)


def _new_id() -> str:
    from uuid import uuid4

    return f"LR-{uuid4().hex[:12]}"


def main_model_store(
    *, root: Path | None = None, audit: LearningAudit | None = None
) -> CharacterLearningStore:
    """The project model's own store (the one that learns from the characters)."""
    return CharacterLearningStore(MAIN_MODEL_OWNER, root=root, audit=audit)


def character_store(
    key: str, *, root: Path | None = None, audit: LearningAudit | None = None
) -> CharacterLearningStore:
    """One character's own store. Refuses an unknown key."""
    return CharacterLearningStore(key, root=root, audit=audit)


__all__ = [
    "DEFAULT_LIMIT",
    "RECORDS_FILENAME",
    "STATS_FILENAME",
    "CharacterLearningStore",
    "LearningStats",
    "character_store",
    "main_model_store",
]
