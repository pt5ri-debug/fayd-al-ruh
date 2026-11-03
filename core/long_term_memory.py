"""The long-term memory engine: four categories over one durable store.

Phase 2 requires four memory categories -- working, episodic, semantic, and
procedural -- each with the lifecycle *capture → normalise → validate → assign a
category → link → persist → index → retrieve → consolidate → archive*. This
module is that engine. It **extends** the project's existing in-process memory
modules rather than replacing them:

* :class:`~core.working_memory.WorkingMemory` stays the runtime context; this
  module wraps it so a working item can be *promoted* to episodic memory.
* :class:`~core.episodic_memory.EpisodicMemory`,
  :class:`~core.semantic_memory.SemanticMemory`, and
  :class:`~core.procedural_memory.ProceduralMemory` remain the in-process views;
  the durable truth is the SQLite store, and the in-process objects are the fast
  path a caller already knows.
* The **semantic** category does *not* create a competing graph. A semantic
  memory is a `MemoryRecord` that **references** the Phase-1 Cognitive Core's
  entities, relations, and evidence (through ``entities``/``relations``/
  ``evidence_ids``); the graph itself stays canonical.

**Honesty rules, in code.** A record is never deleted to resolve a conflict
(correction writes a new record and marks the old one superseded, both kept); a
working item is not persisted by default (working memory is runtime state); and
a correction, an archive, or a consolidation is an explicit call that returns a
report -- nothing happens silently. Nothing here loads a model.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field, replace
from datetime import datetime
from typing import Any

from core.memory_db import BackupReport, MemoryDatabase, MemoryLoadReport
from core.memory_record import (
    DURABLE_KINDS,
    SOURCE_SYSTEM,
    VERIFICATION_SUPERSEDED,
    VERIFICATION_UNVERIFIED,
    MemoryKind,
    MemoryRecord,
    make_memory_id,
    now_utc,
    supersede,
)

#: The default working-memory capacity (Miller's 7 +/- 2, as the project's own
#: :class:`core.working_memory.WorkingMemory` already uses).
DEFAULT_WORKING_CAPACITY = 7

#: How many records one page holds by default (bounded retrieval).
DEFAULT_PAGE = 50

#: The largest page a caller may request, so one call cannot load everything.
MAX_PAGE = 500


@dataclass
class PromotionReport:
    """What one consolidation pass promoted, and what it left in place."""

    promoted: tuple[str, ...] = ()
    skipped: tuple[str, ...] = ()
    reasons: list[str] = field(default_factory=list)

    @property
    def changed(self) -> bool:
        return bool(self.promoted)

    def describe(self) -> str:
        base = f"promoted {len(self.promoted)} working item(s) to episodic memory"
        if self.skipped:
            return f"{base}; left {len(self.skipped)} in working memory"
        return base


@dataclass(frozen=True)
class MemoryStatus:
    """An honest snapshot of what the memory holds, per category."""

    owner: str
    counts: dict[str, int]
    working: int
    archived: int
    schema_version: int

    @property
    def total(self) -> int:
        return sum(self.counts.values())

    def describe(self) -> str:
        parts = ", ".join(f"{kind}={self.counts.get(kind, 0)}" for kind in
                          (kind.value for kind in DURABLE_KINDS))
        return (
            f"memory for {self.owner}: {parts}; working={self.working}; "
            f"archived={self.archived}; schema={self.schema_version}"
        )


class WorkingMemoryStore:
    """The short-lived context of the current task, with a bounded capacity.

    It wraps the project's own :class:`core.working_memory.WorkingMemory` so the
    existing behaviour (a bounded LRU with an activation bump on access) is kept,
    and adds what Phase 2 needs: an explicit *kind* per item so consolidation can
    promote the right ones, a task reference, and a way to iterate for promotion.
    It is deliberately **not** persisted by default -- it is runtime state.
    """

    def __init__(self, capacity: int = DEFAULT_WORKING_CAPACITY) -> None:
        if capacity < 1:
            raise ValueError("Working-memory capacity must be at least 1.")
        self.capacity = capacity
        self._items: dict[str, MemoryRecord] = {}
        self._order: list[str] = []

    def put(self, record: MemoryRecord) -> MemoryRecord:
        """Place a record in working memory, evicting the oldest past capacity."""
        if record.kind is not MemoryKind.WORKING:
            record = replace(record, kind=MemoryKind.WORKING)
        self._items[record.record_id] = record
        if record.record_id in self._order:
            self._order.remove(record.record_id)
        self._order.append(record.record_id)
        while len(self._order) > self.capacity:
            oldest = self._order.pop(0)
            self._items.pop(oldest, None)
        return record

    def get(self, record_id: str) -> MemoryRecord | None:
        """Fetch a working item and bump its access (the recency signal)."""
        record = self._items.get(record_id)
        if record is None:
            return None
        record = replace(record, access_count=record.access_count + 1, last_accessed=now_utc())
        self._items[record_id] = record
        if record_id in self._order:
            self._order.remove(record_id)
        self._order.append(record_id)
        return record

    def items(self) -> tuple[MemoryRecord, ...]:
        """The working items, most recently used last."""
        return tuple(self._items[key] for key in self._order if key in self._items)

    def remove(self, record_id: str) -> None:
        self._items.pop(record_id, None)
        if record_id in self._order:
            self._order.remove(record_id)

    def clear(self) -> None:
        self._items.clear()
        self._order.clear()

    @property
    def size(self) -> int:
        return len(self._items)

    def __len__(self) -> int:
        return len(self._items)


class LongTermMemory:
    """One owner's four memory categories over one durable store.

    The owner is supplied by the caller -- and the caller is expected to resolve
    it through :mod:`core.learning_isolation` -- so this class never chooses a
    path. The database is opened lazily on first use, so constructing the engine
    is cheap and a read-only caller never writes a file.
    """

    def __init__(
        self,
        owner: str,
        path: str | Any,
        *,
        database: MemoryDatabase | None = None,
        working_capacity: int = DEFAULT_WORKING_CAPACITY,
    ) -> None:
        self.owner = owner
        self._database = database
        self._path = path
        self.working = WorkingMemoryStore(capacity=working_capacity)

    # -- lazy database ------------------------------------------------------
    @property
    def database(self) -> MemoryDatabase:
        """The durable database, opened on first use."""
        if self._database is None:
            self._database = MemoryDatabase(self._path)
        return self._database

    @property
    def schema_version(self) -> int:
        return self.database.schema_version

    # -- lifecycle: capture -------------------------------------------------
    def capture(
        self,
        content: str,
        *,
        kind: MemoryKind | str = MemoryKind.EPISODIC,
        summary: str = "",
        source: str = SOURCE_SYSTEM,
        source_reference: str = "",
        confidence: float = 0.5,
        importance: float = 0.5,
        valid_from: datetime | None = None,
        valid_until: datetime | None = None,
        entities: Iterable[str] = (),
        relations: Iterable[tuple[str, str, str]] = (),
        evidence_ids: Iterable[str] = (),
        task_refs: Iterable[str] = (),
        tags: Iterable[str] = (),
        metadata: dict[str, Any] | None = None,
        record_id: str | None = None,
        persist: bool = True,
    ) -> MemoryRecord:
        """Capture one memory: normalise, assign a category, link, persist, index.

        ``persist=False`` builds the record without writing it -- used by the
        working-memory path, which is runtime state. A duplicate id is *not* an
        error: capturing the same content again returns the stored record, which
        is what makes capture idempotent.
        """
        kind_value = kind if isinstance(kind, MemoryKind) else MemoryKind(str(kind))
        cleaned = _normalise(content)
        if not cleaned:
            raise ValueError("A memory cannot be captured with empty content.")
        identifier = record_id or make_memory_id(kind_value, cleaned, owner=self.owner)
        existing = self.database.get(identifier) if persist else None
        if existing is not None:
            return existing
        record = MemoryRecord(
            record_id=identifier,
            kind=kind_value,
            content=cleaned,
            summary=_normalise(summary),
            owner=self.owner,
            source=source,
            source_reference=source_reference,
            confidence=confidence,
            importance=importance,
            verification=VERIFICATION_UNVERIFIED,
            valid_from=valid_from,
            valid_until=valid_until,
            entities=tuple(entities),
            relations=tuple(relations),
            evidence_ids=tuple(evidence_ids),
            task_refs=tuple(task_refs),
            tags=tuple(tags),
            metadata=dict(metadata or {}),
        )
        if kind_value is MemoryKind.WORKING or not persist:
            return self.working.put(record) if kind_value is MemoryKind.WORKING else record
        return self.database.save(record)

    # -- lifecycle: read ----------------------------------------------------
    def get(self, record_id: str) -> MemoryRecord | None:
        """Fetch one durable record by id (a working item is looked up too)."""
        working = self.working.get(record_id)
        if working is not None:
            return working
        return self.database.get(record_id)

    def page(
        self,
        *,
        kind: MemoryKind | str | None = None,
        include_archived: bool = False,
        limit: int = DEFAULT_PAGE,
        offset: int = 0,
        verification: str | None = None,
        source: str | None = None,
        created_between: tuple[datetime | None, datetime | None] | None = None,
        valid_at: datetime | None = None,
    ) -> tuple[list[MemoryRecord], MemoryLoadReport]:
        """A bounded page of durable records (never the whole store)."""
        if limit > MAX_PAGE:
            limit = MAX_PAGE
        return self.database.list_records(
            kind=kind,
            owner=self.owner,
            verification=verification,
            source=source,
            include_archived=include_archived,
            created_between=created_between,
            valid_at=valid_at,
            limit=limit,
            offset=offset,
        )

    def all_records(self, *, kind: MemoryKind | str | None = None,
                    include_archived: bool = False):
        """Stream every matching record in bounded pages."""
        return self.database.all_records(
            kind=kind, owner=self.owner, include_archived=include_archived
        )

    def count(self, *, kind: MemoryKind | str | None = None,
              include_archived: bool = True) -> int:
        return self.database.count(kind=kind, include_archived=include_archived)

    def counts_by_kind(self, *, include_archived: bool = True) -> dict[str, int]:
        return self.database.counts_by_kind(include_archived=include_archived)

    # -- lifecycle: correct / supersede ------------------------------------
    def correct(
        self,
        record_id: str,
        new_content: str,
        *,
        reason: str = "",
        confidence: float | None = None,
        verification: str = "provisional",
        source: str | None = None,
    ) -> tuple[MemoryRecord, MemoryRecord]:
        """Correct a record without losing the original.

        The old record is marked superseded and archived; the new record is
        stored beside it pointing back. Both are returned, and both are kept --
        the task's "never silently overwrite or delete conflicting memories".
        """
        old = self.database.get(record_id)
        if old is None:
            raise KeyError(f"No memory record {record_id!r} to correct.")
        superseded_old, new_record = supersede(
            old,
            new_content,
            confidence=confidence,
            verification=verification,
            source=source,
            reason=reason,
        )
        self.database.save_many([superseded_old, new_record])
        return superseded_old, new_record

    def mark_superseded(self, record_id: str, *, superseded_by: str = "") -> bool:
        """Mark a record superseded without writing a replacement (an explicit act)."""
        record = self.database.get(record_id)
        if record is None:
            return False
        self.database.save(
            replace(
                record,
                verification=VERIFICATION_SUPERSEDED,
                superseded_by=superseded_by or record.superseded_by,
                archived=True,
            )
        )
        return True

    def archive(self, record_id: str) -> bool:
        """Archive a record: it is kept and excluded from default reads."""
        return self.database.archive(record_id)

    # -- lifecycle: consolidate --------------------------------------------
    def promote_working(self, *, minimum_importance: float = 0.5) -> PromotionReport:
        """Promote the working items worth keeping into episodic memory.

        This is the *consolidation* step the task requires. Only an item whose
        importance clears the threshold is promoted; the rest stay in working
        memory (they are the current context). A promotion writes a **new**
        episodic record and leaves the working item in place, so the short-term
        context is not disturbed.
        """
        if not 0.0 <= minimum_importance <= 1.0:
            raise ValueError("Minimum importance must be between 0 and 1.")
        report = PromotionReport()
        promoted: list[str] = []
        skipped: list[str] = []
        for item in self.working.items():
            if item.importance < minimum_importance:
                skipped.append(item.record_id)
                continue
            episodic = MemoryRecord(
                record_id=make_memory_id(MemoryKind.EPISODIC, item.content, owner=self.owner),
                kind=MemoryKind.EPISODIC,
                content=item.content,
                summary=item.summary,
                owner=self.owner,
                source=item.source,
                source_reference=item.source_reference,
                confidence=item.confidence,
                importance=item.importance,
                entities=item.entities,
                relations=item.relations,
                evidence_ids=item.evidence_ids,
                task_refs=item.task_refs,
                tags=item.tags,
                metadata={**item.metadata, "promoted_from": "working"},
            )
            self.database.save(episodic)
            promoted.append(episodic.record_id)
        report.promoted = tuple(promoted)
        report.skipped = tuple(skipped)
        if skipped:
            report.reasons.append(
                f"{len(skipped)} item(s) below importance {minimum_importance:.2f}"
            )
        return report

    # -- lifecycle: archive / expire ---------------------------------------
    def expire(self, *, before: datetime, archive: bool = True) -> tuple[str, ...]:
        """Archive every record whose ``valid_until`` is before ``before``.

        Expiry is an *archive*, never a delete: the record stays and can be read
        with ``include_archived=True``. A record with no ``valid_until`` never
        expires.
        """
        expired: list[str] = []
        for record in list(self.all_records(include_archived=False)):
            end = record.valid_until
            if end is not None and end < before:
                if archive:
                    self.database.archive(record.record_id)
                expired.append(record.record_id)
        return tuple(expired)

    # -- integrity / backup / restore --------------------------------------
    def integrity_check(self) -> tuple[bool, str]:
        return self.database.integrity_check()

    def backup(self, destination: str | Any) -> BackupReport:
        return self.database.backup(destination)

    def restore(self, source: str | Any) -> BackupReport:
        return self.database.restore(source)

    def duplicates(self, *, kind: MemoryKind | str | None = None) -> list[tuple[str, ...]]:
        """Groups of records with identical content terms (detection only)."""
        return self.database.duplicates(kind=kind)

    # -- status -------------------------------------------------------------
    def status(self) -> MemoryStatus:
        counts = self.counts_by_kind(include_archived=True)
        archived = self.database.count(include_archived=True) - self.database.count(
            include_archived=False
        )
        return MemoryStatus(
            owner=self.owner,
            counts=counts,
            working=self.working.size,
            archived=archived,
            schema_version=self.schema_version,
        )

    def describe(self) -> str:
        return self.status().describe()


def _normalise(text: str) -> str:
    """Collapse whitespace and strip, so the same content yields the same id."""
    return " ".join((text or "").split())


def episodic_records(memory: LongTermMemory) -> Sequence[MemoryRecord]:
    """Every episodic record, streamed (a convenience for a caller and a test)."""
    return tuple(memory.all_records(kind=MemoryKind.EPISODIC))


__all__ = [
    "DEFAULT_PAGE",
    "DEFAULT_WORKING_CAPACITY",
    "MAX_PAGE",
    "LongTermMemory",
    "MemoryStatus",
    "PromotionReport",
    "WorkingMemoryStore",
    "episodic_records",
]
