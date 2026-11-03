"""Memory management: inspect, search, correct, archive, export, back up, restore.

Phase 2 requires the memory to be *manageable*, not merely stored: a user must be
able to look at a record, see where it came from, correct it through an auditable
operation, mark it superseded, archive it, export it, back it up, restore it, and
find duplicates -- **without anything being deleted automatically**.

Two objects live here:

* :class:`MemoryManager` -- the operations over one owner's memory. Every mutating
  operation is explicit and returns a report; none is triggered by a read.
* :class:`PersonalityMemoryRegistry` -- one memory per personality, resolved
  through :mod:`core.learning_isolation`. This is what makes the task's isolation
  rule *structural*: a caller asks for a personality's memory **as** a reader, and
  the registry refuses a read the policy does not allow *before* it resolves a
  path. Natasha's memory is private, so no other personality and not even the main
  model can reach it through this interface.

Nothing here deletes a memory record. ``forget`` archives; a confirmed ``reset``
is the only thing that clears, and it is named as a destructive act.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from core.learning_isolation import (
    LearningAccessError,
    decide,
    is_owner,
    owner_for,
)
from core.long_term_memory import LongTermMemory
from core.memory_db import BackupReport, MemoryLoadReport
from core.memory_record import (
    MemoryKind,
    MemoryRecord,
    now_utc,
)
from core.memory_retrieval import DEFAULT_LIMIT, MemoryRetriever, RetrievalReport
from core.paths import memory_root_dir

#: The file name the durable memory database uses inside an owner's folder.
MEMORY_DB_NAME = "memory.sqlite3"

#: The file name a default export/backup uses.
EXPORT_NAME = "memory-export.json"
BACKUP_NAME = "memory-backup.sqlite3"

#: The memory database schema version this build writes (re-exported for docs).
MEMORY_EXPORT_VERSION = 1


@dataclass
class ExportReport:
    """What an export wrote and how much of it."""

    ok: bool
    path: str
    records: int = 0
    message: str = ""

    def describe(self) -> str:
        return f"{'ok' if self.ok else 'failed'}: {self.message} ({self.path})"


@dataclass
class DuplicateGroup:
    """A set of records with identical content terms (detected, never merged)."""

    record_ids: tuple[str, ...]
    kinds: tuple[str, ...] = ()
    owners: tuple[str, ...] = ()

    @property
    def size(self) -> int:
        return len(self.record_ids)

    def describe(self) -> str:
        return f"{self.size} records share content: {', '.join(self.record_ids)}"


@dataclass
class ManagementReport:
    """The outcome of one management operation, in one readable object."""

    action: str
    changed: int = 0
    record_ids: tuple[str, ...] = ()
    message: str = ""
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return not self.message.lower().startswith("refused")

    def describe(self) -> str:
        base = f"{self.action}: {self.changed} record(s) affected"
        return f"{base} -- {self.message}" if self.message else base


class MemoryManager:
    """The management surface over one owner's long-term memory."""

    def __init__(
        self,
        memory: LongTermMemory,
        *,
        retriever: MemoryRetriever | None = None,
        export_dir: str | Path | None = None,
    ) -> None:
        self.memory = memory
        self.retriever = retriever or MemoryRetriever(memory)
        self.export_dir = Path(export_dir) if export_dir is not None else None

    # -- inspect ------------------------------------------------------------
    def inspect(self, record_id: str) -> MemoryRecord | None:
        """One record, with everything it carries (a working item is included)."""
        return self.memory.get(record_id)

    def provenance(self, record_id: str) -> dict[str, Any]:
        """Where a record came from and how it is regarded.

        Returns the source, the reference, the confidence, the verification
        status, the timestamps, and the supersession links -- the whole chain, so
        a correction is visible rather than hidden.
        """
        record = self.memory.get(record_id)
        if record is None:
            return {}
        return {
            "record_id": record.record_id,
            "kind": record.kind.value,
            "owner": record.owner,
            "source": record.source,
            "source_reference": record.source_reference,
            "confidence": record.confidence,
            "verification": record.verification,
            "trusted": record.trusted,
            "created_at": record.created_at.isoformat(),
            "valid_from": record.valid_from.isoformat() if record.valid_from else "",
            "valid_until": record.valid_until.isoformat() if record.valid_until else "",
            "supersedes": record.supersedes,
            "superseded_by": record.superseded_by,
            "archived": record.archived,
            "entities": list(record.entities),
            "relations": [list(relation) for relation in record.relations],
            "evidence_ids": list(record.evidence_ids),
        }

    def search(
        self,
        query: str,
        *,
        kinds: Iterable[MemoryKind | str] | None = None,
        limit: int = DEFAULT_LIMIT,
        include_archived: bool = False,
        touch: bool = False,
    ) -> RetrievalReport:
        """Search the memory through the one retrieval interface.

        ``touch=False`` by default here: a *management* search is an inspection,
        not a use, so it does not reinforce anything. The answer path calls the
        retriever directly and does touch.
        """
        return self.retriever.retrieve(
            query,
            kinds=kinds,
            limit=limit,
            include_archived=include_archived,
            touch=touch,
        )

    def page(
        self,
        *,
        kind: MemoryKind | str | None = None,
        limit: int = 50,
        offset: int = 0,
        include_archived: bool = False,
        verification: str | None = None,
        created_between: tuple[datetime | None, datetime | None] | None = None,
    ) -> tuple[list[MemoryRecord], MemoryLoadReport]:
        """A bounded page for a UI list (never the whole store)."""
        return self.memory.page(
            kind=kind,
            limit=limit,
            offset=offset,
            include_archived=include_archived,
            verification=verification,
            created_between=created_between,
        )

    # -- correct / supersede ------------------------------------------------
    def correct(
        self,
        record_id: str,
        new_content: str,
        *,
        reason: str = "",
        confidence: float | None = None,
        verification: str = "provisional",
    ) -> ManagementReport:
        """Correct a record through a new, superseding record (old one kept)."""
        try:
            old, new = self.memory.correct(
                record_id,
                new_content,
                reason=reason,
                confidence=confidence,
                verification=verification,
            )
        except KeyError:
            return ManagementReport("correct", 0, (), f"refused: no record {record_id!r}")
        return ManagementReport(
            "correct",
            2,
            (old.record_id, new.record_id),
            f"superseded {old.record_id} with {new.record_id}"
            + (f" ({reason})" if reason else ""),
        )

    def mark_superseded(self, record_id: str, *, superseded_by: str = "") -> ManagementReport:
        """Mark a record superseded without writing a replacement."""
        if not self.memory.mark_superseded(record_id, superseded_by=superseded_by):
            return ManagementReport("supersede", 0, (), f"refused: no record {record_id!r}")
        return ManagementReport("supersede", 1, (record_id,), "marked superseded")

    def verify(self, record_id: str, *, status: str = "verified") -> ManagementReport:
        """Explicitly set a record's verification status (an auditable act).

        This is the *only* way a record becomes trusted. Capture never verifies.
        """
        record = self.memory.get(record_id)
        if record is None:
            return ManagementReport("verify", 0, (), f"refused: no record {record_id!r}")
        from dataclasses import replace

        from core.memory_record import VERIFICATIONS

        if status not in VERIFICATIONS:
            return ManagementReport("verify", 0, (), f"refused: unknown status {status!r}")
        self.memory.database.save(replace(record, verification=status))
        return ManagementReport("verify", 1, (record_id,), f"verification set to {status}")

    # -- archive ------------------------------------------------------------
    def archive(self, record_id: str) -> ManagementReport:
        """Archive a record: kept, and excluded from default reads."""
        if not self.memory.archive(record_id):
            return ManagementReport("archive", 0, (), f"refused: no record {record_id!r}")
        return ManagementReport("archive", 1, (record_id,), "archived")

    def archive_expired(self, *, before: datetime | None = None) -> ManagementReport:
        """Archive every record whose validity ended before ``before``."""
        moment = before or now_utc()
        expired = self.memory.expire(before=moment, archive=True)
        return ManagementReport(
            "archive-expired", len(expired), expired, f"validity ended before {moment.date()}"
        )

    # -- duplicate detection ------------------------------------------------
    def duplicates(self, *, kind: MemoryKind | str | None = None) -> tuple[DuplicateGroup, ...]:
        """Detect near-duplicates by identical content terms -- never merge them.

        A duplicate may carry different provenance or a later correction, so the
        manager only *reports* the groups; a caller decides what to do.
        """
        groups: list[DuplicateGroup] = []
        for ids in self.memory.duplicates(kind=kind):
            records = [self.memory.database.get(record_id) for record_id in ids]
            present = [record for record in records if record is not None]
            groups.append(
                DuplicateGroup(
                    record_ids=tuple(ids),
                    kinds=tuple(sorted({record.kind.value for record in present})),
                    owners=tuple(sorted({record.owner for record in present})),
                )
            )
        return tuple(groups)

    # -- export / backup / restore -----------------------------------------
    def export(self, destination: str | Path | None = None) -> ExportReport:
        """Write the memory as a portable JSON document (provenance included)."""
        target = Path(destination) if destination is not None else (
            (self.export_dir / EXPORT_NAME) if self.export_dir is not None else None
        )
        if target is None:
            return ExportReport(False, "", 0, "no export destination was given")
        target.parent.mkdir(parents=True, exist_ok=True)
        records = [record.to_primitive() for record in self.memory.all_records(include_archived=True)]
        document = {
            "version": MEMORY_EXPORT_VERSION,
            "owner": self.memory.owner,
            "exported_at": now_utc().isoformat(),
            "count": len(records),
            "records": records,
        }
        try:
            target.write_text(json.dumps(document, ensure_ascii=False, indent=2), encoding="utf-8")
        except OSError as error:
            return ExportReport(False, str(target), 0, f"export failed: {error}")
        return ExportReport(True, str(target), len(records), "exported")

    def backup(self, destination: str | Path | None = None) -> BackupReport:
        """Write a consistent copy of the database (SQLite online backup)."""
        target = Path(destination) if destination is not None else (
            (self.export_dir / BACKUP_NAME) if self.export_dir is not None else None
        )
        if target is None:
            return BackupReport(False, "", 0, "no backup destination was given")
        return self.memory.backup(target)

    def restore(self, source: str | Path) -> BackupReport:
        """Restore from a validated backup (the current file is kept as a safety copy)."""
        return self.memory.restore(source)

    # -- reset (destructive, named as such) --------------------------------
    def reset(self, *, confirmed: bool = False) -> ManagementReport:
        """Clear every durable record -- refused unless ``confirmed=True``."""
        if not confirmed:
            return ManagementReport(
                "reset", 0, (), "refused: reset is destructive and needs confirmed=True"
            )
        before = self.memory.database.count(include_archived=True)
        self.memory.database.clear()
        return ManagementReport("reset", before, (), f"cleared {before} record(s)")


class PersonalityMemoryRegistry:
    """One long-term memory per personality, isolated by the project's own policy.

    The registry resolves an owner's database path from
    :func:`core.learning_isolation.store_dir` and enforces the same access rules
    the learning stores use: a personality reads its own memory, the main model
    may read a *shareable* personality's memory, and **Natasha's memory is private**
    -- refused before any path is resolved. Each personality therefore has its own
    episodic and procedural memory, and a cross-personality read is impossible
    through this interface rather than merely discouraged.

    Shared *semantic* knowledge is the Phase-1 Cognitive Core, not this store: a
    semantic memory here is one personality's own record that *references* the
    shared graph. Reading shared knowledge never transfers a private experience.
    """

    def __init__(self, *, data_root: str | Path | None = None) -> None:
        self._data_root = Path(data_root) if data_root is not None else None
        self._stores: dict[str, LongTermMemory] = {}

    def _db_path(self, owner: str) -> Path:
        """The database path for ``owner`` -- its own isolated folder.

        ``data/memory/<owner>/memory.sqlite3`` (or ``<data_root>/memory/...`` for a
        test), so one owner's database can never be opened as another's. The owner
        name determines the folder; a caller never supplies a path.
        """
        resolved = owner_for(owner)
        base = memory_root_dir() if self._data_root is None else (self._data_root / "memory")
        return base / resolved / MEMORY_DB_NAME

    def owner_memory(self, owner: str) -> LongTermMemory:
        """The memory of ``owner`` (no access check -- this is the *own* store)."""
        resolved = owner_for(owner)
        if not is_owner(resolved):
            raise LearningAccessError(f"{owner!r} is not a known memory owner.")
        if resolved not in self._stores:
            self._stores[resolved] = LongTermMemory(resolved, self._db_path(resolved))
        return self._stores[resolved]

    def read_memory(self, reader: str, owner: str) -> LongTermMemory:
        """``owner``'s memory **as read by** ``reader`` -- refused if not permitted.

        The policy is consulted *first*, so a refused read never even resolves a
        path. Natasha's memory cannot be read by anyone but Natasha.
        """
        decision = decide(reader, owner)
        if not decision.allowed:
            raise LearningAccessError(decision.reason)
        return self.owner_memory(owner)

    def can_read(self, reader: str, owner: str) -> bool:
        """Whether ``reader`` may read ``owner``'s memory (a pure policy check)."""
        return decide(reader, owner).allowed

    def readable_owners(self, reader: str) -> tuple[str, ...]:
        """Every owner whose memory ``reader`` may read (its own first)."""
        from core.learning_isolation import OWNERS

        resolved = owner_for(reader)
        return tuple(owner for owner in OWNERS if self.can_read(resolved, owner))

    def manager_for(self, owner: str) -> MemoryManager:
        """A management surface over ``owner``'s own memory."""
        memory = self.owner_memory(owner)
        return MemoryManager(memory)

    def close(self) -> None:
        """Drop the cached stores (the databases hold no open connection)."""
        self._stores.clear()


__all__ = [
    "BACKUP_NAME",
    "EXPORT_NAME",
    "MEMORY_DB_NAME",
    "MEMORY_EXPORT_VERSION",
    "DuplicateGroup",
    "ExportReport",
    "ManagementReport",
    "MemoryManager",
    "PersonalityMemoryRegistry",
]


def memory_sequence(memory: LongTermMemory, kind: MemoryKind | str | None = None) -> Sequence[MemoryRecord]:
    """A convenience: every record of a kind, as a tuple (bounded by the store)."""
    return tuple(memory.all_records(kind=kind))
