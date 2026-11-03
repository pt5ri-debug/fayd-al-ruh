"""The durable memory database: one SQLite file per owner, versioned and atomic.

This is the persistence layer Phase 2 adds. It is deliberately built on
``sqlite3`` from the **standard library** -- the project's own
``storage/sqlite_store.py`` proves the pattern -- so it adds no dependency, works
on Windows 11, and runs entirely locally.

Design decisions, each answering a Phase-2 requirement:

* **Transactional writes.** Every write runs inside a transaction; a failure
  rolls back, so a half-written memory is never left behind.
* **Schema versioning and migrations.** ``meta.schema_version`` records the
  version on disk and :meth:`MemoryDatabase.migrate` applies the numbered steps,
  each in its own transaction. A database from a *newer* build is refused rather
  than read with a schema this build does not understand.
* **Bounded reads.** :meth:`MemoryDatabase.list_records` pages with ``LIMIT`` /
  ``OFFSET`` and never materialises the whole table, so a large memory does not
  have to fit in RAM.
* **Corrupt-record tolerance.** A row whose JSON payload will not parse is
  *skipped and counted* in a :class:`MemoryLoadReport`, never a crash.
* **Backup and recovery.** :meth:`backup` uses SQLite's own online backup API to
  write a consistent copy; :meth:`restore` validates the copy before replacing
  anything, so a bad backup cannot destroy a good database.

Nothing here loads a model or reaches the network.
"""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from core.memory_record import (
    DURABLE_KINDS,
    MemoryKind,
    MemoryRecord,
    make_memory_id,
)

#: The schema version this build writes and understands.
SCHEMA_VERSION = 1

#: Tables are created with ``IF NOT EXISTS`` so opening an empty file is safe.
_SCHEMA = """
CREATE TABLE IF NOT EXISTS meta (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS memories (
    record_id TEXT PRIMARY KEY,
    kind TEXT NOT NULL,
    owner TEXT NOT NULL,
    source TEXT NOT NULL,
    verification TEXT NOT NULL,
    confidence REAL NOT NULL,
    importance REAL NOT NULL,
    created_at TEXT NOT NULL,
    valid_from TEXT,
    valid_until TEXT,
    access_count INTEGER NOT NULL DEFAULT 0,
    last_accessed TEXT,
    superseded_by TEXT NOT NULL DEFAULT '',
    supersedes TEXT NOT NULL DEFAULT '',
    archived INTEGER NOT NULL DEFAULT 0,
    payload TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_memories_owner_kind ON memories(owner, kind);
CREATE INDEX IF NOT EXISTS idx_memories_kind ON memories(kind);
CREATE INDEX IF NOT EXISTS idx_memories_created ON memories(created_at);
CREATE INDEX IF NOT EXISTS idx_memories_verification ON memories(verification);
CREATE TABLE IF NOT EXISTS memory_terms (
    record_id TEXT NOT NULL,
    term TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_memory_terms_term ON memory_terms(term);
CREATE INDEX IF NOT EXISTS idx_memory_terms_record ON memory_terms(record_id);
"""


class MemoryDatabaseError(RuntimeError):
    """A memory-database failure a caller must handle (never swallowed)."""


@dataclass
class MemoryLoadReport:
    """What one read actually did -- including what it could not read."""

    loaded: int = 0
    skipped: int = 0
    reasons: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        """True when nothing was skipped."""
        return self.skipped == 0

    def describe(self) -> str:
        base = f"loaded {self.loaded} record(s)"
        if self.skipped:
            return f"{base}; skipped {self.skipped} unreadable row(s)"
        return base


@dataclass(frozen=True)
class BackupReport:
    """The outcome of a backup or restore, with the real path involved."""

    ok: bool
    path: str
    records: int = 0
    message: str = ""

    def describe(self) -> str:
        return f"{'ok' if self.ok else 'failed'}: {self.message} ({self.path})"


class MemoryDatabase:
    """One owner's durable memory store.

    The path is supplied by the caller -- and the caller is
    :class:`core.long_term_memory.LongTermMemory`, which resolves it from the
    isolation policy -- so a database never chooses where another owner's records
    live.
    """

    def __init__(self, path: str | Path, *, timeout: float = 10.0) -> None:
        self.path = Path(path)
        self.timeout = timeout
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    # -- connection / schema ------------------------------------------------
    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path, timeout=self.timeout)
        connection.row_factory = sqlite3.Row
        # WAL keeps a reader from blocking a writer; ``foreign_keys`` is off
        # because the only link between rows is ``superseded_by``, a soft pointer.
        try:
            connection.execute("PRAGMA journal_mode=WAL")
        except sqlite3.DatabaseError:
            # A filesystem that cannot do WAL (some network shares) still works
            # with the default journal; the task says local-only, but a failure
            # here must not stop the store from opening.
            pass
        return connection

    @contextmanager
    def _transaction(self) -> Iterator[sqlite3.Connection]:
        """Run a block in one transaction; roll back on any error."""
        connection = self._connect()
        try:
            connection.execute("BEGIN")
            yield connection
            connection.commit()
        except BaseException:
            connection.rollback()
            raise
        finally:
            connection.close()

    def _initialize(self) -> None:
        connection = self._connect()
        try:
            connection.executescript(_SCHEMA)
            row = connection.execute(
                "SELECT value FROM meta WHERE key='schema_version'"
            ).fetchone()
            if row is None:
                connection.execute(
                    "INSERT INTO meta(key, value) VALUES('schema_version', ?)",
                    (str(SCHEMA_VERSION),),
                )
            connection.commit()
        finally:
            connection.close()
        self.migrate()

    @property
    def schema_version(self) -> int:
        """The schema version recorded on disk."""
        connection = self._connect()
        try:
            row = connection.execute(
                "SELECT value FROM meta WHERE key='schema_version'"
            ).fetchone()
        finally:
            connection.close()
        try:
            return int(row["value"]) if row is not None else SCHEMA_VERSION
        except (TypeError, ValueError):
            return SCHEMA_VERSION

    def migrate(self) -> int:
        """Bring the database up to :data:`SCHEMA_VERSION`.

        Returns the version now on disk. A database written by a *newer* build is
        refused: reading it with an older schema would silently mis-handle a
        column this build does not know. There is one migration step today (the
        initial schema), so the loop is a no-op on a fresh file -- but the
        mechanism is in place, which is the requirement.
        """
        current = self.schema_version
        if current > SCHEMA_VERSION:
            raise MemoryDatabaseError(
                f"The memory database was written by a newer build "
                f"(schema {current} > {SCHEMA_VERSION}); refusing to read it."
            )
        if current == SCHEMA_VERSION:
            return current
        # Future migrations append numbered steps here, each in its own
        # transaction, so a failure leaves the previous version intact.
        for target in range(current + 1, SCHEMA_VERSION + 1):
            with self._transaction() as connection:
                connection.execute(
                    "INSERT INTO meta(key, value) VALUES('schema_version', ?) "
                    "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
                    (str(target),),
                )
        return self.schema_version

    # -- writes -------------------------------------------------------------
    def save(self, record: MemoryRecord) -> MemoryRecord:
        """Insert or replace one record and its search terms, atomically.

        Replacing is how a *revision* lands: the caller stores the new record and
        the superseded old one together through
        :meth:`core.long_term_memory.LongTermMemory.correct`, so both are kept.
        """
        payload = json.dumps(record.to_primitive(), ensure_ascii=False, sort_keys=True)
        terms = sorted(record.terms)
        with self._transaction() as connection:
            connection.execute(
                """
                INSERT INTO memories(
                    record_id, kind, owner, source, verification, confidence,
                    importance, created_at, valid_from, valid_until, access_count,
                    last_accessed, superseded_by, supersedes, archived, payload
                ) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                ON CONFLICT(record_id) DO UPDATE SET
                    kind=excluded.kind, owner=excluded.owner, source=excluded.source,
                    verification=excluded.verification, confidence=excluded.confidence,
                    importance=excluded.importance, created_at=excluded.created_at,
                    valid_from=excluded.valid_from, valid_until=excluded.valid_until,
                    access_count=excluded.access_count, last_accessed=excluded.last_accessed,
                    superseded_by=excluded.superseded_by, supersedes=excluded.supersedes,
                    archived=excluded.archived, payload=excluded.payload
                """,
                (
                    record.record_id,
                    record.kind.value,
                    record.owner,
                    record.source,
                    record.verification,
                    record.confidence,
                    record.importance,
                    record.created_at.isoformat(),
                    record.valid_from.isoformat() if record.valid_from else None,
                    record.valid_until.isoformat() if record.valid_until else None,
                    record.access_count,
                    record.last_accessed.isoformat() if record.last_accessed else None,
                    record.superseded_by,
                    record.supersedes,
                    1 if record.archived else 0,
                    payload,
                ),
            )
            connection.execute(
                "DELETE FROM memory_terms WHERE record_id=?", (record.record_id,)
            )
            connection.executemany(
                "INSERT INTO memory_terms(record_id, term) VALUES(?,?)",
                [(record.record_id, term) for term in terms],
            )
        return record

    def save_many(self, records: Sequence[MemoryRecord]) -> int:
        """Store several records in **one** transaction (all or nothing)."""
        if not records:
            return 0
        payloads = [
            (
                record,
                json.dumps(record.to_primitive(), ensure_ascii=False, sort_keys=True),
                sorted(record.terms),
            )
            for record in records
        ]
        with self._transaction() as connection:
            for record, payload, terms in payloads:
                connection.execute(
                    """
                    INSERT INTO memories(
                        record_id, kind, owner, source, verification, confidence,
                        importance, created_at, valid_from, valid_until, access_count,
                        last_accessed, superseded_by, supersedes, archived, payload
                    ) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                    ON CONFLICT(record_id) DO UPDATE SET
                        kind=excluded.kind, owner=excluded.owner, source=excluded.source,
                        verification=excluded.verification, confidence=excluded.confidence,
                        importance=excluded.importance, created_at=excluded.created_at,
                        valid_from=excluded.valid_from, valid_until=excluded.valid_until,
                        access_count=excluded.access_count, last_accessed=excluded.last_accessed,
                        superseded_by=excluded.superseded_by, supersedes=excluded.supersedes,
                        archived=excluded.archived, payload=excluded.payload
                    """,
                    (
                        record.record_id,
                        record.kind.value,
                        record.owner,
                        record.source,
                        record.verification,
                        record.confidence,
                        record.importance,
                        record.created_at.isoformat(),
                        record.valid_from.isoformat() if record.valid_from else None,
                        record.valid_until.isoformat() if record.valid_until else None,
                        record.access_count,
                        record.last_accessed.isoformat() if record.last_accessed else None,
                        record.superseded_by,
                        record.supersedes,
                        1 if record.archived else 0,
                        payload,
                    ),
                )
                connection.execute(
                    "DELETE FROM memory_terms WHERE record_id=?", (record.record_id,)
                )
                connection.executemany(
                    "INSERT INTO memory_terms(record_id, term) VALUES(?,?)",
                    [(record.record_id, term) for term in terms],
                )
        return len(records)

    def touch(self, record_ids: Sequence[str], *, moment: datetime | None = None) -> int:
        """Increment the access count of the records a retrieval returned.

        This is the *only* reinforcement the store applies, and it is bounded by
        the caller (one increment per retrieval), so there is no feedback loop.
        """
        if not record_ids:
            return 0
        stamp = (moment or datetime.now(UTC)).isoformat()
        with self._transaction() as connection:
            connection.executemany(
                "UPDATE memories SET access_count=access_count+1, last_accessed=? "
                "WHERE record_id=?",
                [(stamp, record_id) for record_id in record_ids],
            )
        # The payload holds the authoritative record; keep it in step with the
        # indexed columns so a read-back agrees with the counters.
        for record_id in record_ids:
            record = self.get(record_id)
            if record is not None:
                record = _with_access(record, stamp)
                self.save(record)
        return len(record_ids)

    def archive(self, record_id: str) -> bool:
        """Mark a record archived (kept, never deleted). Returns whether it existed."""
        record = self.get(record_id)
        if record is None:
            return False
        from dataclasses import replace

        self.save(replace(record, archived=True))
        return True

    # -- reads --------------------------------------------------------------
    def get(self, record_id: str) -> MemoryRecord | None:
        """One record by id, or ``None`` (a corrupt row reads as ``None``)."""
        connection = self._connect()
        try:
            row = connection.execute(
                "SELECT payload FROM memories WHERE record_id=?", (record_id,)
            ).fetchone()
        finally:
            connection.close()
        if row is None:
            return None
        return _record_from_payload(row["payload"])

    def exists(self, record_id: str) -> bool:
        """Whether a record id is present (cheap, no payload parse)."""
        connection = self._connect()
        try:
            row = connection.execute(
                "SELECT 1 FROM memories WHERE record_id=? LIMIT 1", (record_id,)
            ).fetchone()
        finally:
            connection.close()
        return row is not None

    def list_records(
        self,
        *,
        kind: MemoryKind | str | None = None,
        owner: str | None = None,
        verification: str | None = None,
        source: str | None = None,
        include_archived: bool = False,
        created_between: tuple[datetime | None, datetime | None] | None = None,
        valid_at: datetime | None = None,
        limit: int = 100,
        offset: int = 0,
        newest_first: bool = True,
    ) -> tuple[list[MemoryRecord], MemoryLoadReport]:
        """A **bounded page** of records plus what could not be read.

        The whole table is never materialised: SQLite applies the filters, the
        ordering, and the ``LIMIT``/``OFFSET``, and only the page's payloads are
        parsed. A row whose payload will not parse is skipped and counted.
        """
        if limit < 1:
            raise ValueError("A memory page must have a positive limit.")
        if offset < 0:
            raise ValueError("A memory page offset cannot be negative.")
        clauses: list[str] = []
        params: list[Any] = []
        if kind is not None:
            clauses.append("kind=?")
            params.append(kind.value if isinstance(kind, MemoryKind) else str(kind))
        if owner is not None:
            clauses.append("owner=?")
            params.append(owner)
        if verification is not None:
            clauses.append("verification=?")
            params.append(verification)
        if source is not None:
            clauses.append("source=?")
            params.append(source)
        if not include_archived:
            clauses.append("archived=0")
        if created_between is not None:
            start, end = created_between
            if start is not None:
                clauses.append("created_at>=?")
                params.append(start.isoformat())
            if end is not None:
                clauses.append("created_at<=?")
                params.append(end.isoformat())
        if valid_at is not None:
            stamp = valid_at.isoformat()
            clauses.append("(valid_from IS NULL OR valid_from<=?)")
            params.append(stamp)
            clauses.append("(valid_until IS NULL OR valid_until>=?)")
            params.append(stamp)
        where = f" WHERE {' AND '.join(clauses)}" if clauses else ""
        order = "DESC" if newest_first else "ASC"
        query = (
            f"SELECT payload FROM memories{where} "
            f"ORDER BY created_at {order}, record_id ASC LIMIT ? OFFSET ?"
        )
        params.extend([limit, offset])
        connection = self._connect()
        try:
            rows = connection.execute(query, params).fetchall()
        finally:
            connection.close()
        records: list[MemoryRecord] = []
        report = MemoryLoadReport()
        for row in rows:
            record = _record_from_payload(row["payload"])
            if record is None:
                report.skipped += 1
                if len(report.reasons) < 10:
                    report.reasons.append("a row payload could not be parsed")
                continue
            records.append(record)
        report.loaded = len(records)
        return records, report

    def all_records(
        self, *, kind: MemoryKind | str | None = None, owner: str | None = None,
        include_archived: bool = False, page_size: int = 500,
    ) -> Iterator[MemoryRecord]:
        """Stream every matching record in bounded pages (never all at once)."""
        offset = 0
        while True:
            page, _ = self.list_records(
                kind=kind, owner=owner, include_archived=include_archived,
                limit=page_size, offset=offset,
            )
            if not page:
                return
            yield from page
            if len(page) < page_size:
                return
            offset += page_size

    def search_terms(self, terms: Sequence[str], *, limit: int = 50) -> list[str]:
        """Record ids that share **any** of ``terms``, ordered by match count.

        The index is a plain term table (no FTS extension), so this works on every
        SQLite build including the one bundled with Python on Windows. It is the
        exact/keyword leg of retrieval; the caller fuses it with the other legs.
        """
        cleaned = sorted({term for term in terms if term})
        if not cleaned:
            return []
        placeholders = ",".join("?" for _ in cleaned)
        connection = self._connect()
        try:
            rows = connection.execute(
                f"""
                SELECT record_id, COUNT(*) AS hits FROM memory_terms
                WHERE term IN ({placeholders})
                GROUP BY record_id
                ORDER BY hits DESC, record_id ASC
                LIMIT ?
                """,
                [*cleaned, limit],
            ).fetchall()
        finally:
            connection.close()
        return [row["record_id"] for row in rows]

    def count(self, *, kind: MemoryKind | str | None = None, include_archived: bool = True) -> int:
        """How many records match -- used by the status line and the gate."""
        clauses: list[str] = []
        params: list[Any] = []
        if kind is not None:
            clauses.append("kind=?")
            params.append(kind.value if isinstance(kind, MemoryKind) else str(kind))
        if not include_archived:
            clauses.append("archived=0")
        where = f" WHERE {' AND '.join(clauses)}" if clauses else ""
        connection = self._connect()
        try:
            row = connection.execute(
                f"SELECT COUNT(*) AS n FROM memories{where}", params
            ).fetchone()
        finally:
            connection.close()
        return int(row["n"]) if row is not None else 0

    def counts_by_kind(self, *, include_archived: bool = True) -> dict[str, int]:
        """A per-category count, so a status line reports every category."""
        result = {kind.value: 0 for kind in DURABLE_KINDS}
        where = "" if include_archived else " WHERE archived=0"
        connection = self._connect()
        try:
            rows = connection.execute(
                f"SELECT kind, COUNT(*) AS n FROM memories{where} GROUP BY kind"
            ).fetchall()
        finally:
            connection.close()
        for row in rows:
            result[str(row["kind"])] = int(row["n"])
        return result

    def duplicates(self, *, kind: MemoryKind | str | None = None) -> list[tuple[str, ...]]:
        """Groups of records sharing the same content terms.

        Detection only: nothing is merged or deleted, because a near-duplicate may
        carry different provenance or a later correction. The caller decides.
        """
        connection = self._connect()
        try:
            if kind is not None:
                kind_value = kind.value if isinstance(kind, MemoryKind) else str(kind)
                rows = connection.execute(
                    """
                    SELECT t.record_id AS record_id, GROUP_CONCAT(t.term, ' ') AS signature
                    FROM memory_terms t
                    JOIN memories m ON m.record_id = t.record_id
                    WHERE m.kind=?
                    GROUP BY t.record_id
                    """,
                    (kind_value,),
                ).fetchall()
            else:
                rows = connection.execute(
                    "SELECT record_id, GROUP_CONCAT(term, ' ') AS signature "
                    "FROM memory_terms GROUP BY record_id"
                ).fetchall()
        finally:
            connection.close()
        buckets: dict[str, list[str]] = {}
        for row in rows:
            signature = " ".join(sorted(set(str(row["signature"]).split())))
            buckets.setdefault(signature, []).append(str(row["record_id"]))
        return [tuple(sorted(ids)) for ids in buckets.values() if len(ids) > 1]

    # -- integrity / backup / restore --------------------------------------
    def integrity_check(self) -> tuple[bool, str]:
        """SQLite's own integrity check, surfaced honestly."""
        connection = self._connect()
        try:
            row = connection.execute("PRAGMA integrity_check").fetchone()
        except sqlite3.DatabaseError as error:  # a corrupt file fails to open
            return False, f"the database could not be opened: {error}"
        finally:
            connection.close()
        message = str(row[0]) if row is not None else "unknown"
        return message.lower() == "ok", message

    def backup(self, destination: str | Path) -> BackupReport:
        """Write a consistent copy of the database to ``destination``.

        Uses SQLite's online backup API, so the copy is transactionally
        consistent even while the database is open, and it is validated before
        the report says ``ok``.
        """
        target = Path(destination)
        target.parent.mkdir(parents=True, exist_ok=True)
        source = self._connect()
        try:
            destination_connection = sqlite3.connect(target)
            try:
                source.backup(destination_connection)
                destination_connection.commit()
            finally:
                destination_connection.close()
        except sqlite3.DatabaseError as error:
            return BackupReport(False, str(target), 0, f"backup failed: {error}")
        finally:
            source.close()
        check = MemoryDatabase(target)
        ok, message = check.integrity_check()
        return BackupReport(ok, str(target), check.count(), message)

    def restore(self, source: str | Path) -> BackupReport:
        """Replace this database with a **validated** backup.

        The candidate is opened and integrity-checked first; only a readable,
        consistent database replaces the current one, and the current file is kept
        as ``<name>.pre-restore`` until the swap succeeds. A bad backup therefore
        cannot destroy a good database.
        """
        candidate = Path(source)
        if not candidate.exists():
            return BackupReport(False, str(candidate), 0, "the backup file does not exist.")
        try:
            check = MemoryDatabase(candidate)
        except (sqlite3.DatabaseError, MemoryDatabaseError) as error:
            return BackupReport(False, str(candidate), 0, f"the backup is unusable: {error}")
        ok, message = check.integrity_check()
        if not ok:
            return BackupReport(False, str(candidate), 0, f"the backup failed its check: {message}")
        safety = self.path.with_suffix(self.path.suffix + ".pre-restore")
        try:
            if self.path.exists():
                safety.write_bytes(self.path.read_bytes())
            self.path.write_bytes(candidate.read_bytes())
        except OSError as error:
            return BackupReport(False, str(candidate), 0, f"restore failed: {error}")
        self._initialize()
        return BackupReport(True, str(candidate), self.count(), "restored from the backup")

    def clear(self) -> None:
        """Remove every record (used by a confirmed reset, never by retention)."""
        with self._transaction() as connection:
            connection.execute("DELETE FROM memories")
            connection.execute("DELETE FROM memory_terms")

    def close(self) -> None:
        """No persistent connection is held; provided for symmetry."""
        return None


def _record_from_payload(payload: Any) -> MemoryRecord | None:
    """Parse one stored payload, returning ``None`` for a corrupt row."""
    try:
        data = json.loads(payload)
    except (TypeError, ValueError):
        return None
    if not isinstance(data, dict):
        return None
    try:
        return MemoryRecord.from_primitive(data)
    except (KeyError, TypeError, ValueError):
        return None


def _with_access(record: MemoryRecord, stamp: str) -> MemoryRecord:
    """A copy of ``record`` with one more access and a fresh access time."""
    from dataclasses import replace

    try:
        moment = datetime.fromisoformat(stamp)
    except ValueError:
        moment = datetime.now(UTC)
    return replace(record, access_count=record.access_count + 1, last_accessed=moment)


__all__ = [
    "SCHEMA_VERSION",
    "BackupReport",
    "MemoryDatabase",
    "MemoryDatabaseError",
    "MemoryLoadReport",
    "make_memory_id",
]
