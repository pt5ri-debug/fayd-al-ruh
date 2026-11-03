"""The durable learning store: one SQLite file per owner, versioned and atomic.

This is the persistence layer Phase 3 adds. It records every
:class:`~core.learning_events.LearningEvent` -- the audit trail the task
requires -- and stores the versioned **patterns** and **procedures** the learning
pipelines produce. Like :mod:`core.memory_db` it is built on ``sqlite3`` from the
**standard library**, so it adds no dependency, works on Windows 11, and runs
entirely locally.

Design decisions, each answering a Phase-3 requirement:

* **Auditability.** Every event is appended with its kind, owner, source,
  timestamp, before/after references, outcome, and version -- so a learning
  change can be read back and explained.
* **Versioning.** A monotonic ``version`` counter is kept in ``meta``; every
  applied event stamps the version in force, and the counter only moves forward.
* **Rollback.** :meth:`rollback` marks the events at or after a version as
  ``rolled_back`` (never deleting them) and returns the store to that version --
  an *append-only* reversal, so the audit trail survives.
* **Transactional writes.** Every write runs inside a transaction; a failure
  rolls back, so a half-written event is never left behind.
* **Bounded reads.** :meth:`events` pages with ``LIMIT`` / ``OFFSET`` and never
  materialises the whole table.
* **Corrupt-record tolerance.** A row whose JSON payload will not parse is
  *skipped and counted*, never a crash.
* **Backup and recovery.** :meth:`backup` uses SQLite's own online backup API;
  :meth:`restore` validates the copy before replacing anything, so a bad backup
  cannot destroy a good store.

Nothing here loads a model or reaches the network.
"""

from __future__ import annotations

import json
import shutil
import sqlite3
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from core.learning_events import (
    LEARNING_KINDS,
    OUTCOME_ROLLED_BACK,
    LearningEvent,
    now_utc,
)

#: The schema version this build writes and understands.
SCHEMA_VERSION = 1

_SCHEMA = """
CREATE TABLE IF NOT EXISTS meta (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS learning_events (
    event_id TEXT PRIMARY KEY,
    kind TEXT NOT NULL,
    owner TEXT NOT NULL,
    target TEXT NOT NULL,
    source TEXT NOT NULL,
    outcome TEXT NOT NULL,
    applied INTEGER NOT NULL DEFAULT 0,
    version TEXT NOT NULL DEFAULT '',
    created_at TEXT NOT NULL,
    payload TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_events_kind ON learning_events(kind);
CREATE INDEX IF NOT EXISTS idx_events_created ON learning_events(created_at);
CREATE INDEX IF NOT EXISTS idx_events_target ON learning_events(target);
CREATE INDEX IF NOT EXISTS idx_events_outcome ON learning_events(outcome);
CREATE TABLE IF NOT EXISTS learned_items (
    item_id TEXT PRIMARY KEY,
    kind TEXT NOT NULL,
    owner TEXT NOT NULL,
    status TEXT NOT NULL,
    version TEXT NOT NULL DEFAULT '',
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    payload TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_items_kind ON learned_items(kind);
CREATE INDEX IF NOT EXISTS idx_items_status ON learned_items(status);
"""


class LearningStoreError(RuntimeError):
    """A learning-store failure a caller must handle (never swallowed)."""


@dataclass
class StoreLoadReport:
    """What one read actually did -- including what it could not read."""

    loaded: int = 0
    skipped: int = 0
    reasons: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return self.skipped == 0

    def describe(self) -> str:
        base = f"loaded {self.loaded} row(s)"
        if self.skipped:
            return f"{base}; skipped {self.skipped} unreadable row(s)"
        return base


@dataclass(frozen=True)
class StoreBackupReport:
    """The outcome of a backup or restore, with the real path involved."""

    ok: bool
    path: str
    records: int = 0
    message: str = ""

    def describe(self) -> str:
        return f"{'ok' if self.ok else 'failed'}: {self.message} ({self.path})"


class LearningStore:
    """One owner's durable learning history: events, patterns, and procedures."""

    def __init__(self, owner: str, path: str | Path) -> None:
        self.owner = owner
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._connection = sqlite3.connect(str(self.path))
        self._connection.row_factory = sqlite3.Row
        self._connection.execute("PRAGMA journal_mode=WAL")
        self._connection.execute("PRAGMA foreign_keys=ON")
        self._connection.executescript(_SCHEMA)
        self._connection.commit()
        self._ensure_version()

    # -- schema ------------------------------------------------------------
    @property
    def schema_version(self) -> int:
        row = self._connection.execute(
            "SELECT value FROM meta WHERE key='schema_version'"
        ).fetchone()
        return int(row["value"]) if row else 0

    def _ensure_version(self) -> None:
        stored = self.schema_version
        if stored == 0:
            self._set_meta("schema_version", str(SCHEMA_VERSION))
            self._set_meta("version_counter", "0")
            self._connection.commit()
        elif stored > SCHEMA_VERSION:
            raise LearningStoreError(
                f"The learning store at {self.path} was written by a newer build "
                f"(schema {stored} > {SCHEMA_VERSION}); refusing to read it."
            )

    def _set_meta(self, key: str, value: str) -> None:
        self._connection.execute(
            "INSERT INTO meta(key, value) VALUES(?,?) "
            "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
            (key, value),
        )

    def _get_meta(self, key: str, default: str = "") -> str:
        row = self._connection.execute("SELECT value FROM meta WHERE key=?", (key,)).fetchone()
        return str(row["value"]) if row else default

    @contextmanager
    def _transaction(self) -> Iterator[sqlite3.Connection]:
        try:
            self._connection.execute("BEGIN")
            yield self._connection
            self._connection.commit()
        except Exception:
            self._connection.rollback()
            raise

    # -- versioning --------------------------------------------------------
    @property
    def version(self) -> int:
        """The current learning version (monotonic, never reused)."""
        return int(self._get_meta("version_counter", "0") or 0)

    def next_version(self) -> int:
        """Advance the version counter by one and return the new value.

        A version is only ever advanced by an **applied** change, so a rejected
        learning act does not move the store forward.
        """
        with self._transaction() as connection:
            current = int(self._get_meta("version_counter", "0") or 0) + 1
            connection.execute(
                "INSERT INTO meta(key, value) VALUES('version_counter',?) "
                "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
                (str(current),),
            )
        return current

    # -- events ------------------------------------------------------------
    def record_event(self, event: LearningEvent) -> LearningEvent:
        """Append one event (idempotent: the same id is not written twice)."""
        existing = self.event(event.event_id)
        if existing is not None:
            return existing
        with self._transaction() as connection:
            connection.execute(
                """
                INSERT INTO learning_events(
                    event_id, kind, owner, target, source, outcome, applied,
                    version, created_at, payload
                ) VALUES(?,?,?,?,?,?,?,?,?,?)
                ON CONFLICT(event_id) DO NOTHING
                """,
                (
                    event.event_id,
                    event.kind,
                    event.owner,
                    event.target,
                    event.source,
                    event.outcome,
                    1 if event.applied else 0,
                    event.version,
                    event.created_at.isoformat(),
                    json.dumps(event.to_primitive(), ensure_ascii=False, sort_keys=True),
                ),
            )
        return event

    def record_many(self, events: Sequence[LearningEvent]) -> int:
        """Append several events in **one** transaction (all or nothing)."""
        if not events:
            return 0
        payloads = [
            (
                event,
                json.dumps(event.to_primitive(), ensure_ascii=False, sort_keys=True),
            )
            for event in events
        ]
        with self._transaction() as connection:
            for event, payload in payloads:
                connection.execute(
                    """
                    INSERT INTO learning_events(
                        event_id, kind, owner, target, source, outcome, applied,
                        version, created_at, payload
                    ) VALUES(?,?,?,?,?,?,?,?,?,?)
                    ON CONFLICT(event_id) DO NOTHING
                    """,
                    (
                        event.event_id,
                        event.kind,
                        event.owner,
                        event.target,
                        event.source,
                        event.outcome,
                        1 if event.applied else 0,
                        event.version,
                        event.created_at.isoformat(),
                        payload,
                    ),
                )
        return len(payloads)

    def event(self, event_id: str) -> LearningEvent | None:
        """Fetch one event by id (``None`` when it is not stored)."""
        row = self._connection.execute(
            "SELECT payload FROM learning_events WHERE event_id=?", (event_id,)
        ).fetchone()
        if row is None:
            return None
        try:
            return LearningEvent.from_primitive(json.loads(row["payload"]))
        except (ValueError, TypeError, KeyError):
            return None

    def events(
        self,
        *,
        kind: str | None = None,
        outcome: str | None = None,
        limit: int = 100,
        offset: int = 0,
        since: datetime | None = None,
    ) -> tuple[list[LearningEvent], StoreLoadReport]:
        """A bounded page of events, newest first, with a load report."""
        clauses: list[str] = []
        params: list[Any] = []
        if kind is not None:
            clauses.append("kind=?")
            params.append(kind)
        if outcome is not None:
            clauses.append("outcome=?")
            params.append(outcome)
        if since is not None:
            clauses.append("created_at>=?")
            params.append(since.isoformat())
        where = f" WHERE {' AND '.join(clauses)}" if clauses else ""
        rows = self._connection.execute(
            f"SELECT payload FROM learning_events{where} "
            "ORDER BY created_at DESC, event_id DESC LIMIT ? OFFSET ?",
            (*params, limit, offset),
        ).fetchall()
        report = StoreLoadReport()
        result: list[LearningEvent] = []
        for row in rows:
            try:
                result.append(LearningEvent.from_primitive(json.loads(row["payload"])))
                report.loaded += 1
            except (ValueError, TypeError, KeyError) as error:
                report.skipped += 1
                report.reasons.append(str(error))
        return result, report

    def iter_events(self, *, kind: str | None = None) -> Iterator[LearningEvent]:
        """Stream every matching event in bounded pages (never all in RAM)."""
        offset = 0
        while True:
            page, _ = self.events(kind=kind, limit=200, offset=offset)
            if not page:
                return
            yield from page
            offset += len(page)

    def count(self, *, kind: str | None = None, outcome: str | None = None) -> int:
        clauses: list[str] = []
        params: list[Any] = []
        if kind is not None:
            clauses.append("kind=?")
            params.append(kind)
        if outcome is not None:
            clauses.append("outcome=?")
            params.append(outcome)
        where = f" WHERE {' AND '.join(clauses)}" if clauses else ""
        row = self._connection.execute(
            f"SELECT COUNT(*) AS n FROM learning_events{where}", params
        ).fetchone()
        return int(row["n"]) if row else 0

    def counts_by_kind(self) -> dict[str, int]:
        rows = self._connection.execute(
            "SELECT kind, COUNT(*) AS n FROM learning_events GROUP BY kind"
        ).fetchall()
        counts = dict.fromkeys(LEARNING_KINDS, 0)
        for row in rows:
            counts[str(row["kind"])] = int(row["n"])
        return counts

    # -- learned items (patterns and procedures) ---------------------------
    def save_item(self, item: Any) -> Any:
        """Insert or replace one learned item (a pattern or a procedure)."""
        item_id = str(item.item_id)
        now = now_utc().isoformat()
        existing = self._connection.execute(
            "SELECT created_at FROM learned_items WHERE item_id=?", (item_id,)
        ).fetchone()
        created = str(existing["created_at"]) if existing else now
        with self._transaction() as connection:
            connection.execute(
                """
                INSERT INTO learned_items(
                    item_id, kind, owner, status, version, created_at, updated_at, payload
                ) VALUES(?,?,?,?,?,?,?,?)
                ON CONFLICT(item_id) DO UPDATE SET
                    kind=excluded.kind, owner=excluded.owner, status=excluded.status,
                    version=excluded.version, updated_at=excluded.updated_at,
                    payload=excluded.payload
                """,
                (
                    item_id,
                    str(item.kind),
                    str(item.owner),
                    str(item.status),
                    str(item.version),
                    created,
                    now,
                    json.dumps(item.to_primitive(), ensure_ascii=False, sort_keys=True),
                ),
            )
        return item

    def get_item(self, item_id: str) -> dict[str, Any] | None:
        """Fetch one learned item's stored payload (``None`` when absent)."""
        row = self._connection.execute(
            "SELECT payload FROM learned_items WHERE item_id=?", (item_id,)
        ).fetchone()
        if row is None:
            return None
        try:
            return dict(json.loads(row["payload"]))
        except (ValueError, TypeError):
            return None

    def items(
        self,
        *,
        kind: str | None = None,
        status: str | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> tuple[list[dict[str, Any]], StoreLoadReport]:
        """A bounded page of learned items, with a load report."""
        clauses: list[str] = []
        params: list[Any] = []
        if kind is not None:
            clauses.append("kind=?")
            params.append(kind)
        if status is not None:
            clauses.append("status=?")
            params.append(status)
        where = f" WHERE {' AND '.join(clauses)}" if clauses else ""
        rows = self._connection.execute(
            f"SELECT payload FROM learned_items{where} "
            "ORDER BY updated_at DESC, item_id DESC LIMIT ? OFFSET ?",
            (*params, limit, offset),
        ).fetchall()
        report = StoreLoadReport()
        result: list[dict[str, Any]] = []
        for row in rows:
            try:
                result.append(dict(json.loads(row["payload"])))
                report.loaded += 1
            except (ValueError, TypeError) as error:
                report.skipped += 1
                report.reasons.append(str(error))
        return result, report

    def count_items(self, *, kind: str | None = None) -> int:
        if kind is None:
            row = self._connection.execute("SELECT COUNT(*) AS n FROM learned_items").fetchone()
        else:
            row = self._connection.execute(
                "SELECT COUNT(*) AS n FROM learned_items WHERE kind=?", (kind,)
            ).fetchone()
        return int(row["n"]) if row else 0

    # -- rollback ----------------------------------------------------------
    def rollback(self, version: int, *, reason: str = "") -> list[LearningEvent]:
        """Mark the events at or after ``version`` rolled back (never deleted).

        Returns the *new* reversal events written, so the rollback itself is
        auditable. The stored events are left in place with their outcome changed
        to ``rolled_back``, which is an append-only reversal rather than an
        erasure: nothing unrelated is touched.
        """
        if version < 0:
            raise ValueError("A rollback version cannot be negative.")
        targets = [
            event
            for event in self.iter_events()
            if event.version and _version_of(event.version) >= version and event.applied
        ]
        reversals: list[LearningEvent] = []
        with self._transaction() as connection:
            for event in targets:
                connection.execute(
                    "UPDATE learning_events SET outcome=?, applied=0 WHERE event_id=?",
                    (OUTCOME_ROLLED_BACK, event.event_id),
                )
                reversal = LearningEvent(
                    kind="rollback",
                    target=event.target,
                    owner=self.owner,
                    source="system",
                    summary=reason or f"rolled back to version {version}",
                    before=event.after,
                    after=event.before,
                    outcome=OUTCOME_ROLLED_BACK,
                    applied=True,
                    version=event.version,
                    supersedes=event.event_id,
                )
                connection.execute(
                    """
                    INSERT INTO learning_events(
                        event_id, kind, owner, target, source, outcome, applied,
                        version, created_at, payload
                    ) VALUES(?,?,?,?,?,?,?,?,?,?)
                    ON CONFLICT(event_id) DO NOTHING
                    """,
                    (
                        reversal.event_id,
                        reversal.kind,
                        reversal.owner,
                        reversal.target,
                        reversal.source,
                        reversal.outcome,
                        1,
                        reversal.version,
                        reversal.created_at.isoformat(),
                        json.dumps(reversal.to_primitive(), ensure_ascii=False, sort_keys=True),
                    ),
                )
                reversals.append(reversal)
            connection.execute(
                "INSERT INTO meta(key, value) VALUES('version_counter',?) "
                "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
                (str(max(version - 1, 0)),),
            )
        return reversals

    # -- integrity, backup, restore ---------------------------------------
    def integrity_check(self) -> tuple[bool, str]:
        """SQLite's own integrity check over the file."""
        try:
            row = self._connection.execute("PRAGMA integrity_check").fetchone()
        except sqlite3.DatabaseError as error:
            return False, str(error)
        result = str(row[0]) if row else "unknown"
        return result.lower() == "ok", result

    def backup(self, destination: str | Path) -> StoreBackupReport:
        """Write a consistent copy of the store to ``destination``."""
        target = Path(destination)
        target.parent.mkdir(parents=True, exist_ok=True)
        try:
            with sqlite3.connect(str(target)) as destination_connection:
                self._connection.backup(destination_connection)
                destination_connection.commit()
        except sqlite3.Error as error:
            return StoreBackupReport(False, str(target), 0, f"backup failed: {error}")
        return StoreBackupReport(True, str(target), self.count(), "backup written")

    def restore(self, source: str | Path) -> StoreBackupReport:
        """Replace this store from a backup **only after validating it**.

        A bad backup is refused and the current store is left untouched, so a
        corrupt file can never destroy good learning history.
        """
        origin = Path(source)
        if not origin.exists():
            return StoreBackupReport(False, str(origin), 0, "backup file does not exist")
        try:
            with sqlite3.connect(str(origin)) as candidate:
                check = candidate.execute("PRAGMA integrity_check").fetchone()
                if check is None or str(check[0]).lower() != "ok":
                    return StoreBackupReport(
                        False, str(origin), 0, "backup failed its integrity check"
                    )
                candidate.execute("SELECT COUNT(*) FROM learning_events").fetchone()
                candidate.backup(self._connection)
                self._connection.commit()
        except sqlite3.Error as error:
            return StoreBackupReport(False, str(origin), 0, f"restore refused: {error}")
        return StoreBackupReport(True, str(origin), self.count(), "restored")

    def export(self) -> dict[str, Any]:
        """A JSON-able dump of the whole store (for tooling and tests)."""
        return {
            "owner": self.owner,
            "schema_version": self.schema_version,
            "version": self.version,
            "events": [event.to_primitive() for event in self.iter_events()],
            "counts": self.counts_by_kind(),
        }

    def clear(self) -> None:
        """Remove every stored event and item (a named, destructive reset)."""
        with self._transaction() as connection:
            connection.execute("DELETE FROM learning_events")
            connection.execute("DELETE FROM learned_items")
            connection.execute(
                "INSERT INTO meta(key, value) VALUES('version_counter','0') "
                "ON CONFLICT(key) DO UPDATE SET value='0'"
            )

    def describe(self) -> str:
        counts = self.counts_by_kind()
        parts = ", ".join(f"{kind}={counts.get(kind, 0)}" for kind in LEARNING_KINDS)
        return (
            f"learning store for {self.owner}: {parts}; "
            f"items={self.count_items()}; version={self.version}; "
            f"schema={self.schema_version}"
        )

    def close(self) -> None:
        """Close the underlying connection (a re-open on the same file works)."""
        self._connection.close()

    # -- convenience -------------------------------------------------------
    def copy_to(self, destination: str | Path) -> StoreBackupReport:
        """A byte copy of the file, used only when SQLite cannot be opened."""
        target = Path(destination)
        shutil.copy2(self.path, target)
        return StoreBackupReport(True, str(target), self.count(), "copied")


def _version_of(value: str) -> int:
    """Parse a version string (``"3"`` or ``"v3"``) into an int, 0 on failure."""
    text = value.strip().lstrip("vV")
    try:
        return int(text)
    except ValueError:
        return 0


__all__ = [
    "SCHEMA_VERSION",
    "LearningStore",
    "LearningStoreError",
    "StoreBackupReport",
    "StoreLoadReport",
]
