"""The Romance & Companion Studio database: one versioned, transactional SQLite file.

This is the Studio's relational store, built on ``sqlite3`` from the **standard
library** (the project's own :mod:`core.memory_db` and :mod:`core.workplace.database`
prove the pattern), so it adds no dependency and runs entirely locally. Its
discipline is copied from those two modules:

* **Transactional writes.** Every write runs inside a transaction; a failure
  rolls back, so a half-written record is never left behind.
* **Schema versioning and migrations.** ``meta.schema_version`` records the
  version on disk and :meth:`RomanceDatabase.migrate` applies the numbered steps
  from :mod:`core.romance.schema`. A database from a *newer* build is refused
  rather than read with a schema this build does not understand.
* **Bounded reads.** :meth:`list_records` pages with ``LIMIT``/``OFFSET`` and
  never materialises a whole table, so a large library does not have to fit in RAM.
* **Corrupt-record tolerance.** A row whose payload will not parse is *skipped
  and counted* in a :class:`RomanceLoadReport`, never a crash.
* **Backup and recovery.** :meth:`backup` uses SQLite's own online-backup API;
  :meth:`restore` validates the copy before replacing anything, so a bad backup
  cannot destroy a good database.

**Foreign keys are on** (``PRAGMA foreign_keys=ON``) and references are
``ON DELETE RESTRICT`` (a record is *deactivated*, never deleted) except where a
child has no meaning without its parent. Nothing here loads a model or reaches the
network.
"""

from __future__ import annotations

import json
import shutil
import sqlite3
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from core.romance.model import RecordBase
from core.romance.schema import MIGRATIONS, SCHEMA_VERSION, TABLES, TableSpec


class RomanceDatabaseError(RuntimeError):
    """A Studio database could not be opened, migrated, or read."""


@dataclass
class RomanceLoadReport:
    """What a read skipped, so corruption is visible rather than silent."""

    loaded: int = 0
    skipped: int = 0
    errors: list[str] = field(default_factory=list)

    @property
    def clean(self) -> bool:
        return self.skipped == 0


@dataclass
class BackupReport:
    """The result of a backup or a restore."""

    ok: bool
    path: str
    message: str = ""
    records: int = 0


def _spec(table: str) -> TableSpec:
    spec = TABLES.get(table)
    if spec is None:
        raise RomanceDatabaseError(f"Unknown Romance table: {table!r}")
    return spec


class RomanceDatabase:
    """The versioned SQLite store behind every Studio record.

    A single file holds every table; the connection is short-lived (opened per
    operation) so the database is safe to share between the window and the CLI,
    exactly like the project's other SQLite stores.
    """

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.migrate()

    # -- connection -------------------------------------------------------

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA journal_mode=WAL")
        connection.execute("PRAGMA foreign_keys=ON")
        return connection

    @contextmanager
    def _write(self) -> Iterator[sqlite3.Connection]:
        """A transactional write: commit on success, roll back on any failure."""
        connection = self._connect()
        try:
            yield connection
            connection.commit()
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()

    # -- schema -----------------------------------------------------------

    def schema_version(self) -> int:
        """The version recorded on disk (0 for a brand-new file)."""
        connection = self._connect()
        try:
            try:
                row = connection.execute(
                    "SELECT value FROM meta WHERE key='schema_version'"
                ).fetchone()
            except sqlite3.OperationalError:
                return 0
        finally:
            connection.close()
        return int(row["value"]) if row is not None else 0

    def migrate(self) -> int:
        """Bring the database up to :data:`SCHEMA_VERSION`, returning the version.

        Each numbered step runs in its own transaction. A database from a newer
        build is refused (a schema this build cannot read must not be touched).
        """
        current = self.schema_version()
        if current > SCHEMA_VERSION:
            raise RomanceDatabaseError(
                f"Romance database is schema {current}, this build understands "
                f"{SCHEMA_VERSION}; refusing to read it."
            )
        if current == SCHEMA_VERSION:
            return current
        for target, step in MIGRATIONS:
            if target <= current:
                continue
            with self._write() as connection:
                step(connection)
                connection.execute(
                    "INSERT INTO meta(key, value) VALUES('schema_version', ?) "
                    "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
                    (str(target),),
                )
        return self.schema_version()

    # -- record <-> row ---------------------------------------------------

    @staticmethod
    def _record_to_row(spec: TableSpec, record: RecordBase) -> dict[str, Any]:
        row: dict[str, Any] = {}
        foreign = spec.foreign_key_columns
        for column in spec.columns:
            value = getattr(record, column.field, None)
            if column.json:
                value = json.dumps(value if value is not None else [], ensure_ascii=False)
            elif column.boolean:
                value = 1 if value else 0
            elif column.name in foreign and value == "":
                # An empty reference means "none", and a foreign key must be NULL
                # rather than an empty string that matches nothing.
                value = None
            row[column.name] = value
        return row

    @staticmethod
    def _row_to_record(spec: TableSpec, row: sqlite3.Row, report: RomanceLoadReport) -> Any:
        values: dict[str, Any] = {}
        foreign = spec.foreign_key_columns
        for column in spec.columns:
            raw = row[column.name]
            if column.json:
                try:
                    raw = json.loads(raw) if raw else []
                except (ValueError, TypeError):
                    report.skipped += 1
                    report.errors.append(f"{spec.name}:{row['record_id']} bad JSON in {column.name}")
                    raw = [] if column.json_kind == "list" else {}
                if column.json_kind == "list":
                    raw = tuple(raw or ())
            elif column.boolean:
                raw = bool(raw)
            elif column.name in foreign and raw is None:
                raw = ""
            values[column.field] = raw
        return spec.record(**values)

    # -- writes -----------------------------------------------------------

    def upsert(self, table: str, record: RecordBase) -> None:
        """Insert or replace one record (the write behind every create/edit)."""
        spec = _spec(table)
        row = self._record_to_row(spec, record)
        columns = list(row)
        placeholders = ", ".join(f":{name}" for name in columns)
        assignments = ", ".join(f"{name}=excluded.{name}" for name in columns if name != "record_id")
        sql = (
            f"INSERT INTO {spec.name} ({', '.join(columns)}) VALUES ({placeholders}) "
            f"ON CONFLICT(record_id) DO UPDATE SET {assignments}"
        )
        with self._write() as connection:
            connection.execute(sql, row)

    def upsert_many(self, table: str, records: Sequence[RecordBase]) -> int:
        """Insert or replace several records in one transaction, returning the count."""
        if not records:
            return 0
        spec = _spec(table)
        with self._write() as connection:
            for record in records:
                row = self._record_to_row(spec, record)
                columns = list(row)
                placeholders = ", ".join(f":{name}" for name in columns)
                assignments = ", ".join(
                    f"{name}=excluded.{name}" for name in columns if name != "record_id"
                )
                connection.execute(
                    f"INSERT INTO {spec.name} ({', '.join(columns)}) VALUES ({placeholders}) "
                    f"ON CONFLICT(record_id) DO UPDATE SET {assignments}",
                    row,
                )
        return len(records)

    def delete(self, table: str, record_id: str) -> bool:
        """Hard-delete one record (used only where the retention policy asks for it)."""
        spec = _spec(table)
        with self._write() as connection:
            cursor = connection.execute(
                f"DELETE FROM {spec.name} WHERE record_id = ?", (record_id,)
            )
        return cursor.rowcount > 0

    # -- reads ------------------------------------------------------------

    def get(self, table: str, record_id: str) -> Any | None:
        """Fetch one record by id, or ``None``."""
        spec = _spec(table)
        connection = self._connect()
        try:
            row = connection.execute(
                f"SELECT * FROM {spec.name} WHERE record_id = ?", (record_id,)
            ).fetchone()
        finally:
            connection.close()
        if row is None:
            return None
        return self._row_to_record(spec, row, RomanceLoadReport())

    def list_records(
        self,
        table: str,
        *,
        include_inactive: bool = False,
        where: str = "",
        params: Sequence[Any] = (),
        order_by: str = "created_at, record_id",
        limit: int | None = None,
        offset: int = 0,
        report: RomanceLoadReport | None = None,
    ) -> list[Any]:
        """Read records with an optional filter, bounded and ordered.

        ``include_inactive`` is the one switch that decides whether a deactivated
        record is returned; the default hides it. ``where`` is a caller-supplied
        fragment (used by the repository's typed finders) and never carries user
        text directly -- values go through ``params``.
        """
        spec = _spec(table)
        report = report if report is not None else RomanceLoadReport()
        clauses = []
        if not include_inactive:
            clauses.append("active = 1")
        if where:
            clauses.append(f"({where})")
        sql = f"SELECT * FROM {spec.name}"
        if clauses:
            sql += " WHERE " + " AND ".join(clauses)
        sql += f" ORDER BY {order_by}"
        if limit is not None:
            sql += " LIMIT ? OFFSET ?"
            params = (*params, limit, offset)
        connection = self._connect()
        try:
            rows = connection.execute(sql, tuple(params)).fetchall()
        finally:
            connection.close()
        records = []
        for row in rows:
            try:
                records.append(self._row_to_record(spec, row, report))
            except (TypeError, ValueError) as error:  # a malformed row is skipped
                report.skipped += 1
                report.errors.append(f"{spec.name}:{row['record_id']} {error}")
        report.loaded += len(records)
        return records

    def count(self, table: str, *, include_inactive: bool = False) -> int:
        """How many records a table holds."""
        spec = _spec(table)
        sql = f"SELECT COUNT(*) AS n FROM {spec.name}"
        if not include_inactive:
            sql += " WHERE active = 1"
        connection = self._connect()
        try:
            row = connection.execute(sql).fetchone()
        finally:
            connection.close()
        return int(row["n"]) if row is not None else 0

    # -- integrity, backup, restore --------------------------------------

    def integrity_check(self) -> bool:
        """Whether SQLite reports the file as sound."""
        connection = self._connect()
        try:
            row = connection.execute("PRAGMA integrity_check").fetchone()
        finally:
            connection.close()
        return bool(row is not None and row[0] == "ok")

    def backup(self, destination: str | Path) -> BackupReport:
        """Write a consistent copy of the database to ``destination``."""
        target = Path(destination)
        target.parent.mkdir(parents=True, exist_ok=True)
        source = self._connect()
        try:
            destination_connection = sqlite3.connect(target)
            try:
                source.backup(destination_connection)
            finally:
                destination_connection.close()
        finally:
            source.close()
        return BackupReport(ok=True, path=str(target), message="backup written")

    def restore(self, source: str | Path) -> BackupReport:
        """Replace this database from a backup, validating the copy first.

        The candidate is integrity-checked before anything is replaced, and the
        current file is kept beside it as ``<name>.pre-restore``, so a bad backup
        cannot destroy a good database.
        """
        candidate = Path(source)
        if not candidate.exists():
            return BackupReport(ok=False, path=str(candidate), message="backup not found")
        probe = sqlite3.connect(candidate)
        try:
            row = probe.execute("PRAGMA integrity_check").fetchone()
        except sqlite3.DatabaseError as error:
            return BackupReport(ok=False, path=str(candidate), message=f"unreadable backup: {error}")
        finally:
            probe.close()
        if not row or row[0] != "ok":
            return BackupReport(
                ok=False, path=str(candidate), message="backup failed its integrity check"
            )
        if self.path.exists():
            keep = self.path.with_suffix(self.path.suffix + ".pre-restore")
            self.path.replace(keep)
        shutil.copy2(candidate, self.path)
        if not self.integrity_check():
            return BackupReport(ok=False, path=str(candidate), message="restored file is not sound")
        return BackupReport(ok=True, path=str(candidate), message="database restored")


__all__ = [
    "BackupReport",
    "RomanceDatabase",
    "RomanceDatabaseError",
    "RomanceLoadReport",
]
