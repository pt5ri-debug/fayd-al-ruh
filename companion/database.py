"""Natasha's own SQLite database: schema, transactions, backups, migrations.

This is a *separate* store from the project's cognitive core. It holds the
companion's conversations, memories, dialogue emotion, and relationship events --
nothing about entities, relations, evidence, or beliefs, so Natasha can never
become the knowledge graph or replace it.

Design rules kept deliberately:

* **Standard library only.** ``sqlite3`` ships with Python, so a base install
  pulls nothing new and no vector database or external service is imposed.
* **Atomic writes.** Every write runs inside a transaction, so a crash cannot
  leave a half-saved message or a memory without its correction.
* **Idempotent schema.** ``CREATE TABLE IF NOT EXISTS`` plus an explicit
  ``PRAGMA user_version`` let the schema be created repeatedly and migrated
  forward later without data loss.
* **No destructive startup.** The database is never cleared on an error or on
  start; a failure is reported, not "fixed" by wiping the user's data.
* **Bounded backups.** :meth:`CompanionDatabase.backup` writes a timestamped
  copy and prunes older ones, so backups cannot grow without limit.
"""

from __future__ import annotations

import sqlite3
import threading
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

#: The current schema version, stored in ``PRAGMA user_version``. Bump it when a
#: migration is added; the migration table below maps version -> statement list.
SCHEMA_VERSION = 1


def _utc_now() -> str:
    """An ISO-8601 UTC timestamp, so ordering is stable and timezone-free."""
    return datetime.now(timezone.utc).isoformat()


_SCHEMA: tuple[str, ...] = (
    """
    CREATE TABLE IF NOT EXISTS conversations (
        id          TEXT PRIMARY KEY,
        title       TEXT NOT NULL,
        created_at  TEXT NOT NULL,
        last_active TEXT NOT NULL
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS messages (
        id              TEXT PRIMARY KEY,
        conversation_id TEXT NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
        seq             INTEGER NOT NULL,
        role            TEXT NOT NULL,
        content         TEXT NOT NULL,
        created_at      TEXT NOT NULL,
        UNIQUE (conversation_id, seq)
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_messages_conversation ON messages(conversation_id, seq)",
    """
    CREATE TABLE IF NOT EXISTS memories (
        id            TEXT PRIMARY KEY,
        kind          TEXT NOT NULL,
        content       TEXT NOT NULL,
        source        TEXT NOT NULL,
        confidence    REAL NOT NULL,
        created_at    TEXT NOT NULL,
        updated_at    TEXT NOT NULL,
        superseded_by TEXT,
        active        INTEGER NOT NULL DEFAULT 1,
        fictional     INTEGER NOT NULL DEFAULT 0
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_memories_kind ON memories(kind)",
    """
    CREATE TABLE IF NOT EXISTS conversation_summaries (
        id              TEXT PRIMARY KEY,
        conversation_id TEXT NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
        summary         TEXT NOT NULL,
        from_seq        INTEGER NOT NULL,
        to_seq          INTEGER NOT NULL,
        created_at      TEXT NOT NULL
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_summaries_conversation ON conversation_summaries(conversation_id)",
    """
    CREATE TABLE IF NOT EXISTS companion_settings (
        key        TEXT PRIMARY KEY,
        value      TEXT NOT NULL,
        updated_at TEXT NOT NULL
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS emotion_states (
        id              INTEGER PRIMARY KEY AUTOINCREMENT,
        conversation_id TEXT NOT NULL,
        state           TEXT NOT NULL,
        created_at      TEXT NOT NULL
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_emotion_conversation ON emotion_states(conversation_id, id)",
    """
    CREATE TABLE IF NOT EXISTS relationship_events (
        id          TEXT PRIMARY KEY,
        kind        TEXT NOT NULL,
        title       TEXT NOT NULL,
        detail      TEXT NOT NULL,
        occurred_at TEXT NOT NULL,
        created_at  TEXT NOT NULL,
        active      INTEGER NOT NULL DEFAULT 1,
        fictional   INTEGER NOT NULL DEFAULT 0
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_relationship_kind ON relationship_events(kind)",
)

#: Migrations keyed by the version they upgrade *from*. A fresh database is
#: created at :data:`SCHEMA_VERSION` directly, so this stays empty until a change
#: to the schema is genuinely needed -- at which point a new entry is added here
#: and ``SCHEMA_VERSION`` is bumped, and an existing database is upgraded in place.
_MIGRATIONS: dict[int, tuple[str, ...]] = {}


class CompanionDatabaseError(RuntimeError):
    """Raised when the companion database cannot be opened or written."""


class CompanionDatabase:
    """A small, transaction-safe SQLite store for Natasha.

    One connection is held for the object's life (required for an in-memory
    database, and cheaper for a file one), guarded by a re-entrant lock so the
    GUI's worker thread and the CLI can share a single companion safely.
    """

    def __init__(self, path: str | Path | None, *, persistent: bool = True) -> None:
        self.persistent = persistent and path is not None
        self._lock = threading.RLock()
        if self.persistent:
            self.path = Path(path)  # type: ignore[arg-type]
            self.path.parent.mkdir(parents=True, exist_ok=True)
            target = str(self.path)
        else:
            # A private in-memory database; nothing is written to disk.
            self.path = None  # type: ignore[assignment]
            target = ":memory:"
        try:
            self._connection = sqlite3.connect(target, check_same_thread=False)
        except sqlite3.Error as error:  # pragma: no cover - filesystem dependent
            raise CompanionDatabaseError(f"Could not open the companion database: {error}") from error
        self._connection.row_factory = sqlite3.Row
        with self._lock:
            self._connection.execute("PRAGMA foreign_keys = ON")
            self._initialise()

    # -- schema -----------------------------------------------------------
    def _initialise(self) -> None:
        """Create the schema and run any pending migrations, idempotently."""
        version = int(self._connection.execute("PRAGMA user_version").fetchone()[0])
        with self._connection:
            if version == 0:
                for statement in _SCHEMA:
                    self._connection.execute(statement)
                self._connection.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")
                return
            while version < SCHEMA_VERSION:
                statements = _MIGRATIONS.get(version)
                if statements is None:
                    # A database from a *newer* build, or a version we cannot
                    # migrate: refuse rather than corrupt it.
                    raise CompanionDatabaseError(
                        f"Unsupported companion database version {version} "
                        f"(this build understands up to {SCHEMA_VERSION})."
                    )
                for statement in statements:
                    self._connection.execute(statement)
                version += 1
                self._connection.execute(f"PRAGMA user_version = {version}")

    @property
    def version(self) -> int:
        """The schema version actually stored in the file."""
        with self._lock:
            return int(self._connection.execute("PRAGMA user_version").fetchone()[0])

    # -- transactions -----------------------------------------------------
    @contextmanager
    def transaction(self):
        """A context manager that commits on success and rolls back on error.

        Yields the live connection so several statements run in one atomic unit.
        The commit happens only when the block finishes cleanly, so a caller's
        statements are never committed early (which a bare ``with connection:``
        would do) and never half-applied on failure.
        """
        with self._lock:
            try:
                yield self._connection
            except BaseException:
                self._connection.rollback()
                raise
            self._connection.commit()

    def execute(self, sql: str, params: Iterable[object] = ()) -> sqlite3.Cursor:
        """Run one statement and commit it (a single atomic write)."""
        with self._lock, self._connection:
            return self._connection.execute(sql, tuple(params))

    def query(self, sql: str, params: Iterable[object] = ()) -> list[sqlite3.Row]:
        """Run a read-only query and return all rows."""
        with self._lock:
            return list(self._connection.execute(sql, tuple(params)))

    def query_one(self, sql: str, params: Iterable[object] = ()) -> sqlite3.Row | None:
        """Run a read-only query and return the first row, or ``None``."""
        with self._lock:
            return self._connection.execute(sql, tuple(params)).fetchone()

    # -- maintenance ------------------------------------------------------
    def backup(self, keep: int = 5, *, directory: Path | None = None) -> Path | None:
        """Write a timestamped backup and prune old ones.

        Returns the backup path, or ``None`` for an in-memory database (which has
        nothing on disk to copy). The copy uses SQLite's own online-backup API, so
        it is consistent even while the database is open, and never grows without
        bound because only the newest ``keep`` copies are retained.
        """
        if not self.persistent or self.path is None:
            return None
        target_dir = directory or self.path.parent / "companion-backups"
        target_dir.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        destination = target_dir / f"{self.path.stem}-{stamp}{self.path.suffix}"
        # Two backups in the same second must not overwrite each other, so a
        # free name is chosen rather than clobbering an earlier copy.
        counter = 1
        while destination.exists():
            destination = target_dir / f"{self.path.stem}-{stamp}-{counter}{self.path.suffix}"
            counter += 1
        with self._lock:
            try:
                with sqlite3.connect(str(destination)) as backup_connection:
                    self._connection.backup(backup_connection)
            except sqlite3.Error as error:  # pragma: no cover - filesystem dependent
                raise CompanionDatabaseError(f"Backup failed: {error}") from error
        self._prune_backups(target_dir, keep)
        return destination

    def _prune_backups(self, directory: Path, keep: int) -> None:
        if keep < 0:
            return
        pattern = f"{self.path.stem}-*{self.path.suffix}" if self.path else "*"
        backups = sorted(directory.glob(pattern))
        for stale in backups[: max(0, len(backups) - keep)]:
            try:
                stale.unlink()
            except OSError:  # pragma: no cover - best effort
                pass

    def list_backups(self, directory: Path | None = None) -> list[Path]:
        """Every backup file, newest last."""
        if not self.persistent or self.path is None:
            return []
        target_dir = directory or self.path.parent / "companion-backups"
        if not target_dir.exists():
            return []
        return sorted(target_dir.glob(f"{self.path.stem}-*{self.path.suffix}"))

    def restore(self, source: Path) -> None:
        """Replace the live database with a backup, atomically.

        The current file is copied aside first, so a bad backup can be undone.
        """
        if not self.persistent or self.path is None:
            raise CompanionDatabaseError("An in-memory database cannot be restored.")
        source = Path(source)
        if not source.is_file():
            raise CompanionDatabaseError(f"Backup not found: {source}")
        with self._lock:
            self._connection.close()
            safety = self.path.with_suffix(self.path.suffix + ".pre-restore")
            try:
                if self.path.exists():
                    safety.write_bytes(self.path.read_bytes())
                self.path.write_bytes(source.read_bytes())
            finally:
                self._connection = sqlite3.connect(str(self.path), check_same_thread=False)
                self._connection.row_factory = sqlite3.Row
                self._connection.execute("PRAGMA foreign_keys = ON")
                self._initialise()

    def close(self) -> None:
        """Close the connection (a no-op if already closed)."""
        with self._lock:
            try:
                self._connection.close()
            except sqlite3.Error:  # pragma: no cover - already closed
                pass


__all__ = [
    "CompanionDatabase",
    "CompanionDatabaseError",
    "SCHEMA_VERSION",
]
