"""A small SQLite ledger that makes ingestion incremental and duplicate-free.

Two questions the pipeline asks before it does any work:

* **Have I already ingested this source item?** -- answered by the *source key*
  plus its fingerprint. If the fingerprint is unchanged, the file, row, or URL
  is skipped without being read again.
* **Is this chunk already stored?** -- answered by the *content hash*. The same
  paragraph appearing in two documents is stored once, so a re-import or an
  overlapping export does not duplicate knowledge.

The ledger is a tiny local SQLite database (the standard library, no new
dependency; :class:`storage.sqlite_store.SqliteStore` is the existing precedent
for a namespaced SQLite store). It holds no document text -- only keys, hashes,
and fingerprints -- so it stays small even over a 100 GB corpus, and it can be
rebuilt from the store when lost.

Recording is split into *plan* and *commit*: the pipeline asks
:meth:`DedupLedger.contains_chunk` while streaming, and calls
:meth:`DedupLedger.record` only once the chunk is actually written, so a crash
mid-batch cannot mark unwritten work as done.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from pathlib import Path

SCHEMA = """
CREATE TABLE IF NOT EXISTS sources (
    source_key TEXT PRIMARY KEY,
    uri        TEXT NOT NULL,
    connector  TEXT NOT NULL,
    fingerprint TEXT NOT NULL DEFAULT '',
    chunks     INTEGER NOT NULL DEFAULT 0,
    bytes      INTEGER NOT NULL DEFAULT 0
);
CREATE TABLE IF NOT EXISTS chunks (
    content_hash TEXT PRIMARY KEY,
    source_key   TEXT NOT NULL,
    doc_id       TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS chunks_by_source ON chunks(source_key);
"""


@dataclass(frozen=True)
class SourceState:
    """What the ledger remembers about one source item."""

    source_key: str
    fingerprint: str
    chunks: int


class DedupLedger:
    """SQLite-backed record of ingested sources and chunk hashes."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        connection = self._connect()
        try:
            connection.executescript(SCHEMA)
            connection.commit()
        finally:
            connection.close()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path)
        connection.execute("PRAGMA journal_mode=WAL")
        return connection

    # -- source items -----------------------------------------------------

    def known_source(self, key: str) -> SourceState | None:
        """The stored state for a source item, or ``None`` when unseen."""
        connection = self._connect()
        try:
            row = connection.execute(
                "SELECT source_key, fingerprint, chunks FROM sources WHERE source_key=?",
                (key,),
            ).fetchone()
        finally:
            connection.close()
        if row is None:
            return None
        return SourceState(source_key=row[0], fingerprint=row[1], chunks=int(row[2]))

    def unchanged(self, key: str, fingerprint: str) -> bool:
        """Whether a source item was already ingested with this fingerprint."""
        state = self.known_source(key)
        return state is not None and bool(fingerprint) and state.fingerprint == fingerprint

    def record_source(
        self,
        key: str,
        *,
        uri: str,
        connector: str,
        fingerprint: str = "",
        chunks: int = 0,
        bytes_written: int = 0,
    ) -> None:
        connection = self._connect()
        try:
            connection.execute(
                "INSERT INTO sources(source_key, uri, connector, fingerprint, chunks, bytes) "
                "VALUES(?,?,?,?,?,?) ON CONFLICT(source_key) DO UPDATE SET "
                "uri=excluded.uri, connector=excluded.connector, "
                "fingerprint=excluded.fingerprint, chunks=excluded.chunks, bytes=excluded.bytes",
                (key, uri, connector, fingerprint, chunks, bytes_written),
            )
            connection.commit()
        finally:
            connection.close()

    # -- chunks -----------------------------------------------------------

    def contains_chunk(self, digest: str) -> bool:
        connection = self._connect()
        try:
            row = connection.execute(
                "SELECT 1 FROM chunks WHERE content_hash=?", (digest,)
            ).fetchone()
        finally:
            connection.close()
        return row is not None

    def record_chunk(self, digest: str, *, source_key: str, doc_id: str) -> None:
        connection = self._connect()
        try:
            connection.execute(
                "INSERT OR IGNORE INTO chunks(content_hash, source_key, doc_id) VALUES(?,?,?)",
                (digest, source_key, doc_id),
            )
            connection.commit()
        finally:
            connection.close()

    def source_chunks(self, source_key: str) -> list[str]:
        """The doc ids recorded for one source item (used before safe deletion)."""
        connection = self._connect()
        try:
            rows = connection.execute(
                "SELECT doc_id FROM chunks WHERE source_key=? ORDER BY doc_id",
                (source_key,),
            ).fetchall()
        finally:
            connection.close()
        return [row[0] for row in rows]

    def drop_source(self, source_key: str) -> int:
        """Forget one source item and its chunk hashes; return chunks dropped."""
        connection = self._connect()
        try:
            removed = connection.execute(
                "SELECT count(*) FROM chunks WHERE source_key=?", (source_key,)
            ).fetchone()[0]
            connection.execute("DELETE FROM chunks WHERE source_key=?", (source_key,))
            connection.execute("DELETE FROM sources WHERE source_key=?", (source_key,))
            connection.commit()
        finally:
            connection.close()
        return int(removed)

    def stats(self) -> dict[str, int]:
        connection = self._connect()
        try:
            sources = connection.execute("SELECT count(*) FROM sources").fetchone()[0]
            chunks = connection.execute("SELECT count(*) FROM chunks").fetchone()[0]
            total_bytes = connection.execute(
                "SELECT COALESCE(sum(bytes), 0) FROM sources"
            ).fetchone()[0]
        finally:
            connection.close()
        return {"sources": int(sources), "chunks": int(chunks), "bytes": int(total_bytes)}

    def reset(self) -> None:
        """Forget everything (a full re-ingest); the store itself is untouched."""
        connection = self._connect()
        try:
            connection.execute("DELETE FROM chunks")
            connection.execute("DELETE FROM sources")
            connection.commit()
        finally:
            connection.close()


__all__ = ["SCHEMA", "DedupLedger", "SourceState"]
