"""A persistent full-text index over a built knowledge base.

``BulkCorpusRetriever`` re-tokenizes and re-embeds every chunk on every query,
which is fine for an occasional CLI search but costs tens of seconds over an
80k-chunk store -- far too slow for an interactive answer loop. This module adds
a SQLite **FTS5** index (standard library; no new dependency) stored next to the
shards, so a query returns BM25-ranked chunk ids in milliseconds and only the
winning chunks are read from disk.

The FTS table is *contentless*: the chunk text stays in the JSONL shards, and
the index stores only term positions plus a small ``rowid -> chunk_id/shard``
map. So a 100 MB+ base is never loaded whole, in either direction.
"""

from __future__ import annotations

import re
import sqlite3
from pathlib import Path
from typing import Any

from core.knowledge_base import KnowledgeBaseStore, KnowledgeChunk

DEFAULT_INDEX_NAME = "index.sqlite"
_TOKEN_RE = re.compile(r"[^\W_]+", flags=re.UNICODE)


def _query_string(query: str) -> str:
    """Turn a free-text query into a safe FTS5 MATCH expression.

    Every token is quoted (so FTS5 operators are treated literally) and joined
    with ``OR``; BM25 then ranks chunks that match more terms higher, which is
    the intended "coverage" behaviour.
    """
    tokens = _TOKEN_RE.findall(query)
    return " OR ".join(f'"{token}"' for token in tokens)


class KnowledgeTextIndex:
    """A buildable, read-only-at-query-time FTS5 index beside the shards."""

    def __init__(self, root: str | Path, *, store: KnowledgeBaseStore | None = None) -> None:
        self.root = Path(root)
        self.path = self.root / DEFAULT_INDEX_NAME
        self.store = store or KnowledgeBaseStore(self.root)

    def exists(self) -> bool:
        return self.path.exists()

    def build(
        self,
        *,
        store: KnowledgeBaseStore | None = None,
        progress: Any | None = None,
        batch: int = 1000,
    ) -> dict[str, Any]:
        """(Re)build the index from the store's shards. Returns a summary."""
        store = store or self.store
        self.root.mkdir(parents=True, exist_ok=True)
        if self.path.exists():
            self.path.unlink()
        for suffix in ("-wal", "-shm"):
            sidecar = self.path.with_name(self.path.name + suffix)
            if sidecar.exists():
                sidecar.unlink()

        connection = sqlite3.connect(self.path)
        rowid = 0
        try:
            connection.execute("PRAGMA journal_mode=WAL")
            connection.execute(
                "CREATE TABLE chunks(rowid INTEGER PRIMARY KEY, chunk_id TEXT, title TEXT, shard INTEGER)"
            )
            connection.execute(
                "CREATE VIRTUAL TABLE fts USING fts5(title, body, content='', tokenize='unicode61')"
            )
            location_rows: list[tuple[int, str, str, int]] = []
            fts_rows: list[tuple[int, str, str]] = []
            for shard_index, shard in enumerate(store.shard_paths()):
                with shard.open(encoding="utf-8") as handle:
                    for line in handle:
                        line = line.strip()
                        if not line:
                            continue
                        chunk = KnowledgeChunk.from_json(line)
                        rowid += 1
                        location_rows.append((rowid, chunk.chunk_id, chunk.title, shard_index))
                        fts_rows.append((rowid, chunk.title, chunk.text))
                        if len(location_rows) >= batch:
                            connection.executemany("INSERT INTO chunks VALUES(?,?,?,?)", location_rows)
                            connection.executemany("INSERT INTO fts(rowid,title,body) VALUES(?,?,?)", fts_rows)
                            location_rows.clear()
                            fts_rows.clear()
                            if progress:
                                progress(f"indexed {rowid} chunks")
            if location_rows:
                connection.executemany("INSERT INTO chunks VALUES(?,?,?,?)", location_rows)
                connection.executemany("INSERT INTO fts(rowid,title,body) VALUES(?,?,?)", fts_rows)
            connection.execute("INSERT INTO fts(fts) VALUES('optimize')")
            connection.commit()
        finally:
            connection.close()
        size = self.path.stat().st_size
        return {"indexed": rowid, "size_bytes": size, "size_mb": round(size / 1024 / 1024, 2)}

    def search(
        self,
        query: str,
        *,
        limit: int = 4,
        store: KnowledgeBaseStore | None = None,
    ) -> list[tuple[float, KnowledgeChunk]]:
        """Return ``(score, chunk)`` for the best matches, best first.

        ``score`` is in ``(0, 1]`` and monotonic in BM25, so it is comparable
        across queries. Chunks are read from the shards, never from the index.
        """
        if not query.strip():
            raise ValueError("Retrieval query cannot be empty.")
        if limit < 1:
            raise ValueError("Retrieval limit must be at least 1.")
        expression = _query_string(query)
        if not expression:
            return []

        connection = sqlite3.connect(f"file:{self.path}?mode=ro", uri=True)
        try:
            rows = connection.execute(
                "SELECT rowid, bm25(fts) FROM fts WHERE fts MATCH ? ORDER BY bm25(fts) LIMIT ?",
                (expression, limit * 4),
            ).fetchall()
            if not rows:
                return []
            magnitudes = [-bm25 for _, bm25 in rows]
            top = max(magnitudes) or 1.0
            wanted = {rowid for rowid, _ in rows}
            location = dict(connection.execute(
                    f"SELECT rowid, chunk_id FROM chunks WHERE rowid IN ({','.join('?' * len(wanted))})",
                    tuple(wanted),
                ))
        finally:
            connection.close()

        chunks = (store or self.store).read_chunks(set(location.values()))
        results: list[tuple[float, KnowledgeChunk]] = []
        for (rowid, _bm25), magnitude in zip(rows, magnitudes, strict=False):
            chunk = chunks.get(location.get(rowid, ""))
            if chunk is not None:
                results.append((round(min(1.0, magnitude / top), 6), chunk))
        return results[:limit]

    def stats(self) -> dict[str, Any]:
        if not self.exists():
            return {"present": False}
        connection = sqlite3.connect(f"file:{self.path}?mode=ro", uri=True)
        try:
            count = connection.execute("SELECT count(*) FROM chunks").fetchone()[0]
        finally:
            connection.close()
        size = self.path.stat().st_size
        return {
            "present": True,
            "path": str(self.path),
            "chunks": count,
            "size_bytes": size,
            "size_mb": round(size / 1024 / 1024, 2),
        }


def build_index(store_dir: str | Path, *, progress: Any | None = None) -> dict[str, Any]:
    """Build the text index for the knowledge base at ``store_dir``."""
    index = KnowledgeTextIndex(store_dir)
    summary = index.build(progress=progress)
    summary["path"] = str(index.path)
    summary["present"] = True
    return summary


__all__ = ["DEFAULT_INDEX_NAME", "KnowledgeTextIndex", "build_index"]
