"""Per-document metadata for ingested sources: hashes, sizes, and provenance.

The knowledge store already keeps each citable *chunk* with a few fields
(``title``, ``source``, ``url``, ``license``, ``category``). What it does not
keep is the record-level bookkeeping an ingest system needs: which registered
source a document came from, the document's content hash (so an unchanged
document is recognisable), its size, media type, and the ids of the chunks it
produced.

This module adds exactly that, as *metadata only*. It never stores document text
-- the store is the single source of truth for content -- so it stays small and
raises no duplication. It reuses the project's existing
:class:`storage.sqlite_store.SqliteStore` for the value payloads and adds one
indexed column (``source_id``) so listing a source's documents is a query, not a
full scan.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from storage.sqlite_store import SqliteStore

_NAMESPACE = "documents"

SCHEMA = """
CREATE TABLE IF NOT EXISTS documents (
    doc_id    TEXT PRIMARY KEY,
    source_id TEXT NOT NULL DEFAULT '',
    uri       TEXT NOT NULL DEFAULT '',
    text_hash TEXT NOT NULL DEFAULT ''
);
CREATE INDEX IF NOT EXISTS documents_by_source ON documents(source_id);
"""


@dataclass(frozen=True)
class DocumentMetadata:
    """Text-free metadata about one ingested document (a source *record*)."""

    doc_id: str
    source_id: str
    uri: str = ""
    title: str = ""
    text_hash: str = ""
    char_count: int = 0
    chunk_ids: tuple[str, ...] = ()
    media_type: str = ""
    language: str = ""
    provider: str = ""
    license: str = ""
    category: str = ""
    tags: tuple[str, ...] = ()
    created_at: str = ""

    def __post_init__(self) -> None:
        if not self.doc_id.strip():
            raise ValueError("DocumentMetadata doc_id cannot be empty.")
        if self.char_count < 0:
            raise ValueError("char_count cannot be negative.")

    def to_primitive(self) -> dict[str, Any]:
        data = asdict(self)
        data["chunk_ids"] = list(self.chunk_ids)
        data["tags"] = list(self.tags)
        return data

    @classmethod
    def from_primitive(cls, data: Mapping[str, Any]) -> DocumentMetadata:
        return cls(
            doc_id=str(data["doc_id"]),
            source_id=str(data.get("source_id", "")),
            uri=str(data.get("uri", "")),
            title=str(data.get("title", "")),
            text_hash=str(data.get("text_hash", "")),
            char_count=int(data.get("char_count", 0)),
            chunk_ids=tuple(str(cid) for cid in data.get("chunk_ids", ())),
            media_type=str(data.get("media_type", "")),
            language=str(data.get("language", "")),
            provider=str(data.get("provider", "")),
            license=str(data.get("license", "")),
            category=str(data.get("category", "")),
            tags=tuple(str(tag) for tag in data.get("tags", ())),
            created_at=str(data.get("created_at", "")),
        )


class DocumentMetadataStore:
    """A SQLite-backed, per-source document-metadata index."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self._store = SqliteStore(self.path)
        self._init_schema()

    def _init_schema(self) -> None:
        connection = self._store._connect()
        try:
            connection.executescript(SCHEMA)
            connection.commit()
        finally:
            connection.close()

    def record(self, metadata: DocumentMetadata) -> DocumentMetadata:
        """Add or replace one document's metadata (payload + indexed columns)."""
        connection = self._store._connect()
        try:
            connection.execute(
                "INSERT INTO documents(doc_id, source_id, uri, text_hash) VALUES(?,?,?,?) "
                "ON CONFLICT(doc_id) DO UPDATE SET source_id=excluded.source_id, "
                "uri=excluded.uri, text_hash=excluded.text_hash",
                (metadata.doc_id, metadata.source_id, metadata.uri, metadata.text_hash),
            )
            connection.commit()
        finally:
            connection.close()
        self._store.save(_NAMESPACE, metadata.doc_id, metadata.to_primitive())
        return metadata

    def get(self, doc_id: str) -> DocumentMetadata | None:
        data = self._store.load(_NAMESPACE, doc_id)
        if not isinstance(data, dict):
            return None
        return DocumentMetadata.from_primitive(data)

    def for_source(self, source_id: str) -> list[DocumentMetadata]:
        """Every document recorded for one registered source, id-ordered."""
        connection = self._store._connect()
        try:
            rows = connection.execute(
                "SELECT doc_id FROM documents WHERE source_id=? ORDER BY doc_id",
                (source_id,),
            ).fetchall()
        finally:
            connection.close()
        items = [self.get(row[0]) for row in rows]
        return [item for item in items if item is not None]

    def remove(self, doc_id: str) -> bool:
        connection = self._store._connect()
        try:
            row = connection.execute(
                "SELECT 1 FROM documents WHERE doc_id=?", (doc_id,)
            ).fetchone()
            if row is None:
                return False
            connection.execute("DELETE FROM documents WHERE doc_id=?", (doc_id,))
            connection.execute(
                "DELETE FROM records WHERE namespace=? AND key=?", (_NAMESPACE, doc_id)
            )
            connection.commit()
        finally:
            connection.close()
        return True

    def count(self) -> int:
        connection = self._store._connect()
        try:
            row = connection.execute("SELECT count(*) FROM documents").fetchone()
        finally:
            connection.close()
        return int(row[0]) if row else 0


def document_hash(text: str) -> str:
    """A stable content hash for a document, so an unchanged one is recognisable."""
    return hashlib.blake2b(text.encode("utf-8", "replace"), digest_size=16).hexdigest()


def document_id(source_id: str, text_hash: str) -> str:
    """A document id that is stable for a given source and content hash."""
    return f"doc:{source_id}:{text_hash[:16]}"


def build_document_metadata(
    *,
    source_id: str,
    uri: str,
    title: str,
    text: str,
    chunk_ids: Sequence[str] = (),
    media_type: str = "",
    language: str = "",
    provider: str = "",
    license: str = "",
    category: str = "",
    tags: tuple[str, ...] = (),
    created_at: str = "",
) -> DocumentMetadata:
    """Assemble :class:`DocumentMetadata` from a record, hashing the text once."""
    digest = document_hash(text)
    return DocumentMetadata(
        doc_id=document_id(source_id, digest),
        source_id=source_id,
        uri=uri,
        title=title,
        text_hash=digest,
        char_count=len(text),
        chunk_ids=tuple(chunk_ids),
        media_type=media_type,
        language=language,
        provider=provider,
        license=license,
        category=category,
        tags=tuple(tags),
        created_at=created_at,
    )


def metadata_path(store_root: str | Path) -> Path:
    """Where document metadata lives, beside the ledger under the store root."""
    return Path(store_root) / "ingest" / "metadata.sqlite"


__all__ = [
    "SCHEMA",
    "DocumentMetadata",
    "DocumentMetadataStore",
    "build_document_metadata",
    "document_hash",
    "document_id",
    "metadata_path",
]
