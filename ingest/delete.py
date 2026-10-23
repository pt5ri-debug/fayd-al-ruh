"""Remove an ingested source from the store, safely and completely.

Deleting is the one mutating operation on the knowledge store, so it is done
through a *rewrite* rather than an in-place edit: every shard is streamed, the
chunks belonging to the source are dropped, and the survivors are written to a
fresh set of shards that replaces the old ones only after all of them are on
disk. If anything fails mid-way, the old shards are still intact, so a delete is
all-or-nothing.

The deletion is scoped by *source key* (the ledger knows which ``doc_id``s came
from a source), so it removes exactly what an import added -- never another
source's chunks and never a chunk that arrived through a different path.

Deleting from the store changes what a full-text index should contain, so the
index is dropped and must be rebuilt (``python -m cli ingest build-index``);
:func:`delete_source` reports that with ``reindex_required=True``.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from core.knowledge_base import KnowledgeBaseStore

from ingest.dedup import DedupLedger


@dataclass(frozen=True)
class DeletionReport:
    """What a deletion removed, and whether the index must be rebuilt."""

    source_key: str
    removed: int
    remaining: int
    reindex_required: bool = True

    def as_dict(self) -> dict:
        return {
            "source_key": self.source_key,
            "removed": self.removed,
            "remaining": self.remaining,
            "reindex_required": self.reindex_required,
        }


class SourceDeleter:
    """Removes a source's chunks by rewriting the store's shards."""

    def __init__(self, store: KnowledgeBaseStore, ledger: DedupLedger) -> None:
        self.store = store
        self.ledger = ledger

    def delete(self, source_key: str) -> DeletionReport:
        """Delete every chunk recorded for ``source_key``; rewrite the store."""
        doc_ids = set(self.ledger.source_chunks(source_key))
        if not doc_ids:
            return DeletionReport(source_key, removed=0, remaining=self.store.count())

        removed = 0
        survivors = []
        for chunk in self.store.iter_chunks():
            if chunk.chunk_id in doc_ids:
                removed += 1
            else:
                survivors.append(chunk)

        self._rewrite(survivors)
        self.ledger.drop_source(source_key)
        self._drop_index()
        return DeletionReport(source_key, removed=removed, remaining=len(survivors))

    def _rewrite(self, chunks: list) -> None:
        """Replace the store's shards with one fresh shard holding ``chunks``."""
        self.store.ensure()
        staging = self.store.root / "shard-00000.jsonl.tmp"
        with staging.open("w", encoding="utf-8") as handle:
            for chunk in chunks:
                handle.write(chunk.to_json() + "\n")
        # Swap only after the replacement is fully written.
        for path in self.store.shard_paths():
            path.unlink()
        staging.replace(self.store.root / "shard-00000.jsonl")

    def _drop_index(self) -> None:
        """Remove the full-text index and its sidecars so it can be rebuilt."""
        from core.knowledge_index import KnowledgeTextIndex

        index_path = Path(KnowledgeTextIndex(self.store.root).path)
        for path in (index_path, index_path.with_name(index_path.name + "-wal"), index_path.with_name(index_path.name + "-shm")):
            if path.exists():
                path.unlink()


def delete_source(
    source_key: str, *, store_dir: str | Path, ledger_dir: str | Path | None = None
) -> DeletionReport:
    """Convenience: build the store, ledger, and deleter, then delete."""
    from ingest.pipeline import ledger_path

    store = KnowledgeBaseStore(store_dir)
    ledger_root = Path(ledger_dir) if ledger_dir is not None else store.root
    ledger = DedupLedger(ledger_path(ledger_root))
    return SourceDeleter(store, ledger).delete(source_key)


__all__ = ["DeletionReport", "SourceDeleter", "delete_source"]
