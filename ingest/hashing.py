"""Content hashing for ingestion: stable ids, change detection, and dedup.

Three different questions need three different hashes, so they are kept apart:

* **content hash** -- over the exact text of a chunk. Two chunks with the same
  bytes are the same knowledge, so this is what duplicate detection keys on
  (``blake2b``, the same family :mod:`core.semantic_memory` and
  :mod:`ml.dataset` already use).
* **source key** -- a stable identity for a *source item* (a file path, a table
  row, a URL). It answers "have I seen this source before?" for incremental runs.
* **file fingerprint** -- size plus mtime for a local file. It answers "has this
  file changed since I last ingested it?" cheaply, without re-reading it.

The digests are keyed (``sha256``), not Python's built-in ``hash``, which is
salted per process and would make ids change between runs -- the same reasoning
:meth:`core.knowledge_update.KnowledgeUpdater._doc_id_for` already documents.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

DIGEST_CHARS = 16


def content_hash(text: str) -> str:
    """A stable, keyed digest of a chunk's exact text."""
    return hashlib.blake2b(text.encode("utf-8"), digest_size=16).hexdigest()


def source_key(uri: str, *, extra: str = "") -> str:
    """A stable identity for a source item within a source.

    ``uri`` names the item (a relative path, a primary-key value rendered as
    text, a URL); ``extra`` distinguishes parts of one item when a connector
    still wants an item-level key (for example a sheet or table name).
    """
    key = f"{uri}\x00{extra}" if extra else uri
    return hashlib.sha256(key.encode("utf-8")).hexdigest()[:DIGEST_CHARS]


def file_fingerprint(path: str | Path) -> str:
    """``size:mtime_ns`` for a local file, or ``""`` when it cannot be read.

    Cheap on purpose: it decides whether a file needs re-reading at all, and is
    never the sole basis for de-duplication (that is :func:`content_hash`).
    """
    source = Path(path)
    try:
        stat = source.stat()
    except OSError:
        return ""
    return f"{stat.st_size}:{stat.st_mtime_ns}"


__all__ = [
    "DIGEST_CHARS",
    "content_hash",
    "file_fingerprint",
    "source_key",
]
