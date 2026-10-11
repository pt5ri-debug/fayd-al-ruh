"""Local-first knowledge ingestion: connectors, a streaming pipeline, and dedup.

This package turns *sources* -- local files, SQL databases, and (optionally,
and only with explicit authorisation) the internet -- into the project's
existing citable knowledge base, without ever loading a whole dataset into
memory.

It is deliberately a **separate pipeline from model training**. It writes only
to the on-disk knowledge store (``core.knowledge_base.KnowledgeBaseStore`` under
the data directory); it never imports ``ml.llm`` or ``ml.release`` and never
writes ``ml/artifacts``, ``ml/llm/checkpoints``, or ``evaluation/baseline.json``.
The training pipeline reads the same store through its own readers
(``ml.knowledge_corpus`` / ``ml.llm.corpus``), unchanged.

The pieces:

* :mod:`ingest.records` -- the small contracts (``SourceRecord``, ``Connector``,
  ``IngestionReport``) every connector speaks.
* :mod:`ingest.registry` -- resolve a source to the connector that can read it.
* :mod:`ingest.connectors` -- one module per family (files, SQL, web).
* :mod:`ingest.pipeline` -- stream records, chunk, and append to the store.
* :mod:`ingest.hashing` / :mod:`ingest.dedup` -- content hashing and a ledger so
  an incremental re-run skips what it already ingested.
* :mod:`ingest.jobs` -- resumable job state for long imports.
* :mod:`ingest.errors` -- a per-source error report that never aborts a batch.
* :mod:`ingest.delete` -- safe, all-or-nothing removal of a source's chunks.
"""

from ingest.records import (
    Connector,
    ConnectorInfo,
    IngestItem,
    IngestionReport,
    SourceRecord,
    SourceSpec,
)

__all__ = [
    "Connector",
    "ConnectorInfo",
    "IngestItem",
    "IngestionReport",
    "SourceRecord",
    "SourceSpec",
]
