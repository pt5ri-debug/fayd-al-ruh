"""Stream sources into the project's existing knowledge store.

The pipeline is the one place that turns a connector's records into stored,
citable chunks. It is deliberately thin because the heavy lifting already lives
elsewhere:

* chunking is :func:`core.knowledge_base.chunk_text` (reused, not rewritten);
* storage is :class:`core.knowledge_base.KnowledgeBaseStore` (sharded JSONL,
  ``manifest.json``, ``checkpoint.json``), the *same* store the retriever, the
  FTS index, and the training-side readers already use;
* de-duplication and incremental skipping are :class:`ingest.dedup.DedupLedger`.

Memory stays flat for a 1-100 GB source because nothing is ever a list of the
whole dataset: the connector *yields* records, a record is chunked, and each
chunk is appended and forgotten. The only per-source accumulation is one item's
chunks at a time, bounded by the source item's size.

It writes to the store and nothing else -- never ``ml/artifacts``,
``ml/llm/checkpoints``, or ``evaluation/baseline.json`` -- which is what keeps
ingestion and training separate.
"""

from __future__ import annotations

from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from core.knowledge_base import (
    DEFAULT_CHUNK_CHARS,
    DEFAULT_MIN_CHUNK,
    DEFAULT_OVERLAP,
    KnowledgeBaseStore,
    KnowledgeChunk,
    chunk_text,
)

from ingest.dedup import DedupLedger
from ingest.errors import (
    ConnectorError,
    ErrorLog,
    IngestCancelled,
    IngestInterrupted,
    IngestionError,
)
from ingest.hashing import content_hash
from ingest.jobs import CANCELLED, DONE, FAILED, JobState, JobStore
from ingest.metadata import (
    DocumentMetadataStore,
    build_document_metadata,
    metadata_path,
)
from ingest.records import (
    Connector,
    ConnectorInfo,
    IngestionReport,
    IngestItem,
    ItemOutcome,
    SourceRecord,
    SourceSpec,
)

#: A shard is rolled over once it reaches roughly this many bytes, so a 100 GB
#: ingest does not leave thousands of tiny shard files behind.
DEFAULT_SHARD_BYTES = 48 * 1024 * 1024

ProgressCallback = Callable[..., None]

#: How many actionable failure messages a job keeps (bounded so job files stay tiny).
MAX_JOB_FAILURES = 200

#: Source label stored on chunks, kept distinct from the wiki build's "wikipedia".
INGESTED_SOURCE = "ingested"


class IngestionPipeline:
    """Streams connector records into a knowledge store, with dedup and progress."""

    def __init__(
        self,
        store: KnowledgeBaseStore,
        ledger: DedupLedger,
        *,
        connectors: dict[str, Connector] | None = None,
        chunk_chars: int = DEFAULT_CHUNK_CHARS,
        overlap: int = DEFAULT_OVERLAP,
        min_chunk: int = DEFAULT_MIN_CHUNK,
        shard_bytes: int = DEFAULT_SHARD_BYTES,
        progress: ProgressCallback | None = None,
        errors: ErrorLog | None = None,
        job: JobState | None = None,
        jobs: JobStore | None = None,
        retries: int = 0,
        metadata: DocumentMetadataStore | None = None,
        on_item: Callable[[ItemOutcome], None] | None = None,
        on_record: Callable[[SourceRecord, int], None] | None = None,
        should_cancel: Callable[[], bool] | None = None,
        durable: bool = False,
    ) -> None:
        if chunk_chars < 1:
            raise ValueError("chunk_chars must be at least 1.")
        if not 0 <= overlap < chunk_chars:
            raise ValueError("overlap must be in [0, chunk_chars).")
        if shard_bytes < 1:
            raise ValueError("shard_bytes must be at least 1.")
        if retries < 0:
            raise ValueError("retries cannot be negative.")
        self.store = store
        self.ledger = ledger
        self.connectors = connectors or {}
        self.chunk_chars = chunk_chars
        self.overlap = overlap
        self.min_chunk = min_chunk
        self.shard_bytes = shard_bytes
        self.progress = progress
        self.errors = errors or ErrorLog()
        self.job = job
        self.jobs = jobs
        self.retries = retries
        # Optional per-item view: the loop reports one :class:`ItemOutcome` per
        # source file/table/URL so a caller can show per-file results without
        # re-walking the sources. The aggregate report is unchanged.
        self.on_item = on_item
        # Optional per-record hook: called with each stored record's title and the
        # number of chunks it produced. A resumable SQL run uses it to advance a
        # batch cursor *after* the record is durably written.
        self.on_record = on_record
        # Optional cooperative cancel: checked between records and between items.
        # A truthy result stops the run without corrupting the source and without
        # marking the job done.
        self.should_cancel = should_cancel
        # Durable mode: fsync each shard write before advancing a checkpoint, so
        # persisted progress never runs ahead of what is on disk. Off for the
        # fast bulk wiki build, on for resumable imports.
        self.durable = durable
        # Per-document metadata (path, hash, timestamp, chunk ids) is *recorded*
        # but never read back by the pipeline, so a metadata-write failure must
        # not fail an ingest that otherwise succeeded. Best-effort, like the
        # manifest flush.
        self.metadata = metadata
        self._shard: int | None = None
        self._shard_bytes: int = 0

    # -- public API -------------------------------------------------------

    def ingest(
        self,
        spec: SourceSpec,
        *,
        connector: Connector | None = None,
        incremental: bool = True,
        connector_name: str = "",
        resume_from: int = 0,
    ) -> IngestionReport:
        """Ingest one source. Never raises for a bad source; records it instead."""
        connector = connector or self._resolve(spec)
        info = connector.info
        report = IngestionReport(sources=1)
        if self.job is not None and not resume_from:
            resume_from = self.job.position
        self._report(self._start_message(spec, info, incremental))
        try:
            self._ingest_plan(
                self._effective_spec(spec),
                connector,
                report,
                incremental=incremental,
                resume_from=resume_from,
            )
        except IngestCancelled:
            report.cancelled = True
            self._report(f"cancelled: {spec.uri}")
        except IngestInterrupted:
            raise
        except (ConnectorError, IngestionError, OSError, ValueError) as error:
            report.failed += 1
            self.errors.add(spec.uri, type(error).__name__, str(error))
            self._report(f"failed: {spec.uri}: {error}")
            self._note_failure(spec.uri, str(error))
        self._flush_manifest()
        self._finish_job(report)
        return report

    def _finish_job(self, report: IngestionReport) -> None:
        if self.job is None:
            return
        if report.cancelled:
            self.job.status = CANCELLED
        elif report.failed:
            self.job.status = FAILED
        else:
            self.job.status = DONE
        self._save_job(report)

    def _save_job(self, report: IngestionReport | None = None) -> None:
        if self.job is None:
            return
        if report is not None:
            self.job.records = report.records
            self.job.chunks = report.chunks
            self.job.bytes_written = report.bytes_written
        if self.jobs is not None:
            self.jobs.save(self.job)

    # -- internals --------------------------------------------------------

    def _effective_spec(self, spec: SourceSpec) -> SourceSpec:
        """The spec with the store's own directory excluded from a directory walk.

        Scanning a folder that happens to contain the knowledge store must not
        ingest the store's own shards as if they were source documents.
        """
        roots = list(spec.option("exclude_roots") or [])
        roots.append(str(self.store.root))
        options = dict(spec.options)
        options["exclude_roots"] = roots
        return SourceSpec(uri=spec.uri, options=options)

    def _resolve(self, spec: SourceSpec) -> Connector:
        from ingest.registry import build_default_registry

        return build_default_registry().resolve(spec)

    def _ingest_plan(
        self,
        spec: SourceSpec,
        connector: Connector,
        report: IngestionReport,
        *,
        incremental: bool,
        resume_from: int,
    ) -> None:
        position = 0
        for item in connector.plan(spec):
            position += 1
            if position <= resume_from:
                continue
            if self._cancelled():
                raise IngestCancelled("cancelled between items")
            if incremental and self.ledger.unchanged(item.source_key, item.fingerprint):
                report.skipped_unchanged += 1
                self._emit_item(item, "unchanged")
                if self.progress and report.skipped_unchanged % 200 == 0:
                    self._report(f"skipped {report.skipped_unchanged} unchanged items")
                self._record_position(position, report)
                continue
            before = (report.records, report.chunks, report.bytes_written)
            try:
                self._ingest_item(connector, spec, item, report)
            except (IngestCancelled, IngestInterrupted):
                raise
            except (ConnectorError, IngestionError, OSError, ValueError) as error:
                report.failed += 1
                self.errors.add(item.uri, type(error).__name__, str(error))
                self._report(f"failed: {item.uri}: {error}")
                self._emit_item(item, "failed", message=str(error))
                self._note_failure(item.uri, str(error))
            else:
                self._emit_item(
                    item,
                    "ok",
                    records=report.records - before[0],
                    chunks=report.chunks - before[1],
                    bytes_written=report.bytes_written - before[2],
                )
            self._record_position(position, report)

    def _cancelled(self) -> bool:
        return self.should_cancel is not None and bool(self.should_cancel())

    def _note_failure(self, source: str, message: str) -> None:
        """Record an actionable per-item failure on the job, bounded in size."""
        if self.job is None:
            return
        if len(self.job.failures) >= MAX_JOB_FAILURES:
            return
        self.job.failures.append(f"{source}: {message.splitlines()[0] if message else 'failed'}")

    def _emit_item(
        self,
        item: IngestItem,
        status: str,
        *,
        records: int = 0,
        chunks: int = 0,
        bytes_written: int = 0,
        message: str = "",
    ) -> None:
        """Report one item's outcome, when a caller asked for the per-item view."""
        if self.on_item is None:
            return
        self.on_item(
            ItemOutcome(
                source_key=item.source_key,
                uri=item.uri,
                title=item.title,
                status=status,
                records=records,
                chunks=chunks,
                bytes_written=bytes_written,
                message=message,
            )
        )

    def _record_position(self, position: int, report: IngestionReport) -> None:
        """Persist progress after each item so a long job is resumable."""
        if self.job is None:
            return
        self.job.position = position
        self._save_job(report)

    def _ingest_item(
        self,
        connector: Connector,
        spec: SourceSpec,
        item: IngestItem,
        report: IngestionReport,
    ) -> None:
        """Ingest one item, retrying a transient read failure a bounded number of times.

        Only the *read* is retried: a failed attempt's counters are rolled back
        and its stream is rebuilt from the start, so a retry never double-counts
        and never marks unwritten work as done.
        """
        attempt = 0
        while True:
            baseline = (report.records, report.chunks, report.bytes_written)
            try:
                self._read_item(connector, item, report)
                break
            except (IngestCancelled, IngestInterrupted):
                # A cancel or a deliberate interruption is the caller's decision,
                # not a transient failure: do not retry it, let it unwind.
                raise
            except (ConnectorError, IngestionError, OSError, ValueError) as error:
                # A failed attempt may have written some chunks; the ledger's
                # content hashes keep the retry from storing them twice, and the
                # report counters are rolled back so the run stays honest.
                report.records, report.chunks, report.bytes_written = baseline
                if attempt >= self.retries:
                    raise
                attempt += 1
                self._report(
                    f"retry {attempt}/{self.retries} for {item.uri}: {error}"
                )

    def _read_item(
        self, connector: Connector, item: IngestItem, report: IngestionReport
    ) -> None:
        item_chunks = 0
        item_bytes = 0
        for record in connector.stream_item(item):
            if self._cancelled():
                # Stop before storing anything for this record; the item's
                # already-written records stay committed, the rest is left for
                # the resume. Nothing is marked done that was not written.
                raise IngestCancelled("cancelled during item")
            report.records += 1
            written, chunk_ids = self._store_record(record, item)
            count = len(chunk_ids)
            item_chunks += count
            item_bytes += written
            report.chunks += count
            report.bytes_written += written
            self._record_metadata(record, item, chunk_ids, connector.info.name)
            if self.on_record is not None:
                # Called only after the record's chunks are durably written, so a
                # caller can advance a batch cursor in step with the store.
                self.on_record(record, count)
        self.ledger.record_source(
            item.source_key,
            uri=item.uri,
            connector=connector.info.name,
            fingerprint=item.fingerprint,
            chunks=item_chunks,
            bytes_written=item_bytes,
        )
        if self.progress and report.records % 50 == 0:
            self._report(
                f"{report.records} records, {report.chunks} chunks, "
                f"{report.bytes_written / 1024 / 1024:.1f} MB"
            )

    def _record_metadata(
        self,
        record: SourceRecord,
        item: IngestItem,
        chunk_ids: list[str],
        connector: str,
    ) -> None:
        """Record per-document metadata (best effort; never fails the ingest).

        The store already holds the text and the ledger holds the dedup keys;
        this adds the *bookkeeping* the store does not keep -- source path,
        filename/extension, content hash, media type, ingestion timestamp, and
        the chunk ids produced -- without duplicating any text.
        """
        if self.metadata is None:
            return
        source_id = item.source_key
        meta = record.metadata or {}
        try:
            metadata = build_document_metadata(
                source_id=source_id,
                uri=str(meta.get("path", item.uri)),
                title=record.title,
                text=record.text,
                chunk_ids=tuple(chunk_ids),
                media_type=str(meta.get("kind", "")),
                language="",
                provider=connector,
                license=record.license,
                category=str(meta.get("category", "")),
                tags=(str(meta.get("relative", "")),) if meta.get("relative") else (),
                created_at=datetime.now(UTC).isoformat(timespec="seconds"),
            )
            self.metadata.record(metadata)
        except (OSError, ValueError):
            # Metadata is a side record; a write failure must not lose the ingest.
            pass

    def _store_record(self, record: SourceRecord, item: IngestItem) -> tuple[int, list[str]]:
        """Chunk one record, store the new chunks, and return (bytes, chunk ids)."""
        pending: list[KnowledgeChunk] = []
        digests: list[str] = []
        for piece in chunk_text(
            record.text, chunk_chars=self.chunk_chars, overlap=self.overlap, min_chunk=self.min_chunk
        ):
            digest = content_hash(piece)
            if self.ledger.contains_chunk(digest):
                continue
            pending.append(
                KnowledgeChunk(
                    chunk_id=self._chunk_id(digest),
                    title=record.title,
                    text=piece,
                    source=record.source or INGESTED_SOURCE,
                    url=record.url,
                    license=record.license,
                    category=str(record.metadata.get("category", "")),
                )
            )
            digests.append(digest)
        if not pending:
            return 0, []
        written = self._write(pending)
        for digest, chunk in zip(digests, pending, strict=False):
            self.ledger.record_chunk(digest, source_key=item.source_key, doc_id=chunk.chunk_id)
        return written, [chunk.chunk_id for chunk in pending]

    def _write(self, chunks: list[KnowledgeChunk]) -> int:
        if self._shard is None:
            self._shard = self.store.next_shard_index()
            self._shard_bytes = 0
        written = self.store.append(self._shard, chunks, sync=self.durable)
        self._shard_bytes += written
        if self._shard_bytes >= self.shard_bytes:
            self._shard = None
            self._shard_bytes = 0
        return written

    def _chunk_id(self, digest: str) -> str:
        # Content-addressed: two identical chunks are one chunk, which is exactly
        # what duplicate detection wants, and the id is stable across runs.
        return f"ingest:{digest}"

    def _flush_manifest(self) -> None:
        """Merge a small honest summary into the store manifest (best effort)."""
        try:
            manifest = self.store.load_manifest()
        except (OSError, ValueError):
            manifest = {}
        ingest = dict(manifest.get("ingest", {}))
        self._merge_report(ingest)
        manifest["ingest"] = ingest
        try:
            self.store.write_manifest(manifest)
        except OSError:
            pass

    def _merge_report(self, ingest: dict[str, Any]) -> None:
        stats = self.ledger.stats()
        ingest["sources"] = stats["sources"]
        ingest["chunks"] = stats["chunks"]
        ingest["bytes"] = stats["bytes"]

    def _report(self, message: str) -> None:
        if self.progress:
            self.progress(message)

    def _start_message(self, spec: SourceSpec, info: ConnectorInfo, incremental: bool) -> str:
        mode = "incremental" if incremental else "full"
        return f"ingesting {spec.uri} with connector {info.name} ({mode})"


def ingest_source(
    spec: SourceSpec,
    *,
    store_dir: str | None = None,
    incremental: bool = True,
    allow_web: bool = False,
    retries: int = 0,
    progress: ProgressCallback | None = None,
) -> tuple[IngestionReport, ErrorLog]:
    """One-shot helper: build the default pipeline and ingest ``spec``."""
    from core.knowledge_base import default_knowledge_base_dir

    from ingest.dedup import DedupLedger
    from ingest.registry import build_default_registry

    store = KnowledgeBaseStore(store_dir or default_knowledge_base_dir())
    ledger = DedupLedger(ledger_path(store.root))
    metadata = DocumentMetadataStore(metadata_path(store.root))
    pipeline = IngestionPipeline(store, ledger, progress=progress, retries=retries, metadata=metadata)
    registry = build_default_registry(allow_web=allow_web)
    connector = registry.resolve(spec)
    report = pipeline.ingest(spec, connector=connector, incremental=incremental)
    return report, pipeline.errors


def ledger_path(store_root: str | Path) -> Path:
    """The dedup ledger beside a store root (kept under the same data folder)."""
    return Path(store_root) / "ingest" / "dedup.sqlite"


def ingest_jobs_dir(store_root: str | Path) -> Path:
    """Where resumable job state lives, beside the ledger under the store root."""
    return Path(store_root) / "ingest" / "jobs"


__all__ = [
    "DEFAULT_SHARD_BYTES",
    "INGESTED_SOURCE",
    "IngestionPipeline",
    "ingest_jobs_dir",
    "ingest_source",
    "ledger_path",
]
