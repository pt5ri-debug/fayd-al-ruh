"""Resumable, idempotent SQL imports: durable progress with safe recovery.

A large SQLite import can be interrupted (a cancel, a crash, a locked table). This
module makes such a run *resumable* and *idempotent*, on top of the pieces that
already exist:

* the connector reads a bounded *window* of a table (:mod:`ingest.connectors.sqlite`);
* the pipeline streams a window's rows into the shared knowledge store, writing
  durably (``fsync``) before it returns (:mod:`ingest.pipeline`);
* the dedup ledger makes re-storing an already-stored row a no-op.

The order of operations is the whole point:

1. read one window (``resume_row`` .. ``resume_row + batch_size``) of a table;
2. **write and fsync** every row's chunks to the store;
3. *then* advance the table's ``resume_count`` and save the job.

Because step 3 happens only after step 2, a checkpoint never claims a batch is
done before its outputs are durably saved. If the process dies between 2 and 3,
the next run re-reads that window; every row already written is a content-hash
duplicate, so it stores nothing twice -- the batch is effectively replayed for
free. That is exactly-once under the *content* identity the store uses, plus a
per-table primary key guard that rejects a window whose first row is already
committed (so a spurious replay is detected even if the text differed).

Honest limits, stated rather than papered over:

* Identity is by **row content hash** (the store's existing dedup). A *complete*
  primary key additionally gives each row a stable name (:class:`TablePlan`).
* Resume is by **row offset**. Offsets are not safe across inserts/deletes/
  reorders, so a table whose structure changed is refused and must be
  re-imported explicitly; a table with no primary key can be resumed by offset
  for a full import but has **no** stable row identity -- row-level sync and
  deletion sync are never claimed.
* There is **no** incremental change detection: a re-run reads the source again.

Job state lives with the rest of the ingestion bookkeeping, under
``<store>/ingest/jobs-sql/`` inside the knowledge subsystem -- never in
``ml/artifacts``, ``ml/llm/checkpoints``, or ``evaluation/``.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from core.knowledge_base import KnowledgeBaseStore
from ingest.connectors.sqlite import SqliteConnector, TablePlan
from ingest.dedup import DedupLedger
from ingest.errors import ConnectorError, IngestCancelled, IngestInterrupted, IngestionError
from ingest.jobs import CANCELLED, DONE, FAILED, RUNNING, write_atomic_text
from ingest.pipeline import IngestionPipeline, ledger_path
from ingest.records import SourceRecord, SourceSpec

#: Rows read and committed per batch. Small enough that losing an uncommitted
#: batch is cheap, large enough that bookkeeping is not the bottleneck.
DEFAULT_BATCH_SIZE = 500
MAX_BATCH_SIZE = 1_000_000
#: Bounded retries for a transient batch failure before the table is failed.
DEFAULT_BATCH_RETRIES = 2
#: How many recent primary-key identities a table remembers for the replay guard.
#: A completeness aid, not a correctness requirement: the content hash is the
#: source of truth, and this only detects a repeated window early.
MAX_COMMITTED_KEYS = 5000

#: Table/phase status.
PENDING = "pending"
IN_PROGRESS = "in_progress"
INTERRUPTED = "interrupted"

#: Schema-change recovery policies.
SCHEMA_FAIL = "fail"
"""Default: refuse to continue when a table's structure changed (safe by default)."""
SCHEMA_FULL_REIMPORT = "full-reimport"
"""Explicit opt-in: discard this table's progress and re-import it from row 0."""


class SqlJobError(IngestionError):
    """Base class for the SQL-import job's failures."""


class SchemaChangedError(SqlJobError):
    """A source table's structure changed; an explicit recovery policy is required."""


#: Backwards-compatible alias for the interrupt marker.
Interrupted = IngestInterrupted


@dataclass
class TableProgress:
    """Durable progress for one table within a job."""

    table: str
    signature: str = ""
    resume_count: int = 0
    """Rows already durably committed; the next window starts after them."""
    resume_supported: bool = True
    stable_identity: bool = False
    status: str = PENDING
    target_rows: int = 0
    committed_primary_keys: list[str] = field(default_factory=list)
    """The most recent stable row identities committed (bounded replay guard)."""

    def remember(self, keys: list[str]) -> None:
        self.committed_primary_keys.extend(keys)
        if len(self.committed_primary_keys) > MAX_COMMITTED_KEYS:
            self.committed_primary_keys = self.committed_primary_keys[-MAX_COMMITTED_KEYS:]


@dataclass
class SqlImportJob:
    """The durable state of one resumable SQL import."""

    job_id: str
    source_uri: str
    connector: str = "sqlite"
    source_path: str = ""
    store_dir: str = ""
    batch_size: int = DEFAULT_BATCH_SIZE
    status: str = RUNNING
    created_at: str = ""
    updated_at: str = ""
    tables: dict[str, TableProgress] = field(default_factory=dict)
    failures: list[str] = field(default_factory=list)
    options: dict[str, str] = field(default_factory=dict)
    """The connector options the job was created with (table/column selection,
    batch settings, root).

    Stored so a *resume* rebuilds the exact same window policy: a resumed run
    must never silently widen or narrow what an earlier run selected, and a GUI
    that only knows the job id can still continue the job faithfully.
    """

    def __post_init__(self) -> None:
        if not self.job_id.strip():
            raise ValueError("SqlImportJob job_id cannot be empty.")
        if not self.source_uri.strip():
            raise ValueError("SqlImportJob source_uri cannot be empty.")

    def touch(self) -> None:
        self.updated_at = datetime.now(timezone.utc).isoformat(timespec="seconds")

    def as_dict(self) -> dict[str, Any]:
        data = asdict(self)
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SqlImportJob":
        raw_tables = data.get("tables", {})
        tables = {
            name: TableProgress(**value) for name, value in raw_tables.items()
        }
        return cls(
            job_id=str(data["job_id"]),
            source_uri=str(data["source_uri"]),
            connector=str(data.get("connector", "sqlite")),
            source_path=str(data.get("source_path", "")),
            store_dir=str(data.get("store_dir", "")),
            batch_size=int(data.get("batch_size", DEFAULT_BATCH_SIZE)),
            status=str(data.get("status", RUNNING)),
            created_at=str(data.get("created_at", "")),
            updated_at=str(data.get("updated_at", "")),
            tables=tables,
            failures=list(data.get("failures", [])),
            options=dict(data.get("options", {})),
        )


class SqlJobStore:
    """A directory of :class:`SqlImportJob` files (``*.sqljob.json``), one per job."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)

    def path(self, job_id: str) -> Path:
        return self.root / f"{_safe_name(job_id)}.sqljob.json"

    def exists(self, job_id: str) -> bool:
        return self.path(job_id).exists()

    def save(self, job: SqlImportJob) -> SqlImportJob:
        job.touch()
        write_atomic_text(
            self.path(job.job_id),
            json.dumps(job.as_dict(), ensure_ascii=False, indent=2),
        )
        return job

    def load(self, job_id: str) -> SqlImportJob | None:
        path = self.path(job_id)
        if not path.exists():
            return None
        return SqlImportJob.from_dict(json.loads(path.read_text(encoding="utf-8")))

    def list(self) -> list[SqlImportJob]:
        if not self.root.exists():
            return []
        jobs = []
        for path in sorted(self.root.glob("*.sqljob.json")):
            try:
                jobs.append(SqlImportJob.from_dict(json.loads(path.read_text(encoding="utf-8"))))
            except (ValueError, TypeError, KeyError):
                continue
        return jobs

    def remove(self, job_id: str) -> bool:
        path = self.path(job_id)
        if path.exists():
            path.unlink()
            return True
        return False


class SqlImportRunner:
    """Runs (and resumes) a resumable, idempotent SQL import."""

    def __init__(
        self,
        store_dir: str | Path,
        *,
        connector: Any | None = None,
        jobs_dir: str | Path | None = None,
        batch_size: int = DEFAULT_BATCH_SIZE,
        retries: int = 0,
        batch_retries: int = DEFAULT_BATCH_RETRIES,
        progress: Callable[[str], None] | None = None,
        should_cancel: Callable[[], bool] | None = None,
        on_record: Callable[[SourceRecord, int], None] | None = None,
    ) -> None:
        if not 1 <= batch_size <= MAX_BATCH_SIZE:
            raise ValueError(f"batch_size must be in [1, {MAX_BATCH_SIZE}].")
        if retries < 0 or batch_retries < 0:
            raise ValueError("retries cannot be negative.")
        self.store_dir = Path(store_dir)
        self.store = KnowledgeBaseStore(self.store_dir)
        self.ledger = DedupLedger(ledger_path(self.store.root))
        self.connector = connector or SqliteConnector()
        self.jobs = SqlJobStore(jobs_dir or sql_jobs_dir(self.store_dir))
        self.batch_size = batch_size
        self.retries = retries
        self.batch_retries = batch_retries
        self.progress = progress
        self.should_cancel = should_cancel
        # Optional hook called after each record is durably stored; a caller may
        # raise :class:`Interrupted` here to simulate a crash at a known point.
        self.on_record = on_record

    # -- public API -------------------------------------------------------

    def run(
        self,
        spec: SourceSpec,
        *,
        job_id: str,
        schema_change: str = SCHEMA_FAIL,
    ) -> SqlImportJob:
        """Import ``spec`` resumably; create the job or continue an existing one."""
        if schema_change not in (SCHEMA_FAIL, SCHEMA_FULL_REIMPORT):
            raise ValueError(f"Unknown schema_change policy: {schema_change!r}.")
        job = self._load_or_create(spec, job_id)
        # Each run recomputes the final status, so start from RUNNING.
        job.status = RUNNING
        try:
            plans = self._plans(spec)
        except (ConnectorError, IngestionError, OSError, ValueError) as error:
            job.status = FAILED
            self._fail(job, f"{spec.uri}: {error}")
            self.jobs.save(job)
            return job

        self._reconcile_schemas(job, plans, policy=schema_change)
        if job.status == FAILED:
            self.jobs.save(job)
            return job
        # Persist the reconciled plan before reading, so an interruption during
        # the very first batch still leaves a complete table entry on disk.
        self.jobs.save(job)

        self._report(f"sql import {job.job_id}: {len(plans)} table(s)")
        for plan in plans:
            if job.status != RUNNING:
                break
            self._run_table(spec, job, plan)
        if job.status == RUNNING:
            job.status = DONE
        self.jobs.save(job)
        return job

    def resume(self, spec: SourceSpec, job_id: str, *, schema_change: str = SCHEMA_FAIL) -> SqlImportJob:
        """Continue an existing job (raises if it does not exist).

        The job's own recorded options are used, so a resume reproduces the
        original table/column selection even when the caller only passes a bare
        URI -- a resumed run never silently changes *what* was selected.
        """
        job = self.jobs.load(job_id)
        if job is None:
            raise SqlJobError(f"No such SQL import job: {job_id}")
        return self.run(_resume_spec(spec, job), job_id=job_id, schema_change=schema_change)

    # -- job lifecycle ----------------------------------------------------

    def _load_or_create(self, spec: SourceSpec, job_id: str) -> SqlImportJob:
        existing = self.jobs.load(job_id)
        if existing is not None:
            return existing
        job = SqlImportJob(
            job_id=job_id,
            source_uri=spec.uri,
            connector=self.connector.info.name,
            source_path=str(spec.options.get("source_path", spec.uri)),
            store_dir=str(self.store_dir),
            batch_size=self.batch_size,
            status=RUNNING,
            created_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
            options=dict(spec.options),
        )
        self.jobs.save(job)
        return job

    def _plans(self, spec: SourceSpec) -> tuple[TablePlan, ...]:
        return tuple(self.connector.describe(spec))

    def _reconcile_schemas(
        self, job: SqlImportJob, plans: tuple[TablePlan, ...], *, policy: str
    ) -> None:
        """Compare each table's structure to the recorded one and react to a change."""
        for plan in plans:
            progress = job.tables.get(plan.table)
            if progress is None:
                job.tables[plan.table] = TableProgress(
                    table=plan.table,
                    signature=plan.signature,
                    resume_supported=plan.resume_supported,
                    stable_identity=plan.stable_identity,
                    status=PENDING,
                    target_rows=plan.total_rows,
                )
                continue
            progress.resume_supported = plan.resume_supported
            progress.stable_identity = plan.stable_identity
            progress.target_rows = plan.total_rows
            if progress.signature and progress.signature != plan.signature:
                if policy == SCHEMA_FAIL:
                    job.status = FAILED
                    self._fail(
                        job,
                        f"{plan.table}: schema changed (recorded {progress.signature}, "
                        f"now {plan.signature}); re-run with schema_change="
                        f"{SCHEMA_FULL_REIMPORT!r} to discard progress and re-import this "
                        f"table from row 0.",
                    )
                    return
                # Explicit safe recovery: drop the table's progress and re-read it.
                progress.signature = plan.signature
                progress.resume_count = 0
                progress.committed_primary_keys = []
                progress.status = PENDING
                self._report(
                    f"{plan.table}: schema changed; re-importing from row 0 "
                    f"(schema_change={SCHEMA_FULL_REIMPORT})"
                )

    def _run_table(self, spec: SourceSpec, job: SqlImportJob, plan: TablePlan) -> None:
        progress = job.tables[plan.table]
        self._report(
            f"table {plan.table}: {plan.total_rows} row(s), "
            f"resume_from={progress.resume_count}, identity="
            f"{'primary-key' if plan.stable_identity else 'content-only'}"
            + ("" if plan.resume_supported else " (single pass)")
        )
        while job.status == RUNNING:
            if progress.target_rows and progress.resume_count >= progress.target_rows:
                progress.status = DONE
                self.jobs.save(job)
                break
            try:
                running = self._run_batch_with_retries(spec, job, plan)
            except Interrupted:
                # A controlled interruption: leave the last committed checkpoint
                # intact and mark the job interrupted; the next resume continues.
                job.status = INTERRUPTED
                progress.status = IN_PROGRESS
                self.jobs.save(job)
                return
            except IngestCancelled:
                job.status = CANCELLED
                progress.status = IN_PROGRESS
                self.jobs.save(job)
                self._report(f"cancelled: {job.job_id} at table {plan.table}")
                return
            if running == 0:
                progress.status = DONE
                break
            progress.resume_count += running
            progress.status = IN_PROGRESS
            self.jobs.save(job)
        # Record the source signature only once the table is fully imported, so an
        # interrupted run does not look "complete" and a change can still be seen.
        if progress.status == DONE and plan.signature:
            progress.signature = plan.signature
        self.jobs.save(job)

    def _run_batch_with_retries(
        self, spec: SourceSpec, job: SqlImportJob, plan: TablePlan
    ) -> int:
        """Read one window, retrying a transient failure a bounded number of times."""
        attempt = 0
        while True:
            try:
                return self._run_batch(spec, job, plan)
            except (Interrupted, IngestCancelled, SchemaChangedError):
                raise
            except (ConnectorError, IngestionError, OSError, ValueError) as error:
                if attempt >= self.batch_retries:
                    job.status = FAILED
                    job.tables[plan.table].status = FAILED
                    self._fail(job, f"{plan.table}: {error}")
                    self.jobs.save(job)
                    self._report(f"failed: {plan.table}: {error}")
                    return 0
                attempt += 1
                self._report(
                    f"retry {attempt}/{self.batch_retries} on {plan.table}: {error}"
                )

    def _run_batch(self, spec: SourceSpec, job: SqlImportJob, plan: TablePlan) -> int:
        """Read and durably store one window; return the number of rows committed.

        Nothing is written to the job file here: the caller advances the
        checkpoint only after this returns, so a crash inside the window cannot
        leave a checkpoint ahead of the data.
        """
        progress = job.tables[plan.table]
        # A table with no deterministic order cannot be windowed safely, so it is
        # read in one full pass; everything else reads one bounded window.
        if progress.resume_supported:
            resume_row, max_rows = progress.resume_count, self.batch_size
        else:
            resume_row, max_rows = 0, 0
        window = SourceSpec(
            uri=spec.uri,
            options={
                **spec.options,
                "include_tables": plan.table,
                # One row per record keeps record count == row count and makes the
                # per-row identity line up exactly with the checkpoint offset.
                "row_group": 1,
                "resume_row": resume_row,
                "max_rows": max_rows,
            },
        )
        identifiers: list[str] = []
        rows_seen = 0
        last_row_end = progress.resume_count

        def _capture_record(record: SourceRecord, _chunks: int) -> None:
            nonlocal rows_seen, last_row_end
            rows_seen += 1
            end = int(record.metadata.get("row_end") or 0)
            if end:
                # The connector numbers rows absolutely in the table, so the last
                # successfully stored row is the exact next checkpoint.
                last_row_end = end
            identifiers.append(_identity_key(record.text))
            if self.on_record is not None:
                self.on_record(record, _chunks)

        pipeline = IngestionPipeline(
            self.store,
            self.ledger,
            retries=self.retries,
            should_cancel=self.should_cancel,
            on_record=_capture_record,
            durable=True,
        )
        report = pipeline.ingest(window, connector=self.connector)
        if report.cancelled:
            raise IngestCancelled(f"cancelled at table {plan.table}")
        if report.failed:
            first = pipeline.errors.errors[0].message if pipeline.errors.errors else ""
            raise SqlJobError(f"{plan.table}: {first or 'a row failed to import'}")

        # Replay guard: the window was read, but if its rows are already committed
        # (a crash between writing and checkpointing), the second pass stored
        # nothing new -- the content hashes deduped it. The checkpoint still
        # advances so the run is not stuck on that window.
        if plan.stable_identity and identifiers and identifiers[0]:
            already = identifiers[0] in set(progress.committed_primary_keys)
            verb = "replayed (no duplicate stored)" if already else f"committed {rows_seen} row(s)"
            self._report(f"table {plan.table}: window at row {progress.resume_count} {verb}")
        if plan.stable_identity:
            progress.remember([key for key in identifiers if key])
        return max(0, last_row_end - progress.resume_count)

    # -- helpers ----------------------------------------------------------

    def _fail(self, job: SqlImportJob, message: str) -> None:
        short = message.splitlines()[0] if message else "failed"
        if len(job.failures) < 200:
            job.failures.append(short)

    def _report(self, message: str) -> None:
        if self.progress:
            self.progress(message)


def _resume_spec(spec: SourceSpec, job: SqlImportJob) -> SourceSpec:
    """Rebuild a resume spec: the caller's URI plus the job's recorded options.

    Explicit options on ``spec`` win (a caller may deliberately re-select), but
    anything the caller did not set falls back to what the job was created with,
    so a resume is faithful by default.
    """
    if not job.options:
        return spec
    return SourceSpec(spec.uri, options={**job.options, **spec.options})


def _identity_key(text: str) -> str:
    """Extract the stable identity token (the leading ``[k=v; ...]``) from a row."""
    if not text:
        return ""
    first_line = text.split("\n", 1)[0]
    if first_line.startswith("[") and "]" in first_line:
        return first_line[1 : first_line.index("]")]
    return ""


def _safe_name(job_id: str) -> str:
    return "".join(char if char.isalnum() or char in "-_." else "_" for char in job_id)


def job_fingerprint(payload: str) -> str:
    """A short stable digest, for callers that want an id from source content."""
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def sql_jobs_dir(store_root: str | Path) -> Path:
    """Where SQL-import job state lives: under the store, inside the subsystem."""
    return Path(store_root) / "ingest" / "jobs-sql"


__all__ = [
    "DEFAULT_BATCH_RETRIES",
    "DEFAULT_BATCH_SIZE",
    "INTERRUPTED",
    "IN_PROGRESS",
    "MAX_BATCH_SIZE",
    "PENDING",
    "SCHEMA_FAIL",
    "SCHEMA_FULL_REIMPORT",
    "Interrupted",
    "SchemaChangedError",
    "SqlImportJob",
    "SqlImportRunner",
    "SqlJobError",
    "SqlJobStore",
    "TableProgress",
    "job_fingerprint",
    "sql_jobs_dir",
]
