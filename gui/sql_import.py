"""The SQL import front end for the knowledge subsystem (Tk-free).

This is the layer between the window's SQL import screen and the SQL ingestion
service that the CLI's ``ingest sql-import`` command already uses
(:mod:`ingest.sql_job` + :mod:`ingest.connectors.sqlite`). It adds **no** new
ingestion mechanism: it *drives* the one the project has, so the window, the CLI,
and the pipeline all share the same read-only connector, the same dedup ledger,
the same durable checkpoint, and the same job files under
``<store>/ingest/jobs-sql``.

What the window needs, and where it comes from:

* **file selection** -- a path the user picks; the file is opened read-only and is
  never created or written (the connector's own guarantee);
* **connection test / schema inspection** -- :meth:`SqlImportService.inspect`
  reads SQLite's own metadata (tables, columns, declared types, primary keys) and
  the current row counts, without reading any row value;
* **table/column selection and exclusion** -- turned into the connector's
  ``include_/exclude_`` options and recorded on the job, so a resume reproduces
  the same policy;
* **limited data preview** -- the connector's *own* rendering of at most a few
  rows, so the preview shows exactly how a row will be stored;
* **long-running import** -- :class:`ingest.sql_job.SqlImportRunner` on the
  caller's worker thread, with a durable checkpoint after every batch;
* **cancel / resume / retry** -- a cooperative ``should_cancel`` predicate and
  the runner's resumable job, never a second implementation.

Only **SQLite** is enabled at this stage; :data:`DATABASE_TYPES` is the single
source of that list, so a second engine is a one-line change here and nowhere
else.

Security, deliberately:

* **read-only** -- the connector opens ``file:...?mode=ro`` and sets
  ``PRAGMA query_only=ON``; nothing here can write to the source;
* **root confinement** -- a source is confined to the folder it came from, so a
  chosen path can never escape into another tree;
* **exclusion before read** -- an excluded column is never in the connection's
  projection, so its values never enter a record, a preview, or the store;
* **no secrets in messages** -- :func:`redact_secrets` strips a password, token,
  or credential-bearing URI from every message this module produces, so a status
  line or an error dialog can never leak one.

The pieces here are Tk-free, so the whole flow is testable without a display.
"""

from __future__ import annotations

import re
import time
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path

from core.knowledge_base import KnowledgeBaseStore, default_knowledge_base_dir
from ingest.connectors.sqlite import SqliteConnector
from ingest.errors import ConnectorError
from ingest.records import SourceSpec
from ingest.sql_job import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_BATCH_RETRIES,
    MAX_BATCH_SIZE,
    SCHEMA_FAIL,
    SCHEMA_FULL_REIMPORT,
    Interrupted,
    SqlImportJob,
    SqlImportRunner,
    SqlJobStore,
    sql_jobs_dir,
)

#: The database engines this front end enables. Only SQLite, for now -- the
#: connector is standard-library-only, so the base install stays dependency-free.
DATABASE_TYPES: tuple[str, ...] = ("SQLite",)

#: File suffixes the SQLite picker offers.
SUPPORTED_SUFFIXES: tuple[str, ...] = (".db", ".sqlite", ".sqlite3")

#: Rows shown in a preview by default, and the hard cap a caller may ask for.
DEFAULT_PREVIEW_ROWS = 5
MAX_PREVIEW_ROWS = 50

#: How many rows a resumable batch groups into one stored record, by default.
DEFAULT_ROW_GROUP = 1

#: Job statuses, re-exported so the view never imports the runner internals.
DONE = "done"
FAILED = "failed"
CANCELLED = "cancelled"
RUNNING = "running"

ProgressCallback = Callable[["SqlProgress"], None]


def picker_filetypes() -> tuple[tuple[str, str], ...]:
    """Tk ``filetypes`` tuples for the SQLite file picker."""
    databases = " ".join(f"*{suffix}" for suffix in SUPPORTED_SUFFIXES)
    return (
        ("SQLite databases", databases),
        ("All files", "*.*"),
    )


# ---------------------------------------------------------------------------
# secret redaction
# ---------------------------------------------------------------------------

#: ``key=value`` / ``key: value`` credential pairs, case-insensitive.
_SECRET_PAIR_RE = re.compile(
    r"(?i)\b(password|passwd|pwd|token|secret|api[_-]?key|access[_-]?key|"
    r"private[_-]?key|authorization|bearer)\b\s*[:=]\s*\S+"
)
#: Credentials inside a URI, with any slash count and an optional scheme:
#: ``scheme://user:pass@host`` and the malformed ``scheme:/user:pass@host`` both
#: reduce to ``...***@host``. Deliberately general, so a connector that renders a
#: path oddly cannot leak a user:password pair.
_URI_CREDENTIAL_RE = re.compile(r"[^/\s:@]+:[^/\s@]+@")


def redact_secrets(message: str) -> str:
    """Remove a password, token, or credential-bearing URI from ``message``.

    This front end never builds a credential into a message itself, but a
    connector error or a user-typed path could carry one, and a status line or an
    error dialog must be safe to show, log, or paste. Two shapes are handled: a
    ``key=value`` pair (``password=hunter2`` -> ``password=***``) and a credential
    inside a URI or path (``sqlite://user:pw@host`` -> ``sqlite://***@host``).
    """
    text = _SECRET_PAIR_RE.sub(lambda match: f"{match.group(1)}=***", message)
    return _URI_CREDENTIAL_RE.sub("***@", text)


# ---------------------------------------------------------------------------
# value objects
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SqlColumn:
    """One column, as SQLite's metadata declares it."""

    name: str
    declared_type: str = ""
    primary_key: bool = False
    nullable: bool = True
    hidden: bool = False


@dataclass(frozen=True)
class TableInfo:
    """One table's structure and honest identity/resume limits (no row values)."""

    name: str
    columns: tuple[SqlColumn, ...] = ()
    primary_key: tuple[str, ...] = ()
    row_count: int = 0
    stable_identity: bool = False
    resume_supported: bool = True
    without_rowid: bool = False
    reason: str = ""

    @property
    def column_names(self) -> tuple[str, ...]:
        return tuple(column.name for column in self.columns)

    @property
    def label(self) -> str:
        key = "PK " + ", ".join(self.primary_key) if self.primary_key else "no PK"
        return f"{self.name}  ({len(self.columns)} col, {self.row_count} rows, {key})"


@dataclass(frozen=True)
class ConnectionCheck:
    """The result of a read-only connection test (never raises for a bad file)."""

    ok: bool
    message: str
    tables: tuple[str, ...] = ()


@dataclass(frozen=True)
class SqlPreview:
    """A bounded preview of how rows will be rendered, using the connector's rules."""

    table: str
    columns: tuple[str, ...] = ()
    rows: tuple[str, ...] = ()
    truncated: bool = False
    message: str = ""


@dataclass(frozen=True)
class SqlProgress:
    """A live progress snapshot for the status area (percent only when measurable)."""

    committed: int = 0
    total: int = 0
    table: str = ""
    message: str = ""

    @property
    def percent(self) -> int | None:
        if self.total <= 0:
            return None
        return min(100, int(self.committed * 100 / self.total))


@dataclass(frozen=True)
class TableReport:
    """One table's durable result within a job."""

    name: str
    status: str
    committed: int = 0
    target: int = 0
    stable_identity: bool = False
    resume_supported: bool = True

    @property
    def ok(self) -> bool:
        return self.status == DONE

    @property
    def label(self) -> str:
        return f"{self.name}: {self.status} ({self.committed}/{self.target} rows)"


@dataclass(frozen=True)
class SqlImportReport:
    """The outcome of one SQL import run, ready for the results pane."""

    job_id: str
    status: str
    records: int = 0
    chunks: int = 0
    tables: tuple[TableReport, ...] = ()
    failures: tuple[str, ...] = ()
    cancelled: bool = False
    duration_seconds: float = 0.0
    message: str = ""

    @property
    def done(self) -> bool:
        return self.status == DONE

    @property
    def failed(self) -> bool:
        return self.status == FAILED

    @property
    def interrupted(self) -> bool:
        return self.status == "interrupted"

    @property
    def can_resume(self) -> bool:
        return self.status in (RUNNING, "interrupted", FAILED, CANCELLED)

    @property
    def ok(self) -> bool:
        return self.done

    def summary(self) -> str:
        if self.status == DONE:
            return (
                f"Done: {self.records} row(s), {self.chunks} chunk(s) stored "
                f"from {len(self.tables)} table(s)."
            )
        if self.status == CANCELLED:
            return f"Cancelled: {self.records} row(s) committed and kept; the rest was not read."
        if self.status == "interrupted":
            return f"Interrupted: {self.records} row(s) committed; resume to continue."
        if self.status == FAILED:
            first = self.failures[0] if self.failures else "the source failed to import"
            return f"Failed: {first}"
        return f"Running: {self.records} row(s) so far."

    def as_dict(self) -> dict[str, object]:
        return {
            "job_id": self.job_id,
            "status": self.status,
            "records": self.records,
            "chunks": self.chunks,
            "cancelled": self.cancelled,
            "duration_seconds": round(self.duration_seconds, 3),
            "tables": [
                {
                    "table": table.name,
                    "status": table.status,
                    "committed": table.committed,
                    "target": table.target,
                }
                for table in self.tables
            ],
            "failures": list(self.failures),
        }


@dataclass(frozen=True)
class SqlJobView:
    """A recorded SQL import job, for the jobs list (list/actions/resume)."""

    job_id: str
    status: str
    source: str
    source_path: str = ""
    batch_size: int = DEFAULT_BATCH_SIZE
    tables: tuple[TableReport, ...] = ()
    failures: tuple[str, ...] = ()
    created_at: str = ""
    updated_at: str = ""

    @property
    def committed(self) -> int:
        return sum(table.committed for table in self.tables)

    @property
    def target(self) -> int:
        return sum(table.target for table in self.tables)

    @property
    def is_running(self) -> bool:
        return self.status == RUNNING

    @property
    def can_resume(self) -> bool:
        return self.status in (RUNNING, "interrupted", FAILED, CANCELLED)

    def label(self) -> str:
        return (
            f"{self.job_id}  [{self.status}]  {self.committed}/{self.target} rows"
            f"  ·  {Path(self.source_path or self.source).name}"
        )


# ---------------------------------------------------------------------------
# service
# ---------------------------------------------------------------------------


class SqlImportService:
    """Drives the SQL ingestion service for the window (the CLI's engine).

    One instance is cheap: it resolves the knowledge store once and creates the
    connector and runner per call, so every operation reads and writes exactly
    where the CLI's ``ingest sql-import`` does.
    """

    def __init__(self, *, store_dir: str | Path | None = None) -> None:
        self.store_dir = (
            Path(store_dir) if store_dir is not None else default_knowledge_base_dir()
        )

    # -- helpers ----------------------------------------------------------

    @property
    def database_types(self) -> tuple[str, ...]:
        """The enabled database engines (SQLite only, at this stage)."""
        return DATABASE_TYPES

    def driver_summaries(self) -> tuple[dict[str, object], ...]:
        """The enabled SQL drivers with their schemes, suffixes, and access mode.

        One entry per :data:`DATABASE_TYPES` engine, describing the *same*
        connector the import uses, so ``list-drivers`` never advertises an engine
        the service cannot actually drive.
        """
        info = self._connector().info
        return (
            {
                "name": info.name,
                "schemes": sorted(info.schemes),
                "suffixes": sorted(info.suffixes),
                "description": info.description,
                "read_only": True,
                "optional_dependency": info.optional_dependency,
                "enabled": True,
            },
        )

    def default_job_id(self, path: str | Path) -> str:
        """A stable job id for ``path``, so re-importing resumes the same job."""
        resolved = str(Path(path).expanduser().resolve())
        from ingest.sql_job import job_fingerprint

        return f"sqlite-{Path(path).stem}-{job_fingerprint(resolved)}"

    def build_spec(
        self,
        path: str | Path,
        *,
        tables: Iterable[str] = (),
        exclude_tables: Iterable[str] = (),
        include_columns: Iterable[str] = (),
        exclude_columns: Iterable[str] = (),
        row_group: int | None = None,
        root: str | Path | None = None,
        timeout: float | None = None,
        extra_options: Mapping[str, str] | None = None,
    ) -> SourceSpec:
        """A :class:`SourceSpec` for ``path`` with the selection turned into options.

        The connector honours the selection *before it reads*, so an excluded
        column is never in the query's projection. ``root`` defaults to the
        database's own folder, confining the source to where the user chose it.
        ``extra_options`` lets a caller (the CLI) set connector options the view
        does not expose -- e.g. ``page`` or ``batch_retries`` -- without that
        caller reimplementing the spec.
        """
        database = Path(path).expanduser()
        uri = str(path) if "://" in str(path) else f"sqlite://{database}"
        options: dict[str, str] = {"source_path": str(database)}
        if extra_options:
            options.update({str(key): str(value) for key, value in extra_options.items()})
        if root is None:
            options["root"] = str(database.parent)
        else:
            options["root"] = str(Path(root).expanduser())
        if tables:
            options["include_tables"] = ",".join(tables)
        if exclude_tables:
            options["exclude_tables"] = ",".join(exclude_tables)
        if include_columns:
            options["include_columns"] = ",".join(include_columns)
        if exclude_columns:
            options["exclude_columns"] = ",".join(exclude_columns)
        if row_group is not None:
            options["row_group"] = str(row_group)
        if timeout is not None:
            options["timeout"] = str(timeout)
        return SourceSpec(uri, options=options)

    def _connector(self) -> SqliteConnector:
        return SqliteConnector()

    def _jobs(self) -> SqlJobStore:
        return SqlJobStore(sql_jobs_dir(self.store_dir))

    # -- connection test and schema --------------------------------------

    def test_connection(self, path: str | Path, *, root: str | Path | None = None) -> ConnectionCheck:
        """Open ``path`` read-only and report whether it is readable.

        Never raises for a bad path, a missing file, a directory, or a file that
        is not a database: the connector's own classified error becomes a
        user-facing message with any credential redacted.
        """
        if not str(path).strip():
            return ConnectionCheck(ok=False, message="Choose a database file first.")
        try:
            infos = self.inspect(path, root=root)
        except (ConnectorError, OSError, ValueError) as error:
            return ConnectionCheck(ok=False, message=redact_secrets(str(error)))
        noun = "table" if len(infos) == 1 else "tables"
        return ConnectionCheck(
            ok=True,
            message=f"Read-only connection OK — {len(infos)} {noun}.",
            tables=tuple(info.name for info in infos),
        )

    def inspect(self, path: str | Path, *, root: str | Path | None = None) -> tuple[TableInfo, ...]:
        """Every table's columns, primary key, row count, and identity limits.

        Reads SQLite metadata only -- no row values -- so it is cheap on a large
        table and safe to call as the user changes the selection.
        """
        spec = self.build_spec(path, root=root)
        connector = self._connector()
        schemas = {schema.name: schema for schema in connector.inspect(spec)}
        plans = {plan.table: plan for plan in connector.describe(spec)}
        infos: list[TableInfo] = []
        for name in sorted(schemas):
            schema = schemas[name]
            plan = plans.get(name)
            infos.append(
                TableInfo(
                    name=name,
                    columns=tuple(
                        SqlColumn(
                            name=column.name,
                            declared_type=column.declared_type,
                            primary_key=column.primary_key > 0,
                            nullable=column.nullable,
                            hidden=bool(column.hidden),
                        )
                        for column in schema.columns
                    ),
                    primary_key=schema.primary_key,
                    row_count=plan.total_rows if plan is not None else 0,
                    stable_identity=plan.stable_identity if plan is not None else schema.has_primary_key,
                    resume_supported=plan.resume_supported if plan is not None else True,
                    without_rowid=schema.without_rowid,
                    reason=plan.reason if plan is not None else "",
                )
            )
        return tuple(infos)

    def preview(
        self,
        path: str | Path,
        *,
        table: str,
        columns: Iterable[str] = (),
        limit: int = DEFAULT_PREVIEW_ROWS,
        root: str | Path | None = None,
    ) -> SqlPreview:
        """A limited preview, rendered by the connector exactly as rows are stored.

        At most ``limit`` rows are read (bounded by :data:`MAX_PREVIEW_ROWS`); the
        excluded columns are never read, so a preview cannot show a hidden value.
        """
        if not table:
            return SqlPreview(table="", message="Choose a table first.")
        bounded = max(1, min(int(limit), MAX_PREVIEW_ROWS))
        spec = self.build_spec(
            path, tables=(table,), include_columns=tuple(columns), row_group=1, root=root
        )
        spec = SourceSpec(
            spec.uri, options={**spec.options, "max_rows": str(bounded)}
        )
        try:
            reader = self._connector().stream(spec)
            rows: list[str] = []
            names: tuple[str, ...] = ()
            for record in reader:
                if not names:
                    names = tuple(str(name) for name in record.metadata.get("columns", ()))
                rows.append(record.text)
                if len(rows) >= bounded:
                    break
        except (ConnectorError, OSError, ValueError) as error:
            return SqlPreview(table=table, columns=tuple(columns), message=redact_secrets(str(error)))
        if not rows:
            return SqlPreview(
                table=table,
                columns=names or tuple(columns),
                message="No rows to preview.",
            )
        return SqlPreview(
            table=table,
            columns=names or tuple(columns),
            rows=tuple(rows),
            truncated=len(rows) >= bounded,
        )

    # -- running ----------------------------------------------------------

    def run_import(
        self,
        path: str | Path,
        *,
        job_id: str,
        tables: Iterable[str] = (),
        exclude_tables: Iterable[str] = (),
        include_columns: Iterable[str] = (),
        exclude_columns: Iterable[str] = (),
        batch_size: int = DEFAULT_BATCH_SIZE,
        schema_change: str = SCHEMA_FAIL,
        row_group: int | None = None,
        root: str | Path | None = None,
        timeout: float | None = None,
        extra_options: Mapping[str, str] | None = None,
        progress: ProgressCallback | None = None,
        should_cancel: Callable[[], bool] | None = None,
    ) -> SqlImportReport:
        """Import the reviewed selection through the shared resumable runner.

        The user is expected to have reviewed the source and the selection before
        this is called; nothing runs implicitly.
        """
        if not str(job_id).strip():
            raise ValueError("A job id is required to import a database.")
        if not 1 <= int(batch_size) <= MAX_BATCH_SIZE:
            raise ValueError(f"batch size must be in [1, {MAX_BATCH_SIZE}].")
        spec = self.build_spec(
            path,
            tables=tables,
            exclude_tables=exclude_tables,
            include_columns=include_columns,
            exclude_columns=exclude_columns,
            row_group=row_group,
            root=root,
            timeout=timeout,
            extra_options=extra_options,
        )
        return self._execute(
            spec,
            job_id=job_id,
            batch_size=int(batch_size),
            schema_change=schema_change,
            progress=progress,
            should_cancel=should_cancel,
        )

    def resume_job(
        self,
        job_id: str,
        *,
        progress: ProgressCallback | None = None,
        should_cancel: Callable[[], bool] | None = None,
        schema_change: str | None = None,
    ) -> SqlImportReport:
        """Continue a recorded job from its durable checkpoint.

        The job's own recorded options and batch size are reused, so a resume
        reproduces what the earlier run selected rather than a fresh guess.
        """
        view = self.job(job_id)
        if view is None:
            raise ValueError(f"No such SQL import job: {job_id}")
        return self._execute(
            SourceSpec(view.source),
            job_id=job_id,
            batch_size=view.batch_size,
            schema_change=schema_change or SCHEMA_FAIL,
            progress=progress,
            should_cancel=should_cancel,
            resume=True,
        )

    def retry_job(
        self,
        job_id: str,
        *,
        progress: ProgressCallback | None = None,
        should_cancel: Callable[[], bool] | None = None,
    ) -> SqlImportReport:
        """Retry a failed job. A retry is a resume of the same durable job."""
        return self.resume_job(job_id, progress=progress, should_cancel=should_cancel)

    def _execute(
        self,
        spec: SourceSpec,
        *,
        job_id: str,
        batch_size: int,
        schema_change: str,
        progress: ProgressCallback | None,
        should_cancel: Callable[[], bool] | None,
        resume: bool = False,
    ) -> SqlImportReport:
        store = KnowledgeBaseStore(self.store_dir)
        before = store.count()
        total = self._selection_total(spec)
        committed = 0
        current_table = ""

        def on_record(_record: object, _chunks: int) -> None:
            nonlocal committed
            committed += 1
            if progress is not None:
                progress(
                    SqlProgress(
                        committed=committed,
                        total=total,
                        table=current_table,
                        message=f"committed {committed} row(s)",
                    )
                )

        def on_progress(message: str) -> None:
            nonlocal current_table
            current_table = _current_table(message, current_table)
            if progress is not None:
                progress(
                    SqlProgress(
                        committed=committed,
                        total=total,
                        table=current_table,
                        message=message,
                    )
                )

        runner = SqlImportRunner(
            self.store_dir,
            batch_size=int(batch_size),
            progress=on_progress,
            should_cancel=should_cancel,
            on_record=on_record,
        )
        started = time.monotonic()
        if resume:
            job = runner.resume(spec, job_id, schema_change=schema_change)
        else:
            job = runner.run(spec, job_id=job_id, schema_change=schema_change)
        duration = time.monotonic() - started
        return self._to_report(job, before=before, after=store.count(), duration=duration)

    def _selection_total(self, spec: SourceSpec) -> int:
        """The selected tables' current row count (0 when it cannot be measured)."""
        try:
            return sum(plan.total_rows for plan in self._connector().describe(spec))
        except (ConnectorError, OSError, ValueError):
            return 0

    def _to_report(self, job: SqlImportJob, *, before: int, after: int, duration: float) -> SqlImportReport:
        tables = tuple(
            TableReport(
                name=progress.table,
                status=progress.status,
                committed=progress.resume_count,
                target=progress.target_rows,
                stable_identity=progress.stable_identity,
                resume_supported=progress.resume_supported,
            )
            for progress in sorted(job.tables.values(), key=lambda value: value.table)
        )
        return SqlImportReport(
            job_id=job.job_id,
            status=job.status,
            records=sum(table.committed for table in tables),
            chunks=max(0, after - before),
            tables=tables,
            failures=tuple(redact_secrets(failure) for failure in job.failures),
            cancelled=job.status == CANCELLED,
            duration_seconds=duration,
        )

    # -- job bookkeeping --------------------------------------------------

    def job(self, job_id: str) -> SqlJobView | None:
        """One recorded job as a view, or ``None``."""
        raw = self._jobs().load(job_id)
        return self._to_view(raw) if raw is not None else None

    def list_jobs(self) -> tuple[SqlJobView, ...]:
        """Every recorded SQL import job, newest first."""
        views = [self._to_view(job) for job in self._jobs().list()]
        views.sort(key=lambda view: view.updated_at, reverse=True)
        return tuple(views)

    def forget_job(self, job_id: str) -> bool:
        """Remove a job's durable state file (after the user is done with it)."""
        return self._jobs().remove(job_id)

    def raw_job(self, job_id: str) -> SqlImportJob | None:
        """The durable job record itself, for callers that need its full state.

        The CLI's ``sql-import --json`` emits this (the same shape it always
        has); the window uses :meth:`job` for a display-ready view.
        """
        return self._jobs().load(job_id)

    def _to_view(self, job: SqlImportJob) -> SqlJobView:
        tables = tuple(
            TableReport(
                name=progress.table,
                status=progress.status,
                committed=progress.resume_count,
                target=progress.target_rows,
                stable_identity=progress.stable_identity,
                resume_supported=progress.resume_supported,
            )
            for progress in sorted(job.tables.values(), key=lambda value: value.table)
        )
        return SqlJobView(
            job_id=job.job_id,
            status=job.status,
            source=job.source_uri,
            source_path=job.source_path,
            batch_size=job.batch_size,
            tables=tables,
            failures=tuple(redact_secrets(failure) for failure in job.failures),
            created_at=job.created_at,
            updated_at=job.updated_at,
        )

    def stats(self) -> dict[str, object]:
        """A small, honest description of the SQL job bookkeeping."""
        jobs = self.list_jobs()
        return {
            "store": str(self.store_dir),
            "jobs": len(jobs),
            "running": sum(1 for job in jobs if job.is_running),
        }

    # -- cross-process cancel markers ------------------------------------

    def cancel_marker(self, job_id: str) -> Path:
        """The file whose presence asks a running import to stop.

        A cancel must work across *processes* -- the CLI starts an import and a
        later ``cancel`` is a different invocation -- so a simple marker file
        under the job directory is the honest mechanism: whoever runs the import
        polls it with :meth:`should_cancel`, and the runner's checkpoint keeps
        every row committed before the stop. The GUI's own in-thread cancel does
        not need it, but the file approach works for both.
        """
        return self._jobs().root / f"{_safe_job_name(job_id)}.cancel"

    def request_cancel(self, job_id: str) -> bool:
        """Ask the running import ``job_id`` to stop; ``False`` if it is not known."""
        if self._jobs().load(job_id) is None:
            return False
        marker = self.cancel_marker(job_id)
        marker.parent.mkdir(parents=True, exist_ok=True)
        marker.write_text("cancel requested\n", encoding="utf-8")
        return True

    def should_cancel(self, job_id: str) -> bool:
        """A predicate for :meth:`run_import` / :meth:`resume_job` that polls the marker."""
        marker = self.cancel_marker(job_id)
        return marker.exists()

    def clear_cancel(self, job_id: str) -> bool:
        """Remove a cancel marker once a run has honoured it."""
        marker = self.cancel_marker(job_id)
        if marker.exists():
            marker.unlink()
            return True
        return False


def _safe_job_name(job_id: str) -> str:
    """A filesystem-safe mirror of the job store's own name rule."""
    return "".join(char if char.isalnum() or char in "-_." else "_" for char in job_id)


def _current_table(message: str, current: str) -> str:
    """The table a progress message is about, keeping the last one seen."""
    if message.startswith("table "):
        rest = message[len("table ") :]
        name = rest.split(":", 1)[0].strip()
        if name:
            return name
    return current


__all__ = [
    "CANCELLED",
    "DATABASE_TYPES",
    "DEFAULT_BATCH_RETRIES",
    "DEFAULT_BATCH_SIZE",
    "DEFAULT_PREVIEW_ROWS",
    "DONE",
    "FAILED",
    "Interrupted",
    "MAX_BATCH_SIZE",
    "MAX_PREVIEW_ROWS",
    "RUNNING",
    "SCHEMA_FAIL",
    "SCHEMA_FULL_REIMPORT",
    "SUPPORTED_SUFFIXES",
    "ConnectionCheck",
    "SqlColumn",
    "SqlImportReport",
    "SqlImportService",
    "SqlJobView",
    "SqlPreview",
    "SqlProgress",
    "TableInfo",
    "TableReport",
    "picker_filetypes",
    "redact_secrets",
]
