"""CLI for the ingestion subsystem: add sources, inspect connectors, search.

    python -m cli ingest add ./papers --store-dir data/knowledge_base
    python -m cli ingest add data.csv
    python -m cli ingest add report.pdf
    python -m cli ingest add locked.pdf --pdf-password secret
    python -m cli ingest add scanned.pdf --ocr --ocr-languages ara+eng
    python -m cli ingest add notes.docx
    python -m cli ingest add sqlite:///data/wells.db
    python -m cli ingest add https://www.spe.org/... --allow-web --approve
    python -m cli ingest list-connectors
    python -m cli ingest stats
    python -m cli ingest search "hydraulic fracturing"

SQLite import (the headless twin of the window's SQL Import screen):

    python -m cli ingest sql-drivers
    python -m cli ingest sql-inspect ./data/wells.db
    python -m cli ingest sql-preview ./data/wells.db --table wells --columns id,name --limit 5
    python -m cli ingest sql-import ./data/wells.db --job wells --batch-size 500 \\
        --tables wells,logs --exclude-columns api_key
    python -m cli ingest sql-status wells
    python -m cli ingest sql-resume wells
    python -m cli ingest sql-cancel wells

Every SQL command drives the *same* service the window uses
(:class:`gui.sql_import.SqlImportService` over :mod:`ingest.sql_job` and
:mod:`ingest.connectors.sqlite`); none of them reimplement import logic. The
database is opened **read-only** (`mode=ro` + ``PRAGMA query_only=ON``): a missing
or unreadable file fails safely and is never created or written. Selections use
``--tables``/``--exclude-tables``/``--columns``/``--exclude-columns`` (exclusion
wins, and an excluded column is never read); ``--batch-size``/``--row-group``/
``--page``/``--timeout`` are validated before the source is touched. A run never
blocks on a prompt, reports progress to stderr, and prints final counts to stdout.

Exit codes (SQL commands):

* ``0`` success;
* ``2`` invalid configuration (bad batch size / limit / selection / job id);
* ``3`` the database could not be opened read-only;
* ``4`` the import is incomplete (cancelled or interrupted; resumable);
* ``5`` an unrecoverable failure (a source error, a schema change, an IO fault).

Local sources run with no network and no extra dependency. A *digital* PDF needs
the optional ``[pdf]`` extra (``pip install fayd-al-ruh[pdf]``); ``--pdf-password``
unlocks an encrypted one. OCR for scanned PDF pages (Stage 3) is opt-in with
``--ocr`` and needs the separate ``[ocr]`` extra plus the Tesseract engine and its
language data (``apt install tesseract-ocr tesseract-ocr-eng tesseract-ocr-ara``);
without it, a scanned PDF is reported, not mis-read. A Word ``.docx``/``.docm``
(Stage 2B) is read with the
standard library only, treating it as an untrusted ZIP package with archive
resource limits. A web source is the one exception: it needs
``--allow-web`` to even register the connector, and ``--approve`` to pass the
permission and human-approval gates, so importing from the internet is a
deliberate, audited act rather than a default.

Ingestion writes only to the knowledge store (``data/knowledge_base`` by
default). It never touches training artifacts or the evaluation baseline, so a
large import cannot change a release by accident.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence

from core.knowledge_base import (
    DEFAULT_CHUNK_CHARS,
    DEFAULT_OVERLAP,
    KnowledgeBaseStore,
    default_knowledge_base_dir,
    iter_local_documents,
)
from core.knowledge_index import KnowledgeTextIndex, build_index
from core.permissions import Capability, PermissionPolicy
from core.rag import BulkCorpusRetriever
from gui.sql_import import SqlImportService, redact_secrets
from ingest.dedup import DedupLedger
from ingest.errors import ConnectorError
from ingest.metadata import DocumentMetadataStore, metadata_path
from ingest.pipeline import IngestionPipeline, ledger_path
from ingest.records import SourceSpec
from ingest.registry import build_default_registry
from ingest.sql_job import SCHEMA_FAIL, SCHEMA_FULL_REIMPORT

_WEB_TOOL = "ingest_web"


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="ingest", description=__doc__)
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument(
        "--store-dir",
        default=argparse.SUPPRESS,
        help="Knowledge-base folder (default: <data>/knowledge_base).",
    )
    parser.add_argument(
        "--store-dir",
        default=None,
        help="Knowledge-base folder (default: <data>/knowledge_base).",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    add = sub.add_parser("add", parents=[common], help="Ingest one source (file, folder, SQL, or URL).")
    add.add_argument("source", help="File, directory, sqlite:// URI, or http(s) URL.")
    add.add_argument("--connector", default=None, help="Force a connector by name.")
    add.add_argument("--option", action="append", default=[], metavar="KEY=VALUE", help="Connector option (repeatable).")
    add.add_argument("--chunk-chars", type=int, default=DEFAULT_CHUNK_CHARS)
    add.add_argument("--overlap", type=int, default=DEFAULT_OVERLAP)
    add.add_argument("--no-incremental", action="store_true", help="Re-read every item, ignoring fingerprints.")
    add.add_argument("--retries", type=int, default=2, help="Per-item retries for a transient read failure.")
    add.add_argument("--no-normalize", action="store_true", help="Store text verbatim (skip normalization).")
    add.add_argument("--fold-arabic", action="store_true", help="Also unify Arabic letter forms (opt-in).")
    add.add_argument("--encoding", default=None, help="Pin the source encoding (else detected).")
    add.add_argument(
        "--pdf-password",
        default=None,
        help="Password for an encrypted PDF (encrypted PDFs are refused otherwise).",
    )
    add.add_argument(
        "--ocr",
        action="store_true",
        help=(
            "Force OCR for scanned PDF pages. Scanned pages are OCR'd automatically "
            "when the [ocr] extra + Tesseract are installed; this forces it on."
        ),
    )
    add.add_argument(
        "--no-ocr",
        action="store_true",
        help="Never OCR; read only the PDF's text layer.",
    )
    add.add_argument(
        "--ocr-languages",
        default=None,
        metavar="ARA+ENG",
        help="Tesseract languages for OCR (default: eng).",
    )
    add.add_argument(
        "--ocr-dpi",
        type=int,
        default=None,
        help="Rasterisation DPI for OCR (default: 200).",
    )
    add.add_argument(
        "--ocr-max-pages",
        type=int,
        default=None,
        help="Cap OCR'd pages per PDF (0 = no cap).",
    )
    add.add_argument(
        "--ocr-time-budget",
        type=float,
        default=None,
        help="Seconds before OCR stops for a PDF (0 = no budget).",
    )
    add.add_argument("--allow-web", action="store_true", help="Register the internet connector.")
    add.add_argument("--approve", action="store_true", help="Grant the external capability for a web source.")
    add.add_argument("--quiet", action="store_true", help="Suppress progress.")
    add.add_argument("--json", action="store_true", dest="as_json", help="Print the report as JSON.")
    add.add_argument("--job", default=None, help="Resumable job id (saved under the store).")

    connectors = sub.add_parser("list-connectors", parents=[common], help="List the connectors and their formats.")
    connectors.add_argument("--allow-web", action="store_true")
    connectors.add_argument("--json", action="store_true", dest="as_json")

    stats = sub.add_parser("stats", parents=[common], help="Show the store and ingestion ledger.")
    stats.add_argument("--json", action="store_true", dest="as_json")

    sub.add_parser("build-index", parents=[common], help="Build the fast full-text index over the store.")

    search = sub.add_parser("search", parents=[common], help="Search the ingested store (offline).")
    search.add_argument("query")
    search.add_argument("--limit", type=int, default=5)

    delete = sub.add_parser("delete", parents=[common], help="Remove an ingested source by key.")
    delete.add_argument("source_key", help="The source key recorded in the ledger.")
    delete.add_argument("--json", action="store_true", dest="as_json")

    sub.add_parser("jobs", parents=[common], help="List resumable ingestion jobs.")

    _add_sql_parsers(sub, common)
    return parser


def _add_sql_parsers(
    sub: argparse._SubParsersAction, common: argparse.ArgumentParser
) -> None:
    """Register the SQLite import commands (the headless twin of the GUI screen).

    Every command drives the *same* :class:`gui.sql_import.SqlImportService` the
    window's SQL Import screen uses, which in turn drives the one resumable
    ingestion service (:mod:`ingest.sql_job` + :mod:`ingest.connectors.sqlite`).
    None of them reimplement import logic.
    """
    select = argparse.ArgumentParser(add_help=False)
    select.add_argument(
        "--tables",
        default="",
        metavar="A,B",
        help="Comma-separated tables to import (default: every table).",
    )
    select.add_argument(
        "--exclude-tables",
        default="",
        metavar="A,B",
        help="Comma-separated tables to skip.",
    )
    select.add_argument(
        "--columns",
        default="",
        metavar="A,B",
        help="Comma-separated columns to read (default: every column).",
    )
    select.add_argument(
        "--exclude-columns",
        default="",
        metavar="A,B",
        help="Comma-separated columns to never read (wins over --columns).",
    )
    batch = argparse.ArgumentParser(add_help=False)
    batch.add_argument("--batch-size", type=int, default=500, help="Rows per committed batch.")
    batch.add_argument("--row-group", type=int, default=None, help="Rows grouped into one stored record.")
    batch.add_argument("--page", type=int, default=None, help="Connector page size (rows per read).")
    batch.add_argument("--timeout", type=float, default=None, help="Seconds to wait for a locked database.")
    batch.add_argument("--quiet", action="store_true", help="Suppress progress.")
    batch.add_argument("--json", action="store_true", dest="as_json")

    drivers = sub.add_parser(
        "sql-drivers", parents=[common], help="List the supported SQL drivers."
    )
    drivers.add_argument("--json", action="store_true", dest="as_json")

    inspect = sub.add_parser(
        "sql-inspect", parents=[common], help="Show a database's schema without reading rows."
    )
    inspect.add_argument("source", help="A sqlite:// URI (or a .db/.sqlite path).")
    inspect.add_argument("--json", action="store_true", dest="as_json")

    preview = sub.add_parser(
        "sql-preview", parents=[common], help="Preview how rows of one table will be stored."
    )
    preview.add_argument("source", help="A sqlite:// URI (or a .db/.sqlite path).")
    preview.add_argument("--table", required=True, help="The table to preview.")
    preview.add_argument("--columns", default="", metavar="A,B", help="Columns to preview.")
    preview.add_argument("--limit", type=int, default=5, help="Rows to preview (max 50).")
    preview.add_argument("--json", action="store_true", dest="as_json")

    sql_import = sub.add_parser(
        "sql-import",
        parents=[common, select, batch],
        help="Resumable, idempotent SQL import with durable progress.",
    )
    sql_import.add_argument("source", help="A sqlite:// URI (or a .db/.sqlite path).")
    sql_import.add_argument(
        "--job",
        default=None,
        help="Job id (reused to resume; derived from the path when omitted).",
    )
    sql_import.add_argument(
        "--schema-change",
        choices=("fail", "full-reimport"),
        default="fail",
        help="How to react when a table's structure changed (default: fail).",
    )
    sql_import.add_argument("--option", action="append", default=[], metavar="KEY=VALUE")

    status = sub.add_parser(
        "sql-status", parents=[common], help="Show one SQL import job's status (or all)."
    )
    status.add_argument("job", nargs="?", default=None, help="Job id (omit to list all).")
    status.add_argument("--json", action="store_true", dest="as_json")

    resume = sub.add_parser(
        "sql-resume", parents=[common, batch], help="Resume an interrupted SQL import job."
    )
    resume.add_argument("job", help="The durable job id to continue.")
    resume.add_argument(
        "--schema-change",
        choices=("fail", "full-reimport"),
        default="fail",
        help="Recovery policy if a table's structure changed (default: fail).",
    )

    cancel = sub.add_parser(
        "sql-cancel", parents=[common], help="Ask a running SQL import to stop safely."
    )
    cancel.add_argument("job", help="The durable job id to cancel.")
    cancel.add_argument("--json", action="store_true", dest="as_json")

    sql_jobs = sub.add_parser(
        "sql-jobs", parents=[common], help="List resumable SQL import jobs."
    )
    sql_jobs.add_argument("--json", action="store_true", dest="as_json")


def _resolve_dir(value: str | None):
    from pathlib import Path

    return Path(value) if value else default_knowledge_base_dir()


def _parse_options(pairs: Sequence[str]) -> dict[str, str]:
    options: dict[str, str] = {}
    for pair in pairs:
        if "=" not in pair:
            raise SystemExit(f"Option {pair!r} must be KEY=VALUE.")
        key, value = pair.split("=", 1)
        options[key.strip()] = value
    return options


def run_add(args: argparse.Namespace) -> int:
    store_dir = _resolve_dir(args.store_dir)
    options = _parse_options(args.option)
    # Flags map onto connector options so the file connector can honour them.
    options.setdefault("normalize", "false" if args.no_normalize else "true")
    options.setdefault("fold_arabic", "true" if args.fold_arabic else "false")
    if args.encoding:
        options.setdefault("encoding", args.encoding)
    if args.pdf_password:
        options.setdefault("pdf_password", args.pdf_password)
    if args.ocr:
        options.setdefault("allow_ocr", "true")
    if args.no_ocr:
        options.setdefault("allow_ocr", "false")
    if args.ocr_languages:
        options.setdefault("ocr_languages", args.ocr_languages)
    if args.ocr_dpi is not None:
        options.setdefault("ocr_dpi", str(args.ocr_dpi))
    if args.ocr_max_pages is not None:
        options.setdefault("ocr_max_pages", str(args.ocr_max_pages))
    if args.ocr_time_budget is not None:
        options.setdefault("ocr_time_budget", str(args.ocr_time_budget))
    spec = SourceSpec(args.source, options=options)
    registry = build_default_registry(allow_web=args.allow_web)
    try:
        connector = registry.get(args.connector) if args.connector else registry.resolve(spec)
    except Exception as error:  # noqa: BLE001 - a resolver error is a user-facing message
        print(f"error: {error}")
        return 1
    if connector is None:
        print(f"error: no connector named {args.connector!r}.")
        return 1

    if connector.info.remote and not _authorise_web(args):
        return 1

    store = KnowledgeBaseStore(store_dir)
    ledger = DedupLedger(ledger_path(store.root))
    metadata = DocumentMetadataStore(metadata_path(store.root))
    progress = (lambda message: print(message, flush=True)) if not args.quiet else None
    job = None
    jobs = None
    if args.job:
        from ingest.jobs import JobState, JobStore
        from ingest.pipeline import ingest_jobs_dir

        jobs = JobStore(ingest_jobs_dir(store.root))
        job = jobs.load(args.job) or JobState(job_id=args.job, source=args.source)
        job.connector = connector.info.name
    pipeline = IngestionPipeline(
        store,
        ledger,
        progress=progress,
        job=job,
        jobs=jobs,
        retries=args.retries,
        metadata=metadata,
    )
    report = pipeline.ingest(
        spec, connector=connector, incremental=not args.no_incremental
    )
    _print_report(report, pipeline, args.as_json)
    return 0 if report.failed == 0 else 1


def _authorise_web(args: argparse.Namespace) -> bool:
    """Run a web import only through the permission and approval gates."""
    policy = PermissionPolicy(
        name="ingest-cli",
        allowed=frozenset(
            {Capability.SYMBOLIC_SIMULATION, Capability.SANDBOX_TASK, Capability.EXTERNAL_TOOL}
        ),
        allowed_external_tools=frozenset({_WEB_TOOL}),
    )
    decision = policy.decide(Capability.EXTERNAL_TOOL, _WEB_TOOL)
    if not decision.allowed or not args.approve:
        print(
            "error: importing from the internet needs --approve "
            "(and the host on allowed_hosts or the source authorized)."
        )
        return False
    return True


def _print_report(report, pipeline, as_json: bool) -> None:
    if as_json:
        payload = report.as_dict()
        payload["errors"] = pipeline.errors.as_dicts()
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return
    data = report.as_dict()
    print(
        f"records {data['records']}, chunks {data['chunks']}, "
        f"{data['bytes_written'] / 1024:.1f} KB written, "
        f"skipped {data['skipped_unchanged']} unchanged"
    )
    if pipeline.errors:
        print(f"{len(pipeline.errors)} source(s) failed:")
        for error in pipeline.errors.as_dicts():
            print(f"  {error['source']}: {error['kind']}: {error['message']}")


def run_list_connectors(args: argparse.Namespace) -> int:
    registry = build_default_registry(allow_web=args.allow_web)
    infos = registry.infos()
    if args.as_json:
        payload = [
            {
                "name": info.name,
                "description": info.description,
                "suffixes": sorted(info.suffixes),
                "schemes": sorted(info.schemes),
                "remote": info.remote,
                "optional_dependency": info.optional_dependency,
            }
            for info in infos
        ]
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return 0
    for info in infos:
        formats = ", ".join(sorted(info.suffixes) or sorted(info.schemes)) or "any path"
        flag = " (remote, denied by default)" if info.remote else ""
        print(f"{info.name:<8} {formats}{flag}")
        print(f"         {info.description}")
    return 0


def run_stats(args: argparse.Namespace) -> int:
    store_dir = _resolve_dir(args.store_dir)
    store = KnowledgeBaseStore(store_dir)
    ledger_stats = DedupLedger(ledger_path(store.root)).stats()
    store_stats = store.stats()
    index_stats = KnowledgeTextIndex(store_dir).stats()
    if args.as_json:
        print(
            json.dumps(
                {"store": store_stats, "ledger": ledger_stats, "index": index_stats},
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0
    print(f"Knowledge base : {store_stats['path']}")
    print(f"Chunks         : {store_stats['chunks']}")
    print(f"Size           : {store_stats['size_mb']:.2f} MB")
    print(f"Ingested items : {ledger_stats['sources']} ({ledger_stats['chunks']} chunks tracked)")
    if index_stats["present"]:
        print(f"Index          : {index_stats['size_mb']:.2f} MB ({index_stats['chunks']} entries)")
    else:
        print("Index          : absent (run `python -m cli ingest build-index`)")
    return 0


def run_build_index(args: argparse.Namespace) -> int:
    store_dir = _resolve_dir(args.store_dir)
    if not KnowledgeBaseStore(store_dir).shard_paths():
        print("No store to index. Ingest a source first with `python -m cli ingest add`.")
        return 1
    print(f"Building text index in {store_dir}")
    summary = build_index(store_dir, progress=lambda message: print(message, flush=True))
    print(f"Done: indexed {summary['indexed']} chunks ({summary['size_mb']:.2f} MB)")
    return 0


def run_search(args: argparse.Namespace) -> int:
    store_dir = _resolve_dir(args.store_dir)
    if not KnowledgeBaseStore(store_dir).shard_paths():
        print("No store to search. Ingest a source first with `python -m cli ingest add`.")
        return 1
    index = KnowledgeTextIndex(store_dir)
    retriever = BulkCorpusRetriever(
        lambda: iter_local_documents(store_dir),
        index=index if index.exists() else None,
    )
    passages = retriever.retrieve(args.query, limit=args.limit)
    if not passages:
        print("No match.")
        return 0
    for passage in passages:
        print(f"[{passage.score:.3f}] {passage.title} ({passage.doc_id})")
        print(f"    {passage.text[:300]}")
    return 0


def run_delete(args: argparse.Namespace) -> int:
    from ingest.delete import delete_source

    store_dir = _resolve_dir(args.store_dir)
    report = delete_source(args.source_key, store_dir=store_dir)
    if args.as_json:
        print(json.dumps(report.as_dict(), ensure_ascii=False, indent=2))
        return 0
    if report.removed == 0:
        print(f"No chunks recorded for source key {args.source_key!r}.")
        return 1
    print(f"Removed {report.removed} chunk(s); {report.remaining} remain.")
    print("Rebuild the index with `python -m cli ingest build-index`.")
    return 0


def run_jobs(args: argparse.Namespace) -> int:
    from ingest.jobs import JobStore
    from ingest.pipeline import ingest_jobs_dir

    store_dir = _resolve_dir(args.store_dir)
    states = JobStore(ingest_jobs_dir(store_dir)).list()
    if not states:
        print("No ingestion jobs recorded.")
        return 0
    for state in states:
        print(
            f"{state.job_id} [{state.status}] pos {state.position} "
            f"chunks {state.chunks} — {state.source}"
        )
    return 0


SQL_EXIT_INVALID = 2
"""Bad configuration: a wrong batch size/limit, a bad selection, a bad job id."""

SQL_EXIT_CONNECTION = 3
"""The database could not be opened read-only (missing, locked, not a database)."""

SQL_EXIT_INCOMPLETE = 4
"""The import finished but is not complete (cancelled or interrupted; resumable)."""

SQL_EXIT_FAILED = 5
"""An unrecoverable failure (a source error, a schema change, an IO fault)."""


def _sql_service(args: argparse.Namespace) -> SqlImportService:
    """The shared SQL service, over the same store dir the window uses."""
    return SqlImportService(store_dir=_resolve_dir(args.store_dir))


def _split_names(value: str | None) -> tuple[str, ...]:
    """Split a comma-separated option into names, trimming spaces."""
    if not value:
        return ()
    return tuple(part.strip() for part in value.split(",") if part.strip())


def _progress_printer(quiet: bool):
    """A progress callback for the service (it emits :class:`SqlProgress` objects)."""
    if quiet:
        return None

    def emit(progress: object) -> None:
        message = getattr(progress, "message", None) or str(progress)
        print(redact_secrets(str(message)), file=sys.stderr, flush=True)

    return emit


def _validate_sql_args(args: argparse.Namespace) -> str | None:
    """Return a human message for an invalid argument, or ``None`` when valid.

    Validation happens *before* the service touches the database, so a bad batch
    size or limit never opens the source at all.
    """
    from ingest.sql_job import MAX_BATCH_SIZE

    batch_size = getattr(args, "batch_size", None)
    if batch_size is not None and not 1 <= batch_size <= MAX_BATCH_SIZE:
        return f"--batch-size must be between 1 and {MAX_BATCH_SIZE}."
    row_group = getattr(args, "row_group", None)
    if row_group is not None and row_group < 1:
        return "--row-group must be at least 1."
    page = getattr(args, "page", None)
    if page is not None and page < 1:
        return "--page must be at least 1."
    timeout = getattr(args, "timeout", None)
    if timeout is not None and timeout < 0:
        return "--timeout cannot be negative."
    limit = getattr(args, "limit", None)
    if limit is not None and limit < 1:
        return "--limit must be at least 1."
    return None


def _sql_extra_options(args: argparse.Namespace) -> dict[str, str]:
    """Connector options the view does not expose, forwarded through build_spec."""
    extras: dict[str, str] = {}
    if getattr(args, "page", None) is not None:
        extras["page"] = str(args.page)
    if getattr(args, "timeout", None) is not None:
        extras["timeout"] = str(args.timeout)
    extras.update(_parse_options(getattr(args, "option", []) or []))
    return extras


def _redact_payload(value: object) -> object:
    """Return ``value`` with every string run through :func:`redact_secrets`.

    A job's ``source_uri``/``source_path`` and its failure list can carry a
    credential-bearing path (or a URI), and the CLI prints them in both human and
    ``--json`` form. Redacting the whole payload (not one field) means a new field
    added later cannot leak a secret by omission.
    """
    if isinstance(value, str):
        return redact_secrets(value)
    if isinstance(value, dict):
        return {key: _redact_payload(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_redact_payload(item) for item in value]
    return value


def _print_sql_report(report, as_json: bool, raw_job=None) -> int:
    """Print an import report and map its outcome to an exit code.

    ``--json`` keeps the durable job shape the CLI always emitted (the full
    :class:`ingest.sql_job.SqlImportJob`) rather than the display report, so an
    existing consumer of ``sql-import --json`` keeps working. Every string is
    secret-redacted first, so a credential in a source path never prints.
    """
    if as_json:
        payload = raw_job.as_dict() if raw_job is not None else report.as_dict()
        print(json.dumps(_redact_payload(payload), ensure_ascii=False, indent=2))
    else:
        print(redact_secrets(report.summary()))
        for table in report.tables:
            print(f"  {table.label}")
        for failure in report.failures:
            print(f"  failed: {redact_secrets(failure)}")
        print(f"job {report.job_id}: {report.status}")
    if report.status == "done":
        return 0
    if report.cancelled or report.interrupted:
        return SQL_EXIT_INCOMPLETE
    return SQL_EXIT_FAILED


def run_sql_drivers(args: argparse.Namespace) -> int:
    rows = _sql_service(args).driver_summaries()
    if args.as_json:
        print(json.dumps(rows, ensure_ascii=False, indent=2))
        return 0
    for row in rows:
        suffixes = ", ".join(row["suffixes"])  # type: ignore[arg-type]
        schemes = ", ".join(row["schemes"])  # type: ignore[arg-type]
        print(f"{row['name']}  (read-only)")
        print(f"  schemes  : {schemes}")
        print(f"  suffixes : {suffixes}")
        print(f"  {row['description']}")
        dependency = row.get("optional_dependency") or "none"
        print(f"  extra    : {dependency}")
    return 0


def run_sql_inspect(args: argparse.Namespace) -> int:
    service = _sql_service(args)
    try:
        tables = service.inspect(args.source)
    except (ConnectorError, OSError, ValueError) as error:
        print(f"error: {redact_secrets(str(error))}")
        return SQL_EXIT_CONNECTION
    if args.as_json:
        payload = [
            {
                "table": info.name,
                "row_count": info.row_count,
                "primary_key": list(info.primary_key),
                "stable_identity": info.stable_identity,
                "resume_supported": info.resume_supported,
                "without_rowid": info.without_rowid,
                "reason": info.reason,
                "columns": [
                    {
                        "name": column.name,
                        "declared_type": column.declared_type,
                        "primary_key": column.primary_key,
                        "nullable": column.nullable,
                        "hidden": column.hidden,
                    }
                    for column in info.columns
                ],
            }
            for info in tables
        ]
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return 0
    if not tables:
        print("No tables to import (every table was excluded or the database is empty).")
        return 0
    for info in tables:
        print(info.label)
        for column in info.columns:
            pk = " PK" if column.primary_key else ""
            nullable = "" if column.nullable else " not null"
            hidden = " hidden" if column.hidden else ""
            print(f"    {column.name}: {column.declared_type or '(untyped)'}{pk}{nullable}{hidden}")
        if info.reason:
            print(f"    limit: {info.reason}")
    return 0


def run_sql_preview(args: argparse.Namespace) -> int:
    invalid = _validate_sql_args(args)
    if invalid:
        print(f"error: {invalid}")
        return SQL_EXIT_INVALID
    service = _sql_service(args)
    preview = service.preview(
        args.source,
        table=args.table,
        columns=_split_names(args.columns),
        limit=args.limit,
    )
    if args.as_json:
        print(
            json.dumps(
                _redact_payload(
                    {
                        "table": preview.table,
                        "columns": list(preview.columns),
                        "rows": list(preview.rows),
                        "truncated": preview.truncated,
                        "message": preview.message,
                    }
                ),
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0
    if preview.message and not preview.rows:
        print(redact_secrets(preview.message))
        return SQL_EXIT_CONNECTION if "No such" in preview.message else 0
    print(f"table {preview.table} — columns: {', '.join(preview.columns) or '(none)'}")
    for row in preview.rows:
        print(row)
    if preview.truncated:
        print(f"(showing at most {args.limit} row(s))")
    return 0


def run_sql_import(args: argparse.Namespace) -> int:
    invalid = _validate_sql_args(args)
    if invalid:
        print(f"error: {invalid}")
        return SQL_EXIT_INVALID
    service = _sql_service(args)
    job_id = args.job or service.default_job_id(args.source)
    try:
        report = service.run_import(
            args.source,
            job_id=job_id,
            tables=_split_names(args.tables),
            exclude_tables=_split_names(args.exclude_tables),
            include_columns=_split_names(args.columns),
            exclude_columns=_split_names(args.exclude_columns),
            batch_size=args.batch_size,
            schema_change=(
                SCHEMA_FULL_REIMPORT if args.schema_change == "full-reimport" else SCHEMA_FAIL
            ),
            row_group=args.row_group,
            extra_options=_sql_extra_options(args),
            progress=_progress_printer(args.quiet),
            should_cancel=lambda: service.should_cancel(job_id),
        )
    except ValueError as error:
        print(f"error: {error}")
        return SQL_EXIT_INVALID
    finally:
        service.clear_cancel(job_id)
    return _print_sql_report(report, args.as_json, raw_job=service.raw_job(job_id))


def run_sql_status(args: argparse.Namespace) -> int:
    service = _sql_service(args)
    if args.job:
        view = service.job(args.job)
        if view is None:
            print(f"error: no such SQL import job: {args.job}")
            return SQL_EXIT_INVALID
        if args.as_json:
            raw = service.raw_job(args.job)
            payload = raw.as_dict() if raw else {}
            print(json.dumps(_redact_payload(payload), ensure_ascii=False, indent=2))
            return 0
        print(f"{view.job_id}  [{view.status}]  {view.committed}/{view.target} rows")
        print(f"  source     : {redact_secrets(view.source_path or view.source)}")
        print(f"  batch size : {view.batch_size}")
        print(f"  updated    : {view.updated_at}")
        print(f"  resumable  : {'yes' if view.can_resume else 'no'}")
        for table in view.tables:
            print(f"    {table.label}")
        for failure in view.failures:
            print(f"    failed: {redact_secrets(failure)}")
        return 0
    jobs = service.list_jobs()
    if args.as_json:
        payloads = [_job_view_dict(view) for view in jobs]
        print(json.dumps(_redact_payload(payloads), ensure_ascii=False, indent=2))
        return 0
    if not jobs:
        print("No SQL import jobs recorded.")
        return 0
    for view in jobs:
        print(
            f"{view.job_id} [{view.status}] {view.committed}/{view.target} rows — "
            f"{redact_secrets(view.source)}"
        )
    return 0


def _job_view_dict(view) -> dict[str, object]:
    return {
        "job_id": view.job_id,
        "status": view.status,
        "source": view.source,
        "source_path": view.source_path,
        "batch_size": view.batch_size,
        "committed": view.committed,
        "target": view.target,
        "updated_at": view.updated_at,
        "resumable": view.can_resume,
        "tables": [
            {
                "table": table.name,
                "status": table.status,
                "committed": table.committed,
                "target": table.target,
            }
            for table in view.tables
        ],
        "failures": list(view.failures),
    }


def run_sql_resume(args: argparse.Namespace) -> int:
    invalid = _validate_sql_args(args)
    if invalid:
        print(f"error: {invalid}")
        return SQL_EXIT_INVALID
    service = _sql_service(args)
    try:
        report = service.resume_job(
            args.job,
            progress=_progress_printer(getattr(args, "quiet", False)),
            should_cancel=lambda: service.should_cancel(args.job),
            schema_change=(
                SCHEMA_FULL_REIMPORT if args.schema_change == "full-reimport" else SCHEMA_FAIL
            ),
        )
    except ValueError as error:
        print(f"error: {error}")
        return SQL_EXIT_INVALID
    finally:
        service.clear_cancel(args.job)
    return _print_sql_report(
        report, getattr(args, "as_json", False), raw_job=service.raw_job(args.job)
    )


def run_sql_cancel(args: argparse.Namespace) -> int:
    service = _sql_service(args)
    requested = service.request_cancel(args.job)
    if args.as_json:
        print(json.dumps({"job_id": args.job, "requested": requested}, indent=2))
        return 0 if requested else SQL_EXIT_INVALID
    if not requested:
        print(f"error: no such SQL import job: {args.job}")
        return SQL_EXIT_INVALID
    print(
        f"Cancel requested for {args.job}. A running import stops at the next "
        "checkpoint; committed rows are kept."
    )
    return 0


def run_sql_jobs(args: argparse.Namespace) -> int:
    service = _sql_service(args)
    jobs = service.list_jobs()
    if args.as_json:
        payloads = [_job_view_dict(view) for view in jobs]
        print(json.dumps(_redact_payload(payloads), ensure_ascii=False, indent=2))
        return 0
    if not jobs:
        print("No SQL import jobs recorded.")
        return 0
    for view in jobs:
        print(
            f"{view.job_id} [{view.status}] {view.committed}/{view.target} rows — "
            f"{redact_secrets(view.source)}"
        )
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(list(argv) if argv is not None else sys.argv[1:])
    if args.command == "add":
        return run_add(args)
    if args.command == "list-connectors":
        return run_list_connectors(args)
    if args.command == "stats":
        return run_stats(args)
    if args.command == "build-index":
        return run_build_index(args)
    if args.command == "search":
        return run_search(args)
    if args.command == "delete":
        return run_delete(args)
    if args.command == "jobs":
        return run_jobs(args)
    if args.command == "sql-drivers":
        return run_sql_drivers(args)
    if args.command == "sql-inspect":
        return run_sql_inspect(args)
    if args.command == "sql-preview":
        return run_sql_preview(args)
    if args.command == "sql-import":
        return run_sql_import(args)
    if args.command == "sql-status":
        return run_sql_status(args)
    if args.command == "sql-resume":
        return run_sql_resume(args)
    if args.command == "sql-cancel":
        return run_sql_cancel(args)
    if args.command == "sql-jobs":
        return run_sql_jobs(args)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
