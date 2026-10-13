"""The Knowledge Import Center: the user-facing ingestion front end.

This is the layer between the window's **＋ attach** control and the existing
:mod:`ingest` pipeline. It adds no new ingestion mechanism -- it *drives* the one
the project already has (connectors, dedup ledger, resumable jobs, per-document
metadata, bounded-batch streaming) so the window, the CLI, and the pipeline all
share the same behaviour instead of duplicating it.

Two import paths, matching the two the user asked for:

* **Upload files** -- one or many files chosen in the window's picker. The path
  the picker returns is a real local path the process can read (the window runs
  on the same machine), so the same file connector reads it; nothing is uploaded
  to a server.
* **Import from a folder** -- a folder on this machine, scanned recursively (or
  not) and imported in bounded batches, with incremental discovery of changed
  files and resumable progress.

Both paths share one flow: *verify -> extract -> dedup -> chunk+store -> record
-> retry*, and both report per-file success/failure so a failed file can be
retried and an interrupted job resumed.

Security and limits, deliberately:

* a source is confined to the folder it came from (``root``), so a path can never
  escape the user's chosen tree;
* nothing here executes a file or follows instructions inside it -- every reader
  only parses bytes into text;
* text is normalized conservatively (Unicode NFC + structure) and never loses
  meaningful Arabic or English content; Arabic letter forms are kept unless the
  caller explicitly asks to fold them;
* the engine streams (CSV/JSONL/TXT-by-line/PDF-by-page/DOCX-by-block), so the
  whole dataset is never read into memory by default.

The pieces here are Tk-free, so the flow is testable without a display.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path

from core.knowledge_base import KnowledgeBaseStore, default_knowledge_base_dir
from core.paths import data_dir
from core.pdf_ocr import PageOcrEngine
from ingest.dedup import DedupLedger
from ingest.jobs import DONE, FAILED, RUNNING, JobState, JobStore
from ingest.metadata import DocumentMetadataStore, metadata_path
from ingest.pipeline import IngestionPipeline, ingest_jobs_dir, ledger_path
from ingest.records import IngestionReport, ItemOutcome, SourceSpec
from ingest.registry import build_default_registry

#: Formats the file connector accepts. Kept here so the window can filter its
#: pickers and print an honest list without importing the connector internals.
SUPPORTED_SUFFIXES: tuple[str, ...] = (
    ".txt",
    ".md",
    ".markdown",
    ".csv",
    ".tsv",
    ".json",
    ".jsonl",
    ".ndjson",
    ".pdf",
    ".docx",
    ".docm",
)

#: Recognised but deliberately unsupported (binary/undocumented), refused with a
#: clear message rather than scrambled into nonsense.
UNSUPPORTED_SUFFIXES: tuple[str, ...] = (".doc", ".rtf")

ProgressCallback = Callable[[str], None]


def _suffixes_for_picker() -> tuple[tuple[str, str], ...]:
    """Tk ``filetypes`` tuples built from the supported list, no new dependency."""
    documents = " ".join(f"*{suffix}" for suffix in SUPPORTED_SUFFIXES)
    return (
        ("Knowledge documents", documents),
        ("Text", "*.txt *.md *.markdown"),
        ("Tabular", "*.csv *.tsv"),
        ("Data", "*.json *.jsonl *.ndjson"),
        ("PDF", "*.pdf"),
        ("Word", "*.docx *.docm"),
        ("All files", "*.*"),
    )


def picker_filetypes() -> tuple[tuple[str, str], ...]:
    """The ``filetypes`` a Tk file picker should offer for ingestion."""
    return _suffixes_for_picker()


@dataclass(frozen=True)
class ImportOutcome:
    """What happened to one imported source, for the per-file results list."""

    source: str
    ok: bool
    records: int = 0
    chunks: int = 0
    bytes_written: int = 0
    skipped_unchanged: int = 0
    message: str = ""

    @property
    def filename(self) -> str:
        return Path(self.source).name or self.source

    @property
    def status(self) -> str:
        if self.ok and self.skipped_unchanged and not self.chunks:
            return "unchanged"
        return "ok" if self.ok else "failed"


@dataclass
class ImportReport:
    """The aggregate result of one import run (files or a folder)."""

    outcomes: list[ImportOutcome] = field(default_factory=list)
    records: int = 0
    chunks: int = 0
    bytes_written: int = 0
    skipped_unchanged: int = 0
    duplicates: int = 0
    duration_seconds: float = 0.0
    job_id: str = ""

    @property
    def succeeded(self) -> int:
        return sum(1 for outcome in self.outcomes if outcome.ok)

    @property
    def failed(self) -> int:
        return sum(1 for outcome in self.outcomes if not outcome.ok)

    @property
    def failures(self) -> list[ImportOutcome]:
        return [outcome for outcome in self.outcomes if not outcome.ok]

    @property
    def ok(self) -> bool:
        return self.failed == 0

    def as_dict(self) -> dict[str, object]:
        return {
            "sources": len(self.outcomes),
            "succeeded": self.succeeded,
            "failed": self.failed,
            "records": self.records,
            "chunks": self.chunks,
            "bytes_written": self.bytes_written,
            "skipped_unchanged": self.skipped_unchanged,
            "duplicates": self.duplicates,
            "duration_seconds": round(self.duration_seconds, 3),
            "job_id": self.job_id,
            "outcomes": [
                {
                    "source": outcome.source,
                    "status": outcome.status,
                    "records": outcome.records,
                    "chunks": outcome.chunks,
                    "message": outcome.message,
                }
                for outcome in self.outcomes
            ],
        }

    def summary(self) -> str:
        if not self.outcomes:
            return "Nothing to import."
        parts = [f"{self.succeeded} succeeded"]
        if self.skipped_unchanged:
            parts.append(f"{self.skipped_unchanged} unchanged")
        if self.failed:
            parts.append(f"{self.failed} failed")
        parts.append(f"{self.chunks} chunk(s), {self.bytes_written / 1024:.1f} KB")
        return ", ".join(parts) + "."


class KnowledgeImporter:
    """Drives the ingestion pipeline for the window's import paths.

    One instance is cheap; it resolves the store, ledger, jobs, and metadata
    stores lazily from a single root so every run writes to the same place. It
    reuses :class:`ingest.pipeline.IngestionPipeline` unchanged, so the security
    guarantees (deny-by-default, root confinement, per-source error isolation,
    bounded retries, resumable jobs) are the pipeline's, not a second copy.
    """

    def __init__(
        self,
        *,
        store_dir: str | Path | None = None,
        recursive: bool = True,
        normalize: bool = True,
        fold_arabic: bool = False,
        allow_ocr: bool | None = None,
        ocr_languages: str = "eng",
        ocr_dpi: int | None = None,
        ocr_max_pages: int | None = None,
        chunk_chars: int | None = None,
        overlap: int | None = None,
        retries: int = 2,
        allow_web: bool = False,
        ocr_engine: "PageOcrEngine | None" = None,
    ) -> None:
        self.store_dir = Path(store_dir) if store_dir is not None else default_knowledge_base_dir()
        self.recursive = recursive
        self.normalize = normalize
        self.fold_arabic = fold_arabic
        self.allow_ocr = allow_ocr
        self.ocr_languages = ocr_languages
        self.ocr_dpi = ocr_dpi
        self.ocr_max_pages = ocr_max_pages
        self.chunk_chars = chunk_chars
        self.overlap = overlap
        self.retries = retries
        self.allow_web = allow_web
        self._registry = build_default_registry(allow_web=allow_web, ocr_engine=ocr_engine)

    # -- stores -----------------------------------------------------------

    def store(self) -> KnowledgeBaseStore:
        return KnowledgeBaseStore(self.store_dir)

    def _ledger(self) -> DedupLedger:
        return DedupLedger(ledger_path(self.store_dir))

    def _jobs(self) -> JobStore:
        return JobStore(ingest_jobs_dir(self.store_dir))

    def _metadata(self) -> DocumentMetadataStore:
        return DocumentMetadataStore(metadata_path(self.store_dir))

    def _pipeline(
        self,
        *,
        progress: ProgressCallback | None,
        job: JobState | None,
        jobs: JobStore | None,
        on_item: "Callable[[ItemOutcome], None] | None" = None,
    ) -> IngestionPipeline:
        kwargs: dict[str, object] = {
            "progress": progress,
            "job": job,
            "jobs": jobs,
            "retries": self.retries,
            "metadata": self._metadata(),
        }
        if self.chunk_chars is not None:
            kwargs["chunk_chars"] = self.chunk_chars
        if self.overlap is not None:
            kwargs["overlap"] = self.overlap
        if on_item is not None:
            kwargs["on_item"] = on_item
        return IngestionPipeline(self.store(), self._ledger(), **kwargs)  # type: ignore[arg-type]

    def _options(
        self,
        *,
        recursive: bool | None = None,
        ocr: bool | None = None,
        ocr_languages: str | None = None,
        ocr_dpi: int | None = None,
        ocr_max_pages: int | None = None,
    ) -> dict[str, str]:
        use_ocr = self.allow_ocr if ocr is None else ocr
        options = {
            "normalize": "true" if self.normalize else "false",
            "fold_arabic": "true" if self.fold_arabic else "false",
            "ocr_languages": ocr_languages or self.ocr_languages,
            "recursive": "true" if (self.recursive if recursive is None else recursive) else "false",
        }
        # ``None`` leaves the choice to the reader, which OCRs a scanned page when
        # the extra is installed; True/False is an explicit opt-in/out.
        if use_ocr is not None:
            options["allow_ocr"] = "true" if use_ocr else "false"
        dpi = self.ocr_dpi if ocr_dpi is None else ocr_dpi
        max_pages = self.ocr_max_pages if ocr_max_pages is None else ocr_max_pages
        if dpi:
            options["ocr_dpi"] = str(dpi)
        if max_pages:
            options["ocr_max_pages"] = str(max_pages)
        return options

    # -- import paths -----------------------------------------------------

    def import_files(
        self,
        paths: Iterable[str | Path],
        *,
        ocr: bool | None = None,
        ocr_languages: str | None = None,
        progress: ProgressCallback | None = None,
    ) -> ImportReport:
        """Import one or many chosen files, each confined to its own folder.

        Every file is imported as its own source (with the file's folder as
        ``root``), so one malformed file is recorded and skipped without
        affecting the others, and the per-file result is available for retry.
        ``ocr=True`` reads scanned PDFs (needs the ``[ocr]`` extra + Tesseract).
        """
        report = ImportReport()
        started = time.monotonic()
        options = self._options(ocr=ocr, ocr_languages=ocr_languages)
        ocr_enabled = options.get("allow_ocr", "true") == "true"
        for raw in paths:
            path = Path(raw).expanduser()
            outcome = self._import_one(
                SourceSpec(str(path), options=options),
                root=str(path.parent),
                progress=progress,
                ocr_enabled=ocr_enabled,
            )
            report.outcomes.append(outcome)
            self._merge(report, outcome)
        report.duration_seconds = time.monotonic() - started
        return report

    def import_folder(
        self,
        folder: str | Path,
        *,
        recursive: bool | None = None,
        ocr: bool | None = None,
        ocr_languages: str | None = None,
        job_id: str | None = None,
        resume: bool = True,
        progress: ProgressCallback | None = None,
    ) -> ImportReport:
        """Import every supported file under ``folder`` as one resumable job.

        The folder is scanned lazily and each file is deduped by its fingerprint,
        so a re-run imports only new or changed files. When ``job_id`` is given
        the job is saved after every file and a later call resumes from the last
        completed item when ``resume`` is true. ``ocr=True`` reads scanned PDFs.
        """
        directory = Path(folder).expanduser()
        report = ImportReport()
        started = time.monotonic()
        spec = SourceSpec(
            str(directory),
            options=self._options(recursive=recursive, ocr=ocr, ocr_languages=ocr_languages),
        )
        job = None
        jobs = None
        resume_from = 0
        if job_id:
            jobs = self._jobs()
            job = jobs.load(job_id) or JobState(job_id=job_id, source=str(directory))
            job.connector = "files"
            # Resume only an *interrupted* run. A completed or failed job is
            # re-scanned from the start so new/changed files are found and any
            # file that failed is retried -- the ledger still skips the files
            # that already succeeded, so the re-scan stays cheap.
            if resume and job.status == RUNNING and job.position:
                resume_from = job.position
            else:
                job.position = 0
            report.job_id = job_id
        connector = self._registry.get("files") or self._registry.resolve(spec)
        ocr_enabled = spec.option("allow_ocr", "true") in {"true", "1", "yes", "on"}
        collected: list[ItemOutcome] = []
        pipeline = self._pipeline(
            progress=progress, job=job, jobs=jobs, on_item=collected.append
        )
        result = pipeline.ingest(
            spec, connector=connector, incremental=True, resume_from=resume_from
        )
        for item in collected:
            report.outcomes.append(
                ImportOutcome(
                    source=item.uri,
                    ok=item.status != "failed",
                    records=item.records,
                    chunks=item.chunks,
                    bytes_written=item.bytes_written,
                    skipped_unchanged=1 if item.status == "unchanged" else 0,
                    message=(
                        _ocr_hint(item.message, ocr_enabled=ocr_enabled)
                        if item.status == "failed"
                        else item.message
                    ),
                )
            )
        report.records += result.records
        report.chunks += result.chunks
        report.bytes_written += result.bytes_written
        report.skipped_unchanged += result.skipped_unchanged
        report.duplicates += result.skipped_duplicates
        report.duration_seconds = time.monotonic() - started
        return report

    def retry_failed(
        self,
        failures: Sequence[ImportOutcome],
        *,
        ocr: bool | None = None,
        ocr_languages: str | None = None,
        progress: ProgressCallback | None = None,
    ) -> ImportReport:
        """Re-import the files that failed, as one bounded run.

        Retrying is just importing the same paths again: the dedup ledger makes a
        re-run cheap and the pipeline's per-source isolation keeps a still-broken
        file from stopping the rest. ``ocr`` lets a retry enable OCR for a scanned
        PDF that failed on the first pass.
        """
        paths = [outcome.source for outcome in failures if not outcome.ok]
        return self.import_files(paths, ocr=ocr, ocr_languages=ocr_languages, progress=progress)

    # -- jobs -------------------------------------------------------------

    def list_jobs(self) -> list[JobState]:
        """Every recorded ingestion job, newest first."""
        return sorted(self._jobs().list(), key=lambda state: state.updated_at, reverse=True)

    def resume_job(self, job_id: str, *, progress: ProgressCallback | None = None) -> ImportReport:
        """Resume an interrupted folder job from its last completed item."""
        state = self._jobs().load(job_id)
        if state is None:
            return ImportReport(job_id=job_id)
        return self.import_folder(state.source, job_id=job_id, resume=True, progress=progress)

    def forget_job(self, job_id: str) -> bool:
        """Remove a job's state file (used after a completed run)."""
        return self._jobs().remove(job_id)

    def stats(self) -> dict[str, object]:
        """A small, honest description of the store and the import bookkeeping."""
        store = self.store()
        ledger = self._ledger().stats()
        store_stats = store.stats()
        return {
            "store": store_stats,
            "ledger": ledger,
            "jobs": len(self.list_jobs()),
        }

    # -- internals --------------------------------------------------------

    def _import_one(
        self,
        spec: SourceSpec,
        *,
        root: str | None,
        progress: ProgressCallback | None,
        ocr_enabled: bool = False,
    ) -> ImportOutcome:
        if root is not None:
            spec = SourceSpec(spec.uri, options={**spec.options, "root": root})
        try:
            connector = self._registry.resolve(spec)
        except Exception as error:  # noqa: BLE001 - resolver errors are user-facing
            return ImportOutcome(source=spec.uri, ok=False, message=str(error))
        if connector is None:
            return ImportOutcome(source=spec.uri, ok=False, message="No connector can read this source.")
        pipeline = self._pipeline(progress=progress, job=None, jobs=None)
        before = len(pipeline.errors)
        result = pipeline.ingest(spec, connector=connector, incremental=True)
        return self._outcome(spec.uri, result, pipeline, before, ocr_enabled=ocr_enabled)

    def _outcome(
        self,
        source: str,
        result: IngestionReport,
        pipeline: IngestionPipeline,
        before: int,
        *,
        ocr_enabled: bool = False,
    ) -> ImportOutcome:
        errors = pipeline.errors.as_dicts()[before:]
        if errors:
            message = _ocr_hint(errors[0]["message"], ocr_enabled=ocr_enabled)
            return ImportOutcome(source=source, ok=False, message=message)
        if result.failed:
            return ImportOutcome(source=source, ok=False, message="The source failed to import.")
        return ImportOutcome(
            source=source,
            ok=True,
            records=result.records,
            chunks=result.chunks,
            bytes_written=result.bytes_written,
            skipped_unchanged=result.skipped_unchanged,
        )

    def _merge(self, report: ImportReport, outcome: ImportOutcome) -> None:
        report.records += outcome.records
        report.chunks += outcome.chunks
        report.bytes_written += outcome.bytes_written
        report.skipped_unchanged += outcome.skipped_unchanged


def _ocr_hint(message: str, *, ocr_enabled: bool) -> str:
    """Append a real next step to a scanned-PDF message.

    A file with no text layer is not an error the reader can fix silently; the
    message must say how to read it. This turns the reader's own wording into an
    instruction the Import Center can act on (enable OCR), rather than repeating
    the CLI-only flag name at a user who has no CLI.
    """
    if "no extractable text" not in message.lower():
        return message
    from core.pdf_ocr import ocr_available

    if ocr_enabled:
        if not ocr_available():
            return message + (
                " OCR is not installed; install the [ocr] extra and the Tesseract engine "
                "to read scanned PDFs."
            )
        return message + (
            " OCR is enabled but found no readable text; the page may be blank or too "
            "low-quality to recognise."
        )
    return message + (
        " Enable OCR to read it: import this PDF again with the “Read scanned PDFs (OCR)” "
        "option (the Import Center window, or `python -m cli import-center --ocr files <path>`); "
        "it needs the optional [ocr] extra plus the Tesseract engine."
    )


def default_store_dir() -> Path:
    """Where imports are stored: the knowledge base beside the project's data."""
    return default_knowledge_base_dir()


def import_root() -> Path:
    """The data directory the import bookkeeping lives under (for display)."""
    return data_dir()


__all__ = [
    "DONE",
    "FAILED",
    "RUNNING",
    "ImportOutcome",
    "ImportReport",
    "JobState",
    "KnowledgeImporter",
    "SUPPORTED_SUFFIXES",
    "UNSUPPORTED_SUFFIXES",
    "default_store_dir",
    "import_root",
    "picker_filetypes",
]
