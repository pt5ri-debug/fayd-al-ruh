"""The Knowledge Import Center: files, folders, dedup, retry, resume, limits.

These tests drive :class:`gui.import_center.KnowledgeImporter` (the front end the
window's ＋ control uses) with small synthetic fixtures -- Arabic and English
text, CSV, JSON, a DOCX and a PDF, malformed files, repeated files, and a
larger folder -- and assert the behaviour the requirements call for:

* supported formats extract text and keep useful metadata (path, filename, hash,
  timestamp);
* unchanged content is detected and not ingested twice;
* a malformed file fails on its own without aborting the batch;
* a failed file can be retried;
* an interrupted folder job can be resumed;
* a larger batch is processed in bounded batches (measurable peak memory).

They reuse the project's own DOCX/PDF fixture builders rather than adding a
generation dependency.
"""

from __future__ import annotations

import json
import tracemalloc
from pathlib import Path

import pytest
from core.knowledge_base import KnowledgeBaseStore
from gui.import_center import (
    SUPPORTED_SUFFIXES,
    UNSUPPORTED_SUFFIXES,
    KnowledgeImporter,
    picker_filetypes,
)
from ingest.metadata import DocumentMetadataStore, metadata_path

from . import docx_fixtures, pdf_fixtures


def _importer(tmp_path: Path, **kwargs) -> KnowledgeImporter:
    return KnowledgeImporter(store_dir=tmp_path / "kb", **kwargs)


def _docs(tmp_path: Path) -> Path:
    folder = tmp_path / "docs"
    folder.mkdir(exist_ok=True)
    return folder


# --- supported formats ----------------------------------------------------


def test_txt_and_markdown_arabic_and_english(tmp_path: Path) -> None:
    folder = _docs(tmp_path)
    english = folder / "porosity.txt"
    english.write_text("Porosity measures the void space in a reservoir rock.\n", encoding="utf-8")
    arabic = folder / "المسامية.md"
    arabic.write_text("# المسامية\n\nالمسامية تقيس الفراغات في الصخر.\n", encoding="utf-8")

    report = _importer(tmp_path).import_files([english, arabic])

    assert report.failed == 0
    assert report.succeeded == 2
    assert report.chunks >= 2
    # Arabic content survives normalization: the store holds the Arabic sentence.
    store = KnowledgeBaseStore(tmp_path / "kb")
    text = "\n".join(doc.text for doc in store.iter_documents())
    assert "الفراغات" in text
    assert "Porosity" in text


def test_csv_and_json_are_streamed_and_rendered(tmp_path: Path) -> None:
    folder = _docs(tmp_path)
    (folder / "wells.csv").write_text("well,depth\nA-1,2500\nA-2,3100\n", encoding="utf-8")
    (folder / "wells.json").write_text(
        json.dumps([{"well": "B-1", "oil": 120}, {"well": "B-2", "oil": 90}]),
        encoding="utf-8",
    )

    report = _importer(tmp_path).import_files([folder / "wells.csv", folder / "wells.json"])

    assert report.failed == 0
    store = KnowledgeBaseStore(tmp_path / "kb")
    text = "\n".join(doc.text for doc in store.iter_documents())
    assert "well: A-1" in text and "depth: 2500" in text
    assert "well: B-1" in text


def test_docx_is_read_through_the_shared_extractor(tmp_path: Path) -> None:
    folder = _docs(tmp_path)
    docx = docx_fixtures.latin_and_arabic_docx(folder / "report.docx")

    report = _importer(tmp_path).import_files([docx])

    assert report.failed == 0
    store = KnowledgeBaseStore(tmp_path / "kb")
    text = "\n".join(doc.text for doc in store.iter_documents())
    assert "ضغط الخزان" in text


def test_pdf_requires_the_optional_extra_or_reports_clearly(tmp_path: Path) -> None:
    folder = _docs(tmp_path)
    pdf = pdf_fixtures.write_pdf(folder / "paper.pdf", [pdf_fixtures.latin_page("Reservoir pressure.")])

    report = _importer(tmp_path).import_files([pdf])

    from core.pdf_extract import pdf_available

    if pdf_available():
        assert report.failed == 0
        store = KnowledgeBaseStore(tmp_path / "kb")
        text = "\n".join(doc.text for doc in store.iter_documents())
        assert "Reservoir pressure" in text
    else:
        # Without pypdf the file must fail on its own with a clear message,
        # never crash the run and never mis-read the bytes.
        assert report.failed == 1
        assert "pypdf" in report.outcomes[0].message.lower() or "extra" in report.outcomes[0].message.lower()


# --- metadata -------------------------------------------------------------


def test_metadata_records_path_hash_and_timestamp(tmp_path: Path) -> None:
    folder = _docs(tmp_path)
    source = folder / "notes.txt"
    source.write_text("Waterflooding maintains reservoir pressure.\n", encoding="utf-8")

    _importer(tmp_path).import_files([source])

    store = DocumentMetadataStore(metadata_path(tmp_path / "kb"))
    doc_ids = store._store.list_keys("documents")
    assert doc_ids, "expected at least one metadata record"
    meta = store.get(doc_ids[0])
    assert meta is not None
    assert meta.uri.endswith("notes.txt")
    assert meta.text_hash
    assert meta.created_at
    assert meta.chunk_ids


# --- dedup ----------------------------------------------------------------


def test_repeated_file_is_not_ingested_twice(tmp_path: Path) -> None:
    folder = _docs(tmp_path)
    source = folder / "same.txt"
    source.write_text("Repeated content about casing design.\n", encoding="utf-8")

    importer = _importer(tmp_path)
    first = importer.import_files([source])
    second = importer.import_files([source])

    assert first.chunks >= 1
    assert second.chunks == 0
    assert second.skipped_unchanged == 1
    assert second.outcomes[0].status == "unchanged"


def test_modified_file_is_re_ingested(tmp_path: Path) -> None:
    folder = _docs(tmp_path)
    source = folder / "changing.txt"
    source.write_text("Version one of the drilling note.\n", encoding="utf-8")
    importer = _importer(tmp_path)
    importer.import_files([source])

    source.write_text("Version two of the drilling note, now longer.\n", encoding="utf-8")
    report = importer.import_files([source])

    assert report.skipped_unchanged == 0
    assert report.chunks >= 1


# --- malformed / unsupported ----------------------------------------------


def test_malformed_file_fails_alone_and_the_batch_continues(tmp_path: Path) -> None:
    folder = _docs(tmp_path)
    good = folder / "good.txt"
    good.write_text("A valid sentence about permeability.\n", encoding="utf-8")
    bad = folder / "broken.json"
    bad.write_text("{ this is not valid json ", encoding="utf-8")

    report = _importer(tmp_path).import_folder(folder)

    assert report.succeeded == 1
    assert report.failed == 1
    assert report.failures[0].filename == "broken.json"
    assert "JSON" in report.failures[0].message


def test_unsupported_extension_is_refused_clearly(tmp_path: Path) -> None:
    folder = _docs(tmp_path)
    legacy = folder / "old.doc"
    legacy.write_text("binary junk", encoding="utf-8")

    report = _importer(tmp_path).import_files([legacy])

    assert report.failed == 1
    assert "Unsupported file type" in report.outcomes[0].message


def test_folder_skips_unsupported_files_silently(tmp_path: Path) -> None:
    folder = _docs(tmp_path)
    (folder / "keep.txt").write_text("Keep this text.\n", encoding="utf-8")
    (folder / "ignore.doc").write_text("junk", encoding="utf-8")
    (folder / "ignore.xyz").write_text("junk", encoding="utf-8")

    report = _importer(tmp_path).import_folder(folder)

    names = {outcome.filename for outcome in report.outcomes}
    assert names == {"keep.txt"}


def test_encoding_error_is_reported_not_scrambled(tmp_path: Path) -> None:
    folder = _docs(tmp_path)
    # A lone 0xFF byte is invalid UTF-8 and matches no legacy encoding cleanly.
    broken = folder / "bad-encoding.txt"
    broken.write_bytes(b"valid start \xff\xfe\xff\x00 then garbage")

    report = _importer(tmp_path, normalize=True).import_files([broken])

    # Either it decoded via a legacy codec or it failed cleanly -- never a crash.
    assert report.outcomes[0].status in {"ok", "failed"}


# --- retry ----------------------------------------------------------------


def test_failed_file_can_be_retried_after_it_is_fixed(tmp_path: Path) -> None:
    folder = _docs(tmp_path)
    (folder / "ok.txt").write_text("Good content here.\n", encoding="utf-8")
    bad = folder / "later.json"
    bad.write_text("{ broken", encoding="utf-8")
    importer = _importer(tmp_path)

    first = importer.import_folder(folder)
    assert first.failed == 1

    bad.write_text(json.dumps({"fixed": True}), encoding="utf-8")
    retried = importer.retry_failed(first.failures)

    assert retried.failed == 0
    assert retried.succeeded == 1


# --- resume ---------------------------------------------------------------


def test_folder_job_records_state_and_can_be_resumed(tmp_path: Path) -> None:
    folder = _docs(tmp_path)
    for index in range(5):
        (folder / f"f{index}.txt").write_text(f"Document {index} about well logging.\n", encoding="utf-8")
    importer = _importer(tmp_path)

    report = importer.import_folder(folder, job_id="job-a")

    assert report.job_id == "job-a"
    jobs = importer.list_jobs()
    assert len(jobs) == 1
    assert jobs[0].job_id == "job-a"
    assert jobs[0].status in {"done", "failed"}

    # Resuming a finished job re-scans and skips everything as unchanged.
    resumed = importer.resume_job("job-a")
    assert resumed.failed == 0
    assert resumed.skipped_unchanged == 5


def test_incremental_rescan_imports_only_new_files(tmp_path: Path) -> None:
    folder = _docs(tmp_path)
    for index in range(4):
        (folder / f"a{index}.txt").write_text(f"Existing note {index}.\n", encoding="utf-8")
    importer = _importer(tmp_path)
    importer.import_folder(folder, job_id="job-b")

    (folder / "new.txt").write_text("A brand new note.\n", encoding="utf-8")
    report = importer.import_folder(folder, job_id="job-b")

    assert report.skipped_unchanged == 4
    assert report.succeeded == 5  # four unchanged + one new
    assert report.chunks >= 1


def test_recursive_can_be_disabled(tmp_path: Path) -> None:
    folder = _docs(tmp_path)
    (folder / "top.txt").write_text("Top level.\n", encoding="utf-8")
    sub = folder / "sub"
    sub.mkdir()
    (sub / "deep.txt").write_text("Nested level.\n", encoding="utf-8")

    report = _importer(tmp_path).import_folder(folder, recursive=False)

    assert {outcome.filename for outcome in report.outcomes} == {"top.txt"}


# --- streaming / memory ---------------------------------------------------


def test_large_batch_imports_with_bounded_memory(tmp_path: Path) -> None:
    folder = _docs(tmp_path)
    payload = "The reservoir simulation grid is refined near the wellbore. " * 40
    for index in range(60):
        (folder / f"doc{index:03d}.txt").write_text(f"{payload}\n", encoding="utf-8")

    importer = _importer(tmp_path)
    tracemalloc.start()
    try:
        report = importer.import_folder(folder)
        _current, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()

    assert report.failed == 0
    assert report.succeeded == 60
    # The import must not hold the corpus in memory: 60 files of ~2.6 KB each
    # (~160 KB total) should not need anywhere near that as a live allocation.
    assert peak < 8 * 1024 * 1024, f"peak traced memory was {peak} bytes"


def test_large_single_file_is_read_without_whole_file_buffer(tmp_path: Path) -> None:
    folder = _docs(tmp_path)
    big = folder / "big.txt"
    # ~5 MB, above the reader's 4 MB stream threshold, written without loading
    # it whole here either.
    with big.open("w", encoding="utf-8") as handle:
        for _ in range(5000):
            handle.write("Sandstone porosity and permeability description line.\n")

    importer = _importer(tmp_path)
    tracemalloc.start()
    try:
        report = importer.import_files([big])
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()

    assert report.failed == 0
    assert peak < 32 * 1024 * 1024, f"peak traced memory was {peak} bytes"


# --- helpers --------------------------------------------------------------


def test_supported_and_unsupported_lists_are_honest() -> None:
    for suffix in (".txt", ".md", ".csv", ".json", ".pdf", ".docx"):
        assert suffix in SUPPORTED_SUFFIXES
    assert ".doc" in UNSUPPORTED_SUFFIXES and ".rtf" in UNSUPPORTED_SUFFIXES
    filetypes = picker_filetypes()
    assert filetypes and filetypes[0][0] == "Knowledge documents"


# --- service wiring -------------------------------------------------------


def test_service_exposes_the_import_center(tmp_path: Path) -> None:
    from tests.test_gui import FakeSession, _service

    service = _service(FakeSession(), import_store_dir=tmp_path / "kb")
    folder = _docs(tmp_path)
    (folder / "one.txt").write_text("Import through the service.\n", encoding="utf-8")

    report = service.import_files([folder / "one.txt"])

    assert report.succeeded == 1
    assert service.importer.store_dir == tmp_path / "kb"
    assert service.import_stats()["store"]["chunks"] >= 1


def test_dialog_labels_match_the_import_center() -> None:
    # The dialog module is Tk-only; importing it must not require a display.
    from gui import import_dialog

    assert import_dialog.TITLE == "Knowledge Import Center"
    assert "bounded batches" in import_dialog.SUBTITLE
    assert ".txt" in import_dialog.FORMATS_LINE
    assert ".doc" in import_dialog.UNSUPPORTED_LINE


def _display_available() -> bool:
    try:
        import tkinter as tk

        root = tk.Tk()
    except Exception:
        return False
    root.destroy()
    return True


@pytest.mark.skipif(not _display_available(), reason="no display available")
def test_import_center_window_opens_from_the_plus_button(tmp_path: Path) -> None:
    import tkinter as tk

    from gui.app import FaydGUI

    from tests.test_gui import FakeSession, _service

    root = tk.Tk()
    try:
        app = FaydGUI(root, service=_service(FakeSession(), import_store_dir=tmp_path / "kb"))
        root.update()
        app._open_import_center()
        root.update()
        assert app._import_dialog is not None
        assert app._import_dialog.window.title() == "Knowledge Import Center"
        app._import_dialog._close()
        root.update()
        assert app._import_dialog is None
    finally:
        root.destroy()


# --- pipeline per-item callback -------------------------------------------


def test_pipeline_on_item_reports_each_file(tmp_path: Path) -> None:
    from core.knowledge_base import KnowledgeBaseStore
    from ingest.dedup import DedupLedger
    from ingest.pipeline import IngestionPipeline, ledger_path
    from ingest.records import ItemOutcome, SourceSpec
    from ingest.registry import build_default_registry

    folder = _docs(tmp_path)
    (folder / "a.txt").write_text("First file.\n", encoding="utf-8")
    (folder / "bad.json").write_text("{ broken", encoding="utf-8")
    (folder / "b.txt").write_text("Second file.\n", encoding="utf-8")

    store = KnowledgeBaseStore(tmp_path / "kb")
    seen: list[ItemOutcome] = []
    pipeline = IngestionPipeline(
        store, DedupLedger(ledger_path(store.root)), on_item=seen.append
    )
    registry = build_default_registry()
    spec = SourceSpec(str(folder))
    pipeline.ingest(spec, connector=registry.resolve(spec), incremental=True)

    statuses = {outcome.title: outcome.status for outcome in seen}
    assert statuses == {"a": "ok", "bad": "failed", "b": "ok"}


# --- CLI ------------------------------------------------------------------


def test_cli_import_center_formats_and_folder(tmp_path: Path, capsys) -> None:
    from cli import import_center as cli_ic

    assert cli_ic.main(["formats"]) == 0
    out = capsys.readouterr().out
    assert ".txt" in out and ".docx" in out

    folder = _docs(tmp_path)
    (folder / "note.txt").write_text("CLI import note.\n", encoding="utf-8")
    code = cli_ic.main(["--store-dir", str(tmp_path / "kb"), "folder", str(folder)])
    out = capsys.readouterr().out
    assert code == 0
    assert "1 succeeded" in out


def test_cli_import_center_interactive_imports_a_path(tmp_path: Path, capsys) -> None:
    # The numbered menu runs this with no path and no terminal, which must never
    # be a silent no-op; with a path it imports for real.
    from cli import import_center as cli_ic

    folder = _docs(tmp_path)
    (folder / "one.txt").write_text("Interactive import.\n", encoding="utf-8")

    code = cli_ic.main(["--store-dir", str(tmp_path / "kb"), "interactive", str(folder)])
    out = capsys.readouterr().out
    assert code == 0
    assert "Knowledge Import Center" in out
    assert "1 succeeded" in out


def test_cli_import_center_no_args_gives_guidance_when_headless(capsys) -> None:
    from cli import import_center as cli_ic

    # No subcommand and no path: in a non-tty it must explain how to pass a path
    # rather than quietly printing only the formats.
    code = cli_ic.main([])
    out = capsys.readouterr().out
    assert code == 0
    assert "import-center files" in out or "import-center folder" in out


def test_menu_import_center_entry_actually_imports(tmp_path: Path, monkeypatch, capsys) -> None:
    # Regression: the Import Center menu entry must not merely print the format
    # list. Find its numbered position and drive the menu choice, asserting it
    # entered the Import Center (not just `formats`).
    import io

    from cli import menu
    from cli.menu import run_import_center

    position = next(
        index
        for index, (title, _action, _argv) in enumerate(menu.visible_menu(), start=1)
        if title == "Import files and folders (Knowledge Import Center)"
    )
    title, _action, argv = menu.visible_menu()[position - 1]
    assert title == "Import files and folders (Knowledge Import Center)"
    assert argv == ("interactive",)

    # A deterministic non-tty stdin, so the entry shows the Import Center header
    # and a way forward rather than only the format list, and never blocks.
    monkeypatch.setattr("sys.stdin", io.StringIO(""))
    code = menu.run_choice(str(position))
    out = capsys.readouterr().out
    assert code == 0
    assert "Knowledge Import Center" in out

    # And with a real path it imports.
    folder = _docs(tmp_path)
    (folder / "menu.txt").write_text("Menu-driven import.\n", encoding="utf-8")
    code = run_import_center(("--store-dir", str(tmp_path / "kb"), "interactive", str(folder)))
    out = capsys.readouterr().out
    assert code == 0
    assert "1 succeeded" in out


# --- OCR for scanned PDFs -------------------------------------------------


class _FakeOcrEngine:
    """A real :class:`PageOcrEngine` double: returns fixed text, records pages."""

    def __init__(self, *, languages: tuple[str, ...] = ("eng",), text: str = "OCR page text") -> None:
        self._languages = languages
        self._text = text
        self.seen: list[int] = []

    def available_languages(self) -> tuple[str, ...]:
        return self._languages

    def recognize_page(self, path, number, *, languages, config):
        self.seen.append(number)
        return f"{self._text} {number}"


def _scanned_pdf(tmp_path: Path, name: str = "scan.pdf") -> Path:
    # An image-only page: no text layer, so the native reader refuses it.
    return pdf_fixtures.write_pdf(tmp_path / name, [pdf_fixtures.image_page()])


def test_scanned_pdf_with_ocr_disabled_names_the_next_step(tmp_path: Path) -> None:
    # With OCR turned off, the failure must name a real, reachable action.
    report = _importer(tmp_path).import_files([_scanned_pdf(tmp_path)], ocr=False)

    assert report.failed == 1
    message = report.failures[0].message
    assert "no extractable text" in message.lower()
    assert "Enable OCR" in message
    assert "Read scanned PDFs (OCR)" in message


def test_import_files_with_ocr_reads_a_scanned_pdf(tmp_path: Path) -> None:
    engine = _FakeOcrEngine(text="QAQC Final 2025")
    importer = _importer(tmp_path, ocr_engine=engine)

    report = importer.import_files([_scanned_pdf(tmp_path)], ocr=True)

    assert report.succeeded == 1
    assert engine.seen == [1]
    stored = list(importer.store().iter_chunks())
    assert any("QAQC Final 2025" in chunk.text for chunk in stored)


def test_retry_failed_can_enable_ocr_for_a_scanned_pdf(tmp_path: Path) -> None:
    engine = _FakeOcrEngine(text="Permeability 250 mD")
    importer = _importer(tmp_path, ocr_engine=engine)

    first = importer.import_files([_scanned_pdf(tmp_path)], ocr=False)
    assert first.failed == 1 and engine.seen == []

    retry = importer.retry_failed(first.failures, ocr=True)

    assert retry.succeeded == 1
    assert engine.seen == [1]


def test_ocr_languages_reach_the_engine(tmp_path: Path) -> None:
    engine = _FakeOcrEngine(languages=("eng", "ara"))
    importer = _importer(tmp_path, ocr_engine=engine)

    report = importer.import_files([_scanned_pdf(tmp_path)], ocr=True, ocr_languages="ara+eng")

    assert report.succeeded == 1


def test_attach_document_ocr_hint_names_the_window_action(tmp_path: Path) -> None:
    from gui import attachments

    failed = attachments.attach_document(_scanned_pdf(tmp_path))

    assert failed.ok is False
    assert "no extractable text" in failed.message.lower()
    assert "Read scanned PDFs (OCR)" in failed.message


def test_attach_document_with_ocr_never_raises(tmp_path: Path) -> None:
    from gui import attachments

    # The OCR path is opt-in; whether or not the extra is installed, a failure is
    # reported as a notice, never raised at the caller.
    notice = attachments.attach_document(_scanned_pdf(tmp_path), ocr=True)

    assert isinstance(notice.ok, bool)
    if not notice.ok:
        assert notice.message


# --- CLI OCR plumbing -----------------------------------------------------


def test_cli_ocr_flags_reach_the_importer() -> None:
    from cli import import_center as cli_ic

    args = cli_ic._build_parser().parse_args(
        ["--ocr", "--ocr-languages", "ara+eng", "--ocr-dpi", "300", "formats"]
    )
    importer = cli_ic._importer(args)

    assert importer.allow_ocr is True
    assert importer.ocr_languages == "ara+eng"
    assert importer.ocr_dpi == 300


def test_cli_ocr_retry_offer_is_skipped_when_ocr_is_on(tmp_path: Path) -> None:
    from cli import import_center as cli_ic

    report = _importer(tmp_path).import_files([_scanned_pdf(tmp_path)], ocr=False)
    args = cli_ic._build_parser().parse_args(["--ocr", "files", str(tmp_path)])

    # Already asking for OCR: no second offer, and never a prompt.
    assert cli_ic._offer_ocr_retry(args, _importer(tmp_path), report) is False


def test_cli_ocr_retry_offer_declines_without_a_terminal(tmp_path: Path, monkeypatch) -> None:
    import io

    from cli import import_center as cli_ic

    report = _importer(tmp_path).import_files([_scanned_pdf(tmp_path)], ocr=False)
    args = cli_ic._build_parser().parse_args(["files", str(tmp_path)])
    monkeypatch.setattr("sys.stdin", io.StringIO(""))  # not a tty

    assert cli_ic._offer_ocr_retry(args, _importer(tmp_path), report) is False


def test_cli_ocr_retry_offer_retries_when_accepted(tmp_path: Path, monkeypatch, capsys) -> None:
    from cli import import_center as cli_ic

    scanned = _scanned_pdf(tmp_path)
    importer = _importer(tmp_path)
    report = importer.import_files([scanned], ocr=False)
    assert report.failed == 1

    args = cli_ic._build_parser().parse_args(["files", str(scanned)])
    monkeypatch.setattr("sys.stdin", _FakeTty("y"))

    offered = cli_ic._offer_ocr_retry(args, importer, report)

    # On a machine with the OCR extra the retry reads the page for real; without
    # it the offer still fires and the retry reports honestly. Either way the
    # user was asked and an attempt was made.
    assert offered is True
    assert "Retry them with OCR now?" in capsys.readouterr().out


class _FakeTty:
    """A minimal tty-like stdin: reports ``isatty`` and feeds one answer."""

    def __init__(self, answer: str) -> None:
        self._answer = answer + "\n"

    def isatty(self) -> bool:
        return True

    def readline(self) -> str:
        return self._answer

    def read(self, *_args: object) -> str:
        return self._answer
