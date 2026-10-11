"""Stage 3: optional OCR for scanned PDF documents.

OCR is a *separate, optional* path. These tests prove three things without ever
requiring the OCR extra:

* it is **off by default** -- a normal PDF is read with no OCR import and no
  engine, and a scanned PDF is reported, not mis-read;
* the detection, config validation, language checks, and raster/temp limits are
  pure and testable on their own; and
* when it *is* enabled, only the low-text pages are OCR'd, a page with real text
  is never replaced, page provenance is kept, and a per-page failure is recorded
  without aborting the document.

PDFs come from :mod:`tests.pdf_fixtures` (hand-built, no PDF library); the engine
is a real callable test double implementing :class:`core.pdf_ocr.PageOcrEngine`,
so no project logic is mocked -- only the external rasteriser/engine boundary.
"""

from __future__ import annotations

import time

import pytest

from core.document_reader import DocumentReadError, read_document
from core.pdf_extract import pdf_available
from core.pdf_ocr import (
    OCR_EXTRA,
    OcrConfig,
    OcrError,
    OcrLanguageError,
    OcrLimitError,
    OcrUnavailableError,
    TesseractOcrEngine,
    _write_bounded_png,
    check_raster_limits,
    extract_pdf_pages,
    ocr_available,
    page_needs_ocr,
)
from ingest.connectors.files import FileConnector
from ingest.errors import ConnectorError
from ingest.records import SourceSpec
from tests.pdf_fixtures import image_page, latin_page, write_pdf

pytestmark = pytest.mark.skipif(not pdf_available(), reason="pypdf (the [pdf] extra) is not installed")

_ARABIC = "ضغط الخزان والنفاذية"


class FakeOcrEngine:
    """A real :class:`PageOcrEngine` double -- no mocking, just a small engine.

    Returns a deterministic per-page string, records the pages it was asked for,
    and can be told to fail a page (once, or always) so retry and failure handling
    are exercised against the real orchestration in :mod:`core.pdf_ocr`.
    """

    def __init__(
        self,
        *,
        languages: tuple[str, ...] = ("eng",),
        fail_pages: tuple[int, ...] = (),
        fail_once: bool = False,
        delay: float = 0.0,
    ) -> None:
        self._languages = languages
        self._fail_pages = set(fail_pages)
        self._fail_once = fail_once
        self._delay = delay
        self.seen: list[int] = []
        self.calls: list[int] = []

    def available_languages(self) -> tuple[str, ...]:
        return self._languages

    def recognize_page(self, path, number, *, languages, config):
        self.calls.append(number)
        if self._delay:
            time.sleep(self._delay)
        if number in self._fail_pages:
            if self._fail_once:
                self._fail_pages.discard(number)
            raise OcrError(f"engine could not read page {number}")
        self.seen.append(number)
        return f"OCR text for page {number}"


# --------------------------------------------------------------------------- #
# Config, detection, limits: pure, no OCR dependency needed
# --------------------------------------------------------------------------- #


def test_ocr_config_is_off_by_default():
    assert OcrConfig().enabled is False


def test_ocr_config_validates_its_limits():
    with pytest.raises(ValueError):
        OcrConfig(languages=())
    with pytest.raises(ValueError):
        OcrConfig(dpi=10)
    with pytest.raises(ValueError):
        OcrConfig(batch_size=0)
    with pytest.raises(ValueError):
        OcrConfig(max_pages=-1)
    with pytest.raises(ValueError):
        OcrConfig(max_pixels_per_page=0)
    with pytest.raises(ValueError):
        OcrConfig(time_budget_seconds=-1)
    with pytest.raises(ValueError):
        OcrConfig(max_temp_bytes=10)
    with pytest.raises(ValueError):
        OcrConfig(retries=-1)


def test_page_needs_ocr_flags_only_thin_text():
    config = OcrConfig(min_chars_per_page=16)
    assert page_needs_ocr("", config) is True
    assert page_needs_ocr("   \n  ", config) is True
    assert page_needs_ocr("12", config) is True  # a bare page number
    assert page_needs_ocr("Permeability measures fluid flow through rock.", config) is False


def test_ocr_available_is_a_bool():
    assert isinstance(ocr_available(), bool)


def test_check_raster_limits_refuses_over_budget():
    config = OcrConfig(max_page_dimension=1000, max_pixels_per_page=500_000)
    check_raster_limits(800, 600, config, 1)  # within both limits: no raise
    with pytest.raises(OcrLimitError) as error:
        check_raster_limits(1200, 600, config, 2)
    assert "dimension" in str(error.value)
    with pytest.raises(OcrLimitError) as error:
        check_raster_limits(900, 900, config, 3)
    assert "px limit" in str(error.value)


def test_write_bounded_png_refuses_an_over_budget_raster():
    image = _tiny_image()
    try:
        with pytest.raises(OcrLimitError):
            _write_bounded_png(image, 10, 1)
    finally:
        image.close()


def _tiny_image():
    from PIL import Image

    return Image.new("L", (64, 64), 255)


def test_default_engine_reports_a_missing_extra_clearly():
    """When the OCR packages are absent the engine names the extra to install."""
    if ocr_available():
        pytest.skip("the OCR extra is installed; the unavailable path cannot be shown")
    with pytest.raises(OcrUnavailableError) as error:
        TesseractOcrEngine().available_languages()
    message = str(error.value)
    assert OCR_EXTRA in message
    assert "tesseract" in message.lower()


def test_resolve_languages_degrades_to_the_present_ones():
    from core.pdf_ocr import resolve_languages

    engine = FakeOcrEngine(languages=("eng",))
    assert resolve_languages(engine, ("ara", "eng")) == ("eng",)
    with pytest.raises(OcrLanguageError) as error:
        resolve_languages(engine, ("ara",))
    assert "ara" in str(error.value)
    assert "tesseract-ocr-ara" in str(error.value)


# --------------------------------------------------------------------------- #
# Orchestration: native extraction + per-page OCR
# --------------------------------------------------------------------------- #


def test_extract_without_ocr_keeps_native_text(tmp_path):
    path = write_pdf(tmp_path / "doc.pdf", [latin_page("Alpha content here."), latin_page("Beta.")])

    result = extract_pdf_pages(path)  # config is None -> OCR never runs

    assert result.used_ocr is False
    assert result.ocr_pages == ()
    assert "Alpha content here." in result.pages[0].text
    assert result.native_pages == (1, 2)


def test_disabled_config_does_not_touch_a_scanned_pdf(tmp_path):
    path = write_pdf(tmp_path / "scan.pdf", [image_page(), image_page()])

    result = extract_pdf_pages(path, config=OcrConfig(enabled=False))

    assert result.used_ocr is False
    assert result.text == ""  # no text layer, and OCR was not asked for
    assert result.page_count == 2


def test_ocr_replaces_only_the_scanned_pages(tmp_path):
    path = write_pdf(
        tmp_path / "mixed.pdf",
        [latin_page("Native page one text."), image_page(), latin_page("Native page three text.")],
    )
    engine = FakeOcrEngine()

    result = extract_pdf_pages(path, config=OcrConfig(enabled=True), engine=engine)

    assert engine.seen == [2]  # only the image-only page was OCR'd
    assert result.ocr_pages == (2,)
    assert "Native page one text." in result.pages[0].text
    assert result.pages[1].text == "OCR text for page 2"
    assert "Native page three text." in result.pages[2].text
    assert result.native_pages == (1, 3)
    assert result.languages_used == ("eng",)


def test_ocr_never_overwrites_a_page_that_already_has_text(tmp_path):
    path = write_pdf(tmp_path / "native.pdf", [latin_page("Real extracted text stays.")])
    engine = FakeOcrEngine()

    result = extract_pdf_pages(path, config=OcrConfig(enabled=True), engine=engine)

    assert engine.calls == []  # nothing was sent to OCR
    assert result.pages[0].text == "Real extracted text stays."


def test_ocr_records_a_page_failure_without_aborting(tmp_path):
    path = write_pdf(tmp_path / "scan.pdf", [image_page(), image_page()])
    engine = FakeOcrEngine(fail_pages=(1,))

    result = extract_pdf_pages(path, config=OcrConfig(enabled=True, retries=0), engine=engine)

    assert result.ocr_pages == (2,)
    assert [failure.number for failure in result.failures] == [1]
    assert "page 1" in result.failures[0].reason


def test_ocr_retries_a_transient_failure(tmp_path):
    path = write_pdf(tmp_path / "scan.pdf", [image_page()])
    engine = FakeOcrEngine(fail_pages=(1,), fail_once=True)

    result = extract_pdf_pages(path, config=OcrConfig(enabled=True, retries=1), engine=engine)

    assert engine.calls == [1, 1]  # first attempt failed, retry succeeded
    assert result.ocr_pages == (1,)
    assert result.failures == ()


def test_ocr_max_pages_leaves_the_rest_unattempted(tmp_path):
    path = write_pdf(tmp_path / "scan.pdf", [image_page(), image_page(), image_page()])
    engine = FakeOcrEngine()

    result = extract_pdf_pages(
        path, config=OcrConfig(enabled=True, max_pages=1), engine=engine
    )

    assert result.ocr_pages == (1,)
    assert result.skipped_pages == (2, 3)
    assert engine.seen == [1]


def test_ocr_batches_are_bounded_and_ordered(tmp_path):
    path = write_pdf(tmp_path / "scan.pdf", [image_page(), image_page(), image_page()])
    engine = FakeOcrEngine()

    result = extract_pdf_pages(
        path, config=OcrConfig(enabled=True, batch_size=2), engine=engine
    )

    assert engine.seen == [1, 2, 3]
    assert result.ocr_pages == (1, 2, 3)


def test_ocr_stops_when_the_time_budget_is_reached(tmp_path):
    path = write_pdf(tmp_path / "scan.pdf", [image_page(), image_page(), image_page()])
    engine = FakeOcrEngine(delay=0.02)

    result = extract_pdf_pages(
        path,
        config=OcrConfig(enabled=True, batch_size=1, time_budget_seconds=0.001),
        engine=engine,
    )

    # The first batch always runs; the budget stops the rest honestly.
    assert result.ocr_pages == (1,)
    assert result.skipped_pages == (2, 3)


def test_ocr_reports_missing_languages_for_the_whole_document(tmp_path):
    path = write_pdf(tmp_path / "scan.pdf", [image_page()])
    engine = FakeOcrEngine(languages=("eng",))

    with pytest.raises(OcrLanguageError):
        extract_pdf_pages(
            path, config=OcrConfig(enabled=True, languages=("ara",)), engine=engine
        )


def test_ocr_text_keeps_a_rtl_hint(tmp_path):
    path = write_pdf(tmp_path / "scan.pdf", [image_page()])

    class ArabicEngine(FakeOcrEngine):
        def recognize_page(self, path, number, *, languages, config):
            return _ARABIC

    result = extract_pdf_pages(
        path, config=OcrConfig(enabled=True), engine=ArabicEngine()
    )

    assert result.pages[0].rtl_hint is True


# --------------------------------------------------------------------------- #
# Reader integration
# --------------------------------------------------------------------------- #


def test_reader_marks_scanned_pages_with_provenance(tmp_path):
    path = write_pdf(tmp_path / "scan.pdf", [image_page(), image_page()])
    engine = FakeOcrEngine()

    document = read_document(path, ocr_config=OcrConfig(enabled=True), ocr_engine=engine)

    assert document.kind == "pdf-ocr"
    assert document.scanned is True
    assert document.ocr_pages == (1, 2)
    assert "OCR text for page 1" in document.text


def test_reader_does_not_ocr_a_digital_pdf(tmp_path):
    path = write_pdf(tmp_path / "doc.pdf", [latin_page("Digital text that is long enough for extraction.")])
    engine = FakeOcrEngine()

    document = read_document(path, ocr_config=OcrConfig(enabled=True), ocr_engine=engine)

    assert document.kind == "pdf"
    assert document.scanned is False
    assert document.ocr_pages == ()
    assert engine.calls == []


def test_reader_without_ocr_reports_a_scan_actionably(tmp_path):
    path = write_pdf(tmp_path / "scan.pdf", [image_page()])
    with pytest.raises(DocumentReadError) as error:
        read_document(path)
    assert "scanned" in str(error.value).lower()


def test_reader_mixed_pdf_keeps_native_pages_and_marks_ocr(tmp_path):
    path = write_pdf(
        tmp_path / "mixed.pdf",
        [latin_page("Native text on page one."), image_page()],
    )

    document = read_document(
        path, ocr_config=OcrConfig(enabled=True), ocr_engine=FakeOcrEngine()
    )

    assert document.ocr_pages == (2,)
    assert "Native text on page one." in document.text
    assert "OCR text for page 2" in document.text
    assert document.page_numbers == (1, 2)


# --------------------------------------------------------------------------- #
# Connector integration
# --------------------------------------------------------------------------- #


def test_connector_without_ocr_refuses_a_scan(tmp_path):
    path = write_pdf(tmp_path / "scan.pdf", [image_page()])
    with pytest.raises(ConnectorError) as error:
        list(FileConnector().stream(SourceSpec(str(path))))
    assert "ocr" in str(error.value).lower()


def test_connector_with_ocr_emits_labelled_records(tmp_path):
    path = write_pdf(tmp_path / "mixed.pdf", [latin_page("Native page text that is long enough."), image_page()])
    spec = SourceSpec(
        str(path),
        options={
            "allow_ocr": "true",
            "ocr_languages": "ara+eng",
            "ocr_min_chars": "16",
        },
    )
    engine = FakeOcrEngine()

    records = list(FileConnector(ocr_engine=engine).stream(spec))

    kinds = [record.metadata["kind"] for record in records]
    assert kinds == ["pdf", "pdf-ocr"]
    ocr_record = records[1]
    assert ocr_record.metadata["ocr"] is True
    assert ocr_record.metadata["page"] == 2
    assert ocr_record.metadata["ocr_languages"] == ["eng"]
    assert ocr_record.title.endswith("(p. 2)")


# --------------------------------------------------------------------------- #
# CLI: opt-in flag and honest failure
# --------------------------------------------------------------------------- #


def _run(argv):
    from cli import ingest as ingest_cli

    return ingest_cli.main(argv)


def test_cli_scan_without_ocr_reports_actionably(tmp_path, capsys):
    path = write_pdf(tmp_path / "scan.pdf", [image_page()])
    code = _run(["add", str(path), "--store-dir", str(tmp_path / "kb"), "--quiet"])
    assert code == 1
    assert "ocr" in capsys.readouterr().out.lower()


def test_cli_ocr_flag_without_the_extra_reports_the_extra(tmp_path, capsys):
    if ocr_available():
        pytest.skip("the OCR extra is installed; the unavailable path cannot be shown")
    path = write_pdf(tmp_path / "scan.pdf", [image_page()])
    code = _run(
        ["add", str(path), "--store-dir", str(tmp_path / "kb"), "--quiet", "--ocr"]
    )
    assert code == 1
    out = capsys.readouterr().out.lower()
    assert "ocr" in out
    assert "tesseract" in out


# --------------------------------------------------------------------------- #
# The real engine, when the extra and the Tesseract data are installed
# --------------------------------------------------------------------------- #


def _image_only_pdf(path, text: str) -> None:
    """Write a genuine image-only PDF (no text layer) with Pillow."""
    from PIL import Image, ImageDraw

    image = Image.new("RGB", (1200, 320), "white")
    ImageDraw.Draw(image).text((40, 140), text, fill="black")
    image.save(path, "PDF")
    image.close()


@pytest.mark.skipif(not ocr_available(), reason="the [ocr] extra + Tesseract are not installed")
def test_real_tesseract_recovers_english_from_a_scan(tmp_path):
    path = tmp_path / "scan.pdf"
    _image_only_pdf(path, "Permeability measures fluid flow through rock")

    document = read_document(
        path, ocr_config=OcrConfig(enabled=True, languages=("eng",))
    )

    assert document.kind == "pdf-ocr"
    assert document.scanned is True
    assert document.ocr_pages == (1,)
    assert "permeability" in document.text.lower()


@pytest.mark.skipif(not ocr_available(), reason="the [ocr] extra + Tesseract are not installed")
def test_real_engine_reports_its_installed_languages():
    languages = TesseractOcrEngine().available_languages()
    assert "eng" in languages
    assert isinstance(languages, tuple)
