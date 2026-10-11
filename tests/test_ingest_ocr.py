"""The optional OCR path for scanned PDFs.

``pypdf`` and the OCR extra are not installed in the base environment, so these
tests exercise the *reader's* fallback logic directly. ``pypdf`` -- a third-party
parser, not our code -- is replaced by a tiny stub whose pages yield no text,
which is exactly the "scanned image" case. The OCR engine itself is a real
callable passed in, so no part of the project's own logic is mocked.
"""

from __future__ import annotations

import sys
import types
from pathlib import Path

import pytest

from core import document_reader
from core.document_reader import DocumentReadError, _default_ocr_engine, read_document


class _EmptyPage:
    def extract_text(self) -> str:
        return ""


class _StubReader:
    is_encrypted = False
    metadata = None

    def __init__(self, *_args, **_kwargs) -> None:
        self.pages = [_EmptyPage()]

    def get_num_pages(self) -> int:
        return len(self.pages)


@pytest.fixture
def scanned_pdf(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A .pdf path plus a pypdf stub whose pages contain no extractable text."""
    stub = types.ModuleType("pypdf")
    stub.PdfReader = _StubReader  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "pypdf", stub)
    path = tmp_path / "scan.pdf"
    path.write_bytes(b"%PDF-1.4 not really a pdf")
    return path


def test_scanned_pdf_without_engine_reports_the_ocr_extra(scanned_pdf):
    with pytest.raises(DocumentReadError) as error:
        read_document(scanned_pdf, allow_ocr=False)
    assert "scanned" in str(error.value).lower()


def test_scanned_pdf_with_engine_returns_ocr_text(scanned_pdf):
    def fake_engine(path: Path) -> str:
        assert path == scanned_pdf
        return "Hydraulic fracturing\ncreates fractures in rock."

    document = read_document(scanned_pdf, ocr=fake_engine)

    assert document.kind == "pdf-ocr"
    assert "Hydraulic fracturing" in document.text
    assert document.paragraphs[0] == "Hydraulic fracturing"


def test_scanned_pdf_engine_returning_nothing_is_refused(scanned_pdf):
    with pytest.raises(DocumentReadError) as error:
        read_document(scanned_pdf, ocr=lambda _path: "   \n  ")
    assert "OCR" in str(error.value)


def test_scanned_pdf_engine_failure_is_reported_not_fatal(scanned_pdf):
    def boom(_path: Path) -> str:
        raise RuntimeError("tesseract missing")

    with pytest.raises(DocumentReadError) as error:
        read_document(scanned_pdf, ocr=boom)
    assert "OCR" in str(error.value)


def test_default_ocr_engine_is_none_when_extra_absent():
    # Neither pypdfium2 nor pytesseract is installed in the base environment.
    assert _default_ocr_engine() is None


def test_text_pdf_does_not_call_ocr(tmp_path, monkeypatch):
    """A PDF with real text must never reach the OCR engine."""
    calls: list[Path] = []

    class _TextPage:
        def extract_text(self) -> str:
            return "This page already has text."

    class _TextReader:
        is_encrypted = False
        metadata = None

        def __init__(self, *_args, **_kwargs) -> None:
            self.pages = [_TextPage()]

        def get_num_pages(self) -> int:
            return len(self.pages)

    stub = types.ModuleType("pypdf")
    stub.PdfReader = _TextReader  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "pypdf", stub)

    path = tmp_path / "text.pdf"
    path.write_bytes(b"%PDF-1.4")
    document = read_document(path, ocr=lambda p: calls.append(p) or "should not run")

    assert document.kind == "pdf"
    assert calls == []


def test_streaming_and_truncation_flags(tmp_path, monkeypatch):
    monkeypatch.setattr(document_reader, "MAX_TEXT_CHARS", 200)
    monkeypatch.setattr(document_reader, "STREAM_THRESHOLD_BYTES", 10)
    path = tmp_path / "huge.txt"
    path.write_text("\n".join(f"line number {i} with some words" for i in range(5000)), encoding="utf-8")

    document = read_document(path)

    assert document.kind == "text"
    assert document.truncated is True
    assert len(document.text) <= 200


def test_small_text_file_is_not_marked_truncated(tmp_path):
    path = tmp_path / "small.txt"
    path.write_text("one line\n", encoding="utf-8")
    assert read_document(path).truncated is False


def test_normalize_flag_applies_normalization(tmp_path):
    path = tmp_path / "arabic.txt"
    path.write_text("أَلْوَان   \n\n\n  Colors\n", encoding="utf-8")

    verbatim = read_document(path)
    normalized = read_document(path, normalize=True, fold_arabic=True)

    assert verbatim.text != normalized.text
    assert normalized.text == "الوان\nColors"
