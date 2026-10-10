"""Reading text out of Word .docx files without any third-party dependency."""

from __future__ import annotations

import zipfile
from pathlib import Path

import pytest

from core.document_reader import (
    DocumentReadError,
    read_document,
)

_DOCUMENT_XML = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">
  <w:body>
    <w:p><w:r><w:t>Reservoir porosity</w:t></w:r></w:p>
    <w:p><w:r><w:t>Porosity is the fraction of pore </w:t></w:r><w:r><w:t>space.</w:t></w:r></w:p>
    <w:p><w:r><w:t>Permeability</w:t></w:r><w:tab/><w:r><w:t>measured in darcy</w:t></w:r></w:p>
    <w:p/>
  </w:body>
</w:document>
"""


def _write_docx(path: Path, xml: str = _DOCUMENT_XML) -> Path:
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("word/document.xml", xml)
    return path


def test_reads_paragraphs_and_joins_runs(tmp_path):
    path = _write_docx(tmp_path / "notes.docx")

    document = read_document(path)

    assert document.kind == "docx"
    assert document.title == "notes"
    assert document.paragraphs[0] == "Reservoir porosity"
    # Two runs in one paragraph are joined into one line.
    assert document.paragraphs[1] == "Porosity is the fraction of pore space."
    # A tab is preserved between runs.
    assert "Permeability\tmeasured in darcy" in document.paragraphs[2]
    # The empty paragraph is dropped.
    assert len(document.paragraphs) == 3
    assert document.word_count == 13


def test_reads_plain_text(tmp_path):
    path = tmp_path / "notes.txt"
    path.write_text("First line\n\nSecond line\n", encoding="utf-8")

    document = read_document(path)

    assert document.kind == "text"
    assert document.paragraphs == ("First line", "Second line")


def test_missing_file_is_refused(tmp_path):
    with pytest.raises(DocumentReadError):
        read_document(tmp_path / "nope.docx")


def test_legacy_doc_is_refused_with_a_hint(tmp_path):
    path = tmp_path / "old.doc"
    path.write_bytes(b"\xd0\xcf\x11\xe0not really a doc")

    with pytest.raises(DocumentReadError) as error:
        read_document(path)

    assert ".docx" in str(error.value)


def test_unsupported_extension_is_refused(tmp_path):
    path = tmp_path / "image.png"
    path.write_bytes(b"\x89PNG")

    with pytest.raises(DocumentReadError):
        read_document(path)


def test_corrupt_zip_is_refused(tmp_path):
    path = tmp_path / "broken.docx"
    path.write_bytes(b"this is not a zip archive")

    with pytest.raises(DocumentReadError):
        read_document(path)


def test_docx_without_text_is_refused(tmp_path):
    # A body whose only paragraph has no runs holds no readable content.
    empty = """<?xml version="1.0"?>
    <w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">
      <w:body><w:p><w:r></w:r></w:p></w:body>
    </w:document>"""
    path = _write_docx(tmp_path / "empty.docx", empty)

    with pytest.raises(DocumentReadError):
        read_document(path)
