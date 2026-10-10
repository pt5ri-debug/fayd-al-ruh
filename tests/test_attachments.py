"""Attaching a document makes it citable evidence; detaching takes it back out."""

from __future__ import annotations

import zipfile
from pathlib import Path

from core.assistant import build_default_assistant
from core.document_reader import DocumentReadError
from gui.attachments import (
    ATTACHED_SOURCE,
    attach_document,
    build_attachment,
    remove_documents,
)

_DOCX_HEAD = (
    '<?xml version="1.0"?>'
    '<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">'
)


def _write_docx(path: Path, lines: list[str]) -> Path:
    body = "".join(f"<w:p><w:r><w:t>{line}</w:t></w:r></w:p>" for line in lines)
    xml = f"{_DOCX_HEAD}<w:body>{body}</w:body></w:document>"
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("word/document.xml", xml)
    return path


def test_build_attachment_chunks_and_tags_documents(tmp_path):
    path = _write_docx(
        tmp_path / "reservoir.docx",
        ["Reservoir rock properties", "Porosity is the fraction of pore space."],
    )

    attachment = build_attachment(path)

    assert attachment.filename == "reservoir.docx"
    assert attachment.title == "reservoir"
    assert attachment.words >= 6
    assert attachment.documents
    for document in attachment.documents:
        assert document.source == ATTACHED_SOURCE
        assert document.doc_id.startswith("attachment:reservoir")


def test_attach_document_reports_success_with_a_preview(tmp_path):
    path = _write_docx(tmp_path / "notes.docx", ["Permeability is measured in darcy."])

    notice = attach_document(path)

    assert notice.ok is True
    assert "darcy" in notice.preview
    assert notice.attachments


def test_attach_document_reports_failure_without_raising(tmp_path):
    notice = attach_document(tmp_path / "missing.docx")

    assert notice.ok is False
    assert "missing.docx" in notice.message
    assert notice.attachments == []


def test_legacy_doc_reports_the_conversion_hint(tmp_path):
    path = tmp_path / "old.doc"
    path.write_bytes(b"\xd0\xcf\x11\xe0")

    notice = attach_document(path)

    assert notice.ok is False
    assert ".docx" in notice.message


def test_attached_document_becomes_citable_evidence(tmp_path):
    assistant = build_default_assistant()
    retriever = assistant.retriever
    path = _write_docx(
        tmp_path / "well.docx",
        [
            "The Zephyrium-9 well flowed at 4200 barrels of oil per day.",
            "Its bottomhole pressure was 5120 psi after a week of production.",
        ],
    )
    attachment = build_attachment(path)
    for document in attachment.documents:
        retriever.index_document(document)

    answer = assistant.ask("What did the Zephyrium-9 well flow at?")

    assert "Zephyrium-9" in answer.text or "4200" in answer.text
    assert any("attachment" in citation for citation in answer.citations)


def test_remove_documents_takes_the_evidence_back_out(tmp_path):
    assistant = build_default_assistant()
    retriever = assistant.retriever
    path = _write_docx(tmp_path / "well.docx", ["Zephyrium-9 flowed at 4200 barrels per day."])
    attachment = build_attachment(path)
    for document in attachment.documents:
        retriever.index_document(document)
    assert retriever.documents.get(attachment.doc_ids[0]) is not None

    removed = remove_documents(retriever, attachment)

    assert removed == len(attachment.doc_ids)
    assert retriever.documents.get(attachment.doc_ids[0]) is None


def test_build_attachment_raises_for_unreadable_file(tmp_path):
    import pytest

    with pytest.raises(DocumentReadError):
        build_attachment(tmp_path / "nothing.docx")
