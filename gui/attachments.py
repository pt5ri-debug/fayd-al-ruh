"""Attached files for a chat turn: read a document, index it, and cite it.

When the user attaches a file (a Word ``.docx`` or a ``.txt``) to a chat
message, its text becomes *first-class evidence*: it is added to the same
document store the grounded assistant retrieves from, so an answer can quote it
and cite it exactly like the built-in corpus. This module holds the pieces that
turn a file on disk into that evidence, kept out of the Tk window so the logic is
testable without a display.

The design mirrors the rest of the project:

* :func:`attach_document` reads the file, splits it into retrievable chunks, and
  returns both the :class:`Document` objects to add and a human-readable notice
  for the transcript;
* :func:`remove_documents` takes them back out when the user detaches a file;
* nothing is written to disk here -- persistence, if any, is the caller's choice.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from core.document_reader import DocumentReadError, DocumentText, read_document
from core.documents import Document
from core.knowledge_base import chunk_text

#: Namespace prefix for attached-file documents, so they are easy to spot.
ATTACHED_SOURCE = "attachment"
_ATTACHMENT_PREFIX = "attachment:"

#: An attachment is indexed in chunks of this many characters. Smaller than the
#: bulk corpus default: a chat attachment is quoted in full, and a tighter chunk
#: keeps a single relevant passage from being diluted by the rest of the file.
_CHUNK_CHARS = 900
_CHUNK_OVERLAP = 120


def _slug(text: str) -> str:
    """A short, filesystem-safe token used inside a document id."""
    cleaned = "".join(ch if ch.isalnum() else "-" for ch in text.casefold())
    collapsed = "-".join(part for part in cleaned.split("-") if part)
    return collapsed[:48] or "file"


@dataclass(frozen=True)
class Attachment:
    """One file attached to the conversation, and the documents it produced."""

    filename: str
    title: str
    text: str
    documents: tuple[Document, ...]
    words: int
    chars: int

    @property
    def doc_ids(self) -> tuple[str, ...]:
        return tuple(document.doc_id for document in self.documents)

    @property
    def is_empty(self) -> bool:
        return not self.documents


@dataclass
class AttachmentNotice:
    """A transcript-ready note about what happened to an attached file."""

    filename: str
    ok: bool
    message: str
    preview: str = ""
    words: int = 0
    attachments: list[Attachment] = field(default_factory=list)


def _chunks(document: DocumentText) -> list[str]:
    chunks = chunk_text(
        document.text,
        chunk_chars=_CHUNK_CHARS,
        overlap=_CHUNK_OVERLAP,
    )
    return chunks or [document.text.strip()]


def build_attachment(path: str | Path) -> Attachment:
    """Read ``path`` and turn it into an :class:`Attachment` of citable documents.

    Raises :class:`~core.document_reader.DocumentReadError` when the file cannot
    be read, so the caller can show the message verbatim.
    """
    source = Path(path).expanduser()
    document = read_document(source)
    slug = _slug(source.stem)
    chunks = _chunks(document)
    multiple = len(chunks) > 1
    documents = tuple(
        Document(
            doc_id=f"{_ATTACHMENT_PREFIX}{slug}-{index}" if multiple else f"{_ATTACHMENT_PREFIX}{slug}",
            title=f"{document.title} — part {index + 1}/{len(chunks)}" if multiple else document.title,
            text=chunk,
            source=ATTACHED_SOURCE,
            metadata={
                "kind": "attachment",
                "filename": source.name,
                "part": index + 1,
                "parts": len(chunks),
            },
        )
        for index, chunk in enumerate(chunks)
    )
    return Attachment(
        filename=source.name,
        title=document.title,
        text=document.text,
        documents=documents,
        words=document.word_count,
        chars=len(document.text),
    )


def attach_document(path: str | Path) -> AttachmentNotice:
    """Read a file and return a notice describing the outcome, never raising.

    On success the notice carries the :class:`Attachment` (its documents ready to
    index) and a preview of the opening text for the transcript. On failure it
    carries the reader's message and ``ok=False``.
    """
    source = Path(path).expanduser()
    try:
        attachment = build_attachment(source)
    except DocumentReadError as error:
        return AttachmentNotice(filename=source.name or str(source), ok=False, message=str(error))
    except (OSError, ValueError) as error:
        return AttachmentNotice(
            filename=source.name or str(source), ok=False, message=f"Could not read the file: {error}"
        )
    return AttachmentNotice(
        filename=attachment.filename,
        ok=True,
        message=(
            f"Attached {attachment.filename} — {attachment.words} words, "
            f"{len(attachment.documents)} passage(s) indexed. I can now answer from it and cite it."
        ),
        preview=_preview(attachment.text),
        words=attachment.words,
        attachments=[attachment],
    )


def _preview(text: str, *, limit: int = 600) -> str:
    """The opening of the document, on a word boundary, for a transcript note."""
    flat = " ".join(text.split())
    if len(flat) <= limit:
        return flat
    return flat[:limit].rsplit(" ", 1)[0] + " …"


def remove_documents(retriever, attachment: Attachment) -> int:
    """Detach an attachment from the retriever; returns how many documents left."""
    removed = 0
    for doc_id in attachment.doc_ids:
        if _remove_one(retriever, doc_id):
            removed += 1
    return removed


def _remove_one(retriever, doc_id: str) -> bool:
    """Remove one document from the store and the semantic index behind a retriever."""
    documents = getattr(retriever, "documents", None)
    if documents is None or getattr(documents, "get", None) is None:
        return False
    if documents.get(doc_id) is None:
        return False
    documents.remove(doc_id)
    index = getattr(retriever, "_index", None)
    if index is not None and hasattr(index, "remove"):
        index.remove(doc_id)
    return True


__all__ = [
    "ATTACHED_SOURCE",
    "Attachment",
    "AttachmentNotice",
    "attach_document",
    "build_attachment",
    "remove_documents",
]
