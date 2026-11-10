"""The Workplace document registry: the files a project owns (the brief's §9).

This is deliberately a **registry**, not a document store. It records a
document's metadata -- type, title, revision, confidentiality, author, issue date,
and a **content hash** -- and it keeps the source file the Workplace itself copied
in (under the git-ignored ``data/workplace/documents``), so a project's repository
is self-contained and a later revision can be compared against the original. It
never re-implements the reader: the text of a document is read through the
project's *existing* :func:`core.document_reader.read_document` (Word, PDF, text,
CSV/JSON, and the optional OCR path), and the file is chunked with the project's
*existing* :func:`core.knowledge_base.chunk_text` when a caller wants it indexed.

The three disciplines the rest of the Workplace uses apply here too:

* a **revision is a new row** (``revision`` + ``supersedes_id``), so the original is
  never mutated and :meth:`DocumentManager.history` can walk the chain;
* a document's **integrity is checked**, not assumed -- :meth:`DocumentManager.verify`
  re-hashes the stored file and reports whether it still matches the recorded hash;
* a **duplicate is reported**, not silently accepted: registering a file whose
  content the registry already holds returns a record naming the copy it matched.

Writes go through the audited :class:`~core.workplace.domain.WorkplaceDomain`, so a
registration is recorded in the audit trail like every other Workplace change.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from hashlib import blake2b
from pathlib import Path

from core.document_reader import DocumentText, read_document
from core.paths import workplace_documents_dir
from core.workplace.domain import WorkplaceDomain
from core.workplace.records import DOCUMENT_TYPES, WorkplaceDocument
from core.workplace.repository import WorkplaceRepository

#: The readable label of each document type (English + Arabic).
DOCUMENT_TYPE_LABELS: dict[str, tuple[str, str]] = {
    "proposal": ("Technical / commercial proposal", "عرض فني / تجاري"),
    "inspection_report": ("Inspection report", "تقرير فحص"),
    "technical_report": ("Technical report", "تقرير فني"),
    "method_statement": ("Method statement", "بيان طريقة العمل"),
    "itp": ("Inspection and test plan", "خطة الفحص والاختبار"),
    "welding_document": ("Welding documentation", "وثائق اللحام"),
    "coating_procedure": ("Coating procedure", "إجراء الدهان"),
    "pqt": ("PQT / qualification record", "سجل تأهيل الإجراءات"),
    "ncr": ("Nonconformance report (NCR)", "تقرير عدم مطابقة"),
    "car": ("Corrective action request (CAR)", "طلب إجراء تصحيحي"),
    "correspondence": ("Technical correspondence", "مراسلات فنية"),
    "minutes": ("Meeting minutes", "محضر اجتماع"),
    "contract": ("Contractual document", "وثيقة تعاقدية"),
    "other": ("Other document", "وثيقة أخرى"),
}


def document_type_label(doc_type: str, *, language: str = "en") -> str:
    """The readable label of a document type (its own key when unknown)."""
    labels = DOCUMENT_TYPE_LABELS.get(doc_type)
    if labels is None:
        return doc_type
    return labels[1] if language == "ar" else labels[0]


def content_hash(data: bytes | str) -> str:
    """A stable content digest (``blake2b``, standard library) for a file or text."""
    payload = data.encode("utf-8") if isinstance(data, str) else data
    return blake2b(payload, digest_size=16).hexdigest()


def _slug(text: str) -> str:
    cleaned = "".join(ch if ch.isalnum() else "-" for ch in text.casefold())
    return "-".join(part for part in cleaned.split("-") if part)[:48] or "doc"


@dataclass(frozen=True)
class DocumentRecord:
    """One document row, with the readable facts a surface needs."""

    record_id: str
    doc_type: str
    title: str
    title_ar: str
    path: str
    content_hash: str
    revision: str
    supersedes_id: str
    confidentiality: str
    project_id: str
    division_id: str
    author_id: str
    approved_by: str
    issued_on: str
    active: bool
    metadata: dict

    @property
    def filename(self) -> str:
        return Path(self.path).name if self.path else ""

    @property
    def type_label(self) -> str:
        return document_type_label(self.doc_type)

    def describe(self, *, language: str = "en") -> str:
        head = self.title or self.filename or self.record_id
        parts = [head, f"[{document_type_label(self.doc_type, language=language)}]"]
        if self.revision:
            parts.append(f"rev {self.revision}")
        if self.confidentiality and self.confidentiality != "internal":
            parts.append(f"({self.confidentiality})")
        return " ".join(parts)

    @classmethod
    def from_record(cls, record: WorkplaceDocument) -> DocumentRecord:
        return cls(
            record_id=record.record_id,
            doc_type=record.doc_type,
            title=record.title,
            title_ar=record.title_ar,
            path=record.path,
            content_hash=record.content_hash,
            revision=record.revision,
            supersedes_id=record.supersedes_id,
            confidentiality=record.confidentiality,
            project_id=record.project_id,
            division_id=record.division_id,
            author_id=record.author_id,
            approved_by=record.approved_by,
            issued_on=record.issued_on,
            active=record.active,
            metadata=dict(record.metadata),
        )


@dataclass(frozen=True)
class DocumentVerification:
    """The result of re-hashing a stored document (the brief's §12 traceability)."""

    record_id: str
    ok: bool
    recorded_hash: str
    actual_hash: str
    message: str

    def describe(self) -> str:
        state = "matches" if self.ok else "MISMATCH"
        return f"{self.record_id}: {state} (recorded {self.recorded_hash[:12]}, actual {self.actual_hash[:12]}) — {self.message}"


class DocumentManager:
    """Register, read, list, verify, and revise a project's documents.

    ``reader`` is the project's own document reader (injectable so a test can
    exercise the registry with no optional dependency); ``documents_dir`` is where
    the Workplace keeps the copies it makes.
    """

    def __init__(
        self,
        repository: WorkplaceRepository,
        domain: WorkplaceDomain | None = None,
        *,
        documents_dir: Path | None = None,
        reader: Callable[..., DocumentText] = read_document,
    ) -> None:
        self.repository = repository
        self.domain = domain
        self.documents_dir = Path(documents_dir) if documents_dir is not None else workplace_documents_dir()
        self._reader = reader

    # -- storage ----------------------------------------------------------

    def _store_bytes(self, data: bytes, *, suffix: str) -> Path:
        """Copy bytes into the Workplace's document folder, named by content hash.

        Content-addressed on purpose: the same bytes are stored once, so a revision
        that did not change the file does not duplicate it, and the file's name
        itself is a check on what it holds.
        """
        digest = content_hash(data)
        folder = self.documents_dir
        folder.mkdir(parents=True, exist_ok=True)
        target = folder / f"{digest}{suffix}"
        if not target.exists():
            target.write_bytes(data)
        return target

    def _store_file(self, source: Path) -> tuple[Path, str]:
        data = source.read_bytes()
        stored = self._store_bytes(data, suffix=source.suffix.lower() or ".dat")
        return stored, content_hash(data)

    # -- write ------------------------------------------------------------

    def register_file(
        self,
        source: str | Path,
        *,
        title: str = "",
        title_ar: str = "",
        project_id: str = "",
        division_id: str = "",
        doc_type: str = "other",
        revision: str = "",
        supersedes_id: str = "",
        confidentiality: str = "internal",
        author_id: str = "",
        issued_on: str = "",
    ) -> DocumentRecord:
        """Copy a file in and register it (a metadata row + the stored copy)."""
        if self.domain is None:
            raise RuntimeError("This registry is read-only (no domain service supplied).")
        path = Path(source).expanduser()
        if not path.exists() or not path.is_file():
            raise ValueError(f"No such file: {path}")
        if doc_type not in DOCUMENT_TYPES:
            raise ValueError(
                f"Unknown document type {doc_type!r}; expected one of "
                f"{', '.join(DOCUMENT_TYPES)}."
            )
        stored, digest = self._store_file(path)
        record = self.domain.register_document(
            title=title.strip() or path.stem,
            title_ar=title_ar.strip(),
            project_id=project_id,
            division_id=division_id,
            doc_type=doc_type,
            path=str(stored),
            content_hash=digest,
            revision=revision.strip(),
            supersedes_id=supersedes_id,
            confidentiality=confidentiality,
            author_id=author_id,
            issued_on=issued_on,
            metadata={"source": "workplace", "original_name": path.name},
        )
        return DocumentRecord.from_record(record)

    def register_text(
        self,
        text: str,
        *,
        title: str,
        doc_type: str = "other",
        **fields: object,
    ) -> DocumentRecord:
        """Register a document whose text is supplied directly (no source file).

        The text is written to a ``.txt`` file in the Workplace's folder so the
        registry stays uniform: every document has a file the project's own reader
        can read, and no document's text lives in a second place.
        """
        if not text.strip():
            raise ValueError("A document needs some text.")
        data = text.encode("utf-8")
        stored = self._store_bytes(data, suffix=".txt")
        # ``register_file`` would re-copy; record the already-stored path directly.
        if self.domain is None:
            raise RuntimeError("This registry is read-only (no domain service supplied).")
        if doc_type not in DOCUMENT_TYPES:
            raise ValueError(
                f"Unknown document type {doc_type!r}; expected one of "
                f"{', '.join(DOCUMENT_TYPES)}."
            )
        record = self.domain.register_document(
            title=title.strip(),
            doc_type=doc_type,
            path=str(stored),
            content_hash=content_hash(data),
            **fields,
        )
        return DocumentRecord.from_record(record)

    def record_revision(
        self,
        record_id: str,
        *,
        revision: str = "",
        title: str = "",
        issued_on: str = "",
    ) -> DocumentRecord:
        """Record a *new* revision row that supersedes ``record_id``.

        The file is shared (the new row points at the same stored bytes) unless the
        caller re-registers a changed file through :meth:`register_file` with
        ``supersedes_id`` set. This keeps a revision chain explicit while never
        mutating the original row.
        """
        if self.domain is None:
            raise RuntimeError("This registry is read-only (no domain service supplied).")
        current = self.repository.get_domain("workplace_documents", record_id)
        if not isinstance(current, WorkplaceDocument):
            raise ValueError(f"No document with id {record_id!r}.")
        record = self.domain.register_document(
            title=title.strip() or current.title,
            title_ar=current.title_ar,
            project_id=current.project_id,
            division_id=current.division_id,
            doc_type=current.doc_type,
            path=current.path,
            content_hash=current.content_hash,
            revision=revision.strip() or current.revision,
            supersedes_id=record_id,
            confidentiality=current.confidentiality,
            author_id=current.author_id,
            issued_on=issued_on or current.issued_on,
            metadata=dict(current.metadata),
        )
        return DocumentRecord.from_record(record)

    # -- read -------------------------------------------------------------

    def document(self, record_id: str) -> DocumentRecord | None:
        record = self.repository.get_domain("workplace_documents", record_id)
        return DocumentRecord.from_record(record) if isinstance(record, WorkplaceDocument) else None

    def documents(
        self,
        *,
        project_id: str | None = None,
        doc_type: str | None = None,
        query: str | None = None,
        include_inactive: bool = False,
    ) -> list[DocumentRecord]:
        """The registry, optionally narrowed by project, type, or a free-text query."""
        rows = self.repository.list_domain(
            "workplace_documents", include_inactive=include_inactive
        )
        records = [row for row in rows if isinstance(row, WorkplaceDocument)]
        if project_id is not None:
            records = [r for r in records if r.project_id == project_id]
        if doc_type is not None:
            records = [r for r in records if r.doc_type == doc_type]
        if query:
            needle = " ".join(query.casefold().split())
            records = [
                r for r in records
                if needle in " ".join(
                    (r.title, r.title_ar, r.revision, r.doc_type, Path(r.path).name)
                ).casefold()
            ]
        records.sort(key=lambda r: (r.issued_on, r.title), reverse=True)
        return [DocumentRecord.from_record(r) for r in records]

    def history(self, record_id: str) -> list[DocumentRecord]:
        """A document's revision chain: the record and everything it superseded."""
        chain = self.domain.document_history(record_id) if self.domain else []
        if not chain:
            record = self.repository.get_domain("workplace_documents", record_id)
            chain = [record] if isinstance(record, WorkplaceDocument) else []
        return [DocumentRecord.from_record(r) for r in chain]

    def read_text(self, record_id: str) -> DocumentText:
        """Read a document's text through the project's *existing* reader."""
        record = self.repository.get_domain("workplace_documents", record_id)
        if not isinstance(record, WorkplaceDocument):
            raise ValueError(f"No document with id {record_id!r}.")
        if not record.path:
            raise ValueError(f"Document {record_id!r} has no stored file.")
        return self._reader(record.path)

    def chunks(self, record_id: str, *, chunk_chars: int = 900, overlap: int = 120) -> list[str]:
        """The document's text as retrievable chunks (the project's own chunker)."""
        from core.knowledge_base import chunk_text

        document = self.read_text(record_id)
        return chunk_text(document.text, chunk_chars=chunk_chars, overlap=overlap)

    def verify(self, record_id: str) -> DocumentVerification:
        """Re-hash the stored file and report whether it still matches the record."""
        record = self.repository.get_domain("workplace_documents", record_id)
        if not isinstance(record, WorkplaceDocument):
            raise ValueError(f"No document with id {record_id!r}.")
        path = Path(record.path)
        if not record.path or not path.exists():
            return DocumentVerification(
                record_id=record_id, ok=False, recorded_hash=record.content_hash,
                actual_hash="", message=f"the stored file is missing ({record.path or 'no path'})",
            )
        actual = content_hash(path.read_bytes())
        ok = actual == record.content_hash
        return DocumentVerification(
            record_id=record_id, ok=ok, recorded_hash=record.content_hash, actual_hash=actual,
            message="the file is unchanged" if ok else "the file has changed since it was registered",
        )

    def find_by_hash(self, digest: str) -> list[DocumentRecord]:
        """Every document whose content hash matches (a duplicate check)."""
        rows = self.repository.list_domain("workplace_documents", include_inactive=True)
        return [
            DocumentRecord.from_record(row)
            for row in rows
            if isinstance(row, WorkplaceDocument) and row.content_hash == digest
        ]

    def types(self, *, language: str = "en") -> list[tuple[str, str]]:
        """Every document type with its readable label, for a picker."""
        return [(key, document_type_label(key, language=language)) for key in DOCUMENT_TYPES]

    def deactivate(self, record_id: str) -> DocumentRecord:
        """Withdraw a document (the soft delete); its history survives."""
        if self.domain is None:
            raise RuntimeError("This registry is read-only (no domain service supplied).")
        updated = self.domain.deactivate("workplace_documents", record_id)
        return DocumentRecord.from_record(updated)


__all__ = [
    "DOCUMENT_TYPE_LABELS",
    "DocumentManager",
    "DocumentRecord",
    "DocumentVerification",
    "content_hash",
    "document_type_label",
]
