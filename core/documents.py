"""Documents and a document store for retrieval-augmented answers.

The project's knowledge lives in an ontology and a curriculum, both of which are
declarative seeds. Retrieval needs *prose* to quote, not just triples, so this
module turns the seeds into citable documents and provides a store for any
additional documents a caller adds.

A :class:`Document` is deliberately plain: an id, a title, a body, a source, and
free metadata. Documents are the unit of citation, so every retrieved answer can
point back to the exact text it used.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable

from core.curriculum_knowledge import courses
from core.domain_knowledge import disciplines
from core.petroleum_corpus import corpus_documents
from storage.json_store import JsonStore

_NAMESPACE = "documents"


@dataclass(frozen=True)
class Document:
    """A citable unit of text."""

    doc_id: str
    title: str
    text: str
    source: str = "internal"
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.doc_id.strip():
            raise ValueError("Document id cannot be empty.")
        if not self.title.strip():
            raise ValueError("Document title cannot be empty.")
        if not self.text.strip():
            raise ValueError("Document text cannot be empty.")

    def to_primitive(self) -> dict[str, Any]:
        return {
            "doc_id": self.doc_id,
            "title": self.title,
            "text": self.text,
            "source": self.source,
            "metadata": self.metadata,
        }

    @classmethod
    def from_primitive(cls, data: dict[str, Any]) -> "Document":
        return cls(
            doc_id=str(data["doc_id"]),
            title=str(data["title"]),
            text=str(data["text"]),
            source=str(data.get("source", "internal")),
            metadata=dict(data.get("metadata", {})),
        )


class DocumentStore:
    """An in-memory store of documents with optional JSON persistence."""

    def __init__(self, store: JsonStore | None = None) -> None:
        self._documents: dict[str, Document] = {}
        self._store = store

    def add(self, document: Document) -> Document:
        if document.doc_id in self._documents:
            raise ValueError(f"Document '{document.doc_id}' already exists.")
        self._documents[document.doc_id] = document
        return document

    def add_text(
        self,
        doc_id: str,
        title: str,
        text: str,
        *,
        source: str = "internal",
        metadata: dict[str, Any] | None = None,
    ) -> Document:
        return self.add(Document(doc_id, title, text, source, metadata or {}))

    def get(self, doc_id: str) -> Document | None:
        return self._documents.get(doc_id)

    def remove(self, doc_id: str) -> Document | None:
        """Remove a document from memory; returns it when present."""
        return self._documents.pop(doc_id, None)

    def documents(self) -> tuple[Document, ...]:
        return tuple(self._documents.values())

    def __len__(self) -> int:
        return len(self._documents)

    def save(self, document: Document) -> None:
        if self._store is None:
            return
        self._documents.setdefault(document.doc_id, document)
        self._store.save(_NAMESPACE, document.doc_id, document.to_primitive())

    def load(self, doc_id: str) -> Document | None:
        if doc_id in self._documents:
            return self._documents[doc_id]
        if self._store is None:
            return None
        record = self._store.load(_NAMESPACE, doc_id)
        if record is None:
            return None
        document = Document.from_primitive(record)
        self._documents[document.doc_id] = document
        return document

    def load_all(self) -> int:
        """Load every persisted document; returns how many were loaded."""
        if self._store is None:
            return 0
        count = 0
        for doc_id in self._store.list_keys(_NAMESPACE):
            if self.load(doc_id) is not None:
                count += 1
        return count


def _discipline_document(discipline) -> Document:
    lines = [f"Discipline {discipline.label}."]
    for label, aliases in discipline.concepts:
        aliases_text = f" (also known as {', '.join(aliases)})" if aliases else ""
        lines.append(f"Concept {label}{aliases_text}.")
    for source, target, relation, _confidence in discipline.relations:
        lines.append(f"{source} {relation} {target}.")
    return Document(
        doc_id=f"discipline:{discipline.key}",
        title=f"Discipline: {discipline.label}",
        text="\n".join(lines),
        source="domain_knowledge",
        metadata={"kind": "discipline", "key": discipline.key},
    )


def _course_document(course) -> Document:
    lines = [f"Course {course.label} in {course.category}."]
    for label, aliases in course.concepts:
        aliases_text = f" (also known as {', '.join(aliases)})" if aliases else ""
        lines.append(f"Concept {label}{aliases_text}.")
    for source, target, relation, _confidence in course.relations:
        lines.append(f"{source} {relation} {target}.")
    return Document(
        doc_id=f"course:{course.key}",
        title=f"Course: {course.label}",
        text="\n".join(lines),
        source="curriculum_knowledge",
        metadata={"kind": "course", "key": course.key, "category": course.category},
    )


def build_knowledge_documents() -> list[Document]:
    """Turn the seeded disciplines, courses, and corpus into citable documents."""
    documents = [_discipline_document(discipline) for discipline in disciplines()]
    documents.extend(_course_document(course) for course in courses())
    documents.extend(corpus_documents())
    return documents


def build_document_store(
    extra: Iterable[Document] = (),
    *,
    store: JsonStore | None = None,
    include_knowledge: bool = True,
) -> DocumentStore:
    """A store preloaded with the knowledge corpus (and any extra documents)."""
    document_store = DocumentStore(store=store)
    if include_knowledge:
        for document in build_knowledge_documents():
            document_store.add(document)
    for document in extra:
        document_store.add(document)
    return document_store


__all__ = [
    "Document",
    "DocumentStore",
    "build_document_store",
    "build_knowledge_documents",
]
