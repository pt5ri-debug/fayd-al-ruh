"""Retrieval-augmented generation over documents, ontology, and memory.

Three signals already exist in the project and each sees a different thing:

* lexical/semantic memory (:mod:`core.memory`) matches shared words and hashed
  embeddings;
* ontology retrieval (:mod:`core.ontology_retrieval`) reaches items connected by
  relations even with no shared words;
* a plain document store has the prose that can actually be quoted.

This module fuses them into one ranked list of :class:`RetrievedPassage`, each
carrying its source and a citation string. An answer built from these passages
can therefore cite exactly where every claim came from -- the "grounded,
evidence-aware" property the architecture is built around.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Iterable
from dataclasses import dataclass

from core.documents import Document, DocumentStore
from core.memory import CognitiveMemory
from core.ontology import Ontology
from core.ontology_retrieval import OntologyGuidedRetriever
from core.semantic_memory import SemanticMemoryIndex, hashed_embedding

_WEIGHT_LEXICAL = 0.7
_WEIGHT_SEMANTIC = 0.1
_WEIGHT_ONTOLOGY = 0.2
_TOKEN_RE = re.compile(r"\w+", flags=re.UNICODE)
_DIACRITICS_RE = re.compile(r"[\u0610-\u061a\u064b-\u065f\u0670\u06d6-\u06ed]")
_STOPWORDS = frozenset(
    {
        # English (primary)
        "a", "an", "the", "what", "how", "is", "are", "of", "to", "in", "on",
        "for", "and", "or", "with", "between", "does", "do", "why", "when",
        "which", "that", "this", "these", "those", "it", "its", "as", "by",
        "from", "at", "be", "can", "will", "would", "should", "there", "their",
        "about", "into", "than", "then", "not", "no", "yes", "if", "so", "such",
        # Arabic (retained so bilingual aliases still normalise cleanly)
        "ما", "ماذا", "كيف", "هل", "من", "في", "على", "إلى", "عن", "هو", "هي",
        "التي", "الذي", "و", "أو", "علاقة", "بين", "مع", "ثم", "قد", "لا", "إن",
        "هذا", "هذه", "ذلك", "تلك", "أي", "كل", "بعض",
    }
)
_ALEF = str.maketrans({"أ": "ا", "إ": "ا", "آ": "ا", "ى": "ي", "ة": "ه", "ؤ": "و", "ئ": "ي"})
_PREFIXES = ("وال", "بال", "فال", "كال", "ال", "و", "ب", "ل", "ك", "ف")


def _normalize(token: str) -> str:
    """Light Arabic normalisation: strip diacritics, unify letters, drop affixes."""
    value = _DIACRITICS_RE.sub("", token).replace("ـ", "").translate(_ALEF).casefold()
    changed = True
    while changed and len(value) > 3:
        changed = False
        for prefix in _PREFIXES:
            if value.startswith(prefix) and len(value) - len(prefix) >= 3:
                value = value[len(prefix):]
                changed = True
                break
    return value


def _tokens(text: str) -> set[str]:
    return {_normalize(token) for token in _TOKEN_RE.findall(text)}


def _query_terms(text: str) -> set[str]:
    return _tokens(text) - _STOPWORDS


def _coverage(query: set[str], document: set[str]) -> float:
    """Fraction of the query's content terms the document contains."""
    if not query:
        return 0.0
    return len(query & document) / len(query)


@dataclass(frozen=True)
class RetrievedPassage:
    """A passage selected as evidence, with the signal that found it."""

    doc_id: str
    title: str
    text: str
    source: str
    score: float
    signal: str
    """Which signal dominated: lexical, semantic, or ontology."""

    @property
    def citation(self) -> str:
        return f"[{self.source}:{self.doc_id}] {self.title}"


class BulkCorpusRetriever:
    """Streaming retrieval over a large, on-disk knowledge base.

    Unlike :class:`RagRetriever`, which indexes a small in-memory store, this
    scans a bulk corpus (tens of thousands of chunks) *without* holding it in
    memory: the source is re-streamed per query and only the top matches are
    kept. That keeps a 100 MB+ knowledge base usable on a modest machine at the
    cost of a linear scan per query.
    """

    def __init__(
        self,
        source: Callable[[], Iterable[Document]] | Iterable[Document],
        *,
        dimensions: int = 128,
    ) -> None:
        self._source = source if callable(source) else (lambda: iter(source))
        self.dimensions = dimensions

    def retrieve(self, query: str, *, limit: int = 4) -> list[RetrievedPassage]:
        if not query.strip():
            raise ValueError("Retrieval query cannot be empty.")
        if limit < 1:
            raise ValueError("Retrieval limit must be at least 1.")

        query_terms = _query_terms(query)
        query_vector = hashed_embedding(query, self.dimensions)
        best: list[RetrievedPassage] = []
        for document in self._source():
            text = f"{document.title} {document.text}"
            lexical = _coverage(query_terms, _tokens(text))
            semantic = max(0.0, SemanticMemoryIndex._cosine(query_vector, hashed_embedding(text, self.dimensions)))
            fused = _WEIGHT_LEXICAL * lexical + _WEIGHT_SEMANTIC * semantic
            if fused <= 0.0:
                continue
            signal = "lexical" if lexical >= semantic else "semantic"
            best.append(
                RetrievedPassage(
                    doc_id=document.doc_id,
                    title=document.title,
                    text=document.text,
                    source=document.source,
                    score=round(fused, 6),
                    signal=signal,
                )
            )
            if len(best) > limit * 8:
                best.sort(key=lambda passage: (-passage.score, passage.doc_id))
                del best[limit:]
        best.sort(key=lambda passage: (-passage.score, passage.doc_id))
        return best[:limit]


class RagRetriever:
    """Rank documents for a query by fusing lexical, semantic, and ontology scores."""

    def __init__(
        self,
        documents: DocumentStore,
        *,
        ontology: Ontology | None = None,
        memory: CognitiveMemory | None = None,
        dimensions: int = 128,
        bulk: BulkCorpusRetriever | None = None,
    ) -> None:
        self.documents = documents
        self.ontology = ontology
        self.memory = memory
        self.dimensions = dimensions
        self.bulk = bulk
        self._index = SemanticMemoryIndex(dimensions)
        for document in documents.documents():
            self._index.add(document.doc_id, f"{document.title} {document.text}")
        self._retriever = (
            OntologyGuidedRetriever(ontology) if ontology is not None else None
        )

    def index_document(self, document: Document) -> None:
        # Idempotent: re-indexing a document already in the store refreshes its
        # vector instead of raising (learned statements are re-indexed on every
        # refresh, and some stores call add() themselves).
        if self.documents.get(document.doc_id) is None:
            self.documents.add(document)
        self._index.add(document.doc_id, f"{document.title} {document.text}")
        if self._retriever is not None:
            self._retriever.invalidate()

    def retrieve(self, query: str, *, limit: int = 4) -> list[RetrievedPassage]:
        if not query.strip():
            raise ValueError("Retrieval query cannot be empty.")
        if limit < 1:
            raise ValueError("Retrieval limit must be at least 1.")

        texts = {document.doc_id: f"{document.title} {document.text}" for document in self.documents.documents()}
        query_terms = _query_terms(query)
        semantic = {
            match.item_id: max(0.0, match.score)
            for match in self._index.search(query, limit=max(limit * 4, len(texts)))
        }
        ontology: dict[str, float] = {}
        if self._retriever is not None:
            ontology = {
                match.item_id: match.score
                for match in self._retriever.rank(texts, query, limit=max(limit * 4, len(texts)))
            }

        passages: list[RetrievedPassage] = []
        for document in self.documents.documents():
            lexical = _coverage(query_terms, _tokens(texts[document.doc_id]))
            semantic_score = semantic.get(document.doc_id, 0.0)
            ontology_score = ontology.get(document.doc_id, 0.0)
            fused = (
                _WEIGHT_LEXICAL * lexical
                + _WEIGHT_SEMANTIC * semantic_score
                + _WEIGHT_ONTOLOGY * ontology_score
            )
            if fused <= 0.0:
                continue
            signal = max(
                (("lexical", lexical), ("semantic", semantic_score), ("ontology", ontology_score)),
                key=lambda item: item[1],
            )[0]
            passages.append(
                RetrievedPassage(
                    doc_id=document.doc_id,
                    title=document.title,
                    text=document.text,
                    source=document.source,
                    score=round(fused, 6),
                    signal=signal,
                )
            )

        passages.sort(key=lambda passage: (-passage.score, passage.doc_id))
        passages = passages[:limit]

        # The bulk corpus is optional and separate: it can only *add* evidence
        # for a query the internal store did not answer, and its results never
        # outrank an internal passage on a tie (stable by score then doc_id).
        if self.bulk is not None and len(passages) < limit:
            seen = {passage.doc_id for passage in passages}
            for extra in self.bulk.retrieve(query, limit=limit - len(passages)):
                if extra.doc_id not in seen:
                    passages.append(extra)
                    seen.add(extra.doc_id)
        return passages[:limit]


__all__ = ["BulkCorpusRetriever", "RagRetriever", "RetrievedPassage"]
