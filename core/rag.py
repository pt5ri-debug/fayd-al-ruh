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

from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import Any

from core.documents import Document, DocumentStore
from core.memory import CognitiveMemory
from core.ontology import Ontology
from core.ontology_retrieval import OntologyGuidedRetriever
from core.semantic_memory import SemanticMemoryIndex, hashed_embedding
from core.tokenize import content_terms as _query_terms
from core.tokenize import tokenize as _tokens

_WEIGHT_LEXICAL = 0.7
_WEIGHT_SEMANTIC = 0.1
_WEIGHT_ONTOLOGY = 0.2
# The learned semantic space replaces the hash-based signal *in the same slot* and
# at the same weight. Raising its weight was tried and reverted: the distributional
# prior is a real improvement to *ranking* (a paraphrase with no shared word can now
# be found), but it is a prior about relatedness, not evidence that the corpus
# actually answers the question. Giving it more weight inflated every in-domain
# score past the weak-evidence threshold, so the assistant stopped consulting the
# web for the questions it should -- the calibrated scale is what keeps that gate
# meaningful, so the signal improves without moving the scale.
# The bulk corpus is BM25-calibrated, so its scores are rescaled before they are
# mixed with the internal (coverage-based) scores. Without a real index the bulk
# retriever's coverage scores are on the same scale as lexical ones and used as is.
_BULK_FUSION_WEIGHT = 0.6


def _load_embeddings():
    """Load the project's learned semantic space, or ``None`` if it cannot be built.

    Lazy and total: a caller that has never trained a space still gets a working
    retriever (the hashed signal backs it), and a broken cache file degrades to
    the same fallback rather than failing a query.
    """
    try:
        from core.embeddings import semantic_space

        return semantic_space()
    except Exception:  # pragma: no cover - a defensive fallback, never the happy path
        return None


def _coverage(query: set[str], document: set[str] | frozenset[str]) -> float:
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
        index: Any | None = None,
    ) -> None:
        self._source = source if callable(source) else (lambda: iter(source))
        self.dimensions = dimensions
        # ``index`` is a KnowledgeTextIndex (or any object with ``exists`` and
        # ``search(query, limit=...) -> [(score, chunk)]``). When present, a
        # query is served from the FTS5 index in milliseconds instead of
        # re-tokenizing the whole corpus.
        self.index = index

    @property
    def indexed(self) -> bool:
        """Whether a built FTS5 index backs this retriever."""
        return self.index is not None and self.index.exists()

    def retrieve(self, query: str, *, limit: int = 4) -> list[RetrievedPassage]:
        if not query.strip():
            raise ValueError("Retrieval query cannot be empty.")
        if limit < 1:
            raise ValueError("Retrieval limit must be at least 1.")

        if self.index is not None and self.index.exists():
            passages = [
                RetrievedPassage(
                    doc_id=chunk.chunk_id,
                    title=chunk.title,
                    text=chunk.text,
                    source=chunk.source,
                    score=score,
                    signal="lexical",
                )
                for score, chunk in self.index.search(query, limit=limit)
            ]
            if passages:
                return passages

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
    """Rank documents for a query by fusing lexical, semantic, and ontology scores.

    The semantic signal is the project's **learned** distributional space
    (:mod:`core.embeddings`) when one is available -- so a query about "flow
    through rock" can reach a passage about permeability even with no shared
    word -- and the older hashed fingerprint otherwise. The learned space is
    loaded lazily and cached, so a caller that never needs it never pays for it.
    """

    def __init__(
        self,
        documents: DocumentStore,
        *,
        ontology: Ontology | None = None,
        memory: CognitiveMemory | None = None,
        dimensions: int = 128,
        bulk: BulkCorpusRetriever | None = None,
        embeddings: bool = True,
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
        self._embeddings = _load_embeddings() if embeddings else None
        # Per-document derived data, keyed by doc_id and validated by its own text.
        # A retrieval needs each document's full text, its content terms, and its
        # vector; none of those depend on the *query*, so recomputing them for every
        # question was pure waste -- and the vectors were the largest remaining cost
        # of a retrieval after the ontology lookup was indexed. The cache is **not**
        # cleared wholesale when the store changes: each entry is checked against the
        # document's current text, so a changed or re-added document recomputes while
        # the hundreds of untouched ones keep their vectors (a blanket invalidation
        # re-embedded the whole store every time one document was learned).
        self._cache: dict[str, tuple[str, frozenset[str], tuple[float, ...] | None]] = {}

    @property
    def embeddings_available(self) -> bool:
        """Whether the learned semantic space is backing this retriever."""
        return self._embeddings is not None

    def _semantic_scores(
        self, query: str, texts: dict[str, str], *, query_vector: tuple[float, ...] | None = None
    ) -> dict[str, float]:
        """The semantic score of every document: learned cosine, else hashed.

        The learned space places a whole document by the *mean* of its known term
        vectors, so two passages about the same thing score alike even when they
        share no word. The hashed index is the fallback and its scores are used
        unchanged.

        The query is embedded **once** and every document's vector is taken from the
        per-document cache, so the cosine is the only per-document work; the old
        form re-embedded the same query for every document.
        """
        if self._embeddings is None:
            return {
                match.item_id: max(0.0, match.score)
                for match in self._index.search(query, limit=max(4, len(texts)))
            }
        from core.embeddings import dot_product

        query_vector = query_vector if query_vector is not None else self._embeddings.embed(query)
        if not any(query_vector):
            return {}
        scores: dict[str, float] = {}
        for doc_id, text in texts.items():
            vector = self._document_vector(doc_id, text)
            if vector is None or not any(vector):
                continue
            # Both vectors come from ``DistributionalSpace.embed``, which returns a
            # unit vector, so the dot product *is* the cosine -- and it skips the two
            # norm passes the general form would redo for every document.
            scores[doc_id] = dot_product(query_vector, vector)
        return scores

    def _document_vector(self, doc_id: str, text: str) -> tuple[float, ...] | None:
        """A document's learned vector, from the cache (``None`` when embeddings are off)."""
        if self._embeddings is None:
            return None
        return self._document_data(doc_id, text)[2]

    def _document_data(
        self, doc_id: str, text: str
    ) -> tuple[str, frozenset[str], tuple[float, ...] | None]:
        """A document's full text, its content terms, and its vector, all cached.

        A document's text and its derived terms and vector do not depend on the
        query, so they are computed once and reused for as long as the document's
        text is unchanged. The entry stores its own text, so a *changed* document is
        recomputed without throwing away every other document's work.
        """
        cached = self._cache.get(doc_id)
        if cached is not None and cached[0] == text:
            return cached
        terms = frozenset(_tokens(text))
        vector = self._embeddings.embed(text) if self._embeddings is not None else None
        entry = (text, terms, vector)
        self._cache[doc_id] = entry
        return entry

    def index_document(self, document: Document) -> None:
        # Idempotent: re-indexing a document already in the store refreshes its
        # vector instead of raising (learned statements are re-indexed on every
        # refresh, and some stores call add() themselves).
        if self.documents.get(document.doc_id) is None:
            self.documents.add(document)
        self._index.add(document.doc_id, f"{document.title} {document.text}")
        self._cache.pop(document.doc_id, None)
        if self._retriever is not None:
            self._retriever.invalidate()

    def remove_document(self, doc_id: str) -> bool:
        """Drop a document from the store and the semantic index; the inverse of
        :meth:`index_document`. Returns whether it was present."""
        if self.documents.get(doc_id) is None:
            return False
        self.documents.remove(doc_id)
        self._index.remove(doc_id)
        self._cache.pop(doc_id, None)
        if self._retriever is not None:
            self._retriever.invalidate()
        return True

    def invalidate(self) -> None:
        """Drop cached ontology-guided rankings after the graph changed.

        Called when the ontology gains a learned relation outside this retriever
        (e.g. a relation replayed from a previous session by
        :mod:`core.knowledge_carry`), so the next query sees the new link.
        """
        if self._retriever is not None:
            self._retriever.invalidate()

    def retrieve(self, query: str, *, limit: int = 4) -> list[RetrievedPassage]:
        if not query.strip():
            raise ValueError("Retrieval query cannot be empty.")
        if limit < 1:
            raise ValueError("Retrieval limit must be at least 1.")

        query_terms = _query_terms(query)
        # One pass over the store: the document's text, its content terms, and its
        # learned vector are all taken from the cache, so a question that repeats
        # costs almost nothing.
        entries: list[tuple[Document, str, frozenset[str], tuple[float, ...] | None]] = []
        for document in self.documents.documents():
            text, terms, vector = self._document_data(
                document.doc_id, f"{document.title} {document.text}"
            )
            entries.append((document, text, terms, vector))
        texts = {document.doc_id: text for document, text, _terms, _vector in entries}
        query_vector = (
            self._embeddings.embed(query) if self._embeddings is not None else None
        )
        semantic = self._semantic_scores(query, texts, query_vector=query_vector)
        ontology: dict[str, float] = {}
        if self._retriever is not None:
            ontology = {
                match.item_id: match.score
                for match in self._retriever.rank(texts, query, limit=max(limit * 4, len(texts)))
            }

        passages: list[RetrievedPassage] = []
        for document, _text, document_terms, _vector in entries:
            lexical = _coverage(query_terms, document_terms)
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

        # The bulk corpus is a first-class evidence source, so its chunks are
        # ranked *with* the internal ones rather than appended after them. The
        # FTS5 index selects the candidates (fast), but its scores are normalised
        # within a query (its best hit is always 1.0), which is not comparable
        # with the internal coverage scores -- so each candidate is rescored by
        # the same query coverage the internal documents use. External evidence is
        # kept slightly below internal evidence (``_BULK_FUSION_WEIGHT``).
        if self.bulk is not None:
            passages.extend(
                self._bulk_passages(query, query_terms, limit, indexed=self.bulk.indexed)
            )

        passages.sort(key=lambda passage: (-passage.score, passage.doc_id))
        return passages[:limit]

    def _bulk_passages(
        self, query: str, query_terms: set[str], limit: int, *, indexed: bool
    ) -> list[RetrievedPassage]:
        """Bulk-corpus candidates, deduplicated by article and rescored by coverage."""
        fetch = max(limit * 4, limit)
        best: dict[str, RetrievedPassage] = {}
        for passage in self.bulk.retrieve(query, limit=fetch):  # type: ignore[union-attr]
            if indexed:
                score = _coverage(query_terms, _tokens(f"{passage.title} {passage.text}"))
            else:
                score = passage.score
            score = min(1.0, score * _BULK_FUSION_WEIGHT)
            if score <= 0.0:
                continue
            key = passage.title or passage.doc_id
            existing = best.get(key)
            if existing is None or score > existing.score:
                best[key] = RetrievedPassage(
                    doc_id=passage.doc_id,
                    title=passage.title,
                    text=passage.text,
                    source=passage.source,
                    score=round(score, 6),
                    signal=passage.signal,
                )
        return list(best.values())


__all__ = ["BulkCorpusRetriever", "RagRetriever", "RetrievedPassage"]
