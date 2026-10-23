"""Every knowledge repository the project holds, exposed to the learners as data.

The seeds in :mod:`core.domain_knowledge`, :mod:`core.curriculum_knowledge`, and
:mod:`core.petroleum_corpus`, plus promoted learned statements, runtime
documents, and the built bulk knowledge base, are all the project "knows". This
module gives the learning layers one uniform view over them so that training is
grounded in the whole of that knowledge rather than in a hand-written sample.

The bulk knowledge base is the one repository that does not fit comfortably: it
is tens of thousands of Wikipedia chunks (~100 MB) against a language model of a
few hundred thousand parameters that sees only a few megabytes per run. Feeding
it whole would either be ignored (mostly unsampled) or dilute the project's own
voice with generic encyclopedic prose. So it is sampled, deliberately:

* it is filtered to chunks that actually match the project's petroleum
  vocabulary, so what is kept is *relevant* reference text, not noise;
* the sample is chosen deterministically by a hash of the chunk id, so the same
  repository always yields the same sample across runs and machines;
* it is bounded by a caller-chosen size and off by default, so tests and
  interactive paths stay fast and a release opts in explicitly.
"""

from __future__ import annotations

import hashlib
import heapq
import re
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field
from pathlib import Path

from core.documents import Document, build_knowledge_documents

# How the bulk sample is chosen. A chunk is relevant when its *title* names a
# specific term from the project's field: a matching title means the article is
# about a domain topic, whereas body mentions are far too common to discriminate.
_KB_MATCH_HITS = 1
# The default size of the relevance-filtered bulk sample a release trains on.
DEFAULT_KB_SAMPLE = 15000
_TOKEN_RE = re.compile(r"[a-z][a-z0-9]+")

# Tokens too common to signal relevance.
_STOP = frozenset(
    {
        "the", "and", "for", "with", "that", "this", "from", "are", "was", "were",
        "which", "their", "have", "has", "been", "into", "than", "then", "when",
        "they", "them", "such", "also", "other", "some", "more", "most", "used",
        "using", "use", "can", "may", "not", "but", "its", "his", "her", "one",
        "two", "all", "any", "each", "many", "over", "under", "between", "within",
        "well", "wells", "field", "fields", "oil", "gas", "water", "rock", "rocks",
    }
)

# Cross-domain words that appear in almost any engineering or science text, so
# matching them says little. Excluded from the relevance vocabulary: a passage
# counts as relevant only when it uses the field's *specific* terms.
_GENERIC = frozenset(
    {
        "engineering", "engineer", "engineers", "science", "sciences", "scientific",
        "physical", "chemistry", "chemical", "physics", "mathematics", "statistics",
        "energy", "power", "process", "processes", "system", "systems", "material",
        "materials", "machine", "machines", "structure", "structures", "design",
        "analysis", "method", "methods", "theory", "model", "models", "data",
        "development", "production", "industry", "industrial", "technology",
        "research", "history", "university", "college", "company", "companies",
        "natural", "general", "modern", "large", "small", "high", "low", "new",
        "measure", "measurement", "measurements", "value", "values", "rate",
        "pressure", "temperature", "volume", "flow", "force", "density", "mass",
        "artificial", "national", "international", "american", "world",
        "human", "information", "management", "safety", "quality", "test", "tests",
    }
)



def _tokens(text: str) -> set[str]:
    return {token for token in _TOKEN_RE.findall(text.casefold()) if token not in _STOP}


@dataclass(frozen=True)
class KnowledgeSample:
    """A deterministic sample of the project's knowledge, with provenance.

    ``documents`` is the sample itself; ``counts`` records how many documents
    came from each repository, and ``available`` records how many the bulk store
    actually held before sampling, so a report can say honestly what was seen
    versus what exists.
    """

    documents: tuple[Document, ...]
    counts: dict[str, int] = field(default_factory=dict)
    available: dict[str, int] = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.documents)

    def sources(self) -> tuple[str, ...]:
        return tuple(sorted(self.counts))


def repository_key(document: Document) -> str:
    """The repository a document came from, as a short stable label."""
    source = document.source or "internal"
    if source.startswith("learned"):
        return "learned"
    if source in {"domain_knowledge", "curriculum_knowledge", "petroleum_corpus"}:
        return source
    if source in {"wikipedia", "knowledge_base"} or document.metadata.get("origin") == "knowledge_base":
        return "knowledge_base"
    return "runtime"


def authored_documents() -> list[Document]:
    """The curated knowledge: disciplines, courses, corpus, and learned facts."""
    return list(build_knowledge_documents())


def runtime_documents() -> list[Document]:
    """Documents written at runtime (internet learning, Jarvis feedback)."""
    from core.paths import data_dir
    from storage.json_store import JsonStore

    store = JsonStore(data_dir() / "documents")
    documents: list[Document] = []
    for key in store.list_keys("documents"):
        record = store.load("documents", key)
        if not isinstance(record, dict):
            continue
        try:
            documents.append(Document.from_primitive(record))
        except (KeyError, ValueError, TypeError):
            continue
    return documents


def petroleum_vocabulary() -> frozenset[str]:
    """The terms that mark a passage as relevant to the project's field.

    Built from the authored knowledge (discipline and course labels, corpus
    titles, concepts, and glossary), so the bulk filter reflects the project's
    own declared domain rather than an outside keyword list.
    """
    global _PETROLEUM_VOCABULARY
    if _PETROLEUM_VOCABULARY is not None:
        return _PETROLEUM_VOCABULARY
    from core.curriculum_knowledge import courses
    from core.domain_knowledge import disciplines
    from core.petroleum_corpus import entries

    terms: set[str] = set()
    for discipline in disciplines():
        terms |= _tokens(discipline.label)
        for label, aliases in discipline.concepts:
            terms |= _tokens(label)
            for alias in aliases:
                terms |= _tokens(alias)
    for course in courses():
        terms |= _tokens(course.label)
        for label, _aliases in course.concepts:
            terms |= _tokens(label)
    for entry in entries():
        terms |= _tokens(entry.title)
        terms |= _tokens(entry.discipline)
        for label, aliases in entry.concepts:
            terms |= _tokens(label)
            for alias in aliases:
                terms |= _tokens(alias)
        for term, _arabic in entry.glossary:
            terms |= _tokens(term)
    _PETROLEUM_VOCABULARY = frozenset(terms - _GENERIC)
    return _PETROLEUM_VOCABULARY


def _hash_rank(chunk_id: str) -> int:
    return int.from_bytes(hashlib.sha1(chunk_id.encode("utf-8")).digest()[:8], "big")


# The vocabulary and the bulk sample are deterministic in the project's
# knowledge, so a process caches them: a release summary and its training would
# otherwise re-stream (and re-rank) the whole bulk store several times over.
_PETROLEUM_VOCABULARY: frozenset[str] | None = None
_KB_SAMPLE_CACHE: dict[tuple[int, int], tuple[tuple[Document, ...], int]] = {}


def knowledge_base_sample(
    *,
    max_chunks: int = 15000,
    march: frozenset[str] | None = None,
    min_hits: int = _KB_MATCH_HITS,
    store_dir: str | Path | None = None,
) -> tuple[list[Document], int]:
    """A relevance-filtered, deterministic sample of the bulk knowledge base.

    Streams every chunk once, keeps only those whose *title* names at least
    ``min_hits`` specific terms from the field's vocabulary, and retains the
    ``max_chunks`` with the smallest hash rank. Because selection is by hash
    (not by position), the sample is stable and spread across the whole store.
    Returns ``(documents, available)`` where ``available`` is how many relevant
    chunks existed.
    """
    from core.knowledge_base import KnowledgeBaseStore, default_knowledge_base_dir

    if max_chunks < 1:
        return [], 0
    root = store_dir if store_dir is not None else default_knowledge_base_dir()
    cacheable = store_dir is None and march is None
    if cacheable:
        cache_key = (max_chunks, min_hits)
        cached = _KB_SAMPLE_CACHE.get(cache_key)
        if cached is not None:
            return list(cached[0]), cached[1]
    store = KnowledgeBaseStore(root)
    if not store.shard_paths():
        return [], 0
    terms = march if march is not None else petroleum_vocabulary()

    keeper: list[tuple[int, int, str]] = []
    documents: dict[str, Document] = {}
    relevant = 0
    sequence = 0
    for chunk in store.iter_chunks():
        if len(_tokens(chunk.title) & terms) < min_hits:
            continue
        relevant += 1
        rank = _hash_rank(chunk.chunk_id)
        sequence += 1
        entry = (-rank, sequence, chunk.chunk_id)
        if len(keeper) < max_chunks:
            heapq.heappush(keeper, entry)
            documents[chunk.chunk_id] = chunk.to_document()
        elif entry > keeper[0]:
            _dropped = heapq.heapreplace(keeper, entry)
            documents[chunk.chunk_id] = chunk.to_document()
    selected_ids = [chunk_id for _rank, _seq, chunk_id in keeper]
    selected_ids.sort()
    selected = [documents[chunk_id] for chunk_id in selected_ids]
    if cacheable:
        _KB_SAMPLE_CACHE[cache_key] = (tuple(selected), relevant)
    return selected, relevant


def knowledge_sample(
    *,
    kb_max_chunks: int = 0,
    include_runtime: bool = True,
    include_authored: bool = True,
    march: frozenset[str] | None = None,
) -> KnowledgeSample:
    """Assemble a sample across every repository the project holds.

    Authored and runtime documents are small and always whole when included. The
    bulk knowledge base is filtered and bounded (``kb_max_chunks``, 0 disables
    it) because it is far too large to use whole.
    """
    documents: list[Document] = []
    available: dict[str, int] = {}
    if include_authored:
        authored = authored_documents()
        documents.extend(authored)
        available["authored"] = len(authored)
    if include_runtime:
        runtime = runtime_documents()
        documents.extend(runtime)
        available["runtime"] = len(runtime)
    if kb_max_chunks > 0:
        sampled, relevant = knowledge_base_sample(max_chunks=kb_max_chunks, march=march)
        documents.extend(sampled)
        available["knowledge_base"] = relevant

    counts: dict[str, int] = {}
    for document in documents:
        key = repository_key(document)
        counts[key] = counts.get(key, 0) + 1
    return KnowledgeSample(documents=tuple(documents), counts=counts, available=available)


def iter_documents(
    *,
    kb_max_chunks: int = 0,
    include_runtime: bool = True,
    march: frozenset[str] | None = None,
) -> Iterator[Document]:
    """Stream the assembled sample rather than materialising it all at once."""
    yield from authored_documents()
    if include_runtime:
        yield from runtime_documents()
    if kb_max_chunks > 0:
        sampled, _relevant = knowledge_base_sample(max_chunks=kb_max_chunks, march=march)
        yield from sampled


def sentences_from_documents(documents: Iterable[Document]) -> list[str]:
    """Split documents into clean, one-sentence training strings."""
    from core.composition import clean, sentences

    line: list[str] = []
    for document in documents:
        for sentence in sentences(clean(document.text)):
            if sentence:
                line.append(sentence)
    return line


__all__ = [
    "DEFAULT_KB_SAMPLE",
    "KnowledgeSample",
    "authored_documents",
    "iter_documents",
    "knowledge_base_sample",
    "knowledge_sample",
    "petroleum_vocabulary",
    "repository_key",
    "runtime_documents",
    "sentences_from_documents",
]
