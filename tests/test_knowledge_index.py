"""Tests for the FTS5 text index over a bulk knowledge base (offline)."""

from __future__ import annotations

from core.knowledge_base import KnowledgeBaseStore, KnowledgeChunk
from core.knowledge_index import KnowledgeTextIndex, build_index
from core.rag import BulkCorpusRetriever


def _store(tmp_path):
    store = KnowledgeBaseStore(tmp_path / "kb")
    store.ensure()
    chunks = [
        KnowledgeChunk(
            chunk_id="kb:Drilling_fluid:0",
            title="Drilling fluid",
            text="Drilling mud cools the bit and carries rock cuttings to the surface.",
            source="wikipedia",
            url="https://en.wikipedia.org/wiki/Drilling_fluid",
            category="Petroleum engineering",
        ),
        KnowledgeChunk(
            chunk_id="kb:Net_present_value:0",
            title="Net present value",
            text="Net present value discounts future cash flows to the present.",
            source="wikipedia",
            url="https://en.wikipedia.org/wiki/Net_present_value",
            category="Economics",
        ),
        KnowledgeChunk(
            chunk_id="kb:Blowout_preventer:0",
            title="Blowout preventer",
            text="A blowout preventer seals the wellbore to stop an uncontrolled release.",
            source="wikipedia",
            url="https://en.wikipedia.org/wiki/Blowout_preventer",
            category="Petroleum engineering",
        ),
    ]
    store.append(0, chunks)
    return store


def test_build_and_search_returns_ranked_chunks(tmp_path):
    store = _store(tmp_path)
    index = KnowledgeTextIndex(tmp_path / "kb", store=store)
    summary = index.build()
    assert summary["indexed"] == 3
    assert index.exists()

    results = index.search("drilling mud cuttings")
    assert results
    score, chunk = results[0]
    assert chunk.chunk_id == "kb:Drilling_fluid:0"
    assert 0.0 < score <= 1.0


def test_search_is_safe_with_fts_operators(tmp_path):
    # Quotes/operators must be treated literally, not as FTS5 syntax.
    store = _store(tmp_path)
    index = KnowledgeTextIndex(tmp_path / "kb", store=store)
    index.build()
    assert index.search(")( \"") == []  # punctuation only -> no terms
    assert index.search("wellbore")[0][1].chunk_id == "kb:Blowout_preventer:0"
    # A wildcard/column operator does not inject FTS5 syntax or raise.
    assert index.search("drilling* OR title:blowout")


def test_missing_index_reports_absent(tmp_path):
    index = KnowledgeTextIndex(tmp_path / "kb")
    assert not index.exists()
    assert index.stats() == {"present": False}


def test_retriever_uses_index_and_falls_back(tmp_path):
    store = _store(tmp_path)
    docs = [chunk.to_document() for chunk in store.iter_chunks()]

    indexed = BulkCorpusRetriever(lambda: iter(docs), index=KnowledgeTextIndex(tmp_path / "kb", store=store))
    # index absent -> streaming fallback still answers
    assert indexed.retrieve("blowout preventer", limit=1)[0].doc_id == "kb:Blowout_preventer:0"

    build_index(tmp_path / "kb")
    indexed = BulkCorpusRetriever(lambda: iter(docs), index=KnowledgeTextIndex(tmp_path / "kb", store=store))
    assert indexed.retrieve("net present value", limit=1)[0].doc_id == "kb:Net_present_value:0"


def test_build_index_summary(tmp_path):
    _store(tmp_path)
    summary = build_index(tmp_path / "kb")
    assert summary["indexed"] == 3
    assert summary["present"] is True
    assert summary["size_bytes"] > 0
