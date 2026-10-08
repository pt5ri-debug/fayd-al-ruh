"""Tests for the buildable bulk knowledge base.

Everything here is offline: the network is replaced by a small fake fetcher, so
the tests exercise chunking, storage, the manifest/checkpoint, streaming
retrieval, and the CLI without touching Wikipedia.
"""

from __future__ import annotations

from core.documents import DocumentStore
from core.knowledge_base import (
    KnowledgeBaseStore,
    KnowledgeChunk,
    build_knowledge_base,
    chunk_text,
    clean_wikitext,
    iter_local_documents,
)
from core.rag import BulkCorpusRetriever, RagRetriever

_ARTICLE = (
    "Drilling fluid, also called mud, is a fluid used to drill boreholes into the "
    "earth. It cools and lubricates the drill bit, carries rock cuttings to the "
    "surface, and controls formation pressure to prevent a blowout. Water-based "
    "mud, oil-based mud, and synthetic-based mud are the main types. "
) * 20


class FakeFetcher:
    """A deterministic stand-in for :class:`WikipediaFetcher`."""

    def __init__(self, per_category: int = 3) -> None:
        self.per_category = per_category
        self.calls: list[str] = []

    def category_titles(self, category: str) -> list[str]:
        self.calls.append(category)
        return [f"{category} article {i}" for i in range(self.per_category)]

    def extracts(self, titles):
        return [(title, _ARTICLE) for title in titles]


def test_chunk_text_splits_and_overlaps():
    chunks = chunk_text("A" * 5000, chunk_chars=1000, overlap=100)
    assert len(chunks) >= 5
    assert all(chunk for chunk in chunks)
    # Overlap means the next chunk starts before the previous one ended.
    assert chunks[0][-100:] == chunks[1][:100]


def test_chunk_text_returns_short_text_as_one_chunk():
    assert chunk_text("a short note") == ["a short note"]
    assert chunk_text("   ") == []


def test_clean_wikitext_removes_markup_keeps_prose():
    raw = (
        "== Drilling fluid ==\n"
        "'''Drilling fluid''' is a fluid.<ref>{{cite web|url=x}}</ref> "
        "See [[Drilling mud|mud]] and [[File:rig.jpg|thumb|caption]]. "
        "{| class=\"wikitable\"\n|-\n| a || b\n|}\n"
        "<!-- hidden -->\n"
    )
    cleaned = clean_wikitext(raw)
    assert "Drilling fluid" in cleaned
    assert "is a fluid" in cleaned
    assert "{{" not in cleaned and "}}" not in cleaned
    assert "[[File" not in cleaned
    assert "<!--" not in cleaned
    assert "<ref" not in cleaned
    # A piped link keeps only its display text.
    assert "mud" in cleaned
    assert "Drilling mud|" not in cleaned


def test_clean_wikitext_handles_empty():
    assert clean_wikitext("") == ""
    assert clean_wikitext("   ") == ""


def test_chunk_text_handles_arabic():
    text = "طين الحفر " * 400
    chunks = chunk_text(text, chunk_chars=500, overlap=50)
    assert chunks
    assert "طين" in chunks[0]


def test_store_append_iter_and_stats(tmp_path):
    store = KnowledgeBaseStore(tmp_path / "kb")
    chunk = KnowledgeChunk(
        chunk_id="kb:test:0",
        title="Test",
        text="hello world",
        source="wikipedia",
        url="https://example.org/x",
    )
    written = store.append(0, [chunk])
    assert written > 0
    assert store.size_bytes() == written
    assert store.count() == 1
    restored = next(store.iter_chunks())
    assert restored.chunk_id == "kb:test:0"
    assert restored.to_document().doc_id == "kb:test:0"


def test_build_reaches_target_and_writes_manifest(tmp_path):
    store_dir = tmp_path / "kb"
    manifest = build_knowledge_base(
        store_dir,
        target_mb=0.02,
        categories=["Drilling technology", "Geology"],
        fetcher=FakeFetcher(per_category=5),
        batch=5,
        sleep=0,
        progress=None,
    )
    assert manifest["total_chunks"] > 0
    assert manifest["total_bytes"] > 0
    store = KnowledgeBaseStore(store_dir)
    assert store.shard_paths()
    assert store.manifest_path.exists()
    assert store.load_manifest()["total_chunks"] == manifest["total_chunks"]


def test_build_is_resumable_via_checkpoint(tmp_path):
    store_dir = tmp_path / "kb"
    fetcher = FakeFetcher(per_category=2)
    build_knowledge_base(
        store_dir,
        target_mb=100.0,
        categories=["Drilling technology"],
        fetcher=fetcher,
        batch=2,
        sleep=0,
    )
    checkpoint = KnowledgeBaseStore(store_dir).load_checkpoint()
    assert "Drilling technology" in checkpoint["done_categories"]
    # Re-running skips the finished category instead of fetching it again.
    before = list(fetcher.calls)
    build_knowledge_base(
        store_dir,
        target_mb=100.0,
        categories=["Drilling technology"],
        fetcher=fetcher,
        batch=2,
        sleep=0,
    )
    assert fetcher.calls == before


def test_iter_local_documents_is_lazy_and_citable(tmp_path):
    store_dir = tmp_path / "kb"
    build_knowledge_base(
        store_dir,
        target_mb=0.02,
        categories=["Geology"],
        fetcher=FakeFetcher(per_category=4),
        batch=4,
        sleep=0,
    )
    documents = list(iter_local_documents(store_dir))
    assert documents
    assert documents[0].doc_id.startswith("kb:")
    assert documents[0].metadata["origin"] == "knowledge_base"
    assert documents[0].metadata["license"]


def test_iter_local_documents_empty_when_missing(tmp_path):
    assert list(iter_local_documents(tmp_path / "absent")) == []


def test_bulk_retriever_ranks_by_relevance():
    from core.documents import Document

    docs = [
        Document(doc_id="kb:a:0", title="Drilling fluid", text="mud cools the drill bit", source="wikipedia"),
        Document(doc_id="kb:b:0", title="Economics", text="net present value of a project", source="wikipedia"),
    ]
    retriever = BulkCorpusRetriever(lambda: iter(docs))
    passages = retriever.retrieve("drilling mud", limit=1)
    assert passages
    assert passages[0].doc_id == "kb:a:0"
    assert "mud" in passages[0].text


def test_rag_retriever_consults_bulk_only_when_weak(tmp_path):
    from core.documents import Document

    internal = DocumentStore()
    internal.add(Document(doc_id="internal:1", title="Reservoir", text="reservoir rock porosity", source="internal"))
    bulk_docs = [Document(doc_id="kb:x:0", title="Waterflood", text="waterflood maintains reservoir pressure", source="wikipedia")]
    retriever = RagRetriever(
        internal,
        bulk=BulkCorpusRetriever(lambda: iter(bulk_docs)),
    )
    # A query the internal store answers keeps its internal evidence first.
    top = retriever.retrieve("reservoir rock porosity", limit=1)
    assert top[0].doc_id == "internal:1"
    # A query with no internal overlap falls through to the bulk corpus.
    top = retriever.retrieve("waterflood pressure", limit=1)
    assert top[0].doc_id == "kb:x:0"


def test_cli_stats_and_search(tmp_path, capsys):
    from cli.knowledge import main

    store_dir = tmp_path / "kb"
    build_knowledge_base(
        store_dir,
        target_mb=0.02,
        categories=["Geology"],
        fetcher=FakeFetcher(per_category=3),
        batch=3,
        sleep=0,
    )
    assert main(["--store-dir", str(store_dir), "stats"]) == 0
    out = capsys.readouterr().out
    assert "Knowledge base" in out
    assert "MB" in out

    assert main(["--store-dir", str(store_dir), "search", "mud", "--limit", "2"]) == 0
    out = capsys.readouterr().out
    assert "mud" in out.lower()


def test_cli_search_without_store_reports_clearly(tmp_path, capsys):
    from cli.knowledge import main

    assert main(["--store-dir", str(tmp_path / "absent"), "search", "anything"]) == 1
    assert "No knowledge base" in capsys.readouterr().out
