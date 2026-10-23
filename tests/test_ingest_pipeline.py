"""Tests for the ingestion pipeline: store integration, dedup, and separation."""

from __future__ import annotations

import sqlite3

from core.knowledge_base import KnowledgeBaseStore, iter_local_documents
from core.knowledge_index import KnowledgeTextIndex, build_index
from core.rag import BulkCorpusRetriever
from ingest.dedup import DedupLedger
from ingest.pipeline import IngestionPipeline, ledger_path
from ingest.records import SourceSpec


def _pipeline(root):
    store = KnowledgeBaseStore(root / "kb")
    ledger = DedupLedger(ledger_path(store.root))
    return store, IngestionPipeline(store, ledger)


def _docs(path, **files):
    for name, text in files.items():
        target = path / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text, encoding="utf-8")
    return path


def test_ingest_stores_chunks_in_the_shared_store(tmp_path):
    source = _docs(tmp_path / "docs", **{"a.txt": "Water makes the soil wet.\n"})
    store, pipeline = _pipeline(tmp_path)
    report = pipeline.ingest(SourceSpec(str(source)))
    assert report.chunks == 1
    assert store.count() == 1
    chunk = next(iter(store.iter_chunks()))
    assert chunk.source == "file"
    assert chunk.chunk_id.startswith("ingest:")


def test_ingest_never_raises_on_a_bad_source(tmp_path):
    _store, pipeline = _pipeline(tmp_path)
    report = pipeline.ingest(SourceSpec(str(tmp_path / "missing")))
    assert report.failed == 1
    assert len(pipeline.errors) == 1
    assert pipeline.errors.errors[0].kind == "ConnectorError"


def test_incremental_rerun_skips_unchanged_items(tmp_path):
    source = _docs(tmp_path / "docs", **{"a.txt": "one\n", "b.txt": "two\n"})
    store, pipeline = _pipeline(tmp_path)
    pipeline.ingest(SourceSpec(str(source)))
    report = pipeline.ingest(SourceSpec(str(source)))
    assert report.skipped_unchanged == 2
    assert report.chunks == 0
    assert store.count() == 2


def test_changed_file_is_re_ingested(tmp_path):
    source = _docs(tmp_path / "docs", **{"a.txt": "one\n"})
    _store, pipeline = _pipeline(tmp_path)
    pipeline.ingest(SourceSpec(str(source)))
    (source / "a.txt").write_text("one\nplus a new line\n", encoding="utf-8")
    report = pipeline.ingest(SourceSpec(str(source)))
    assert report.skipped_unchanged == 0
    assert report.chunks >= 1


def test_identical_content_is_deduplicated(tmp_path):
    source = _docs(
        tmp_path / "docs",
        **{"a.txt": "The same paragraph repeated.\n", "b.txt": "The same paragraph repeated.\n"},
    )
    store, pipeline = _pipeline(tmp_path)
    report = pipeline.ingest(SourceSpec(str(source)))
    # Two records, but only one stored chunk: the content hash matched.
    assert report.records == 2
    assert store.count() == 1


def test_store_directory_is_not_ingested_as_a_source(tmp_path):
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "a.txt").write_text("real content\n", encoding="utf-8")
    store = KnowledgeBaseStore(docs / "kb")
    pipeline = IngestionPipeline(store, DedupLedger(ledger_path(tmp_path / "led")))
    pipeline.ingest(SourceSpec(str(docs)))
    # A re-run must skip the file and must not have ingested the store's shards.
    report = pipeline.ingest(SourceSpec(str(docs)))
    assert report.skipped_unchanged == 1
    assert store.count() == 1


def test_ingested_text_is_retrievable_through_the_fts_index(tmp_path):
    source = _docs(
        tmp_path / "docs",
        **{
            "r.txt": "Pressure drives oil toward the well. Permeability measures fluid flow.",
            "g.txt": "Cement seals the well from other layers. A core reveals porosity.",
        },
    )
    store, pipeline = _pipeline(tmp_path)
    pipeline.ingest(SourceSpec(str(source)))
    build_index(store.root)
    retriever = BulkCorpusRetriever(
        lambda: iter_local_documents(store.root), index=KnowledgeTextIndex(store.root)
    )
    passages = retriever.retrieve("permeability fluid", limit=3)
    assert passages
    assert "Permeability" in passages[0].text


def test_sql_source_flows_through_the_pipeline(tmp_path):
    database = tmp_path / "data.db"
    connection = sqlite3.connect(database)
    connection.execute("CREATE TABLE wells (name TEXT, depth INTEGER)")
    connection.executemany("INSERT INTO wells VALUES(?,?)", [("A-1", 1000), ("A-2", 1200)])
    connection.commit()
    connection.close()
    store, pipeline = _pipeline(tmp_path)
    report = pipeline.ingest(SourceSpec(f"sqlite://{database}"))
    assert report.chunks == 1
    assert store.count() == 1


def test_manifest_records_the_ledger_summary(tmp_path):
    source = _docs(tmp_path / "docs", **{"a.txt": "content\n"})
    store, pipeline = _pipeline(tmp_path)
    pipeline.ingest(SourceSpec(str(source)))
    manifest = store.load_manifest()
    assert manifest["ingest"]["sources"] == 1
    assert manifest["ingest"]["chunks"] == 1


def test_ingest_source_helper_returns_report_and_errors(tmp_path):
    from ingest.pipeline import ingest_source

    source = _docs(tmp_path / "docs", **{"a.txt": "helper content\n"})
    report, errors = ingest_source(SourceSpec(str(source)), store_dir=str(tmp_path / "kb2"))
    assert report.chunks == 1
    assert errors.ok
