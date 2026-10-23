"""Tests for resumable jobs and safe source deletion."""

from __future__ import annotations

from core.knowledge_base import KnowledgeBaseStore
from ingest.dedup import DedupLedger
from ingest.delete import SourceDeleter, delete_source
from ingest.jobs import DONE, JobState, JobStore
from ingest.pipeline import IngestionPipeline, ingest_jobs_dir, ledger_path
from ingest.records import SourceSpec


def _docs(tmp_path, **files):
    for name, text in files.items():
        target = tmp_path / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text, encoding="utf-8")
    return tmp_path


# -- jobs ------------------------------------------------------------------


def test_job_state_requires_id_and_source():
    import pytest

    with pytest.raises(ValueError):
        JobState(job_id="", source="x")
    with pytest.raises(ValueError):
        JobState(job_id="j", source="")


def test_job_store_roundtrip(tmp_path):
    store = JobStore(tmp_path / "jobs")
    state = JobState(job_id="j1", source="/docs", connector="files", position=3, chunks=9)
    store.save(state)
    loaded = store.load("j1")
    assert loaded is not None
    assert loaded.position == 3
    assert loaded.chunks == 9
    assert loaded.updated_at  # touched on save


def test_job_store_list_and_remove(tmp_path):
    store = JobStore(tmp_path / "jobs")
    store.save(JobState(job_id="a", source="/a"))
    store.save(JobState(job_id="b", source="/b"))
    assert {state.job_id for state in store.list()} == {"a", "b"}
    assert store.remove("a") is True
    assert store.remove("a") is False
    assert {state.job_id for state in store.list()} == {"b"}


def test_pipeline_records_a_resumable_job(tmp_path):
    source = _docs(tmp_path / "docs", **{"a.txt": "one\n", "b.txt": "two\n", "c.txt": "three\n"})
    store = KnowledgeBaseStore(tmp_path / "kb")
    ledger = DedupLedger(ledger_path(store.root))
    jobs = JobStore(ingest_jobs_dir(store.root))
    job = JobState(job_id="run-1", source=str(source))
    pipeline = IngestionPipeline(store, ledger, job=job, jobs=jobs)
    report = pipeline.ingest(SourceSpec(str(source)))
    assert report.chunks == 3
    saved = jobs.load("run-1")
    assert saved is not None
    assert saved.position == 3
    assert saved.status == DONE


def test_resumed_job_skips_completed_items(tmp_path):
    source = _docs(tmp_path / "docs", **{"a.txt": "one\n", "b.txt": "two\n", "c.txt": "three\n"})
    store = KnowledgeBaseStore(tmp_path / "kb")
    ledger = DedupLedger(ledger_path(store.root))
    jobs = JobStore(ingest_jobs_dir(store.root))
    # Pretend the first two items already completed.
    jobs.save(JobState(job_id="run-2", source=str(source), position=2))
    job = jobs.load("run-2")
    pipeline = IngestionPipeline(store, ledger, job=job, jobs=jobs)
    report = pipeline.ingest(SourceSpec(str(source)))
    assert report.records == 1  # only the third item is read


# -- deletion --------------------------------------------------------------


def _seed(tmp_path):
    source = _docs(
        tmp_path / "docs",
        **{
            "a.txt": "Pressure drives oil.\n",
            "b.txt": "Cement seals the well.\n",
        },
    )
    store = KnowledgeBaseStore(tmp_path / "kb")
    ledger = DedupLedger(ledger_path(store.root))
    pipeline = IngestionPipeline(store, ledger)
    pipeline.ingest(SourceSpec(str(source / "a.txt")))
    pipeline.ingest(SourceSpec(str(source / "b.txt")))
    return store, ledger, source


def test_delete_removes_only_the_named_source(tmp_path):
    store, ledger, source = _seed(tmp_path)
    assert store.count() == 2
    key = next(item.source_key for item in _items(source / "a.txt"))
    report = SourceDeleter(store, ledger).delete(key)
    assert report.removed == 1
    assert store.count() == 1
    remaining = [chunk.text for chunk in store.iter_chunks()]
    assert any("Cement" in text for text in remaining)
    assert all("Pressure" not in text for text in remaining)


def test_delete_reports_when_nothing_recorded(tmp_path):
    store, ledger, _source = _seed(tmp_path)
    report = SourceDeleter(store, ledger).delete("unknown-key")
    assert report.removed == 0
    assert store.count() == 2


def test_delete_by_key_helper(tmp_path):
    store, _ledger, source = _seed(tmp_path)
    key = next(item.source_key for item in _items(source / "a.txt"))
    report = delete_source(key, store_dir=store.root, ledger_dir=store.root)
    assert report.removed == 1
    assert store.count() == 1


def test_delete_drops_a_stale_index(tmp_path):
    from core.knowledge_base import iter_local_documents
    from core.knowledge_index import KnowledgeTextIndex, build_index

    store, ledger, source = _seed(tmp_path)
    build_index(store.root)
    assert KnowledgeTextIndex(store.root).exists()
    key = next(item.source_key for item in _items(source / "a.txt"))
    SourceDeleter(store, ledger).delete(key)
    assert not KnowledgeTextIndex(store.root).exists()
    # The store is still readable after the rewrite.
    assert len(list(iter_local_documents(store.root))) == 1


def _items(path):
    from ingest.connectors.files import FileConnector

    return list(FileConnector().plan(SourceSpec(str(path))))
