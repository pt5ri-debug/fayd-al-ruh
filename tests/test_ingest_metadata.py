"""Tests for per-document metadata and the explicit ingest interfaces."""

from __future__ import annotations

import pytest
from ingest.interfaces import IngestionJob, JobStateStore, SourceConnector
from ingest.jobs import JobState, JobStore
from ingest.metadata import (
    DocumentMetadata,
    DocumentMetadataStore,
    build_document_metadata,
    document_hash,
    document_id,
    metadata_path,
)


def test_document_metadata_rejects_bad_fields():
    with pytest.raises(ValueError):
        DocumentMetadata(doc_id="", source_id="s")
    with pytest.raises(ValueError):
        DocumentMetadata(doc_id="d", source_id="s", char_count=-1)


def test_document_hash_is_stable_and_content_sensitive():
    assert document_hash("hello") == document_hash("hello")
    assert document_hash("hello") != document_hash("hello!")
    assert len(document_hash("x")) == 32


def test_document_id_is_stable_for_source_and_hash():
    digest = document_hash("text")
    assert document_id("src", digest) == document_id("src", digest)
    assert document_id("src", digest) != document_id("other", digest)


def test_build_document_metadata_hashes_the_text_once():
    metadata = build_document_metadata(
        source_id="s", uri="u", title="T", text="hello world", chunk_ids=["ingest:ab"]
    )
    assert metadata.text_hash == document_hash("hello world")
    assert metadata.char_count == len("hello world")
    assert metadata.chunk_ids == ("ingest:ab",)
    assert metadata.doc_id.startswith("doc:s:")


def test_metadata_store_record_get_for_source_remove(tmp_path):
    store = DocumentMetadataStore(metadata_path(tmp_path))
    metadata = build_document_metadata(
        source_id="src1", uri="u", title="A", text="alpha", chunk_ids=["ingest:1"]
    )
    store.record(metadata)
    assert store.count() == 1
    loaded = store.get(metadata.doc_id)
    assert loaded.title == "A"
    assert loaded.chunk_ids == ("ingest:1",)
    assert [item.doc_id for item in store.for_source("src1")] == [metadata.doc_id]
    assert store.for_source("other") == []
    assert store.remove(metadata.doc_id) is True
    assert store.remove(metadata.doc_id) is False
    assert store.count() == 0


def test_metadata_store_indexes_by_source(tmp_path):
    store = DocumentMetadataStore(metadata_path(tmp_path))
    for index, source in enumerate(("a", "a", "b")):
        store.record(
            build_document_metadata(
                source_id=source, uri=f"u{index}", title="T", text=f"text {index}"
            )
        )
    assert len(store.for_source("a")) == 2
    assert len(store.for_source("b")) == 1


def test_metadata_store_replaces_on_same_doc_id(tmp_path):
    store = DocumentMetadataStore(metadata_path(tmp_path))
    metadata = build_document_metadata(source_id="s", uri="u", title="old", text="same")
    store.record(metadata)
    updated = DocumentMetadata(
        doc_id=metadata.doc_id, source_id="s", uri="u", title="new", text_hash=metadata.text_hash
    )
    store.record(updated)
    assert store.count() == 1
    assert store.get(metadata.doc_id).title == "new"


def test_source_connector_interface_matches_real_connectors():
    from ingest.connectors.files import FileConnector
    from ingest.connectors.sql import SqlConnector

    assert isinstance(FileConnector(), SourceConnector)
    assert isinstance(SqlConnector(), SourceConnector)


def test_job_interfaces_match_real_objects(tmp_path):
    assert isinstance(JobState(job_id="j", source="s"), IngestionJob)
    assert isinstance(JobStore(tmp_path / "jobs"), JobStateStore)


def test_job_state_store_protocol_rejects_impostors():
    class NotAStore:
        pass

    assert not isinstance(NotAStore(), JobStateStore)
