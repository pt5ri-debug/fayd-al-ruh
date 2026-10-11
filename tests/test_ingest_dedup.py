"""Tests for the dedup ledger: source state, chunk hashes, and stats."""

from __future__ import annotations

from ingest.dedup import DedupLedger
from ingest.hashing import content_hash, file_fingerprint, source_key


def test_hashes_are_stable_and_distinct():
    assert content_hash("abc") == content_hash("abc")
    assert content_hash("abc") != content_hash("abd")
    assert source_key("a") == source_key("a")
    assert source_key("a") != source_key("b")
    assert source_key("a", extra="t1") != source_key("a", extra="t2")


def test_file_fingerprint_missing_file_is_empty(tmp_path):
    assert file_fingerprint(tmp_path / "nope") == ""
    target = tmp_path / "a.txt"
    target.write_text("x", encoding="utf-8")
    assert file_fingerprint(target).startswith("1:")


def test_ledger_tracks_sources_and_chunks(tmp_path):
    ledger = DedupLedger(tmp_path / "dedup.sqlite")
    assert ledger.known_source("k") is None
    ledger.record_source("k", uri="a.txt", connector="files", fingerprint="1:2", chunks=2, bytes_written=10)
    state = ledger.known_source("k")
    assert state is not None and state.chunks == 2
    assert ledger.unchanged("k", "1:2")
    assert not ledger.unchanged("k", "9:9")
    assert not ledger.unchanged("unknown", "1:2")


def test_ledger_tracks_chunk_hashes(tmp_path):
    ledger = DedupLedger(tmp_path / "dedup.sqlite")
    digest = content_hash("some text")
    assert not ledger.contains_chunk(digest)
    ledger.record_chunk(digest, source_key="k", doc_id="ingest:x")
    assert ledger.contains_chunk(digest)
    # Recording the same hash twice is a no-op, not an error.
    ledger.record_chunk(digest, source_key="k", doc_id="ingest:x")
    assert ledger.source_chunks("k") == ["ingest:x"]


def test_ledger_drop_and_stats(tmp_path):
    ledger = DedupLedger(tmp_path / "dedup.sqlite")
    ledger.record_source("k", uri="a", connector="files", fingerprint="1:2", chunks=1, bytes_written=5)
    ledger.record_chunk("h", source_key="k", doc_id="d")
    assert ledger.stats() == {"sources": 1, "chunks": 1, "bytes": 5}
    assert ledger.drop_source("k") == 1
    assert ledger.stats() == {"sources": 0, "chunks": 0, "bytes": 0}


def test_ledger_reset_clears_everything(tmp_path):
    ledger = DedupLedger(tmp_path / "dedup.sqlite")
    ledger.record_source("k", uri="a", connector="files")
    ledger.record_chunk("h", source_key="k", doc_id="d")
    ledger.reset()
    assert ledger.stats()["sources"] == 0
    assert not ledger.contains_chunk("h")
