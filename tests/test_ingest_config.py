"""Tests for the ingestion configuration (storage + env + explicit precedence)."""

from __future__ import annotations

import pytest

from ingest.config import (
    ENV_PREFIX,
    IngestConfig,
    load_ingest_config,
    save_ingest_config,
)
from ingest.pipeline import DEFAULT_SHARD_BYTES


def test_ingest_config_defaults_match_the_pipeline():
    config = IngestConfig()
    assert config.incremental is True
    assert config.allow_web is False  # deny by default
    assert config.allow_ocr is None  # automatic: OCR'd only when the extra is present
    assert config.shard_bytes == DEFAULT_SHARD_BYTES


def test_ingest_config_ocr_tristate_round_trips(tmp_path):
    # ``None`` is automatic, ``True`` forces OCR, ``False`` refuses it.
    for value in (None, True, False):
        save_ingest_config(IngestConfig(allow_ocr=value), root=tmp_path)
        assert load_ingest_config(root=tmp_path).allow_ocr is value


def test_ingest_config_allow_ocr_environment_override(tmp_path):
    loaded = load_ingest_config(
        root=tmp_path, environ={f"{ENV_PREFIX}ALLOW_OCR": "false"}
    )
    assert loaded.allow_ocr is False


def test_ingest_config_rejects_inconsistent_overlap():
    with pytest.raises(ValueError):
        IngestConfig(chunk_chars=100, overlap=100)
    with pytest.raises(ValueError):
        IngestConfig(chunk_chars=0)


def test_ingest_config_resolves_paths_under_the_store_root():
    config = IngestConfig(store_dir="/srv/kb")
    assert config.resolved_store_dir().as_posix() == "/srv/kb"
    assert config.resolved_ledger_path().as_posix().endswith("/srv/kb/ingest/dedup.sqlite")
    assert config.resolved_jobs_dir().as_posix().endswith("/srv/kb/ingest/jobs")


def test_ingest_config_save_and_load_round_trip(tmp_path):
    save_ingest_config(IngestConfig(store_dir="/kb", chunk_chars=500, overlap=10), root=tmp_path)
    loaded = load_ingest_config(root=tmp_path)
    assert loaded.store_dir == "/kb"
    assert loaded.chunk_chars == 500
    assert loaded.overlap == 10


def test_ingest_config_environment_overrides_stored_value(tmp_path):
    save_ingest_config(IngestConfig(chunk_chars=500, overlap=10), root=tmp_path)
    loaded = load_ingest_config(
        root=tmp_path,
        environ={f"{ENV_PREFIX}CHUNK_CHARS": "400", f"{ENV_PREFIX}ALLOW_WEB": "true"},
    )
    assert loaded.chunk_chars == 400
    assert loaded.allow_web is True


def test_ingest_config_explicit_argument_beats_environment(tmp_path):
    loaded = load_ingest_config(
        root=tmp_path,
        environ={f"{ENV_PREFIX}CHUNK_CHARS": "400", f"{ENV_PREFIX}OVERLAP": "10"},
        chunk_chars=111,
        overlap=5,
    )
    assert loaded.chunk_chars == 111
    assert loaded.overlap == 5


def test_ingest_config_ignores_unknown_stored_fields(tmp_path):
    from storage.json_store import JsonStore

    JsonStore(tmp_path).save(
        "ingest", "config", {"chunk_chars": 300, "overlap": 10, "nonsense": 1}
    )
    loaded = load_ingest_config(root=tmp_path)
    assert loaded.chunk_chars == 300
    assert not hasattr(loaded, "nonsense")


def test_ingest_config_malformed_stored_value_falls_back(tmp_path):
    from storage.json_store import JsonStore

    JsonStore(tmp_path).save("ingest", "config", {"chunk_chars": "not-an-int"})
    loaded = load_ingest_config(root=tmp_path)
    assert loaded.chunk_chars == IngestConfig().chunk_chars


def test_ingest_config_invalid_override_raises(tmp_path):
    # An override the caller supplied is checked strictly, not silently dropped.
    with pytest.raises(ValueError):
        load_ingest_config(root=tmp_path, chunk_chars=10)  # below the default overlap


def test_ingest_config_bad_environment_value_is_ignored(tmp_path):
    loaded = load_ingest_config(root=tmp_path, environ={f"{ENV_PREFIX}CHUNK_CHARS": "oops"})
    assert loaded.chunk_chars == IngestConfig().chunk_chars
