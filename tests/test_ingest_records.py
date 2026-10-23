"""Tests for the ingestion contracts: records, specs, reports, and the registry."""

from __future__ import annotations

import pytest
from ingest.errors import ConnectorError
from ingest.records import records_from_texts
from ingest.registry import ConnectorRegistry, build_default_registry

from ingest import (
    ConnectorInfo,
    IngestionReport,
    IngestItem,
    SourceRecord,
    SourceSpec,
)


def test_source_spec_requires_a_uri():
    with pytest.raises(ValueError):
        SourceSpec(uri="   ")


def test_source_spec_option_returns_default():
    spec = SourceSpec(uri="x", options={"a": 1})
    assert spec.option("a") == 1
    assert spec.option("missing", "fallback") == "fallback"


def test_source_record_rejects_empty_fields():
    with pytest.raises(ValueError):
        SourceRecord(title="", text="t", source="file")
    with pytest.raises(ValueError):
        SourceRecord(title="t", text=" ", source="file")
    with pytest.raises(ValueError):
        SourceRecord(title="t", text="t", source="")


def test_ingest_item_requires_key_uri_title():
    with pytest.raises(ValueError):
        IngestItem(source_key="", uri="u", title="t")
    with pytest.raises(ValueError):
        IngestItem(source_key="k", uri="", title="t")
    with pytest.raises(ValueError):
        IngestItem(source_key="k", uri="u", title="")


def test_connector_info_rejects_empty_name_or_description():
    with pytest.raises(ValueError):
        ConnectorInfo(name="", description="d")
    with pytest.raises(ValueError):
        ConnectorInfo(name="n", description="  ")


def test_ingestion_report_as_dict():
    report = IngestionReport(records=3, chunks=5)
    assert report.as_dict()["records"] == 3
    assert report.as_dict()["chunks"] == 5
    assert report.as_dict()["failed"] == 0


def test_records_from_texts_skips_blank_text():
    records = list(records_from_texts([("a", "one"), ("b", "   ")], source="file"))
    assert [r.title for r in records] == ["a"]


class _StubConnector:
    def __init__(self, info):
        self.info = info

    def plan(self, spec):
        return iter(())

    def stream_item(self, item):
        return iter(())

    def stream(self, spec):
        return iter(())


def test_registry_rejects_duplicate_names():
    info = ConnectorInfo(name="dup", description="d")
    registry = ConnectorRegistry()
    registry.register(_StubConnector(info))
    with pytest.raises(ValueError):
        registry.register(_StubConnector(info))


def test_registry_resolves_by_scheme_then_suffix_then_default():
    registry = ConnectorRegistry()
    sql = _StubConnector(ConnectorInfo(name="sql", description="d", schemes=frozenset({"sqlite"})))
    files = _StubConnector(
        ConnectorInfo(name="files", description="d", suffixes=frozenset({".csv"}), default=True)
    )
    registry.register(sql)
    registry.register(files)
    assert registry.resolve(SourceSpec("sqlite:///db")).info.name == "sql"
    assert registry.resolve(SourceSpec("report.csv")).info.name == "files"
    # A bare directory (no scheme, no suffix) falls to the default connector.
    assert registry.resolve(SourceSpec("/data/folder")).info.name == "files"


def test_registry_rejects_unknown_source_without_default():
    registry = ConnectorRegistry()
    registry.register(
        _StubConnector(ConnectorInfo(name="sql", description="d", schemes=frozenset({"sqlite"})))
    )
    with pytest.raises(ConnectorError):
        registry.resolve(SourceSpec("https://example.com"))


def test_registry_does_not_treat_windows_drive_as_scheme():
    from ingest.registry import _scheme

    assert _scheme("C:\\data\\file.txt") == ""
    assert _scheme("sqlite:///db") == "sqlite"


def test_default_registry_registers_files_and_sql_but_not_web():
    registry = build_default_registry()
    names = registry.names()
    assert "files" in names
    assert "sql" in names
    assert "web" not in names


def test_default_registry_can_opt_into_web():
    registry = build_default_registry(allow_web=True)
    assert "web" in registry.names()
