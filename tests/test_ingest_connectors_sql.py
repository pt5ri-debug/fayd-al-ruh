"""Tests for the SQL connector: streaming, read-only safety, and drivers."""

from __future__ import annotations

import sqlite3

import pytest

from ingest.connectors.sql import SqlConnector, _require_read_only, _safe_identifier
from ingest.errors import ConnectorError
from ingest.records import SourceSpec


def _database(path, rows=3):
    connection = sqlite3.connect(path)
    connection.execute("CREATE TABLE wells (name TEXT, depth INTEGER)")
    connection.executemany(
        "INSERT INTO wells VALUES(?,?)",
        [(f"A-{index}", 1000 + index * 100) for index in range(rows)],
    )
    connection.commit()
    connection.close()
    return path


def test_lists_tables_and_streams_rows(tmp_path):
    database = _database(tmp_path / "data.db", rows=5)
    connector = SqlConnector()
    items = list(connector.plan(SourceSpec(f"sqlite://{database}")))
    assert [item.title for item in items] == ["wells"]
    records = list(connector.stream_item(items[0]))
    assert "name: A-0" in records[0].text
    assert "depth: 1400" in records[0].text
    assert records[0].source == "sql"


def test_stream_groups_rows_into_parts(tmp_path):
    database = _database(tmp_path / "data.db", rows=25)
    spec = SourceSpec(f"sqlite://{database}", options={"row_group": 10, "page": 7})
    records = list(SqlConnector().stream(spec))
    assert len(records) == 3  # 25 rows / 10 per group
    assert all(r.title.startswith("wells") for r in records)


def test_flat_memory_streams_large_table(tmp_path):
    database = _database(tmp_path / "big.db", rows=5000)
    spec = SourceSpec(f"sqlite://{database}", options={"row_group": 500, "page": 250})
    records = list(SqlConnector().stream(spec))
    assert len(records) == 10
    assert all(len(r.text.splitlines()) == 500 for r in records)


def test_plan_fingerprint_reflects_rows(tmp_path):
    database = _database(tmp_path / "data.db", rows=4)
    item = next(iter(SqlConnector().plan(SourceSpec(f"sqlite://{database}"))))
    assert item.fingerprint.startswith("4:")


def test_explicit_query_is_used(tmp_path):
    database = _database(tmp_path / "data.db", rows=5)
    spec = SourceSpec(
        f"sqlite://{database}",
        options={"query": "SELECT name FROM wells WHERE depth > 1100", "row_group": 10},
    )
    records = list(SqlConnector().stream(spec))
    text = records[0].text
    assert "A-2" in text
    assert "A-0" not in text


def test_mutating_statement_is_refused(tmp_path):
    database = _database(tmp_path / "data.db")
    spec = SourceSpec(f"sqlite://{database}", options={"query": "DELETE FROM wells"})
    with pytest.raises(ConnectorError):
        list(SqlConnector().stream(spec))


def test_require_read_only_accepts_select_and_with():
    _require_read_only("SELECT * FROM t")
    _require_read_only("  with x as (select 1) select * from x")
    with pytest.raises(ConnectorError):
        _require_read_only("DROP TABLE t")


def test_safe_identifier_rejects_injection():
    assert _safe_identifier("wells") == "wells"
    for bad in ("wells; DROP TABLE x", "wells--", "a b", "1abc", ""):
        with pytest.raises(ConnectorError):
            _safe_identifier(bad)


def test_missing_database_reports_a_connector_error(tmp_path):
    with pytest.raises(ConnectorError):
        list(SqlConnector().stream(SourceSpec(f"sqlite://{tmp_path / 'none.db'}")))


def test_driver_without_package_reports_an_install_hint(tmp_path):
    spec = SourceSpec("postgresql://host/db", options={"tables": ["wells"], "dsn": "x"})
    items = list(SqlConnector().plan(spec))
    with pytest.raises(ConnectorError) as error:
        list(SqlConnector().stream_item(items[0]))
    assert "psycopg" in str(error.value)


def test_driver_without_dsn_reports_a_connector_error(tmp_path):
    # Fake a non-sqlite scheme by using a driver scheme with no dsn.
    from ingest.records import IngestItem

    item = IngestItem(
        source_key="k",
        uri="postgresql://host/db#t",
        title="t",
        metadata={"table": "t", "scheme": "postgresql", "dsn": ""},
    )
    with pytest.raises(ConnectorError) as error:
        list(SqlConnector().stream_item(item))
    assert "dsn" in str(error.value)
