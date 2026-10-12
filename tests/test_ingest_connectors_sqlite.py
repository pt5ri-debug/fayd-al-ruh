"""Tests for the standard-library SQLite connector and its pipeline integration.

The connector is read-only, never creates a database, reads the schema from
SQLite's own metadata, and streams rows in bounded ``fetchmany`` batches. These
tests build real temporary databases (including tables with composite and absent
primary keys, Arabic text, NULLs, numeric/date/blob values, and duplicates) and
assert both the read behaviour and that the source file is left unchanged.
"""

from __future__ import annotations

import hashlib
import sqlite3
from pathlib import Path

import pytest

from core.knowledge_base import KnowledgeBaseStore, iter_local_documents
from core.knowledge_index import KnowledgeTextIndex, build_index
from core.rag import BulkCorpusRetriever
from ingest.connectors.sqlite import (
    MAX_PAGE,
    MAX_ROW_GROUP,
    SqliteAccessError,
    SqliteConnector,
    SqliteOptionError,
    SqliteOptions,
    SqlitePathError,
    SqliteReadError,
    SqliteSchemaError,
    SqliteValueError,
    _quote,
    _render_cell,
    _translate,
)
from ingest.pipeline import IngestionPipeline, ingest_source, ledger_path
from ingest.dedup import DedupLedger
from ingest.records import SourceSpec


def _db(path: Path, script: list[str]) -> Path:
    connection = sqlite3.connect(path)
    try:
        connection.execute("PRAGMA foreign_keys=ON")
        for statement in script:
            connection.execute(statement)
        connection.commit()
    finally:
        connection.close()
    return path


def _wells(path: Path) -> Path:
    return _db(
        path,
        [
            "CREATE TABLE wells (id INTEGER PRIMARY KEY, name TEXT NOT NULL, depth INTEGER)",
            "INSERT INTO wells VALUES (1, 'A-1', 1000)",
            "INSERT INTO wells VALUES (2, 'A-2', 1200)",
            "INSERT INTO wells VALUES (3, 'A-3', 1400)",
        ],
    )


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


# ---------------------------------------------------------------------------
# schema discovery
# ---------------------------------------------------------------------------


def test_discovers_tables_columns_and_declared_types(tmp_path):
    database = _wells(tmp_path / "wells.db")
    schema = SqliteConnector().inspect(SourceSpec(f"sqlite://{database}"))
    assert [table.name for table in schema] == ["wells"]
    wells = schema[0]
    assert [column.name for column in wells.columns] == ["id", "name", "depth"]
    assert [column.declared_type for column in wells.columns] == ["INTEGER", "TEXT", "INTEGER"]
    assert wells.primary_key == ("id",)
    assert wells.has_primary_key
    assert not wells.composite_key


def test_discovers_composite_primary_key_in_order(tmp_path):
    database = _db(
        tmp_path / "composite.db",
        [
            "CREATE TABLE reading (well TEXT, taken_at TEXT, value REAL, PRIMARY KEY(well, taken_at))",
        ],
    )
    schema = SqliteConnector().inspect(SourceSpec(f"sqlite://{database}"))[0]
    assert schema.primary_key == ("well", "taken_at")
    assert schema.composite_key


def test_discovers_composite_foreign_key(tmp_path):
    database = _db(
        tmp_path / "fk.db",
        [
            "CREATE TABLE parent (a INTEGER, b TEXT, PRIMARY KEY(a, b))",
            "CREATE TABLE child (id INTEGER PRIMARY KEY, pa INTEGER, pb TEXT, "
            "FOREIGN KEY(pa, pb) REFERENCES parent(a, b))",
        ],
    )
    tables = {table.name: table for table in SqliteConnector().inspect(SourceSpec(f"sqlite://{database}"))}
    assert tables["child"].foreign_keys
    fk = tables["child"].foreign_keys[0]
    assert fk.columns == ("pa", "pb")
    assert fk.referenced_table == "parent"
    assert fk.referenced_columns == ("a", "b")


def test_flags_a_without_rowid_table(tmp_path):
    database = _db(
        tmp_path / "wr.db",
        [
            "CREATE TABLE wr (k TEXT PRIMARY KEY, v TEXT) WITHOUT ROWID",
            "INSERT INTO wr VALUES ('k1', 'v1')",
        ],
    )
    schema = SqliteConnector().inspect(SourceSpec(f"sqlite://{database}"))[0]
    assert schema.without_rowid
    assert schema.primary_key == ("k",)


def test_table_without_primary_key_is_read_but_has_no_stable_identity(tmp_path):
    database = _db(
        tmp_path / "nopk.db",
        ["CREATE TABLE log (msg TEXT)", "INSERT INTO log VALUES ('first')", "INSERT INTO log VALUES ('second')"],
    )
    schema = SqliteConnector().inspect(SourceSpec(f"sqlite://{database}"))[0]
    assert schema.primary_key == ()
    assert not schema.has_primary_key
    records = list(SqliteConnector().stream(SourceSpec(f"sqlite://{database}")))
    assert records
    # No row identity is invented: positional rows are numbered, not identified.
    assert records[0].metadata["stable_identity"] is False
    assert records[0].metadata["pk_columns"] == []


# ---------------------------------------------------------------------------
# read-only, never creating
# ---------------------------------------------------------------------------


def test_missing_path_is_a_clear_error_and_creates_nothing(tmp_path):
    missing = tmp_path / "does-not-exist.db"
    with pytest.raises(SqlitePathError):
        list(SqliteConnector().stream(SourceSpec(f"sqlite://{missing}")))
    assert not missing.exists()


def test_directory_path_is_refused(tmp_path):
    with pytest.raises(SqlitePathError):
        SqliteConnector().inspect(SourceSpec(f"sqlite://{tmp_path}"))


def test_import_leaves_the_source_file_unchanged(tmp_path):
    database = _wells(tmp_path / "wells.db")
    before = _digest(database)
    ingest_source(SourceSpec(f"sqlite://{database}"), store_dir=str(tmp_path / "kb"))
    assert _digest(database) == before


def test_read_only_connection_refuses_writes(tmp_path):
    database = _wells(tmp_path / "wells.db")
    connector = SqliteConnector()
    options = SqliteOptions()
    path = database.resolve()
    with connector._connect(path, options) as connection:  # noqa: SLF001 - asserting the read-only contract
        with pytest.raises(sqlite3.OperationalError):
            connection.execute("DELETE FROM wells")


def test_locked_database_reports_an_access_error(tmp_path):
    database = _db(
        tmp_path / "lock.db",
        ["CREATE TABLE t (a TEXT)", "INSERT INTO t VALUES ('x')"],
    )
    writer = sqlite3.connect(database)
    writer.execute("PRAGMA journal_mode=DELETE")
    writer.execute("BEGIN EXCLUSIVE")
    writer.execute("INSERT INTO t VALUES ('y')")
    try:
        with pytest.raises(SqliteAccessError):
            list(SqliteConnector().stream(SourceSpec(f"sqlite://{database}", options={"timeout": "0.3"})))
    finally:
        writer.rollback()
        writer.close()


def test_not_a_database_reports_a_schema_error(tmp_path):
    bad = tmp_path / "bad.db"
    bad.write_text("this is not a sqlite database", encoding="utf-8")
    with pytest.raises(SqliteSchemaError):
        SqliteConnector().inspect(SourceSpec(f"sqlite://{bad}"))


def test_malformed_utf8_value_reports_a_value_error(tmp_path):
    database = _db(tmp_path / "utf.db", ["CREATE TABLE t (a TEXT)"])
    connection = sqlite3.connect(database)
    connection.execute("INSERT INTO t VALUES (CAST(X'FFFE41' AS TEXT))")
    connection.commit()
    connection.close()
    with pytest.raises(SqliteValueError):
        list(SqliteConnector().stream(SourceSpec(f"sqlite://{database}")))


def test_translate_maps_an_interrupted_read_to_a_read_error():
    assert isinstance(_translate(sqlite3.OperationalError("interrupted"), "t"), SqliteReadError)
    assert isinstance(_translate(sqlite3.OperationalError("database is locked"), "t"), SqliteAccessError)
    assert isinstance(_translate(sqlite3.DatabaseError("file is not a database"), "t"), SqliteSchemaError)


# ---------------------------------------------------------------------------
# value rendering: no silent change of meaning
# ---------------------------------------------------------------------------


def test_arabic_and_english_text_is_preserved(tmp_path):
    database = _db(
        tmp_path / "text.db",
        [
            "CREATE TABLE notes (id INTEGER PRIMARY KEY, body TEXT)",
            "INSERT INTO notes VALUES (1, 'ضغط المسام drives oil flow')",
        ],
    )
    record = list(SqliteConnector().stream(SourceSpec(f"sqlite://{database}")))[0]
    assert "ضغط المسام drives oil flow" in record.text


def test_null_is_distinct_from_empty_text(tmp_path):
    database = _db(
        tmp_path / "nulls.db",
        [
            "CREATE TABLE t (id INTEGER PRIMARY KEY, note TEXT)",
            "INSERT INTO t VALUES (1, NULL)",
            "INSERT INTO t VALUES (2, '')",
        ],
    )
    record = list(SqliteConnector().stream(SourceSpec(f"sqlite://{database}")))[0]
    assert "note: NULL" in record.text
    assert "note: ''" in record.text


def test_numeric_and_datetime_values_keep_their_representation(tmp_path):
    database = _db(
        tmp_path / "types.db",
        [
            "CREATE TABLE t (id INTEGER PRIMARY KEY, amount REAL, ts TEXT, n INTEGER)",
            "INSERT INTO t VALUES (1, 1.5, '2023-01-02T03:04:05', 7)",
        ],
    )
    record = list(SqliteConnector().stream(SourceSpec(f"sqlite://{database}")))[0]
    assert "amount: 1.5" in record.text
    assert "ts: 2023-01-02T03:04:05" in record.text
    assert "n: 7" in record.text


def test_blob_is_rendered_as_hex_not_decoded(tmp_path):
    database = _db(tmp_path / "blob.db", ["CREATE TABLE t (id INTEGER PRIMARY KEY, b BLOB)"])
    connection = sqlite3.connect(database)
    connection.execute("INSERT INTO t VALUES (1, ?)", (b"\x00\xff",))
    connection.commit()
    connection.close()
    record = list(SqliteConnector().stream(SourceSpec(f"sqlite://{database}")))[0]
    assert "b: X'00ff'" in record.text


def test_render_cell_is_documented_and_stable():
    assert _render_cell(None) == "NULL"
    assert _render_cell("") == "''"
    assert _render_cell("x") == "x"
    assert _render_cell(7) == "7"
    assert _render_cell(1.5) == "1.5"
    assert _render_cell(b"\x01") == "X'01'"


# ---------------------------------------------------------------------------
# bounded streaming and option validation
# ---------------------------------------------------------------------------


def test_streams_in_bounded_row_groups(tmp_path):
    statements = ["CREATE TABLE n (i INTEGER PRIMARY KEY)"]
    statements += [f"INSERT INTO n VALUES ({i})" for i in range(25)]
    database = _db(tmp_path / "many.db", statements)
    spec = SourceSpec(f"sqlite://{database}", options={"row_group": "10", "page": "7"})
    records = list(SqliteConnector().stream(spec))
    assert len(records) == 3  # 25 rows, 10 per record
    assert [record.metadata["row_start"] for record in records] == [1, 11, 21]
    assert [record.metadata["row_end"] for record in records] == [10, 20, 25]


def test_page_and_row_group_are_validated(tmp_path):
    database = _wells(tmp_path / "w.db")
    for bad in ({"page": "0"}, {"page": "100001"}, {"row_group": "-1"}, {"page": "abc"}):
        with pytest.raises(SqliteOptionError):
            SqliteOptions.from_spec(SourceSpec(f"sqlite://{database}", options=bad))
    # Boundaries are accepted.
    SqliteOptions.from_spec(SourceSpec("sqlite://x", options={"page": str(MAX_PAGE)}))
    SqliteOptions.from_spec(SourceSpec("sqlite://x", options={"row_group": str(MAX_ROW_GROUP)}))


def test_plan_does_not_read_any_row(tmp_path):
    # Plan must be cheap: schema only, no data read. A table whose rows raise on
    # read (malformed text) still plans without touching the values.
    database = _db(tmp_path / "p.db", ["CREATE TABLE t (a TEXT)"])
    connection = sqlite3.connect(database)
    connection.execute("INSERT INTO t VALUES (CAST(X'FFFE41' AS TEXT))")
    connection.commit()
    connection.close()
    items = list(SqliteConnector().plan(SourceSpec(f"sqlite://{database}")))
    assert [item.title for item in items] == ["t"]


def test_plan_reports_no_fingerprint_so_no_incremental_skip(tmp_path):
    database = _wells(tmp_path / "w.db")
    item = next(iter(SqliteConnector().plan(SourceSpec(f"sqlite://{database}"))))
    # Incremental synchronisation is deliberately not implemented: no fingerprint.
    assert item.fingerprint == ""


# ---------------------------------------------------------------------------
# inclusion/exclusion rules and identifiers
# ---------------------------------------------------------------------------


def test_excluded_columns_are_not_read_or_rendered(tmp_path):
    database = _db(
        tmp_path / "sec.db",
        [
            "CREATE TABLE users (id INTEGER PRIMARY KEY, name TEXT, api_key TEXT)",
            "INSERT INTO users VALUES (1, 'Aisha', 'SUPER-SECRET-TOKEN')",
        ],
    )
    spec = SourceSpec(f"sqlite://{database}", options={"exclude_columns": "api_key"})
    records = list(SqliteConnector().stream(spec))
    assert records
    assert "SUPER-SECRET-TOKEN" not in records[0].text
    assert "api_key" not in records[0].metadata["columns"]


def test_include_columns_restricts_the_projection(tmp_path):
    database = _wells(tmp_path / "w.db")
    spec = SourceSpec(f"sqlite://{database}", options={"include_columns": "id, depth"})
    record = list(SqliteConnector().stream(spec))[0]
    assert record.metadata["columns"] == ["id", "depth"]
    assert "name" not in record.text


def test_table_inclusion_and_exclusion(tmp_path):
    database = _db(
        tmp_path / "two.db",
        [
            "CREATE TABLE keep (id INTEGER PRIMARY KEY, v TEXT)",
            "CREATE TABLE other (id INTEGER PRIMARY KEY, v TEXT)",
            "INSERT INTO keep VALUES (1, 'kept')",
            "INSERT INTO other VALUES (1, 'dropped')",
        ],
    )
    excluded = list(SqliteConnector().stream(SourceSpec(f"sqlite://{database}", options={"exclude_tables": "other"})))
    assert {record.metadata["table"] for record in excluded} == {"keep"}
    included = list(SqliteConnector().stream(SourceSpec(f"sqlite://{database}", options={"include_tables": "other"})))
    assert {record.metadata["table"] for record in included} == {"other"}


def test_quote_doubles_embedded_quotes_and_refuses_empty():
    assert _quote("wells") == '"wells"'
    assert _quote('a"b') == '"a""b"'
    with pytest.raises(SqliteOptionError):
        _quote("")


def test_identifiers_with_spaces_and_reserved_words_are_quoted(tmp_path):
    database = _db(
        tmp_path / "odd.db",
        [
            'CREATE TABLE "select" ("id" INTEGER PRIMARY KEY, "na me" TEXT)',
            'INSERT INTO "select" VALUES (1, \'value with space\')',
        ],
    )
    schema = SqliteConnector().inspect(SourceSpec(f"sqlite://{database}"))[0]
    assert schema.name == "select"
    record = list(SqliteConnector().stream(SourceSpec(f"sqlite://{database}")))[0]
    assert "na me: value with space" in record.text


def test_an_external_query_option_is_ignored(tmp_path):
    # Natural-language / operator SQL is never executed: a query= option is inert.
    database = _wells(tmp_path / "w.db")
    spec = SourceSpec(f"sqlite://{database}", options={"query": "DELETE FROM wells"})
    records = list(SqliteConnector().stream(spec))
    assert records  # the table is still read normally
    assert "A-1" in records[0].text


# ---------------------------------------------------------------------------
# pipeline integration (shared store, no second chunker/index)
# ---------------------------------------------------------------------------


def _pipeline(root: Path) -> tuple[KnowledgeBaseStore, IngestionPipeline]:
    store = KnowledgeBaseStore(root / "kb")
    return store, IngestionPipeline(store, DedupLedger(ledger_path(store.root)))


def test_pipeline_stores_sqlite_rows_in_the_shared_store(tmp_path):
    database = _wells(tmp_path / "wells.db")
    store, pipeline = _pipeline(tmp_path)
    report = pipeline.ingest(SourceSpec(f"sqlite://{database}"))
    assert report.failed == 0
    assert report.chunks >= 1
    assert store.count() == report.chunks
    chunk = next(iter(store.iter_chunks()))
    assert chunk.source == "sqlite"


def test_sqlite_rows_are_retrievable_through_the_fts_index(tmp_path):
    database = _db(
        tmp_path / "kb.db",
        [
            "CREATE TABLE topics (id INTEGER PRIMARY KEY, title TEXT, body TEXT)",
            "INSERT INTO topics VALUES (1, 'Permeability', 'Permeability measures fluid flow.')",
            "INSERT INTO topics VALUES (2, 'Porosity', 'Porosity is the void fraction of rock.')",
        ],
    )
    store, pipeline = _pipeline(tmp_path)
    pipeline.ingest(SourceSpec(f"sqlite://{database}"))
    build_index(store.root)
    retriever = BulkCorpusRetriever(
        lambda: iter_local_documents(store.root), index=KnowledgeTextIndex(store.root)
    )
    passages = retriever.retrieve("permeability fluid flow", limit=3)
    assert passages
    assert "Permeability" in passages[0].text


def test_excluded_column_never_reaches_the_knowledge_store(tmp_path):
    database = _db(
        tmp_path / "sec.db",
        [
            "CREATE TABLE accounts (id INTEGER PRIMARY KEY, holder TEXT, secret TEXT)",
            "INSERT INTO accounts VALUES (1, 'Aisha', 'DO-NOT-STORE-ME')",
        ],
    )
    store, pipeline = _pipeline(tmp_path)
    pipeline.ingest(SourceSpec(f"sqlite://{database}", options={"exclude_columns": "secret"}))
    stored = "\n".join(chunk.text for chunk in store.iter_chunks())
    assert "DO-NOT-STORE-ME" not in stored
    assert "Aisha" in stored


def test_identical_content_across_tables_is_deduplicated(tmp_path):
    database = _db(
        tmp_path / "dup.db",
        [
            "CREATE TABLE a (id INTEGER PRIMARY KEY, body TEXT)",
            "CREATE TABLE b (id INTEGER PRIMARY KEY, body TEXT)",
            "INSERT INTO a VALUES (1, 'The same paragraph repeated.')",
            "INSERT INTO b VALUES (1, 'The same paragraph repeated.')",
        ],
    )
    store, pipeline = _pipeline(tmp_path)
    report = pipeline.ingest(SourceSpec(f"sqlite://{database}"))
    assert report.records >= 2
    assert store.count() == 1  # content-addressed dedup, not a second store


def test_rerun_reads_again_but_stores_no_duplicate_chunks(tmp_path):
    database = _wells(tmp_path / "wells.db")
    store, pipeline = _pipeline(tmp_path)
    pipeline.ingest(SourceSpec(f"sqlite://{database}"))
    first = store.count()
    report = pipeline.ingest(SourceSpec(f"sqlite://{database}"))
    # No fingerprint, so nothing is skipped as "unchanged"...
    assert report.skipped_unchanged == 0
    # ...but the content hashes keep the second run from storing anything new.
    assert report.chunks == 0
    assert store.count() == first


def test_registry_resolves_sqlite_uri_to_the_sqlite_connector():
    from ingest.registry import build_default_registry

    registry = build_default_registry()
    assert registry.resolve(SourceSpec("sqlite:///data/wells.db")).info.name == "sqlite"
    assert registry.get("sqlite") is not None


def test_registry_connectors_have_distinct_schemes():
    from ingest.registry import build_default_registry

    infos = {info.name: info for info in build_default_registry().infos()}
    assert infos["sqlite"].schemes == frozenset({"sqlite", "sqlite3"})
    assert "sqlite" not in infos["sql"].schemes
    assert infos["sql"].optional_dependency == "psycopg"


def test_import_failure_leaves_the_source_unchanged(tmp_path):
    bad = tmp_path / "bad.db"
    bad.write_text("not a database at all", encoding="utf-8")
    before = _digest(bad)
    store, pipeline = _pipeline(tmp_path)
    report = pipeline.ingest(SourceSpec(f"sqlite://{bad}"))
    assert report.failed == 1
    assert pipeline.errors
    assert _digest(bad) == before


def test_locked_import_reports_failure_without_touching_the_source(tmp_path):
    database = _db(
        tmp_path / "lock.db",
        ["CREATE TABLE t (a TEXT)", "INSERT INTO t VALUES ('x')"],
    )
    before = _digest(database)
    writer = sqlite3.connect(database)
    writer.execute("PRAGMA journal_mode=DELETE")
    writer.execute("BEGIN EXCLUSIVE")
    writer.execute("INSERT INTO t VALUES ('y')")
    try:
        store, pipeline = _pipeline(tmp_path)
        report = pipeline.ingest(SourceSpec(f"sqlite://{database}", options={"timeout": "0.3"}))
        assert report.failed == 1
    finally:
        writer.rollback()
        writer.close()
    assert _digest(database) == before


def test_root_confinement_allows_relative_paths_inside_the_root(tmp_path):
    root = tmp_path / "data"
    root.mkdir()
    _wells(root / "wells.db")
    spec = SourceSpec("sqlite://wells.db", options={"root": str(root)})
    records = list(SqliteConnector().stream(spec))
    assert records
    assert "A-1" in records[0].text


def test_root_confinement_refuses_a_path_outside_the_root(tmp_path):
    root = tmp_path / "data"
    root.mkdir()
    _wells(tmp_path / "outside.db")
    spec = SourceSpec("sqlite://../outside.db", options={"root": str(root)})
    with pytest.raises(SqlitePathError):
        list(SqliteConnector().stream(spec))


def test_foreign_key_metadata_is_carried_onto_the_record(tmp_path):
    database = _db(
        tmp_path / "fk.db",
        [
            "CREATE TABLE parent (id INTEGER PRIMARY KEY)",
            "CREATE TABLE child (id INTEGER PRIMARY KEY, parent_id INTEGER, "
            "FOREIGN KEY(parent_id) REFERENCES parent(id))",
            "INSERT INTO parent VALUES (1)",
            "INSERT INTO child VALUES (1, 1)",
        ],
    )
    records = {
        record.metadata["table"]: record
        for record in SqliteConnector().stream(SourceSpec(f"sqlite://{database}"))
    }
    keys = records["child"].metadata["foreign_keys"]
    assert keys and keys[0]["referenced_table"] == "parent"
    assert keys[0]["columns"] == ["parent_id"]
