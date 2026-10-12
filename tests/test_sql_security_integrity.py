"""Security and source-integrity review of the SQL ingestion implementation.

Phase 4 — Step 6. These tests are an *external* check on the guarantees the SQL
import claims; they drive the real connector, runner, store, and CLI over
isolated temporary SQLite files (no mocks, no production database). Each test
names the property it pins:

1. SQLite sources open read-only; a source is never silently created or modified.
2. The importer issues no write operation against the source.
3. Identifiers are validated and quoted, and no caller value is interpolated.
4. No natural-language question reaches SQL execution.
5. Credentials never appear in a printed message (human or ``--json``).
6. Excluded columns never enter the knowledge store.
7. Imported data, job checkpoints, and the FTS index stay out of the ML dirs.
8. An interrupted import resumes without losing committed data or duplicating.
9. Schema changes and unsupported sync are explicit, not silent.
10. The source's schema and logical contents are unchanged (hash + row compare).

The source is compared before and after by schema signature, full logical content
(rows, keyed), and a SHA-256 file hash: the file hash is an *additional* signal,
not the only one, because a same-size rewrite would evade it.
"""

from __future__ import annotations

import hashlib
import io
import contextlib
import sqlite3
from pathlib import Path

import pytest

from cli import ingest as cli
from core.knowledge_base import KnowledgeBaseStore
from core.knowledge_index import KnowledgeTextIndex, build_index
from core.paths import artifacts_dir, checkpoints_dir
from ingest.connectors.sqlite import SqliteConnector, SqliteOptions, _quote
from ingest.pipeline import ledger_path
from ingest.records import SourceSpec
from ingest.sql_job import (
    SCHEMA_FAIL,
    SCHEMA_FULL_REIMPORT,
    Interrupted,
    SqlImportRunner,
    sql_jobs_dir,
)


# ---------------------------------------------------------------------------
# fixtures / helpers
# ---------------------------------------------------------------------------


def _db(path: Path, script: list[str]) -> Path:
    connection = sqlite3.connect(path)
    try:
        for statement in script:
            connection.execute(statement)
        connection.commit()
    finally:
        connection.close()
    return path


def _wells_with_secret(path: Path) -> Path:
    return _db(
        path,
        [
            "CREATE TABLE wells (id INTEGER PRIMARY KEY, name TEXT, depth INTEGER, api_key TEXT)",
            "INSERT INTO wells VALUES (1, 'A-1', 1000, 'SECRET-TOKEN-1')",
            "INSERT INTO wells VALUES (2, 'A-2', 1200, 'SECRET-TOKEN-2')",
            "INSERT INTO wells VALUES (3, 'A-3', 1400, 'SECRET-TOKEN-3')",
        ],
    )


def _file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _logical_contents(path: Path) -> dict[str, list[tuple]]:
    """Every table's rows, keyed by table name, ordered by rowid (a full compare)."""
    connection = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    try:
        tables = [
            row[0]
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'"
            )
        ]
        return {
            table: connection.execute(f"SELECT * FROM {_quote(table)} ORDER BY rowid").fetchall()
            for table in tables
        }
    finally:
        connection.close()


def _schema_sql(path: Path) -> dict[str, str]:
    connection = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    try:
        return {
            name: (sql or "")
            for name, sql in connection.execute(
                "SELECT name, sql FROM sqlite_master WHERE type IN ('table','index')"
            )
        }
    finally:
        connection.close()


# ---------------------------------------------------------------------------
# 1 & 2 & 10. read-only open, no source write, source unchanged
# ---------------------------------------------------------------------------


def test_open_refuses_every_write_statement(tmp_path):
    """Property 1 & 2: the read-only handle rejects INSERT/UPDATE/DELETE/DDL."""
    database = _wells_with_secret(tmp_path / "w.db")
    connector = SqliteConnector()
    with connector._connect(database.resolve(), SqliteOptions()) as connection:  # noqa: SLF001
        for statement in (
            "DELETE FROM wells",
            "INSERT INTO wells VALUES (4, 'x', 0, 'y')",
            "UPDATE wells SET name='z'",
            "DROP TABLE wells",
            "CREATE TABLE extra (a)",
        ):
            with pytest.raises(sqlite3.OperationalError):
                connection.execute(statement)


def test_import_changes_neither_schema_nor_contents_nor_file(tmp_path):
    """Property 10: full-file hash AND schema AND logical rows all unchanged."""
    database = _wells_with_secret(tmp_path / "w.db")
    before_hash = _file_hash(database)
    before_schema = _schema_sql(database)
    before_rows = _logical_contents(database)

    runner = SqlImportRunner(tmp_path / "kb", batch_size=2)
    job = runner.run(SourceSpec(f"sqlite://{database}"), job_id="unchanged")
    assert job.status == "done"

    assert _file_hash(database) == before_hash
    assert _schema_sql(database) == before_schema
    assert _logical_contents(database) == before_rows


def test_missing_source_is_never_created_by_any_entry_point(tmp_path):
    """Property 1: no code path silently creates the database file."""
    missing = tmp_path / "absent.db"
    spec = SourceSpec(f"sqlite://{missing}")
    with pytest.raises(Exception):
        list(SqliteConnector().stream(spec))
    SqlImportRunner(tmp_path / "kb").run(spec, job_id="missing")
    assert not missing.exists()


# ---------------------------------------------------------------------------
# 3. identifiers validated/quoted; values never interpolated
# ---------------------------------------------------------------------------


def test_quote_doubles_quotes_and_rejects_empty_or_nul():
    assert _quote('a"b') == '"a""b"'
    assert _quote("select") == '"select"'
    with pytest.raises(Exception):
        _quote("")
    with pytest.raises(Exception):
        _quote("bad\x00name")


def test_hostile_table_and_column_names_cannot_inject_sql(tmp_path):
    """A table/column literally named as SQL is quoted, never executed."""
    database = _db(
        tmp_path / "hostile.db",
        [
            'CREATE TABLE "wells; DROP TABLE wells" ('
            '"id" INTEGER PRIMARY KEY, "name) VALUES (1); --" TEXT)',
            'INSERT INTO "wells; DROP TABLE wells" VALUES (1, \'safe\')',
        ],
    )
    records = list(SqliteConnector().stream(SourceSpec(f"sqlite://{database}")))
    assert records
    assert "safe" in records[0].text
    # The malicious identifier was read as data structure, not executed: the
    # table still exists under its literal name (it was never dropped).
    assert "wells; DROP TABLE wells" in _schema_sql(database)


def test_a_query_option_is_inert_and_the_table_still_reads(tmp_path):
    """Property 3: caller-supplied text is never a SQL statement."""
    database = _wells_with_secret(tmp_path / "w.db")
    spec = SourceSpec(
        f"sqlite://{database}",
        options={"query": "DROP TABLE wells", "exclude_columns": "api_key"},
    )
    records = list(SqliteConnector().stream(spec))
    assert records and "A-1" in records[0].text
    assert _logical_contents(database)["wells"]  # still present


# ---------------------------------------------------------------------------
# 4. no natural-language question can cause SQL execution
# ---------------------------------------------------------------------------


def test_natural_language_never_reaches_sql_execution(tmp_path, monkeypatch):
    """Property 4: only the internal read-only SELECT/PRAGMA is ever executed."""
    import ingest.connectors.sqlite as sqlite_mod

    database = _wells_with_secret(tmp_path / "w.db")
    executed: list[str] = []

    class _RecordingCursor:
        def __init__(self, inner):
            self._inner = inner

        def execute(self, sql, *args, **kwargs):
            executed.append(str(sql))
            return self._inner.execute(sql, *args, **kwargs)

        def __getattr__(self, name):
            return getattr(self._inner, name)

    class _RecordingConnection:
        def __init__(self, inner):
            self._inner = inner

        def execute(self, sql, *args, **kwargs):
            executed.append(str(sql))
            return self._inner.execute(sql, *args, **kwargs)

        def cursor(self):
            return _RecordingCursor(self._inner.cursor())

        def close(self):
            self._inner.close()

        def __getattr__(self, name):
            return getattr(self._inner, name)

    real_open = sqlite_mod._open_read_only  # noqa: SLF001
    monkeypatch.setattr(
        sqlite_mod,
        "_open_read_only",
        lambda path, timeout: _RecordingConnection(real_open(path, timeout)),
    )

    runner = SqlImportRunner(tmp_path / "kb", batch_size=5)
    runner.run(SourceSpec(f"sqlite://{database}"), job_id="nl")

    assert executed  # something was read
    forbidden = ("insert", "update", "delete", "drop", "alter", "create", "attach")
    for statement in executed:
        lowered = statement.casefold()
        assert not any(word in lowered for word in forbidden), statement
        assert lowered.lstrip().startswith(("select", "pragma", "with")), statement


# ---------------------------------------------------------------------------
# 5. credentials never printed (service redaction + CLI redaction)
# ---------------------------------------------------------------------------


def _run_cli(argv: list[str]) -> tuple[int, str, str]:
    out, err = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        code = cli.main(argv)
    return code, out.getvalue(), err.getvalue()


def test_cli_never_prints_a_credential_from_a_source_uri(tmp_path):
    """Property 5: import/status/jobs, human and JSON, redact the source URI."""
    store = tmp_path / "kb"
    dsn = "sqlite://user:hunter2@host/wells.db"
    _run_cli(["sql-import", dsn, "--job", "leaky", "--quiet", "--store-dir", str(store)])

    for argv in (
        ["sql-import", dsn, "--job", "l2", "--quiet"],
        ["sql-import", dsn, "--job", "l3", "--quiet", "--json"],
        ["sql-status", "leaky"],
        ["sql-status", "leaky", "--json"],
        ["sql-jobs"],
        ["sql-jobs", "--json"],
    ):
        _code, out, err = _run_cli([*argv, "--store-dir", str(store)])
        assert "hunter2" not in out, argv
        assert "hunter2" not in err, argv


def test_service_redacts_failures_and_never_stores_a_live_secret(tmp_path):
    """Property 5: the report's failures are redacted before a caller sees them."""
    from gui.sql_import import SqlImportService

    service = SqlImportService(store_dir=tmp_path / "kb")
    report = service.run_import(
        "sqlite://user:hunter2@host/nowhere.db", job_id="fail"
    )
    assert report.failed
    assert all("hunter2" not in failure for failure in report.failures)
    assert "hunter2" not in report.summary()


# ---------------------------------------------------------------------------
# 6. excluded columns never enter the store
# ---------------------------------------------------------------------------


def test_excluded_column_never_reaches_the_store_or_the_index(tmp_path):
    """Property 6: an excluded sensitive column is neither read nor stored/indexed."""
    database = _wells_with_secret(tmp_path / "w.db")
    store_dir = tmp_path / "kb"
    spec = SourceSpec(f"sqlite://{database}", options={"exclude_columns": "api_key"})
    runner = SqlImportRunner(store_dir, batch_size=5)
    assert runner.run(spec, job_id="excl").status == "done"

    store = KnowledgeBaseStore(store_dir)
    stored = "\n".join(chunk.text for chunk in store.iter_chunks())
    assert "SECRET-TOKEN" not in stored
    assert "api_key" not in stored

    build_index(store_dir)
    index = KnowledgeTextIndex(store_dir)
    hits = index.search("SECRET-TOKEN", limit=5)
    assert all("SECRET-TOKEN" not in chunk.text for _score, chunk in hits)


# ---------------------------------------------------------------------------
# 7. ingestion data / checkpoints / index stay out of the ML dirs
# ---------------------------------------------------------------------------


def test_import_writes_stay_out_of_training_and_artifact_dirs(tmp_path, monkeypatch):
    """Property 7: imported data, job files, and the index live under the store.

    A training-artifact sentinel under the ML artifact dir must be byte-identical
    after an import plus an index build, and the job files must sit under the
    knowledge store -- not under ``ml/artifacts`` or ``ml/llm/checkpoints``.
    """
    monkeypatch.setenv("FAYD_DATA_DIR", str(tmp_path / "fayd-data"))
    database = _wells_with_secret(tmp_path / "w.db")
    store_dir = tmp_path / "kb"

    artifacts, checkpoints = artifacts_dir(), checkpoints_dir()
    # The ML dirs are disjoint from the default knowledge store and from ours.
    assert not artifacts.is_relative_to(store_dir)
    assert not checkpoints.is_relative_to(store_dir)

    artifacts.mkdir(parents=True, exist_ok=True)
    sentinel = artifacts / "release_manifest.json"
    sentinel.write_text('{"trained": true}', encoding="utf-8")
    sentinel_hash = _file_hash(sentinel)

    SqlImportRunner(store_dir, batch_size=5).run(
        SourceSpec(f"sqlite://{database}"), job_id="sep"
    )
    build_index(store_dir)

    # The training artifact is untouched, and nothing was written beside it.
    assert _file_hash(sentinel) == sentinel_hash
    assert list(artifacts.iterdir()) == [sentinel]
    assert not checkpoints.exists() or list(checkpoints.iterdir()) == []

    # All ingestion state is under the knowledge store.
    jobs_dir = sql_jobs_dir(store_dir)
    assert jobs_dir == store_dir / "ingest" / "jobs-sql"
    assert jobs_dir.is_relative_to(store_dir)
    assert any(jobs_dir.glob("*.sqljob.json"))
    assert (store_dir / "index.sqlite").exists()
    assert ledger_path(KnowledgeBaseStore(store_dir).root).is_relative_to(store_dir)


# ---------------------------------------------------------------------------
# 8. interruption: committed data kept, resumed, no duplication
# ---------------------------------------------------------------------------


def test_interrupt_then_resume_keeps_data_without_duplicates(tmp_path):
    """Property 8: commit-then-checkpoint means resume replays for free."""
    database = _db(
        tmp_path / "w.db",
        ["CREATE TABLE t (id INTEGER PRIMARY KEY, v TEXT)"]
        + [f"INSERT INTO t VALUES ({i}, 'value {i}')" for i in range(1, 26)],
    )
    store_dir = tmp_path / "kb"
    calls = {"n": 0}

    def on_record(_record, _chunks):
        calls["n"] += 1
        if calls["n"] == 10:
            raise Interrupted("simulated crash")

    crashed = SqlImportRunner(store_dir, batch_size=10, on_record=on_record).run(
        SourceSpec(f"sqlite://{database}"), job_id="res"
    )
    assert crashed.status == "interrupted"
    committed_after_crash = KnowledgeBaseStore(store_dir).count()
    assert committed_after_crash == 10

    resumed = SqlImportRunner(store_dir, batch_size=10).resume(
        SourceSpec(f"sqlite://{database}"), "res"
    )
    assert resumed.status == "done"
    assert resumed.tables["t"].resume_count == 25
    # Committed data survived and the replayed window produced no duplicate chunk.
    assert KnowledgeBaseStore(store_dir).count() == 25

    ids = [chunk.chunk_id for chunk in KnowledgeBaseStore(store_dir).iter_chunks()]
    assert len(ids) == len(set(ids)) == 25


# ---------------------------------------------------------------------------
# 9. schema change / unsupported sync are explicit
# ---------------------------------------------------------------------------


def test_schema_change_is_refused_without_explicit_policy(tmp_path):
    database = _wells_with_secret(tmp_path / "w.db")
    store_dir = tmp_path / "kb"
    SqlImportRunner(store_dir, batch_size=5).run(
        SourceSpec(f"sqlite://{database}"), job_id="sch"
    )
    # Structural change: add a column (data-only edits must NOT trip this).
    connection = sqlite3.connect(database)
    connection.execute("ALTER TABLE wells ADD COLUMN extra TEXT")
    connection.commit()
    connection.close()

    job = SqlImportRunner(store_dir, batch_size=5).run(
        SourceSpec(f"sqlite://{database}"), job_id="sch", schema_change=SCHEMA_FAIL
    )
    assert job.status == "failed"
    assert any("schema changed" in failure for failure in job.failures)


def test_full_reimport_policy_is_explicit_and_recovers(tmp_path):
    database = _wells_with_secret(tmp_path / "w.db")
    store_dir = tmp_path / "kb"
    SqlImportRunner(store_dir, batch_size=5).run(
        SourceSpec(f"sqlite://{database}"), job_id="full"
    )
    connection = sqlite3.connect(database)
    connection.execute("ALTER TABLE wells ADD COLUMN extra TEXT")
    connection.commit()
    connection.close()

    job = SqlImportRunner(store_dir, batch_size=5).run(
        SourceSpec(f"sqlite://{database}"),
        job_id="full",
        schema_change=SCHEMA_FULL_REIMPORT,
    )
    assert job.status == "done"
    assert job.tables["wells"].resume_count == 3


def test_unsupported_sync_is_reported_not_claimed(tmp_path):
    """A table with no primary key is read but has no stable row identity."""
    database = _db(
        tmp_path / "nopk.db",
        ["CREATE TABLE t (a TEXT, b TEXT)", "INSERT INTO t VALUES ('x', 'y')"],
    )
    plans = SqliteConnector().describe(SourceSpec(f"sqlite://{database}"))
    assert plans[0].stable_identity is False
    assert plans[0].reason  # an honest explanation is always provided


# ---------------------------------------------------------------------------
# 10 (cont.) source integrity is checked by content, not the hash alone
# ---------------------------------------------------------------------------


def test_source_integrity_compared_by_content_and_hash(tmp_path):
    """A same-size rewrite evades a bare file hash; the row compare catches it."""
    database = _wells_with_secret(tmp_path / "w.db")
    before_hash = _file_hash(database)
    before_rows = _logical_contents(database)

    SqlImportRunner(tmp_path / "kb", batch_size=5).run(
        SourceSpec(f"sqlite://{database}"), job_id="integ"
    )

    assert _file_hash(database) == before_hash
    assert _logical_contents(database) == before_rows
    # And the pipeline never holds the source open (a write now succeeds).
    connection = sqlite3.connect(database)
    try:
        connection.execute("BEGIN IMMEDIATE")
        connection.rollback()
    finally:
        connection.close()


def test_no_prompt_and_no_terminal_input(tmp_path, monkeypatch):
    """Commands are non-interactive: a missing TTY never blocks a run."""
    store = tmp_path / "kb"
    database = _wells_with_secret(tmp_path / "w.db")
    monkeypatch.setattr("sys.stdin", None)
    code, _out, _err = _run_cli(
        ["sql-import", str(database), "--job", "np", "--quiet", "--store-dir", str(store)]
    )
    assert code == 0
