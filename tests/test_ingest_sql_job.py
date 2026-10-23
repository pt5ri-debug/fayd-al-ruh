"""Resumable, idempotent SQL imports: durable progress and safe recovery.

These tests drive the real connector, pipeline, store, and job store (no mocks):
each builds a small SQLite file, imports it through :class:`SqlImportRunner`, and
checks the durable checkpoint, the store contents, and the resume behaviour.
"""

from __future__ import annotations

import sqlite3

import pytest
from core.knowledge_base import KnowledgeBaseStore
from ingest.records import SourceSpec
from ingest.sql_job import (
    DEFAULT_BATCH_SIZE,
    INTERRUPTED,
    SCHEMA_FULL_REIMPORT,
    Interrupted,
    SqlImportJob,
    SqlImportRunner,
    SqlJobError,
    SqlJobStore,
    TableProgress,
    sql_jobs_dir,
)


def _make_db(path, rows: int = 25, *, columns: str = "id INTEGER PRIMARY KEY, v TEXT") -> str:
    connection = sqlite3.connect(path)
    try:
        connection.execute(f"CREATE TABLE t({columns})")
        connection.executemany(
            "INSERT INTO t(id, v) VALUES(?, ?)",
            [(i, f"value {i}") for i in range(1, rows + 1)],
        )
        connection.commit()
    finally:
        connection.close()
    return str(path)


def _spec(db_path) -> SourceSpec:
    return SourceSpec(f"sqlite://{db_path}", options={"source_path": str(db_path)})


def _runner(store_dir, **kwargs) -> SqlImportRunner:
    return SqlImportRunner(store_dir, **kwargs)


# -- job state -------------------------------------------------------------


def test_job_state_roundtrip(tmp_path):
    store = SqlJobStore(tmp_path / "jobs")
    job = SqlImportJob(job_id="j1", source_uri="sqlite:///x.db", batch_size=7)
    job.tables["t"] = TableProgress(table="t", signature="abc", resume_count=3)
    store.save(job)
    loaded = store.load("j1")
    assert loaded is not None
    assert loaded.batch_size == 7
    assert loaded.tables["t"].resume_count == 3
    assert loaded.updated_at


def test_job_state_requires_ids():
    with pytest.raises(ValueError):
        SqlImportJob(job_id="", source_uri="x")
    with pytest.raises(ValueError):
        SqlImportJob(job_id="j", source_uri="")


def test_job_store_list_and_remove(tmp_path):
    store = SqlJobStore(tmp_path / "jobs")
    store.save(SqlImportJob(job_id="a", source_uri="/a"))
    store.save(SqlImportJob(job_id="b", source_uri="/b"))
    assert {job.job_id for job in store.list()} == {"a", "b"}
    assert store.remove("a") is True
    assert store.remove("a") is False
    assert {job.job_id for job in store.list()} == {"b"}


def test_sql_jobs_live_inside_the_store(tmp_path):
    assert sql_jobs_dir(tmp_path) == tmp_path / "ingest" / "jobs-sql"


# -- full import and tracking ---------------------------------------------


def test_full_import_tracks_rows_and_marks_done(tmp_path):
    db = _make_db(tmp_path / "w.db", rows=25)
    store_dir = tmp_path / "kb"
    runner = _runner(store_dir, batch_size=10)
    job = runner.run(_spec(db), job_id="full")

    assert job.status == "done"
    progress = job.tables["t"]
    assert progress.status == "done"
    assert progress.resume_count == 25
    assert progress.target_rows == 25
    assert progress.signature  # recorded on completion
    # One chunk per row (one row per record, distinct text).
    assert KnowledgeBaseStore(store_dir).count() == 25


def test_batches_are_bounded_by_batch_size(tmp_path):
    db = _make_db(tmp_path / "w.db", rows=25)
    messages: list[str] = []
    runner = _runner(tmp_path / "kb", batch_size=10, progress=messages.append)
    runner.run(_spec(db), job_id="batches")
    committed = [m for m in messages if "committed" in m]
    # 25 rows / 10 per batch -> three committed windows.
    assert len(committed) == 3


def test_rerun_is_idempotent(tmp_path):
    db = _make_db(tmp_path / "w.db", rows=25)
    store_dir = tmp_path / "kb"
    runner = _runner(store_dir, batch_size=10)
    runner.run(_spec(db), job_id="once")
    before = KnowledgeBaseStore(store_dir).count()

    # A second run re-reads but stores nothing new (content-hash dedup).
    again = runner.run(_spec(db), job_id="once")
    after = KnowledgeBaseStore(store_dir).count()
    assert before == after == 25
    assert again.status == "done"
    assert again.tables["t"].resume_count == 25


# -- interruption and resume ----------------------------------------------


def test_interruption_keeps_committed_batch_and_resumes(tmp_path):
    db = _make_db(tmp_path / "w.db", rows=25)
    store_dir = tmp_path / "kb"
    calls = {"n": 0}

    def on_record(record, chunks):
        calls["n"] += 1
        if calls["n"] == 10:  # crash during the first batch, after 10 writes
            raise Interrupted("simulated crash")

    crashing = SqlImportRunner(store_dir, batch_size=10, on_record=on_record)
    job = crashing.run(_spec(db), job_id="resume")
    assert job.status == INTERRUPTED
    # The crash happened before the checkpoint advanced, but the writes are durable.
    assert job.tables["t"].resume_count == 0
    assert KnowledgeBaseStore(store_dir).count() == 10

    resumed = _runner(store_dir, batch_size=10).resume(_spec(db), "resume")
    assert resumed.status == "done"
    assert resumed.tables["t"].resume_count == 25
    # The re-read window is deduped: still one chunk per row.
    assert KnowledgeBaseStore(store_dir).count() == 25


def test_checkpoint_never_ahead_of_committed_data(tmp_path):
    """The invariant: resume_count is only advanced after a durably stored batch."""

    db = _make_db(tmp_path / "w.db", rows=8)
    store_dir = tmp_path / "kb"
    store = SqlJobStore(sql_jobs_dir(store_dir))

    def on_record(record, chunks):
        # While a batch is being written, the persisted checkpoint must not move.
        saved = store.load("inv")
        assert saved is not None
        assert saved.tables["t"].resume_count < 8

    runner = SqlImportRunner(store_dir, batch_size=4, on_record=on_record)
    job = runner.run(_spec(db), job_id="inv")
    assert job.status == "done"


# -- schema change ---------------------------------------------------------


def test_schema_change_is_refused_by_default(tmp_path):
    db = _make_db(tmp_path / "w.db", rows=5)
    store_dir = tmp_path / "kb"
    runner = _runner(store_dir, batch_size=10)
    runner.run(_spec(db), job_id="schema")

    connection = sqlite3.connect(db)
    connection.execute("ALTER TABLE t ADD COLUMN extra TEXT")
    connection.commit()
    connection.close()

    job = runner.run(_spec(db), job_id="schema")
    assert job.status == "failed"
    assert any("schema changed" in message for message in job.failures)


def test_schema_change_can_be_reimported_explicitly(tmp_path):
    db = _make_db(tmp_path / "w.db", rows=5)
    store_dir = tmp_path / "kb"
    runner = _runner(store_dir, batch_size=10)
    runner.run(_spec(db), job_id="schema2")

    connection = sqlite3.connect(db)
    connection.execute("ALTER TABLE t ADD COLUMN extra TEXT")
    connection.commit()
    connection.close()

    job = runner.run(_spec(db), job_id="schema2", schema_change=SCHEMA_FULL_REIMPORT)
    assert job.status == "done"
    assert job.tables["t"].status == "done"


def test_unknown_schema_policy_is_rejected(tmp_path):
    db = _make_db(tmp_path / "w.db", rows=2)
    with pytest.raises(ValueError):
        _runner(tmp_path / "kb").run(_spec(db), job_id="x", schema_change="whatever")


# -- data-only change does not look like a schema change ------------------


def test_data_only_change_keeps_signature(tmp_path):
    db = _make_db(tmp_path / "w.db", rows=5)
    store_dir = tmp_path / "kb"
    runner = _runner(store_dir, batch_size=10)
    job = runner.run(_spec(db), job_id="data")
    signature = job.tables["t"].signature

    # Insert more rows: the structure is identical, so the signature is stable.
    connection = sqlite3.connect(db)
    connection.execute("INSERT INTO t(id, v) VALUES(99, 'value 99')")
    connection.commit()
    connection.close()
    again = runner.run(_spec(db), job_id="data")
    assert again.status == "done"
    assert again.tables["t"].signature == signature


# -- resume of a missing job ----------------------------------------------


def test_resume_without_a_job_raises(tmp_path):
    db = _make_db(tmp_path / "w.db", rows=2)
    with pytest.raises(SqlJobError):
        _runner(tmp_path / "kb").resume(_spec(db), "does-not-exist")


# -- table plan / identity honesty ----------------------------------------


def test_describe_reports_identity_and_resume(tmp_path):
    from ingest.connectors.sqlite import SqliteConnector

    db = _make_db(tmp_path / "w.db", rows=3)
    plan = SqliteConnector().describe(_spec(db))[0]
    assert plan.total_rows == 3
    assert plan.primary_key == ("id",)
    assert plan.stable_identity is True
    assert plan.resume_supported is True
    assert plan.signature


def test_describe_without_primary_key_is_honest(tmp_path):
    from ingest.connectors.sqlite import SqliteConnector

    path = tmp_path / "nopk.db"
    connection = sqlite3.connect(path)
    connection.execute("CREATE TABLE t(a TEXT, b TEXT)")
    connection.executemany("INSERT INTO t VALUES(?, ?)", [(f"a{i}", f"b{i}") for i in range(4)])
    connection.commit()
    connection.close()

    plan = SqliteConnector().describe(_spec(path))[0]
    assert plan.stable_identity is False
    assert plan.resume_supported is True  # a rowid gives a deterministic order
    assert "no primary key" in plan.reason.lower()


# -- connector windows -----------------------------------------------------


def test_row_window_reads_exactly_the_batch(tmp_path):
    from ingest.connectors.sqlite import SqliteConnector

    db = _make_db(tmp_path / "w.db", rows=10)
    window = SourceSpec(
        f"sqlite://{db}",
        options={"resume_row": "3", "max_rows": "4", "row_group": "1"},
    )
    records = list(SqliteConnector().stream(window))
    assert [record.metadata["row_start"] for record in records] == [4, 5, 6, 7]
    assert [record.metadata["row_end"] for record in records] == [4, 5, 6, 7]


def test_schema_signature_detects_structure_not_data(tmp_path):
    from ingest.connectors.sqlite import SqliteConnector

    db = _make_db(tmp_path / "w.db", rows=3)
    connector = SqliteConnector()
    before = connector.describe(_spec(db))[0].signature

    connection = sqlite3.connect(db)
    connection.execute("INSERT INTO t(id, v) VALUES(10, 'x')")
    connection.commit()
    connection.close()
    assert connector.describe(_spec(db))[0].signature == before

    connection = sqlite3.connect(db)
    connection.execute("ALTER TABLE t ADD COLUMN extra TEXT")
    connection.commit()
    connection.close()
    assert connector.describe(_spec(db))[0].signature != before


# -- cancel ----------------------------------------------------------------


def test_cancel_stops_and_marks_cancelled(tmp_path):
    db = _make_db(tmp_path / "w.db", rows=25)
    store_dir = tmp_path / "kb"
    state = {"stop": False}

    def on_record(record, chunks):
        state["stop"] = True  # cancel after the first row

    runner = SqlImportRunner(
        store_dir,
        batch_size=10,
        on_record=on_record,
        should_cancel=lambda: state["stop"],
    )
    job = runner.run(_spec(db), job_id="cancel")
    assert job.status == "cancelled"
    assert KnowledgeBaseStore(store_dir).count() == 1


def test_batch_size_bounds_are_validated(tmp_path):
    with pytest.raises(ValueError):
        SqlImportRunner(tmp_path, batch_size=0)
    with pytest.raises(ValueError):
        SqlImportRunner(tmp_path, batch_size=DEFAULT_BATCH_SIZE + 10_000_000)


def test_missing_source_is_failed_not_raised(tmp_path):
    runner = _runner(tmp_path / "kb", batch_size=5)
    missing = SourceSpec(f"sqlite://{tmp_path / 'nope.db'}",
                         options={"source_path": str(tmp_path / "nope.db")})
    job = runner.run(missing, job_id="missing")
    assert job.status == "failed"
    assert job.failures
