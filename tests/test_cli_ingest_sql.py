"""Tests for the SQLite import commands in the ingestion CLI.

These drive the real :class:`gui.sql_import.SqlImportService` over temporary
SQLite databases, so they exercise the same service, storage, checkpoints, and
source identity the GUI uses -- not a second implementation.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest

from cli import ingest as ingest_cli


def _run(argv):
    return ingest_cli.main(argv)


def _db(tmp_path: Path, *, rows: int = 8, name: str = "wells.db") -> Path:
    """A small SQLite database with a sensitive column, for the tests below."""
    path = tmp_path / name
    connection = sqlite3.connect(path)
    connection.execute(
        "CREATE TABLE wells(id INTEGER PRIMARY KEY, name TEXT, api_key TEXT)"
    )
    connection.execute(
        "CREATE TABLE logs(id INTEGER PRIMARY KEY, well_id INTEGER, note TEXT)"
    )
    connection.executemany(
        "INSERT INTO wells(id, name, api_key) VALUES(?, ?, ?)",
        [(i, f"w{i}", f"secret{i}") for i in range(1, rows + 1)],
    )
    connection.executemany(
        "INSERT INTO logs(id, well_id, note) VALUES(?, ?, ?)",
        [(i, i, f"n{i}") for i in range(1, rows + 1)],
    )
    connection.commit()
    connection.close()
    return path


def _stored_text(store_dir: Path) -> str:
    from core.knowledge_base import KnowledgeBaseStore

    return " ".join(chunk.text for chunk in KnowledgeBaseStore(store_dir).iter_chunks())


# --- shared service identity ----------------------------------------------


def test_cli_and_gui_share_the_same_service() -> None:
    """The CLI must drive the GUI's service, never a private import path."""
    from gui.sql_import import SqlImportService

    assert ingest_cli.SqlImportService is SqlImportService


# --- drivers --------------------------------------------------------------


def test_sql_drivers_lists_sqlite(tmp_path, capsys):
    assert _run(["sql-drivers", "--store-dir", str(tmp_path / "kb")]) == 0
    out = capsys.readouterr().out
    assert "sqlite" in out
    assert "read-only" in out


def test_sql_drivers_json(tmp_path, capsys):
    assert _run(["sql-drivers", "--json", "--store-dir", str(tmp_path / "kb")]) == 0
    rows = json.loads(capsys.readouterr().out)
    assert rows and rows[0]["name"] == "sqlite"
    assert rows[0]["read_only"] is True
    assert rows[0]["enabled"] is True
    assert ".db" in rows[0]["suffixes"]


# --- inspect --------------------------------------------------------------


def test_sql_inspect_shows_schema_without_reading_rows(tmp_path, capsys):
    db = _db(tmp_path)
    assert _run(["sql-inspect", str(db), "--store-dir", str(tmp_path / "kb")]) == 0
    out = capsys.readouterr().out
    assert "wells" in out and "logs" in out
    assert "api_key" in out
    assert "PK id" in out


def test_sql_inspect_json(tmp_path, capsys):
    db = _db(tmp_path)
    assert _run(["sql-inspect", str(db), "--json", "--store-dir", str(tmp_path / "kb")]) == 0
    payload = json.loads(capsys.readouterr().out)
    names = {table["table"] for table in payload}
    assert {"wells", "logs"} <= names
    wells = next(table for table in payload if table["table"] == "wells")
    assert wells["primary_key"] == ["id"]
    assert wells["row_count"] == 8
    assert {column["name"] for column in wells["columns"]} >= {"id", "name", "api_key"}


def test_sql_inspect_missing_database_returns_connection_code(tmp_path, capsys):
    code = _run(["sql-inspect", str(tmp_path / "missing.db"), "--store-dir", str(tmp_path / "kb")])
    assert code == ingest_cli.SQL_EXIT_CONNECTION
    assert "No such SQLite database" in capsys.readouterr().out


def test_sql_inspect_not_a_database_returns_connection_code(tmp_path, capsys):
    bad = tmp_path / "notes.txt"
    bad.write_text("this is not a database", encoding="utf-8")
    code = _run(["sql-inspect", str(bad), "--store-dir", str(tmp_path / "kb")])
    assert code == ingest_cli.SQL_EXIT_CONNECTION
    assert capsys.readouterr().out.startswith("error:")


# --- preview --------------------------------------------------------------


def test_sql_preview_is_bounded_and_honours_columns(tmp_path, capsys):
    db = _db(tmp_path)
    code = _run(
        [
            "sql-preview",
            str(db),
            "--table",
            "wells",
            "--columns",
            "id,name",
            "--limit",
            "3",
            "--store-dir",
            str(tmp_path / "kb"),
        ]
    )
    assert code == 0
    out = capsys.readouterr().out
    assert "id: 1; name: w1" in out
    assert "secret1" not in out  # the excluded column is never read
    assert "showing at most 3 row(s)" in out


def test_sql_preview_rejects_a_bad_limit(tmp_path, capsys):
    db = _db(tmp_path)
    code = _run(
        ["sql-preview", str(db), "--table", "wells", "--limit", "0", "--store-dir", str(tmp_path / "kb")]
    )
    assert code == ingest_cli.SQL_EXIT_INVALID
    assert "--limit must be at least 1." in capsys.readouterr().out


# --- import ---------------------------------------------------------------


def test_sql_import_runs_and_reports_counts(tmp_path, capsys):
    db = _db(tmp_path)
    store_dir = tmp_path / "kb"
    code = _run(
        ["sql-import", str(db), "--job", "j1", "--batch-size", "3", "--store-dir", str(store_dir)]
    )
    assert code == 0
    out = capsys.readouterr().out
    assert "wells: done (8/8 rows)" in out
    assert "Done: 16 row(s)" in out


def test_sql_import_json_keeps_the_durable_job_shape(tmp_path, capsys):
    db = _db(tmp_path)
    code = _run(
        ["sql-import", str(db), "--job", "j1", "--quiet", "--json", "--store-dir", str(tmp_path / "kb")]
    )
    assert code == 0
    payload = json.loads(capsys.readouterr().out)
    # The shape the CLI always emitted (the durable job record).
    assert payload["job_id"] == "j1"
    assert payload["tables"]["wells"]["status"] == "done"
    assert payload["tables"]["wells"]["resume_count"] == 8


def test_sql_import_excludes_a_column_before_it_is_read(tmp_path):
    db = _db(tmp_path)
    store_dir = tmp_path / "kb"
    code = _run(
        [
            "sql-import",
            str(db),
            "--job",
            "j1",
            "--tables",
            "wells",
            "--exclude-columns",
            "api_key",
            "--quiet",
            "--store-dir",
            str(store_dir),
        ]
    )
    assert code == 0
    stored = _stored_text(store_dir)
    assert "secret1" not in stored
    assert "w1" in stored


def test_sql_import_selects_one_table(tmp_path, capsys):
    db = _db(tmp_path)
    store_dir = tmp_path / "kb"
    code = _run(
        ["sql-import", str(db), "--job", "j1", "--tables", "logs", "--quiet", "--store-dir", str(store_dir)]
    )
    assert code == 0
    text = _stored_text(store_dir)
    assert "n1" in text  # logs rows stored
    assert "w1" not in text  # wells deselected


def test_sql_import_derives_a_job_id_when_none_is_given(tmp_path, capsys):
    db = _db(tmp_path, rows=2)
    store_dir = tmp_path / "kb"
    assert _run(["sql-import", str(db), "--quiet", "--store-dir", str(store_dir)]) == 0
    capsys.readouterr()
    assert _run(["sql-jobs", "--json", "--store-dir", str(store_dir)]) == 0
    jobs = json.loads(capsys.readouterr().out)
    assert len(jobs) == 1
    assert jobs[0]["job_id"].startswith("sqlite-")


@pytest.mark.parametrize("option", ["--batch-size", "--row-group", "--page"])
def test_sql_import_rejects_invalid_numbers(tmp_path, capsys, option):
    db = _db(tmp_path)
    code = _run(["sql-import", str(db), option, "0", "--store-dir", str(tmp_path / "kb")])
    assert code == ingest_cli.SQL_EXIT_INVALID
    assert capsys.readouterr().out.startswith("error:")


def test_sql_import_rejects_an_out_of_range_batch_size(tmp_path, capsys):
    db = _db(tmp_path)
    from ingest.sql_job import MAX_BATCH_SIZE

    code = _run(
        [
            "sql-import",
            str(db),
            "--batch-size",
            str(MAX_BATCH_SIZE + 1),
            "--store-dir",
            str(tmp_path / "kb"),
        ]
    )
    assert code == ingest_cli.SQL_EXIT_INVALID


def test_sql_import_failure_returns_nonzero(tmp_path, capsys):
    # A missing database is a source failure the import cannot recover from.
    code = _run(
        ["sql-import", str(tmp_path / "missing.db"), "--job", "j1", "--quiet", "--store-dir", str(tmp_path / "kb")]
    )
    assert code == ingest_cli.SQL_EXIT_FAILED


def test_sql_import_progress_goes_to_stderr(tmp_path, capsys):
    db = _db(tmp_path)
    code = _run(["sql-import", str(db), "--job", "j1", "--store-dir", str(tmp_path / "kb")])
    assert code == 0
    captured = capsys.readouterr()
    assert "sql import" in captured.err  # progress never pollutes stdout
    assert "sql import" not in captured.out


# --- status, jobs ---------------------------------------------------------


def test_sql_status_reports_one_job(tmp_path, capsys):
    db = _db(tmp_path)
    store_dir = tmp_path / "kb"
    _run(["sql-import", str(db), "--job", "j1", "--quiet", "--store-dir", str(store_dir)])
    capsys.readouterr()
    assert _run(["sql-status", "j1", "--store-dir", str(store_dir)]) == 0
    out = capsys.readouterr().out
    assert "j1  [done]  16/16 rows" in out
    assert "resumable  : no" in out


def test_sql_status_unknown_job_is_invalid(tmp_path, capsys):
    assert _run(["sql-status", "nope", "--store-dir", str(tmp_path / "kb")]) == ingest_cli.SQL_EXIT_INVALID
    assert "no such SQL import job" in capsys.readouterr().out


def test_sql_jobs_lists_and_handles_empty(tmp_path, capsys):
    assert _run(["sql-jobs", "--store-dir", str(tmp_path / "kb")]) == 0
    assert "No SQL import jobs" in capsys.readouterr().out

    db = _db(tmp_path, rows=2)
    store_dir = tmp_path / "kb"
    _run(["sql-import", str(db), "--job", "listed", "--quiet", "--store-dir", str(store_dir)])
    capsys.readouterr()
    assert _run(["sql-jobs", "--store-dir", str(store_dir)]) == 0
    assert "listed [done]" in capsys.readouterr().out


# --- cancel and resume ----------------------------------------------------


def test_sql_cancel_unknown_job_is_invalid(tmp_path, capsys):
    assert _run(["sql-cancel", "nope", "--store-dir", str(tmp_path / "kb")]) == ingest_cli.SQL_EXIT_INVALID
    assert "no such SQL import job" in capsys.readouterr().out


def test_sql_cancel_requests_and_resume_completes(tmp_path, capsys):
    from ingest.sql_job import SqlImportJob, SqlJobStore, sql_jobs_dir

    db = _db(tmp_path, rows=20, name="big.db")
    store_dir = tmp_path / "kb"
    # Record the durable job, then ask it to stop before it starts.
    SqlJobStore(sql_jobs_dir(store_dir)).save(
        SqlImportJob(job_id="c1", source_uri=f"sqlite://{db}")
    )
    assert _run(["sql-cancel", "c1", "--store-dir", str(store_dir)]) == 0
    assert "Cancel requested" in capsys.readouterr().out

    code = _run(["sql-import", str(db), "--job", "c1", "--batch-size", "5", "--quiet", "--store-dir", str(store_dir)])
    assert code == ingest_cli.SQL_EXIT_INCOMPLETE
    assert "Cancelled:" in capsys.readouterr().out

    # The job stayed resumable and the resume finishes it.
    assert _run(["sql-resume", "c1", "--batch-size", "5", "--quiet", "--store-dir", str(store_dir)]) == 0
    out = capsys.readouterr().out
    assert "Done: 40 row(s)" in out


def test_sql_resume_unknown_job_is_invalid(tmp_path, capsys):
    assert _run(["sql-resume", "nope", "--store-dir", str(tmp_path / "kb")]) == ingest_cli.SQL_EXIT_INVALID
    assert "No such SQL import job" in capsys.readouterr().out


# --- safety ---------------------------------------------------------------


def test_sql_import_opens_the_source_read_only(tmp_path, capsys):
    db = _db(tmp_path)
    before = db.read_bytes()
    assert _run(["sql-import", str(db), "--job", "j1", "--quiet", "--store-dir", str(tmp_path / "kb")]) == 0
    assert db.read_bytes() == before  # the source is never written


def test_table_names_are_treated_as_names_not_sql(tmp_path, capsys):
    """A selection is a name filter, never executed as SQL: an injection is inert."""
    db = _db(tmp_path, rows=2)
    store_dir = tmp_path / "kb"
    payload = 'wells; DROP TABLE wells; --'
    code = _run(
        ["sql-import", str(db), "--job", "j1", "--tables", payload, "--quiet", "--store-dir", str(store_dir)]
    )
    # No table matches the odd name, so nothing is imported; the database is intact.
    assert code == 0
    connection = sqlite3.connect(db)
    tables = {row[0] for row in connection.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    connection.close()
    assert {"wells", "logs"} <= tables


def test_secrets_are_redacted_in_messages(tmp_path, capsys):
    """A credential-bearing URI in an error must not be printed verbatim."""
    from gui.sql_import import redact_secrets

    assert redact_secrets("password=hunter2") == "password=***"
    assert "hunter2" not in redact_secrets("sqlite://user:hunter2@host/db")


def test_commands_never_prompt_without_a_terminal(tmp_path, monkeypatch, capsys):
    """Automated environments must not block: every command returns on its own."""
    import io

    monkeypatch.setattr("sys.stdin", io.StringIO(""))
    db = _db(tmp_path, rows=2)
    for argv in (
        ["sql-drivers"],
        ["sql-inspect", str(db)],
        ["sql-preview", str(db), "--table", "wells"],
        ["sql-import", str(db), "--job", "j1", "--quiet"],
        ["sql-status", "j1"],
        ["sql-jobs"],
    ):
        assert _run([*argv, "--store-dir", str(tmp_path / "kb")]) == 0
