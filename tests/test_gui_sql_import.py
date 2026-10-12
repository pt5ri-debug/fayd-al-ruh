"""The SQL Database Import front end: connection, schema, selection, jobs.

These tests drive :class:`gui.sql_import.SqlImportService` -- the layer the
window's SQL import screen uses -- against temporary SQLite databases (never a
real database). They assert the behaviour the requirements call for:

* an invalid path and a failed connection are reported, not raised;
* the schema display lists tables, columns, declared types, and keys;
* excluded columns never enter a record, a preview, or the knowledge store;
* a reviewed selection imports through the shared resumable service;
* a job failure is reported with an actionable message and can be retried;
* cancel, resume, and retry use the shared durable checkpoint.

The window parts are skipped unless a display (or Xvfb) is available, exactly
like ``tests/test_gui.py``.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from core.knowledge_base import KnowledgeBaseStore
from gui.sql_import import (
    DATABASE_TYPES,
    MAX_PREVIEW_ROWS,
    SCHEMA_FULL_REIMPORT,
    SqlImportService,
    SqlProgress,
    redact_secrets,
)
from ingest.sql_job import sql_jobs_dir


def _make_db(path: Path, rows: int = 20, *, schema: str = "id INTEGER PRIMARY KEY, v TEXT") -> Path:
    connection = sqlite3.connect(path)
    try:
        connection.execute(f"CREATE TABLE wells({schema})")
        names = [column.split()[0] for column in schema.split(",")]
        if len(names) == 2:
            connection.executemany(
                f"INSERT INTO wells({names[0]}, {names[1]}) VALUES(?, ?)",
                [(i, f"value {i}") for i in range(1, rows + 1)],
            )
        connection.commit()
    finally:
        connection.close()
    return path


def _service(tmp_path: Path) -> SqlImportService:
    return SqlImportService(store_dir=tmp_path / "kb")


# --- connection and schema ------------------------------------------------


def test_only_sqlite_is_enabled() -> None:
    assert DATABASE_TYPES == ("SQLite",)


def test_invalid_path_is_reported_not_raised(tmp_path: Path) -> None:
    service = _service(tmp_path)
    check = service.test_connection(tmp_path / "missing.db")
    assert check.ok is False
    assert "missing.db" in check.message


def test_directory_path_is_refused(tmp_path: Path) -> None:
    service = _service(tmp_path)
    check = service.test_connection(tmp_path)
    assert check.ok is False
    assert "database file" in check.message.lower()


def test_a_file_that_is_not_a_database_is_reported(tmp_path: Path) -> None:
    service = _service(tmp_path)
    bogus = tmp_path / "not-a-db.db"
    bogus.write_text("this is not a database", encoding="utf-8")
    check = service.test_connection(bogus)
    assert check.ok is False
    assert check.message


def test_empty_path_asks_for_a_file(tmp_path: Path) -> None:
    check = _service(tmp_path).test_connection("")
    assert check.ok is False
    assert "choose a database" in check.message.lower()


def test_schema_display_lists_tables_columns_types_and_keys(tmp_path: Path) -> None:
    db = _make_db(tmp_path / "wells.db", rows=3, schema="id INTEGER PRIMARY KEY, v TEXT")
    infos = _service(tmp_path).inspect(db)
    assert [info.name for info in infos] == ["wells"]
    wells = infos[0]
    assert wells.primary_key == ("id",)
    assert wells.stable_identity is True
    assert wells.resume_supported is True
    assert wells.row_count == 3
    by_name = {column.name: column for column in wells.columns}
    assert by_name["id"].primary_key is True
    assert by_name["id"].declared_type == "INTEGER"
    assert by_name["v"].declared_type == "TEXT"


def test_schema_shows_a_composite_primary_key(tmp_path: Path) -> None:
    db = _make_db(
        tmp_path / "wells.db", rows=2, schema="a INTEGER, b INTEGER, v TEXT, PRIMARY KEY(a, b)"
    )
    info = _service(tmp_path).inspect(db)[0]
    assert info.primary_key == ("a", "b")
    assert info.stable_identity is True


def test_table_without_primary_key_reports_no_stable_identity(tmp_path: Path) -> None:
    db = _make_db(tmp_path / "wells.db", rows=2, schema="a INTEGER, v TEXT")
    info = _service(tmp_path).inspect(db)[0]
    assert info.primary_key == ()
    assert info.stable_identity is False
    # A rowid still gives a deterministic order, so resume-by-offset is offered.
    assert info.resume_supported is True


# --- preview and exclusions ----------------------------------------------


def test_preview_is_bounded_and_uses_the_connectors_rendering(tmp_path: Path) -> None:
    db = _make_db(tmp_path / "wells.db", rows=20)
    preview = _service(tmp_path).preview(db, table="wells", limit=3)
    assert len(preview.rows) == 3
    assert preview.truncated is True
    assert preview.columns == ("id", "v")
    # The connector's documented identity prefix is present, so the preview is
    # exactly how a row will be stored.
    assert preview.rows[0].startswith("[id=1]")


def test_preview_limit_is_capped(tmp_path: Path) -> None:
    db = _make_db(tmp_path / "wells.db", rows=200)
    preview = _service(tmp_path).preview(db, table="wells", limit=9999)
    assert len(preview.rows) <= MAX_PREVIEW_ROWS


def test_excluded_column_is_never_previewed(tmp_path: Path) -> None:
    db = _make_db(tmp_path / "wells.db", rows=5, schema="id INTEGER PRIMARY KEY, secret TEXT")
    preview = _service(tmp_path).preview(db, table="wells", columns=("id",))
    assert "secret" not in preview.columns
    assert all("secret" not in row for row in preview.rows)


def test_include_columns_restricts_the_preview(tmp_path: Path) -> None:
    db = _make_db(tmp_path / "wells.db", rows=5, schema="id INTEGER PRIMARY KEY, a TEXT, b TEXT")
    preview = _service(tmp_path).preview(db, table="wells", columns=("id", "a"))
    assert preview.columns == ("id", "a")


# --- successful import ----------------------------------------------------


def test_import_stores_rows_in_the_shared_store(tmp_path: Path) -> None:
    db = _make_db(tmp_path / "wells.db", rows=20)
    service = _service(tmp_path)
    report = service.run_import(db, job_id="j1", batch_size=7)
    assert report.status == "done"
    assert report.records == 20
    assert report.chunks == 20
    assert KnowledgeBaseStore(tmp_path / "kb").count() == 20
    assert report.summary().startswith("Done:")


def test_import_is_idempotent(tmp_path: Path) -> None:
    db = _make_db(tmp_path / "wells.db", rows=10)
    service = _service(tmp_path)
    service.run_import(db, job_id="j1", batch_size=4)
    before = KnowledgeBaseStore(tmp_path / "kb").count()
    service.run_import(db, job_id="j1", batch_size=4)
    assert KnowledgeBaseStore(tmp_path / "kb").count() == before


def test_excluded_column_never_reaches_the_store(tmp_path: Path) -> None:
    db = _make_db(tmp_path / "wells.db", rows=10, schema="id INTEGER PRIMARY KEY, api TEXT")
    service = _service(tmp_path)
    report = service.run_import(db, job_id="j1", exclude_columns=("api",))
    assert report.status == "done"
    text = "\n".join(doc.text for doc in KnowledgeBaseStore(tmp_path / "kb").iter_documents())
    assert "value" not in text  # the connector renders `api: API(<i>)`; ensure excluded
    assert "api" not in text
    assert "id" in text


def test_selected_tables_only_are_imported(tmp_path: Path) -> None:
    db = tmp_path / "wells.db"
    connection = sqlite3.connect(db)
    connection.execute("CREATE TABLE a(id INTEGER PRIMARY KEY, v TEXT)")
    connection.execute("CREATE TABLE b(id INTEGER PRIMARY KEY, w TEXT)")
    connection.executemany("INSERT INTO a(id, v) VALUES(?, ?)", [(i, f"a{i}") for i in range(3)])
    connection.executemany("INSERT INTO b(id, w) VALUES(?, ?)", [(i, f"b{i}") for i in range(4)])
    connection.commit()
    connection.close()
    report = _service(tmp_path).run_import(db, job_id="sel", tables=("a",))
    assert report.status == "done"
    assert {table.name for table in report.tables} == {"a"}
    assert report.records == 3


def test_import_job_options_are_recorded_for_resume(tmp_path: Path) -> None:
    db = _make_db(tmp_path / "wells.db", rows=5, schema="id INTEGER PRIMARY KEY, secret TEXT")
    service = _service(tmp_path)
    service.run_import(db, job_id="j1", exclude_columns=("secret",))
    view = service.job("j1")
    assert view is not None
    # The recorded options survive on disk, so a resume reproduces the policy.
    from ingest.sql_job import SqlJobStore

    raw = SqlJobStore(sql_jobs_dir(tmp_path / "kb")).load("j1")
    assert raw is not None
    assert raw.options.get("exclude_columns") == "secret"


# --- progress -------------------------------------------------------------


def test_progress_is_reported_with_a_measurable_percent(tmp_path: Path) -> None:
    db = _make_db(tmp_path / "wells.db", rows=20)
    seen: list[SqlProgress] = []
    _service(tmp_path).run_import(
        db, job_id="j1", batch_size=5, progress=seen.append
    )
    assert seen
    assert seen[-1].committed == 20
    assert seen[-1].total == 20
    assert seen[-1].percent == 100
    assert any(snapshot.table == "wells" for snapshot in seen)


# --- failures, cancel, resume, retry --------------------------------------


def test_missing_database_reports_an_actionable_failure(tmp_path: Path) -> None:
    service = _service(tmp_path)
    report = service.run_import(tmp_path / "missing.db", job_id="j1")
    assert report.status == "failed"
    assert report.failures
    assert "missing.db" in report.failures[0]
    assert report.can_resume is True


def test_cancel_stops_and_keeps_committed_work(tmp_path: Path) -> None:
    db = _make_db(tmp_path / "wells.db", rows=20)
    service = _service(tmp_path)
    report = service.run_import(db, job_id="j1", batch_size=5, should_cancel=lambda: True)
    assert report.status == "cancelled"
    assert report.cancelled is True
    # Nothing beyond the already-committed work is claimed.
    assert report.summary().startswith("Cancelled:")


def test_resume_continues_from_the_checkpoint(tmp_path: Path) -> None:
    db = _make_db(tmp_path / "wells.db", rows=20)
    service = _service(tmp_path)
    service.run_import(db, job_id="j1", batch_size=5, should_cancel=lambda: True)
    view = service.job("j1")
    assert view is not None and view.can_resume
    report = service.resume_job("j1")
    assert report.status == "done"
    assert report.records == 20


def test_retry_after_a_schema_change_failure(tmp_path: Path) -> None:
    db = _make_db(tmp_path / "wells.db", rows=5, schema="id INTEGER PRIMARY KEY, v TEXT")
    service = _service(tmp_path)
    service.run_import(db, job_id="j1", batch_size=2)
    # Change the structure: a resume must refuse until an explicit policy is set.
    connection = sqlite3.connect(db)
    connection.execute("ALTER TABLE wells ADD COLUMN extra TEXT")
    connection.commit()
    connection.close()
    failed = service.resume_job("j1")
    assert failed.status == "failed"
    assert failed.failures and "schema changed" in failed.failures[0]
    # Retrying with the explicit recovery policy re-imports successfully.
    recovered = service.resume_job("j1", schema_change=SCHEMA_FULL_REIMPORT)
    assert recovered.status == "done"
    assert recovered.records == 5


def test_resume_of_an_unknown_job_raises(tmp_path: Path) -> None:
    service = _service(tmp_path)
    with pytest.raises(ValueError):
        service.resume_job("does-not-exist")


# --- jobs bookkeeping -----------------------------------------------------


def test_jobs_are_listed_and_can_be_forgotten(tmp_path: Path) -> None:
    db = _make_db(tmp_path / "wells.db", rows=4)
    service = _service(tmp_path)
    service.run_import(db, job_id="j1")
    views = service.list_jobs()
    assert [view.job_id for view in views] == ["j1"]
    assert views[0].committed == 4
    # A completed job is not resumable (there is nothing left to read).
    assert views[0].status == "done"
    assert views[0].can_resume is False
    assert service.forget_job("j1") is True
    assert service.list_jobs() == ()
    assert service.forget_job("j1") is False


def test_default_job_id_is_stable_and_path_specific(tmp_path: Path) -> None:
    service = _service(tmp_path)
    a = tmp_path / "a.db"
    b = tmp_path / "b.db"
    assert service.default_job_id(a) == service.default_job_id(a)
    assert service.default_job_id(a) != service.default_job_id(b)


# --- secrets --------------------------------------------------------------


def test_redact_secrets_removes_passwords_tokens_and_uri_credentials() -> None:
    assert "hunter2" not in redact_secrets("password=hunter2")
    assert redact_secrets("password=hunter2") == "password=***"
    assert redact_secrets("token: abc123") == "token=***"
    assert "pw" not in redact_secrets("sqlite://user:pw@host/db")
    assert redact_secrets("sqlite://user:pw@host/db") == "sqlite://***@host/db"
    # A harmless message is unchanged.
    assert redact_secrets("Read-only connection OK") == "Read-only connection OK"


def test_error_messages_never_leak_a_credential(tmp_path: Path) -> None:
    service = _service(tmp_path)
    check = service.test_connection("sqlite://user:supersecret@host/db")
    assert "supersecret" not in check.message


# --- window (only when a display is available) ----------------------------


def _display_available() -> bool:
    try:
        import tkinter as tk

        root = tk.Tk()
    except Exception:
        return False
    root.destroy()
    return True


class _FakeService:
    """The minimum a dialog needs: a ``sql_import`` service and a format list."""

    def __init__(self, sql: SqlImportService) -> None:
        self.sql_import = sql


class _Theme:
    base = "#101010"
    surface = "#161616"
    surface_raised = "#1e1e1e"
    surface_deep = "#0b0b0b"
    contrast = "#ffffff"
    foreground = "#e0e0e0"
    text_secondary = "#a0a0a0"
    text_dim = "#707070"
    danger = "#d05050"
    interactive_active = "#2a2a2a"
    border_subtle = "#242424"


@pytest.mark.skipif(not _display_available(), reason="no display available")
def test_sql_import_dialog_shows_schema_and_reports_a_bad_file(tmp_path: Path) -> None:
    import tkinter as tk

    from gui.sql_import_dialog import SqlImportDialog

    db = _make_db(tmp_path / "wells.db", rows=6)
    service = _FakeService(SqlImportService(store_dir=tmp_path / "kb"))
    root = tk.Tk()
    try:
        dialog = SqlImportDialog(root, service, theme=_Theme(), font="TkDefaultFont", mono="TkFixedFont")
        # A bad path is reported, and the schema stays empty.
        dialog._path_var.set(str(tmp_path / "missing.db"))
        dialog._on_test()
        assert dialog._tables == {}
        # A good path loads the schema into the tree.
        dialog._path_var.set(str(db))
        dialog._on_test()
        assert set(dialog._tables) == {"wells"}
        assert "wells" in dialog._table_tree.get_children()
        # Selecting the table renders its columns (with the PK marked).
        dialog._table_tree.selection_set("wells")
        dialog._on_table_select()
        labels = [
            child.cget("text")
            for child in dialog._columns_frame.winfo_children()
            if child.winfo_class() == "Checkbutton"
        ]
        assert any("id" in label and "PK" in label for label in labels)
        dialog._close()
    finally:
        root.destroy()


@pytest.mark.skipif(not _display_available(), reason="no display available")
def test_import_center_opens_the_sql_screen(tmp_path: Path) -> None:
    import tkinter as tk

    from gui.import_dialog import ImportCenterDialog
    from gui.sql_import_dialog import SqlImportDialog

    class _Service:
        sql_import = SqlImportService(store_dir=tmp_path / "kb")

        def import_jobs(self):  # noqa: ANN201 - the dialog only needs an iterable
            return ()

    root = tk.Tk()
    try:
        center = ImportCenterDialog(
            root, _Service(), theme=_Theme(), font="TkDefaultFont", mono="TkFixedFont"
        )
        center._on_open_sql_import()
        assert isinstance(center._sql_dialog, SqlImportDialog)
        center._sql_dialog._close()
        center._close()
    finally:
        root.destroy()
