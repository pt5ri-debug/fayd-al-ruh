"""The SQL Database Import screen: the window's SQL source page.

Reached from the ＋ menu, this page adds a SQL database as a knowledge source:

* **1. Database file** -- choose a SQLite file, test the read-only connection,
  and see the tables it holds;
* **2. Tables and columns** -- pick which tables to import and uncheck any column
  that must never be read (a secret column), with a limited data preview;
* **3. Start import** -- review the batch size and start the import explicitly;
* **4. Import jobs** -- see durable jobs and Resume / Retry / Forget one.

It is a thin view over :class:`gui.sql_import.SqlImportService`, which drives the
same ingestion service the CLI's ``ingest sql-import`` command uses. The window
only chooses a file, checks boxes, and starts the work on a worker thread so Tk
never blocks; the readable progress, the durable checkpoint, cancellation, resume,
and the exclusion policy are the service's and the connector's, not a copy.

The class is Tk-only and imported lazily with the rest of the window, so the CLI
never touches it.
"""

from __future__ import annotations

import threading
from collections.abc import Callable
from functools import partial
from pathlib import Path
from typing import Any, Literal

import tkinter as tk
from tkinter import filedialog, ttk

from gui.sql_import import (
    DATABASE_TYPES,
    DEFAULT_BATCH_SIZE,
    MAX_BATCH_SIZE,
    SCHEMA_FULL_REIMPORT,
    SqlImportReport,
    TableInfo,
    picker_filetypes,
    redact_secrets,
)
from gui.typography import resolve
from gui.window_controls import install_window_controls

#: The page title and one-line description.
TITLE = "SQL Database Import"
SUBTITLE = (
    "Add a SQLite database to the knowledge base. The database is opened read-only "
    "and is never written to; each row is stored as a citable chunk. SQLite is the "
    "only database type enabled at this stage."
)
FORMATS_LINE = "Enabled database type: " + ", ".join(DATABASE_TYPES) + "."

#: Excluding a column is by name across the whole database (the connector's rule),
#: stated here so the UI is honest about what unchecking a box means.
EXCLUSION_NOTE = (
    "Unchecking a column excludes it by name everywhere in this database, before "
    "any value is read — so its data never reaches a record, a preview, or the store."
)


class SqlImportDialog:
    """A modal-ish window that imports a SQLite database into the knowledge base."""

    def __init__(
        self,
        parent: tk.Tk | tk.Toplevel,
        service: Any,
        *,
        theme: Any,
        font: str,
        mono: str,
        on_close: Callable[[], None] | None = None,
    ) -> None:
        self.service = service
        self.sql = service.sql_import
        self.theme = theme
        self.font = font
        self.mono = mono
        self._type = resolve(parent, ui_family=font, mono_family=mono)
        self._on_close = on_close
        self._busy = False
        self._cancel = threading.Event()
        self._tables: dict[str, TableInfo] = {}
        self._exclusions: dict[str, set[str]] = {}
        self._exclude_columns: set[str] = set()
        self._last_report: SqlImportReport | None = None

        self._path_var = tk.StringVar()
        self._batch_var = tk.StringVar(value=str(DEFAULT_BATCH_SIZE))
        self._only_selected_var = tk.BooleanVar(value=False)

        self.window = tk.Toplevel(parent)
        self.window.title(TITLE)
        self.window.configure(bg=theme.base)
        self.window.geometry("820x760")
        self.window.minsize(640, 540)
        self.window.transient(parent)
        self.window.protocol("WM_DELETE_WINDOW", self._close)
        self._build()
        self.refresh_jobs()

    # -- layout -----------------------------------------------------------

    def _build(self) -> None:
        theme = self.theme
        bar = tk.Frame(self.window, bg=theme.titlebar)
        bar.pack(fill="x")
        header = tk.Frame(bar, bg=theme.titlebar)
        header.pack(fill="x", padx=20, pady=(18, 8))
        title_row = tk.Frame(header, bg=theme.titlebar)
        title_row.pack(fill="x")
        tk.Label(
            title_row, text=TITLE, bg=theme.titlebar, fg=theme.contrast,
            font=self._type.font("heading1", weight="bold"), anchor="w",
        ).pack(side="left")
        self._window_controls = install_window_controls(
            self.window, title_row, theme=theme, typography=self._type, on_close=self._close
        )
        tk.Label(
            header, text=SUBTITLE, bg=theme.titlebar, fg=theme.text_secondary,
            font=self._type.font("xs"), anchor="w", justify="left", wraplength=760,
        ).pack(fill="x", pady=(4, 0))
        tk.Label(
            header, text=FORMATS_LINE, bg=theme.titlebar, fg=theme.text_dim,
            font=self._type.font("2xs"), anchor="w",
        ).pack(fill="x", pady=(6, 0))
        tk.Frame(self.window, bg=theme.titlebar_edge, height=1).pack(fill="x")

        self._build_file_section()
        self._build_schema_section()
        self._build_start_section()
        self._build_jobs_section()
        self._build_results_section()

    def _section(self, title: str) -> tk.Frame:
        theme = self.theme
        frame = tk.Frame(self.window, bg=theme.surface, highlightthickness=1,
                         highlightbackground=theme.border_subtle)
        frame.pack(fill="x", padx=20, pady=(10, 0))
        tk.Label(
            frame, text=title, bg=theme.surface, fg=theme.contrast,
            font=self._type.font("sm", weight="bold"), anchor="w",
        ).pack(fill="x", padx=12, pady=(10, 2))
        return frame

    def _button(self, parent: tk.Misc, text: str, command: Any) -> tk.Button:
        theme = self.theme
        return tk.Button(
            parent, text=text, command=command, bg=theme.surface_raised,
            fg=theme.foreground, activebackground=theme.interactive_active,
            activeforeground=theme.contrast, relief="flat", borderwidth=0,
            font=self._type.font("xs", weight="bold"), padx=12, pady=4, cursor="hand2",
        )

    def _build_file_section(self) -> None:
        theme = self.theme
        frame = self._section("1.  Database file")
        row = tk.Frame(frame, bg=theme.surface)
        row.pack(fill="x", padx=12, pady=(2, 6))
        self._choose_button = self._button(row, "Choose database…", self._on_choose_file)
        self._choose_button.pack(side="left")
        self._path_label = tk.Label(
            row, text="No database chosen", bg=theme.surface, fg=theme.text_dim,
            font=self._type.font("xs"), anchor="w",
        )
        self._path_label.pack(side="left", padx=10, fill="x", expand=True)
        row2 = tk.Frame(frame, bg=theme.surface)
        row2.pack(fill="x", padx=12, pady=(0, 12))
        self._test_button = self._button(row2, "Test connection (read-only)", self._on_test)
        self._test_button.pack(side="left")
        self._connection_label = tk.Label(
            row2, text="", bg=theme.surface, fg=theme.text_secondary,
            font=self._type.font("xs"), anchor="w",
        )
        self._connection_label.pack(side="left", padx=10, fill="x", expand=True)

    def _build_schema_section(self) -> None:
        theme = self.theme
        frame = self._section("2.  Tables and columns")
        body = tk.Frame(frame, bg=theme.surface)
        body.pack(fill="both", expand=True, padx=12, pady=(2, 12))
        left = tk.Frame(body, bg=theme.surface)
        left.pack(side="left", fill="both", expand=True)
        right = tk.Frame(body, bg=theme.surface)
        right.pack(side="left", fill="both", expand=True, padx=(10, 0))

        tk.Label(
            left, text="Tables", bg=theme.surface, fg=theme.text_secondary,
            font=self._type.font("xs", weight="bold"), anchor="w",
        ).pack(fill="x")
        self._table_tree = ttk.Treeview(
            left, columns=("rows", "pk"), show="tree headings", height=6,
            selectmode="extended",
        )
        self._table_tree.heading("#0", text="Table")
        self._table_tree.heading("rows", text="Rows")
        self._table_tree.heading("pk", text="Key")
        self._table_tree.column("#0", width=150, anchor="w")
        self._table_tree.column("rows", width=60, anchor="e")
        self._table_tree.column("pk", width=60, anchor="center")
        self._table_tree.pack(fill="both", expand=True, pady=(4, 4))
        self._table_tree.bind("<<TreeviewSelect>>", self._on_table_select)
        self._only_selected_check = tk.Checkbutton(
            left, text="Import only the selected tables", variable=self._only_selected_var,
            bg=theme.surface, fg=theme.foreground, selectcolor=theme.surface_raised,
            activebackground=theme.surface, activeforeground=theme.contrast,
            font=self._type.font("xs"), highlightthickness=0, borderwidth=0, anchor="w",
        )
        self._only_selected_check.pack(fill="x")

        tk.Label(
            right, text="Columns (uncheck to exclude)", bg=theme.surface,
            fg=theme.text_secondary, font=self._type.font("xs", weight="bold"), anchor="w",
        ).pack(fill="x")
        self._columns_frame = tk.Frame(right, bg=theme.surface_deep, height=120)
        self._columns_frame.pack(fill="both", expand=True, pady=(4, 4))
        tk.Label(
            right, text=EXCLUSION_NOTE, bg=theme.surface, fg=theme.text_dim,
            font=self._type.font("2xs"), anchor="w", justify="left", wraplength=360,
        ).pack(fill="x")
        row = tk.Frame(right, bg=theme.surface)
        row.pack(fill="x", pady=(6, 0))
        self._preview_button = self._button(row, "Preview rows", self._on_preview)
        self._preview_button.pack(side="left")

    def _build_start_section(self) -> None:
        theme = self.theme
        frame = self._section("3.  Start import")
        row = tk.Frame(frame, bg=theme.surface)
        row.pack(fill="x", padx=12, pady=(2, 12))
        tk.Label(
            row, text="Batch size (rows per committed batch)", bg=theme.surface,
            fg=theme.text_secondary, font=self._type.font("xs"),
        ).pack(side="left")
        self._batch_entry = tk.Entry(
            row, textvariable=self._batch_var, width=10, bg=theme.surface_deep,
            fg=theme.foreground, insertbackground=theme.foreground, relief="flat",
            font=self._type.mono("code"),
        )
        self._batch_entry.pack(side="left", padx=8)
        self._start_button = self._button(row, "Start import", self._on_start)
        self._start_button.pack(side="right")
        self._cancel_button = self._button(row, "Cancel", self._on_cancel)

    def _build_jobs_section(self) -> None:
        theme = self.theme
        frame = self._section("4.  Import jobs")
        self._jobs_frame = tk.Frame(frame, bg=theme.surface)
        self._jobs_frame.pack(fill="x", padx=12, pady=(2, 6))
        row = tk.Frame(frame, bg=theme.surface)
        row.pack(fill="x", padx=12, pady=(0, 12))
        self._button(row, "Refresh", self.refresh_jobs).pack(side="left")

    def _build_results_section(self) -> None:
        theme = self.theme
        frame = tk.Frame(self.window, bg=theme.base)
        frame.pack(fill="both", expand=True, padx=20, pady=(12, 4))
        top = tk.Frame(frame, bg=theme.base)
        top.pack(fill="x")
        self._progress = ttk.Progressbar(top, mode="indeterminate", length=180)
        self._progress_determinate = ttk.Progressbar(top, mode="determinate", length=180, maximum=100)
        self._status = tk.Label(
            top, text="Ready.", bg=theme.base, fg=theme.text_secondary,
            font=self._type.font("xs"), anchor="w",
        )
        self._status.pack(side="left")
        self._results = tk.Text(
            frame, height=8, bg=theme.surface_deep, fg=theme.foreground,
            font=self._type.mono("code"), relief="flat", wrap="none", padx=10, pady=8,
            highlightthickness=0, borderwidth=0, state="disabled",
        )
        self._results.pack(fill="both", expand=True, pady=(6, 12))

    # -- file and connection ---------------------------------------------

    def _on_choose_file(self) -> None:
        path = filedialog.askopenfilename(
            parent=self.window, title="Choose a SQLite database",
            filetypes=list(picker_filetypes()),
        )
        if not path:
            return
        self._path_var.set(path)
        self._path_label.configure(text=redact_secrets(path), fg=self.theme.foreground)
        self._on_test()

    def _on_test(self) -> None:
        path = self._path_var.get()
        check = self.sql.test_connection(path)
        self._connection_label.configure(
            text=redact_secrets(check.message),
            fg=self.theme.text_secondary if check.ok else self.theme.danger,
        )
        if check.ok:
            self._load_schema(path)
        else:
            self._clear_schema()

    def _clear_schema(self) -> None:
        self._tables = {}
        for item in self._table_tree.get_children():
            self._table_tree.delete(item)
        self._render_columns(None)

    def _load_schema(self, path: str) -> None:
        try:
            infos = self.sql.inspect(path)
        except Exception as error:  # noqa: BLE001 - shown, never crashes Tk
            self._set_status(redact_secrets(f"Could not read the schema: {error}"), error=True)
            self._clear_schema()
            return
        self._tables = {info.name: info for info in infos}
        self._exclusions = {name: set() for name in self._tables}
        for item in self._table_tree.get_children():
            self._table_tree.delete(item)
        for info in infos:
            key = "PK " + ", ".join(info.primary_key) if info.primary_key else "no PK"
            self._table_tree.insert(
                "", "end", iid=info.name, text=info.name,
                values=(info.row_count, key),
            )

    # -- columns and preview ---------------------------------------------

    def _on_table_select(self, _event: Any = None) -> None:
        selected = self._table_tree.selection()
        table = selected[0] if selected else None
        self._render_columns(table)

    def _render_columns(self, table: str | None) -> None:
        theme = self.theme
        for child in self._columns_frame.winfo_children():
            child.destroy()
        if table is None or table not in self._tables:
            tk.Label(
                self._columns_frame, text="Select a table to see its columns.",
                bg=theme.surface_deep, fg=theme.text_dim, font=self._type.font("xs"), anchor="w",
            ).pack(fill="x", padx=8, pady=8)
            return
        info = self._tables[table]
        excluded = self._exclusions.setdefault(table, set())
        for column in info.columns:
            label = column.name + (f"  [{column.declared_type}]" if column.declared_type else "")
            if column.primary_key:
                label += "  (PK)"
            var = tk.BooleanVar(value=column.name not in excluded)
            check = tk.Checkbutton(
                self._columns_frame, text=label, variable=var,
                command=partial(self._toggle_column, table, column.name, var),
                bg=theme.surface_deep, fg=theme.foreground, selectcolor=theme.surface_raised,
                activebackground=theme.surface_deep, activeforeground=theme.contrast,
                font=self._type.font("xs"), highlightthickness=0, borderwidth=0, anchor="w",
            )
            check.pack(fill="x", padx=8, pady=1)

    def _toggle_column(self, table: str, column: str, var: tk.BooleanVar) -> None:
        excluded = self._exclusions.setdefault(table, set())
        if var.get():
            excluded.discard(column)
        else:
            excluded.add(column)
        self._exclude_columns = set().union(*self._exclusions.values()) if self._exclusions else set()

    def _on_preview(self) -> None:
        path = self._path_var.get()
        selection = self._table_tree.selection()
        if not path or not selection:
            self._set_status("Choose a database and a table first.", error=True)
            return
        table = selection[0]
        info = self._tables.get(table)
        columns = tuple(
            column.name for column in (info.columns if info else ())
            if column.name not in self._exclude_columns
        )
        preview = self.sql.preview(path, table=table, columns=columns, limit=5)
        if preview.message and not preview.rows:
            self._set_results(redact_secrets(preview.message))
            return
        lines = [f"{preview.table}  ({', '.join(preview.columns)})", ""]
        lines.extend(preview.rows)
        if preview.truncated:
            lines.append("… (preview limited)")
        self._set_results("\n".join(lines))

    # -- import -----------------------------------------------------------

    def _selected_tables(self) -> tuple[str, ...]:
        if not self._only_selected_var.get():
            return ()  # empty include list = every table
        return tuple(self._table_tree.selection())

    def _batch_size(self) -> int:
        try:
            value = int(self._batch_var.get())
        except ValueError:
            return DEFAULT_BATCH_SIZE
        return max(1, min(value, MAX_BATCH_SIZE))

    def _on_start(self) -> None:
        if self._busy:
            return
        path = self._path_var.get()
        if not path:
            self._set_status("Choose a database first.", error=True)
            return
        if self._only_selected_var.get() and not self._table_tree.selection():
            self._set_status("Select at least one table, or import all tables.", error=True)
            return
        job_id = self.sql.default_job_id(path)
        tables = self._selected_tables()
        exclude_columns = tuple(sorted(self._exclude_columns))
        batch = self._batch_size()
        self._cancel = threading.Event()
        self._start(
            f"Importing {Path(path).name}…",
            lambda progress: self.sql.run_import(
                path, job_id=job_id, tables=tables,
                exclude_columns=exclude_columns, batch_size=batch, progress=progress,
                should_cancel=self._cancel.is_set,
            ),
        )

    def _on_cancel(self) -> None:
        if self._busy:
            self._cancel.set()
            self._set_status("Cancelling at the next safe point…")

    def _on_resume_job(self, job_id: str) -> None:
        if self._busy:
            return
        self._cancel = threading.Event()
        self._start(
            f"Resuming job {job_id}…",
            lambda progress: self.sql.resume_job(
                job_id, progress=progress, should_cancel=self._cancel.is_set
            ),
        )

    def _on_retry_job(self, job_id: str, *, full_reimport: bool = False) -> None:
        if self._busy:
            return
        self._cancel = threading.Event()
        schema_change = SCHEMA_FULL_REIMPORT if full_reimport else None
        self._start(
            f"Retrying job {job_id}…",
            lambda progress: self.sql.resume_job(
                job_id, progress=progress, should_cancel=self._cancel.is_set,
                schema_change=schema_change,
            ),
        )

    def _on_forget_job(self, job_id: str) -> None:
        self.sql.forget_job(job_id)
        self.refresh_jobs()

    # -- worker -----------------------------------------------------------

    def _start(self, message: str, job: Any) -> None:
        self._busy = True
        self._set_buttons_enabled(False)
        self._set_status(message)
        self._cancel_button.pack(side="right", padx=8)
        self._progress_determinate.pack_forget()
        self._progress.pack(side="right", padx=8)
        self._progress.start(12)

        def run() -> None:
            try:
                report = job(self._on_progress)
                self.window.after(0, partial(self._finish, report, None))
            except Exception as error:  # noqa: BLE001 - shown, never crashes Tk
                self.window.after(0, partial(self._finish, None, error))

        threading.Thread(target=run, daemon=True).start()

    def _on_progress(self, snapshot: Any) -> None:
        self.window.after(0, partial(self._render_progress, snapshot))

    def _render_progress(self, snapshot: Any) -> None:
        percent = snapshot.percent
        if percent is None:
            self._set_status(f"Importing… {snapshot.message}")
        else:
            if not self._progress_determinate.winfo_ismapped():
                self._progress.stop()
                self._progress.pack_forget()
                self._progress_determinate.pack(side="right", padx=8)
            self._progress_determinate.configure(value=percent)
            where = f" — {snapshot.table}" if snapshot.table else ""
            self._set_status(f"Importing{where}… {snapshot.committed}/{snapshot.total} rows ({percent}%)")

    def _finish(self, report: Any, error: Exception | None) -> None:
        self._busy = False
        self._progress.stop()
        self._progress.pack_forget()
        self._progress_determinate.pack_forget()
        self._cancel_button.pack_forget()
        self._set_buttons_enabled(True)
        if error is not None:
            self._set_status(redact_secrets(f"Import failed: {error}"), error=True)
            return
        assert report is not None
        self._last_report = report
        self._render_report(report)
        self.refresh_jobs()

    # -- rendering --------------------------------------------------------

    def _render_report(self, report: SqlImportReport) -> None:
        lines = [report.summary(), ""]
        for table in report.tables:
            marker = {"done": "✓", "failed": "✗"}.get(table.status, "…")
            note = "" if table.stable_identity else "  (no stable row identity)"
            lines.append(f"{marker}  {table.label}{note}")
        for failure in report.failures:
            lines.append(f"    {failure}")
        self._set_results("\n".join(lines))
        self._set_status(report.summary(), error=report.failed)

    def refresh_jobs(self) -> None:
        theme = self.theme
        for child in self._jobs_frame.winfo_children():
            child.destroy()
        try:
            jobs = self.sql.list_jobs()
        except Exception:  # noqa: BLE001 - a jobs read must not break the page
            jobs = ()
        if not jobs:
            tk.Label(
                self._jobs_frame, text="No SQL import jobs recorded yet.",
                bg=theme.surface, fg=theme.text_dim, font=self._type.font("xs"), anchor="w",
            ).pack(fill="x")
            return
        for job in jobs:
            row = tk.Frame(self._jobs_frame, bg=theme.surface)
            row.pack(fill="x", pady=1)
            tk.Label(
                row, text=job.label(), bg=theme.surface, fg=theme.text_secondary,
                font=self._type.font("2xs"), anchor="w",
            ).pack(side="left", fill="x", expand=True)
            if job.can_resume:
                self._button(row, "Resume", lambda jid=job.job_id: self._on_resume_job(jid)).pack(
                    side="right", padx=(6, 0)
                )
            if job.status == "failed":
                self._button(
                    row, "Retry", lambda jid=job.job_id: self._on_retry_job(jid)
                ).pack(side="right")
            self._button(row, "Forget", lambda jid=job.job_id: self._on_forget_job(jid)).pack(
                side="right"
            )

    def _set_results(self, text: str) -> None:
        self._results.configure(state="normal")
        self._results.delete("1.0", "end")
        self._results.insert("1.0", redact_secrets(text))
        self._results.configure(state="disabled")

    def _set_status(self, text: str, *, error: bool = False) -> None:
        colour = self.theme.danger if error else self.theme.text_secondary
        self._status.configure(text=redact_secrets(text), fg=colour)

    def _set_buttons_enabled(self, enabled: bool) -> None:
        state: Literal["normal", "disabled"] = "normal" if enabled else "disabled"
        for button in (
            self._choose_button,
            self._test_button,
            self._preview_button,
            self._start_button,
        ):
            button.configure(state=state)

    def _close(self) -> None:
        if self._on_close is not None:
            self._on_close()
        self.window.destroy()


__all__ = ["EXCLUSION_NOTE", "FORMATS_LINE", "SUBTITLE", "TITLE", "SqlImportDialog"]
