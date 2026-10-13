"""The Knowledge Import Center: the window's file/folder import page.

Reached from the composer's **＋** control, this page offers the two import paths
the user asked for and a jobs list:

* **Upload files** -- choose one or many supported files; each is imported and
  its result shown.
* **Import from a local folder** -- choose a folder, optionally scan its
  subfolders, and import every supported file in bounded batches.
* **Import jobs** -- see recorded jobs and resume an interrupted one.

It is a thin view over :class:`gui.import_center.KnowledgeImporter` (which drives
the existing :mod:`ingest` pipeline): the window only chooses paths, starts the
work on a worker thread so Tk never blocks, and renders the per-file results.
Because the app runs on the user's own machine, a folder the picker returns is a
real local path the process can read -- there is no server to upload to, and no
arbitrary path is ever constructed from text.

The class is Tk-only and imported lazily with the rest of the window, so the
CLI never touches it.
"""

from __future__ import annotations

import threading
from collections.abc import Callable
from functools import partial
from pathlib import Path
from typing import Any, Literal

import tkinter as tk
from tkinter import filedialog, ttk

from core import optional_deps
from gui.import_center import (
    SUPPORTED_SUFFIXES,
    UNSUPPORTED_SUFFIXES,
    ImportReport,
    picker_filetypes,
)

#: The header line, kept as the page title.
TITLE = "Knowledge Import Center"

#: One-line description of what the page does, shown under the title.
SUBTITLE = (
    "Import documents into the knowledge base. Files are read locally in bounded "
    "batches; nothing is uploaded and no macro or script is ever run."
)

#: A short, honest list of the formats the file connector accepts.
FORMATS_LINE = "Supported: " + ", ".join(SUPPORTED_SUFFIXES) + "."
#: The formats that are recognised but refused, with the reason.
UNSUPPORTED_LINE = (
    "Not supported (binary/undocumented): " + ", ".join(UNSUPPORTED_SUFFIXES)
    + " — save as .docx instead."
)


class ImportCenterDialog:
    """A modal-ish window that imports files and folders into the knowledge base."""

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
        self.theme = theme
        self.font = font
        self.mono = mono
        self._on_close = on_close
        self._busy = False
        self._last_report: ImportReport | None = None
        self._folder_var = tk.StringVar()
        self._recursive_var = tk.BooleanVar(value=True)
        self._ocr_var = tk.BooleanVar(value=True)
        self._ocr_force_var = tk.BooleanVar(value=False)
        self._ocr_languages_var = tk.StringVar(value="eng")

        self.window = tk.Toplevel(parent)
        self.window.title(TITLE)
        self.window.configure(bg=theme.base)
        self.window.geometry("720x640")
        self.window.minsize(560, 480)
        self.window.transient(parent)
        self.window.protocol("WM_DELETE_WINDOW", self._close)
        self._build()
        self.refresh_jobs()

    # -- layout -----------------------------------------------------------

    def _build(self) -> None:
        theme = self.theme
        header = tk.Frame(self.window, bg=theme.base)
        header.pack(fill="x", padx=20, pady=(18, 8))
        tk.Label(
            header, text=TITLE, bg=theme.base, fg=theme.contrast,
            font=(self.font, 15, "bold"), anchor="w",
        ).pack(fill="x")
        tk.Label(
            header, text=SUBTITLE, bg=theme.base, fg=theme.text_secondary,
            font=(self.font, 9), anchor="w", justify="left", wraplength=660,
        ).pack(fill="x", pady=(4, 0))
        tk.Label(
            header, text=FORMATS_LINE, bg=theme.base, fg=theme.text_dim,
            font=(self.font, 8), anchor="w", justify="left", wraplength=660,
        ).pack(fill="x", pady=(6, 0))
        tk.Label(
            header, text=UNSUPPORTED_LINE, bg=theme.base, fg=theme.text_dim,
            font=(self.font, 8), anchor="w", justify="left", wraplength=660,
        ).pack(fill="x")

        self._build_upload_section()
        self._build_folder_section()
        self._build_sql_section()
        self._build_jobs_section()
        self._build_results_section()

    def _section(self, title: str) -> tk.Frame:
        theme = self.theme
        frame = tk.Frame(self.window, bg=theme.surface, highlightthickness=1,
                         highlightbackground=theme.border_subtle)
        frame.pack(fill="x", padx=20, pady=(10, 0))
        tk.Label(
            frame, text=title, bg=theme.surface, fg=theme.contrast,
            font=(self.font, 10, "bold"), anchor="w",
        ).pack(fill="x", padx=12, pady=(10, 2))
        return frame

    def _button(self, parent: tk.Misc, text: str, command: Any) -> tk.Button:
        theme = self.theme
        button = tk.Button(
            parent, text=text, command=command, bg=theme.surface_raised,
            fg=theme.foreground, activebackground=theme.interactive_active,
            activeforeground=theme.contrast, relief="flat", borderwidth=0,
            font=(self.font, 9, "bold"), padx=12, pady=4, cursor="hand2",
        )
        return button

    def _build_upload_section(self) -> None:
        theme = self.theme
        frame = self._section("1.  Upload files")
        row = tk.Frame(frame, bg=theme.surface)
        row.pack(fill="x", padx=12, pady=(2, 12))
        self._upload_button = self._button(row, "Choose files…", self._on_upload)
        self._upload_button.pack(side="left")
        tk.Label(
            row, text="Select one or many files. Each is imported on its own.",
            bg=theme.surface, fg=theme.text_secondary, font=(self.font, 9),
        ).pack(side="left", padx=10)

    def _build_folder_section(self) -> None:
        theme = self.theme
        frame = self._section("2.  Import from a local folder")
        row = tk.Frame(frame, bg=theme.surface)
        row.pack(fill="x", padx=12, pady=(2, 6))
        self._folder_button = self._button(row, "Choose folder…", self._on_choose_folder)
        self._folder_button.pack(side="left")
        self._folder_label = tk.Label(
            row, text="No folder chosen", bg=theme.surface, fg=theme.text_dim,
            font=(self.font, 9), anchor="w",
        )
        self._folder_label.pack(side="left", padx=10, fill="x", expand=True)
        row2 = tk.Frame(frame, bg=theme.surface)
        row2.pack(fill="x", padx=12, pady=(0, 4))
        self._recursive_check = tk.Checkbutton(
            row2, text="Include subfolders", variable=self._recursive_var,
            bg=theme.surface, fg=theme.foreground, selectcolor=theme.surface_raised,
            activebackground=theme.surface, activeforeground=theme.contrast,
            font=(self.font, 9), highlightthickness=0, borderwidth=0, anchor="w",
        )
        self._recursive_check.pack(side="left")
        self._folder_import_button = self._button(row2, "Import folder", self._on_import_folder)
        self._folder_import_button.pack(side="right")
        row3 = tk.Frame(frame, bg=theme.surface)
        row3.pack(fill="x", padx=12, pady=(0, 12))
        self._ocr_check = tk.Checkbutton(
            row3, text="Read scanned PDFs (OCR)", variable=self._ocr_var,
            bg=theme.surface, fg=theme.foreground, selectcolor=theme.surface_raised,
            activebackground=theme.surface, activeforeground=theme.contrast,
            font=(self.font, 9), highlightthickness=0, borderwidth=0, anchor="w",
        )
        self._ocr_check.pack(side="left")
        self._ocr_force_check = tk.Checkbutton(
            row3, text="force low-text pages", variable=self._ocr_force_var,
            bg=theme.surface, fg=theme.text_secondary, selectcolor=theme.surface_raised,
            activebackground=theme.surface, activeforeground=theme.contrast,
            font=(self.font, 8), highlightthickness=0, borderwidth=0, anchor="w",
        )
        self._ocr_force_check.pack(side="left", padx=(8, 0))
        tk.Label(
            row3, text="languages", bg=theme.surface, fg=theme.text_secondary,
            font=(self.font, 9),
        ).pack(side="left", padx=(12, 4))
        self._ocr_languages_entry = tk.Entry(
            row3, textvariable=self._ocr_languages_var, width=10,
            bg=theme.surface_deep, fg=theme.foreground, insertbackground=theme.foreground,
            relief="flat", font=(self.mono, 9),
        )
        self._ocr_languages_entry.pack(side="left")
        tk.Label(
            row3, text="(e.g. ara+eng; needs the [ocr] extra + Tesseract)",
            bg=theme.surface, fg=theme.text_dim, font=(self.font, 8),
        ).pack(side="left", padx=6)
        # Offer the install right where OCR is chosen, but only when it is missing.
        self._ocr_install_button = self._button(
            row3, "Install OCR support…", self._on_install_ocr
        )
        if optional_deps.is_available(optional_deps.FEATURE_OCR):
            self._ocr_install_button.pack_forget()
        else:
            self._ocr_install_button.pack(side="right")

    def _build_sql_section(self) -> None:
        theme = self.theme
        frame = self._section("3.  Import a SQL database")
        row = tk.Frame(frame, bg=theme.surface)
        row.pack(fill="x", padx=12, pady=(2, 12))
        self._sql_button = self._button(row, "Open SQL Database Import…", self._on_open_sql_import)
        self._sql_button.pack(side="left")
        tk.Label(
            row,
            text="Add a SQLite database: read-only connection test, schema and column "
                 "selection, a limited preview, and a resumable import.",
            bg=theme.surface, fg=theme.text_secondary, font=(self.font, 9),
            justify="left", wraplength=520,
        ).pack(side="left", padx=10)

    def _build_jobs_section(self) -> None:
        theme = self.theme
        frame = self._section("4.  Import jobs")
        self._jobs_frame = tk.Frame(frame, bg=theme.surface)
        self._jobs_frame.pack(fill="x", padx=12, pady=(2, 8))
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
        self._status = tk.Label(
            top, text="Ready.", bg=theme.base, fg=theme.text_secondary,
            font=(self.font, 9), anchor="w",
        )
        self._status.pack(side="left")
        self._retry_button = self._button(top, "Retry failed", self._on_retry)
        self._retry_ocr_button = self._button(top, "Retry with OCR", self._on_retry_ocr)
        # The retry button appears only when there is something to retry.
        self._results = tk.Text(
            frame, height=10, bg=theme.surface_deep, fg=theme.foreground,
            font=(self.mono, 9), relief="flat", wrap="none", padx=10, pady=8,
            highlightthickness=0, borderwidth=0, state="disabled",
        )
        self._results.pack(fill="both", expand=True, pady=(6, 12))

    # -- actions ----------------------------------------------------------

    def _on_upload(self) -> None:
        if self._busy:
            return
        paths = filedialog.askopenfilenames(
            parent=self.window, title="Upload files to the knowledge base",
            filetypes=list(picker_filetypes()),
        )
        if not paths:
            return
        self._start("Importing %d file(s)…" % len(paths),
                    lambda progress: self.service.import_files(
                        paths, ocr=self._ocr_choice(), progress=progress))

    def _on_choose_folder(self) -> None:
        folder = filedialog.askdirectory(
            parent=self.window, title="Choose a folder to import", mustexist=True,
        )
        if not folder:
            return
        self._folder_var.set(folder)
        self._folder_label.configure(text=folder, fg=self.theme.foreground)

    def _on_import_folder(self) -> None:
        if self._busy:
            return
        folder = self._folder_var.get()
        if not folder:
            self._set_status("Choose a folder first.", error=True)
            return
        recursive = bool(self._recursive_var.get())
        job_id = "folder-" + Path(folder).name
        self._start(
            f"Importing {folder}…",
            lambda progress: self.service.import_folder(
                folder, recursive=recursive, ocr=self._ocr_choice(),
                ocr_languages=self._ocr_languages(), job_id=job_id, progress=progress
            ),
        )

    def _ocr_choice(self) -> bool | None:
        """The tri-state OCR choice from the two checkboxes.

        ``Read scanned PDFs (OCR)`` unchecked means never OCR; checked means
        automatic (only pages with no text layer); with ``force low-text pages``
        it means force OCR on every low-text page.
        """
        if not self._ocr_var.get():
            return False
        return True if self._ocr_force_var.get() else None

    def _ocr_languages(self) -> str:
        """The language codes typed in the OCR row, or the engine default."""
        return self._ocr_languages_var.get().strip() or "eng"

    def _on_retry(self, *, ocr: bool | None = None) -> None:
        if self._busy or self._last_report is None:
            return
        failures = self._last_report.failures
        if not failures:
            return
        use_ocr = self._ocr_choice() if ocr is None else ocr
        self._start(
            f"Retrying {len(failures)} file(s)…",
            lambda progress: self.service.retry_failed(
                failures, ocr=use_ocr, ocr_languages=self._ocr_languages(), progress=progress
            ),
        )

    def _on_retry_ocr(self) -> None:
        """Retry the scanned-PDF failures with OCR turned on for this run."""
        self._ocr_var.set(True)
        self._on_retry(ocr=True)

    def _on_install_ocr(self) -> None:
        """Install the optional OCR packages from within the Import Center.

        A scanned PDF cannot be imported without them, and the project is not on
        PyPI, so the old "pip install fayd-al-ruh[ocr]" hint could not work here.
        This runs the bundled requirements file instead and reports the outcome.
        """
        if self._busy:
            return
        if optional_deps.is_available(optional_deps.FEATURE_OCR):
            self._set_status("OCR support is already installed.")
            self._ocr_install_button.pack_forget()
            return
        if optional_deps.install_command(optional_deps.FEATURE_OCR) is None:
            self._set_status(optional_deps.install_hint(optional_deps.FEATURE_OCR), error=True)
            return
        self._start("Installing OCR support…", self._install_ocr_job)

    def _install_ocr_job(self, progress: Any) -> "optional_deps.InstallResult":
        result = optional_deps.install(optional_deps.FEATURE_OCR, progress=progress)
        return result

    def _finish_install(self, result: "optional_deps.InstallResult") -> None:
        self._set_status(result.message, error=not result.ok)
        if result.ok and optional_deps.is_available(optional_deps.FEATURE_OCR):
            self._ocr_var.set(True)
            self._ocr_install_button.pack_forget()

    def _on_resume_job(self, job_id: str) -> None:
        if self._busy:
            return
        self._start(
            f"Resuming job {job_id}…",
            lambda progress: self.service.resume_import_job(job_id, progress=progress),
        )

    def _on_forget_job(self, job_id: str) -> None:
        self.service.importer.forget_job(job_id)
        self.refresh_jobs()

    def _on_open_sql_import(self) -> None:
        """Open the SQL Database Import screen, reusing this window's service."""
        from gui.sql_import_dialog import SqlImportDialog

        parent = self.window
        self._sql_dialog = SqlImportDialog(
            parent, self.service, theme=self.theme, font=self.font, mono=self.mono,
            on_close=lambda: setattr(self, "_sql_dialog", None),
        )

    # -- worker -----------------------------------------------------------

    def _start(self, message: str, job: Any) -> None:
        """Run an import on a worker thread so the window never freezes."""
        self._busy = True
        self._set_buttons_enabled(False)
        self._set_status(message)
        self._progress.pack(side="right", padx=8)
        self._progress.start(12)

        def run() -> None:
            try:
                report = job(self._progress_line)
                self.window.after(0, partial(self._finish, report, None))
            except Exception as error:  # noqa: BLE001 - shown, never crashes Tk
                self.window.after(0, partial(self._finish, None, error))

        threading.Thread(target=run, daemon=True).start()

    def _progress_line(self, message: str) -> None:
        self.window.after(0, lambda: self._set_status(message))

    def _finish(self, report: Any, error: Exception | None) -> None:
        self._busy = False
        self._progress.stop()
        self._progress.pack_forget()
        self._set_buttons_enabled(True)
        if error is not None:
            self._set_status(f"Import failed: {error}", error=True)
            return
        if isinstance(report, optional_deps.InstallResult):
            self._finish_install(report)
            return
        assert report is not None
        self._last_report = report
        self._render_report(report)
        self.refresh_jobs()

    # -- rendering --------------------------------------------------------

    def _render_report(self, report: ImportReport) -> None:
        lines: list[str] = []
        lines.append(report.summary())
        lines.append("")
        for outcome in report.outcomes:
            marker = {"ok": "✓", "unchanged": "=", "failed": "✗"}.get(outcome.status, "?")
            detail = ""
            if outcome.status == "ok":
                detail = f"{outcome.records} record(s), {outcome.chunks} chunk(s)"
            elif outcome.status == "unchanged":
                detail = "already imported (unchanged)"
            else:
                detail = outcome.message
            lines.append(f"{marker}  {outcome.filename}  —  {detail}")
        self._set_results("\n".join(lines))
        if report.failed:
            self._set_status(
                f"Done: {report.succeeded} succeeded, {report.failed} failed.",
                error=True,
            )
            self._retry_button.pack(side="right", padx=8)
            if self._scanned_failures(report):
                self._retry_ocr_button.pack(side="right")
            else:
                self._retry_ocr_button.pack_forget()
        else:
            self._set_status(f"Done: {report.summary()}")
            self._retry_button.pack_forget()
            self._retry_ocr_button.pack_forget()

    @staticmethod
    def _scanned_failures(report: ImportReport) -> list[Any]:
        """The failures whose message says the PDF had no text layer."""
        return [
            outcome
            for outcome in report.failures
            if "no extractable text" in (outcome.message or "").lower()
        ]

    def refresh_jobs(self) -> None:
        """Rebuild the jobs list from the recorded job state."""
        theme = self.theme
        for child in self._jobs_frame.winfo_children():
            child.destroy()
        try:
            jobs = self.service.import_jobs()
        except Exception:  # noqa: BLE001 - a jobs read must not break the page
            jobs = []
        if not jobs:
            tk.Label(
                self._jobs_frame, text="No import jobs recorded yet.",
                bg=theme.surface, fg=theme.text_dim, font=(self.font, 9), anchor="w",
            ).pack(fill="x")
            return
        for state in jobs:
            row = tk.Frame(self._jobs_frame, bg=theme.surface)
            row.pack(fill="x", pady=1)
            label = f"{state.job_id}  [{state.status}]  pos {state.position}  ·  {state.source}"
            tk.Label(
                row, text=label, bg=theme.surface, fg=theme.text_secondary,
                font=(self.font, 8), anchor="w",
            ).pack(side="left", fill="x", expand=True)
            if state.status == "running":
                self._button(row, "Resume", lambda jid=state.job_id: self._on_resume_job(jid)).pack(
                    side="right", padx=(6, 0)
                )
            self._button(row, "Forget", lambda jid=state.job_id: self._on_forget_job(jid)).pack(
                side="right"
            )

    def _set_results(self, text: str) -> None:
        self._results.configure(state="normal")
        self._results.delete("1.0", "end")
        self._results.insert("1.0", text)
        self._results.configure(state="disabled")

    def _set_status(self, text: str, *, error: bool = False) -> None:
        colour = self.theme.danger if error else self.theme.text_secondary
        self._status.configure(text=text, fg=colour)

    def _set_buttons_enabled(self, enabled: bool) -> None:
        state: Literal["normal", "disabled"] = "normal" if enabled else "disabled"
        for button in (
            self._upload_button,
            self._folder_button,
            self._folder_import_button,
        ):
            button.configure(state=state)

    def _close(self) -> None:
        if self._on_close is not None:
            self._on_close()
        self.window.destroy()


__all__ = ["FORMATS_LINE", "SUBTITLE", "TITLE", "ImportCenterDialog"]
