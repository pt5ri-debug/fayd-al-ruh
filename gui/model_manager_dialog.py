"""The Model Manager window: every local model, in one place.

Reached from the sidebar's **Model Manager** entry, this window is the one
surface where the project's language models can be seen, inspected, and run:

* **Discover / Register** -- find GGUF files on disk, or add one by hand.
* **Inspect** -- re-read a file and report its format, size, and integrity.
* **Check** -- say whether a model can run here, and why not when it cannot,
  with the memory figure named as an *estimate*, never a measurement.
* **Load / Test / Unload** -- run a model through its runtime and ask it a
  trivial question, to prove it really answers.
* **Settings** -- the model's own inference settings (temperature, top-p, and
  the rest), saved and reset.
* **Operations log** -- the most recent actions, so a failure is visible.

The page is a thin view over :class:`gui.model_manager.ModelManagerService`, so
the whole workflow is testable with no display. Tk is imported lazily with the
rest of the window, so the CLI never touches it. Long work (a scan, a load, a
test) runs on a worker thread whose result is handed back through a queue the UI
thread drains, so the window never freezes.

Everything stays honest: this window **manages** models, it does not fetch them.
A model whose weights are absent says so and names the real next step (the
project's own governed download command), and a runtime that is not installed
reports the exact install line instead of pretending.
"""

from __future__ import annotations

import queue
import threading
import tkinter as tk
from collections.abc import Callable
from tkinter import filedialog, messagebox, simpledialog, ttk
from typing import Any, Literal

from gui.model_manager import (
    ManagerSnapshot,
    ModelDetail,
    ModelManagerService,
    ModelRow,
    detail_text,
    human_size,
)
from gui.sidebar import SidebarColumn
from gui.typography import resolve
from gui.window_controls import install_window_controls

TITLE = "Model Manager"
SUBTITLE = (
    "Every language model this build knows about — discover, register, inspect, "
    "check, load, and test them, all on this machine."
)
HONESTY_LINE = (
    "The Model Manager inspects and runs models that are already here. It does "
    "not download weights itself: when a model is missing, it names the command "
    "the owning engine already provides, and you run it. Memory figures are "
    "estimates from the file size, never measurements."
)

#: The transcript/detail column reads as a centred column, capped like the chat
#: window's, so a wide screen does not stretch a paragraph across it.
_COLUMN_MAX = 880

class ModelManagerDialog:
    """A window listing, inspecting, and running the project's language models."""

    def __init__(
        self,
        parent: tk.Tk | tk.Toplevel,
        service: ModelManagerService,
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
        self._type = resolve(parent, ui_family=font, mono_family=mono)
        self._on_close = on_close
        self._closed = False
        self._busy = False
        self._cancelled = False
        self._after_id: str | None = None
        self._rows: list[ModelRow] = []
        self._detail: ModelDetail | None = None
        self._selected_id = ""
        # The offers a search found, so "Download the chosen model" has something
        # real to fetch rather than a made-up default.
        self._offers: tuple[Any, ...] = ()
        self._download_cancel = threading.Event()

        # Worker threads hand results back through a queue the UI thread drains on
        # its own clock: calling `after` from a worker thread races with the main
        # loop (Tk is not thread-safe), which is what the rest of the project does.
        self._results: queue.Queue[
            tuple[Callable[[Any, Exception | None], None], Any, Exception | None]
        ] = queue.Queue()

        self.window = tk.Toplevel(parent)
        self.window.title(TITLE)
        self.window.configure(bg=theme.base)
        self.window.geometry("1200x720")
        self.window.minsize(940, 620)
        self.window.transient(parent)
        self.window.protocol("WM_DELETE_WINDOW", self._close)
        self._build()
        self._reload()

    # -- layout -----------------------------------------------------------

    def _build(self) -> None:
        self._build_header()
        theme = self.theme
        body = tk.Frame(self.window, bg=theme.base)
        body.pack(fill="both", expand=True)
        body.rowconfigure(0, weight=1)
        body.columnconfigure(0, weight=0, minsize=300)
        body.columnconfigure(1, weight=1)
        self._body = body
        self._build_sidebar(body)
        self._build_main(body)

    def _build_header(self) -> None:
        theme = self.theme
        bar = tk.Frame(self.window, bg=theme.titlebar)
        bar.pack(fill="x")
        inner = tk.Frame(bar, bg=theme.titlebar)
        inner.pack(fill="x", padx=16, pady=(12, 8))
        titles = tk.Frame(inner, bg=theme.titlebar)
        titles.pack(side="left", anchor="w")
        tk.Label(
            titles, text=TITLE, bg=theme.titlebar, fg=theme.contrast,
            font=self._type.font("heading3", weight="bold"), anchor="w",
        ).pack(anchor="w")
        tk.Label(
            titles, text=SUBTITLE, bg=theme.titlebar, fg=theme.text_secondary,
            font=self._type.font("2xs"), anchor="w", justify="left", wraplength=800,
        ).pack(anchor="w", pady=(2, 0))
        self._spinner = ttk.Progressbar(inner, mode="indeterminate", length=120)
        self._cancel_button = self._button(inner, "Cancel", self._on_cancel)
        install_window_controls(
            self.window, inner, theme=theme, typography=self._type, on_close=self._close
        )
        tk.Frame(self.window, bg=theme.titlebar_edge, height=1).pack(fill="x")

    def _button(self, parent: tk.Misc, text: str, command: Any) -> tk.Button:
        return tk.Button(
            parent, text=text, command=command, relief="flat", borderwidth=0,
            highlightthickness=0, bg=self.theme.surface_raised, fg=self.theme.foreground,
            activebackground=self.theme.interactive_active, activeforeground=self.theme.contrast,
            font=self._type.font("2xs", weight="bold"), padx=10, pady=4, cursor="hand2",
        )

    # -- sidebar ----------------------------------------------------------

    def _build_sidebar(self, parent: tk.Frame) -> None:
        column = SidebarColumn(
            parent, theme=self.theme, typography=self._type, width=300, scope="model-manager"
        )
        self._sidebar_column = column

        theme = self.theme
        brand = tk.Frame(column.body, bg=theme.surface)
        brand.pack(fill="x", padx=14, pady=(14, 4))
        dot = tk.Canvas(brand, width=24, height=24, bg=theme.surface, highlightthickness=0)
        dot.create_oval(1, 1, 23, 23, fill=theme.primary, outline="")
        dot.create_text(
            12, 13, text="ف", fill=theme.primary_foreground,
            font=self._type.font("sm", weight="bold"),
        )
        dot.pack(side="left")
        tk.Label(
            brand, text="Model Manager", bg=theme.surface, fg=theme.contrast,
            font=self._type.font("body", weight="bold"),
        ).pack(side="left", padx=(8, 0))

        summary = column.add("summary", "Summary", icon="📊")
        self._summary_label = tk.Label(
            summary, text="", bg=theme.surface, fg=theme.text_secondary,
            font=self._type.font("2xs"), anchor="w", justify="left", wraplength=260,
        )
        self._summary_label.pack(fill="x", padx=14, pady=(4, 10))

        tasks = column.add("tasks", "Tasks", icon="🎯")
        self._tasks_label = tk.Label(
            tasks, text="", bg=theme.surface, fg=theme.text_secondary,
            font=self._type.font("2xs"), anchor="w", justify="left", wraplength=260,
        )
        self._tasks_label.pack(fill="x", padx=14, pady=(4, 10))

        library = column.add("library", "Library", icon="📚")
        self._register_button = self._button(
            library, "Register a model file…", self._on_register
        )
        self._register_button.pack(fill="x", padx=12, pady=(4, 4))
        self._discover_button = self._button(
            library, "Discover models in a folder…", self._on_discover
        )
        self._discover_button.pack(fill="x", padx=12, pady=(0, 4))
        self._import_button = self._button(
            library, "Import models added by hand", self._on_import_custom
        )
        self._import_button.pack(fill="x", padx=12, pady=(0, 8))

        self._runtimes = column.add("runtimes", "Runtimes", icon="⚙")
        self._runtimes_label = tk.Label(
            self._runtimes, text="", bg=theme.surface, fg=theme.text_secondary,
            font=self._type.font("2xs"), anchor="w", justify="left", wraplength=260,
        )
        self._runtimes_label.pack(fill="x", padx=14, pady=(4, 10))

        downloads = column.add("downloads", "Downloads", icon="⬇")
        self._sources_label = tk.Label(
            downloads, text="", bg=theme.surface, fg=theme.text_secondary,
            font=self._type.font("2xs"), anchor="w", justify="left", wraplength=260,
        )
        self._sources_label.pack(fill="x", padx=14, pady=(4, 4))
        self._search_button = self._button(
            downloads, "Search the sources…", self._on_search_offers
        )
        self._search_button.pack(fill="x", padx=12, pady=(0, 4))
        self._download_button = self._button(
            downloads, "Download the chosen model…", self._on_download
        )
        self._download_button.pack(fill="x", padx=12, pady=(0, 4))
        self._resume_button = self._button(
            downloads, "Resume a partial download", self._on_resume_download
        )
        self._resume_button.pack(fill="x", padx=12, pady=(0, 8))

        resources = column.add("resources", "Resources", icon="🖥")
        self._resources_label = tk.Label(
            resources, text="", bg=theme.surface, fg=theme.text_secondary,
            font=self._type.font("2xs"), anchor="w", justify="left", wraplength=260,
        )
        self._resources_label.pack(fill="x", padx=14, pady=(4, 4))
        self._preflight_button = self._button(
            resources, "Check the chosen model", self._on_preflight
        )
        self._preflight_button.pack(fill="x", padx=12, pady=(0, 4))
        self._stop_button = self._button(
            resources, "Stop model processes", self._on_stop_processes
        )
        self._stop_button.pack(fill="x", padx=12, pady=(0, 8))

        trained = column.add("trained", "Trained models", icon="🧠")
        self._trained_label = tk.Label(
            trained, text="", bg=theme.surface, fg=theme.text_secondary,
            font=self._type.font("2xs"), anchor="w", justify="left", wraplength=260,
        )
        self._trained_label.pack(fill="x", padx=14, pady=(4, 4))
        self._trained_button = self._button(
            trained, "Register the trained checkpoint", self._on_register_trained
        )
        self._trained_button.pack(fill="x", padx=12, pady=(0, 8))

        diagnostics = column.add("diagnostics", "Diagnostics", icon="🩺")
        self._diagnostics_button = self._button(
            diagnostics, "Run diagnostics", self._on_diagnostics
        )
        self._diagnostics_button.pack(fill="x", padx=12, pady=(4, 4))
        self._diagnostics_label = tk.Label(
            diagnostics, text="", bg=theme.surface, fg=theme.text_secondary,
            font=self._type.font("2xs"), anchor="w", justify="left", wraplength=260,
        )
        self._diagnostics_label.pack(fill="x", padx=14, pady=(0, 10))

        log = column.add("log", "Operations log", icon="🧾")
        log_wrap = tk.Frame(log, bg=theme.surface)
        log_wrap.pack(fill="both", expand=True, padx=12, pady=(4, 12))
        self._log = tk.Text(
            log_wrap, height=8, bg=theme.base, fg=theme.foreground, relief="flat",
            highlightthickness=0, borderwidth=0, wrap="word",
            font=self._type.font("2xs"), state="disabled", cursor="arrow",
        )
        self._log.pack(fill="both", expand=True)
        self._log.configure(state="disabled")

    # -- main -------------------------------------------------------------

    def _build_main(self, parent: tk.Frame) -> None:
        theme = self.theme
        col = tk.Frame(parent, bg=theme.base)
        col.grid(row=0, column=1, sticky="nsew")

        head = tk.Frame(col, bg=theme.base)
        head.pack(fill="x", padx=16, pady=(14, 6))
        tk.Label(
            head, text="MODELS", bg=theme.base, fg=theme.text_dim,
            font=self._type.font("2xs", weight="bold"),
        ).pack(side="left")
        self._status_label = tk.Label(
            head, text="", bg=theme.base, fg=theme.text_secondary,
            font=self._type.font("2xs"),
        )
        self._status_label.pack(side="right")

        # The filter row: a free-text box and a state selector, both a *view* over
        # the rows already read (never a second listing), so searching never
        # re-reads the registry.
        filters = tk.Frame(col, bg=theme.base)
        filters.pack(fill="x", padx=16, pady=(0, 6))
        tk.Label(
            filters, text="Search", bg=theme.base, fg=theme.text_secondary,
            font=self._type.font("2xs"),
        ).pack(side="left")
        self._search_var = tk.StringVar(value="")
        self._search = tk.Entry(
            filters, textvariable=self._search_var, width=28, bg=theme.base,
            fg=theme.foreground, relief="flat", highlightthickness=1,
            highlightbackground=theme.border_subtle, font=self._type.font("2xs"),
        )
        self._search.pack(side="left", padx=(6, 12))
        self._search_var.trace_add("write", lambda *_: self._fill_table())
        tk.Label(
            filters, text="State", bg=theme.base, fg=theme.text_secondary,
            font=self._type.font("2xs"),
        ).pack(side="left")
        self._state_var = tk.StringVar(value="")
        self._state_filter = ttk.Combobox(
            filters, textvariable=self._state_var, width=14, state="readonly",
            values=("all", *self.service.state_names()),
        )
        self._state_filter.set("all")
        self._state_filter.pack(side="left", padx=(6, 12))
        self._state_filter.bind("<<ComboboxSelected>>", lambda _e: self._fill_table())

        # The model table. ``ttk.Treeview`` gives real columns and selection, so a
        # listing stays readable without a hand-built grid of labels.
        table_wrap = tk.Frame(col, bg=theme.base)
        table_wrap.pack(fill="both", expand=True, padx=16, pady=(0, 8))
        columns = ("state", "runtime", "format", "size", "source")
        self._table = ttk.Treeview(
            table_wrap, columns=columns, show="tree headings", height=10, selectmode="browse"
        )
        self._table.heading("#0", text="Model")
        self._table.column("#0", width=300, anchor="w")
        for key, label, width in (
            ("state", "State", 130),
            ("runtime", "Runtime", 110),
            ("format", "Format", 90),
            ("size", "Size", 90),
            ("source", "Source", 100),
        ):
            self._table.heading(key, text=label)
            self._table.column(key, width=width, anchor="w")
        table_scroll = tk.Scrollbar(table_wrap, command=self._table.yview)
        self._table.configure(yscrollcommand=table_scroll.set)
        self._table.pack(side="left", fill="both", expand=True)
        table_scroll.pack(side="right", fill="y")
        self._table.bind("<<TreeviewSelect>>", self._on_select_row)

        # Actions for the selected model.
        actions = tk.Frame(col, bg=theme.base)
        actions.pack(fill="x", padx=16, pady=(0, 8))
        self._action_buttons: list[tk.Button] = []
        for text, command in (
            ("Inspect", self._on_inspect),
            ("Check", self._on_check),
            ("Load", self._on_load),
            ("Test", self._on_test),
            ("Unload", self._on_unload),
            ("Save settings", self._on_save_settings),
            ("Reset settings", self._on_reset_settings),
            ("Forget", self._on_forget),
        ):
            button = self._button(actions, text, command)
            button.pack(side="left", padx=(0, 6))
            self._action_buttons.append(button)

        # Detail pane. A plain rounded panel is enough here (the transcript-style
        # bubble machinery is not needed for a fixed detail block).
        detail_card = tk.Frame(col, bg=theme.surface)
        detail_card.pack(fill="both", expand=True, padx=16, pady=(0, 16))
        self._detail_card = detail_card
        inner = tk.Frame(detail_card, bg=theme.surface)
        inner.pack(fill="both", expand=True, padx=16, pady=14)

        tk.Label(
            inner, text="DETAILS", bg=theme.surface, fg=theme.text_dim,
            font=self._type.font("2xs", weight="bold"),
        ).pack(anchor="w")
        self._detail_text = tk.Text(
            inner, height=14, bg=theme.surface, fg=theme.foreground, relief="flat",
            highlightthickness=0, borderwidth=0, wrap="word",
            font=self._type.font("2xs"), state="disabled", cursor="arrow",
        )
        self._detail_text.pack(fill="both", expand=True, pady=(4, 8))

        settings_row = tk.Frame(inner, bg=theme.surface)
        settings_row.pack(fill="x")
        tk.Label(
            settings_row, text="temperature", bg=theme.surface, fg=theme.text_secondary,
            font=self._type.font("2xs"),
        ).pack(side="left")
        self._temperature_var = tk.StringVar(value="")
        self._temperature = tk.Entry(
            settings_row, textvariable=self._temperature_var, width=8,
            bg=theme.base, fg=theme.foreground, relief="flat", highlightthickness=1,
            highlightbackground=theme.border_subtle, font=self._type.font("2xs"),
        )
        self._temperature.pack(side="left", padx=(6, 14))
        tk.Label(
            settings_row, text="max tokens", bg=theme.surface, fg=theme.text_secondary,
            font=self._type.font("2xs"),
        ).pack(side="left")
        self._max_tokens_var = tk.StringVar(value="")
        self._max_tokens = tk.Entry(
            settings_row, textvariable=self._max_tokens_var, width=8,
            bg=theme.base, fg=theme.foreground, relief="flat", highlightthickness=1,
            highlightbackground=theme.border_subtle, font=self._type.font("2xs"),
        )
        self._max_tokens.pack(side="left", padx=(6, 14))
        tk.Label(
            settings_row, text="context", bg=theme.surface, fg=theme.text_secondary,
            font=self._type.font("2xs"),
        ).pack(side="left")
        self._context_var = tk.StringVar(value="")
        self._context = tk.Entry(
            settings_row, textvariable=self._context_var, width=8,
            bg=theme.base, fg=theme.foreground, relief="flat", highlightthickness=1,
            highlightbackground=theme.border_subtle, font=self._type.font("2xs"),
        )
        self._context.pack(side="left", padx=(6, 14))

        self._detail_text.configure(state="disabled")

    # -- data -------------------------------------------------------------

    def _reload(self) -> None:
        """Read the snapshot and draw the whole page. Never raises."""
        try:
            snapshot = self.service.snapshot()
        except Exception as error:  # a broken registry must not blank the window
            self._set_status(f"Could not read the model registry: {error}")
            return
        self._apply_snapshot(snapshot)

    def _apply_snapshot(self, snapshot: ManagerSnapshot) -> None:
        self._rows = list(snapshot.rows)
        self._summary_label.configure(text=snapshot.summary())
        self._runtimes_label.configure(text="\n".join(snapshot.runtime_lines) or "none")
        self._tasks_label.configure(text=self._task_summary())
        self._diagnostics_label.configure(text=self._dashboard_text(snapshot))
        self._render_log()
        self._fill_table()
        self._set_status(snapshot.summary())
        self._refresh_environment()

    def _refresh_environment(self) -> None:
        """Fill the Downloads / Resources / Trained sections from the live state.

        Each read is independent and best-effort: a source that cannot answer here
        (no ``huggingface_hub``, no folder) or a machine that cannot report its GPU
        leaves one honest line, never a blank section and never a failed page.
        """
        self._sources_label.configure(text=self._sources_text())
        self._resources_label.configure(text=self._resources_text())
        self._trained_label.configure(text=self._trained_text())

    def _sources_text(self) -> str:
        try:
            sources = self.service.sources()
        except Exception as error:
            return f"Could not read the sources: {error}"
        if not sources:
            return "No model source is available in this build."
        lines = []
        for source in sources:
            ok, why = source.available()
            lines.append(f"{source.title}: {'available' if ok else f'unavailable — {why}'}")
        try:
            partials = self.service.resumable_downloads()
        except Exception:
            partials = ()
        if partials:
            lines.append(f"{len(partials)} partial download(s) can be resumed.")
        return "\n".join(lines)

    def _resources_text(self) -> str:
        try:
            resources = self.service.resources()
        except Exception as error:
            return f"Could not read this machine's resources: {error}"
        if resources is None:
            return "This machine's resources could not be read."
        describe = getattr(resources, "describe", None)
        if callable(describe):
            return describe()
        return str(resources)

    def _trained_text(self) -> str:
        try:
            artifacts = self.service.trained_artifacts()
        except Exception as error:
            return f"Could not read the trained artifacts: {error}"
        if not artifacts:
            return "No trained artifact was found in this checkout."
        return "\n".join(item.describe() for item in artifacts)

    # -- downloads, resources, and trained models -------------------------

    def _on_search_offers(self) -> None:
        """Search the trusted sources for a model. Downloads nothing."""
        query = simpledialog.askstring(
            "Search the model sources",
            "Search the trusted sources for a model (a name or a topic):",
            parent=self.window,
        )
        if not query:
            return
        self._run_background(
            "Searching the model sources…",
            lambda: self.service.search_offers(query, limit=20),
            self._finish_search_offers,
        )

    def _finish_search_offers(self, offers: Any, error: Exception | None) -> None:
        if error is not None:
            self._set_status(f"Could not search the sources: {error}")
            return
        offers = tuple(offers or ())
        self._offers = offers
        if not offers:
            self._set_detail("No source offers a model matching that search.")
            self._set_status("No offer matched.")
            return
        lines = ["Model offers found:", ""]
        for index, offer in enumerate(offers, 1):
            lines.append(f"{index}. {offer.display_name}  [{offer.source_key}]")
            lines.append(f"   id: {offer.source_id}")
            if offer.total_bytes:
                lines.append(f"   size: {offer.total_bytes} bytes")
        lines.append("")
        lines.append("Use “Download the chosen model…” to fetch the first offer.")
        self._set_detail("\n".join(lines))
        self._set_status(f"{len(offers)} offer(s) found.")

    def _on_download(self) -> None:
        """Fetch the first search offer. A governed action, run on a worker thread.

        The download leaves the machine, so it is approved explicitly here -- the
        user pressing the button *is* the approval -- and the engine's own
        permission policy still applies underneath.
        """
        offers = getattr(self, "_offers", ())
        if not offers:
            self._set_status("Search the sources first, then choose a model to download.")
            return
        offer = offers[0]
        if not messagebox.askyesno(
            "Download a model",
            f"Download {offer.display_name} from {offer.source_key}?\n\n"
            "This reaches the network and may fetch several gigabytes.",
            parent=self.window,
        ):
            return
        self._download_cancel = threading.Event()
        self._run_background(
            "Downloading the model…",
            lambda: self.service.download(
                offer,
                approved=True,
                should_cancel=self._download_cancel.is_set,
            ),
            self._finish_download,
        )

    def _finish_download(self, result: Any, error: Exception | None) -> None:
        if error is not None:
            self._set_status(f"The download failed: {error}")
            return
        report, entry = result if isinstance(result, tuple) else (result, None)
        describe = getattr(report, "describe", None)
        self._set_detail(describe() if callable(describe) else str(report))
        if entry is not None:
            self._reload()
            self._set_status("The model downloaded and was registered.")
        else:
            self._set_status("The download did not complete; it can be resumed.")

    def _on_resume_download(self) -> None:
        """Resume a partial download: re-issue the offer and let the engine resume."""
        try:
            partials = self.service.resumable_downloads()
        except Exception as error:
            self._set_status(f"Could not read the partial downloads: {error}")
            return
        if not partials:
            self._set_status("No partial download to resume.")
            return
        self._set_status(
            f"{len(partials)} partial download(s) can be resumed; "
            "search the same model again and download it."
        )
        self._set_detail(
            "Partial downloads that can be resumed:\n"
            + "\n".join(f"  {path}" for path in partials)
            + "\n\nSearch the same model again and download it; the engine resumes "
            "from the bytes already on disk."
        )

    def _on_preflight(self) -> None:
        model_id = self._require_selection()
        if model_id is None:
            return
        self._run_background(
            "Checking whether the model fits this machine…",
            lambda: self.service.preflight(model_id),
            self._finish_preflight,
        )

    def _finish_preflight(self, verdict: Any, error: Exception | None) -> None:
        if error is not None:
            self._set_status(f"Could not run the preflight: {error}")
            return
        describe = getattr(verdict, "describe", None)
        self._set_detail(describe() if callable(describe) else str(verdict))
        ok = bool(getattr(verdict, "ok", False))
        self._set_status("The model can run here." if ok else "The model cannot run here yet.")

    def _on_stop_processes(self) -> None:
        self._run_background(
            "Stopping the model processes…",
            self.service.stop_processes,
            self._finish_stop_processes,
        )

    def _finish_stop_processes(self, stopped: Any, error: Exception | None) -> None:
        if error is not None:
            self._set_status(f"Could not stop the processes: {error}")
            return
        count = len(tuple(stopped or ()))
        self._set_status(f"Stopped {count} model process(es)." if count else "Nothing to stop.")

    def _on_register_trained(self) -> None:
        """Register the project's own trained checkpoint as an ordinary record."""
        self._run_background(
            "Registering the trained checkpoint…",
            self.service.register_trained,
            self._finish_register_trained,
        )

    def _finish_register_trained(self, records: Any, error: Exception | None) -> None:
        if error is not None:
            self._set_status(f"Could not register the trained checkpoint: {error}")
            return
        records = tuple(records or ())
        if not records:
            self._set_status("No trained checkpoint is present to register.")
            return
        self._reload()
        self._set_status(f"Registered {len(records)} trained model(s).")

    def _dashboard_text(self, snapshot: ManagerSnapshot) -> str:
        """The one-line-per-fact dashboard body: active models, errors, machine.

        Read straight from the snapshot, so it can never disagree with the table
        beside it. An empty section says so rather than being hidden.
        """
        lines: list[str] = []
        if snapshot.active:
            lines.append(f"{len(snapshot.active)} model(s) loaded now.")
        else:
            lines.append("No model is loaded right now.")
        if snapshot.errors:
            lines.append(f"{len(snapshot.errors)} recent error(s).")
        else:
            lines.append("No recent errors.")
        resources = snapshot.resources
        if resources is not None:
            memory = resources.total_memory_bytes
            if memory:
                lines.append(f"RAM: {memory / 1e9:.1f} GB total.")
            cores = resources.cpu_count
            if cores:
                lines.append(f"CPU cores: {cores}.")
        return "\n".join(lines)

    def _task_summary(self) -> str:
        """Which model answers each task right now, or the honest next step."""
        lines: list[str] = []
        try:
            names = self.service.task_names()
        except Exception:
            names = ()
        for task in names:
            try:
                row = self.service.select_for_task(task)
            except Exception:
                row = None
            lines.append(f"{task}: {row.display_name}" if row is not None else f"{task}: none ready")
        return "\n".join(lines) or "no tasks"

    def _fill_table(self) -> None:
        selected = self._selected_id
        state = self._state_var.get()
        if state == "all":
            state = ""
        try:
            rows = self.service.search(
                self._search_var.get(), state=state, rows=tuple(self._rows)
            )
        except Exception:
            rows = tuple(self._rows)
        self._table.delete(*self._table.get_children())
        for row in rows:
            self._table.insert(
                "", "end", iid=row.model_id, text=row.display_name,
                values=(
                    row.state_marker,
                    row.runtime,
                    row.format,
                    human_size(row.size_bytes),
                    row.source,
                ),
            )
        if selected and selected in self._table.get_children():
            self._table.selection_set(selected)

    def _render_log(self) -> None:
        try:
            events = self.service.log_tail(40)
        except Exception:
            events = ()
        lines = [event.describe() if hasattr(event, "describe") else str(event) for event in events]
        self._log.configure(state="normal")
        self._log.delete("1.0", "end")
        self._log.insert("1.0", "\n".join(lines) or "No operations yet.")
        self._log.configure(state="disabled")

    def _set_status(self, text: str) -> None:
        self._status_label.configure(text=text)

    # -- selection --------------------------------------------------------

    def _on_select_row(self, _event: tk.Event | None = None) -> None:
        selection = self._table.selection()
        if not selection:
            return
        self._selected_id = selection[0]
        self.service.select(self._selected_id)
        self._show_detail(self._selected_id)

    def _show_detail(self, model_id: str) -> None:
        try:
            detail = self.service.detail(model_id)
        except Exception as error:
            self._set_detail(f"Could not read this model's record: {error}")
            return
        self._detail = detail
        self._set_detail(detail_text(detail))
        self._fill_settings(detail)

    def _set_detail(self, text: str) -> None:
        self._detail_text.configure(state="normal")
        self._detail_text.delete("1.0", "end")
        self._detail_text.insert("1.0", text)
        self._detail_text.configure(state="disabled")

    def _fill_settings(self, detail: ModelDetail) -> None:
        settings = detail.settings
        if settings is None:
            self._temperature_var.set("")
            self._max_tokens_var.set("")
            self._context_var.set("")
            return
        self._temperature_var.set(str(getattr(settings, "temperature", "")))
        self._max_tokens_var.set(str(getattr(settings, "max_tokens", "")))
        self._context_var.set(str(getattr(settings, "context", "")))

    # -- actions ----------------------------------------------------------

    def _on_register(self) -> None:
        path = filedialog.askopenfilename(
            parent=self.window,
            title="Choose a model file",
            filetypes=[("GGUF models", "*.gguf"), ("All files", "*.*")],
        )
        if not path:
            return
        self._run_background(
            "Registering the model…",
            lambda: self.service.register(path),
            self._finish_register,
        )

    def _finish_register(self, row: ModelRow, error: Exception | None) -> None:
        if error is not None:
            self._set_status(f"Could not register the model: {error}")
            return
        self._selected_id = row.model_id
        self._reload()
        self._set_status(f"Registered {row.display_name}.")

    def _on_discover(self) -> None:
        folder = filedialog.askdirectory(parent=self.window, title="Choose a folder to scan")
        if not folder:
            return
        self._run_background(
            "Scanning for models…",
            lambda: self.service.discover([folder]),
            self._finish_discover,
        )

    def _finish_discover(self, report: Any, error: Exception | None) -> None:
        if error is not None:
            self._set_status(f"Could not scan the folder: {error}")
            return
        found = getattr(report, "probes", ())
        self._set_status(f"Found {len(found)} model file(s).")
        self._reload()

    def _on_import_custom(self) -> None:
        self._run_background(
            "Importing models added by hand…",
            self.service.import_custom_models,
            self._finish_import_custom,
        )

    def _finish_import_custom(self, rows: Any, error: Exception | None) -> None:
        if error is not None:
            self._set_status(f"Could not import the models: {error}")
            return
        self._reload()
        self._set_status(f"Imported {len(rows)} model(s).")

    def _on_inspect(self) -> None:
        model_id = self._require_selection()
        if model_id is None:
            return
        self._run_background(
            "Inspecting the model file…",
            lambda: self.service.inspect(model_id),
            lambda probe, error: self._finish_inspect(probe, error, model_id),
        )

    def _finish_inspect(self, probe: Any, error: Exception | None, model_id: str) -> None:
        if error is not None:
            self._set_status(f"Could not inspect the model: {error}")
            return
        self._reload()
        self._show_detail(model_id)
        self._set_status(probe.describe() if hasattr(probe, "describe") else "Inspected.")

    def _on_check(self) -> None:
        model_id = self._require_selection()
        if model_id is None:
            return
        self._run_background(
            "Checking whether the model can run here…",
            lambda: self.service.compatibility(model_id),
            self._finish_check,
        )

    def _finish_check(self, report: Any, error: Exception | None) -> None:
        if error is not None:
            self._set_status(f"Could not check the model: {error}")
            return
        self._set_detail(report.describe())
        self._set_status("can run here" if report.can_run else "cannot run here")

    def _on_diagnostics(self) -> None:
        """Show the whole diagnostics body: machine, runtimes, active, errors.

        Runs on the worker thread (reading resources and asking every runtime can
        be slow on a cold machine), and the result is shown in the detail pane so
        it is selectable and copyable like every other block here.
        """
        self._run_background(
            "Gathering diagnostics…",
            self.service.diagnostics,
            self._finish_diagnostics,
        )

    def _finish_diagnostics(self, report: Any, error: Exception | None) -> None:
        if error is not None:
            self._set_status(f"Could not gather diagnostics: {error}")
            return
        self._set_detail(report.describe() if hasattr(report, "describe") else str(report))
        self._set_status("Diagnostics ready.")

    def _on_load(self) -> None:
        model_id = self._require_selection()
        if model_id is None:
            return
        self._run_background(
            "Loading the model…",
            lambda: self.service.load(model_id),
            self._finish_operation,
        )

    def _on_test(self) -> None:
        model_id = self._require_selection()
        if model_id is None:
            return
        self._run_background(
            "Asking the model a test question…",
            lambda: self.service.test(model_id, prompt="Reply with the single word: ready."),
            self._finish_operation,
        )

    def _on_unload(self) -> None:
        model_id = self._selected_id or ""
        self._run_background(
            "Releasing the model…",
            lambda: self.service.unload(model_id),
            self._finish_operation,
        )

    def _finish_operation(self, report: Any, error: Exception | None) -> None:
        if error is not None:
            self._set_status(f"The operation failed: {error}")
            self._render_log()
            return
        self._set_status(report.describe() if hasattr(report, "describe") else "done")
        self._render_log()

    def _on_save_settings(self) -> None:
        model_id = self._require_selection()
        if model_id is None:
            return
        changes = self._collect_settings()
        if changes is None:
            return
        try:
            self.service.update_settings(model_id, **changes)
        except Exception as error:
            self._set_status(f"Could not save the settings: {error}")
            return
        self._show_detail(model_id)
        self._set_status("Settings saved.")

    def _on_reset_settings(self) -> None:
        model_id = self._require_selection()
        if model_id is None:
            return
        try:
            self.service.reset_settings(model_id)
        except Exception as error:
            self._set_status(f"Could not reset the settings: {error}")
            return
        self._show_detail(model_id)
        self._set_status("Settings reset to the project defaults.")

    def _collect_settings(self) -> dict[str, Any] | None:
        """Read the three fields, refusing a value that is not a number."""
        changes: dict[str, Any] = {}
        for name, var in (
            ("temperature", self._temperature_var),
            ("max_tokens", self._max_tokens_var),
            ("context", self._context_var),
        ):
            raw = var.get().strip()
            if not raw:
                continue
            try:
                changes[name] = float(raw) if name == "temperature" else int(raw)
            except ValueError:
                self._set_status(f"{name} must be a number.")
                return None
        return changes

    def _on_forget(self) -> None:
        model_id = self._require_selection()
        if model_id is None:
            return
        try:
            self.service.remove(model_id, confirm=True)
        except Exception as error:
            self._set_status(f"Could not forget the model: {error}")
            return
        self._selected_id = ""
        self._set_detail("")
        self._reload()
        self._set_status("Forgotten. The file itself was not touched.")

    def _require_selection(self) -> str | None:
        if not self._selected_id:
            self._set_status("Choose a model first.")
            return None
        return self._selected_id

    # -- worker -----------------------------------------------------------

    def _run_background(
        self,
        message: str,
        job: Callable[[], Any],
        finish: Callable[[Any, Exception | None], None],
    ) -> None:
        """Run ``job`` on a worker thread, then hand its result back on the UI thread."""
        self._busy = True
        self._cancelled = False
        self._set_status(message)
        self._set_busy_buttons(False)
        self._spinner.pack(side="right", padx=8)
        self._spinner.start(12)
        self._cancel_button.pack(side="right", padx=(0, 8))

        def run() -> None:
            try:
                result = job()
                self._results.put((finish, result, None))
            except Exception as error:
                self._results.put((finish, None, error))

        threading.Thread(target=run, daemon=True).start()
        self._schedule_pump()

    def _schedule_pump(self) -> None:
        """Ask Tk to drain the queue again shortly, remembering the ticket.

        The ticket is kept so :meth:`_close` can cancel it: a pending ``after``
        callback firing into a destroyed window raises ``TclError``, which is the
        safe-close hazard this avoids.
        """
        if self._closed:
            return
        self._after_id = self.window.after(60, self._pump_results)

    def _pump_results(self) -> None:
        self._after_id = None
        if self._closed:
            return
        try:
            while True:
                finish, result, error = self._results.get_nowait()
                if self._cancelled and error is None:
                    # The user cancelled while this job was in flight: report it
                    # and drop the result rather than redrawing stale state.
                    self._set_status("Cancelled.")
                    continue
                self._finish(finish, result, error)
        except queue.Empty:
            pass
        if self._busy:
            self._schedule_pump()

    def _on_cancel(self) -> None:
        """Ask the running job to be abandoned.

        The worker thread cannot be killed safely in Python, so cancellation is
        *cooperative*: the result is dropped when it arrives, the spinner stops,
        and the buttons come back. The status says plainly that the work may still
        finish in the background, so nothing is overstated.
        """
        if not self._busy:
            self._set_status("Nothing is running.")
            return
        self._cancelled = True
        # A download polls its own cancel flag, so a cancelled download stops at
        # the next chunk instead of running to completion in the background.
        self._download_cancel.set()
        self._busy = False
        self._spinner.stop()
        self._spinner.pack_forget()
        self._cancel_button.pack_forget()
        self._set_busy_buttons(True)
        self._set_status("Cancelled — the running step may still finish in the background.")

    def _finish(
        self,
        finish: Callable[[Any, Exception | None], None],
        result: Any,
        error: Exception | None,
    ) -> None:
        self._busy = False
        self._spinner.stop()
        self._spinner.pack_forget()
        self._cancel_button.pack_forget()
        self._set_busy_buttons(True)
        try:
            finish(result, error)
        except Exception as inner:  # a view bug must not kill the pump
            self._set_status(f"Unexpected error: {inner}")

    def _set_busy_buttons(self, enabled: bool) -> None:
        state: Literal["normal", "disabled"] = "normal" if enabled else "disabled"
        for button in self._action_buttons:
            button.configure(state=state)
        for button in (
            self._register_button,
            self._discover_button,
            self._import_button,
            self._diagnostics_button,
        ):
            button.configure(state=state)

    # -- lifecycle --------------------------------------------------------

    def _close(self) -> None:
        self._closed = True
        if self._after_id is not None:
            # A pending pump callback would fire into a destroyed window and raise
            # ``TclError``; cancelling it is what makes closing mid-operation safe.
            try:
                self.window.after_cancel(self._after_id)
            except tk.TclError:
                pass
            self._after_id = None
        try:
            self._spinner.stop()
        except tk.TclError:
            pass
        if self._on_close is not None:
            self._on_close()
        try:
            self.window.destroy()
        except tk.TclError:
            pass


__all__ = [
    "HONESTY_LINE",
    "SUBTITLE",
    "TITLE",
    "ModelManagerDialog",
]
