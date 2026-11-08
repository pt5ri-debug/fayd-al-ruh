"""The Maintenance window: backup/restore, diagnostics, and keyboard shortcuts.

Phase 5 of the UI restructuring closes the gaps in
``docs/ui/UI_GAPS_AND_MISSING_FEATURES.md``. Three of them need a *surface*: a
user must be able to back up and restore their own data, read the machine's and
the models' diagnostics, and find out which keys the program binds. All three are
served by one window over the Tk-free :class:`gui.maintenance.MaintenanceService`,
so the window and ``python -m cli maintenance`` read **one** service -- no copy,
check, or figure is computed twice.

It is built the way every other window in the project is built: the shared
:class:`gui.components.DialogHeader` title band, the shared
:class:`gui.sidebar.SidebarColumn` on the left, and a worker thread whose result
is handed back through a queue the UI thread drains (Tk is not thread-safe).

Honest by construction: a restore is a **governed** action that asks for
confirmation first (it replaces the current data), a store that is absent is named
as absent, and a failure is shown with its own reason rather than reported as a
success.
"""

from __future__ import annotations

import queue
import threading
import tkinter as tk
from collections.abc import Callable
from tkinter import filedialog, messagebox
from typing import Any

from gui.dialog_base import ThemedDialog
from gui.maintenance import STORE_TITLES, MaintenanceService, owner_choices
from gui.sidebar import SidebarColumn
from gui.typography import resolve
from gui.window_controls import fit_geometry

TITLE = "Maintenance"
SUBTITLE = (
    "Back up and restore your own data, read the machine's and the models' "
    "diagnostics, and see the keyboard shortcuts the program binds."
)
HONESTY_LINE = (
    "Backup copies your real stores; Restore is governed and asks first, then "
    "keeps the current data aside as a safety copy. Diagnostics report what the "
    "machine and the Model Manager actually say, never a guess."
)

DEFAULT_WIDTH = 1080
DEFAULT_HEIGHT = 700
MIN_WIDTH = 900
MIN_HEIGHT = 560
SIDEBAR_WIDTH = 320

_COLUMN_MAX = 820
_MAX_CARD_LINES = 20


class MaintenanceDialog(ThemedDialog):
    """The one window for backup/restore, diagnostics, and shortcuts."""

    def __init__(
        self,
        parent: tk.Tk | tk.Toplevel,
        service: MaintenanceService,
        *,
        theme: Any,
        font: str,
        mono: str,
        overrides: object | None = None,
        list_overrides: object | None = None,
        on_close: Callable[[], None] | None = None,
    ) -> None:
        self.service = service
        self.theme = theme
        self.font = font
        self.mono = mono
        self._type = resolve(parent, ui_family=font, mono_family=mono, overrides=overrides)
        self._overrides = overrides
        self._list_overrides = list_overrides
        self._on_close = on_close
        self._closed = False
        self._busy = False
        self._pump_after: str | None = None
        self._cards: list[tk.Text] = []
        self._results: queue.Queue[
            tuple[Callable[[Any, Exception | None], None], Any, Exception | None]
        ] = queue.Queue()
        self._store_var = tk.StringVar(value=next(iter(STORE_TITLES)))
        self._owner_choices = owner_choices() or ("fayd_al_ruh",)

        self.window = tk.Toplevel(parent)
        self.window.title(TITLE)
        self.window.configure(bg=theme.base)
        self.window.geometry(
            fit_geometry(self.window, DEFAULT_WIDTH, DEFAULT_HEIGHT, min_height=MIN_HEIGHT)
        )
        self.window.minsize(MIN_WIDTH, MIN_HEIGHT)
        self.window.transient(parent)
        self.window.protocol("WM_DELETE_WINDOW", self._close)
        self._themed_build()

    def _after_build(self) -> None:
        """Run the window's post-build work (first build, and after a re-theme)."""
        self._reload()

    # -- layout -----------------------------------------------------------

    def _build(self) -> None:
        self._build_header()
        theme = self.theme
        body = tk.Frame(self.window, bg=theme.base)
        body.pack(fill="both", expand=True)
        body.rowconfigure(0, weight=1)
        body.columnconfigure(0, weight=0, minsize=SIDEBAR_WIDTH)
        body.columnconfigure(1, weight=1)
        self._build_sidebar(body)
        self._build_main(body)

    def _build_header(self) -> None:
        from gui.components import DialogHeader

        header = DialogHeader(
            self.window, title=TITLE, subtitle=SUBTITLE, theme=self.theme,
            typography=self._type, on_close=self._close,
        )
        self._window_controls = header.controls

    def _button(self, parent: tk.Misc, text: str, command: Any) -> tk.Button:
        from gui.components import button

        return button(
            parent, text, command, theme=self.theme, typography=self._type,
            variant="secondary", small=True, anchor="w",
        )

    def _build_sidebar(self, parent: tk.Frame) -> None:
        column = SidebarColumn(
            parent, theme=self.theme, typography=self._type, width=SIDEBAR_WIDTH,
            scope="maintenance",
        )
        self._sidebar_column = column
        theme = self.theme

        stores = column.add("stores", "Stores", icon="🗄", open=True)
        self._stores_label = tk.Label(
            stores, text="", bg=theme.surface, fg=theme.text_secondary,
            font=self._type.font("2xs"), anchor="w", justify="left", wraplength=280,
        )
        self._stores_label.pack(fill="x", padx=14, pady=(4, 4))
        tk.Label(
            stores, text="Which store:", bg=theme.surface, fg=theme.text_dim,
            font=self._type.font("2xs"), anchor="w",
        ).pack(fill="x", padx=14, pady=(4, 0))
        picker = tk.OptionMenu(stores, self._store_var, *STORE_TITLES)
        picker.configure(
            bg=theme.base, fg=theme.foreground, activebackground=theme.interactive_hover,
            activeforeground=theme.foreground, relief="flat", highlightthickness=0,
            font=self._type.font("2xs"), anchor="w",
        )
        picker["menu"].configure(bg=theme.base, fg=theme.foreground, font=self._type.font("2xs"))
        picker.pack(fill="x", padx=12, pady=(2, 6))
        self._button(stores, "⤓   Back up this store", self._on_backup).pack(
            fill="x", padx=12, pady=2
        )
        self._button(stores, "⤒   Restore this store…", self._on_restore).pack(
            fill="x", padx=12, pady=(2, 8)
        )

        diagnostics = column.add("diagnostics", "Diagnostics", icon="🩺", open=True)
        self._button(diagnostics, "⟳   Read diagnostics", self._on_diagnostics).pack(
            fill="x", padx=12, pady=(4, 2)
        )
        self._button(diagnostics, "📜   Show the operation log", self._on_log).pack(
            fill="x", padx=12, pady=(2, 8)
        )

        shortcuts = column.add("shortcuts", "Keyboard shortcuts", icon="⌨", open=False)
        self._shortcuts_label = tk.Label(
            shortcuts, text="", bg=theme.surface, fg=theme.text_secondary,
            font=self._type.font("2xs"), anchor="w", justify="left", wraplength=280,
        )
        self._shortcuts_label.pack(fill="x", padx=14, pady=(4, 8))

        memory = column.add("memory", "Memory & Knowledge", icon="🧠", open=False)
        tk.Label(
            memory, text="Which personality:", bg=theme.surface, fg=theme.text_dim,
            font=self._type.font("2xs"), anchor="w",
        ).pack(fill="x", padx=14, pady=(4, 0))
        self._owner_var = tk.StringVar(value=self._owner_choices[0])
        owner_picker = tk.OptionMenu(memory, self._owner_var, *self._owner_choices)
        owner_picker.configure(
            bg=theme.base, fg=theme.foreground, activebackground=theme.interactive_hover,
            activeforeground=theme.foreground, relief="flat", highlightthickness=0,
            font=self._type.font("2xs"), anchor="w",
        )
        owner_picker["menu"].configure(
            bg=theme.base, fg=theme.foreground, font=self._type.font("2xs")
        )
        owner_picker.pack(fill="x", padx=12, pady=(2, 4))
        self._button(memory, "🧠   Show this memory", self._on_memory).pack(
            fill="x", padx=12, pady=2
        )
        tk.Label(
            memory, text="Search the knowledge base:", bg=theme.surface, fg=theme.text_dim,
            font=self._type.font("2xs"), anchor="w",
        ).pack(fill="x", padx=14, pady=(6, 0))
        self._kb_query = tk.Entry(
            memory, bg=theme.base, fg=theme.foreground, relief="flat", highlightthickness=0,
            insertbackground=theme.foreground, font=self._type.font("2xs"),
        )
        self._kb_query.pack(fill="x", padx=12, pady=(2, 2))
        self._kb_query.bind("<Return>", lambda _e: self._on_kb_search())
        self._button(memory, "🔎   Search the knowledge base", self._on_kb_search).pack(
            fill="x", padx=12, pady=(2, 8)
        )

        about = column.add("about", "About this page", icon="ℹ", open=False)
        tk.Label(
            about, text=HONESTY_LINE, bg=theme.surface, fg=theme.text_dim,
            font=self._type.font("2xs"), anchor="w", justify="left", wraplength=280,
        ).pack(fill="x", padx=14, pady=(4, 8))

    def _build_main(self, parent: tk.Frame) -> None:
        theme = self.theme
        col = tk.Frame(parent, bg=theme.base)
        col.grid(row=0, column=1, sticky="nsew")

        head = tk.Frame(col, bg=theme.base)
        head.pack(fill="x", padx=16, pady=(14, 6))
        tk.Label(
            head, text="MAINTENANCE", bg=theme.base, fg=theme.text_dim,
            font=self._type.font("2xs", weight="bold"),
        ).pack(side="left")
        self._status_label = tk.Label(
            head, text="", bg=theme.base, fg=theme.text_secondary,
            font=self._type.font("2xs"),
        )
        self._status_label.pack(side="right")

        holder = tk.Frame(col, bg=theme.base)
        holder.pack(fill="both", expand=True, padx=16, pady=(0, 16))
        self._canvas = tk.Canvas(holder, bg=theme.base, highlightthickness=0, borderwidth=0)
        self._canvas.pack(side="left", fill="both", expand=True)
        scroll = tk.Scrollbar(holder, orient="vertical", command=self._canvas.yview)
        scroll.pack(side="right", fill="y")
        self._canvas.configure(yscrollcommand=scroll.set)
        self._transcript = tk.Frame(self._canvas, bg=theme.base)
        self._transcript_id = self._canvas.create_window((0, 0), window=self._transcript, anchor="nw")
        self._transcript.bind(
            "<Configure>",
            lambda _e: self._canvas.configure(scrollregion=self._canvas.bbox("all")),
        )
        self._canvas.bind(
            "<Configure>", lambda e: self._canvas.itemconfigure(self._transcript_id, width=e.width)
        )
        for widget in (self._canvas, self._transcript):
            widget.bind("<MouseWheel>", self._on_wheel)
            widget.bind("<Button-4>", self._on_wheel)
            widget.bind("<Button-5>", self._on_wheel)

    def _on_wheel(self, event: tk.Event) -> None:
        delta = getattr(event, "delta", 0)
        if delta:
            self._canvas.yview_scroll(int(-delta / 120), "units")
        elif getattr(event, "num", None) == 4:
            self._canvas.yview_scroll(-1, "units")
        elif getattr(event, "num", None) == 5:
            self._canvas.yview_scroll(1, "units")

    # -- data -------------------------------------------------------------

    def _reload(self) -> None:
        """Read the stores and the shortcuts, then draw the page."""
        try:
            statuses = self.service.store_statuses()
        except Exception as error:  # a broken store read must not blank the window
            self._set_status(f"Could not read the stores: {error}")
            return
        self._stores_label.configure(text="\n".join(status.describe() for status in statuses))
        self._shortcuts_label.configure(
            text="\n".join(f"{s.keys:<12} {s.action}" for s in self.service.shortcuts())
        )
        self._set_status("Ready")

    # -- actions ----------------------------------------------------------

    def _on_backup(self) -> None:
        if self._busy:
            return
        store = self._store_var.get()
        self._run_background(
            f"Backing up {STORE_TITLES.get(store, store)}…",
            lambda: self.service.backup(store),
            self._finish_report,
        )

    def _on_restore(self) -> None:
        if self._busy:
            return
        store = self._store_var.get()
        path = filedialog.askdirectory(
            parent=self.window, title=f"Choose a {STORE_TITLES.get(store, store)} backup"
        )
        if not path:
            return
        if not messagebox.askyesno(
            TITLE,
            "Restoring replaces the current store with the chosen backup.\n\n"
            "The current data is kept aside as a safety copy. Continue?",
            parent=self.window,
        ):
            return
        self._run_background(
            f"Restoring {STORE_TITLES.get(store, store)}…",
            lambda: self.service.restore(store, path),
            self._finish_report,
        )

    def _finish_report(self, report: Any, error: Exception | None) -> None:
        if error is not None:
            self._add_note(f"The operation failed: {error}", title="Problem")
            return
        level = "Backup" if report.ok else "Problem"
        self._add_note(report.describe(), title=level)
        self._reload()

    def _on_diagnostics(self) -> None:
        if self._busy:
            return
        self._run_background("Reading diagnostics…", self.service.diagnostics, self._finish_diagnostics)

    def _finish_diagnostics(self, lines: Any, error: Exception | None) -> None:
        if error is not None:
            self._add_note(f"Could not read the diagnostics: {error}", title="Problem")
            return
        body = "\n".join(f"{line.label}: {line.value}" for line in lines) or "Nothing to report."
        self._add_note(body, title="Diagnostics")

    def _on_log(self) -> None:
        if self._busy:
            return
        self._run_background("Reading the operation log…", self.service.recent_log, self._finish_log)

    def _finish_log(self, entries: Any, error: Exception | None) -> None:
        if error is not None:
            self._add_note(f"Could not read the log: {error}", title="Problem")
            return
        body = "\n".join(entries) or "No operation has been logged yet."
        self._add_note(body, title="Operation log")

    def _on_memory(self) -> None:
        if self._busy:
            return
        owner = self._owner_var.get()
        self._run_background(
            f"Reading {owner}'s memory…",
            lambda: (self.service.memory_summary(owner), self.service.memory_lines(owner)),
            self._finish_memory,
        )

    def _finish_memory(self, payload: Any, error: Exception | None) -> None:
        if error is not None:
            self._add_note(f"Could not read the memory: {error}", title="Problem")
            return
        summary, lines = payload
        body = "\n".join(line.describe() for line in lines) or "Nothing is remembered yet."
        self._add_note(f"{summary}\n\n{body}", title="Memory")

    def _on_kb_search(self) -> None:
        if self._busy:
            return
        query = self._kb_query.get().strip()
        if not query:
            self._add_note("Type a query to search the knowledge base.", title="Knowledge base")
            return
        self._run_background(
            f"Searching the knowledge base for {query!r}…",
            lambda: self.service.knowledge_search(query),
            self._finish_kb_search,
        )

    def _finish_kb_search(self, hits: Any, error: Exception | None) -> None:
        if error is not None:
            self._add_note(f"Could not search the knowledge base: {error}", title="Problem")
            return
        body = "\n\n".join(hit.describe() for hit in hits) or (
            "Nothing matched -- the knowledge base may be empty, or its index not built."
        )
        self._add_note(body, title="Knowledge base")

    # -- transcript -------------------------------------------------------

    def _add_note(self, text: str, *, title: str = "") -> None:
        theme = self.theme
        width = self._canvas.winfo_width() if hasattr(self, "_canvas") else 0
        pad = 12 if width <= 1 else max(12, (width - _COLUMN_MAX) // 2)
        row = tk.Frame(self._transcript, bg=theme.base)
        row.pack(fill="x", padx=(pad, pad), pady=6)
        if title:
            tk.Label(
                row, text=title, bg=theme.base, fg=theme.text_dim,
                font=self._type.font("2xs", weight="bold"), anchor="w",
            ).pack(fill="x", anchor="w")
        card = tk.Frame(row, bg=theme.surface)
        card.pack(fill="x")
        body = tk.Text(
            card, height=1, bg=theme.surface, fg=theme.foreground, relief="flat",
            highlightthickness=0, borderwidth=0, wrap="word", cursor="arrow",
            font=self._type.mono("2xs") if "\n" in text else self._type.font("2xs"),
        )
        body.insert("1.0", text)
        body.configure(state="disabled")
        body.pack(fill="x", padx=12, pady=10)
        self._fit_height(body)
        self._cards.append(body)
        self.window.after(30, lambda: self._canvas.yview_moveto(1.0))

    def _fit_height(self, body: tk.Text) -> None:
        try:
            body.update_idletasks()
            lines = int(body.index("end-1c").split(".")[0])
        except tk.TclError:
            return
        body.configure(height=max(1, min(lines, _MAX_CARD_LINES)))

    def _set_status(self, text: str) -> None:
        if hasattr(self, "_status_label"):
            self._status_label.configure(text=text)

    # -- worker thread ----------------------------------------------------

    def _run_background(
        self,
        note: str,
        job: Callable[[], Any],
        finish: Callable[[Any, Exception | None], None],
    ) -> None:
        self._busy = True
        self._set_status(note)

        def run() -> None:
            try:
                result = job()
                error: Exception | None = None
            except Exception as exc:
                result, error = None, exc
            self._results.put((finish, result, error))

        threading.Thread(target=run, daemon=True).start()
        self._schedule_pump()

    def _schedule_pump(self) -> None:
        try:
            self._pump_after = self.window.after(60, self._pump)
        except tk.TclError:
            self._pump_after = None

    def _pump(self) -> None:
        self._pump_after = None
        try:
            finish, result, error = self._results.get_nowait()
        except queue.Empty:
            if self._busy:
                self._schedule_pump()
            return
        self._busy = False
        self._set_status("")
        finish(result, error)

    def _close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self._pump_after is not None:
            try:
                self.window.after_cancel(self._pump_after)
            except tk.TclError:
                pass
            self._pump_after = None
        try:
            self.window.destroy()
        except tk.TclError:
            pass
        if self._on_close is not None:
            self._on_close()


__all__ = ["SUBTITLE", "TITLE", "MaintenanceDialog"]
