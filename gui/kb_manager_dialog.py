"""The Knowledge Base Manager window: fill a character's corpus from one place.

Opened from the main window's sidebar (**Knowledge Base Manager**), this window
gives one character's knowledge corpus four ways in, all through the engine in
:mod:`gui.kb_manager`:

1. **Documents** -- *Add files…* and *Add a folder…*, read with the same reader
   the Import Center uses (text, CSV/JSON, digital PDF, Word, optional OCR).
2. **The web** -- type a query and *Search the web*, then fetch the **whole page**
   of any result, or paste a listing URL and *Fetch this page and its links* to
   import a forum or an index in one action. The web path is off until the
   **Search the web** checkbox is ticked, which is the user's explicit approval.
3. **Paste text** -- the window's own composer at the bottom; whatever is typed or
   pasted there is added to the selected character's corpus.
4. **Choose the character** -- the sidebar's *Character* list; every source above
   flows into the character that is selected.

The window is built the way every other window in the project is built -- the same
:class:`~gui.app.Theme` tokens, the shared :class:`~gui.chat_composer.ChatComposer`,
:class:`~gui.sidebar.SidebarColumn`, and :class:`~gui.window_controls.WindowControls`
-- and all logic lives in the service, so the workflow is tested with no display.
Tk is imported lazily with the rest of the window, so the CLI never touches it.
"""

from __future__ import annotations

import queue
import threading
import tkinter as tk
from collections.abc import Callable
from functools import partial
from tkinter import filedialog
from typing import Any

from gui import clipboard, rich_text
from gui.chat_composer import ChatComposer
from gui.dialog_base import ThemedDialog
from gui.kb_manager import KnowledgeBaseManagerService, WebFetchReport
from gui.sidebar import SidebarColumn
from gui.typography import resolve
from gui.window_controls import fit_geometry, install_window_controls

TITLE = "Knowledge Base Manager"

SUBTITLE = (
    "Fill one character's own knowledge corpus. Add documents, search the web and "
    "fetch whole pages, or paste text -- everything you add goes into the character "
    "selected on the left."
)

#: The transcript reads as a centred column, capped like the chat window's.
_COLUMN_MAX = 860

#: The sidebar is this wide; the conversation column takes the rest.
SIDEBAR_WIDTH = 320

DEFAULT_WIDTH = 1180
DEFAULT_HEIGHT = 700
MIN_WIDTH = 940
MIN_HEIGHT = 620


class KnowledgeBaseManagerDialog(ThemedDialog):
    """A window that adds documents, web pages, and pasted text to one corpus."""

    def __init__(
        self,
        parent: tk.Tk | tk.Toplevel,
        service: KnowledgeBaseManagerService,
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
        self._list_type = resolve(
            parent, ui_family=font, mono_family=mono, overrides=list_overrides
        )
        self._on_close = on_close
        self._closed = False
        self._busy = False
        self._pump_after: str | None = None
        self._rows: list[tuple[str, str, tk.Frame, tk.Text, Any]] = []
        # Worker threads hand results back through a queue the UI thread drains.
        self._results: queue.Queue[
            tuple[Callable[[Any, Exception | None], None], Any, Exception | None]
        ] = queue.Queue()

        self._paste_var = tk.StringVar(value="")
        self._query_var = tk.StringVar(value="")
        self._character_var = tk.StringVar(value=service.selected)
        self._web_var = tk.BooleanVar(value=service.web_available())
        self._follow_var = tk.BooleanVar(value=True)
        self._search_results: list[Any] = []

        self.window = tk.Toplevel(parent)
        self.window.title(TITLE)
        self.window.configure(bg=theme.base)
        self.window.geometry(self._initial_geometry())
        self.window.minsize(MIN_WIDTH, MIN_HEIGHT)
        self.window.transient(parent)
        self.window.protocol("WM_DELETE_WINDOW", self._close)

        self._themed_build()

    def _after_build(self) -> None:
        """Refresh the labels a rebuild must redraw (idempotent)."""
        self._refresh_status()

    def _initial_geometry(self) -> str:
        return fit_geometry(self.window, DEFAULT_WIDTH, DEFAULT_HEIGHT, min_height=MIN_HEIGHT)

    # -- layout -----------------------------------------------------------

    def _build(self) -> None:
        self._build_header()
        theme = self.theme
        body = tk.Frame(self.window, bg=theme.base)
        body.pack(fill="both", expand=True)
        body.rowconfigure(0, weight=1)
        body.columnconfigure(0, weight=0, minsize=SIDEBAR_WIDTH)
        body.columnconfigure(1, weight=1)
        self._body = body
        self._build_sidebar(body)
        self._build_conversation(body)

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
            font=self._type.font("2xs"), anchor="w", justify="left", wraplength=780,
        ).pack(anchor="w", pady=(2, 0))
        self._window_controls = install_window_controls(
            self.window, inner, theme=theme, typography=self._type, on_close=self._close
        )
        tk.Frame(self.window, bg=theme.titlebar_edge, height=1).pack(fill="x")

    def _button(self, parent: tk.Misc, text: str, command: Any, *, small: bool = False) -> tk.Button:
        return tk.Button(
            parent, text=text, command=command, relief="flat", borderwidth=0,
            highlightthickness=0, bg=self.theme.surface_raised, fg=self.theme.foreground,
            activebackground=self.theme.interactive_active, activeforeground=self.theme.contrast,
            font=self._type.font("2xs", weight="bold"), padx=10, pady=4 if small else 6,
            cursor="hand2", anchor="w",
        )

    # -- sidebar ----------------------------------------------------------

    def _build_sidebar(self, parent: tk.Frame) -> None:
        column = SidebarColumn(
            parent,
            theme=self.theme,
            typography=self._type,
            width=SIDEBAR_WIDTH,
            scope="knowledge-base-manager",
            state_root=getattr(self.service, "data_root", None),
        )
        self._sidebar_column = column
        self._sidebar_canvas = column.canvas
        self._sidebar_scrollbar = column.scrollbar
        self._sidebar = column.body

        self._build_character_section(column.add("character", "Character", icon="🎭"))
        self._build_documents_section(column.add("documents", "1. Documents", icon="📄"))
        self._build_web_section(column.add("web", "2. Web search and pages", icon="🌐"))
        self._build_status_section(column.add("status", "Corpus", icon="📊"))

    def _build_character_section(self, sidebar: tk.Frame) -> None:
        theme = self.theme
        holder = tk.Frame(sidebar, bg=theme.surface)
        holder.pack(fill="x", padx=10, pady=(4, 0))
        for character in self.service.characters():
            tk.Radiobutton(
                holder, text=character.label, value=character.owner,
                variable=self._character_var, command=partial(self._on_character, character.owner),
                bg=theme.surface, fg=theme.foreground, selectcolor=theme.surface,
                activebackground=theme.surface, activeforeground=theme.foreground,
                font=self._type.font("xs"), anchor="w", padx=6, highlightthickness=0,
                bd=0, cursor="hand2", justify="left", wraplength=SIDEBAR_WIDTH - 44,
            ).pack(fill="x")

    def _build_documents_section(self, sidebar: tk.Frame) -> None:
        holder = tk.Frame(sidebar, bg=self.theme.surface)
        holder.pack(fill="x", padx=12, pady=(4, 0))
        self._button(holder, "＋  Add files…", self._on_add_files, small=True).pack(
            fill="x", pady=(0, 4)
        )
        self._button(holder, "🗀  Add a folder…", self._on_add_folder, small=True).pack(fill="x")

    def _build_web_section(self, sidebar: tk.Frame) -> None:
        theme = self.theme
        holder = tk.Frame(sidebar, bg=theme.surface)
        holder.pack(fill="x", padx=12, pady=(4, 0))

        self._web_check = tk.Checkbutton(
            holder, text="Search the web (approve network access)",
            variable=self._web_var, command=self._on_web_toggle,
            bg=theme.surface, fg=theme.foreground, selectcolor=theme.surface,
            activebackground=theme.surface, activeforeground=theme.foreground,
            font=self._type.font("2xs"), anchor="w", highlightthickness=0, bd=0,
            cursor="hand2", wraplength=SIDEBAR_WIDTH - 40, justify="left",
        )
        self._web_check.pack(fill="x", pady=(0, 6))

        query_row = tk.Frame(holder, bg=theme.surface)
        query_row.pack(fill="x")
        self._query_entry = tk.Entry(
            query_row, textvariable=self._query_var, bg=theme.base, fg=theme.foreground,
            insertbackground=theme.foreground, relief="flat", highlightthickness=0,
            font=self._type.font("xs"),
        )
        self._query_entry.pack(side="left", fill="x", expand=True, ipady=4, padx=(0, 6))
        self._query_entry.bind("<Return>", lambda _e: self._on_search())
        self._button(query_row, "Search", self._on_search, small=True).pack(side="right")

        self._results_list = tk.Listbox(
            holder, bg=theme.base, fg=theme.foreground, relief="flat", highlightthickness=0,
            borderwidth=0, selectbackground=theme.interactive_active, selectforeground=theme.contrast,
            font=self._list_type.font("2xs"), activestyle="none", exportselection=False,
            height=7, width=32,
        )
        self._results_list.pack(fill="x", pady=(6, 0))

        self._button(
            holder, "⇩  Fetch the selected page(s)", self._on_fetch_selected, small=True
        ).pack(fill="x", pady=(6, 0))

        url_row = tk.Frame(holder, bg=theme.surface)
        url_row.pack(fill="x", pady=(6, 0))
        self._url_var = tk.StringVar(value="")
        self._url_entry = tk.Entry(
            url_row, textvariable=self._url_var, bg=theme.base, fg=theme.foreground,
            insertbackground=theme.foreground, relief="flat", highlightthickness=0,
            font=self._type.font("xs"),
        )
        self._url_entry.pack(fill="x", ipady=4)
        self._url_entry.bind("<Return>", lambda _e: self._on_fetch_url())
        self._button(
            holder, "🌐  Fetch this page (and its links)", self._on_fetch_url, small=True
        ).pack(fill="x", pady=(6, 0))
        tk.Checkbutton(
            holder, text="Follow the page's links", variable=self._follow_var,
            bg=theme.surface, fg=theme.text_secondary, selectcolor=theme.surface,
            activebackground=theme.surface, activeforeground=theme.foreground,
            font=self._type.font("2xs"), anchor="w", highlightthickness=0, bd=0, cursor="hand2",
        ).pack(fill="x", pady=(4, 0))
        self._button(
            holder, "🗂  Walk the board (every page)", self._on_walk_board, small=True
        ).pack(fill="x", pady=(6, 0))
        pages_row = tk.Frame(holder, bg=theme.surface)
        pages_row.pack(fill="x", pady=(4, 0))
        tk.Label(
            pages_row, text="board pages:", bg=theme.surface, fg=theme.text_secondary,
            font=self._type.font("2xs"), anchor="w",
        ).pack(side="left")
        self._pages_var = tk.IntVar(value=10)
        tk.Spinbox(
            pages_row, from_=1, to=200, width=5, textvariable=self._pages_var,
            bg=theme.base, fg=theme.foreground, relief="flat", highlightthickness=0,
            buttonbackground=theme.surface_raised, font=self._type.font("2xs"),
        ).pack(side="right")
        match_row = tk.Frame(holder, bg=theme.surface)
        match_row.pack(fill="x", pady=(4, 0))
        tk.Label(
            match_row, text="only links containing:", bg=theme.surface,
            fg=theme.text_secondary, font=self._type.font("2xs"), anchor="w",
        ).pack(side="left")
        self._match_var = tk.StringVar(value="/threads/")
        tk.Entry(
            match_row, textvariable=self._match_var, bg=theme.base, fg=theme.foreground,
            insertbackground=theme.foreground, relief="flat", highlightthickness=0,
            width=12, font=self._type.font("2xs"),
        ).pack(side="right")
        self._descend_var = tk.BooleanVar(value=True)
        tk.Checkbutton(
            holder, text="Also enter sub-forums (a category holds most stories)",
            variable=self._descend_var, bg=theme.surface, fg=theme.text_secondary,
            selectcolor=theme.surface, activebackground=theme.surface,
            activeforeground=theme.foreground, font=self._type.font("2xs"), anchor="w",
            highlightthickness=0, bd=0, cursor="hand2", wraplength=SIDEBAR_WIDTH - 40,
            justify="left",
        ).pack(fill="x", pady=(4, 0))

    def _build_status_section(self, sidebar: tk.Frame) -> None:
        theme = self.theme
        holder = tk.Frame(sidebar, bg=theme.surface)
        holder.pack(fill="x", padx=12, pady=(4, 0))
        self._status_label = tk.Label(
            holder, text="", bg=theme.surface, fg=theme.text_secondary,
            font=self._type.font("2xs"), anchor="w", justify="left",
            wraplength=SIDEBAR_WIDTH - 40,
        )
        self._status_label.pack(fill="x")
        self._button(holder, "⟳  Refresh", self._refresh_status, small=True).pack(
            fill="x", pady=(6, 0)
        )
        self._button(holder, "⚙  Build the search index", self._on_build_index, small=True).pack(
            fill="x", pady=(4, 0)
        )

    # -- conversation column ----------------------------------------------

    def _build_conversation(self, parent: tk.Frame) -> None:
        theme = self.theme
        col = tk.Frame(parent, bg=theme.base)
        col.grid(row=0, column=1, sticky="nsew")

        self._build_composer(col)

        container = tk.Frame(col, bg=theme.base)
        container.pack(fill="both", expand=True)
        self._canvas = tk.Canvas(container, bg=theme.base, highlightthickness=0)
        self._canvas.pack(side="left", fill="both", expand=True)
        self._scrollbar = tk.Scrollbar(container, command=self._canvas.yview)
        self._scrollbar.pack(side="right", fill="y")
        self._canvas.configure(yscrollcommand=self._scrollbar.set)
        self._transcript = tk.Frame(self._canvas, bg=theme.base)
        self._window = self._canvas.create_window((0, 0), window=self._transcript, anchor="nw")
        self._transcript.bind("<Configure>", self._on_transcript_configure)
        self._canvas.bind("<Configure>", self._on_canvas_configure)
        for widget in (self._canvas, self._transcript):
            widget.bind("<MouseWheel>", self._on_mousewheel)
            widget.bind("<Button-4>", self._on_mousewheel)
            widget.bind("<Button-5>", self._on_mousewheel)
        self._add_message("note", self._welcome())

    def _welcome(self) -> str:
        character = self.service.selected_character()
        return (
            f"Adding to {character.name}'s corpus. Use the panel on the left to add "
            "documents, search the web, or paste text below. Every source you add "
            "goes into this character's own knowledge base."
        )

    def _build_composer(self, parent: tk.Frame) -> None:
        composer = ChatComposer(
            parent, theme=self.theme, typography=self._type,
            placeholder="Paste or type text to add to the selected character's corpus…",
            on_submit=self._on_paste, outer_bg=self.theme.base, with_attach=False,
            variable=self._paste_var, max_lines=8,
        )
        composer.wrap.pack(side="bottom", fill="x", padx=self._column_pad(), pady=(8, 12))
        self._composer = composer
        self._entry = composer.entry
        composer.focus()

    # -- geometry helpers --------------------------------------------------

    def _column_pad(self) -> int:
        try:
            width = self._transcript.winfo_width() or self._canvas.winfo_width()
        except (tk.TclError, AttributeError):
            width = _COLUMN_MAX
        if width <= _COLUMN_MAX:
            return 16
        return max(16, (width - _COLUMN_MAX) // 2)

    # -- transcript -------------------------------------------------------

    def _add_message(self, role: str, text: str, *, title: str = "", mono_body: bool = False) -> tk.Text:
        theme = self.theme
        pad = self._column_pad()
        row = tk.Frame(self._transcript, bg=theme.base)
        row.pack(fill="x", padx=(pad, pad), pady=6)

        head = tk.Frame(row, bg=theme.base)
        head.pack(fill="x", pady=(0, 2))
        label = title or {
            "user": "You",
            "note": "Note",
            "source": "Source",
            "result": "Result",
        }.get(role, self.service.selected_character().name)
        tk.Label(
            head, text=label, bg=theme.base, fg=theme.text_dim,
            font=self._type.font("xs", weight="bold"), anchor="w",
        ).pack(side="left")

        widget = tk.Text(
            row, height=1, wrap="word", relief="flat", borderwidth=0, highlightthickness=0,
            bg=theme.base, fg=theme.foreground,
            font=self._type.mono("code") if mono_body else self._type.font("body"),
            padx=0, pady=0, cursor="xterm", takefocus=1,
        )
        rich_text.apply_body_spacing(widget, self._type)
        if mono_body:
            widget.insert("1.0", text)
        else:
            self._insert_rich(widget, text)
        widget.pack(fill="x")
        self._rows.append((role, text, row, widget, None))
        self._make_read_only(widget)
        for wheel in ("<MouseWheel>", "<Button-4>", "<Button-5>"):
            widget.bind(wheel, self._on_bubble_wheel, add="+")
        self._fit_height(widget)
        self._scroll_to_bottom()
        return widget

    def _insert_rich(self, widget: tk.Text, text: str) -> None:
        rich_text.insert_markdown(widget, text, theme=self.theme, typography=self._type)

    def _make_read_only(self, widget: tk.Text) -> None:
        """Read-only but still selectable and copyable, like a chat message."""
        clipboard.make_read_only(widget)

    def _fit_height(self, widget: tk.Text) -> None:
        try:
            widget.update_idletasks()
            lines = int(widget.index("end-1c").split(".")[0])
            widget.configure(height=max(1, lines))
        except (tk.TclError, ValueError):
            pass

    def _on_transcript_configure(self, _event: tk.Event) -> None:
        try:
            self._canvas.configure(scrollregion=self._canvas.bbox("all"))
        except tk.TclError:
            pass

    def _on_canvas_configure(self, event: tk.Event) -> None:
        try:
            self._canvas.itemconfigure(self._window, width=event.width)
        except tk.TclError:
            pass

    def _scroll_to_bottom(self) -> None:
        try:
            self._canvas.update_idletasks()
            self._canvas.yview_moveto(1.0)
        except tk.TclError:
            pass

    def _on_bubble_wheel(self, event: tk.Event) -> str:
        delta = -1 if getattr(event, "delta", 0) > 0 or getattr(event, "num", 0) == 4 else 1
        try:
            self._canvas.yview_scroll(delta, "units")
        except tk.TclError:
            pass
        return "break"

    def _on_mousewheel(self, event: tk.Event) -> str:
        return self._on_bubble_wheel(event)

    # -- character ---------------------------------------------------------

    def _on_character(self, owner: str) -> None:
        try:
            character = self.service.set_character(owner)
        except KeyError as error:
            self._add_message("note", str(error))
            return
        self._add_message("note", f"Now adding to {character.name}'s corpus.")
        self._refresh_status()

    # -- documents ---------------------------------------------------------

    def _on_add_files(self) -> None:
        if self._busy:
            return
        paths = filedialog.askopenfilenames(parent=self.window, title="Add documents")
        if not paths:
            return
        self._add_message("user", "Add files: " + ", ".join(paths))
        self._start_work(
            lambda: self.service.import_files(paths),
            lambda report, error: self._render_import(report, error, "files"),
        )

    def _on_add_folder(self) -> None:
        if self._busy:
            return
        folder = filedialog.askdirectory(parent=self.window, title="Add a folder")
        if not folder:
            return
        self._add_message("user", f"Add folder: {folder}")
        self._start_work(
            lambda: self.service.import_folder(folder, recursive=True),
            lambda report, error: self._render_import(report, error, "folder"),
        )

    def _render_import(self, report: Any, error: Exception | None, what: str) -> None:
        if error is not None:
            self._add_message("note", f"Could not add the {what}: {error}")
            return
        self._add_message("source", f"{report.summary()}")
        for outcome in getattr(report, "failures", [])[:10]:
            self._add_message("note", f"! {outcome.source}: {outcome.message}")
        self._refresh_status()

    # -- web ---------------------------------------------------------------

    def _on_web_toggle(self) -> None:
        self.service.set_web_approval(bool(self._web_var.get()))
        state = "on" if self.service.web_available() else "off"
        self._add_message("note", f"Web access is now {state}.")

    def _on_search(self) -> None:
        if self._busy:
            return
        query = self._query_var.get().strip()
        if not query:
            return
        self._add_message("user", f"Search the web for: {query}")
        self._start_work(lambda: self.service.search_web(query), self._render_search)

    def _render_search(self, report: WebFetchReport, error: Exception | None) -> None:
        if error is not None:
            self._add_message("note", f"Search failed: {error}")
            return
        self._results_list.delete(0, "end")
        self._search_results = list(report.results)
        if not report.allowed:
            self._add_message("note", report.reason)
            return
        if not report.results:
            self._add_message("note", "No results were returned.")
            return
        for result in report.results:
            self._results_list.insert("end", f"{result.title} — {result.url}")
        self._add_message(
            "result",
            f"{len(report.results)} result(s). Select one or more, then **Fetch the "
            "selected page(s)** to import the whole page.",
        )

    def _on_walk_board(self) -> None:
        if self._busy:
            return
        url = self._url_var.get().strip()
        if not url:
            self._add_message("note", "Type a board URL to walk first.")
            return
        try:
            pages = max(1, int(self._pages_var.get()))
        except (tk.TclError, ValueError):
            pages = 10
        self._add_message("user", f"Walk the board at {url} (up to {pages} page(s))")
        match = self._match_var.get().strip()
        descend = bool(self._descend_var.get())
        self._start_work(
            lambda: self.service.walk_board(
                url, max_pages=pages, match=match or None, descend=descend
            ),
            lambda report, error: self._render_board(report, error),
        )

    def _render_board(self, report: WebFetchReport, error: Exception | None) -> None:
        if error is not None:
            self._add_message("note", f"Could not walk the board: {error}")
            return
        if not report.allowed:
            self._add_message("note", report.reason)
            return
        self._results_list.delete(0, "end")
        self._search_results = list(report.results)
        for result in report.results:
            self._results_list.insert("end", result.url)
        self._add_message(
            "result",
            f"{report.summary()} Select the stories you want, then **Fetch the "
            "selected page(s)** to import them.",
        )

    def _on_fetch_selected(self) -> None:
        if self._busy:
            return
        selection = self._results_list.curselection()
        urls = [self._search_results[i].url for i in selection if i < len(self._search_results)]
        if not urls:
            self._add_message("note", "Select one or more results to fetch first.")
            return
        self._add_message("user", "Fetch: " + ", ".join(urls))
        self._start_work(
            lambda: self.service.import_urls(urls),
            lambda report, error: self._render_web(report, error),
        )

    def _on_fetch_url(self) -> None:
        if self._busy:
            return
        url = self._url_var.get().strip()
        if not url:
            self._add_message("note", "Type a page URL to fetch first.")
            return
        follow = bool(self._follow_var.get())
        self._add_message("user", f"Fetch {url}" + (" and its links" if follow else ""))
        self._start_work(
            lambda: self.service.fetch_and_import(url, follow_links=follow),
            lambda report, error: self._render_web(report, error),
        )

    def _render_web(self, report: WebFetchReport, error: Exception | None) -> None:
        if error is not None:
            self._add_message("note", f"Could not fetch: {error}")
            return
        if not report.allowed:
            self._add_message("note", report.reason)
            return
        self._add_message("source", report.summary())
        for page in report.pages[:25]:
            mark = "✓" if page.ok else "✗"
            detail = f"{page.chunks} chunk(s)" if page.ok else (page.message or "failed")
            self._add_message("note", f"{mark} {page.url} — {detail}", mono_body=True)
        self._refresh_status()

    # -- paste -------------------------------------------------------------

    def _on_paste(self) -> None:
        if self._busy:
            return
        text = self._composer.text().strip()
        if not text:
            return
        self._composer.clear()
        preview = text if len(text) <= 200 else text[:200] + "…"
        self._add_message("user", preview)
        self._start_work(
            lambda: self.service.import_text(text, title=_first_line(text)),
            lambda report, error: self._render_import(report, error, "text"),
        )

    # -- status ------------------------------------------------------------

    def _on_build_index(self) -> None:
        if self._busy:
            return
        self._start_work(
            lambda: self.service.build_index(),
            lambda result, error: self._render_index(result, error),
        )

    def _render_index(self, result: Any, error: Exception | None) -> None:
        if error is not None:
            self._add_message("note", f"Could not build the index: {error}")
            return
        count = (result or {}).get("indexed", 0)
        self._add_message("note", f"Index built for {count} chunk(s).")
        self._refresh_status()

    def _refresh_status(self) -> None:
        try:
            self._status_label.configure(text=self.service.describe())
        except (AttributeError, tk.TclError):
            pass

    # -- background work --------------------------------------------------

    def _start_work(
        self, job: Callable[[], Any], finish: Callable[[Any, Exception | None], None]
    ) -> None:
        self._busy = True
        self._set_send_enabled(False)

        def run() -> None:
            try:
                self._results.put((finish, job(), None))
            except Exception as error:  # a failed job reports, never crashes the window
                self._results.put((finish, None, error))

        threading.Thread(target=run, daemon=True).start()
        self._schedule_pump()

    def _schedule_pump(self) -> None:
        if self._closed:
            return
        self._pump_after = self.window.after(60, self._pump_results)

    def _pump_results(self) -> None:
        self._pump_after = None
        try:
            finish, result, error = self._results.get_nowait()
        except queue.Empty:
            if self._busy and not self._closed:
                self._schedule_pump()
            return
        self._busy = False
        self._set_send_enabled(True)
        finish(result, error)

    def _set_send_enabled(self, enabled: bool) -> None:
        try:
            self._composer.set_send_enabled(enabled)
        except (AttributeError, tk.TclError):
            pass

    # -- lifecycle --------------------------------------------------------

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
        finally:
            if self._on_close is not None:
                self._on_close()


def _first_line(text: str) -> str:
    """A short title for pasted text: its first non-empty line, capped."""
    for line in text.splitlines():
        stripped = line.strip()
        if stripped:
            return stripped[:80]
    return ""


__all__ = [
    "DEFAULT_HEIGHT",
    "DEFAULT_WIDTH",
    "MIN_HEIGHT",
    "MIN_WIDTH",
    "SIDEBAR_WIDTH",
    "SUBTITLE",
    "TITLE",
    "KnowledgeBaseManagerDialog",
]
