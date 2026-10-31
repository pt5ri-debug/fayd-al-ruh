"""Doc's Studio: read any document, then ask about it, in one window.

Reached from the chat composer's ＋ menu, this page turns a file into something
you can read and question:

* **Open a document** -- text, Markdown, CSV/TSV, JSON/JSONL, Word
  ``.docx``/``.docm``, a PDF (digital *or* scanned), or a plain image.
* **Read it** -- the extracted text is shown in the transcript, and each page
  that has no text layer is rendered to an image and shown in the preview column
  on the right.
* **Ask about it** -- the composer sends a question to the local **Qwen3-VL**
  vision model, which is shown the document's text *and* the page images, so a
  scanned page, a table, or a figure can be read as well as a paragraph.

The page is a thin view over :class:`gui.docs_studio.DocsStudioService`, so the
whole workflow is testable with no display. Tk is imported lazily with the rest
of the window, so the CLI never touches it. Long work (loading a large PDF,
rendering its pages, running the model) happens on a worker thread whose result
is handed back through a queue the UI thread drains, so the window never freezes.

Everything stays honest: the model runs **locally** through
``llama-cpp-python``, every answer is labelled as the local model's own reading
(never a cited extract), and when the model or its runtime is missing the page
says so and offers the real next step instead of pretending.
"""

from __future__ import annotations

import queue
import threading
import tkinter as tk
from collections.abc import Callable
from pathlib import Path
from tkinter import filedialog, ttk
from tkinter import font as tkfont
from typing import Any, Literal

from core.answer_provenance import AnswerProvenance
from core.docs_model import HEADER
from core.docs_studio import SUPPORTED_SUFFIXES, DocsStudioError
from core.voices import detect_language

from gui import markdown as md
from gui import rich_text
from gui.app import RADIUS_LARGE, RoundedBox
from gui.chat_composer import ChatComposer
from gui.docs_studio import DocsStudioService, picker_filetypes
from gui.image_preview import load_scaled
from gui.message_footer import add_provenance_label
from gui.sidebar import SidebarColumn
from gui.typography import resolve
from gui.window_controls import install_window_controls

TITLE = "Doc's Studio"
SUBTITLE = (
    "Open any document — text, Word, PDF, a scanned PDF, or an image — read its "
    "contents here, then ask questions about it."
)
HONESTY_LINE = (
    "Answers come from a local Qwen3-VL vision model running on this machine; "
    "nothing is sent off it. Every answer is the model's own reading, not a cited "
    "extract from verified sources."
)

#: The preview column (and so the page image) is this fraction of the window
#: width, both before a document is opened and after one.
PREVIEW_FRACTION = 0.30

#: Horizontal padding, in pixels: the side padding of every section card, and the
#: extra padding inside the preview card.
SECTION_PAD = 16
PREVIEW_PAD = 10

#: The transcript reads as a centred column, capped like the chat window's.
_COLUMN_MAX = 860

#: A message taller than this scrolls inside its own card rather than the column.
_MAX_BUBBLE_LINES = 40


class DocsDialog:
    """A window that loads documents and answers questions about them."""

    def __init__(
        self,
        parent: tk.Tk | tk.Toplevel,
        service: DocsStudioService,
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
        self._photo: tk.PhotoImage | None = None
        self._preview_source = ""
        self._preview_fit_width = 0
        self._current_key = ""
        self._document_keys: list[str] = []
        self._page_numbers: list[int] = []
        self._current_page_index = 0
        self._row_frames: list[tuple[str, str, tk.Frame, tk.Text, RoundedBox | None]] = []

        # Worker threads hand results back through a queue the UI thread drains on
        # its own clock: calling `after` from a worker thread races with the main
        # loop (Tk is not thread-safe), which is what the rest of the project does.
        self._results: queue.Queue[
            tuple[Callable[[Any, Exception | None], None], Any, Exception | None]
        ] = queue.Queue()
        self._progress: queue.Queue[str] = queue.Queue()

        self._question_var = tk.StringVar(value="")

        self.window = tk.Toplevel(parent)
        self.window.title(TITLE)
        self.window.configure(bg=theme.base)
        self.window.geometry("1240x720")
        self.window.minsize(960, 640)
        self.window.transient(parent)
        self.window.protocol("WM_DELETE_WINDOW", self._close)
        self._build()
        self._refresh_model_note()

    # -- layout -----------------------------------------------------------

    def _build(self) -> None:
        self._build_header()
        theme = self.theme
        body = tk.Frame(self.window, bg=theme.base)
        body.pack(fill="both", expand=True)
        body.rowconfigure(0, weight=1)
        body.columnconfigure(0, weight=0, minsize=280)
        body.columnconfigure(1, weight=1)
        body.columnconfigure(2, weight=0)
        self._body = body
        body.bind("<Configure>", self._on_body_configure)
        self._build_sidebar(body)
        self._build_conversation(body)
        self._build_preview(body)
        self._layout_columns()

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
            font=self._type.font("2xs"), anchor="w", justify="left", wraplength=820,
        ).pack(anchor="w", pady=(2, 0))
        self._spinner = ttk.Progressbar(inner, mode="indeterminate", length=120)
        self._window_controls = install_window_controls(
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
        # A stack of collapsible sections (the shared :class:`~gui.sidebar.SidebarColumn`):
        # the brand and the open-document actions, the document list, the read/save
        # actions, and the model block -- each under its own foldable header.
        column = SidebarColumn(
            parent,
            theme=self.theme,
            typography=self._type,
            width=280,
            scope="docs-studio",
        )
        self._sidebar_column = column
        self._sidebar_canvas = column.canvas
        self._sidebar_scrollbar = column.scrollbar
        self._sidebar = column.body

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
            brand, text="Doc's Studio", bg=theme.surface, fg=theme.contrast,
            font=self._type.font("body", weight="bold"),
        ).pack(side="left", padx=(8, 0))

        actions = column.add("open", "Open a document", icon="📂")
        self._open_button = self._button(actions, "Open a document…", self._on_open_document)
        self._open_button.pack(fill="x", padx=12, pady=(4, 4))
        self._open_folder_button = self._button(
            actions, "Open a folder…", self._on_open_folder
        )
        self._open_folder_button.pack(fill="x", padx=12, pady=(0, 4))
        self._close_button = self._button(
            actions, "Close this document", self._on_close_document
        )
        self._close_button.pack(fill="x", padx=12, pady=(0, 4))
        self._close_button.configure(state="disabled")

        docs = column.add("documents", "Open documents", icon="📚")
        list_wrap = tk.Frame(docs, bg=theme.surface)
        list_wrap.pack(fill="both", expand=True, padx=12, pady=(4, 6))
        self._document_list = tk.Listbox(
            list_wrap, bg=theme.base, fg=theme.foreground, relief="flat",
            highlightthickness=0, borderwidth=0, selectbackground=theme.interactive_active,
            selectforeground=theme.contrast, font=self._type.font("xs"), activestyle="none",
            exportselection=False, height=6,
        )
        self._document_list.pack(fill="both", expand=True)
        self._document_list.bind("<<ListboxSelect>>", self._on_select_document)

        info = tk.Frame(docs, bg=theme.surface)
        info.pack(fill="x", padx=14, pady=(0, 6))
        self._document_note = tk.Label(
            info, text="No document is open.", bg=theme.surface, fg=theme.text_secondary,
            font=self._type.font("2xs"), anchor="w", justify="left", wraplength=250,
        )
        self._document_note.pack(fill="x")

        text_actions = column.add("text", "Text and reading", icon="📝")
        self._read_button = self._button(
            text_actions, "Read the document (scanned pages)", self._on_read_document
        )
        self._read_button.pack(fill="x", padx=12, pady=(4, 4))
        self._read_button.configure(state="disabled")
        self._save_text_button = self._button(
            text_actions, "Save the extracted text…", self._on_save_text
        )
        self._save_text_button.pack(fill="x", padx=12, pady=(0, 4))
        self._save_text_button.configure(state="disabled")

        model_box = column.add("model", "Model", icon="🧠")
        model_inner = tk.Frame(model_box, bg=theme.surface)
        model_inner.pack(fill="x", padx=14, pady=(4, 14))
        self._model_note = tk.Label(
            model_inner, text="", bg=theme.surface, fg=theme.text_secondary,
            font=self._type.font("2xs"), anchor="w", justify="left", wraplength=250,
        )
        self._model_note.pack(fill="x", pady=(0, 4))
        self._download_button = self._button(
            model_inner, "Download the model…", self._on_download_model
        )
        self._download_button.pack(fill="x")

    # -- conversation -----------------------------------------------------

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
        self._canvas.configure(yscrollcommand=self._scrollbar.set)
        self._transcript = tk.Frame(self._canvas, bg=theme.base)
        self._window = self._canvas.create_window((0, 0), window=self._transcript, anchor="nw")
        self._transcript.bind("<Configure>", self._on_transcript_configure)
        self._canvas.bind("<Configure>", self._on_canvas_configure)
        for widget in (self._canvas, self._transcript):
            widget.bind("<MouseWheel>", self._on_mousewheel)
            widget.bind("<Button-4>", self._on_mousewheel)
            widget.bind("<Button-5>", self._on_mousewheel)
        self._add_message("note", HONESTY_LINE)

    def _build_composer(self, parent: tk.Frame) -> None:
        """The shared rounded composer card, pinned to the bottom like the chat window."""
        composer = ChatComposer(
            parent, theme=self.theme, typography=self._type,
            placeholder="Ask a question about the document…", on_submit=self._on_ask,
            outer_bg=self.theme.base, with_attach=False, variable=self._question_var,
        )
        composer.wrap.pack(side="bottom", fill="x", padx=self._column_pad(), pady=(8, 12))
        self._composer = composer
        self._question = composer.entry
        self._ask_button = self._button(composer.right_tools, "Ask", self._on_ask)
        composer.add_tool(self._ask_button)

    # -- preview ----------------------------------------------------------

    def _build_preview(self, parent: tk.Frame) -> None:
        theme = self.theme
        side = tk.Frame(parent, bg=theme.base)
        side.grid(row=0, column=2, sticky="nsew")
        self._side_column = side

        card = tk.Frame(side, bg=theme.surface)
        card.pack(fill="both", expand=True, padx=(0, SECTION_PAD), pady=(SECTION_PAD, SECTION_PAD))
        head = tk.Frame(card, bg=theme.surface)
        head.pack(fill="x", padx=PREVIEW_PAD, pady=(PREVIEW_PAD, 4))
        tk.Label(
            head, text="PAGE", bg=theme.surface, fg=theme.text_dim,
            font=self._type.font("2xs", weight="bold"),
        ).pack(side="left")
        self._page_label = tk.Label(
            head, text="—", bg=theme.surface, fg=theme.text_secondary,
            font=self._type.font("2xs"),
        )
        self._page_label.pack(side="right")

        self._preview = tk.Label(
            card, bg=theme.base, fg=theme.text_dim, font=self._type.font("2xs"),
            text="Open a document to see its pages here.", wraplength=240, justify="center",
        )
        self._preview.pack(fill="both", expand=True, padx=PREVIEW_PAD, pady=(0, 6))

        nav = tk.Frame(card, bg=theme.surface)
        nav.pack(fill="x", padx=PREVIEW_PAD, pady=(0, PREVIEW_PAD))
        self._prev_button = self._button(nav, "‹ Previous", self._on_previous_page)
        self._prev_button.pack(side="left")
        self._next_button = self._button(nav, "Next ›", self._on_next_page)
        self._next_button.pack(side="right")
        self._prev_button.configure(state="disabled")
        self._next_button.configure(state="disabled")

    def _on_body_configure(self, _event: tk.Event | None = None) -> None:
        self._layout_columns()

    def _layout_columns(self) -> None:
        """Pin the preview column to ``PREVIEW_FRACTION`` of the window width."""
        try:
            width = self.window.winfo_width()
        except tk.TclError:  # the window is closing
            return
        if width <= 1:
            return
        self._body.columnconfigure(
            2,
            minsize=max(160, int(width * PREVIEW_FRACTION) + 2 * SECTION_PAD + 2 * PREVIEW_PAD),
        )
        if self._preview_source:
            self.window.after_idle(self._refit_preview)

    def _refit_preview(self) -> None:
        if not self._preview_source:
            return
        self.window.update_idletasks()
        width = self._preview.winfo_width()
        if width > 1 and width != self._preview_fit_width:
            self._show_preview(self._preview_source)

    def _show_preview(self, path: str) -> None:
        """Show ``path`` fitted to the preview box, and remember the fit width."""
        self._preview_source = path
        width = self._preview.winfo_width()
        if width <= 1:
            width = 240
        photo = load_scaled(path, width)
        if photo is None:
            self._photo = None
            self._preview.configure(image="", text="This page cannot be previewed.")
            return
        self._photo = photo
        self._preview.configure(image=photo, text="")
        self._preview_fit_width = width

    # -- column geometry --------------------------------------------------

    def _column_pad(self) -> int:
        # Called while the composer is built, before the transcript exists, so the
        # widgets may not be there yet; the base pad is the right answer then.
        try:
            width = self._transcript.winfo_width() or self._canvas.winfo_width()
        except (tk.TclError, AttributeError):
            width = _COLUMN_MAX
        if width <= _COLUMN_MAX:
            return 16
        return max(16, (width - _COLUMN_MAX) // 2)

    def _available_column(self) -> int:
        try:
            width = self._canvas.winfo_width()
        except (tk.TclError, AttributeError):
            return _COLUMN_MAX
        if width <= 2:
            return _COLUMN_MAX
        return max(160, min(_COLUMN_MAX, width - 2 * self._column_pad()))

    def _bubble_fonts(self) -> tuple[tkfont.Font, tkfont.Font]:
        return (
            tkfont.Font(font=self._type.font("body")),
            tkfont.Font(font=self._type.mono("code")),
        )

    def _bubble_content_width(self, text: str, *, role: str) -> int:
        normal, mono = self._bubble_fonts()
        widest = 0
        for line in md.iter_styled_lines(text):
            font = mono if line.in_code else normal
            widest = max(widest, font.measure(line.text))
        if role == "user":
            return max(80, min(widest, int(_COLUMN_MAX * 0.84) - 52))
        return max(120, min(widest, self._available_column()))

    def _bubble_text_width(self, text: str, *, role: str) -> int:
        normal, _mono = self._bubble_fonts()
        char_px = max(1, normal.measure("0"))
        content = self._bubble_content_width(text, role=role)
        # Round *down*: a ``Text`` width is whole characters, so rounding up can
        # ask for a few pixels more than the column and force a horizontal scroll.
        return max(1, content // char_px)

    def _bubble_card_width(self, text: str) -> int:
        return self._bubble_content_width(text, role="user") + 52

    # -- transcript -------------------------------------------------------

    def _add_message(
        self, role: str, text: str, *, title: str = "", mono_body: bool = False,
        rich: bool = True, provenance: str = "",
    ) -> tk.Text:
        """Add one message row in the chat window's style and return its widget."""
        theme = self.theme
        pad = self._column_pad()
        row = tk.Frame(self._transcript, bg=theme.base)
        row.pack(fill="x", padx=(pad, pad), pady=6)
        card: RoundedBox | None = None

        if role == "user":
            card = RoundedBox(
                row, bg=theme.user_bubble, fill=theme.user_bubble, outline=theme.user_bubble,
                corner=RADIUS_LARGE, parent_bg=theme.base,
            )
            card.pack(side="right", anchor="e")
            widget = tk.Text(
                card.inner, height=1, wrap="word", relief="flat", borderwidth=0,
                highlightthickness=0, bg=theme.user_bubble, fg=theme.user_bubble_fg,
                font=self._type.font("body"), padx=14, pady=8, cursor="xterm", takefocus=1, width=0,
            )
            widget.insert("1.0", text)
            widget.pack(fill="both", expand=True)
            card.configure(width=self._bubble_card_width(text))
            widget.configure(width=self._bubble_text_width(text, role="user"))
        else:
            head = tk.Frame(row, bg=theme.base)
            head.pack(fill="x", pady=(0, 4))
            if role in ("assistant", "model", "docs"):
                self._avatar(head)
                padx = (6, 0)
            else:
                padx = (0, 0)
            tk.Label(
                head, text=title or ("Doc's Studio" if role != "note" else "Note"),
                bg=theme.base, fg=theme.text_dim, font=self._type.font("xs", weight="bold"),
                anchor="w",
            ).pack(side="left", padx=padx)
            widget = tk.Text(
                row, height=1, wrap="word", relief="flat", borderwidth=0, highlightthickness=0,
                bg=theme.base, fg=theme.foreground,
                font=self._type.mono("code") if mono_body else self._type.font("body"),
                padx=0, pady=0, cursor="xterm", takefocus=1,
            )
            rich_text.apply_body_spacing(widget, self._type)
            if rich and not mono_body:
                self._insert_rich(widget, text)
            else:
                widget.insert("1.0", text)
            widget.pack(anchor="w")
            widget.configure(width=self._bubble_text_width(text, role=role))

        self._row_frames.append((role, text, row, widget, card))
        self._make_read_only(widget)
        for wheel in ("<MouseWheel>", "<Button-4>", "<Button-5>"):
            widget.bind(wheel, self._on_bubble_wheel, add="+")
        self._fit_height(widget)
        if provenance.strip():
            add_provenance_label(row, provenance, theme=theme, typography=self._type)
        self._scroll_to_bottom()
        return widget

    def _avatar(self, parent: tk.Misc) -> None:
        badge = tk.Canvas(parent, width=24, height=24, bg=self.theme.base, highlightthickness=0)
        badge.create_oval(1, 1, 23, 23, fill=self.theme.primary, outline="")
        badge.create_text(
            12, 13, text="ف", fill=self.theme.primary_foreground,
            font=self._type.font("sm", weight="bold"),
        )
        badge.pack(side="left")

    def _insert_rich(self, widget: tk.Text, text: str) -> None:
        rich_text.insert_markdown(widget, text, theme=self.theme, typography=self._type)

    def _provenance_line(self, result: Any) -> str:
        """Which vision model read the document, and which pages it used.

        The Doc's Studio answers with its own local vision model, so the model line
        names it from the service's own honest status (never a guess) and the extra
        line names the pages the answer actually read -- the studio's equivalent of
        the chat windows' citations. An unavailable model says so honestly rather
        than naming weights that did not run.
        """
        status = self.service.model_status()
        available = bool(getattr(status, "available", False))
        model = status.model if available else ""
        pages = tuple(
            getattr(result, "pages_used", ()) or getattr(result, "pages_read", ()) or ()
        )
        extra = ("Pages read: " + ", ".join(str(page) for page in pages),) if pages else ()
        label = f"Doc's Studio: {model} (local vision model, in-process)." if available else ""
        return AnswerProvenance(
            model_label=label,
            mouth="docs",
            model_key=model,
            language=detect_language(getattr(result, "question", "")),
            retrieval_only=not available,
            extra=extra,
        ).describe()

    def _fit_height(self, widget: tk.Text) -> None:
        widget.update_idletasks()
        measured = widget.count("1.0", "end", "displaylines")
        lines = int(measured[0]) if measured else 1
        widget.configure(height=min(max(lines, 1), _MAX_BUBBLE_LINES))

    @staticmethod
    def _make_read_only(widget: tk.Text) -> None:
        """Read-only but still selectable and copyable, like a chat message."""

        def block(event: tk.Event) -> str:
            if int(event.state) & 0x0004 or int(event.state) & 0x0008:
                return ""
            if event.keysym in {"Left", "Right", "Up", "Down", "Home", "End", "Prior", "Next"}:
                return ""
            return "break"

        for sequence in ("<Key>", "<<Paste>>", "<<Cut>>", "<<Clear>>"):
            widget.bind(sequence, block)

    def _on_bubble_wheel(self, event: tk.Event) -> str:
        widget = event.widget
        spans = widget.yview() if isinstance(widget, tk.Text) else (0.0, 1.0)
        if spans[0] <= 0.0 and spans[1] >= 1.0:
            self._on_mousewheel(event)
            return "break"
        return ""

    def _on_transcript_configure(self, _event: tk.Event) -> None:
        self._canvas.configure(scrollregion=self._canvas.bbox("all"))
        self._toggle_scrollbar()

    def _on_canvas_configure(self, event: tk.Event) -> None:
        self._canvas.itemconfigure(self._window, width=event.width)
        self._reflow()

    def _reflow(self) -> None:
        pad = self._column_pad()
        for _role, _text, row, _widget, _card in self._row_frames:
            row.pack_configure(padx=(pad, pad))
        self._remetric_rows()

    def _remetric_rows(self) -> None:
        for role, text, _row, widget, card in self._row_frames:
            try:
                if card is not None:
                    card.configure(width=self._bubble_card_width(text))
                widget.configure(width=self._bubble_text_width(text, role=role))
            except tk.TclError:
                continue
            self._fit_height(widget)

    def _toggle_scrollbar(self) -> None:
        first, last = self._canvas.yview()
        if first <= 0.0 and last >= 1.0:
            self._scrollbar.pack_forget()
        else:
            self._scrollbar.pack(side="right", fill="y")

    def _on_mousewheel(self, event: tk.Event) -> None:
        delta = -1 * (event.delta // 120) if getattr(event, "delta", 0) else (
            -1 if getattr(event, "num", 0) == 4 else 1
        )
        self._canvas.yview_scroll(delta, "units")
        self._toggle_scrollbar()

    def _scroll_to_bottom(self) -> None:
        self._canvas.update_idletasks()
        self._canvas.yview_moveto(1.0)
        self._toggle_scrollbar()

    # -- background work ---------------------------------------------------

    def _run_background(
        self,
        message: str,
        job: Callable[[], Any],
        finish: Callable[[Any, Exception | None], None],
    ) -> None:
        """Run ``job`` on a worker thread, then hand its result back on the UI thread."""
        self._busy = True
        self._set_status(message)
        self._set_busy_buttons(False)
        self._spinner.pack(side="right", padx=8)
        self._spinner.start(12)

        def run() -> None:
            try:
                result = job()
                self._results.put((finish, result, None))
            except Exception as error:
                self._results.put((finish, None, error))

        threading.Thread(target=run, daemon=True).start()
        self.window.after(60, self._pump_results)

    def _pump_results(self) -> None:
        """Drain finished jobs and progress lines on the UI thread."""
        self._drain_progress()
        try:
            finish, result, error = self._results.get_nowait()
        except queue.Empty:
            if self._busy and not self._closed:
                self.window.after(60, self._pump_results)
            return
        self._busy = False
        self._spinner.stop()
        self._spinner.pack_forget()
        self._set_busy_buttons(True)
        finish(result, error)

    def _drain_progress(self) -> None:
        latest = ""
        while True:
            try:
                latest = self._progress.get_nowait()
            except queue.Empty:
                break
        if latest:
            self._set_status(latest)

    def _set_busy_buttons(self, enabled: bool) -> None:
        state: Literal["normal", "disabled"] = "normal" if enabled else "disabled"
        for button in (self._ask_button, self._open_button, self._open_folder_button):
            button.configure(state=state)
        if not enabled:
            self._download_button.configure(state="disabled")
            self._close_button.configure(state="disabled")
            self._save_text_button.configure(state="disabled")
            self._read_button.configure(state="disabled")
        else:
            self._download_button.configure(state="normal")
            self._sync_document_buttons()

    def _set_status(self, message: str, *, error: bool = False) -> None:
        self._document_note.configure(
            text=message, fg=self.theme.danger if error else self.theme.text_secondary
        )

    # -- documents ---------------------------------------------------------

    def _on_open_document(self) -> None:
        path = filedialog.askopenfilename(
            parent=self.window, title="Open a document", filetypes=list(picker_filetypes())
        )
        if path:
            self._load(path)

    def _on_open_folder(self) -> None:
        folder = filedialog.askdirectory(parent=self.window, title="Open a folder of documents")
        if not folder:
            return
        opened = 0
        errors: list[str] = []
        for child in sorted(Path(folder).iterdir()):
            if child.is_file() and child.suffix.lower() in SUPPORTED_SUFFIXES:
                try:
                    self.service.open_document(child)
                    opened += 1
                except DocsStudioError as error:
                    errors.append(f"{child.name}: {error}")
        self._refresh_documents()
        self._add_message(
            "note",
            f"Opened {opened} document(s) from {folder}."
            + (f"\nSkipped {len(errors)}: " + "; ".join(errors[:4]) if errors else ""),
            title="Folder",
        )

    def _load(self, path: str) -> None:
        self._add_message("note", f"Opening **{Path(path).name}**…", title="Working")

        def job() -> Any:
            return self.service.open_document(path)

        def finish(result: Any, error: Exception | None) -> None:
            if error is not None:
                self._add_message("note", f"Could not open the document: {error}", title="Problem")
                return
            self._refresh_documents()
            self._select_document(result.key)
            self._describe_document(result)

        self._run_background(f"Reading {Path(path).name}…", job, finish)

    def _describe_document(self, opened: Any) -> None:
        document = opened.document
        lines = [f"**{document.filename}** — {document.summary()}"]
        if document.notes:
            lines.append("Notes: " + "; ".join(document.notes[:5]))
        if document.vision_pages:
            pages = ", ".join(str(number) for number in document.vision_pages[:12])
            lines.append(
                f"Pages read as images (no text layer): {pages}"
                + ("…" if len(document.vision_pages) > 12 else "")
            )
        if document.text.strip():
            preview = document.text[:1500]
            if len(document.text) > len(preview):
                preview += "\n\n… (the rest is available to the model when you ask)"
            lines.append(preview)
        elif not document.vision_pages:
            lines.append("This document has no readable content.")
        self._add_message("note", "\n\n".join(lines), title="Document")
        self._show_page(0)
        self._maybe_auto_read(opened)

    def _maybe_auto_read(self, opened: Any) -> None:
        """Read a document's image pages with the vision model, when it is ready.

        A scanned document has no text layer, so without this the window would show
        only a page image and a "ask a question" note -- its *contents* would never
        appear. When the model is installed the pages are transcribed automatically;
        when it is not, the real next step is stated instead of leaving the user
        with a blank-looking document.
        """
        if not opened.document.vision_pages:
            return
        status = self.service.model_status()
        if not status.available:
            self._add_message(
                "note",
                "This document has no text layer, so its content is read from the page "
                f"image(s) by the vision model — which is not ready yet. {status.hint or status.summary()}",
                title="Reading needs the model",
            )
            return
        self._read_document(opened.key)

    def _read_document(self, key: str) -> None:
        """Transcribe the current document's image pages and show the contents."""
        if not key:
            return

        def job() -> Any:
            return self.service.read(key, progress=lambda message: self._progress.put(message))

        def finish(result: Any, error: Exception | None) -> None:
            if error is not None:
                self._add_message("note", f"Could not read the document: {error}", title="Problem")
                return
            if not result.ok:
                self._add_message("note", result.message, title="Cannot read")
                return
            if not result.text:
                self._add_message(
                    "note", result.message or "The model returned no readable text.",
                    title="No text found",
                )
                return
            self._add_message(
                "docs", f"{HEADER}\n\n{result.text}", title="Doc's Studio — contents",
                provenance=self._provenance_line(result),
            )

        self._run_background("Reading the document…", job, finish)

    def _refresh_documents(self) -> None:
        self._document_list.delete(0, "end")
        self._document_keys = []
        for opened in self.service.documents():
            self._document_keys.append(opened.key)
            self._document_list.insert("end", opened.filename)
        self._sync_document_buttons()

    def _sync_document_buttons(self) -> None:
        has_document = bool(self._current_key)
        state: Literal["normal", "disabled"] = "normal" if has_document else "disabled"
        self._close_button.configure(state=state)
        self._save_text_button.configure(state=state)
        self._read_button.configure(state=state)

    def _on_select_document(self, _event: tk.Event | None = None) -> None:
        selection = self._document_list.curselection()
        if not selection:
            return
        index = int(selection[0])
        if 0 <= index < len(self._document_keys):
            self._select_document(self._document_keys[index], show=True)

    def _select_document(self, key: str, *, show: bool = False) -> None:
        self._current_key = key
        opened = self.service.get(key)
        if opened is None:
            return
        self._sync_document_buttons()
        self._page_numbers = list(opened.document.vision_pages) or [
            page.number for page in opened.document.pages
        ]
        self._current_page_index = 0
        if show:
            self._describe_document(opened)
        else:
            self._show_page(0)
        self._update_page_controls()

    def _on_close_document(self) -> None:
        if not self._current_key:
            return
        self.service.close_document(self._current_key)
        self._current_key = ""
        self._page_numbers = []
        self._photo = None
        self._preview.configure(image="", text="Open a document to see its pages here.")
        self._page_label.configure(text="—")
        self._refresh_documents()
        self._set_status("No document is open.")

    def _on_read_document(self) -> None:
        """Read the current document's scanned pages on demand."""
        if not self._current_key:
            return
        opened = self.service.get(self._current_key)
        if opened is not None and not opened.document.vision_pages:
            self._add_message(
                "note",
                "This document has a text layer, so its contents are already shown as "
                "text above — there is nothing extra for the vision model to read.",
                title="Nothing to read",
            )
            return
        self._read_document(self._current_key)

    def _on_save_text(self) -> None:
        if not self._current_key:
            return
        opened = self.service.get(self._current_key)
        default = f"{Path(opened.filename).stem}.txt" if opened else "document.txt"
        target = filedialog.asksaveasfilename(
            parent=self.window, title="Save the extracted text", initialfile=default,
            defaultextension=".txt", filetypes=[("Text", "*.txt"), ("All files", "*.*")],
        )
        if not target:
            return
        try:
            path = self.service.save_text(self._current_key, target)
        except DocsStudioError as error:
            self._add_message("note", f"Could not save the text: {error}", title="Problem")
            return
        self._add_message("note", f"Saved the extracted text to `{path}`.", title="Saved")

    # -- pages -------------------------------------------------------------

    def _show_page(self, index: int) -> None:
        if not self._page_numbers:
            return
        index = max(0, min(index, len(self._page_numbers) - 1))
        self._current_page_index = index
        number = self._page_numbers[index]
        opened = self.service.get(self._current_key)
        if opened is None:
            return
        page = opened.document.page(number)
        if page is None or not page.image_path:
            self._photo = None
            self._preview_source = ""
            self._preview.configure(
                image="",
                text=f"Page {number} has a text layer; its text is shown in the transcript.",
            )
        else:
            self._show_preview(page.image_path)
        self._page_label.configure(text=f"{index + 1} / {len(self._page_numbers)}  (page {number})")
        self._update_page_controls()

    def _update_page_controls(self) -> None:
        count = len(self._page_numbers)
        self._prev_button.configure(
            state="normal" if count and self._current_page_index > 0 else "disabled"
        )
        self._next_button.configure(
            state="normal" if count and self._current_page_index < count - 1 else "disabled"
        )

    def _on_previous_page(self) -> None:
        self._show_page(self._current_page_index - 1)

    def _on_next_page(self) -> None:
        self._show_page(self._current_page_index + 1)

    # -- asking ------------------------------------------------------------

    def _on_ask(self) -> None:
        question = self._question_var.get().strip()
        if not question:
            return
        if not self._current_key:
            self._add_message(
                "note", "Open a document first, then ask a question about it.", title="No document"
            )
            return
        self._composer.clear()
        self._add_message("user", question)

        key = self._current_key

        def job() -> Any:
            return self.service.ask(key, question)

        def finish(result: Any, error: Exception | None) -> None:
            if error is not None:
                self._add_message("note", f"Could not answer: {error}", title="Problem")
                return
            if not result.ok:
                self._add_message("note", result.message, title="Cannot answer")
                return
            self._add_message(
                "docs", f"{HEADER}\n\n{result.text}", title="Doc's Studio",
                provenance=self._provenance_line(result),
            )

        self._run_background("Reading the document…", job, finish)

    # -- model -------------------------------------------------------------

    def _refresh_model_note(self) -> None:
        status = self.service.model_status()
        self._model_note.configure(text=status.summary())
        self._download_button.configure(
            state="disabled" if status.available else "normal"
        )

    def _on_download_model(self) -> None:
        names = self.service.model_names()
        if not names:
            return
        chosen = names[0]
        self._add_message(
            "note",
            "Downloading the vision model is a one-time, human-approved action "
            f"(~3.4 GB). Run `python -m cli docs download {chosen} --approve` in a "
            "terminal, or drop the `.gguf` and its `mmproj-*.gguf` into the model "
            "folder.",
            title="Download the model",
        )

    # -- lifecycle ---------------------------------------------------------

    def _close(self) -> None:
        self._closed = True
        if self._on_close is not None:
            self._on_close()
        try:
            self.window.destroy()
        except tk.TclError:
            pass


__all__ = ["HONESTY_LINE", "PREVIEW_FRACTION", "SUBTITLE", "TITLE", "DocsDialog"]
