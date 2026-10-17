"""The Presentation Studio window: discuss, outline, review slides, export a .pptx.

This page turns a conversation into a real PowerPoint file, in the order a deck is
actually made:

* **Discuss.** Talk the subject over with the project's *existing* assistant --
  the same grounded assistant and knowledge base the chat window uses. Every
  answer is shown with its citations, exactly as the chat would.
* **Outline.** Ask the assistant for an outline, or write one yourself; the
  outline is numbered and can be revised until you approve a revision.
* **Draft the slides.** The assistant writes each section's slide content, one
  reviewable batch at a time.
* **Review every slide.** Each slide is individually numbered; you can edit its
  title, body, table, equations, notes, reorder it, remove it, and approve it.
  An edit after approval is never silent -- it reopens the content and the export
  gate re-arms -- and every change is recorded as a revision you can restore.
* **Approve the content**, then **export** a complete ``.pptx`` containing the
  approved slides in exactly the approved order.

The window is a thin view over :class:`gui.presentation.PresentationService`, so
all of the workflow is testable with no display. Tk is imported lazily with the
rest of the window, so the CLI never touches it.
"""

from __future__ import annotations

import queue
import threading
from collections.abc import Callable
from typing import Any, Literal

import tkinter as tk
from tkinter import filedialog, font as tkfont, simpledialog, ttk

from core.presentation import (
    CONTENT_TYPES,
    STATE_CONTENT_APPROVED,
    STATE_EXPORTED,
    STATE_LABELS,
)
from gui import markdown as md
from gui import rich_text
from gui.app import RADIUS_LARGE, RoundedBox
from gui.typography import resolve

#: The transcript reads as a centred column, capped like the chat window's.
_COLUMN_MAX = 860

#: A message taller than this scrolls inside its own card rather than the column.
_MAX_BUBBLE_LINES = 40

TITLE = "Presentation Studio"
SUBTITLE = (
    "Build a real PowerPoint deck from a conversation: discuss the subject, agree an "
    "outline, draft and review every numbered slide, then export a .pptx that contains "
    "exactly the slides you approved, in exactly the order you approved them."
)
HONESTY_LINE = (
    "Discussion, outline, and slide text come from the project's own assistant — the same "
    "grounded assistant and knowledge base the chat window uses, with its citations. "
    "Nothing is exported until you approve the content, and the .pptx is written locally "
    "with no extra package."
)

#: The workflow, in order, with the sidebar button that advances it.
STEPS: tuple[tuple[str, str], ...] = (
    ("discuss", "1. Discuss the subject"),
    ("outline", "2. Draft the outline"),
    ("review-outline", "3. Review the outline"),
    ("approve-outline", "4. Approve the outline"),
    ("slides", "5. Draft the slides"),
    ("review-slides", "6. Review the slides"),
    ("approve-content", "7. Approve the content"),
    ("export", "8. Export the .pptx"),
)


class PresentationDialog:
    """A window that builds a PowerPoint deck from a discussion, slide by slide."""

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
        self._type = resolve(parent, ui_family=font, mono_family=mono)
        self._on_close = on_close
        self._busy = False
        self._closed = False
        self._project_id = ""
        self._selected_slide = ""
        self._outline_revision = 0
        # Worker threads hand results back through a queue the UI thread drains on
        # its own clock: calling `after` from a worker thread races with the main
        # loop (Tk is not thread-safe), which is what the rest of the project does.
        self._results: queue.Queue[
            tuple[Callable[[Any, Exception | None], None], Any, Exception | None]
        ] = queue.Queue()
        self._progress: queue.Queue[str] = queue.Queue()
        # Each entry is (role, text, row frame, message widget, its rounded card
        # or ``None``); the text is kept so a resize can re-measure the row's
        # width instead of leaving it sized for the old column.
        self._row_frames: list[tuple[str, str, tk.Frame, tk.Text, RoundedBox | None]] = []
        self._project_ids: list[str] = []
        self._slide_ids: list[str] = []
        # Long steps (drafting a deck) report progress through this queue so the
        # status line shows the work rather than looking frozen.
        if hasattr(service, "set_progress"):
            service.set_progress(self._progress.put)
        self._question_var = tk.StringVar(value="")
        self._title_var = tk.StringVar(value="")
        self._slide_title_var = tk.StringVar(value="")
        self._slide_type_var = tk.StringVar(value="bullets")

        self.window = tk.Toplevel(parent)
        self.window.title(TITLE)
        self.window.configure(bg=theme.base)
        self.window.geometry("1200x680")
        self.window.minsize(940, 620)
        self.window.transient(parent)
        self.window.protocol("WM_DELETE_WINDOW", self._close)
        self._build()
        self._open_first_project()

    # -- layout -----------------------------------------------------------

    def _build(self) -> None:
        theme = self.theme
        self._build_header()
        body = tk.Frame(self.window, bg=theme.base)
        body.pack(fill="both", expand=True)
        body.rowconfigure(0, weight=1)
        body.columnconfigure(0, weight=0, minsize=300)
        body.columnconfigure(1, weight=1)
        self._build_sidebar(body)
        self._build_conversation(body)

    def _build_header(self) -> None:
        theme = self.theme
        bar = tk.Frame(self.window, bg=theme.base)
        bar.pack(fill="x")
        inner = tk.Frame(bar, bg=theme.base)
        inner.pack(fill="x", padx=16, pady=(12, 8))
        titles = tk.Frame(inner, bg=theme.base)
        titles.pack(side="left", anchor="w")
        tk.Label(
            titles, text=TITLE, bg=theme.base, fg=theme.contrast,
            font=self._type.font("heading3", weight="bold"), anchor="w",
        ).pack(anchor="w")
        tk.Label(
            titles, text=SUBTITLE, bg=theme.base, fg=theme.text_secondary,
            font=self._type.font("2xs"), anchor="w", justify="left", wraplength=780,
        ).pack(anchor="w", pady=(2, 0))
        self._spinner = ttk.Progressbar(inner, mode="indeterminate", length=120)
        tk.Frame(self.window, bg=theme.border_subtle, height=1).pack(fill="x")

    def _section_label(self, parent: tk.Misc, text: str) -> None:
        tk.Label(
            parent, text=text.upper(), bg=self.theme.surface, fg=self.theme.text_dim,
            font=self._type.font("2xs", weight="bold"), anchor="w",
        ).pack(fill="x", padx=14, pady=(10, 2))

    def _button(self, parent: tk.Misc, text: str, command: Any) -> tk.Button:
        return tk.Button(
            parent, text=text, command=command, relief="flat", borderwidth=0,
            highlightthickness=0, bg=self.theme.surface_raised, fg=self.theme.foreground,
            activebackground=self.theme.interactive_active, activeforeground=self.theme.contrast,
            font=self._type.font("2xs", weight="bold"), padx=10, pady=4, cursor="hand2",
        )

    # -- sidebar ----------------------------------------------------------

    def _build_sidebar(self, parent: tk.Frame) -> None:
        theme = self.theme
        sidebar = tk.Frame(parent, bg=theme.surface)
        sidebar.grid(row=0, column=0, sticky="nsew")

        brand = tk.Frame(sidebar, bg=theme.surface)
        brand.pack(fill="x", padx=14, pady=(14, 4))
        dot = tk.Canvas(brand, width=24, height=24, bg=theme.surface, highlightthickness=0)
        dot.create_oval(1, 1, 23, 23, fill=theme.primary, outline="")
        dot.create_text(
            12, 13, text="ف", fill=theme.primary_foreground,
            font=self._type.font("sm", weight="bold"),
        )
        dot.pack(side="left")
        tk.Label(
            brand, text=TITLE, bg=theme.surface, fg=theme.contrast,
            font=self._type.font("heading3", weight="bold"), anchor="w",
        ).pack(side="left", padx=(8, 0))

        self._section_label(sidebar, "Projects")
        self._project_list = tk.Listbox(
            sidebar, bg=theme.base, fg=theme.foreground, selectbackground=theme.primary,
            selectforeground=theme.primary_foreground, relief="flat", highlightthickness=0,
            font=self._type.font("xs"), height=5, activestyle="none",
        )
        self._project_list.pack(fill="x", padx=14, pady=(0, 6))
        self._project_list.bind("<<ListboxSelect>>", self._on_project_select)
        row = tk.Frame(sidebar, bg=theme.surface)
        row.pack(fill="x", padx=14, pady=(0, 4))
        self._button(row, "New presentation", self._on_new_project).pack(side="left")

        self._project_label = tk.Label(
            sidebar, text="No project selected.", bg=theme.surface, fg=theme.text_secondary,
            font=self._type.font("xs", weight="bold"), anchor="w", justify="left", wraplength=260,
        )
        self._project_label.pack(fill="x", padx=14, pady=(2, 2))
        self._state_label = tk.Label(
            sidebar, text="", bg=theme.surface, fg=theme.text_dim, font=self._type.font("2xs"),
            anchor="w", justify="left", wraplength=260,
        )
        self._state_label.pack(fill="x", padx=14, pady=(0, 4))

        self._section_label(sidebar, "The workflow")
        self._step_buttons: dict[str, tk.Button] = {}
        for key, label in STEPS:
            button = self._button(sidebar, label, lambda k=key: self._run_step(k))
            button.pack(fill="x", padx=14, pady=1)
            self._step_buttons[key] = button

        self._section_label(sidebar, "Slides")
        self._slide_list = tk.Listbox(
            sidebar, bg=theme.base, fg=theme.foreground, selectbackground=theme.primary,
            selectforeground=theme.primary_foreground, relief="flat", highlightthickness=0,
            font=self._type.font("xs"), height=6, activestyle="none",
        )
        self._slide_list.pack(fill="both", expand=True, padx=14, pady=(0, 6))
        self._slide_list.bind("<<ListboxSelect>>", self._on_slide_select)

        self._section_label(sidebar, "Export")
        gate = tk.Frame(sidebar, bg=theme.surface)
        gate.pack(fill="x", padx=14, pady=(0, 4))
        self._export_button = self._button(gate, "Export the .pptx", self._on_export)
        self._export_button.pack(side="left")
        self._export_button.configure(state="disabled")
        self._button(gate, "Save as…", self._on_export_as).pack(side="left", padx=(6, 0))
        self._export_note = tk.Label(
            sidebar, text="", bg=theme.surface, fg=theme.text_secondary,
            font=self._type.font("2xs"), anchor="w", justify="left", wraplength=260,
        )
        self._export_note.pack(fill="x", padx=14, pady=(4, 14))

    # -- conversation -----------------------------------------------------

    def _build_conversation(self, parent: tk.Frame) -> None:
        theme = self.theme
        col = tk.Frame(parent, bg=theme.base)
        col.grid(row=0, column=1, sticky="nsew")

        self._build_composer(col)
        self._build_slide_editor(col)

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
        """The rounded composer card, pinned to the bottom like the chat window."""
        theme = self.theme
        wrap = tk.Frame(parent, bg=theme.base)
        wrap.pack(side="bottom", fill="x", padx=self._column_pad(), pady=(8, 12))
        card = RoundedBox(
            wrap, bg=theme.base, fill=theme.surface, outline=theme.border_subtle,
            corner=RADIUS_LARGE, parent_bg=theme.base,
        )
        card.pack(fill="x")
        inner = card.inner
        self._question = tk.Text(
            inner, height=2, wrap="word", relief="flat", borderwidth=0, highlightthickness=0,
            bg=theme.surface, fg=theme.foreground, insertbackground=theme.primary,
            font=self._type.font("body"), padx=4, pady=4,
        )
        self._question.pack(fill="x", padx=14, pady=(12, 6))
        actions = tk.Frame(inner, bg=theme.surface)
        actions.pack(fill="x", padx=12, pady=(0, 10))
        self._ask_button = self._button(actions, "Discuss", self._on_discuss)
        self._ask_button.pack(side="right")
        tk.Label(
            actions, text="Ask the assistant about the subject…", bg=theme.surface,
            fg=theme.text_dim, font=self._type.font("2xs"), anchor="w",
        ).pack(side="left")

    # -- slide editor panel ------------------------------------------------

    def _build_slide_editor(self, parent: tk.Frame) -> None:
        """The per-slide editor, revealed above the composer when a slide is chosen."""
        theme = self.theme
        panel = tk.Frame(parent, bg=theme.surface)
        head = tk.Frame(panel, bg=theme.surface)
        head.pack(fill="x", padx=12, pady=(10, 4))
        self._slide_label = tk.Label(
            head, text="Select a slide to edit it.", bg=theme.surface, fg=theme.text_secondary,
            font=self._type.font("xs", weight="bold"), anchor="w",
        )
        self._slide_label.pack(side="left")
        self._button(head, "Close", self._hide_slide_editor).pack(side="right")

        form = tk.Frame(panel, bg=theme.surface)
        form.pack(fill="x", padx=12, pady=(0, 4))
        tk.Label(
            form, text="Title", bg=theme.surface, fg=theme.text_dim,
            font=self._type.font("2xs", weight="bold"), anchor="w",
        ).pack(anchor="w")
        tk.Entry(
            form, textvariable=self._slide_title_var, bg=theme.base, fg=theme.foreground,
            insertbackground=theme.primary, relief="flat", highlightthickness=0,
            font=self._type.font("body"),
        ).pack(fill="x", pady=(2, 6))
        type_row = tk.Frame(form, bg=theme.surface)
        type_row.pack(fill="x", pady=(0, 4))
        tk.Label(
            type_row, text="Type", bg=theme.surface, fg=theme.text_dim,
            font=self._type.font("2xs", weight="bold"),
        ).pack(side="left")
        tk.OptionMenu(
            type_row, self._slide_type_var, *CONTENT_TYPES,
        ).pack(side="left", padx=(6, 0))
        tk.Label(
            form, text="Content (one line per bullet / table row / equation)", bg=theme.surface,
            fg=theme.text_dim, font=self._type.font("2xs", weight="bold"), anchor="w",
        ).pack(anchor="w")
        self._slide_body = tk.Text(
            panel, wrap="word", undo=True, bg=theme.base, fg=theme.foreground,
            insertbackground=theme.primary, relief="flat", highlightthickness=0,
            font=self._type.font("body"), padx=8, pady=8, height=8,
        )
        self._slide_body.pack(fill="both", expand=True, padx=12, pady=(2, 6))
        notes_row = tk.Frame(panel, bg=theme.surface)
        notes_row.pack(fill="x", padx=12, pady=(0, 4))
        tk.Label(
            notes_row, text="Speaker notes", bg=theme.surface, fg=theme.text_dim,
            font=self._type.font("2xs", weight="bold"),
        ).pack(side="left")
        self._slide_notes = tk.Entry(
            notes_row, bg=theme.base, fg=theme.foreground, insertbackground=theme.primary,
            relief="flat", highlightthickness=0, font=self._type.font("xs"),
        )
        self._slide_notes.pack(side="left", fill="x", expand=True, padx=(6, 0))

        actions = tk.Frame(panel, bg=theme.surface)
        actions.pack(fill="x", padx=12, pady=(0, 10))
        self._button(actions, "Save slide", self._on_save_slide).pack(side="left")
        self._button(actions, "Approve slide", self._on_approve_slide).pack(side="left", padx=(6, 0))
        self._button(actions, "Withdraw approval", self._on_unapprove_slide).pack(side="left", padx=(6, 0))
        self._button(actions, "Move up", lambda: self._on_move(-1)).pack(side="left", padx=(6, 0))
        self._button(actions, "Move down", lambda: self._on_move(1)).pack(side="left", padx=(6, 0))
        self._button(actions, "Revisions…", self._on_revisions).pack(side="left", padx=(6, 0))
        self._button(actions, "Remove", self._on_remove_slide).pack(side="left", padx=(6, 0))
        self._slide_panel = panel

    def _show_slide_editor(self) -> None:
        if not self._slide_panel.winfo_ismapped():
            self._slide_panel.pack(side="bottom", fill="x", padx=self._column_pad(), pady=(0, 6))

    def _hide_slide_editor(self) -> None:
        self._slide_panel.pack_forget()

    # -- transcript --------------------------------------------------------

    def _column_pad(self) -> int:
        """Side padding that centres the transcript in a readable column."""
        width = self._canvas.winfo_width() if hasattr(self, "_canvas") else 0
        if width <= 1:
            return 16
        return max(16, (width - _COLUMN_MAX) // 2)

    def _available_column(self) -> int:
        """The transcript column's pixel width, capped for readability."""
        width = self._canvas.winfo_width()
        if width <= 1:
            return _COLUMN_MAX
        return max(160, min(_COLUMN_MAX, width - 2 * self._column_pad()))

    def _bubble_fonts(self) -> tuple[tkfont.Font, tkfont.Font]:
        """The proportional and mono fonts a message measures its text with.

        Measured at the same sizes the body and code tags draw with, so the width
        a message reserves matches the text it will hold.
        """
        return (
            tkfont.Font(font=self._type.font("body")),
            tkfont.Font(font=self._type.mono("code")),
        )

    def _bubble_content_width(self, text: str, *, role: str) -> int:
        """The pixel width a message's text needs, never wider than the column.

        A ``Text`` widget's ``width`` is in *characters*, so a pixel figure used
        directly would reserve thousands of characters and force a horizontal
        scrollbar; measuring the widest line in the body font is what keeps a
        message inside the column and still lets it shrink to short text.
        """
        normal, mono = self._bubble_fonts()
        widest = 0
        for line in md.iter_styled_lines(text):
            font = mono if line.in_code else normal
            widest = max(widest, font.measure(line.text))
        if role == "user":
            return max(80, min(widest, int(_COLUMN_MAX * 0.84) - 52))
        return max(120, min(widest, self._available_column()))

    def _bubble_text_width(self, text: str, *, role: str) -> int:
        """The ``Text`` width, in characters, that yields the content width."""
        normal, _mono = self._bubble_fonts()
        char_px = max(1, normal.measure("0"))
        content = self._bubble_content_width(text, role=role)
        # Round *down*: a ``Text`` width is whole characters, so rounding up can
        # ask for a few pixels more than the column and force a horizontal scroll.
        return max(1, content // char_px)

    def _bubble_card_width(self, text: str) -> int:
        """The pixel width of a user bubble's rounded card, chrome included."""
        return self._bubble_content_width(text, role="user") + 52

    def _remetric_rows(self) -> None:
        """Re-measure every message after the column width changed (a resize)."""
        for role, text, _row, widget, card in self._row_frames:
            try:
                if card is not None:
                    card.configure(width=self._bubble_card_width(text))
                widget.configure(width=self._bubble_text_width(text, role=role))
            except tk.TclError:
                continue
            self._fit_height(widget)

    def _add_message(
        self, role: str, text: str, *, title: str = "", mono_body: bool = False, rich: bool = True
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
            if role in ("assistant", "model"):
                self._avatar(head)
                padx = (6, 0)
            else:
                padx = (0, 0)
            tk.Label(
                head, text=title or ("Assistant" if role == "assistant" else "Studio"),
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
        """Run ``job`` on a worker thread, then hand its result back on the UI thread.

        Tk is not thread-safe, so the worker only ever *queues* its result; the UI
        thread applies it through the pump below.
        """
        self._busy = True
        self._set_status(message)
        self._set_busy_buttons(False)
        self._spinner.pack(side="right", padx=8)
        self._spinner.start(12)

        def run() -> None:
            try:
                result = job()
                self._results.put((finish, result, None))
            except Exception as error:  # noqa: BLE001 - shown, never crashes Tk
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
        self._ask_button.configure(state=state)
        for button in self._step_buttons.values():
            button.configure(state=state)
        if not enabled:
            self._export_button.configure(state="disabled")

    def _set_status(self, message: str, *, error: bool = False) -> None:
        self._export_note.configure(
            text=message, fg=self.theme.danger if error else self.theme.text_secondary
        )

    def _report_error(self, error: Exception) -> None:
        self._add_message("note", f"Could not complete that step: {error}", title="Problem")

    # -- project state -----------------------------------------------------

    def _open_first_project(self) -> None:
        projects = self.service.list_projects()
        if projects:
            self._select_project(projects[0].project_id)
        else:
            self._add_message(
                "note",
                "Press **New presentation** to start, then discuss the subject with the "
                "assistant in the box below.",
                title="Getting started",
            )
        self._refresh_projects()

    def _refresh_projects(self) -> None:
        self._project_list.delete(0, "end")
        self._project_ids = []
        for project in self.service.list_projects():
            self._project_ids.append(project.project_id)
            self._project_list.insert("end", f"{project.project_id}  {project.title}")

    def _on_project_select(self, _event: tk.Event | None = None) -> None:
        selection = self._project_list.curselection()
        if not selection:
            return
        project_id = self._project_ids[selection[0]]
        if project_id != self._project_id:
            self._select_project(project_id)

    def _select_project(self, project_id: str) -> None:
        self._project_id = project_id
        self._selected_slide = ""
        self._hide_slide_editor()
        self.refresh()

    def refresh(self) -> None:
        """Reload the project's state, slide list, and export gate."""
        if not self._project_id:
            return
        try:
            project = self.service.get(self._project_id)
        except Exception as error:  # noqa: BLE001 - a deleted project must not crash the window
            self._report_error(error)
            return
        state = STATE_LABELS.get(project.state, project.state)
        self._project_label.configure(text=f"{project.title}")
        self._state_label.configure(
            text=f"{project.project_id}  ·  {state}\n"
            f"{len(project.ordered_slides())} slide(s), "
            f"{len(project.approved_slides)} approved"
        )
        self._refresh_slides(project)
        self._refresh_gate(project)
        self._refresh_steps(project.state)

    def _refresh_steps(self, state: str) -> None:
        """Enable only the workflow steps that make sense in this state."""
        allowed = {
            "discuss": True,
            "outline": state in ("DISCUSSION", "OUTLINE_DRAFT", "OUTLINE_REVIEW"),
            "review-outline": state in ("OUTLINE_DRAFT", "OUTLINE_REVIEW"),
            "approve-outline": state in ("OUTLINE_DRAFT", "OUTLINE_REVIEW"),
            "slides": state in ("SLIDE_DRAFTING", "SLIDE_REVIEW"),
            "review-slides": state in ("SLIDE_DRAFTING", "SLIDE_REVIEW"),
            "approve-content": state in ("SLIDE_DRAFTING", "SLIDE_REVIEW"),
            "export": state in (STATE_CONTENT_APPROVED, STATE_EXPORTED, "EXPORT_FAILED"),
        }
        for key, button in self._step_buttons.items():
            if self._busy:
                button.configure(state="disabled")
            else:
                button.configure(state="normal" if allowed.get(key, False) else "disabled")

    def _refresh_slides(self, project: Any) -> None:
        self._slide_list.delete(0, "end")
        self._slide_ids = []
        for slide in project.ordered_slides():
            self._slide_ids.append(slide.slide_id)
            mark = "✓" if slide.approved else "·"
            self._slide_list.insert("end", f"{slide.number}. {mark} {slide.title or '(untitled)'}")

    def _refresh_gate(self, project: Any) -> None:
        allowed, reason = self.service.can_export(project.project_id)
        self._export_button.configure(state="normal" if allowed and not self._busy else "disabled")
        if allowed:
            self._set_status(
                f"Ready to export {len(project.approved_slides)} approved slide(s) to a .pptx."
            )
        else:
            self._set_status(reason)

    # -- workflow steps ----------------------------------------------------

    def _run_step(self, key: str) -> None:
        handler = {
            "outline": self._on_draft_outline,
            "review-outline": self._on_review_outline,
            "approve-outline": self._on_approve_outline,
            "slides": self._on_draft_slides,
            "review-slides": self._on_review_slides,
            "approve-content": self._on_approve_content,
            "export": self._on_export,
        }.get(key)
        if handler is None:
            return
        if not self._project_id:
            self._set_status("Create a presentation first.", error=True)
            return
        handler()

    def _on_new_project(self) -> None:
        title = simpledialog.askstring(TITLE, "Title of the presentation:", parent=self.window)
        if not title:
            return
        try:
            project = self.service.new_project(title)
        except Exception as error:  # noqa: BLE001 - shown, never crashes Tk
            self._report_error(error)
            return
        self._refresh_projects()
        self._select_project(project.project_id)
        self._add_message(
            "note",
            f"Created **{project.title}** ({project.project_id}). Discuss the subject in the "
            "box below — every answer is cited, and the discussion is what the outline is "
            "drawn from.",
            title="New presentation",
        )

    def _on_discuss(self) -> None:
        if not self._project_id:
            self._set_status("Create a presentation first.", error=True)
            return
        question = self._question.get("1.0", "end").strip()
        if not question:
            self._set_status("Type a question to discuss first.", error=True)
            return
        self._question.delete("1.0", "end")
        self._add_message("user", question)

        def job() -> Any:
            return self.service.discuss(self._project_id, question)

        def finish(result: Any, error: Exception | None) -> None:
            if error is not None:
                self._report_error(error)
                return
            self._add_message("assistant", result.answer, title="Assistant")
            self.refresh()

        self._run_background("Asking the assistant…", job, finish)

    def _on_draft_outline(self) -> None:
        """Ask the user for an optional outline, then draft one."""
        note = simpledialog.askstring(
            TITLE,
            "Optional: write the outline yourself (one section per line, sub-sections "
            "indented). Leave empty to ask the assistant.",
            parent=self.window,
        )
        self.draft_outline(note if note and note.strip() else None)

    def draft_outline(self, text: str | None = None) -> None:
        """Draft an outline (from ``text``, or by asking the assistant)."""

        def job() -> Any:
            return self.service.draft_outline(self._project_id, text=text)

        def finish(result: Any, error: Exception | None) -> None:
            if error is not None:
                self._report_error(error)
                return
            self._outline_revision = result.revision
            self._add_message(
                "note",
                f"**Outline revision {result.revision}** — about {result.estimated_slides} "
                f"slide(s):\n\n{self.service.render_outline(result)}",
                title="Outline",
            )
            self.refresh()

        self._run_background("Drafting the outline…", job, finish)

    def _on_review_outline(self) -> None:
        try:
            project = self.service.review_outline(self._project_id)
        except Exception as error:  # noqa: BLE001 - shown, never crashes Tk
            self._report_error(error)
            return
        self._add_message(
            "note",
            f"The outline is now under review ({STATE_LABELS.get(project.state, project.state)}). "
            "Approve the revision you want, or draft another.",
            title="Outline review",
        )
        self.refresh()

    def _on_approve_outline(self) -> None:
        try:
            project = self.service.approve_outline(self._project_id)
        except Exception as error:  # noqa: BLE001 - shown, never crashes Tk
            self._report_error(error)
            return
        self._add_message(
            "note",
            f"Approved outline revision {project.approved_outline_revision}. Now draft the "
            "slide content.",
            title="Outline approved",
        )
        self.refresh()

    def _on_draft_slides(self) -> None:
        """Ask which section to draft, then draft its slides."""
        heading = simpledialog.askstring(
            TITLE,
            "Draft slides for one section (type its heading), or leave empty to draft every "
            "remaining section.",
            parent=self.window,
        )
        self.draft_slides(heading)

    def draft_slides(self, heading: str | None = None) -> None:
        """Draft the slides for one outline section, or every remaining section."""
        if heading is not None and not heading.strip():
            heading = None
        elif heading is not None:
            heading = heading.strip()

        def job() -> Any:
            return self.service.draft_slides_from_outline(self._project_id, heading=heading)

        def finish(result: Any, error: Exception | None) -> None:
            if error is not None:
                self._report_error(error)
                return
            if not result:
                self._set_status("Nothing new to draft; every section already has a slide.")
                return
            self._add_message(
                "note",
                f"Drafted {len(result)} slide(s). Review each one in the list on the left.",
                title="Slides drafted",
            )
            self.refresh()

        self._run_background("Drafting the slides…", job, finish)

    def _on_review_slides(self) -> None:
        try:
            project = self.service.review_slides(self._project_id)
        except Exception as error:  # noqa: BLE001 - shown, never crashes Tk
            self._report_error(error)
            return
        self._add_message(
            "note",
            f"The slides are now under review ({STATE_LABELS.get(project.state, project.state)}). "
            "Select a slide to edit, reorder, or approve it.",
            title="Slide review",
        )
        self.refresh()

    def _on_approve_content(self) -> None:
        try:
            project = self.service.approve_content(self._project_id)
        except Exception as error:  # noqa: BLE001 - shown, never crashes Tk
            self._set_status(str(error), error=True)
            return
        self._add_message(
            "note",
            f"Content approved ({STATE_LABELS.get(project.state, project.state)}). "
            f"{len(project.approved_slides)} slide(s) are ready to export.",
            title="Content approved",
        )
        self.refresh()

    # -- slide handlers ----------------------------------------------------

    def _on_slide_select(self, _event: tk.Event | None = None) -> None:
        selection = self._slide_list.curselection()
        if not selection:
            return
        slide_id = self._slide_ids[selection[0]]
        self._selected_slide = slide_id
        self._load_slide(slide_id)
        self._show_slide_editor()

    def _load_slide(self, slide_id: str) -> None:
        try:
            slide = self.service.get(self._project_id).slide(slide_id)
        except Exception as error:  # noqa: BLE001 - shown, never crashes Tk
            self._report_error(error)
            return
        self._slide_label.configure(
            text=f"Slide {slide.number} ({slide.slide_id})"
            + ("  ·  approved" if slide.approved else "")
        )
        self._slide_title_var.set(slide.title)
        self._slide_type_var.set(slide.content_type)
        body = "\n".join(
            "|".join(row) for row in slide.table
        ) if slide.content_type == "table" else "\n".join(slide.content)
        self._slide_body.delete("1.0", "end")
        self._slide_body.insert("1.0", body)
        self._slide_notes.delete(0, "end")
        self._slide_notes.insert(0, slide.notes)

    def _read_slide_fields(self) -> dict[str, Any]:
        """The editor's values, shaped for the slide's content type."""
        content_type = self._slide_type_var.get()
        lines = [line for line in self._slide_body.get("1.0", "end").splitlines() if line.strip()]
        fields: dict[str, Any] = {
            "title": self._slide_title_var.get().strip(),
            "content_type": content_type,
            "notes": self._slide_notes.get().strip(),
        }
        if content_type == "table":
            fields["table"] = [line.split("|") for line in lines]
            fields["content"] = []
        elif content_type == "equation":
            fields["equations"] = lines
            fields["content"] = []
        elif content_type == "image":
            fields["images"] = lines
            fields["content"] = []
        else:
            fields["content"] = lines
            fields["table"] = []
        return fields

    def _on_save_slide(self) -> None:
        if not self._selected_slide:
            return
        fields = self._read_slide_fields()

        def job() -> Any:
            return self.service.edit_slide(self._project_id, self._selected_slide, **fields)

        def finish(result: Any, error: Exception | None) -> None:
            if error is not None:
                self._set_status(str(error), error=True)
                return
            project = self.service.get(self._project_id)
            note = f"Saved slide {result.number}."
            if project.approval_outdated:
                note += " The content approval is now outdated; approve the slides again."
            self._set_status(note)
            self.refresh()

        self._run_background("Saving the slide…", job, finish)

    def _on_approve_slide(self) -> None:
        if not self._selected_slide:
            return
        try:
            slide = self.service.approve_slide(self._project_id, self._selected_slide)
        except Exception as error:  # noqa: BLE001 - shown, never crashes Tk
            self._set_status(str(error), error=True)
            return
        self._set_status(f"Approved slide {slide.number}.")
        self.refresh()

    def _on_unapprove_slide(self) -> None:
        if not self._selected_slide:
            return
        try:
            slide = self.service.unapprove_slide(self._project_id, self._selected_slide)
        except Exception as error:  # noqa: BLE001 - shown, never crashes Tk
            self._set_status(str(error), error=True)
            return
        self._set_status(f"Withdrew the approval of slide {slide.number}.")
        self.refresh()

    def _on_move(self, delta: int) -> None:
        if not self._selected_slide:
            return
        try:
            project = self.service.get(self._project_id)
            slide = project.slide(self._selected_slide)
            self.service.reorder_slide(self._project_id, self._selected_slide, slide.order + delta)
        except Exception as error:  # noqa: BLE001 - shown, never crashes Tk
            self._set_status(str(error), error=True)
            return
        self._set_status("Reordered the slide.")
        self.refresh()

    def _on_remove_slide(self) -> None:
        if not self._selected_slide:
            return
        try:
            self.service.remove_slide(self._project_id, self._selected_slide)
        except Exception as error:  # noqa: BLE001 - shown, never crashes Tk
            self._set_status(str(error), error=True)
            return
        self._selected_slide = ""
        self._hide_slide_editor()
        self._set_status("Removed the slide.")
        self.refresh()

    def _on_revisions(self) -> None:
        if not self._selected_slide:
            return
        try:
            items = self.service.revisions(self._project_id, self._selected_slide)
        except Exception as error:  # noqa: BLE001 - shown, never crashes Tk
            self._set_status(str(error), error=True)
            return
        if not items:
            self._set_status("This slide has no recorded revisions yet.")
            return
        lines = "\n".join(
            f"{item.revision}. {'approved' if item.approved else 'draft'} "
            f"({item.reason}) — {item.content.get('title', '')}"
            for item in items
        )
        choice = simpledialog.askstring(
            TITLE,
            f"Revisions of this slide (oldest first):\n\n{lines}\n\n"
            "Type the revision number to restore, or leave empty to cancel:",
            parent=self.window,
        )
        if not choice or not choice.strip().isdigit():
            return
        try:
            self.service.restore_slide_revision(
                self._project_id, self._selected_slide, int(choice.strip())
            )
        except Exception as error:  # noqa: BLE001 - shown, never crashes Tk
            self._set_status(str(error), error=True)
            return
        self._set_status(f"Restored revision {choice.strip()}; approve it again before exporting.")
        self._load_slide(self._selected_slide)
        self.refresh()

    # -- export ------------------------------------------------------------

    def _on_export(self) -> None:
        self._export_to(None)

    def _on_export_as(self) -> None:
        if not self._project_id:
            self._set_status("Create a presentation first.", error=True)
            return
        path = filedialog.asksaveasfilename(
            parent=self.window,
            title="Save the presentation as",
            defaultextension=".pptx",
            filetypes=[("PowerPoint presentation", "*.pptx")],
        )
        if not path:
            return
        self._export_to(path)

    def _export_to(self, path: str | None) -> None:
        if not self._project_id:
            self._set_status("Create a presentation first.", error=True)
            return

        def job() -> Any:
            return self.service.export(self._project_id, path=path)

        def finish(result: Any, error: Exception | None) -> None:
            if error is not None:
                self._set_status(str(error), error=True)
                return
            if not result.ok:
                self._set_status(result.message, error=True)
                return
            self._add_message(
                "note",
                f"**Exported {result.slide_count} slide(s)** to:\n\n`{result.path}`\n\n"
                f"{result.size_bytes:,} bytes. The deck contains exactly the slides you "
                "approved, in the order you approved them.",
                title="Exported",
            )
            self._set_status(result.message)
            self.refresh()

        self._run_background("Writing the .pptx…", job, finish)

    # -- lifecycle ---------------------------------------------------------

    def _close(self) -> None:
        self._closed = True
        if self._on_close is not None:
            self._on_close()
        self.window.destroy()


__all__ = ["PresentationDialog", "STEPS", "SUBTITLE", "TITLE"]
