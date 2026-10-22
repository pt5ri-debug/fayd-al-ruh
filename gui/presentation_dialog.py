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
  title, body, table, equations, notes, its section/subsection, reorder it,
  remove it, regenerate it, and approve it. A command box accepts a plain
  instruction ("shorten slides 11-15", "move slide 18 before slide 17") that the
  shared parser turns into the same operation. An edit after approval is never
  silent -- it reopens the content and the export gate re-arms -- and every change
  is recorded as a revision you can restore.
* **Approve the content**, then **export** a complete ``.pptx`` containing the
  approved slides in exactly the approved order.

The window is a thin view over :class:`gui.presentation.PresentationService`, so
all of the workflow is testable with no display. Tk is imported lazily with the
rest of the window, so the CLI never touches it.
"""

from __future__ import annotations

import queue
import subprocess
import sys
import threading
from collections.abc import Callable
from pathlib import Path
from typing import Any, Literal

import tkinter as tk
from tkinter import filedialog, font as tkfont, messagebox, simpledialog, ttk

from core.presentation import (
    CONTENT_TYPES,
    DEFAULT_SLIDE_BATCH,
    STATE_CONTENT_APPROVED,
    STATE_EXPORTED,
    STATE_LABELS,
    PresentationError,
    theme_names,
)
from gui import markdown as md
from gui import rich_text
from gui.app import RADIUS_LARGE, RoundedBox
from gui.chat_composer import ChatComposer
from gui.typography import resolve
from gui.window_controls import install_window_controls

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
    "Nothing is exported until you approve the content. The .pptx is written locally, in "
    "editable PowerPoint shapes, with the optional python-pptx engine (the built-in "
    "writer is used when it is not installed)."
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
        on_open_final_editor: Callable[[], None] | None = None,
        model_status: Any = None,
        ensure_author: Callable[[], Any] | None = None,
        on_open_course: Callable[[str], None] | None = None,
    ) -> None:
        self.service = service
        self.theme = theme
        self.font = font
        self.mono = mono
        self._type = resolve(parent, ui_family=font, mono_family=mono)
        self._on_close = on_close
        # Called when "Approve the content" opens the final-editing window. The
        # chat window owns that window's lifetime, so this dialog only asks for it.
        self._on_open_final_editor = on_open_final_editor
        # An optional callable returning the authoring model's honest status, shown
        # so the user knows whether a dedicated model is drafting or the assistant.
        self._model_status = model_status
        # Called on the *worker* thread before a draft, so the Studio's own model is
        # loaded there (once) rather than at construction -- loading a multi-gigabyte
        # GGUF must never block the window opening, and the load cost is paid only
        # when a draft is actually asked for.
        self._ensure_author = ensure_author
        # Called with a course key when the deck reaches suitable course content, so
        # the chat window can open the Course window. The Studio only *asks*; it
        # never owns that window.
        self._on_open_course = on_open_course
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
        self._slide_section_var = tk.StringVar(value="")
        self._slide_subsection_var = tk.StringVar(value="")

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
        self._window_controls = install_window_controls(
            self.window, inner, theme=theme, typography=self._type, on_close=self._close
        )
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
        manage = tk.Frame(sidebar, bg=theme.surface)
        manage.pack(fill="x", padx=14, pady=(0, 4))
        self._rename_button = self._button(manage, "Rename", self._on_rename_project)
        self._rename_button.pack(side="left")
        self._duplicate_button = self._button(manage, "Duplicate", self._on_duplicate_project)
        self._duplicate_button.pack(side="left", padx=(6, 0))
        self._delete_button = self._button(manage, "Delete", self._on_delete_project)
        self._delete_button.pack(side="left", padx=(6, 0))

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
        batch_row = tk.Frame(sidebar, bg=theme.surface)
        batch_row.pack(fill="x", padx=14, pady=(0, 4))
        tk.Label(
            batch_row, text="Draft", bg=theme.surface, fg=theme.text_dim,
            font=self._type.font("2xs", weight="bold"),
        ).pack(side="left")
        self._batch_var = tk.StringVar(value=str(self.service.batch_size))
        tk.Spinbox(
            batch_row, from_=1, to=50, width=3, textvariable=self._batch_var,
            command=self._on_batch_size, relief="flat", highlightthickness=0,
            bg=theme.base, fg=theme.foreground, font=self._type.font("2xs"),
            buttonbackground=theme.surface_raised,
        ).pack(side="left", padx=(6, 0))
        tk.Label(
            batch_row, text="slides per batch", bg=theme.surface, fg=theme.text_dim,
            font=self._type.font("2xs"),
        ).pack(side="left", padx=(6, 0))
        self._slide_list = tk.Listbox(
            sidebar, bg=theme.base, fg=theme.foreground, selectbackground=theme.primary,
            selectforeground=theme.primary_foreground, relief="flat", highlightthickness=0,
            font=self._type.font("xs"), height=6, activestyle="none",
        )
        self._slide_list.pack(fill="both", expand=True, padx=14, pady=(0, 6))
        self._slide_list.bind("<<ListboxSelect>>", self._on_slide_select)

        self._section_label(sidebar, "Deck look")
        look = tk.Frame(sidebar, bg=theme.surface)
        look.pack(fill="x", padx=14, pady=(0, 6))
        tk.Label(
            look, text="Theme", bg=theme.surface, fg=theme.text_dim,
            font=self._type.font("2xs"),
        ).pack(anchor="w")
        self._theme_choice = tk.StringVar(value="professional")
        theme_picker = ttk.Combobox(
            look, textvariable=self._theme_choice, state="readonly",
            values=list(theme_names()), font=self._type.font("2xs"),
        )
        theme_picker.pack(fill="x")
        theme_picker.bind("<<ComboboxSelected>>", self._on_theme_change)

        shape = tk.Frame(look, bg=theme.surface)
        shape.pack(fill="x", pady=(4, 0))
        tk.Label(
            shape, text="Size", bg=theme.surface, fg=theme.text_dim,
            font=self._type.font("2xs"),
        ).pack(side="left")
        self._aspect_choice = tk.StringVar(value="16:9")
        aspect_picker = ttk.Combobox(
            shape, textvariable=self._aspect_choice, state="readonly", width=6,
            values=["16:9", "4:3"], font=self._type.font("2xs"),
        )
        aspect_picker.pack(side="left", padx=(6, 0))
        aspect_picker.bind("<<ComboboxSelected>>", self._on_aspect_change)
        self._numbers_var = tk.BooleanVar(value=True)
        tk.Checkbutton(
            shape, text="Numbers", variable=self._numbers_var, command=self._on_numbers_change,
            bg=theme.surface, fg=theme.foreground, selectcolor=theme.base,
            activebackground=theme.surface, font=self._type.font("2xs"),
        ).pack(side="left", padx=(6, 0))

        self._section_label(sidebar, "Export")
        gate = tk.Frame(sidebar, bg=theme.surface)
        gate.pack(fill="x", padx=14, pady=(0, 4))
        self._export_button = self._button(gate, "Export the .pptx", self._on_export)
        self._export_button.pack(side="left")
        self._export_button.configure(state="disabled")
        self._button(gate, "Save as…", self._on_export_as).pack(side="left", padx=(6, 0))
        check_row = tk.Frame(sidebar, bg=theme.surface)
        check_row.pack(fill="x", padx=14, pady=(0, 4))
        self._check_button = self._button(check_row, "Check the deck", self._on_validate)
        self._check_button.pack(side="left")
        self._preview_button = self._button(check_row, "Preview…", self._on_preview)
        self._preview_button.pack(side="left", padx=(6, 0))
        self._download_button = self._button(
            check_row, "Open the folder", self._on_open_folder
        )
        self._download_button.pack(side="left", padx=(6, 0))
        self._download_button.configure(state="disabled")
        self._engine_note = tk.Label(
            sidebar, text="", bg=theme.surface, fg=theme.text_secondary,
            font=self._type.font("2xs"), anchor="w", justify="left", wraplength=260,
        )
        self._engine_note.pack(fill="x", padx=14, pady=(4, 0))
        # Which model drafts the outline and slides: the Studio's own Qwen3 GGUF
        # when installed, else the project's assistant. Said plainly, so the user
        # knows what is writing the content and what to install for the better one.
        self._model_note = tk.Label(
            sidebar, text="", bg=theme.surface, fg=theme.text_secondary,
            font=self._type.font("2xs"), anchor="w", justify="left", wraplength=260,
        )
        self._model_note.pack(fill="x", padx=14, pady=(4, 0))
        self._export_note = tk.Label(
            sidebar, text="", bg=theme.surface, fg=theme.text_secondary,
            font=self._type.font("2xs"), anchor="w", justify="left", wraplength=260,
        )
        self._export_note.pack(fill="x", padx=14, pady=(4, 0))

        self._section_label(sidebar, "Course content")
        course_row = tk.Frame(sidebar, bg=theme.surface)
        course_row.pack(fill="x", padx=14, pady=(0, 4))
        # Enabled only once the deck's own slides match one of the project's
        # courses; the window opens when suitable content is reached, never before.
        self._course_button = self._button(
            course_row, "Open the course…", self._on_course_button
        )
        self._course_button.pack(side="left")
        self._course_button.configure(state="disabled")
        self._course_note = tk.Label(
            sidebar, text="", bg=theme.surface, fg=theme.text_secondary,
            font=self._type.font("2xs"), anchor="w", justify="left", wraplength=260,
        )
        self._course_note.pack(fill="x", padx=14, pady=(4, 14))

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
        """The shared rounded composer card, pinned to the bottom like the chat window."""
        composer = ChatComposer(
            parent, theme=self.theme, typography=self._type,
            placeholder="Ask the assistant about the subject…", on_submit=self._on_discuss,
            outer_bg=self.theme.base, with_attach=False,
        )
        composer.wrap.pack(side="bottom", fill="x", padx=self._column_pad(), pady=(8, 12))
        self._composer = composer
        self._question = composer.entry
        # The Studio's own action sits on the composer's action toolbar, beside the
        # shared Send disc, so the panel is one consistent composer everywhere.
        self._ask_button = self._button(composer.right_tools, "Discuss", self._on_discuss)
        composer.add_tool(self._ask_button)

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
        place_row = tk.Frame(form, bg=theme.surface)
        place_row.pack(fill="x", pady=(0, 4))
        tk.Label(
            place_row, text="Section", bg=theme.surface, fg=theme.text_dim,
            font=self._type.font("2xs", weight="bold"),
        ).pack(side="left")
        tk.Entry(
            place_row, textvariable=self._slide_section_var, bg=theme.base, fg=theme.foreground,
            insertbackground=theme.primary, relief="flat", highlightthickness=0,
            font=self._type.font("xs"), width=18,
        ).pack(side="left", padx=(6, 8))
        tk.Label(
            place_row, text="Subsection", bg=theme.surface, fg=theme.text_dim,
            font=self._type.font("2xs", weight="bold"),
        ).pack(side="left")
        tk.Entry(
            place_row, textvariable=self._slide_subsection_var, bg=theme.base,
            fg=theme.foreground, insertbackground=theme.primary, relief="flat",
            highlightthickness=0, font=self._type.font("xs"), width=18,
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
        self._button(actions, "Approve all slides", self._on_approve_all_slides).pack(side="left", padx=(6, 0))
        self._button(actions, "Withdraw approval", self._on_unapprove_slide).pack(side="left", padx=(6, 0))
        self._button(actions, "Move up", lambda: self._on_move(-1)).pack(side="left", padx=(6, 0))
        self._button(actions, "Move down", lambda: self._on_move(1)).pack(side="left", padx=(6, 0))
        self._button(actions, "Regenerate", self._on_regenerate_slide).pack(side="left", padx=(6, 0))
        self._button(actions, "Revisions…", self._on_revisions).pack(side="left", padx=(6, 0))
        self._button(actions, "Remove", self._on_remove_slide).pack(side="left", padx=(6, 0))

        command_row = tk.Frame(panel, bg=theme.surface)
        command_row.pack(fill="x", padx=12, pady=(0, 10))
        tk.Label(
            command_row, text="Command", bg=theme.surface, fg=theme.text_dim,
            font=self._type.font("2xs", weight="bold"),
        ).pack(side="left")
        self._command_var = tk.StringVar(value="")
        tk.Entry(
            command_row, textvariable=self._command_var, bg=theme.base, fg=theme.foreground,
            insertbackground=theme.primary, relief="flat", highlightthickness=0,
            font=self._type.font("xs"),
        ).pack(side="left", fill="x", expand=True, padx=(6, 6))
        self._button(command_row, "Run", self._on_run_command).pack(side="left")
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
        for button in (self._rename_button, self._duplicate_button, self._delete_button):
            button.configure(state="disabled")
        if not enabled:
            self._export_button.configure(state="disabled")
            self._check_button.configure(state="disabled")
            self._preview_button.configure(state="disabled")
            self._download_button.configure(state="disabled")
            self._course_button.configure(state="disabled")
        else:
            self._sync_project_buttons()

    def _set_status(self, message: str, *, error: bool = False) -> None:
        self._export_note.configure(
            text=message, fg=self.theme.danger if error else self.theme.text_secondary
        )

    def _report_error(self, error: Exception) -> None:
        self._add_message("note", f"Could not complete that step: {error}", title="Problem")

    # -- project state -----------------------------------------------------

    def _open_first_project(self) -> None:
        # Build the list first so selecting the first project can also mark its row.
        self._refresh_projects()
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

    def _refresh_projects(self) -> None:
        self._project_list.delete(0, "end")
        self._project_ids = []
        for project in self.service.list_projects():
            self._project_ids.append(project.project_id)
            # One shared row format for the window and the CLI's `list`.
            self._project_list.insert("end", self.service.status(project.project_id).dashboard_row())
        self._sync_project_buttons()

    def _sync_project_buttons(self) -> None:
        """A management button is live only when a project is selected."""
        state: Literal["normal", "disabled"] = "normal" if self._project_id else "disabled"
        for button in (self._rename_button, self._duplicate_button, self._delete_button):
            button.configure(state=state)

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
        if project_id in self._project_ids:
            index = self._project_ids.index(project_id)
            self._project_list.selection_clear(0, "end")
            self._project_list.selection_set(index)
            self._project_list.see(index)
        self._sync_project_buttons()
        self.refresh()

    def refresh(self, *, keep_note: bool = False) -> None:
        """Reload the project's state, slide list, and export gate.

        ``keep_note`` leaves the status line alone, so a caller that has just
        reported something (a save, an approval, a command's result) does not
        have its confirmation replaced by the export-gate hint.
        """
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
        self._refresh_gate(project, keep_note=keep_note)
        self._refresh_steps(project.state)
        self._sync_project_buttons()

    def _refresh_steps(self, state: str) -> None:
        """Enable only the workflow steps that make sense in this state.

        Step 7 (*Approve the content*) is enabled only once the deck is under
        review, because that is the first state the state machine accepts an
        approval from -- offering it while the slides are still being drafted made
        it look available and then fail. Step 6 (*Review the slides*) stays
        available through drafting so the user can start reviewing early.
        """
        allowed = {
            "discuss": True,
            "outline": state in ("DISCUSSION", "OUTLINE_DRAFT", "OUTLINE_REVIEW"),
            "review-outline": state in ("OUTLINE_DRAFT", "OUTLINE_REVIEW"),
            "approve-outline": state in ("OUTLINE_DRAFT", "OUTLINE_REVIEW"),
            "slides": state in ("SLIDE_DRAFTING", "SLIDE_REVIEW"),
            "review-slides": state in ("SLIDE_DRAFTING", "SLIDE_REVIEW"),
            "approve-content": state in ("SLIDE_REVIEW",),
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
            place = f"  [{slide.topic}]" if slide.topic else ""
            self._slide_list.insert(
                "end", f"{slide.number}. {mark} {slide.title or '(untitled)'}{place}"
            )

    def _refresh_gate(self, project: Any, *, keep_note: bool = False) -> None:
        allowed, reason = self.service.can_export(project.project_id)
        self._export_button.configure(state="normal" if allowed and not self._busy else "disabled")
        self._check_button.configure(state="normal" if not self._busy else "disabled")
        self._preview_button.configure(state="normal" if not self._busy else "disabled")
        self._sync_download_button(project)
        self._sync_settings(project)
        self._sync_course(project)
        self._engine_note.configure(text=self._engine_line())
        self._model_note.configure(text=self._model_line())
        if keep_note:
            return
        if allowed:
            self._set_status(
                f"Ready to export {len(project.approved_slides)} approved slide(s) to a .pptx."
            )
        else:
            self._set_status(reason)

    def _sync_download_button(self, project: Any) -> None:
        """Enable "Open the folder" only when a validated export exists on disk."""
        last = next((item for item in reversed(project.exports) if item.ok), None)
        has_file = last is not None and bool(last.path) and Path(last.path).exists()
        enabled = has_file and not self._busy
        self._download_button.configure(state="normal" if enabled else "disabled")

    def _sync_settings(self, project: Any) -> None:
        """Mirror the project's stored settings into the widgets (no callbacks)."""
        settings = getattr(project, "settings", None)
        if settings is None:
            return
        self._theme_choice.set(settings.theme)
        self._aspect_choice.set(settings.aspect)
        self._numbers_var.set(bool(settings.show_slide_numbers))

    def _sync_course(self, project: Any) -> None:
        """Offer the Course window only once the deck reaches suitable content.

        The deck's *own* slides decide this: a course the deck is substantially
        about (the service's ``best_course``) enables the button and names itself,
        so the window opens at a meaningful moment rather than at random. Before
        that the button is disabled and the note says what would enable it.
        """
        best = None
        try:
            best = self.service.best_course(project.project_id)
        except Exception:  # noqa: BLE001 - a status line must never crash the window
            best = None
        if best is not None:
            self._course_button.configure(state="normal" if not self._busy else "disabled")
            self._course_note.configure(
                text=(
                    f"This deck is about **{best.label}** ({best.branch}) — "
                    f"{best.match_count} of its concepts are covered. Open the course "
                    "to see what is still missing."
                )
            )
        else:
            self._course_button.configure(state="disabled")
            self._course_note.configure(
                text=(
                    "Draft the slides first — the course window opens when the deck's "
                    "content matches one of the project's own courses."
                )
            )

    def _model_line(self) -> str:
        """Say which model drafts the outline and slides, and how to get the better one."""
        status = None
        if callable(self._model_status):
            try:
                status = self._model_status()
            except Exception:  # noqa: BLE001 - a status line must never crash the window
                status = None
        if status is not None and getattr(status, "available", False):
            backend = getattr(status, "backend", "")
            where = "in-process" if backend == "in-process" else "Ollama"
            return (
                f"Drafting model: {getattr(status, 'model', 'qwen3:8b')} ({where}) — "
                "outlines and slides are written by it."
            )
        return (
            "Drafting model: the project's own assistant. Install the Studio's Qwen3 "
            "model for purpose-built slides: `python -m pip install -r "
            "requirements-presentation.txt` then `python -m cli presentation "
            "model-download --approve`."
        )

    def _engine_line(self) -> str:
        """Say which render engine an export would actually use, and why."""
        try:
            engine = self.service.resolve_engine(None)
        except Exception as error:  # noqa: BLE001 - never let a status line crash the window
            return str(error)
        if engine == "pptx":
            return "Engine: python-pptx — real, editable shapes."
        return (
            "Engine: built-in writer (python-pptx is not installed; "
            "run `python -m cli deps --install pptx` for editable shapes)."
        )

    def _on_theme_change(self, _event: Any = None) -> None:
        self._save_settings(theme=self._theme_choice.get())

    def _on_aspect_change(self, _event: Any = None) -> None:
        self._save_settings(aspect=self._aspect_choice.get())

    def _on_numbers_change(self) -> None:
        self._save_settings(show_slide_numbers=bool(self._numbers_var.get()))

    def _save_settings(self, **changes: Any) -> None:
        if not self._project_id:
            return
        try:
            self.service.update_settings(self._project_id, **changes)
        except Exception as error:  # noqa: BLE001 - report, do not crash the window
            self._set_status(str(error), error=True)
            return
        self._set_status("Deck settings saved.")

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

    def _on_rename_project(self) -> None:
        if not self._project_id:
            self._set_status("Select a presentation first.", error=True)
            return
        try:
            current = self.service.get(self._project_id)
        except Exception as error:  # noqa: BLE001 - shown, never crashes Tk
            self._report_error(error)
            return
        title = simpledialog.askstring(
            TITLE, "New title:", initialvalue=current.title, parent=self.window
        )
        if not title or not title.strip():
            return
        try:
            project = self.service.rename_project(self._project_id, title)
        except Exception as error:  # noqa: BLE001 - shown, never crashes Tk
            self._report_error(error)
            return
        self._refresh_projects()
        self._select_project(project.project_id)
        self._add_message("note", f"Renamed to **{project.title}**.", title="Renamed")

    def _on_duplicate_project(self) -> None:
        if not self._project_id:
            self._set_status("Select a presentation first.", error=True)
            return
        try:
            clone = self.service.duplicate_project(self._project_id)
        except Exception as error:  # noqa: BLE001 - shown, never crashes Tk
            self._report_error(error)
            return
        self._refresh_projects()
        self._select_project(clone.project_id)
        self._add_message(
            "note",
            f"Duplicated as **{clone.title}** ({clone.project_id}) with "
            f"{len(clone.slides)} slide(s). The original is unchanged.",
            title="Duplicated",
        )

    def _on_delete_project(self) -> None:
        if not self._project_id:
            self._set_status("Select a presentation first.", error=True)
            return
        try:
            current = self.service.get(self._project_id)
        except Exception as error:  # noqa: BLE001 - shown, never crashes Tk
            self._report_error(error)
            return
        confirmed = messagebox.askyesno(
            TITLE,
            f"Delete {current.project_id} — {current.title!r}?\n\n"
            "The stored project is removed. Any .pptx already exported stays on disk.",
            parent=self.window,
        )
        if not confirmed:
            return
        try:
            self.service.delete(self._project_id)
        except Exception as error:  # noqa: BLE001 - shown, never crashes Tk
            self._report_error(error)
            return
        self._project_id = ""
        self._selected_slide = ""
        self._hide_slide_editor()
        self._refresh_projects()
        self._open_first_project()

    def _on_discuss(self) -> None:
        if not self._project_id:
            self._set_status("Create a presentation first.", error=True)
            return
        question = self._composer.text().strip()
        if not question:
            self._set_status("Type a question to discuss first.", error=True)
            return
        self._composer.clear()
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

    def _prepare_draft(self) -> None:
        """Wire the Studio's own authoring model, if one is installed.

        Called on the worker thread just before a draft. It is best-effort: with no
        model installed, or on a load failure, the Studio simply drafts with the
        project's assistant, which is the documented fallback.
        """
        ensure = self._ensure_author
        if not callable(ensure):
            return
        try:
            ensure()
        except Exception:  # noqa: BLE001 - a missing model is a normal state
            pass

    def draft_outline(self, text: str | None = None) -> None:
        """Draft an outline (from ``text``, or by asking the assistant)."""

        def job() -> Any:
            self._prepare_draft()
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
            self._prepare_draft()
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
                f"Drafted {len(result)} slide(s). Review each one in the list on the left — "
                "select a slide to edit, regenerate, or approve it, or type a command "
                "such as “shorten slides 3-5”.",
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
        """Approve the whole deck, then open the final editor.

        Approving the content is the workflow's *gate*: the state machine requires
        every slide to be approved first (``gui.presentation.approve_content``),
        because the export must contain exactly the approved slides. Rather than
        make the user approve each slide by hand and then press this again, this
        step approves every non-empty slide that is not approved yet, then moves
        the project to *Content approved* and opens the **final editor**, where the
        deck is polished and the .pptx is written.

        A slide that is empty is never approved silently: the service refuses, and
        the reason is shown in the transcript and the status line so the user knows
        exactly which slide to fill or remove.
        """
        try:
            # Approve every slide in one save; the service refuses an empty slide
            # *by number*, so the message names the slide that needs attention.
            self.service.approve_slides(self._project_id)
            project = self.service.approve_content(self._project_id)
        except Exception as error:  # noqa: BLE001 - shown, never crashes Tk
            self._set_status(str(error), error=True)
            self._report_error(error)
            return
        self._add_message(
            "note",
            f"Content approved ({STATE_LABELS.get(project.state, project.state)}). "
            f"{len(project.approved_slides)} slide(s) are ready to export. "
            "The final editor is open — polish any slide there and press "
            "**Export the .pptx**.",
            title="Content approved",
        )
        self.refresh()
        self._open_final_editor()

    def _open_final_editor(self) -> None:
        """Open the final-editing window for the approved deck.

        The window is a *sibling* of this one and drives the same service, so the
        two views can never disagree: an edit made there is visible here on the
        next refresh. When a final editor is already open for this project it is
        brought to the front instead of a second one being created.
        """
        opener = getattr(self, "_on_open_final_editor", None)
        if callable(opener):
            opener()

    # -- course window -----------------------------------------------------

    def _on_course_button(self) -> None:
        """Open the Course window for the course this deck is about.

        Only enabled once the deck's own content matches a course strongly enough
        (``_sync_course``), so the window never opens on a deck with no curriculum
        behind it. The *best* match is the one the button names, so the button and
        the window can never disagree.
        """
        if not self._project_id:
            return
        try:
            best = self.service.best_course(self._project_id)
        except Exception as error:  # noqa: BLE001 - shown, never crashes Tk
            self._report_error(error)
            return
        if best is None:
            self._set_status(
                "The deck does not match a course yet — draft more content first."
            )
            return
        self.open_course(best.key)

    def open_course(self, course_key: str) -> None:
        """Ask the window's owner to open the Course window for ``course_key``."""
        opener = getattr(self, "_on_open_course", None)
        if callable(opener):
            opener(course_key)

    def use_course_concepts(self, text: str) -> None:
        """Put curriculum concepts into the Studio's own question box.

        The Course window hands the deck's *missing* concepts here; the user then
        decides whether to ask the assistant about them. The Studio never drafts
        on its own, so the user keeps control of what enters the deck.
        """
        current = self._composer.text().strip()
        merged = f"{current}\n{text}".strip() if current else text
        self._composer.set_text(merged)
        self._set_status("Concepts copied into the question box — press Discuss to use them.")

    def reveal_slide_number(self, number: int) -> None:
        """Select and show the slide with this 1-based number, if it exists."""
        for index, slide_id in enumerate(self._slide_ids):
            try:
                slide = self.service.get(self._project_id).slide(slide_id)
            except Exception:  # noqa: BLE001 - a stale list must not crash Tk
                return
            if slide.number == number:
                self._slide_list.selection_clear(0, "end")
                self._slide_list.selection_set(index)
                self._slide_list.see(index)
                self._selected_slide = slide_id
                self._load_slide(slide_id)
                self._show_slide_editor()
                return

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
        # Loading a slide makes it the editor's target, so a save/revision acts on
        # the slide the user is looking at even when the editor was opened
        # programmatically (not through a list selection).
        self._selected_slide = slide_id
        self._slide_label.configure(
            text=f"Slide {slide.number} ({slide.slide_id})"
            + (f"  ·  {slide.topic}" if slide.topic else "")
            + ("  ·  approved" if slide.approved else "")
        )
        self._slide_title_var.set(slide.title)
        self._slide_type_var.set(slide.content_type)
        self._slide_section_var.set(slide.section)
        self._slide_subsection_var.set(slide.subsection)
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
            "section": self._slide_section_var.get().strip(),
            "subsection": self._slide_subsection_var.get().strip(),
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
            self.refresh(keep_note=True)

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
        self.refresh(keep_note=True)

    def _on_approve_all_slides(self) -> None:
        """Approve every non-empty slide in one action.

        The export contains exactly the approved slides, so the content gate needs
        every slide approved. Approving them one at a time is what made the
        workflow stall at step 7; this is the one-click equivalent, and an empty
        slide is reported rather than silently approved.
        """
        try:
            project = self.service.approve_slides(self._project_id)
        except Exception as error:  # noqa: BLE001 - shown, never crashes Tk
            self._set_status(str(error), error=True)
            self._report_error(error)
            return
        self._set_status(
            f"Approved {len(project.approved_slides)} slide(s). "
            "Press “Approve the content” when the deck is ready."
        )
        self.refresh(keep_note=True)

    def _on_unapprove_slide(self) -> None:
        if not self._selected_slide:
            return
        try:
            slide = self.service.unapprove_slide(self._project_id, self._selected_slide)
        except Exception as error:  # noqa: BLE001 - shown, never crashes Tk
            self._set_status(str(error), error=True)
            return
        self._set_status(f"Withdrew the approval of slide {slide.number}.")
        self.refresh(keep_note=True)

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
        self.refresh(keep_note=True)

    def _on_batch_size(self) -> None:
        """Apply the chosen drafting batch size (validated by the service)."""
        try:
            size = self.service.set_batch_size(int(self._batch_var.get()))
        except Exception as error:  # noqa: BLE001 - a bad value is shown, not raised
            self._batch_var.set(str(DEFAULT_SLIDE_BATCH))
            self._set_status(str(error), error=True)
            return
        self._batch_var.set(str(size))
        self._set_status(f"Drafting will add up to {size} slide(s) per batch.")

    def _on_regenerate_slide(self) -> None:
        """Re-draft the selected slide, keeping its number and heading."""
        if not self._selected_slide:
            return
        try:
            slide = self.service.get(self._project_id).slide(self._selected_slide)
        except Exception as error:  # noqa: BLE001 - shown, never crashes Tk
            self._report_error(error)
            return
        number = slide.number

        def job() -> Any:
            return self.service.regenerate_slides(self._project_id, [number])

        def finish(result: Any, error: Exception | None) -> None:
            if error is not None:
                self._set_status(str(error), error=True)
                return
            self._load_slide(self._selected_slide)
            self._set_status(f"Regenerated slide {number}; approve it again before exporting.")
            self.refresh(keep_note=True)

        self._run_background(f"Regenerating slide {number}…", job, finish)

    def _on_run_command(self) -> None:
        """Run the typed slide instruction (the shared command parser)."""
        text = self._command_var.get().strip()
        if not text:
            return
        if not self._project_id:
            self._set_status("Create a presentation first.", error=True)
            return
        self._command_var.set("")

        def job() -> Any:
            return self.service.apply_slide_command(self._project_id, text)

        def finish(result: Any, error: Exception | None) -> None:
            if error is not None:
                self._set_status(str(error), error=True)
                return
            self._set_status(result.message, error=not result.ok)
            self.refresh(keep_note=True)

        self._run_background("Running the slide command…", job, finish)

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
        self.refresh(keep_note=True)

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
        self.refresh(keep_note=True)

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

    # -- export validation and preview -------------------------------------

    def _on_validate(self) -> None:
        """Check the deck's structure and numbering without writing a file."""
        if not self._project_id:
            self._set_status("Select a presentation first.", error=True)
            return
        try:
            result = self.service.validate_project(self._project_id)
        except Exception as error:  # noqa: BLE001 - shown, never crashes Tk
            self._set_status(str(error), error=True)
            return
        lines = [result.summary()]
        for issue in result.issues:
            lines.append(f"- issue: {issue}")
        for warning in result.warnings:
            lines.append(f"- warning: {warning}")
        self._add_message(
            "note",
            "\n".join(lines),
            title="Deck check" if result.ok else "Deck check — not ready",
        )
        self._set_status(result.summary(), error=not result.ok)

    def _on_preview(self) -> None:
        """Show the exact slides an export would write, in order, as text."""
        if not self._project_id:
            self._set_status("Select a presentation first.", error=True)
            return
        try:
            deck = self.service.preview(self._project_id)
        except Exception as error:  # noqa: BLE001 - shown, never crashes Tk
            self._set_status(str(error), error=True)
            return
        lines = [f"**{deck.title}** — {len(deck.slides)} slide(s) will be written.", ""]
        for slide in deck.slides:
            lines.append(f"**{slide.number}. {slide.title or '(untitled)'}**")
            for item in slide.content:
                lines.append(f"- {item}")
            for row in slide.table:
                lines.append("| " + " | ".join(row) + " |")
            for equation in slide.equations:
                lines.append(f"= {equation}")
            lines.append("")
        self._add_message("note", "\n".join(lines), title="Preview")

    def _on_open_folder(self) -> None:
        """Open the folder that holds the last validated export (a local action)."""
        if not self._project_id:
            self._set_status("Select a presentation first.", error=True)
            return
        try:
            project = self.service.get(self._project_id)
        except Exception as error:  # noqa: BLE001 - shown, never crashes Tk
            self._report_error(error)
            return
        last = next((item for item in reversed(project.exports) if item.ok), None)
        if last is None or not last.path:
            self._set_status("Export a deck first.", error=True)
            return
        directory = Path(last.path).parent
        self._reveal_directory(directory)

    def _reveal_directory(self, directory: Path) -> None:
        """Ask the OS to open a folder, tolerating a headless or locked desktop."""
        try:
            if sys.platform.startswith("darwin"):
                subprocess.Popen(["open", str(directory)])
            elif sys.platform.startswith("win"):
                subprocess.Popen(["explorer", str(directory)])
            else:
                subprocess.Popen(["xdg-open", str(directory)])
        except Exception as error:  # noqa: BLE001 - the path is still shown, so never crash
            self._set_status(f"Could not open the folder ({error}). It is at {directory}.")
            return
        self._set_status(f"Opened {directory}.")

    def _export_to(self, path: str | None) -> None:
        if not self._project_id:
            self._set_status("Create a presentation first.", error=True)
            return

        def job() -> Any:
            # Validate the deck's structure *before* writing, so a numbering gap or
            # an unapproved slide is reported instead of producing a bad download.
            check = self.service.validate_project(self._project_id)
            if not check.ok:
                raise PresentationError(check.summary())
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
                f"{result.size_bytes:,} bytes, written by the **{result.engine}** engine. "
                "The deck contains exactly the slides you approved, in the order you "
                "approved them.",
                title="Exported",
            )
            for warning in getattr(result, "warnings", ()):
                self._add_message("note", f"⚠ {warning}", title="Check")
            self._set_status(result.message)
            self.refresh(keep_note=True)

        self._run_background("Writing the .pptx…", job, finish)

    # -- lifecycle ---------------------------------------------------------

    def _close(self) -> None:
        self._closed = True
        if self._on_close is not None:
            self._on_close()
        self.window.destroy()


__all__ = ["PresentationDialog", "STEPS", "SUBTITLE", "TITLE"]
