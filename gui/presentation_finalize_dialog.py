"""The final-editing window: polish an approved deck and write the .pptx.

This window opens the moment the user presses **Approve the content** in
:mod:`gui.presentation_dialog`. That is the natural boundary: the *content* is
settled (the state machine has moved the project to ``CONTENT_APPROVED``), and
what remains is presentation work -- fixing a title's wording, reordering two
slides, setting the deck's theme and size, then exporting.

It is built exactly the way the Studio and the chat windows are built (the same
theme tokens, the same type scale, the same rounded composer and centred
column), and it drives the *same* :class:`gui.presentation.PresentationService`
as the Studio and the CLI -- so there is one set of slides, one diff, and one
export gate, never a second copy of the deck.

Two honesty rules carried over from the Studio:

* editing a slide after approval **re-arms the gate**: the service records the
  change, the approval becomes outdated, and the export button is disabled with
  the reason shown until the slides are approved again;
* the export contains **exactly** the approved slides, in the approved order.

Tk is imported here, at module import, which is why this module is only ever
imported lazily from :mod:`gui.app` -- ``cli/`` must never import Tk.
"""

from __future__ import annotations

import queue
import subprocess
import sys
import threading
import tkinter as tk
from collections.abc import Callable
from pathlib import Path
from tkinter import filedialog, simpledialog, ttk
from typing import Any, Literal

from core.best_effort import best_effort
from core.presentation import (
    CONTENT_TYPES,
    PresentationError,
    theme_names,
)

from gui import rich_text
from gui.app import RADIUS_LARGE, RoundedBox
from gui.chat_composer import ChatComposer
from gui.typography import resolve
from gui.window_controls import install_window_controls

TITLE = "Final presentation editor"
SUBTITLE = (
    "Your content is approved. Polish the slides here, then export — the .pptx "
    "contains exactly the slides you approved, in the approved order. Editing a "
    "slide re-arms the approval, so nothing changes after the fact by accident."
)

#: The transcript reads as a centred column, capped like the chat window's.
_COLUMN_MAX = 860

#: A message taller than this scrolls inside its own card rather than the column.
_MAX_BUBBLE_LINES = 40


class PresentationFinalizeDialog:
    """Polish an approved deck and export it, in the chat window's own style."""

    def __init__(
        self,
        parent: tk.Tk | tk.Toplevel,
        service: Any,
        *,
        project_id: str,
        theme: Any,
        font: str,
        mono: str,
        on_close: Callable[[], None] | None = None,
        on_edit: Callable[[], None] | None = None,
    ) -> None:
        self.service = service
        self.theme = theme
        self.font = font
        self.mono = mono
        self._type = resolve(parent, ui_family=font, mono_family=mono)
        self._on_close = on_close
        # Called after any edit/approval change so the Studio window behind can
        # refresh its own view; the two share one service, so a plain refresh is
        # enough and no state is copied between them.
        self._on_edit = on_edit
        self._busy = False
        self._closed = False
        self._project_id = project_id
        self._selected_slide = ""
        self._results: queue.Queue[
            tuple[Callable[[Any, Exception | None], None], Any, Exception | None]
        ] = queue.Queue()
        self._progress: queue.Queue[str] = queue.Queue()
        self._row_frames: list[tuple[str, str, tk.Frame, tk.Text, RoundedBox | None]] = []
        self._slide_ids: list[str] = []
        self._question_var = tk.StringVar(value="")
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
        self._add_message("note", SUBTITLE, title="Final review")
        self.refresh()

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
        self._spinner = ttk.Progressbar(inner, mode="indeterminate", length=120)
        self._window_controls = install_window_controls(
            self.window, inner, theme=theme, typography=self._type, on_close=self._close
        )
        tk.Frame(self.window, bg=theme.titlebar_edge, height=1).pack(fill="x")

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
            brand, text="Final editor", bg=theme.surface, fg=theme.contrast,
            font=self._type.font("heading3", weight="bold"), anchor="w",
        ).pack(side="left", padx=(8, 0))

        self._project_label = tk.Label(
            sidebar, text="", bg=theme.surface, fg=theme.text_secondary,
            font=self._type.font("xs", weight="bold"), anchor="w", justify="left", wraplength=260,
        )
        self._project_label.pack(fill="x", padx=14, pady=(2, 2))
        self._state_label = tk.Label(
            sidebar, text="", bg=theme.surface, fg=theme.text_dim, font=self._type.font("2xs"),
            anchor="w", justify="left", wraplength=260,
        )
        self._state_label.pack(fill="x", padx=14, pady=(0, 4))

        self._section_label(sidebar, "Slides")
        self._slide_list = tk.Listbox(
            sidebar, bg=theme.base, fg=theme.foreground, selectbackground=theme.primary,
            selectforeground=theme.primary_foreground, relief="flat", highlightthickness=0,
            font=self._type.font("xs"), height=10, activestyle="none",
        )
        self._slide_list.pack(fill="both", expand=True, padx=14, pady=(0, 6))
        self._slide_list.bind("<<ListboxSelect>>", self._on_slide_select)

        order = tk.Frame(sidebar, bg=theme.surface)
        order.pack(fill="x", padx=14, pady=(0, 6))
        self._button(order, "Move up", lambda: self._on_move(-1)).pack(side="left")
        self._button(order, "Move down", lambda: self._on_move(1)).pack(side="left", padx=(6, 0))
        self._button(order, "Remove", self._on_remove_slide).pack(side="left", padx=(6, 0))

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
        self._download_button = self._button(check_row, "Open the folder", self._on_open_folder)
        self._download_button.pack(side="left", padx=(6, 0))
        self._download_button.configure(state="disabled")
        self._export_note = tk.Label(
            sidebar, text="", bg=theme.surface, fg=theme.text_secondary,
            font=self._type.font("2xs"), anchor="w", justify="left", wraplength=260,
        )
        self._export_note.pack(fill="x", padx=14, pady=(4, 14))

    # -- conversation -----------------------------------------------------

    def _build_conversation(self, parent: tk.Frame) -> None:
        theme = self.theme
        right = tk.Frame(parent, bg=theme.base)
        right.grid(row=0, column=1, sticky="nsew")
        self._build_transcript(right)
        self._build_slide_panel(right)
        self._build_composer(right)

    def _build_transcript(self, parent: tk.Frame) -> None:
        theme = self.theme
        holder = tk.Frame(parent, bg=theme.base)
        holder.pack(fill="both", expand=True)
        self._canvas = tk.Canvas(holder, bg=theme.base, highlightthickness=0, borderwidth=0)
        self._canvas.pack(side="left", fill="both", expand=True)
        scroll = ttk.Scrollbar(holder, orient="vertical", command=self._canvas.yview)
        scroll.pack(side="right", fill="y")
        self._canvas.configure(yscrollcommand=scroll.set)
        self._transcript = tk.Frame(self._canvas, bg=theme.base)
        self._transcript_id = self._canvas.create_window(
            (0, 0), window=self._transcript, anchor="nw"
        )
        self._transcript.bind(
            "<Configure>",
            lambda _e: self._canvas.configure(scrollregion=self._canvas.bbox("all")),
        )
        self._canvas.bind("<Configure>", self._on_canvas_resize)
        self._canvas.bind_all("<MouseWheel>", self._on_wheel)

    def _on_canvas_resize(self, event: tk.Event) -> None:
        self._canvas.itemconfigure(self._transcript_id, width=event.width)

    def _on_wheel(self, event: tk.Event) -> None:
        self._canvas.yview_scroll(int(-event.delta / 120), "units")

    def _column_pad(self) -> int:
        width = self._canvas.winfo_width() if hasattr(self, "_canvas") else 0
        if width <= 1:
            return 16
        return max(16, (width - _COLUMN_MAX) // 2)

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
        else:
            header = tk.Frame(row, bg=theme.base)
            header.pack(fill="x", anchor="w")
            if title:
                tk.Label(
                    header, text=title, bg=theme.base, fg=theme.text_dim,
                    font=self._type.font("2xs", weight="bold"), anchor="w",
                ).pack(anchor="w")
            widget = tk.Text(
                row, height=1, wrap="word", relief="flat", borderwidth=0,
                highlightthickness=0, bg=theme.base, fg=theme.foreground,
                font=self._type.font("mono" if mono_body else "body"),
                padx=2, pady=2, cursor="xterm", takefocus=1, width=0,
            )
        widget.pack(fill="x", expand=True, anchor="w", pady=(2, 0))
        widget.configure(state="normal")
        if rich and role != "user" and not mono_body:
            rich_text.insert_markdown(widget, text, theme=theme, typography=self._type)
        else:
            widget.insert("1.0", text)
        widget.configure(state="disabled")
        self._fit_height(widget, mono_body=mono_body)
        self._row_frames.append((role, text, row, widget, card))
        self.window.after(30, lambda: self._canvas.yview_moveto(1.0))
        return widget

    def _fit_height(self, widget: tk.Text, *, mono_body: bool = False) -> None:
        lines = int(widget.index("end-1c").split(".")[0])
        widget.configure(height=min(max(lines, 1), _MAX_BUBBLE_LINES))

    def _set_status(self, message: str, *, error: bool = False) -> None:
        self._export_note.configure(
            text=message, fg=self.theme.danger if error else self.theme.text_secondary
        )

    def _report_error(self, error: Exception) -> None:
        self._add_message("note", f"Could not complete that step: {error}", title="Problem")

    # -- background work --------------------------------------------------

    def _run_background(
        self,
        message: str,
        job: Callable[[], Any],
        finish: Callable[[Any, Exception | None], None],
    ) -> None:
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
        for button in (self._check_button, self._preview_button):
            button.configure(state=state)
        if not enabled:
            self._export_button.configure(state="disabled")
            self._download_button.configure(state="disabled")
        else:
            self.refresh(keep_note=True)

    # -- slide panel ------------------------------------------------------

    def _build_slide_panel(self, parent: tk.Frame) -> None:
        theme = self.theme
        panel = tk.Frame(parent, bg=theme.surface, highlightthickness=1,
                         highlightbackground=theme.border_subtle)
        head = tk.Frame(panel, bg=theme.surface)
        head.pack(fill="x", padx=12, pady=(8, 2))
        self._slide_label = tk.Label(
            head, text="Select a slide to edit it.", bg=theme.surface, fg=theme.contrast,
            font=self._type.font("xs", weight="bold"), anchor="w",
        )
        self._slide_label.pack(side="left")
        self._button(head, "Close", self._hide_slide_editor).pack(side="right")

        meta = tk.Frame(panel, bg=theme.surface)
        meta.pack(fill="x", padx=12, pady=(0, 4))
        tk.Label(meta, text="Title", bg=theme.surface, fg=theme.text_dim,
                 font=self._type.font("2xs")).pack(side="left")
        tk.Entry(
            meta, textvariable=self._slide_title_var, bg=theme.base, fg=theme.foreground,
            insertbackground=theme.primary, relief="flat", highlightthickness=0,
            font=self._type.font("xs"),
        ).pack(side="left", fill="x", expand=True, padx=(6, 10))
        tk.Label(meta, text="Type", bg=theme.surface, fg=theme.text_dim,
                 font=self._type.font("2xs")).pack(side="left")
        type_picker = ttk.Combobox(
            meta, textvariable=self._slide_type_var, state="readonly", width=10,
            values=list(CONTENT_TYPES), font=self._type.font("2xs"),
        )
        type_picker.pack(side="left", padx=(6, 0))

        sec = tk.Frame(panel, bg=theme.surface)
        sec.pack(fill="x", padx=12, pady=(0, 4))
        tk.Label(sec, text="Section", bg=theme.surface, fg=theme.text_dim,
                 font=self._type.font("2xs")).pack(side="left")
        tk.Entry(
            sec, textvariable=self._slide_section_var, bg=theme.base, fg=theme.foreground,
            insertbackground=theme.primary, relief="flat", highlightthickness=0,
            font=self._type.font("xs"),
        ).pack(side="left", fill="x", expand=True, padx=(6, 10))
        tk.Label(sec, text="Subsection", bg=theme.surface, fg=theme.text_dim,
                 font=self._type.font("2xs")).pack(side="left")
        tk.Entry(
            sec, textvariable=self._slide_subsection_var, bg=theme.base, fg=theme.foreground,
            insertbackground=theme.primary, relief="flat", highlightthickness=0,
            font=self._type.font("xs"),
        ).pack(side="left", fill="x", expand=True, padx=(6, 0))

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
        self._button(actions, "Approve all", self._on_approve_all_slides).pack(side="left", padx=(6, 0))
        self._button(actions, "Regenerate", self._on_regenerate_slide).pack(side="left", padx=(6, 0))
        self._button(actions, "Add a slide…", self._on_add_slide).pack(side="left", padx=(6, 0))
        self._slide_panel = panel

    def _show_slide_editor(self) -> None:
        if not self._slide_panel.winfo_ismapped():
            self._slide_panel.pack(side="bottom", fill="x", padx=self._column_pad(), pady=(0, 6))

    def _hide_slide_editor(self) -> None:
        self._slide_panel.pack_forget()

    # -- composer ---------------------------------------------------------

    def _build_composer(self, parent: tk.Frame) -> None:
        composer = ChatComposer(
            parent, theme=self.theme, typography=self._type,
            placeholder="Describe what you want to change in the deck…",
            on_submit=self._on_ask, outer_bg=self.theme.base, with_attach=False,
            variable=self._question_var,
        )
        composer.wrap.pack(fill="x", padx=16, pady=(0, 12))
        self._composer = composer
        # The deck's own action sits on the composer's action toolbar, beside the
        # shared Send disc, so the panel is one consistent composer everywhere.
        composer.add_tool(self._button(composer.right_tools, "Send", self._on_ask))

    def _on_ask(self) -> None:
        """Apply a plain instruction to the whole deck (the shared command parser)."""
        text = self._question_var.get().strip()
        if not text or not self._project_id:
            return
        self._question_var.set("")
        self._add_message("user", text)

        def job() -> Any:
            return self.service.apply_slide_command(self._project_id, text)

        def finish(result: Any, error: Exception | None) -> None:
            if error is not None:
                self._report_error(error)
                return
            self._add_message("note", result.message, title="Applied")
            self.refresh(keep_note=True)

        self._run_background("Applying the change…", job, finish)

    # -- state ------------------------------------------------------------

    def refresh(self, *, keep_note: bool = False) -> None:
        try:
            project = self.service.get(self._project_id)
        except Exception as error:
            self._report_error(error)
            return
        from core.presentation import STATE_LABELS

        state = STATE_LABELS.get(project.state, project.state)
        self._project_label.configure(text=project.title)
        self._state_label.configure(
            text=f"{project.project_id}  ·  {state}\n"
            f"{len(project.ordered_slides())} slide(s), "
            f"{len(project.approved_slides)} approved"
        )
        self._refresh_slides(project)
        self._sync_settings(project)
        self._refresh_gate(project, keep_note=keep_note)

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

    def _sync_settings(self, project: Any) -> None:
        settings = getattr(project, "settings", None)
        if settings is None:
            return
        self._theme_choice.set(getattr(settings, "theme", self._theme_choice.get()))
        aspect = getattr(settings, "aspect", "16:9")
        self._aspect_choice.set("4:3" if str(aspect).startswith("4") else "16:9")
        self._numbers_var.set(bool(getattr(settings, "show_numbers", True)))

    def _refresh_gate(self, project: Any, *, keep_note: bool = False) -> None:
        allowed, reason = self.service.can_export(self._project_id)
        self._export_button.configure(state="normal" if allowed and not self._busy else "disabled")
        self._check_button.configure(state="normal" if not self._busy else "disabled")
        self._preview_button.configure(state="normal" if not self._busy else "disabled")
        self._sync_download_button(project)
        if keep_note:
            return
        if allowed:
            self._set_status(
                f"Ready to export {len(project.approved_slides)} approved slide(s) to a .pptx."
            )
        else:
            self._set_status(reason)

    def _sync_download_button(self, project: Any) -> None:
        last = next((item for item in reversed(project.exports) if item.ok), None)
        enabled = last is not None and bool(last.path) and not self._busy
        self._download_button.configure(state="normal" if enabled else "disabled")

    # -- slide handlers ---------------------------------------------------

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
        except Exception as error:
            self._report_error(error)
            return
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
        body = (
            "\n".join("|".join(row) for row in slide.table)
            if slide.content_type == "table"
            else "\n".join(slide.content)
        )
        self._slide_body.delete("1.0", "end")
        self._slide_body.insert("1.0", body)
        self._slide_notes.delete(0, "end")
        self._slide_notes.insert(0, slide.notes)

    def _read_slide_fields(self) -> dict[str, Any]:
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
                self._report_error(error)
                return
            project = self.service.get(self._project_id)
            note = f"Saved slide {result.number}."
            if project.approval_outdated:
                note += " The content approval is now outdated; approve the slides again."
            self._set_status(note)
            self.refresh(keep_note=True)
            self._notify_edit()

        self._run_background("Saving the slide…", job, finish)

    def _on_approve_slide(self) -> None:
        if not self._selected_slide:
            return
        try:
            slide = self.service.approve_slide(self._project_id, self._selected_slide)
        except Exception as error:
            self._set_status(str(error), error=True)
            return
        self._set_status(f"Approved slide {slide.number}.")
        self.refresh(keep_note=True)
        self._notify_edit()

    def _on_approve_all_slides(self) -> None:
        try:
            project = self.service.approve_slides(self._project_id)
        except Exception as error:
            self._set_status(str(error), error=True)
            self._report_error(error)
            return
        self._set_status(f"Approved {len(project.approved_slides)} slide(s).")
        self.refresh(keep_note=True)
        self._notify_edit()

    def _on_move(self, delta: int) -> None:
        if not self._selected_slide:
            return
        try:
            project = self.service.get(self._project_id)
            slide = project.slide(self._selected_slide)
            self.service.reorder_slide(self._project_id, self._selected_slide, slide.order + delta)
        except Exception as error:
            self._set_status(str(error), error=True)
            return
        self._set_status("Reordered the slide.")
        self.refresh(keep_note=True)
        self._notify_edit()

    def _on_remove_slide(self) -> None:
        if not self._selected_slide:
            return
        try:
            self.service.remove_slide(self._project_id, self._selected_slide)
        except Exception as error:
            self._set_status(str(error), error=True)
            return
        self._selected_slide = ""
        self._hide_slide_editor()
        self._set_status("Removed the slide.")
        self.refresh(keep_note=True)
        self._notify_edit()

    def _on_add_slide(self) -> None:
        """Append a blank slide the user then fills in and approves."""
        try:
            spec = _prompt_for_slide(self.window)
        except Exception as error:
            self._report_error(error)
            return
        if spec is None:
            return
        try:
            slide = self.service.insert_slide(self._project_id, spec)
        except Exception as error:
            self._set_status(str(error), error=True)
            return
        self._set_status(f"Added slide {slide.number}; fill it in and approve it.")
        self.refresh(keep_note=True)
        self._notify_edit()

    def _on_regenerate_slide(self) -> None:
        """Re-draft the selected slide with the Studio's own authoring model."""
        if not self._selected_slide:
            return
        try:
            slide = self.service.get(self._project_id).slide(self._selected_slide)
        except Exception as error:
            self._report_error(error)
            return
        number = slide.number

        def job() -> Any:
            return self.service.regenerate_slides(self._project_id, [number])

        def finish(result: Any, error: Exception | None) -> None:
            if error is not None:
                self._report_error(error)
                return
            self._load_slide(self._selected_slide)
            self._set_status(f"Regenerated slide {number}; approve it again before exporting.")
            self.refresh(keep_note=True)
            self._notify_edit()

        self._run_background(f"Regenerating slide {number}…", job, finish)

    def _notify_edit(self) -> None:
        if self._on_edit is not None:
            self._on_edit()

    # -- deck look --------------------------------------------------------

    def _apply_settings(self, **fields: Any) -> None:
        try:
            self.service.update_settings(self._project_id, **fields)
        except Exception as error:
            self._set_status(str(error), error=True)
            return
        self._set_status("Updated the deck's look.")
        self.refresh(keep_note=True)

    def _on_theme_change(self, _event: tk.Event | None = None) -> None:
        self._apply_settings(theme=self._theme_choice.get())

    def _on_aspect_change(self, _event: tk.Event | None = None) -> None:
        self._apply_settings(aspect=self._aspect_choice.get())

    def _on_numbers_change(self) -> None:
        self._apply_settings(show_numbers=bool(self._numbers_var.get()))

    # -- validation, preview, export --------------------------------------

    def _on_validate(self) -> None:
        try:
            result = self.service.validate_project(self._project_id)
        except Exception as error:
            self._report_error(error)
            return
        lines = [result.summary()]
        for issue in result.issues:
            lines.append(f"- issue: {issue}")
        for warning in result.warnings:
            lines.append(f"- warning: {warning}")
        self._add_message(
            "note", "\n".join(lines), title="Deck check" if result.ok else "Deck check — not ready"
        )
        self._set_status(result.summary(), error=not result.ok)

    def _on_preview(self) -> None:
        try:
            deck = self.service.preview(self._project_id)
        except Exception as error:
            self._report_error(error)
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
        def job() -> Any:
            check = self.service.validate_project(self._project_id)
            if not check.ok:
                raise PresentationError(check.summary())
            return self.service.export(self._project_id, path=path)

        def finish(result: Any, error: Exception | None) -> None:
            if error is not None:
                self._report_error(error)
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
                "approved them.\n\nPress **Open the folder** to show it in your file "
                "manager.",
                title="Exported",
            )
            for warning in getattr(result, "warnings", ()):
                self._add_message("note", f"⚠ {warning}", title="Check")
            self._set_status(result.message)
            self.refresh(keep_note=True)

        self._run_background("Writing the .pptx…", job, finish)

    def _on_open_folder(self) -> None:
        try:
            project = self.service.get(self._project_id)
        except Exception as error:
            self._report_error(error)
            return
        last = next((item for item in reversed(project.exports) if item.ok), None)
        if last is None or not last.path:
            self._set_status("Export a deck first.", error=True)
            return
        self._reveal_directory(Path(last.path).parent)

    def _reveal_directory(self, directory: Path) -> None:
        """Ask the OS to open a folder, tolerating a headless or locked desktop."""
        try:
            if sys.platform.startswith("darwin"):
                subprocess.Popen(["open", str(directory)])
            elif sys.platform.startswith("win"):
                subprocess.Popen(["explorer", str(directory)])
            else:
                subprocess.Popen(["xdg-open", str(directory)])
        except Exception as error:
            self._set_status(f"Could not open {directory}: {error}", error=True)

    # -- lifecycle --------------------------------------------------------

    def _close(self) -> None:
        self._closed = True
        with best_effort("unbind the canvas wheel handler"):
            self._canvas.unbind_all("<MouseWheel>")
        if self._on_close is not None:
            self._on_close()
        self.window.destroy()


def _prompt_for_slide(parent: tk.Misc) -> Any:
    """Ask for a new slide's title and heading; return a ``SlideSpec`` or ``None``."""
    from gui.presentation import SlideSpec

    title = simpledialog.askstring(TITLE, "Title of the new slide:", parent=parent)
    if title is None:
        return None
    heading = simpledialog.askstring(
        TITLE,
        "Which section does it belong to? (optional)",
        parent=parent,
    )
    body = simpledialog.askstring(
        TITLE,
        "Bullets, one per line (optional):",
        parent=parent,
    )
    bullets = [line.strip() for line in (body or "").splitlines() if line.strip()]
    section = (heading or "").strip()
    return SlideSpec(
        title=title.strip() or "(untitled)",
        content=bullets,
        section=section,
        subsection="",
        sources=(section,) if section else (),
    )


__all__ = ["SUBTITLE", "TITLE", "PresentationFinalizeDialog"]
