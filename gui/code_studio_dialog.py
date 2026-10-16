"""The Code Studio window: edit the project's own code, test it, then merge.

This is the project's "code host in a window". It borrows the model a developer
already uses on a code host -- a branch per change, reviewed and merged when it is
good -- and implements it with the project's own ``git``:

* **A workspace is an isolated copy.** *New workspace* makes a ``git worktree``
  under ``data/studio/workspaces/<slug>`` on its own ``studio/<slug>`` branch, cut
  from ``main``. Everything you and the coding model change happens there, so the
  running checkout is never touched and a broken change cannot stop the program.
* **The coding model edits it.** *Ask the model* sends a plain-language request to
  a local coding model (``qwen2.5-coder:7b`` through Ollama by default) which reads
  and writes files *inside the workspace only*, through a strict JSON protocol.
  The request appears as the user's own message, and every step the model takes is
  shown live in the transcript as it happens -- first by first, like the project's
  own chat window -- so its work is reviewable rather than magic.
* **Test, review, merge.** *Run checks* runs the project's own test command inside
  the workspace; *Show diff* is the change as a unified diff; and **Merge** appears
  only when the checks have passed and ``main`` has not moved on, so a merge can
  never overwrite newer work. *Discard* throws the whole change away.

The window is a thin view over :class:`gui.code_studio.CodeStudioService`, so all
of the logic is testable with no display. Tk is imported lazily with the rest of
the window, so the CLI never touches it.
"""

from __future__ import annotations

import queue
import threading
from collections.abc import Callable
from typing import Any

import tkinter as tk
from tkinter import font as tkfont
from tkinter import ttk

from core.coder_model import PULL_HINT
from gui import markdown as md
from gui.app import RADIUS_LARGE, RoundedBox

#: The Studio reads as a centred conversation, capped like the chat window's
#: transcript so long lines stay readable on a wide screen.
_COLUMN_MAX = 820

#: A message taller than this scrolls inside its own card rather than the column.
_MAX_BUBBLE_LINES = 40

TITLE = "Code Studio"
SUBTITLE = (
    "Change the project's own code safely: every change gets its own isolated workspace (a "
    "git worktree on its own branch). Ask the local coding model to make the change, review "
    "the diff, run the tests, then merge — or discard it. The running program is never edited "
    "directly, so a half-finished change can never break it."
)
HONESTY_LINE = (
    "The coding model's edits are real file writes inside the workspace, and nothing it writes "
    "reaches the project until you press Merge — which is refused unless the checks pass and "
    "the base branch has not moved on."
)

#: The default coding model, shown in the status line so the choice is explicit.
DEFAULT_MODEL_NOTE = f"Coding model: qwen2.5-coder:7b (Ollama). If it is missing: {PULL_HINT}"


class CodeStudioDialog:
    """A window for editing, testing, and merging changes to the project."""

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
        self._closed = False
        self._current_slug: str = ""
        self._current_file: str = ""
        self._dirty = False
        # Worker threads hand results back through a queue that the UI thread
        # drains on its own clock. Calling `after` from a worker thread races with
        # the main loop (Tk is not thread-safe), which is what the project's own
        # window avoids the same way.
        self._results: queue.Queue[tuple[Callable[[Any, Exception | None], None], Any, Exception | None]] = (
            queue.Queue()
        )
        # Live agent steps, pushed from the worker thread and drained on the UI
        # thread (Tk is not thread-safe), so the model's work appears as it happens.
        self._steps: queue.Queue[Any] = queue.Queue()
        self._live: tk.Text | None = None

        self._task_var = tk.StringVar(value="")
        self._request_var = tk.StringVar(value="")

        self.window = tk.Toplevel(parent)
        self.window.title(TITLE)
        self.window.configure(bg=theme.base)
        self.window.geometry("1180x608")
        self.window.minsize(900, 600)
        self.window.transient(parent)
        self.window.protocol("WM_DELETE_WINDOW", self._close)
        self._build()
        self.refresh()

    # -- layout -----------------------------------------------------------

    # -- layout -----------------------------------------------------------

    def _build(self) -> None:
        """Assemble the window the way the chat window is assembled.

        A slim header, a sidebar for the workspace, and a centred conversation
        column with its composer pinned to the bottom -- the same shape and the
        same theme tokens as the main chat window, so the Studio reads as part of
        the program rather than a separate tool.
        """
        theme = self.theme
        # Set the first time the model's work is shown; the tests read them after.
        self._trace: Any = None
        self._review: Any = None
        self._row_frames: list[tk.Frame] = []

        self._build_header()

        body = tk.Frame(self.window, bg=theme.base)
        body.pack(fill="both", expand=True)
        body.rowconfigure(0, weight=1)
        body.columnconfigure(0, weight=0, minsize=280)
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
            font=(self.font, 14, "bold"), anchor="w",
        ).pack(anchor="w")
        tk.Label(
            titles, text=SUBTITLE, bg=theme.base, fg=theme.text_secondary,
            font=(self.font, 8), anchor="w", justify="left", wraplength=760,
        ).pack(anchor="w", pady=(2, 0))

        self._spinner = ttk.Progressbar(inner, mode="indeterminate", length=120)
        tk.Frame(self.window, bg=theme.border_subtle, height=1).pack(fill="x")

    # -- sidebar ----------------------------------------------------------

    def _section_label(self, parent: tk.Misc, text: str) -> None:
        tk.Label(
            parent, text=text.upper(), bg=self.theme.surface, fg=self.theme.text_dim,
            font=(self.font, 8, "bold"), anchor="w",
        ).pack(fill="x", padx=14, pady=(10, 2))

    def _build_sidebar(self, parent: tk.Frame) -> None:
        theme = self.theme
        sidebar = tk.Frame(parent, bg=theme.surface)
        sidebar.grid(row=0, column=0, sticky="nsew")

        brand = tk.Frame(sidebar, bg=theme.surface)
        brand.pack(fill="x", padx=14, pady=(14, 4))
        dot = tk.Canvas(brand, width=24, height=24, bg=theme.surface, highlightthickness=0)
        dot.create_oval(1, 1, 23, 23, fill=theme.primary, outline="")
        dot.create_text(12, 13, text="ف", fill=theme.primary_foreground, font=(self.font, 10, "bold"))
        dot.pack(side="left")
        tk.Label(
            brand, text=TITLE, bg=theme.surface, fg=theme.contrast,
            font=(self.font, 12, "bold"), anchor="w",
        ).pack(side="left", padx=(8, 0))

        self._status_line = tk.Label(
            sidebar, text=DEFAULT_MODEL_NOTE, bg=theme.surface, fg=theme.text_dim,
            font=(self.font, 8), anchor="w", justify="left", wraplength=248,
        )
        self._status_line.pack(fill="x", padx=14, pady=(0, 2))

        self._section_label(sidebar, "Workspaces")
        self._workspace_list = tk.Listbox(
            sidebar, bg=theme.base, fg=theme.foreground, selectbackground=theme.primary,
            selectforeground=theme.primary_foreground, relief="flat", highlightthickness=0,
            font=(self.font, 9), height=6, activestyle="none",
        )
        self._workspace_list.pack(fill="x", padx=14, pady=(0, 6))
        self._workspace_list.bind("<<ListboxSelect>>", self._on_workspace_select)

        tk.Label(
            sidebar, text="Describe the change, then create its workspace:",
            bg=theme.surface, fg=theme.text_secondary, font=(self.font, 8),
            anchor="w", justify="left", wraplength=248,
        ).pack(fill="x", padx=14, pady=(0, 3))
        tk.Entry(
            sidebar, textvariable=self._task_var, bg=theme.base, fg=theme.foreground,
            insertbackground=theme.primary, relief="flat", highlightthickness=0,
            font=(self.font, 9),
        ).pack(fill="x", padx=14, pady=(0, 4), ipady=4)
        row = tk.Frame(sidebar, bg=theme.surface)
        row.pack(fill="x", padx=14, pady=(0, 6))
        self._button(row, "New workspace", self._on_create_workspace).pack(side="left")
        self._button(row, "Refresh", self.refresh).pack(side="left", padx=(6, 0))

        self._workspace_label = tk.Label(
            sidebar, text="No workspace selected.", bg=theme.surface, fg=theme.text_secondary,
            font=(self.font, 9, "bold"), anchor="w", justify="left", wraplength=248,
        )
        self._workspace_label.pack(fill="x", padx=14, pady=(0, 2))

        self._section_label(sidebar, "Files")
        self._file_list = tk.Listbox(
            sidebar, bg=theme.base, fg=theme.foreground, selectbackground=theme.primary,
            selectforeground=theme.primary_foreground, relief="flat", highlightthickness=0,
            font=(self.mono, 8), height=8, activestyle="none",
        )
        self._file_list.pack(fill="both", expand=True, padx=14, pady=(0, 6))
        self._file_list.bind("<<ListboxSelect>>", self._on_file_select)

        self._section_label(sidebar, "Review and merge")
        gate = tk.Frame(sidebar, bg=theme.surface)
        gate.pack(fill="x", padx=14, pady=(0, 4))
        self._merge_button = self._button(gate, "Merge into the project", self._on_merge)
        self._merge_button.pack(side="left")
        self._merge_button.configure(state="disabled")
        self._button(gate, "Discard workspace", self._on_discard).pack(side="left", padx=(6, 0))
        self._checks_label = tk.Label(
            sidebar, text="Checks have not run yet.", bg=theme.surface, fg=theme.text_dim,
            font=(self.font, 8), anchor="w", justify="left", wraplength=248,
        )
        self._checks_label.pack(fill="x", padx=14)
        self._merge_note = tk.Label(
            sidebar, text="", bg=theme.surface, fg=theme.text_secondary, font=(self.font, 8),
            anchor="w", justify="left", wraplength=248,
        )
        self._merge_note.pack(fill="x", padx=14, pady=(4, 14))

    # -- conversation -----------------------------------------------------

    def _build_conversation(self, parent: tk.Frame) -> None:
        theme = self.theme
        col = tk.Frame(parent, bg=theme.base)
        col.grid(row=0, column=1, sticky="nsew")

        self._build_composer(col)
        self._build_editor_panel(col)

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
        entry = tk.Entry(
            inner, textvariable=self._request_var, bg=theme.surface, fg=theme.foreground,
            insertbackground=theme.primary, relief="flat", highlightthickness=0,
            font=(self.font, 11),
        )
        entry.pack(fill="x", padx=14, pady=(12, 6))
        entry.bind("<Return>", lambda _event: self._on_ask_model())

        actions = tk.Frame(inner, bg=theme.surface)
        actions.pack(fill="x", padx=12, pady=(0, 10))
        self._ask_button = self._button(actions, "Ask the model", self._on_ask_model)
        self._ask_button.pack(side="right")
        self._checks_button = self._button(actions, "Run checks", self._on_run_checks)
        self._checks_button.pack(side="right", padx=(0, 6))
        self._button(actions, "Show diff", self._on_show_diff).pack(side="right", padx=(0, 6))
        self._button(actions, "Save file", self._on_save_file).pack(side="right", padx=(0, 6))

    def _build_editor_panel(self, parent: tk.Frame) -> None:
        """The file editor, revealed in the column when a file is opened."""
        theme = self.theme
        panel = tk.Frame(parent, bg=theme.surface)
        head = tk.Frame(panel, bg=theme.surface)
        head.pack(fill="x", padx=12, pady=(10, 4))
        self._file_label = tk.Label(
            head, text="Select a file to open it.", bg=theme.surface, fg=theme.text_secondary,
            font=(self.mono, 9), anchor="w",
        )
        self._file_label.pack(side="left")
        self._button(head, "Close", self._hide_editor).pack(side="right")
        self._button(head, "Save file", self._on_save_file).pack(side="right", padx=(0, 6))

        self._editor = tk.Text(
            panel, wrap="none", undo=True, bg=theme.base, fg=theme.foreground,
            insertbackground=theme.primary, relief="flat", highlightthickness=0,
            font=(self.mono, 9), padx=8, pady=8, height=14,
        )
        self._editor.pack(fill="both", expand=True, padx=12, pady=(0, 12))
        self._editor.bind("<<Modified>>", self._on_editor_modified)
        self._editor_panel = panel

    def _show_editor(self) -> None:
        if not self._editor_panel.winfo_ismapped():
            self._editor_panel.pack(side="bottom", fill="x", padx=self._column_pad(), pady=(0, 6))

    def _hide_editor(self) -> None:
        self._editor_panel.pack_forget()

    # -- transcript -------------------------------------------------------

    def _column_pad(self) -> int:
        """Side padding that centres the transcript in a readable column."""
        width = self._canvas.winfo_width() if hasattr(self, "_canvas") else 0
        if width <= 1:
            return 16
        return max(16, (width - _COLUMN_MAX) // 2)

    def _available_column(self) -> int:
        width = self._canvas.winfo_width()
        if width <= 1:
            return _COLUMN_MAX
        return max(160, min(_COLUMN_MAX, width - 2 * self._column_pad()))

    def _bubble_fonts(self) -> tuple[tkfont.Font, tkfont.Font]:
        return tkfont.Font(font=(self.font, 11)), tkfont.Font(font=(self.mono, 9))

    def _bubble_width(self, text: str, *, role: str = "model") -> int:
        normal, mono = self._bubble_fonts()
        widest = 0
        for line in md.iter_styled_lines(text or " "):
            font = mono if line.in_code else normal
            widest = max(widest, font.measure(line.text))
        if role == "user":
            return max(80, min(widest, int(_COLUMN_MAX * 0.84) - 52))
        return max(160, min(widest, self._available_column()))

    def _add_message(
        self, role: str, text: str, *, title: str = "", mono_body: bool = False, rich: bool = True
    ) -> tk.Text:
        """Add one message row in the chat window's style and return its widget."""
        theme = self.theme
        pad = self._column_pad()
        row = tk.Frame(self._transcript, bg=theme.base)
        row.pack(fill="x", padx=(pad, pad), pady=6)
        self._row_frames.append(row)

        if role == "user":
            card = RoundedBox(
                row, bg=theme.user_bubble, fill=theme.user_bubble, outline=theme.user_bubble,
                corner=RADIUS_LARGE, parent_bg=theme.base,
            )
            card.pack(side="right", anchor="e")
            widget = tk.Text(
                card.inner, height=1, wrap="word", relief="flat", borderwidth=0,
                highlightthickness=0, bg=theme.user_bubble, fg=theme.user_bubble_fg,
                font=(self.font, 11), padx=14, pady=8, cursor="xterm", takefocus=1, width=0,
            )
            widget.insert("1.0", text)
            widget.pack(fill="both", expand=True)
            card.configure(width=self._bubble_width(text, role="user"))
        else:
            head = tk.Frame(row, bg=theme.base)
            head.pack(fill="x", pady=(0, 4))
            if role == "model":
                self._avatar(head)
                padx = (6, 0)
            else:
                padx = (0, 0)
            tk.Label(
                head, text=title or ("Coding model" if role == "model" else "Status"),
                bg=theme.base, fg=theme.text_dim, font=(self.font, 9, "bold"), anchor="w",
            ).pack(side="left", padx=padx)
            widget = tk.Text(
                row, height=1, wrap="word", relief="flat", borderwidth=0, highlightthickness=0,
                bg=theme.base, fg=theme.foreground,
                font=(self.mono, 9) if mono_body else (self.font, 11),
                padx=0, pady=0, cursor="xterm", takefocus=1, spacing1=1, spacing3=1,
            )
            if rich and not mono_body:
                self._insert_rich(widget, text)
            else:
                widget.insert("1.0", text)
            widget.pack(anchor="w")
            widget.configure(width=self._bubble_width(text))

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
            12, 13, text="ف", fill=self.theme.primary_foreground, font=(self.font, 10, "bold")
        )
        badge.pack(side="left")

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
        for row in self._row_frames:
            row.pack_configure(padx=(pad, pad))

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

    # -- markdown rendering ----------------------------------------------

    def _insert_rich(self, widget: tk.Text, text: str) -> None:
        self._configure_tags(widget)
        for index, styled in enumerate(md.iter_styled_lines(text)):
            if index:
                widget.insert("end", "\n")
            self._insert_styled_line(widget, styled)

    def _configure_tags(self, widget: tk.Text) -> None:
        theme = self.theme
        widget.tag_configure(
            "heading", font=(self.font, 15, "bold"), foreground=theme.contrast, spacing1=6, spacing3=4
        )
        widget.tag_configure(
            "heading3", font=(self.font, 12, "bold"), foreground=theme.contrast, spacing1=4, spacing3=3
        )
        widget.tag_configure("bullet", lmargin1=16, lmargin2=30)
        widget.tag_configure(
            "quote", lmargin1=16, lmargin2=16, foreground=theme.text_secondary, background=theme.surface_raised
        )
        widget.tag_configure(
            "code", font=(self.mono, 10), background=theme.code_bg, foreground=theme.code_fg,
            lmargin1=10, lmargin2=10, spacing1=2, spacing3=2,
        )
        widget.tag_configure("table", font=(self.mono, 10), foreground=theme.foreground, lmargin1=8)
        widget.tag_configure("rule", foreground=theme.border)
        widget.tag_configure("bold", font=(self.font, 11, "bold"), foreground=theme.contrast)
        widget.tag_configure(
            "inline_code", font=(self.mono, 10), background=theme.code_bg, foreground=theme.code_fg
        )
        widget.tag_configure("kw", foreground=theme.primary)
        widget.tag_configure("str", foreground=theme.success)
        widget.tag_configure("cmt", foreground=theme.text_dim)

    def _insert_styled_line(self, widget: tk.Text, styled: md.StyledLine) -> None:
        start = widget.index("end-1c")
        widget.insert("end", styled.text)
        end = widget.index("end-1c")
        tag = None
        if styled.in_code:
            tag = "code"
        elif styled.kind == md.HEADING:
            tag = "heading" if styled.level <= 2 else "heading3"
        elif styled.kind in (md.BULLET, md.NUMBER):
            tag = "bullet"
        elif styled.kind == md.QUOTE:
            tag = "quote"
        elif styled.kind == md.TABLE:
            tag = "table"
        elif styled.kind == md.RULE:
            tag = "rule"
        if tag:
            widget.tag_add(tag, start, end)
        if styled.in_code:
            self._highlight_code_line(widget, styled.text, start)
        elif styled.kind not in (md.CODE, md.RULE):
            self._inline_tags(widget, styled.text, start)

    def _inline_tags(self, widget: tk.Text, line: str, line_start: str) -> None:
        row = line_start.split(".")[0]
        for segment in md.parse_inline(line):
            if segment.kind == "bold":
                token = f"**{segment.text}**"
                idx = line.find(token)
                if idx >= 0:
                    widget.tag_add("bold", f"{row}.{idx}", f"{row}.{idx + len(token)}")
        position = 0
        while True:
            opening = line.find("`", position)
            if opening < 0:
                break
            closing = line.find("`", opening + 1)
            if closing < 0:
                break
            widget.tag_add("inline_code", f"{row}.{opening}", f"{row}.{closing + 1}")
            position = closing + 1

    def _highlight_code_line(self, widget: tk.Text, line: str, start: str) -> None:
        row = start.split(".")[0]
        for segment in md.highlight_code(line):
            if segment.kind == "plain" or segment.text == "\n":
                continue
            tag = {"keyword": "kw", "string": "str", "comment": "cmt"}.get(segment.kind)
            if tag is None:
                continue
            idx = line.find(segment.text)
            if idx >= 0:
                widget.tag_add(tag, f"{row}.{idx}", f"{row}.{idx + len(segment.text)}")

    # -- widgets ----------------------------------------------------------

    def _button(self, parent: tk.Misc, text: str, command: Any) -> tk.Button:
        theme = self.theme
        return tk.Button(
            parent, text=text, command=command, bg=theme.surface_raised,
            fg=theme.foreground, activebackground=theme.interactive_active,
            activeforeground=theme.contrast, relief="flat", borderwidth=0,
            font=(self.font, 9, "bold"), padx=12, pady=4, cursor="hand2",
        )

    # -- state ------------------------------------------------------------

    def refresh(self) -> None:
        """Reload the workspace list and the model status line."""
        state = self.service.state()
        self._status_line.configure(text=state.coder.summary() if state.coder else "")
        self._workspace_list.delete(0, "end")
        self._workspaces = state.workspaces
        for workspace in state.workspaces:
            label = workspace.slug
            if workspace.checks_passed is True:
                label += "   ✓ checks passed"
            elif workspace.checks_passed is False:
                label += "   ✗ checks failed"
            self._workspace_list.insert("end", label)
        if not state.repository:
            self._workspace_label.configure(
                text="This is not a git checkout, so the Studio cannot make workspaces."
            )
        if self._current_slug and not any(w.slug == self._current_slug for w in state.workspaces):
            self._current_slug = ""
            self._workspace_label.configure(text="No workspace selected.")

    def _selected_slug(self) -> str:
        selection = self._workspace_list.curselection()
        if not selection:
            return ""
        return self._workspaces[selection[0]].slug

    # -- workspaces -------------------------------------------------------

    def _on_workspace_select(self, _event: tk.Event | None = None) -> None:
        slug = self._selected_slug()
        if not slug:
            return
        self._current_slug = slug
        workspace = self.service.workspace(slug)
        self._workspace_label.configure(
            text=f"{workspace.slug}  ·  branch {workspace.branch}  ·  from "
            f"{workspace.base_revision}"
            + (f"  ·  {workspace.task}" if workspace.task else "")
        )
        self._checks_label.configure(
            text=workspace.checks_summary or "Checks have not run yet.",
            fg=self.theme.text_secondary if workspace.checks_passed else self.theme.text_dim,
        )
        self._refresh_files()
        self._refresh_merge_state()
        self._set_trace(workspace.task or "")

    def _on_create_workspace(self) -> None:
        task = self._task_var.get().strip()
        if not task:
            self._set_merge_note("Describe the change first.", error=True)
            return
        if self._busy:
            return
        self._run_background(
            "Creating the workspace…",
            lambda: self.service.create_workspace(task),
            self._after_create,
        )

    def _after_create(self, report: Any, error: Exception | None) -> None:
        if error is not None:
            self._set_merge_note(f"Could not create the workspace: {error}", error=True)
            return
        if not report.ok:
            self._set_merge_note(report.message, error=True)
            return
        self._task_var.set("")
        self.refresh()
        self._set_merge_note(report.message)

    def _on_discard(self) -> None:
        slug = self._current_slug or self._selected_slug()
        if not slug:
            self._set_merge_note("Select a workspace to discard.", error=True)
            return
        report = self.service.remove_workspace(slug)
        self._current_slug = ""
        self._current_file = ""
        self._editor.delete("1.0", "end")
        self._file_list.delete(0, "end")
        self._set_review("")
        self.refresh()
        self._set_merge_note(report.message, error=not report.ok)

    # -- editor -----------------------------------------------------------

    def _refresh_files(self) -> None:
        self._file_list.delete(0, "end")
        if not self._current_slug:
            return
        try:
            entries = self.service.list_files(self._current_slug)
        except Exception as error:  # noqa: BLE001 - shown, never crashes Tk
            self._set_merge_note(f"Could not list files: {error}", error=True)
            return
        self._files = entries
        for entry in entries:
            self._file_list.insert("end", entry.path)

    def _on_file_select(self, _event: tk.Event | None = None) -> None:
        selection = self._file_list.curselection()
        if not selection:
            return
        relative = self._files[selection[0]].path
        report = self.service.read_file(self._current_slug, relative)
        if not report.ok:
            self._set_merge_note(report.message, error=True)
            return
        self._current_file = relative
        self._file_label.configure(text=relative)
        self._show_editor()
        self._editor.delete("1.0", "end")
        self._editor.insert("1.0", report.detail)
        self._editor.edit_modified(False)
        self._dirty = False

    def _on_editor_modified(self, _event: tk.Event | None = None) -> None:
        if self._editor.edit_modified():
            self._dirty = True
            self._file_label.configure(text=f"{self._current_file}  (unsaved)")
            self._editor.edit_modified(False)

    def _on_save_file(self) -> None:
        if not self._current_slug or not self._current_file:
            self._set_merge_note("Open a file first.", error=True)
            return
        content = self._editor.get("1.0", "end-1c")
        report = self.service.save_file(self._current_slug, self._current_file, content)
        self._set_merge_note(report.message, error=not report.ok)
        if report.ok:
            self._dirty = False
            self._file_label.configure(text=self._current_file)
            self._refresh_merge_state(keep_note=True)

    # -- model ------------------------------------------------------------

    def _on_ask_model(self) -> None:
        request = self._request_var.get().strip()
        if not self._current_slug:
            self._set_merge_note("Select a workspace first.", error=True)
            return
        if not request:
            self._set_merge_note("Describe the change you want the model to make.", error=True)
            return
        if self._busy:
            return
        # Show the request as the user's own message, then clear the composer, so
        # the conversation reads like a chat: question first, then the model's work.
        self._add_message("user", request)
        self._request_var.set("")
        # A live bubble the model's steps are appended to as they happen.
        self._live = self._add_message("model", "", title="Coding model", mono_body=True)
        self._run_background(
            "The coding model is working…",
            lambda: self.service.run_agent(self._current_slug, request, on_step=self._on_agent_step),
            self._after_ask,
        )

    def _on_agent_step(self, step: Any) -> None:
        """Queue one live step; the UI thread renders it (never touch Tk here)."""
        self._steps.put(step)

    def _drain_steps(self) -> None:
        """Append any queued agent steps to the live bubble."""
        appended = False
        while True:
            try:
                step = self._steps.get_nowait()
            except queue.Empty:
                break
            lines = step.live_lines()
            if not lines:
                continue
            if self._live is not None:
                text = self._live.get("1.0", "end-1c")
                prefix = "\n" if text.strip() else ""
                self._live.insert("end", prefix + "\n".join(lines))
                appended = True
        if appended and self._live is not None:
            self._fit_height(self._live)
            self._scroll_to_bottom()

    def _after_ask(self, result: Any, error: Exception | None) -> None:
        self._drain_steps()
        self._live = None
        if error is not None:
            self._set_trace(f"The coding model failed: {error}")
            return
        report, trace = result
        if trace is not None:
            self._set_trace(trace.render())
        else:
            self._set_trace(report.message)
        self._set_merge_note(report.message, error=not report.ok)
        self.refresh()
        self._refresh_files()
        self._refresh_merge_state(keep_note=True)

    # -- diff and checks --------------------------------------------------

    def _on_show_diff(self) -> None:
        if not self._current_slug:
            self._set_merge_note("Select a workspace first.", error=True)
            return
        report = self.service.diff(self._current_slug)
        self._set_review(report.detail if report.ok else report.message)

    def _on_run_checks(self) -> None:
        if not self._current_slug:
            self._set_merge_note("Select a workspace first.", error=True)
            return
        if self._busy:
            return
        self._checks_label.configure(text="Running the checks…", fg=self.theme.text_dim)
        self._run_background(
            "Running the project's checks…",
            lambda: self.service.run_checks(self._current_slug),
            self._after_checks,
        )

    def _after_checks(self, result: Any, error: Exception | None) -> None:
        if error is not None:
            self._checks_label.configure(text=f"Checks could not run: {error}", fg=self.theme.danger)
            return
        report, _check = result
        self._set_review(report.detail or report.message)
        self._checks_label.configure(
            text=report.message, fg=self.theme.text_secondary if report.ok else self.theme.danger
        )
        self.refresh()
        self._refresh_merge_state()

    # -- merge ------------------------------------------------------------

    def _refresh_merge_state(self, *, keep_note: bool = False) -> None:
        """Enable Merge only when the gate allows it.

        ``keep_note`` leaves the current message alone, so a caller that has just
        reported something to the user (a save, a model run) does not have its
        confirmation replaced by the merge hint.
        """
        if not self._current_slug:
            self._merge_button.configure(state="disabled")
            if not keep_note:
                self._merge_note.configure(text="")
            return
        report = self.service.can_merge(self._current_slug)
        self._merge_button.configure(state="normal" if report.ok else "disabled")
        if not keep_note:
            self._merge_note.configure(
                text=report.message, fg=self.theme.text_secondary if report.ok else self.theme.text_dim
            )

    def _on_merge(self) -> None:
        if not self._current_slug or self._busy:
            return
        slug = self._current_slug
        self._run_background(
            f"Merging `{slug}` into the project…",
            lambda: self.service.merge(slug),
            self._after_merge,
        )

    def _after_merge(self, report: Any, error: Exception | None) -> None:
        if error is not None:
            self._set_merge_note(f"Merge failed: {error}", error=True)
            return
        self._set_merge_note(report.message, error=not report.ok)
        if report.ok:
            self._set_review(report.detail)
            self._current_slug = ""
            self._current_file = ""
            self._editor.delete("1.0", "end")
            self._file_list.delete(0, "end")
            self._workspace_label.configure(text="Merged. Create a new workspace for the next change.")
        self.refresh()
        # A merged workspace is gone, so Merge must go back to disabled rather than
        # stay lit for a selection that no longer exists.
        self._refresh_merge_state(keep_note=True)

    # -- helpers ----------------------------------------------------------

    def _run_background(self, message: str, job: Callable[[], Any], finish: Callable[[Any, Exception | None], None]) -> None:
        """Run ``job`` on a worker thread, then hand its result back on the UI thread."""
        self._busy = True
        self._set_merge_note(message)
        self._set_buttons_enabled(False)
        self._spinner.pack(side="right", padx=8)
        self._spinner.start(12)

        def run() -> None:
            try:
                result = job()
                self._results.put((finish, result, None))
            except Exception as error:  # noqa: BLE001 - shown, never crashes Tk
                self._results.put((finish, None, error))

        threading.Thread(target=run, daemon=True).start()
        self._schedule_pump()

    def _schedule_pump(self) -> None:
        """Drain finished jobs on the UI thread, then look again shortly."""
        self._pump_results()
        if not self._closed:
            self.window.after(60, self._schedule_pump)

    def _pump_results(self) -> None:
        # Render the model's live steps first, so the transcript keeps pace with
        # the worker even before the run finishes.
        self._drain_steps()
        while True:
            try:
                finish, result, error = self._results.get_nowait()
            except queue.Empty:
                return
            self._busy = False
            self._spinner.stop()
            self._spinner.pack_forget()
            self._set_buttons_enabled(True)
            finish(result, error)

    def _set_buttons_enabled(self, enabled: bool) -> None:
        self._ask_button.configure(state="normal" if enabled else "disabled")
        self._checks_button.configure(state="normal" if enabled else "disabled")
        if not enabled:
            self._merge_button.configure(state="disabled")
        else:
            self._refresh_merge_state()

    def _set_trace(self, text: str) -> None:
        """Show the model's work as a transcript message, not a bare panel."""
        if not text:
            return
        self._trace = self._add_message("model", text, title="Coding model", mono_body=True)

    def _set_review(self, text: str) -> None:
        """Show a diff (or a checks result) as a monospaced transcript message."""
        if not text:
            return
        self._review = self._add_message("note", text, title="Diff and checks", mono_body=True)

    def _set_merge_note(self, message: str, *, error: bool = False) -> None:
        self._merge_note.configure(
            text=message, fg=self.theme.danger if error else self.theme.text_secondary
        )

    def _close(self) -> None:
        self._closed = True
        if self._on_close is not None:
            self._on_close()
        self.window.destroy()


__all__ = ["DEFAULT_MODEL_NOTE", "HONESTY_LINE", "SUBTITLE", "TITLE", "CodeStudioDialog"]
