"""The Tkinter window: a professional desktop front end for Fayd Al-Ruh.

Tkinter ships with CPython, so the GUI adds no dependency. Long work (an answer
or a sidebar command) runs on a worker thread and posts results back through a
queue; Tk is only ever touched from the main thread, as Tk requires.

The look is deliberately restrained: a flat, high-contrast palette, a generous
transcript with role-labelled message cards, and pill-shaped controls, so the
window reads like a modern desktop assistant rather than a bare form.

This module is imported lazily by the CLI, so importing it never opens a window
and never breaks the command-line path.
"""

from __future__ import annotations

import contextlib
import io
import queue
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import tkinter as tk
from tkinter import filedialog
from tkinter import font as tkfont
from tkinter import simpledialog

from core.paths import data_dir
from core.version import __version__
from gui.actions import ACTIONS, GuiAction, SessionItem, saved_session_items
from gui.attachments import Attachment, AttachmentNotice
from gui.options import build_parser
from gui.service import ChatReply, ChatService
from storage.json_store import JsonStore

# --------------------------------------------------------------------- theme


@dataclass(frozen=True)
class Theme:
    """One appearance: every colour the window draws with."""

    name: str
    sidebar_bg: str
    sidebar_hover: str
    sidebar_active: str
    main_bg: str
    card_bg: str
    user_bg: str
    user_fg: str
    user_accent: str
    toolbar_bg: str
    border: str
    text: str
    muted: str
    faint: str
    accent: str
    accent_hover: str
    accent_text: str
    disabled: str


DARK_THEME = Theme(
    name="dark",
    sidebar_bg="#17171C",
    sidebar_hover="#24242C",
    sidebar_active="#2D2D38",
    main_bg="#1E1E24",
    card_bg="#2A2A33",
    user_bg="#2563A8",
    user_fg="#FFFFFF",
    user_accent="#DCE8F7",
    toolbar_bg="#1A1A20",
    border="#33333D",
    text="#ECECF1",
    muted="#9A9AA8",
    faint="#6E6E7A",
    accent="#12A594",
    accent_hover="#0F8C7E",
    accent_text="#04211D",
    disabled="#2C4A46",
)

LIGHT_THEME = Theme(
    name="light",
    sidebar_bg="#F2F2F5",
    sidebar_hover="#E4E4EA",
    sidebar_active="#D9D9E0",
    main_bg="#FFFFFF",
    card_bg="#F4F4F7",
    user_bg="#2563A8",
    user_fg="#FFFFFF",
    user_accent="#DCE8F7",
    toolbar_bg="#FAFAFC",
    border="#E0E0E6",
    text="#1B1B22",
    muted="#5C5C69",
    faint="#8A8A97",
    accent="#0E9384",
    accent_hover="#0B7C70",
    accent_text="#FFFFFF",
    disabled="#BFD9D4",
)

THEMES = {theme.name: theme for theme in (DARK_THEME, LIGHT_THEME)}

#: The window's saved appearance lives in the user's data directory.
_THEME_NAMESPACE = "gui"
_THEME_KEY = "appearance"

# A paragraph-cap for a single message; longer notes scroll inside their card.
_MAX_BUBBLE_LINES = 40


def _pick_font() -> str:
    """A clean UI font, falling back to whatever Tk offers."""
    for name in ("Segoe UI", "Helvetica Neue", "DejaVu Sans", "Helvetica", "Arial"):
        if name in tkfont.families():
            return name
    return "TkDefaultFont"


def load_theme_name() -> str:
    """The saved theme name, defaulting to dark; never raises on a bad store."""
    try:
        saved = JsonStore(data_dir()).load(_THEME_NAMESPACE, _THEME_KEY)
    except Exception:  # a broken store must not stop the window opening
        return DARK_THEME.name
    name = saved.get("theme") if isinstance(saved, dict) else None
    return name if name in THEMES else DARK_THEME.name


def save_theme_name(name: str) -> None:
    """Persist the chosen theme; a store error is silently ignored."""
    if name not in THEMES:
        return
    try:
        JsonStore(data_dir()).save(_THEME_NAMESPACE, _THEME_KEY, {"theme": name})
    except Exception:
        pass


class _Tooltip:
    """A small hover hint for a control, shown after a short delay."""

    def __init__(self, widget: tk.Widget, text: str, theme: Theme, font: str) -> None:
        self._widget = widget
        self._text = text
        self._theme = theme
        self._font = font
        self._window: tk.Toplevel | None = None
        self._after: str | None = None
        widget.bind("<Enter>", self._schedule, add="+")
        widget.bind("<Leave>", self._hide, add="+")
        widget.bind("<ButtonPress>", self._hide, add="+")

    def _schedule(self, _event: tk.Event) -> None:
        self._after = self._widget.after(450, self._show)

    def _show(self) -> None:
        if self._window is not None:
            return
        window = tk.Toplevel(self._widget)
        window.wm_overrideredirect(True)
        window.configure(bg=self._theme.border)
        tk.Label(
            window, text=self._text, bg=self._theme.card_bg, fg=self._theme.text,
            font=(self._font, 9), padx=8, pady=4, relief="flat",
        ).pack()
        x = self._widget.winfo_rootx() + 4
        y = self._widget.winfo_rooty() - 32
        window.wm_geometry(f"+{max(x, 0)}+{max(y, 0)}")
        self._window = window

    def _hide(self, _event: tk.Event | None = None) -> None:
        if self._after is not None:
            self._widget.after_cancel(self._after)
            self._after = None
        if self._window is not None:
            self._window.destroy()
            self._window = None


class LineStream(io.TextIOBase):
    """A text stream that forwards each complete line to a queue.

    Training entry points print their live progress to stderr; the GUI redirects
    stderr through this stream so each line reaches the transcript as it is
    written, instead of arriving only after the run ends.
    """

    def __init__(self, lines: "queue.Queue[str]") -> None:
        self._lines = lines
        self._buffer = ""

    def write(self, text: str) -> int:
        self._buffer += text
        while "\n" in self._buffer:
            line, self._buffer = self._buffer.split("\n", 1)
            if line.strip():
                self._lines.put(line)
        return len(text)

    def flush(self) -> None:
        if self._buffer.strip():
            self._lines.put(self._buffer)
        self._buffer = ""


class FaydGUI:
    """The application window and its wiring to :class:`ChatService`."""

    def __init__(
        self,
        root: tk.Tk,
        *,
        service: ChatService | None = None,
        model_checkpoint: str | None = None,
        no_web: bool = False,
        session: str | None = None,
        theme: str | None = None,
    ) -> None:
        self.root = root
        self.service = service or ChatService(
            no_web=no_web,
            model_checkpoint=model_checkpoint,
            session=session,
        )
        self._results: queue.Queue[tuple[str, Any]] = queue.Queue()
        self._lines: queue.Queue[str] = queue.Queue()
        self._busy = False
        self._streaming_label: tk.Text | None = None
        # Live output is coalesced and rendered at a bounded rate: a fast training
        # loop can emit thousands of lines, and reconfiguring on every one would
        # freeze the window.
        self._pending: list[str] = []
        self._last_render = 0.0
        self._bubbles: list[tk.Text] = []
        self._bubble_roles: dict[tk.Text, str] = {}
        self._font = _pick_font()
        self._thinking: tk.Label | None = None
        self._thinking_row: tk.Widget | None = None
        self._thinking_phase = 0

        # The theme is read from the user's saved preference unless one was passed
        # in; both toggle later through _toggle_theme.
        name = theme if theme in THEMES else load_theme_name()
        self._theme = THEMES[name]

        # The transcript is selectable and copyable; Delete clears it, Ctrl+C
        # copies the selection, Ctrl+A selects the whole transcript, and a
        # right-click offers the same as a menu.
        root.bind("<Control-c>", self._on_copy)
        root.bind("<Control-C>", self._on_copy)
        root.bind("<Control-a>", self._on_select_all)
        root.bind("<Control-A>", self._on_select_all)
        root.bind("<Delete>", self._on_clear_transcript)
        root.bind("<Button-3>", self._on_right_click)
        root.bind("<Control-n>", self._on_new_chat)
        root.bind("<Control-s>", self._on_save)
        self._clipboard_menu = tk.Menu(root, tearoff=0)
        self._clipboard_menu.add_command(label="Copy", command=self._on_copy)
        self._clipboard_menu.add_command(
            label="Copy conversation", command=self._on_copy_conversation
        )
        self._clipboard_menu.add_command(label="Select all", command=self._on_select_all)
        self._clipboard_menu.add_separator()
        self._clipboard_menu.add_command(label="Clear transcript", command=self._clear_transcript)

        root.title("Fayd Al-Ruh")
        root.geometry("1160x768")
        root.minsize(880, 560)

        self._build_ui()

    # ------------------------------------------------------------------ layout

    def _build_ui(self) -> None:
        """Build (or rebuild) the whole window for the current theme.

        Called again by :meth:`_toggle_theme`; the transcript is snapshotted
        first and restored, so switching appearance keeps every message.
        """
        had_messages = bool(self._bubbles)
        self._bubbles = []
        self._bubble_roles = {}
        self._pending = []
        self._streaming_label = None
        self._thinking_row = None
        self._thinking = None
        for child in self.root.winfo_children():
            if isinstance(child, tk.Menu):  # the clipboard menu outlives a rebuild
                continue
            child.destroy()
        self.root.configure(bg=self._theme.main_bg)

        self._build_sidebar()
        self._build_main()
        self._set_source_label()
        if had_messages:
            self._build_thinking()  # ready, but hidden until a job starts
        else:
            self._show_hero()
        self._set_send_enabled(not self._busy)
        if not getattr(self, "_pumping", False):
            self._pumping = True
            self._pump()

    def _build_sidebar(self) -> None:
        sidebar = tk.Frame(self.root, bg=self._theme.sidebar_bg, width=272)
        sidebar.pack(side="left", fill="y")
        sidebar.pack_propagate(False)

        self._brand(sidebar)

        self._status = tk.Label(
            sidebar, text=self._status_text(), bg=self._theme.sidebar_bg, fg=self._theme.faint,
            font=(self._font, 9), anchor="w", justify="left", padx=18, pady=12,
        )
        self._status.pack(side="bottom", fill="x")

        # The actions and saved chats can outgrow the window, so the body scrolls
        # behind a visible scrollbar that appears only when it actually overflows.
        body = tk.Frame(sidebar, bg=self._theme.sidebar_bg)
        body.pack(side="top", fill="both", expand=True)
        self._side_canvas = tk.Canvas(body, bg=self._theme.sidebar_bg, highlightthickness=0, width=256)
        self._side_canvas.pack(side="left", fill="both", expand=True)
        self._side_scrollbar = tk.Scrollbar(body, command=self._side_canvas.yview)
        self._side_canvas.configure(yscrollcommand=self._side_scrollbar.set)
        panel = tk.Frame(self._side_canvas, bg=self._theme.sidebar_bg)
        window = self._side_canvas.create_window((0, 0), window=panel, anchor="nw")
        panel.bind("<Configure>", self._on_side_panel_configure)
        self._side_canvas.bind("<Configure>", lambda e: self._side_canvas.itemconfigure(window, width=e.width))
        for widget in (self._side_canvas, panel):
            widget.bind("<MouseWheel>", self._on_sidebar_wheel)
            widget.bind("<Button-4>", self._on_sidebar_wheel)
            widget.bind("<Button-5>", self._on_sidebar_wheel)

        self._sidebar_button(panel, "＋   New chat", self._on_new_chat).pack(
            fill="x", padx=12, pady=(4, 10)
        )

        self._section_label(panel, "Answering with")
        self._source_var = tk.StringVar(value=self.service.source)
        for value, label in (("sources", "From sources (cited)"), ("model", "From trained model")):
            self._radio(panel, label, value).pack(fill="x", padx=12, pady=1)

        self._sidebar_button(panel, "⤓   Save conversation", self._on_save).pack(
            fill="x", padx=12, pady=(10, 12)
        )

        self._section_label(panel, "Actions")
        self._actions_panel = tk.Frame(panel, bg=self._theme.sidebar_bg)
        self._actions_panel.pack(fill="x")
        self._action_buttons = self._populate_actions(self._actions_panel)

        self._section_label(panel, "Saved chats", top=14)
        self._sessions_panel = tk.Frame(panel, bg=self._theme.sidebar_bg)
        self._sessions_panel.pack(fill="x")
        self._session_buttons: list[tk.Widget] = []
        self.refresh_sessions()

        self._section_label(panel, "Appearance", top=14)
        row = tk.Frame(panel, bg=self._theme.sidebar_bg)
        row.pack(fill="x", padx=12, pady=(0, 14))
        right = "light mode  ☀" if self._theme.name == "dark" else "dark mode  ☾"
        self._sidebar_button(row, right, self._toggle_theme, small=True).pack(
            fill="x", pady=(0, 4)
        )
        tk.Label(
            row, text="Theme is saved for next launch.", bg=self._theme.sidebar_bg,
            fg=self._theme.faint, font=(self._font, 8), anchor="w", padx=12,
        ).pack(fill="x")

    def _on_side_panel_configure(self, _event: tk.Event) -> None:
        self._side_canvas.configure(scrollregion=self._side_canvas.bbox("all"))
        self._toggle_sidebar_scrollbar()

    def _toggle_sidebar_scrollbar(self) -> None:
        """Show the sidebar scrollbar only when its content overflows."""
        first, last = self._side_canvas.yview()
        if first <= 0.0 and last >= 1.0:
            self._side_scrollbar.pack_forget()
        else:
            self._side_scrollbar.pack(side="right", fill="y")

    def _on_sidebar_wheel(self, event: tk.Event) -> str:
        delta = -1 * (event.delta // 120) if getattr(event, "delta", 0) else (
            -1 if getattr(event, "num", 0) == 4 else 1
        )
        self._side_canvas.yview_scroll(delta, "units")
        self._toggle_sidebar_scrollbar()
        return "break"

    def _toggle_theme(self) -> str:
        """Switch between dark and light, persist the choice, and rebuild."""
        self._theme = LIGHT_THEME if self._theme.name == "dark" else DARK_THEME
        save_theme_name(self._theme.name)
        self._rebuild_preserving_transcript()
        return "break"

    def _rebuild_preserving_transcript(self) -> None:
        transcript = self._snapshot_transcript()
        self._build_ui()
        for role, text in transcript:
            self._add_bubble(role, text)
        self._scroll_to_bottom()

    def _snapshot_transcript(self) -> list[tuple[str, str]]:
        """The visible messages as (role, text), so a rebuild can restore them."""
        return [
            (self._bubble_roles.get(widget, "assistant"), widget.get("1.0", "end-1c"))
            for widget in self._bubbles
        ]

    def _brand(self, sidebar: tk.Frame) -> None:
        header = tk.Frame(sidebar, bg=self._theme.sidebar_bg)
        header.pack(fill="x", padx=18, pady=(18, 6))
        dot = tk.Canvas(header, width=28, height=28, bg=self._theme.sidebar_bg, highlightthickness=0)
        dot.create_oval(2, 2, 26, 26, fill=self._theme.accent, outline="")
        dot.create_text(14, 15, text="ف", fill=self._theme.accent_text, font=(self._font, 12, "bold"))
        dot.pack(side="left")
        titles = tk.Frame(header, bg=self._theme.sidebar_bg)
        titles.pack(side="left", padx=(10, 0))
        tk.Label(
            titles, text="Fayd Al-Ruh", bg=self._theme.sidebar_bg, fg=self._theme.text,
            font=(self._font, 14, "bold"), anchor="w",
        ).pack(anchor="w")
        tk.Label(
            titles, text=f"v{__version__}  ·  petroleum mind", bg=self._theme.sidebar_bg,
            fg=self._theme.faint, font=(self._font, 8), anchor="w",
        ).pack(anchor="w")

    def _section_label(self, parent: tk.Misc, text: str, *, top: int = 6) -> tk.Label:
        label = tk.Label(
            parent, text=text.upper(), bg=self._theme.sidebar_bg, fg=self._theme.faint,
            font=(self._font, 8, "bold"), anchor="w", padx=18,
        )
        label.pack(fill="x", pady=(top, 4))
        return label

    def _radio(self, parent: tk.Misc, label: str, value: str) -> tk.Radiobutton:
        return tk.Radiobutton(
            parent, text=label, value=value, variable=self._source_var,
            command=self._on_source_change, bg=self._theme.sidebar_bg, fg=self._theme.text,
            selectcolor=self._theme.sidebar_bg, activebackground=self._theme.sidebar_bg,
            activeforeground=self._theme.text, font=(self._font, 10),
            anchor="w", padx=6, highlightthickness=0, bd=0, cursor="hand2",
        )

    def _populate_actions(self, parent: tk.Misc) -> list[tk.Widget]:
        """Fill the Actions section with one button per entry (no duplicates)."""
        buttons: list[tk.Widget] = []
        for action in ACTIONS:
            button = self._action_button(parent, action)
            button.pack(fill="x", padx=12, pady=1)
            buttons.append(button)
        return buttons

    def refresh_sessions(self) -> None:
        """Rebuild the Saved chats list from the store, marking the current one."""
        for widget in self._session_buttons:
            widget.destroy()
        self._session_buttons = []
        try:
            saved = self.service.list_sessions()
        except Exception:  # a broken store must not blank the sidebar
            saved = ()
        items = saved_session_items(saved, current=self.service.session)
        if not items:
            hint = tk.Label(
                self._sessions_panel, text="No saved chats yet.", bg=self._theme.sidebar_bg,
                fg=self._theme.faint, font=(self._font, 9), anchor="w", padx=18, pady=2,
            )
            hint.pack(fill="x")
            self._session_buttons.append(hint)
            return
        for item in items:
            button = self._session_button(self._sessions_panel, item)
            button.pack(fill="x", padx=12, pady=1)
            self._session_buttons.append(button)

    def _session_button(self, parent: tk.Misc, item: SessionItem) -> tk.Label:
        """A sidebar button that resumes one saved chat."""
        return self._sidebar_button(
            parent, item.label, lambda: self._on_resume_session(item), small=True,
            active=item.is_current,
        )

    def _action_button(self, parent: tk.Misc, action: GuiAction) -> tk.Label:
        """A sidebar button bound to one action (avoids a late-bound lambda)."""
        return self._sidebar_button(
            parent, action.label, lambda: self._on_action(action), small=True
        )

    def _sidebar_button(
        self,
        parent: tk.Misc,
        text: str,
        command: Callable[[], Any],
        *,
        small: bool = False,
        active: bool = False,
    ) -> tk.Label:
        """A flat, hoverable label that behaves like a button."""
        size = 9 if small else 11
        base = self._theme.sidebar_active if active else self._theme.sidebar_bg
        button = tk.Label(
            parent, text=text, bg=base, fg=self._theme.text,
            font=(self._font, size), anchor="w", padx=12,
            pady=(4 if small else 9),
            justify="left", wraplength=224 if small else 0, cursor="hand2",
        )
        button.bind("<Enter>", lambda _e: button.configure(bg=self._theme.sidebar_hover))
        button.bind("<Leave>", lambda _e: button.configure(bg=base))
        button.bind("<Button-1>", lambda _e: command())
        return button

    def _build_main(self) -> None:
        main = tk.Frame(self.root, bg=self._theme.main_bg)
        main.pack(side="left", fill="both", expand=True)

        self._build_toolbar(main)

        # Scrollable transcript.
        container = tk.Frame(main, bg=self._theme.main_bg)
        container.pack(fill="both", expand=True)
        self._canvas = tk.Canvas(container, bg=self._theme.main_bg, highlightthickness=0)
        self._canvas.pack(side="left", fill="both", expand=True)
        self._scrollbar = tk.Scrollbar(container, command=self._canvas.yview)
        self._canvas.configure(yscrollcommand=self._scrollbar.set)

        self._transcript = tk.Frame(self._canvas, bg=self._theme.main_bg)
        self._window = self._canvas.create_window((0, 0), window=self._transcript, anchor="nw")
        self._transcript.bind("<Configure>", self._on_transcript_configure)
        self._canvas.bind("<Configure>", self._on_canvas_configure)
        for widget in (self._canvas, self._transcript):
            widget.bind("<MouseWheel>", self._on_mousewheel)
            widget.bind("<Button-4>", self._on_mousewheel)
            widget.bind("<Button-5>", self._on_mousewheel)

        self._build_composer(main)

        # A friendly empty state, shown until the first message arrives.
        self._empty: tk.Widget | None = None

    def _show_hero(self) -> None:
        """The welcome/empty state, shown whenever the transcript is empty."""
        if self._empty is not None:
            return
        self._empty = tk.Frame(self._transcript, bg=self._theme.main_bg)
        self._empty.pack(pady=(64, 0), padx=24)
        tk.Label(
            self._empty, text="ف", bg=self._theme.main_bg, fg=self._theme.accent,
            font=(self._font, 34, "bold"),
        ).pack()
        tk.Label(
            self._empty, text="Ask anything about petroleum engineering",
            bg=self._theme.main_bg, fg=self._theme.text, font=(self._font, 18, "bold"),
        ).pack(pady=(8, 0))
        tk.Label(
            self._empty,
            text="Reservoir rock properties, drilling, production, petrophysics …",
            bg=self._theme.main_bg, fg=self._theme.muted, font=(self._font, 11),
        ).pack(pady=(6, 0))
        tk.Label(
            self._empty,
            text="“From sources” gives cited answers · “From model” shows what the trained model learned",
            bg=self._theme.main_bg, fg=self._theme.faint, font=(self._font, 9), justify="center",
            wraplength=560,
        ).pack(pady=(14, 0))

    def _build_toolbar(self, main: tk.Frame) -> None:
        bar = tk.Frame(main, bg=self._theme.toolbar_bg)
        bar.pack(fill="x")
        inner = tk.Frame(bar, bg=self._theme.toolbar_bg)
        inner.pack(fill="x", padx=18, pady=10)
        tk.Label(
            inner, text="Chat", bg=self._theme.toolbar_bg, fg=self._theme.text,
            font=(self._font, 13, "bold"), anchor="w",
        ).pack(side="left")
        self._mode_label = tk.Label(
            inner, text="", bg=self._theme.toolbar_bg, fg=self._theme.muted,
            font=(self._font, 10), anchor="e",
        )
        self._mode_label.pack(side="right")
        tk.Frame(main, bg=self._theme.border, height=1).pack(fill="x")

    def _build_composer(self, main: tk.Frame) -> None:
        composer = tk.Frame(main, bg=self._theme.main_bg)
        composer.pack(fill="x", padx=18, pady=(10, 16))

        # Files attached to this conversation appear as removable chips above the
        # input, like a modern chat composer.
        self._attachments_row = tk.Frame(composer, bg=self._theme.main_bg)
        self._attachment_chips: list[tk.Widget] = []
        self._attachments_row.pack(side="top", fill="x")

        field = tk.Frame(composer, bg=self._theme.card_bg)
        field.pack(side="left", fill="both", expand=True)

        # A "+" button beside the input opens a file picker for Word documents.
        self._attach_button = tk.Button(
            field, text="＋", command=self._on_attach_clicked,
            bg=self._theme.card_bg, fg=self._theme.muted,
            activebackground=self._theme.sidebar_hover, activeforeground=self._theme.text,
            font=(self._font, 15, "bold"), relief="flat", borderwidth=0,
            padx=10, pady=0, cursor="hand2",
        )
        self._attach_button.pack(side="left", fill="y")
        self._attach_tip = _Tooltip(
            self._attach_button, "Attach a Word (.docx) or text file", self._theme, self._font
        )

        self._entry = tk.Text(
            field, height=2, bg=self._theme.card_bg, fg=self._theme.text,
            insertbackground=self._theme.accent, font=(self._font, 11),
            relief="flat", wrap="word", padx=14, pady=10,
            highlightthickness=1, highlightbackground=self._theme.border,
            highlightcolor=self._theme.accent, borderwidth=0,
        )
        self._entry.pack(side="left", fill="both", expand=True)
        self._entry.bind("<Return>", self._on_enter)
        self._entry.bind("<Shift-Return>", lambda _e: None)

        column = tk.Frame(composer, bg=self._theme.main_bg)
        column.pack(side="right", fill="y", padx=(8, 0))
        self._send = tk.Button(
            column, text="Send", command=self._on_send, bg=self._theme.accent,
            fg=self._theme.accent_text, activebackground=self._theme.accent_hover, activeforeground=self._theme.accent_text,
            font=(self._font, 11, "bold"), relief="flat", borderwidth=0,
            padx=22, pady=10, cursor="hand2",
        )
        self._send.pack(side="top", fill="x")
        tk.Label(
            column, text="Return · Shift+Return = newline", bg=self._theme.main_bg,
            fg=self._theme.faint, font=(self._font, 8),
        ).pack(side="top", pady=(4, 0))
        self._render_attachment_chips()
        self._entry.focus_set()

    # -------------------------------------------------------------- transcript

    def _add_bubble(self, role: str, text: str, *, extra: str = "") -> tk.Text:
        """Add one message card and return its selectable read-only text widget."""
        row = tk.Frame(self._transcript, bg=self._theme.main_bg)
        row.pack(fill="x", padx=18, pady=5)
        self._hide_empty()

        if role == "user":
            row.pack_configure(anchor="e")
            bg, fg, accent = self._theme.user_bg, self._theme.user_fg, self._theme.user_accent
        elif role == "note":
            bg, fg, accent = self._theme.main_bg, self._theme.faint, self._theme.muted
        else:
            bg, fg, accent = self._theme.card_bg, self._theme.text, self._theme.muted

        card = tk.Frame(row, bg=bg)
        card.pack(side="right" if role == "user" else "left")

        top = tk.Frame(card, bg=bg)
        top.pack(fill="x", padx=14, pady=(8, 0))
        tk.Label(
            top, text=self._role_label(role), bg=bg, fg=accent,
            font=(self._font, 8, "bold"), anchor="w",
        ).pack(side="left")

        # A read-only Text, not a Label: its text can be selected and copied.
        widget = tk.Text(
            card,
            height=1, wrap="word", relief="flat", borderwidth=0,
            highlightthickness=0, bg=bg, fg=fg, font=(self._font, 11),
            padx=14, pady=(2 if role == "note" else 4), cursor="xterm", takefocus=1,
        )
        widget.insert("1.0", text if not extra else f"{text}\n\n{extra}")
        self._make_read_only(widget)
        widget.pack(fill="x")
        for wheel in ("<MouseWheel>", "<Button-4>", "<Button-5>"):
            widget.bind(wheel, self._on_bubble_wheel, add="+")
        self._fit_height(widget)

        footer = tk.Frame(card, bg=bg)
        footer.pack(fill="x", padx=14, pady=(0, 8))
        copy = tk.Label(
            footer, text="⧉ Copy", bg=bg, fg=accent,
            font=(self._font, 8), cursor="hand2",
        )
        copy.pack(side="left")
        copy.bind("<Button-1>", lambda _e: self._copy_widget(widget))

        self._bubbles.append(widget)
        self._bubble_roles[widget] = role
        self._scroll_to_bottom()
        return widget

    @staticmethod
    def _role_label(role: str) -> str:
        if role == "user":
            return "YOU"
        if role == "note":
            return "STATUS"
        return "FAYD AL-RUH"

    def _copy_widget(self, widget: tk.Text) -> None:
        self._copy_text(widget.get("1.0", "end-1c"))

    @staticmethod
    def _make_read_only(widget: tk.Text) -> None:
        """Keep a Text editable-looking but read-only, yet still selectable/copyable.

        Blocking only the mutating keys (rather than ``state="disabled"``) keeps
        selection, Ctrl+C, and the mouse working normally.
        """
        def block(event: tk.Event) -> str:
            if int(event.state) & 0x0004 or int(event.state) & 0x0008:  # Control / Command
                return ""  # let accelerators (copy, select-all) through
            if event.keysym in {"Left", "Right", "Up", "Down", "Home", "End", "Prior", "Next"}:
                return ""  # allow navigation
            return "break"

        for sequence in ("<Key>", "<<Paste>>", "<<Cut>>", "<<Clear>>"):
            widget.bind(sequence, block)

    def _on_bubble_wheel(self, event: tk.Event) -> str:
        """Scroll the transcript when the bubble has nothing of its own to scroll."""
        widget = event.widget
        spans = widget.yview() if isinstance(widget, tk.Text) else (0.0, 1.0)
        if spans[0] <= 0.0 and spans[1] >= 1.0:
            self._on_mousewheel(event)
            return "break"
        return ""

    def _fit_height(self, widget: tk.Text) -> None:
        """Size a read-only Text to its wrapped content, up to a scroll cap."""
        widget.update_idletasks()
        measured = widget.count("1.0", "end-1c", "displaylines")
        lines = int(measured[0]) if measured else 1
        widget.configure(height=min(max(lines, 1), _MAX_BUBBLE_LINES))

    def _bubble_texts(self) -> list[str]:
        return [widget.get("1.0", "end-1c") for widget in self._bubbles]

    def _hide_empty(self) -> None:
        empty = getattr(self, "_empty", None)
        if empty is not None:
            empty.destroy()
            self._empty = None

    def _on_right_click(self, event: tk.Event) -> None:
        try:
            self._clipboard_menu.tk_popup(event.x_root, event.y_root)
        finally:
            self._clipboard_menu.grab_release()

    def _copy_text(self, text: str) -> None:
        if not text:
            return
        self.root.clipboard_clear()
        self.root.clipboard_append(text)
        self.root.update_idletasks()

    def _selected_text(self) -> str:
        """The selection from the focused Text widget, falling back to any selection."""
        widget = self.root.focus_get()
        if isinstance(widget, tk.Text):
            try:
                if widget.tag_ranges("sel"):
                    return widget.get("sel.first", "sel.last")
            except tk.TclError:
                pass
        for candidate in self._bubbles:
            try:
                if candidate.tag_ranges("sel"):
                    return candidate.get("sel.first", "sel.last")
            except tk.TclError:
                continue
        return ""

    def _on_copy(self, _event: tk.Event | None = None) -> str:
        """Copy the focused or transcript selection; never clobber it with nothing."""
        selection = self._selected_text()
        if not selection:
            return ""  # nothing selected: leave the clipboard (and composer) alone
        self._copy_text(selection)
        return "break"

    def _on_copy_conversation(self) -> None:
        self._copy_text("\n\n".join(self._bubble_texts()))

    def _on_select_all(self, _event: tk.Event | None = None) -> str:
        if self.root.focus_get() is self._entry:
            return ""  # keep the composer's own editing untouched
        for widget in self._bubbles:
            widget.tag_add("sel", "1.0", "end-1c")
        return "break"

    def _on_clear_transcript(self, _event: tk.Event | None = None) -> str:
        if self.root.focus_get() is self._entry:
            return ""  # Delete while typing edits the composer, never the transcript
        self._clear_transcript()
        return "break"

    def _clear_transcript(self) -> None:
        for widget in list(self._bubbles):
            self._destroy_bubble(widget)
        self._bubbles.clear()
        self._bubble_roles.clear()
        self._show_hero()

    @staticmethod
    def _destroy_bubble(widget: tk.Text) -> None:
        """Remove a message widget and its wrapping card and row."""
        card = widget.master
        if card is None:
            return
        row = card.master
        (row if row is not None else card).destroy()

    def _on_transcript_configure(self, _event: tk.Event) -> None:
        self._canvas.configure(scrollregion=self._canvas.bbox("all"))
        self._toggle_scrollbar()

    def _on_canvas_configure(self, event: tk.Event) -> None:
        self._canvas.itemconfigure(self._window, width=event.width)

    def _toggle_scrollbar(self) -> None:
        """Show the scrollbar only when the transcript actually overflows."""
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

    # --------------------------------------------------------------- behavior

    def _set_source_label(self) -> None:
        self._mode_label.configure(text=self._source_label_text())

    def _status_text(self) -> str:
        return (
            f"Session  {self.service.session or 'unsaved'}\n"
            "Ctrl+N new · Ctrl+S save · Ctrl+C copy"
        )

    def _on_enter(self, _event: tk.Event) -> str:
        self._on_send()
        return "break"  # keep the newline out of the box

    def _on_send(self) -> None:
        if self._busy:
            return
        message = self._entry.get("1.0", "end").strip()
        if not message:
            return
        self._entry.delete("1.0", "end")
        self._add_bubble("user", message)
        self._show_thinking("Thinking")
        self._start_work(lambda: self.service.ask(message))

    def _on_new_chat(self, _event: tk.Event | None = None) -> str:
        if self._busy:
            return "break"
        conversation_id = self.service.new_conversation()
        self._hide_thinking()
        self._reset_attachments()
        self._add_bubble("note", f"Started a new conversation ({conversation_id}).")
        self._update_status()
        self.refresh_sessions()
        return "break"

    def _on_save(self, _event: tk.Event | None = None) -> str:
        if self._busy:
            return "break"
        self._add_bubble("note", self.service.save())
        self._update_status()
        self.refresh_sessions()
        return "break"

    def _on_resume_session(self, item: SessionItem) -> None:
        if self._busy or item.is_current:
            return
        if self.service.resume(item.session_id):
            self._reset_attachments()
            self._add_bubble("note", f"Resumed conversation {item.session_id}.")
            self._update_status()
            self.refresh_sessions()
        else:
            self._add_bubble("note", f"Could not resume conversation {item.session_id}.")
            self.refresh_sessions()

    # ------------------------------------------------------------ attachments

    def _on_attach_clicked(self) -> None:
        """Open a file picker for a Word document and attach the choice."""
        path = filedialog.askopenfilename(
            parent=self.root,
            title="Attach a document",
            filetypes=[
                ("Word documents", "*.docx *.docm"),
                ("Text files", "*.txt"),
                ("All files", "*.*"),
            ],
        )
        if path:
            self._on_attach(path)

    def _on_attach(self, path: str) -> AttachmentNotice:
        """Read and index a file, then show it in the transcript and as a chip."""
        notice = self.service.attach_file(path)
        if not notice.ok:
            self._add_bubble("note", f"Could not attach {notice.filename}: {notice.message}")
            return notice
        summary = notice.message
        if notice.preview:
            summary = f"{summary}\n\nPreview:\n{notice.preview}"
        self._add_bubble("note", summary)
        self._render_attachment_chips()
        return notice

    def _render_attachment_chips(self) -> None:
        """Rebuild the removable chips for the files attached to this chat."""
        if not hasattr(self, "_attachments_row"):
            return
        for widget in self._attachment_chips:
            widget.destroy()
        self._attachment_chips = []
        attachments = self.service.attachments
        # The row stays packed (an empty frame has no visible height); re-packing
        # it after forgetting would move it behind the input and give it no space.
        self._attachments_row.pack_configure(pady=(0, 6) if attachments else 0)
        for attachment in attachments:
            self._attachment_chips.append(self._attachment_chip(attachment))

    def _attachment_chip(self, attachment: Attachment) -> tk.Widget:
        chip = tk.Frame(self._attachments_row, bg=self._theme.card_bg)
        chip.pack(side="left", padx=(0, 6), pady=2)
        tk.Label(
            chip, text=f"📄 {attachment.filename}", bg=self._theme.card_bg,
            fg=self._theme.text, font=(self._font, 9), padx=8, pady=3,
        ).pack(side="left")
        close = tk.Label(
            chip, text="✕", bg=self._theme.card_bg, fg=self._theme.muted,
            font=(self._font, 9, "bold"), padx=8, pady=3, cursor="hand2",
        )
        close.pack(side="left")
        close.bind("<Enter>", lambda _e: close.configure(fg=self._theme.accent))
        close.bind("<Leave>", lambda _e: close.configure(fg=self._theme.muted))
        close.bind("<Button-1>", lambda _e: self._detach(attachment))
        return chip

    def _detach(self, attachment: Attachment) -> None:
        removed = self.service.remove_attachment(attachment)
        self._render_attachment_chips()
        self._add_bubble(
            "note",
            f"Detached {attachment.filename}."
            if removed
            else f"Detached {attachment.filename} (its text was already gone).",
        )

    def _reset_attachments(self) -> None:
        """Detach every file, used when the conversation changes."""
        self.service.clear_attachments()
        self._render_attachment_chips()

    def _update_status(self) -> None:
        self._status.configure(text=self._status_text())

    def _on_source_change(self) -> None:
        want = self._source_var.get()
        if want == self.service.source:
            return
        try:
            if want == "model":
                self.service.use_model()
            else:
                self.service.use_sources()
        except (FileNotFoundError, OSError, ValueError) as error:
            self._source_var.set(self.service.source)
            self._add_bubble(
                "note",
                f"Could not load the trained model: {error}\n"
                "Train it first with the “Train the language model” action.",
            )
            return
        self._set_source_label()
        self._add_bubble(
            "note",
            "Now answering from the trained model (no sources)."
            if want == "model"
            else "Now answering from sources, with citations.",
        )

    def _on_action(self, action: GuiAction) -> None:
        if self._busy:
            return
        argv = action.argv
        if action.needs_input:
            question = self._prompt_for_question(action.label)
            if not question:
                return
            argv = (*argv, question)
        self._add_bubble("note", f"Running: {action.label} …")
        self._streaming_label = self._add_bubble("note", f"{action.label} — live output:")
        self._start_work(lambda: self._run_action(action, argv))

    def _prompt_for_question(self, title: str) -> str:
        """Ask for the extra argument an action needs; empty means cancelled."""
        return (simpledialog.askstring(title, "Question:", parent=self.root) or "").strip()

    def _run_action(self, action: GuiAction, argv: tuple[str, ...]) -> str:
        # Capture stdout (the report) while streaming stderr -- the live training
        # progress -- into the transcript line by line, so a long run never looks
        # frozen and the operator can watch the time remaining.
        buffer = io.StringIO()
        with contextlib.redirect_stdout(buffer), contextlib.redirect_stderr(LineStream(self._lines)):
            action.command(argv)
        output = buffer.getvalue().strip()
        return output or f"{action.label} finished."

    def _append_progress_line(self, text: str) -> None:
        """Queue one live output line; the pump flushes them at a bounded rate."""
        self._pending.append(text)

    def _flush_progress(self, *, force: bool = False) -> None:
        """Render queued progress lines, coalesced and rate-limited.

        A training loop can emit many lines between pumps; joining them into one
        widget update keeps the window responsive instead of reconfiguring a
        widget and re-scrolling on every single line.
        """
        if not self._pending:
            return
        now = time.monotonic()
        if not force and now - self._last_render < 0.15:
            return
        text = "\n".join(self._pending)
        self._pending.clear()
        self._last_render = now
        if self._streaming_label is None:
            self._add_bubble("note", text)
        else:
            self._streaming_label.insert("end", f"\n{text}")
            self._fit_height(self._streaming_label)
        self._scroll_to_bottom()

    # ------------------------------------------------------- thinking indicator

    def _build_thinking(self) -> None:
        """Build the (initially hidden) Thinking card once, ready to reveal."""
        row = tk.Frame(self._transcript, bg=self._theme.main_bg)
        card = tk.Frame(row, bg=self._theme.card_bg)
        card.pack(side="left")
        self._thinking = tk.Label(
            card, text="Thinking·", bg=self._theme.card_bg, fg=self._theme.muted,
            font=(self._font, 10, "italic"), padx=16, pady=10,
        )
        self._thinking.pack()
        self._thinking_row = row
        self._thinking_base = "Thinking"
        self._thinking_phase = 1

    def _show_thinking(self, label: str) -> None:
        """Reveal the animated indicator so a pending answer never looks frozen."""
        if self._thinking is None or self._thinking_row is None:
            self._build_thinking()
        self._thinking_base = label
        self._thinking_phase = 1
        if self._thinking is not None:
            self._thinking.configure(text=f"{label}·")
        if self._thinking_row is not None:
            self._thinking_row.pack(fill="x", padx=18, pady=5)
        self._hide_empty()
        self._scroll_to_bottom()
        self.root.after(380, self._animate_thinking)

    def _animate_thinking(self) -> None:
        if self._thinking is None or self._thinking_row is None:
            return
        if not self._thinking_row.winfo_ismapped():
            return
        self._thinking_phase = self._thinking_phase % 3 + 1
        self._thinking.configure(text=f"{self._thinking_base}{'·' * self._thinking_phase}")
        self.root.after(380, self._animate_thinking)

    def _hide_thinking(self) -> None:
        if self._thinking_row is not None:
            self._thinking_row.pack_forget()

    # -------------------------------------------------------------- threading

    def _start_work(self, job) -> None:
        """Run ``job`` on a worker thread; its result becomes an assistant bubble."""
        self._busy = True
        self._set_send_enabled(False)
        self._status.configure(text="Working…\nThis can take a moment.")

        def run() -> None:
            try:
                self._results.put(("ok", job()))
            except Exception as error:  # surfaced in the transcript, never crashes Tk
                self._results.put(("error", error))

        threading.Thread(target=run, daemon=True).start()

    def _set_send_enabled(self, enabled: bool) -> None:
        self._send.configure(
            state="normal" if enabled else "disabled",
            bg=self._theme.accent if enabled else self._theme.disabled,
            cursor="hand2" if enabled else "arrow",
        )

    def _pump(self) -> None:
        """Drain worker results and live progress lines on the Tk main thread."""
        self._drain_lines(cap=500)
        self._flush_progress()
        try:
            kind, payload = self._results.get_nowait()
        except queue.Empty:
            kind = None
        if kind is not None:
            # The job has ended, so every line it printed is already queued;
            # drain them all into the same live bubble before showing the result.
            self._drain_lines(cap=None)
            self._flush_progress(force=True)
            self._hide_thinking()
            if kind == "ok":
                self._render_result(payload)
            else:
                self._add_bubble("assistant", f"Something went wrong: {payload}")
            self._busy = False
            self._streaming_label = None
            self._set_send_enabled(True)
            self._update_status()
            self.refresh_sessions()
        self.root.after(100, self._pump)

    def _drain_lines(self, *, cap: int | None) -> None:
        """Move queued live-output lines into the pending buffer, optionally capped."""
        drained = 0
        while cap is None or drained < cap:
            try:
                self._append_progress_line(self._lines.get_nowait())
                drained += 1
            except queue.Empty:
                break

    def _render_result(self, payload: Any) -> None:
        if isinstance(payload, ChatReply):
            self._add_bubble("assistant", payload.text)
            if payload.hint:
                self._add_bubble("note", payload.hint)
        else:
            self._add_bubble("assistant", str(payload))
        self._mode_label.configure(text=self._source_label_text())
        self._entry.focus_set()

    def _source_label_text(self) -> str:
        return (
            "Answering from the trained model (no sources)"
            if self.service.source == "model"
            else "Answering from sources (cited)"
        )


def main(argv: list[str] | None = None) -> int:
    """Launch the GUI. Returns a non-zero code when no display is available."""
    args = build_parser().parse_args(argv)
    try:
        root = tk.Tk()
    except tk.TclError as error:
        print(f"The graphical interface needs a display: {error}")
        print("Use the command line instead: python -m cli chat")
        return 1

    service = ChatService(
        no_web=args.no_web,
        root=Path(args.root) if args.root else None,
        read_write=args.read_write,
        model_checkpoint=args.model_checkpoint,
        session=args.session,
    )
    if args.mind:
        try:
            service.use_model()
        except (FileNotFoundError, OSError, ValueError) as error:
            print(f"Could not load the trained model: {error}")
            return 1

    FaydGUI(root, service=service)
    root.mainloop()
    return 0


__all__ = ["FaydGUI", "main"]
