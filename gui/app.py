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

# ------------------------------------------------------- OpenHands design tokens
#
# Colours, radii and spacing mirror the OpenHands chat UI (the OpenHands
# frontend's ``src/tailwind.css``): the cool-grey scale, a gold primary accent,
# and HeroUI's small/large radii. Keeping the same tokens is what makes the
# window read like OpenHands rather than a generic form.

#: The small↔large radius pair OpenHands uses (HeroUI ``radius.small``/``large``).
RADIUS_SMALL = 5
RADIUS_LARGE = 20


@dataclass(frozen=True)
class Theme:
    """One appearance: every colour the window draws with.

    Field names follow OpenHands' semantic tokens (``surface``, ``tertiary``,
    ``border_subtle`` …) so the mapping back to the web UI stays obvious.
    """

    name: str
    # Surfaces, from deepest to most raised.
    surface_deep: str      # context menus, deep insets
    base: str              # app shell background
    surface: str           # cards, composer, sidebar
    surface_raised: str    # hover/selected tint inside a panel
    # Text, from most to least prominent.
    contrast: str          # strongest contrast (assistant text, titles)
    foreground: str        # primary readable text
    text_secondary: str    # secondary text
    text_dim: str          # placeholders, very muted labels
    # Lines and state.
    border: str            # standard 1px stroke
    border_subtle: str     # within-panel separators, list dividers
    primary: str           # gold accent — send button, brand, focus
    primary_foreground: str  # text/icon on the primary fill
    interactive_hover: str   # hover on the base surface
    interactive_active: str  # selected/active on the base surface
    user_bubble: str       # user message fill
    user_bubble_fg: str    # user message text
    success: str
    danger: str
    disabled: str


DARK_THEME = Theme(
    name="dark",
    surface_deep="#05070A",     # cool-grey-975
    base="#0B0E14",             # cool-grey-950
    surface="#21252F",          # cool-grey-925
    surface_raised="#2C313F",   # cool-grey-900
    contrast="#F7F9FC",         # cool-grey-50
    foreground="#C3CDDC",       # cool-grey-300
    text_secondary="#A3B0C4",   # cool-grey-400
    text_dim="#7E8A9E",         # cool-grey-500
    border="#4B5468",           # cool-grey-700
    border_subtle="#383F50",    # cool-grey-800
    primary="#C9B974",          # gold
    primary_foreground="#0B0E14",
    interactive_hover="#2C313F",  # cool-grey-900
    interactive_active="#383F50",  # cool-grey-800
    user_bubble="#2C313F",        # cool-grey-900, like OpenHands' bg-tertiary
    user_bubble_fg="#C3CDDC",
    success="#A5E75E",
    danger="#E76A5E",
    disabled="#383F50",
)

LIGHT_THEME = Theme(
    name="light",
    surface_deep="#EEF2F7",     # cool-grey-100
    base="#F7F9FC",             # cool-grey-50
    surface="#FFFFFF",
    surface_raised="#EEF2F7",   # cool-grey-100
    contrast="#0B0E14",         # cool-grey-950
    foreground="#2C313F",       # cool-grey-900
    text_secondary="#4B5468",   # cool-grey-700
    text_dim="#7E8A9E",         # cool-grey-500
    border="#DCE3EE",           # cool-grey-200
    border_subtle="#EEF2F7",    # cool-grey-100
    primary="#8A6E1F",          # darkened gold, AA on the light base
    primary_foreground="#FFFFFF",
    interactive_hover="#EEF2F7",  # cool-grey-100
    interactive_active="#DCE3EE",  # cool-grey-200
    user_bubble="#EEF2F7",        # cool-grey-100
    user_bubble_fg="#0B0E14",
    success="#3F7D1E",
    danger="#C0473C",
    disabled="#DCE3EE",
)

THEMES = {theme.name: theme for theme in (DARK_THEME, LIGHT_THEME)}

#: The window's saved appearance lives in the user's data directory.
_THEME_NAMESPACE = "gui"
_THEME_KEY = "appearance"

# A paragraph-cap for a single message; longer notes scroll inside their card.
_MAX_BUBBLE_LINES = 40

#: The chat column's maximum width, so long lines stay readable and the
#: transcript reads like a centred conversation (OpenHands caps it the same way).
_COLUMN_MAX = 820


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


class RoundedBox(tk.Canvas):
    """A canvas that paints a rounded-rectangle surface, then holds real widgets.

    Tk has no corner radius, so a rounded panel (the composer, a user bubble) is
    a Canvas that draws the rounded fill and border and hosts its children in an
    interior frame placed with :meth:`place`. ``corner`` is the HeroUI radius
    (5 for small controls, 20 for the composer and message cards).
    """

    def __init__(
        self,
        parent: tk.Misc,
        *,
        bg: str,
        fill: str,
        outline: str,
        corner: int = RADIUS_SMALL,
        width_pad: int = 24,
        height_pad: int = 8,
        parent_bg: str | None = None,
    ) -> None:
        super().__init__(
            parent, bg=parent_bg or bg, highlightthickness=0, borderwidth=0,
            takefocus=0,
        )
        self._fill = fill
        self._outline = outline
        self._corner = corner
        self._pad_x = width_pad // 2
        self._pad_y = height_pad // 2
        self._shape: int | None = None
        self.inner = tk.Frame(self, bg=fill)
        # relwidth/height fill the canvas exactly, with the padding as an offset,
        # so the interior can never spill past the rounded background.
        self.inner.place(
            x=self._pad_x, y=self._pad_y, relwidth=1, width=-self._pad_x * 2
        )
        self.inner.bind("<Configure>", self._sync_height)
        self.bind("<Configure>", self._redraw)

    def _sync_height(self, _event: tk.Event | None = None) -> None:
        """Match the canvas height to its content, growing with wrapped text."""
        wanted = self.inner.winfo_reqheight() + self._pad_y * 2
        if wanted > 2 and self.winfo_reqheight() != wanted:
            self.configure(height=wanted)

    def _redraw(self, _event: tk.Event | None = None) -> None:
        self.delete("shape")
        w, h = self.winfo_width(), self.winfo_height()
        if w <= 2 or h <= 2:
            return
        self._shape = self._rounded(1, 1, w - 1, h - 1, self._corner)
        # Draw first, then push it behind the interior frame.
        self.tag_lower(self._shape)

    def set_outline(self, colour: str) -> None:
        """Recolour the border — used to show focus on the composer."""
        self._outline = colour
        if self._shape is not None:
            self.itemconfigure(self._shape, outline=colour)

    def _rounded(self, x1: int, y1: int, x2: int, y2: int, r: int) -> int:
        points = [
            x1 + r, y1, x2 - r, y1, x2, y1, x2, y1 + r,
            x2, y2 - r, x2, y2, x2 - r, y2, x1 + r, y2,
            x1, y2, x1, y2 - r, x1, y1 + r, x1, y1,
        ]
        return self.create_polygon(
            points, smooth=True, splinesteps=24,
            fill=self._fill, outline=self._outline, width=1,
        )


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
            window, text=self._text, bg=self._theme.surface, fg=self._theme.foreground,
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
        self.root.configure(bg=self._theme.base)

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
        sidebar = tk.Frame(self.root, bg=self._theme.surface, width=280)
        sidebar.pack(side="left", fill="y")
        sidebar.pack_propagate(False)
        # A hairline divides the sidebar from the conversation, like OpenHands.
        tk.Frame(sidebar, bg=self._theme.border_subtle, width=1).pack(side="right", fill="y")

        self._brand(sidebar)

        # A filled "New chat" button, the sidebar's one primary action.
        new_chat = tk.Button(
            sidebar, text="＋   New chat", command=self._on_new_chat,
            bg=self._theme.surface_raised, fg=self._theme.contrast,
            activebackground=self._theme.interactive_active,
            activeforeground=self._theme.contrast,
            font=(self._font, 10, "bold"), relief="flat", borderwidth=0,
            anchor="w", padx=14, pady=8, cursor="hand2",
        )
        new_chat.bind("<Enter>", lambda _e: new_chat.configure(bg=self._theme.interactive_active))
        new_chat.bind("<Leave>", lambda _e: new_chat.configure(bg=self._theme.surface_raised))
        new_chat.pack(fill="x", padx=12, pady=(2, 6))

        self._sidebar_button(sidebar, "⤓   Save conversation", self._on_save, small=True).pack(
            fill="x", padx=12, pady=(0, 6)
        )

        # Everything below the header scrolls behind a scrollbar that appears
        # only when it actually overflows.
        body = tk.Frame(sidebar, bg=self._theme.surface)
        body.pack(side="top", fill="both", expand=True)
        self._side_canvas = tk.Canvas(body, bg=self._theme.surface, highlightthickness=0, width=1)
        self._side_canvas.pack(side="left", fill="both", expand=True)
        self._side_scrollbar = tk.Scrollbar(body, command=self._side_canvas.yview)
        self._side_canvas.configure(yscrollcommand=self._side_scrollbar.set)
        panel = tk.Frame(self._side_canvas, bg=self._theme.surface)
        window = self._side_canvas.create_window((0, 0), window=panel, anchor="nw")
        panel.bind("<Configure>", self._on_side_panel_configure)
        self._side_canvas.bind("<Configure>", lambda e: self._side_canvas.itemconfigure(window, width=e.width))
        for widget in (self._side_canvas, panel):
            widget.bind("<MouseWheel>", self._on_sidebar_wheel)
            widget.bind("<Button-4>", self._on_sidebar_wheel)
            widget.bind("<Button-5>", self._on_sidebar_wheel)

        self._section_label(panel, "Conversations", top=8)
        self._sessions_panel = tk.Frame(panel, bg=self._theme.surface)
        self._sessions_panel.pack(fill="x")
        self._session_buttons: list[tk.Widget] = []
        self.refresh_sessions()

        self._section_label(panel, "Answering with", top=14)
        self._source_var = tk.StringVar(value=self.service.source)
        for value, label in (("sources", "From sources (cited)"), ("model", "From trained model")):
            self._radio(panel, label, value).pack(fill="x", padx=14, pady=1)

        self._section_label(panel, "Actions", top=14)
        self._actions_panel = tk.Frame(panel, bg=self._theme.surface)
        self._actions_panel.pack(fill="x")
        self._action_buttons = self._populate_actions(self._actions_panel)

        self._section_label(panel, "Appearance", top=14)
        row = tk.Frame(panel, bg=self._theme.surface)
        row.pack(fill="x", padx=12, pady=(0, 14))
        right = "light mode  ☀" if self._theme.name == "dark" else "dark mode  ☾"
        self._sidebar_button(row, right, self._toggle_theme, small=True).pack(
            fill="x", pady=(0, 4)
        )
        tk.Label(
            row, text="Theme is saved for next launch.", bg=self._theme.surface,
            fg=self._theme.text_dim, font=(self._font, 8), anchor="w", padx=12,
        ).pack(fill="x")

        self._status = tk.Label(
            sidebar, text=self._status_text(), bg=self._theme.surface, fg=self._theme.text_dim,
            font=(self._font, 8), anchor="w", justify="left", padx=18, pady=10,
        )
        self._status.pack(side="bottom", fill="x")

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
        header = tk.Frame(sidebar, bg=self._theme.surface)
        header.pack(fill="x", padx=18, pady=(18, 10))
        dot = tk.Canvas(header, width=30, height=30, bg=self._theme.surface, highlightthickness=0)
        dot.create_oval(1, 1, 29, 29, fill=self._theme.primary, outline="")
        dot.create_text(15, 16, text="ف", fill=self._theme.primary_foreground, font=(self._font, 13, "bold"))
        dot.pack(side="left")
        titles = tk.Frame(header, bg=self._theme.surface)
        titles.pack(side="left", padx=(10, 0))
        tk.Label(
            titles, text="Fayd Al-Ruh", bg=self._theme.surface, fg=self._theme.contrast,
            font=(self._font, 14, "bold"), anchor="w",
        ).pack(anchor="w")
        tk.Label(
            titles, text=f"v{__version__}  ·  petroleum mind", bg=self._theme.surface,
            fg=self._theme.text_dim, font=(self._font, 8), anchor="w",
        ).pack(anchor="w")

    def _section_label(self, parent: tk.Misc, text: str, *, top: int = 6) -> tk.Label:
        label = tk.Label(
            parent, text=text.upper(), bg=self._theme.surface, fg=self._theme.text_dim,
            font=(self._font, 8, "bold"), anchor="w", padx=18,
        )
        label.pack(fill="x", pady=(top, 4))
        return label

    def _radio(self, parent: tk.Misc, label: str, value: str) -> tk.Radiobutton:
        return tk.Radiobutton(
            parent, text=label, value=value, variable=self._source_var,
            command=self._on_source_change, bg=self._theme.surface, fg=self._theme.foreground,
            selectcolor=self._theme.surface, activebackground=self._theme.surface,
            activeforeground=self._theme.foreground, font=(self._font, 10),
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
                self._sessions_panel, text="No saved chats yet.", bg=self._theme.surface,
                fg=self._theme.text_dim, font=(self._font, 9), anchor="w", padx=18, pady=2,
            )
            hint.pack(fill="x")
            self._session_buttons.append(hint)
            return
        for item in items:
            button = self._session_button(self._sessions_panel, item)
            button.pack(fill="x", padx=10, pady=1)
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
        base = self._theme.surface_raised if active else self._theme.surface
        button = tk.Label(
            parent, text=text, bg=base, fg=self._theme.foreground,
            font=(self._font, size), anchor="w", padx=12,
            pady=(4 if small else 9),
            justify="left", wraplength=232 if small else 0, cursor="hand2",
        )
        button.bind("<Enter>", lambda _e: button.configure(bg=self._theme.interactive_hover))
        button.bind("<Leave>", lambda _e: button.configure(bg=base))
        button.bind("<Button-1>", lambda _e: command())
        return button

    def _build_main(self) -> None:
        main = tk.Frame(self.root, bg=self._theme.base)
        main.pack(side="left", fill="both", expand=True)

        self._build_toolbar(main)

        # Scrollable transcript, its rows centred into a readable column.
        container = tk.Frame(main, bg=self._theme.base)
        container.pack(fill="both", expand=True)
        self._canvas = tk.Canvas(container, bg=self._theme.base, highlightthickness=0)
        self._canvas.pack(side="left", fill="both", expand=True)
        self._scrollbar = tk.Scrollbar(container, command=self._canvas.yview)
        self._canvas.configure(yscrollcommand=self._scrollbar.set)

        self._transcript = tk.Frame(self._canvas, bg=self._theme.base)
        self._rows: list[tk.Frame] = []
        self._row_of: dict[tk.Text, tk.Frame] = {}
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

    def _column_pad(self) -> int:
        """Side padding that centres the transcript in a readable column."""
        canvas = getattr(self, "_canvas", None)
        width = canvas.winfo_width() if canvas is not None else 0
        if width <= 1:
            return 20
        return max(20, (width - _COLUMN_MAX) // 2)

    def _user_width(self, text: str) -> int:
        """A width for a right-aligned bubble: fit the longest line, capped."""
        font = tkfont.Font(family=self._font, size=12)
        widest = max((font.measure(line) for line in text.split("\n")), default=0)
        return max(120, min(widest + 40, int(_COLUMN_MAX * 0.84)))

    def _show_hero(self) -> None:
        """The welcome/empty state, shown whenever the transcript is empty."""
        if self._empty is not None:
            return
        self._empty = tk.Frame(self._transcript, bg=self._theme.base)
        self._empty.pack(pady=(72, 0), padx=24)
        badge = tk.Canvas(
            self._empty, width=64, height=64, bg=self._theme.base, highlightthickness=0
        )
        badge.create_oval(1, 1, 63, 63, fill=self._theme.primary, outline="")
        badge.create_text(32, 34, text="ف", fill=self._theme.primary_foreground, font=(self._font, 26, "bold"))
        badge.pack()
        tk.Label(
            self._empty, text="Fayd Al-Ruh",
            bg=self._theme.base, fg=self._theme.contrast, font=(self._font, 24, "bold"),
        ).pack(pady=(14, 0))
        tk.Label(
            self._empty, text="Ask anything about petroleum engineering",
            bg=self._theme.base, fg=self._theme.foreground, font=(self._font, 14),
        ).pack(pady=(6, 0))
        tk.Label(
            self._empty,
            text="Reservoir rock properties, drilling, production, petrophysics …",
            bg=self._theme.base, fg=self._theme.text_secondary, font=(self._font, 11),
        ).pack(pady=(10, 0))
        tk.Label(
            self._empty,
            text="“From sources” gives cited answers · “From model” shows what the trained model learned",
            bg=self._theme.base, fg=self._theme.text_dim, font=(self._font, 9), justify="center",
            wraplength=560,
        ).pack(pady=(16, 0))

    def _build_toolbar(self, main: tk.Frame) -> None:
        bar = tk.Frame(main, bg=self._theme.base)
        bar.pack(fill="x")
        inner = tk.Frame(bar, bg=self._theme.base)
        inner.pack(fill="x", padx=18, pady=9)
        tk.Label(
            inner, text="Chat", bg=self._theme.base, fg=self._theme.contrast,
            font=(self._font, 13, "bold"), anchor="w",
        ).pack(side="left")
        self._mode_label = tk.Label(
            inner, text="", bg=self._theme.base, fg=self._theme.text_secondary,
            font=(self._font, 10), anchor="e",
        )
        self._mode_label.pack(side="right")
        tk.Frame(main, bg=self._theme.border_subtle, height=1).pack(fill="x")

    def _build_composer(self, main: tk.Frame) -> None:
        composer = tk.Frame(main, bg=self._theme.base)
        composer.pack(fill="x", padx=self._column_pad(), pady=(12, 16))

        # Files attached to this conversation appear as removable chips above the
        # input, like a modern chat composer.
        self._attachments_row = tk.Frame(composer, bg=self._theme.base)
        self._attachment_chips: list[tk.Widget] = []
        self._attachments_row.pack(side="top", fill="x", pady=(0, 6))

        # The composer is a rounded surface card, matching OpenHands' 15px panel.
        self._composer_box = RoundedBox(
            composer, bg=self._theme.surface, fill=self._theme.surface,
            outline=self._theme.border, corner=RADIUS_LARGE, parent_bg=self._theme.base,
        )
        self._composer_box.pack(side="top", fill="x")
        field = self._composer_box.inner

        # A "+" button beside the input opens a file picker for Word documents.
        self._attach_button = tk.Button(
            field, text="＋", command=self._on_attach_clicked,
            bg=self._theme.surface, fg=self._theme.text_dim,
            activebackground=self._theme.surface, activeforeground=self._theme.contrast,
            font=(self._font, 16, "bold"), relief="flat", borderwidth=0,
            padx=12, pady=0, cursor="hand2",
        )
        self._attach_button.pack(side="left", fill="y")
        self._attach_tip = _Tooltip(
            self._attach_button, "Attach a Word (.docx) or text file", self._theme, self._font
        )

        self._entry = tk.Text(
            field, height=2, width=1, bg=self._theme.surface, fg=self._theme.contrast,
            insertbackground=self._theme.primary, font=(self._font, 11),
            relief="flat", wrap="word", padx=6, pady=12,
            highlightthickness=0, borderwidth=0,
        )
        self._entry.pack(side="left", fill="both", expand=True)
        self._entry.bind("<Return>", self._on_enter)
        self._entry.bind("<Shift-Return>", lambda _e: None)
        self._entry.bind("<FocusIn>", lambda _e: self._composer_box.set_outline(self._theme.primary))
        self._entry.bind("<FocusOut>", lambda _e: self._composer_box.set_outline(self._theme.border))

        # A circular send button, like OpenHands' size-8 round control.
        self._send = tk.Button(
            field, text="↑", command=self._on_send, bg=self._theme.primary,
            fg=self._theme.primary_foreground, activebackground=self._theme.primary,
            activeforeground=self._theme.primary_foreground,
            font=(self._font, 14, "bold"), relief="flat", borderwidth=0,
            padx=0, pady=2, width=2, cursor="hand2",
        )
        self._send.pack(side="right", padx=(6, 2), pady=8)
        self._render_attachment_chips()
        self._entry.focus_set()

    # -------------------------------------------------------------- transcript

    def _add_bubble(self, role: str, text: str, *, extra: str = "") -> tk.Text:
        """Add one message row and return its selectable read-only text widget.

        User messages are a small rounded bubble on the right; assistant and note
        messages are full-width rows in the centred column, the assistant's with a
        gold avatar, mirroring the OpenHands transcript.
        """
        pad = self._column_pad()
        row = tk.Frame(self._transcript, bg=self._theme.base)
        row.pack(fill="x", padx=(pad, pad), pady=6)
        self._rows.append(row)
        self._hide_empty()

        if role == "user":
            card = RoundedBox(
                row, bg=self._theme.user_bubble, fill=self._theme.user_bubble,
                outline=self._theme.user_bubble, corner=RADIUS_LARGE,
                parent_bg=self._theme.base,
            )
            card.pack(side="right", anchor="e")
            inner = card.inner
            widget = tk.Text(
                inner, height=1, wrap="word", relief="flat", borderwidth=0,
                highlightthickness=0, bg=self._theme.user_bubble, fg=self._theme.user_bubble_fg,
                font=(self._font, 11), padx=14, pady=8, cursor="xterm", takefocus=1,
                width=0,
            )
            widget.insert("1.0", text)
            self._make_read_only(widget)
            widget.pack(fill="both", expand=True)
            self._fit_height(widget)
            card.configure(width=self._user_width(text))
        else:
            bg = self._theme.base
            head = tk.Frame(row, bg=bg)
            head.pack(fill="x", pady=(0, 4))
            if role == "note":
                accent, fg = self._theme.text_dim, self._theme.text_secondary
            else:
                self._avatar(head)
                accent, fg = self._theme.text_dim, self._theme.foreground
            tk.Label(
                head, text=self._role_label(role), bg=bg, fg=accent,
                font=(self._font, 9, "bold"), anchor="w",
            ).pack(side="left", padx=(6 if role != "note" else 0, 0))

            widget = tk.Text(
                row, height=1, wrap="word", relief="flat", borderwidth=0,
                highlightthickness=0, bg=bg, fg=fg, font=(self._font, 11),
                padx=(6 if role != "note" else 0), pady=0, cursor="xterm", takefocus=1,
            )
            widget.insert("1.0", text if not extra else f"{text}\n\n{extra}")
            self._make_read_only(widget)
            widget.pack(fill="x")
            self._fit_height(widget)

        for wheel in ("<MouseWheel>", "<Button-4>", "<Button-5>"):
            widget.bind(wheel, self._on_bubble_wheel, add="+")

        footer = tk.Frame(row, bg=self._theme.base)
        footer.pack(fill="x", pady=(2, 0))
        copy = tk.Label(
            footer, text="⧉ Copy", bg=self._theme.base, fg=self._theme.text_dim,
            font=(self._font, 8), cursor="hand2",
        )
        copy.pack(side="right" if role == "user" else "left")
        copy.bind("<Button-1>", lambda _e: self._copy_widget(widget))
        copy.bind("<Enter>", lambda _e: copy.configure(fg=self._theme.foreground))
        copy.bind("<Leave>", lambda _e: copy.configure(fg=self._theme.text_dim))

        self._bubbles.append(widget)
        self._bubble_roles[widget] = role
        self._row_of[widget] = row
        self._scroll_to_bottom()
        return widget

    def _avatar(self, parent: tk.Misc) -> None:
        """A small gold disc with the brand letter, beside an assistant message."""
        badge = tk.Canvas(parent, width=24, height=24, bg=self._theme.base, highlightthickness=0)
        badge.create_oval(1, 1, 23, 23, fill=self._theme.primary, outline="")
        badge.create_text(12, 13, text="ف", fill=self._theme.primary_foreground, font=(self._font, 10, "bold"))
        badge.pack(side="left")

    def _reflow(self) -> None:
        """Recentre rows and resize bubbles after the window width changes."""
        pad = self._column_pad()
        for row in self._rows:
            row.pack_configure(padx=(pad, pad))
        for widget in self._bubbles:
            if self._bubble_roles.get(widget) == "user":
                self._fit_height(widget)
                box = widget.master.master if widget.master is not None else None
                if isinstance(box, RoundedBox):
                    box.configure(width=self._user_width(widget.get("1.0", "end-1c")))

    @staticmethod
    def _role_label(role: str) -> str:
        if role == "user":
            return "You"
        if role == "note":
            return "Status"
        return "Fayd Al-Ruh"

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
        self._rows.clear()
        self._row_of.clear()
        self._show_hero()

    def _destroy_bubble(self, widget: tk.Text) -> None:
        """Remove a message widget and its wrapping row."""
        row: tk.Misc | None = self._row_of.pop(widget, None)
        if row is None:
            row = widget.master
        if row in self._rows:
            self._rows.remove(row)
        if row is not None:
            row.destroy()

    def _on_transcript_configure(self, _event: tk.Event) -> None:
        self._canvas.configure(scrollregion=self._canvas.bbox("all"))
        self._toggle_scrollbar()

    def _on_canvas_configure(self, event: tk.Event) -> None:
        self._canvas.itemconfigure(self._window, width=event.width)
        self._reflow()
        if self._empty is not None:
            self._empty.pack_configure(padx=self._column_pad())

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
        chip = tk.Frame(self._attachments_row, bg=self._theme.surface_raised)
        chip.pack(side="left", padx=(0, 6), pady=2)
        tk.Label(
            chip, text=f"📄 {attachment.filename}", bg=self._theme.surface_raised,
            fg=self._theme.foreground, font=(self._font, 9), padx=8, pady=3,
        ).pack(side="left")
        close = tk.Label(
            chip, text="✕", bg=self._theme.surface_raised, fg=self._theme.text_dim,
            font=(self._font, 9, "bold"), padx=8, pady=3, cursor="hand2",
        )
        close.pack(side="left")
        close.bind("<Enter>", lambda _e: close.configure(fg=self._theme.primary))
        close.bind("<Leave>", lambda _e: close.configure(fg=self._theme.text_dim))
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
        """Build the (initially hidden) Thinking row once, ready to reveal."""
        row = tk.Frame(self._transcript, bg=self._theme.base)
        self._thinking_row = row
        self._thinking = tk.Label(
            row, text="Thinking·", bg=self._theme.base, fg=self._theme.text_secondary,
            font=(self._font, 10, "italic"),
        )
        self._thinking.pack(side="left")
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
            self._thinking_row.pack(fill="x", padx=(self._column_pad(), self._column_pad()), pady=6)
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
            bg=self._theme.primary if enabled else self._theme.disabled,
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
