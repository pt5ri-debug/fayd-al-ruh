"""The Tkinter window: a ChatGPT-style desktop front end for Fayd Al-Ruh.

Tkinter ships with CPython, so the GUI adds no dependency. Long work (an answer
or a sidebar command) runs on a worker thread and posts results back through a
queue; Tk is only ever touched from the main thread, as Tk requires.

The layout follows ChatGPT closely: a collapsible left sidebar (logo, New chat,
search, conversation history grouped by date, tools, settings), a slim header
that names the active **operating mode** and the model selector, a centred
transcript with Markdown-rendered messages, and a rounded composer pinned to the
bottom. Two operating modes are offered -- **AI Mode** (conversation only) and
**Jarvics Mode** (conversation plus sandboxed tools) -- with one obvious switch
and a persisted choice; see :mod:`gui.modes`.

The window is a thin shell over :class:`gui.service.ChatService`; it holds no
answer logic of its own. This module is imported lazily by the CLI, so importing
it never opens a window and never breaks the command-line path.
"""

from __future__ import annotations

import contextlib
import io
import queue
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Any

import tkinter as tk
from tkinter import filedialog
from tkinter import font as tkfont
from tkinter import simpledialog

from core import optional_deps
from core.paths import data_dir
from core.version import __version__
from gui import markdown as md
from gui.actions import ACTIONS, GuiAction
from gui.attachments import Attachment, AttachmentNotice
from gui.modes import (
    MODES,
    HistoryItem,
    ModeSpec,
    group_by_date,
    resolve_mode,
    search_items,
)
from gui.options import build_parser
from gui.service import ChatReply, ChatService

# ------------------------------------------------------- OpenHands design tokens
#
# Colours, radii and spacing mirror the OpenHands chat UI (the OpenHands
# frontend's ``src/tailwind.css``): the cool-grey scale, a gold primary accent,
# and HeroUI's small/large radii. Keeping the same tokens is what makes the
# window read like OpenHands rather than a generic form.

#: The small↔large radius pair OpenHands uses (HeroUI ``radius.small``/``large``).
RADIUS_SMALL = 5
RADIUS_LARGE = 20

#: The send control is a disc, so its side length is a single diameter in pixels.
SEND_BUTTON_SIZE = 36

#: Below this width the sidebar starts hidden (a drawer), like ChatGPT on mobile.
_COMPACT_WIDTH = 900
#: The sidebar's expanded width and its collapsed rail width.
_SIDEBAR_WIDTH = 288
_RAIL_WIDTH = 0


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
    code_bg: str           # fenced/inline code background
    code_fg: str           # fenced code text
    quote_bar: str         # blockquote accent
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
    code_bg="#151A24",            # a touch deeper than the base
    code_fg="#DCE3EE",
    quote_bar="#4B5468",
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
    code_bg="#F0F3F8",
    code_fg="#1B2029",
    quote_bar="#C4CEDD",
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


def _pick_mono() -> str:
    """A monospace font for code, falling back to whatever Tk offers."""
    for name in ("Consolas", "Menlo", "DejaVu Sans Mono", "Courier New", "Courier"):
        if name in tkfont.families():
            return name
    return "TkFixedFont"


def load_theme_name() -> str:
    """The saved theme name, defaulting to dark; never raises on a bad store."""
    from storage.json_store import JsonStore

    try:
        saved = JsonStore(data_dir()).load(_THEME_NAMESPACE, _THEME_KEY)
    except Exception:  # a broken store must not stop the window opening
        return DARK_THEME.name
    name = saved.get("theme") if isinstance(saved, dict) else None
    return name if name in THEMES else DARK_THEME.name


def save_theme_name(name: str) -> None:
    """Persist the chosen theme; a store error is silently ignored."""
    from storage.json_store import JsonStore

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


class CircleButton(tk.Canvas):
    """A circular button: a canvas-painted disc that still behaves like a button.

    Tk's ``Button`` cannot be round, so the send control is a Canvas that draws a
    filled circle and a glyph, and forwards clicks to a command. ``enabled`` (and
    the fill/ink) can be changed later without rebuilding it, which is how the
    window greys the control out while it is busy.
    """

    def __init__(
        self,
        parent: tk.Misc,
        *,
        size: int,
        bg: str,
        fill: str,
        ink: str,
        text: str,
        command: Callable[[], Any],
    ) -> None:
        super().__init__(
            parent, width=size, height=size, bg=bg, highlightthickness=0,
            borderwidth=0, takefocus=0, cursor="hand2",
        )
        self._size = size
        self._fill = fill
        self._ink = ink
        self._command = command
        self._enabled = True
        self._shape = self.create_oval(1, 1, size - 1, size - 1, fill=fill, outline="")
        self._glyph = self.create_text(
            size // 2, size // 2, text=text, fill=ink,
            font=("TkDefaultFont", max(11, size // 3), "bold"),
        )
        self.bind("<Button-1>", self._on_click)

    def _on_click(self, _event: tk.Event) -> None:
        if self._enabled:
            self._command()

    def set_state(self, *, enabled: bool, fill: str, ink: str) -> None:
        """Repaint the disc for an enabled/disabled state."""
        self._enabled = enabled
        self._fill = fill
        self._ink = ink
        self.itemconfigure(self._shape, fill=fill)
        self.itemconfigure(self._glyph, fill=ink)
        self.configure(cursor="hand2" if enabled else "arrow")


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


@dataclass
class _Row:
    """One transcript row and the widgets that make it up."""

    role: str
    widget: tk.Text
    row: tk.Frame
    text: str
    #: The full rendered text (``text`` plus any ``extra``), used to size the bubble.
    body: str = ""


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
        operating_mode: str | None = None,
    ) -> None:
        self.root = root
        self.service = service or ChatService(
            no_web=no_web,
            model_checkpoint=model_checkpoint,
            session=session,
            operating_mode=operating_mode,
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
        self._rows: list[_Row] = []
        self._font = _pick_font()
        self._mono = _pick_mono()
        self._thinking: tk.Label | None = None
        self._thinking_row: tk.Widget | None = None
        self._thinking_phase = 0
        # Sidebar visibility. Wide windows show it; a compact window hides it as
        # a drawer that the header's menu button toggles.
        self._sidebar_visible = True
        self._auto_hidden = False
        self._settings_open = False
        self._action_buttons: list[tk.Widget] = []
        self._retry_text = ""
        self._pending_message: str | None = None
        self._confirm_bar: tk.Frame | None = None
        self._composer_lift = 0
        self._history_query = tk.StringVar()

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
        root.geometry("1180x780")
        root.minsize(420, 520)

        self._build_ui()
        root.bind("<Configure>", self._on_root_configure)

    # ------------------------------------------------------------------ layout

    @property
    def _bubbles(self) -> list[tk.Text]:
        """The message text widgets, in order (kept for tests and callers)."""
        return [entry.widget for entry in self._rows]

    @property
    def _session_buttons(self) -> list[tk.Widget]:
        """The sidebar history widgets, in order (kept for tests and callers)."""
        return self._history_buttons

    def _build_ui(self) -> None:
        """Build (or rebuild) the whole window for the current theme.

        Called again by :meth:`_toggle_theme`; the transcript is snapshotted
        first and restored, so switching appearance keeps every message.
        """
        snapshot = self._snapshot_transcript()
        self._rows = []
        self._pending = []
        self._streaming_label = None
        self._thinking_row = None
        self._thinking = None
        self._confirm_bar = None
        for child in self.root.winfo_children():
            if isinstance(child, tk.Menu):  # the clipboard menu outlives a rebuild
                continue
            child.destroy()
        self.root.configure(bg=self._theme.base)

        shell = tk.Frame(self.root, bg=self._theme.base)
        shell.pack(fill="both", expand=True)
        self._build_sidebar(shell)
        self._build_main(shell)
        self._apply_mode_label()
        for role, text in snapshot:
            self._add_bubble(role, text)
        if not snapshot:
            self._show_hero()
        self._set_send_enabled(not self._busy)
        if not getattr(self, "_pumping", False):
            self._pumping = True
            self._pump()

    def _build_sidebar(self, shell: tk.Frame) -> None:
        """The left sidebar: brand, New chat, search, history, tools, settings."""
        sidebar = tk.Frame(shell, bg=self._theme.surface, width=_SIDEBAR_WIDTH)
        sidebar.pack(side="left", fill="y")
        sidebar.pack_propagate(False)
        self._sidebar = sidebar
        tk.Frame(sidebar, bg=self._theme.border_subtle, width=1).pack(side="right", fill="y")

        self._brand(sidebar)

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

        self._search_entry = tk.Entry(
            sidebar, textvariable=self._history_query, bg=self._theme.surface_raised,
            fg=self._theme.foreground, insertbackground=self._theme.primary,
            relief="flat", font=(self._font, 10), highlightthickness=0,
        )
        self._search_entry.pack(fill="x", padx=12, pady=(0, 8), ipady=6)
        self._search_entry.insert(0, "")
        self._history_query.trace_add("write", lambda *_: self.refresh_history())
        self._search_placeholder()

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

        self._section_label(panel, "Chats", top=2)
        self._history_panel = tk.Frame(panel, bg=self._theme.surface)
        self._history_panel.pack(fill="x")
        self._history_buttons: list[tk.Widget] = []
        self.refresh_history()

        self._section_label(panel, "Tools", top=14)
        self._tools_panel = tk.Frame(panel, bg=self._theme.surface)
        self._tools_panel.pack(fill="x")
        self._render_tools()

        self._section_label(panel, "Settings", top=14)
        settings = tk.Frame(panel, bg=self._theme.surface)
        settings.pack(fill="x")
        self._settings_button = self._sidebar_button(
            settings, "⚙   Settings", self._toggle_settings, small=True
        )
        self._settings_button.pack(fill="x", padx=12, pady=(0, 4))
        self._settings_panel = tk.Frame(panel, bg=self._theme.surface)
        self._settings_panel.pack(fill="x")

        self._section_label(panel, "Appearance", top=10)
        row = tk.Frame(panel, bg=self._theme.surface)
        row.pack(fill="x", padx=12, pady=(0, 14))
        right = "light mode  ☀" if self._theme.name == "dark" else "dark mode  ☾"
        self._sidebar_button(row, right, self._toggle_theme, small=True).pack(
            fill="x", pady=(0, 4)
        )
        tk.Label(
            row, text="Theme and mode are saved for next launch.", bg=self._theme.surface,
            fg=self._theme.text_dim, font=(self._font, 8), anchor="w", padx=12,
        ).pack(fill="x")

        self._status = tk.Label(
            sidebar, text=self._status_text(), bg=self._theme.surface, fg=self._theme.text_dim,
            font=(self._font, 8), anchor="w", justify="left", padx=18, pady=10,
        )
        self._status.pack(side="bottom", fill="x")

        if self._is_compact():
            sidebar.pack_forget()
            self._sidebar_visible = False

    def _search_placeholder(self) -> None:
        """Show a greyed hint in the search box while it is empty."""
        if self._history_query.get():
            return
        self._search_entry.configure(fg=self._theme.text_dim)
        self._search_entry.insert(0, "Search chats")

        def clear(_event: tk.Event) -> None:
            if self._history_query.get() == "Search chats":
                self._search_entry.delete(0, "end")
                self._search_entry.configure(fg=self._theme.foreground)

        def restore(_event: tk.Event) -> None:
            if not self._history_query.get():
                self._search_entry.configure(fg=self._theme.text_dim)
                self._search_entry.insert(0, "Search chats")

        self._search_entry.bind("<FocusIn>", clear)
        self._search_entry.bind("<FocusOut>", restore)

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

    # ------------------------------------------------------------- appearance

    def _toggle_theme(self) -> str:
        """Switch between dark and light, persist the choice, and rebuild."""
        self._theme = LIGHT_THEME if self._theme.name == "dark" else DARK_THEME
        save_theme_name(self._theme.name)
        self._rebuild_preserving_transcript()
        return "break"

    def _rebuild_preserving_transcript(self) -> None:
        transcript = self._snapshot_transcript()
        self._build_ui()
        # _build_ui restores the transcript itself; nothing more to do.
        if transcript and self._rows:
            self._scroll_to_bottom()

    def _snapshot_transcript(self) -> list[tuple[str, str]]:
        """The visible messages as (role, text), so a rebuild can restore them."""
        return [(entry.role, entry.text) for entry in self._rows]

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

    # ----------------------------------------------------------------- history

    def refresh_history(self) -> None:
        """Rebuild the Chats list from the store, grouped by date, with search."""
        panel = getattr(self, "_history_panel", None)
        if panel is None:
            return
        for widget in self._history_buttons:
            widget.destroy()
        self._history_buttons = []
        try:
            items = self.service.saved_session_summaries()
        except Exception:  # a broken store must not blank the sidebar
            items = ()
        query = self._history_query.get()
        if query == "Search chats":
            query = ""
        items = search_items(items, query)
        if not items:
            hint = tk.Label(
                self._history_panel,
                text="No matching chats." if query else "No saved chats yet.",
                bg=self._theme.surface, fg=self._theme.text_dim, font=(self._font, 9),
                anchor="w", padx=18, pady=2,
            )
            hint.pack(fill="x")
            self._history_buttons.append(hint)
            return
        for bucket, group in group_by_date(items):
            header = tk.Label(
                self._history_panel, text=bucket, bg=self._theme.surface,
                fg=self._theme.text_dim, font=(self._font, 8, "bold"), anchor="w", padx=18,
            )
            header.pack(fill="x", pady=(6, 1))
            self._history_buttons.append(header)
            for item in group:
                button = self._history_button(self._history_panel, item)
                button.pack(fill="x", padx=10, pady=1)
                self._history_buttons.append(button)

    def _history_button(self, parent: tk.Misc, item: HistoryItem) -> tk.Label:
        label = f"{item.label}   ✓" if item.is_current else item.label
        return self._sidebar_button(
            parent, label, lambda: self._on_resume_session(item), small=True, active=item.is_current
        )

    def _render_tools(self) -> None:
        """List the sandbox tools in the sidebar; dim them when tools are off."""
        for child in self._tools_panel.winfo_children():
            child.destroy()
        if not self.service.tool_names:
            tk.Label(
                self._tools_panel, text="No tools available.", bg=self._theme.surface,
                fg=self._theme.text_dim, font=(self._font, 9), anchor="w", padx=18, pady=2,
            ).pack(fill="x")
            return
        enabled = self.service.tools_enabled
        colour = self._theme.foreground if enabled else self._theme.text_dim
        for name in self.service.tool_names:
            tk.Label(
                self._tools_panel, text=f"• {name}", bg=self._theme.surface, fg=colour,
                font=(self._font, 9), anchor="w", padx=20, pady=1,
            ).pack(fill="x")
        note = (
            "Available in Jarvics Mode."
            if enabled
            else "Switch to Jarvics Mode to use them."
        )
        tk.Label(
            self._tools_panel, text=note, bg=self._theme.surface, fg=self._theme.text_dim,
            font=(self._font, 8), anchor="w", padx=20, pady=2, wraplength=232, justify="left",
        ).pack(fill="x")

    def _toggle_settings(self) -> None:
        """Expand/collapse the settings panel inside the sidebar."""
        if self._settings_open:
            for child in self._settings_panel.winfo_children():
                child.destroy()
            self._settings_open = False
            return
        self._settings_open = True
        panel = self._settings_panel
        tk.Label(
            panel, text="Answering mouth", bg=self._theme.surface, fg=self._theme.text_secondary,
            font=(self._font, 8, "bold"), anchor="w", padx=18,
        ).pack(fill="x", pady=(6, 2))
        self._source_var = tk.StringVar(value=self.service.source)
        for value, label in (("sources", "From sources (cited)"), ("model", "From trained model")):
            self._radio(panel, label, value).pack(fill="x", padx=14, pady=1)
        self._sidebar_button(
            panel, "⤓   Save conversation", self._on_save, small=True
        ).pack(fill="x", padx=12, pady=(6, 0))
        self._sidebar_button(
            panel, "⌫   Clear transcript", self._clear_transcript, small=True
        ).pack(fill="x", padx=12, pady=(2, 0))
        self._section_label(panel, "Actions", top=8)
        for action in ACTIONS:
            button = self._sidebar_button(
                panel, action.label, partial(self._on_action, action), small=True
            )
            button.pack(fill="x", padx=12, pady=1)
            self._action_buttons.append(button)

    def _radio(self, parent: tk.Misc, label: str, value: str) -> tk.Radiobutton:
        return tk.Radiobutton(
            parent, text=label, value=value, variable=self._source_var,
            command=self._on_source_change, bg=self._theme.surface, fg=self._theme.foreground,
            selectcolor=self._theme.surface, activebackground=self._theme.surface,
            activeforeground=self._theme.foreground, font=(self._font, 10),
            anchor="w", padx=6, highlightthickness=0, bd=0, cursor="hand2",
        )

    # -------------------------------------------------------------------- main

    def _build_main(self, shell: tk.Frame) -> None:
        main = tk.Frame(shell, bg=self._theme.base)
        main.pack(side="left", fill="both", expand=True)
        self._main_frame = main

        self._build_header(main)

        container = tk.Frame(main, bg=self._theme.base)
        container.pack(fill="both", expand=True)
        self._canvas = tk.Canvas(container, bg=self._theme.base, highlightthickness=0)
        self._canvas.pack(side="left", fill="both", expand=True)
        self._scrollbar = tk.Scrollbar(container, command=self._canvas.yview)
        self._canvas.configure(yscrollcommand=self._scrollbar.set)

        self._transcript = tk.Frame(self._canvas, bg=self._theme.base)
        self._row_frames: list[tk.Frame] = []
        self._window = self._canvas.create_window((0, 0), window=self._transcript, anchor="nw")
        self._transcript.bind("<Configure>", self._on_transcript_configure)
        self._canvas.bind("<Configure>", self._on_canvas_configure)
        for widget in (self._canvas, self._transcript):
            widget.bind("<MouseWheel>", self._on_mousewheel)
            widget.bind("<Button-4>", self._on_mousewheel)
            widget.bind("<Button-5>", self._on_mousewheel)

        self._empty: tk.Widget | None = None
        self._build_composer(main)

    def _build_header(self, main: tk.Frame) -> None:
        """The slim top bar: mode switch, model selector, and menu controls."""
        bar = tk.Frame(main, bg=self._theme.base)
        bar.pack(fill="x")
        inner = tk.Frame(bar, bg=self._theme.base)
        inner.pack(fill="x", padx=12, pady=8)

        # Left: the sidebar toggle (also the mobile drawer handle) and the title.
        self._sidebar_toggle = self._icon_button(
            inner, "☰", self._toggle_sidebar, tip="Show or hide the sidebar"
        )
        self._sidebar_toggle.pack(side="left")

        # The mode switch: one obvious control, the active mode shown by name.
        self._mode_var = tk.StringVar(value=self.service.operating_mode.label)
        self._mode_menu = tk.OptionMenu(
            inner, self._mode_var, *[mode.label for mode in MODES], command=self._on_mode_menu,
        )
        self._mode_menu.configure(
            bg=self._theme.surface_raised, fg=self._theme.contrast, activebackground=self._theme.interactive_active,
            activeforeground=self._theme.contrast, relief="flat", highlightthickness=0,
            font=(self._font, 10, "bold"), padx=10, pady=4, bd=0, cursor="hand2",
            indicatoron=True,
        )
        self._mode_menu["menu"].configure(bg=self._theme.surface, fg=self._theme.foreground)
        self._mode_menu.pack(side="left", padx=(8, 0))
        self._mode_label = tk.Label(
            inner, text="", bg=self._theme.base, fg=self._theme.text_secondary,
            font=(self._font, 9), anchor="w",
        )
        self._mode_label.pack(side="left", padx=(10, 0))

        # Right: the model selector (the window's real mouth switch) and theme.
        self._source_var_menu = tk.StringVar(value=self._source_choice_label())
        self._source_menu = tk.OptionMenu(
            inner, self._source_var_menu,
            "From sources (cited)", "From trained model", command=self._on_source_menu,
        )
        self._source_menu.configure(
            bg=self._theme.base, fg=self._theme.foreground, activebackground=self._theme.surface_raised,
            activeforeground=self._theme.contrast, relief="flat", highlightthickness=0,
            font=(self._font, 9), padx=6, pady=3, bd=0, cursor="hand2",
        )
        self._source_menu["menu"].configure(bg=self._theme.surface, fg=self._theme.foreground)
        self._source_menu.pack(side="right")
        self._icon_button(inner, "☾" if self._theme.name == "dark" else "☀", self._toggle_theme, tip="Toggle theme").pack(
            side="right", padx=(0, 6)
        )

        tk.Frame(main, bg=self._theme.border_subtle, height=1).pack(fill="x")

    def _icon_button(self, parent: tk.Misc, glyph: str, command: Callable[[], Any], *, tip: str = "") -> tk.Button:
        button = tk.Button(
            parent, text=glyph, command=command, bg=self._theme.base, fg=self._theme.foreground,
            activebackground=self._theme.surface_raised, activeforeground=self._theme.contrast,
            font=(self._font, 13), relief="flat", borderwidth=0, padx=8, pady=2, cursor="hand2",
        )
        if tip:
            _Tooltip(button, tip, self._theme, self._font)
        return button

    def _column_pad(self) -> int:
        """Side padding that centres the transcript in a readable column."""
        canvas = getattr(self, "_canvas", None)
        width = canvas.winfo_width() if canvas is not None else 0
        if width <= 1:
            return 20
        return max(16, (width - _COLUMN_MAX) // 2)

    def _bubble_fonts(self) -> tuple[tkfont.Font, tkfont.Font]:
        """The proportional and mono fonts a bubble measures its text with."""
        return (
            tkfont.Font(family=self._font, size=11),
            tkfont.Font(family=self._mono, size=10),
        )

    def _available_column(self) -> int:
        """The transcript column's pixel width, capped for readability."""
        canvas = getattr(self, "_canvas", None)
        width = canvas.winfo_width() if canvas is not None else 0
        if width <= 1:
            return _COLUMN_MAX
        return max(160, min(_COLUMN_MAX, width - 2 * self._column_pad()))

    def _bubble_content_width(self, text: str, *, role: str) -> int:
        """The pixel width a bubble's text may occupy: fit the widest line, capped.

        The height is set separately by :meth:`_fit_height`; together the two make a
        bubble grow with the amount of text it holds instead of being a fixed box.
        A user bubble is right-aligned and a little narrower than the column; an
        assistant bubble may use the full column but never overflows it.
        """
        normal, mono = self._bubble_fonts()
        widest = 0
        for line in md.iter_styled_lines(text):
            font = mono if line.in_code else normal
            widest = max(widest, font.measure(line.text))
        if role == "user":
            return max(80, min(widest, int(_COLUMN_MAX * 0.84) - 52))
        pad = 6 if role != "note" else 0
        return max(120, min(widest, self._available_column() - 2 * pad))

    def _bubble_text_width(self, text: str, *, role: str) -> int:
        """The ``Text`` width, in characters, that yields the content width."""
        normal, _mono = self._bubble_fonts()
        char_px = max(1, normal.measure("0"))
        content = self._bubble_content_width(text, role=role)
        return max(1, -(-content // char_px))

    def _bubble_card_width(self, text: str) -> int:
        """The pixel width of a user bubble's rounded card, chrome included."""
        return self._bubble_content_width(text, role="user") + 52

    def _show_hero(self) -> None:
        """The welcome/empty state, shown whenever the transcript is empty."""
        if self._empty is not None:
            return
        self._empty = tk.Frame(self._transcript, bg=self._theme.base)
        self._empty.pack(pady=(64, 0), padx=24)
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
            self._empty, text=self._hero_subtitle(),
            bg=self._theme.base, fg=self._theme.text_secondary, font=(self._font, 11),
            justify="center", wraplength=560,
        ).pack(pady=(10, 0))
        tk.Label(
            self._empty,
            text="Answering mouth: “From sources” cites passages · “From trained model” shows what it learned",
            bg=self._theme.base, fg=self._theme.text_dim, font=(self._font, 9), justify="center",
            wraplength=560,
        ).pack(pady=(16, 0))

    def _hero_subtitle(self) -> str:
        if self.service.tools_enabled:
            return (
                "Jarvics Mode: ask a question, or give a command — list files, read a "
                "file, search, or calculate — run inside the sandbox."
            )
        return "AI Mode: reservoir rock, drilling, production, petrophysics … answers with citations."

    # -------------------------------------------------------------- transcript

    def _add_bubble(self, role: str, text: str, *, extra: str = "") -> tk.Text:
        """Add one message row and return its selectable read-only text widget."""
        pad = self._column_pad()
        row = tk.Frame(self._transcript, bg=self._theme.base)
        row.pack(fill="x", padx=(pad, pad), pady=6)
        self._row_frames.append(row)
        self._hide_empty()

        if role == "user":
            card = RoundedBox(
                row, bg=self._theme.user_bubble, fill=self._theme.user_bubble,
                outline=self._theme.user_bubble, corner=RADIUS_LARGE,
                parent_bg=self._theme.base,
            )
            card.pack(side="right", anchor="e")
            widget = tk.Text(
                card.inner, height=1, wrap="word", relief="flat", borderwidth=0,
                highlightthickness=0, bg=self._theme.user_bubble, fg=self._theme.user_bubble_fg,
                font=(self._font, 11), padx=14, pady=8, cursor="xterm", takefocus=1, width=0,
            )
            widget.insert("1.0", text)
            self._make_read_only(widget)
            widget.pack(fill="both", expand=True)
            card.configure(width=self._bubble_card_width(text))
            self._fit_height(widget)
            body = text
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
                spacing1=1, spacing3=1,
            )
            body = f"{text}\n\n{extra}" if extra else text
            self._insert_rich(widget, body)
            self._make_read_only(widget)
            widget.configure(width=self._bubble_text_width(body, role=role))
            widget.pack(anchor="w")
            self._fit_height(widget)

        for wheel in ("<MouseWheel>", "<Button-4>", "<Button-5>"):
            widget.bind(wheel, self._on_bubble_wheel, add="+")

        footer = tk.Frame(row, bg=self._theme.base)
        footer.pack(fill="x", pady=(2, 0))
        if role != "user":
            self._copy_label(footer, "⧉ Copy", widget)
            retry = tk.Label(
                footer, text="↻ Retry", bg=self._theme.base, fg=self._theme.text_dim,
                font=(self._font, 8), cursor="hand2",
            )
            retry.pack(side="left", padx=(10, 0))
            retry.bind("<Button-1>", lambda _e: self._on_retry())
            retry.bind("<Enter>", lambda _e: retry.configure(fg=self._theme.foreground))
            retry.bind("<Leave>", lambda _e: retry.configure(fg=self._theme.text_dim))
        else:
            self._copy_label(footer, "⧉ Copy", widget)

        self._rows.append(_Row(role=role, widget=widget, row=row, text=text, body=body))
        self._scroll_to_bottom()
        return widget

    def _copy_label(self, footer: tk.Frame, text: str, widget: tk.Text) -> None:
        copy = tk.Label(
            footer, text=text, bg=self._theme.base, fg=self._theme.text_dim,
            font=(self._font, 8), cursor="hand2",
        )
        copy.pack(side="right" if widget.cget("bg") == self._theme.user_bubble else "left")
        copy.bind("<Button-1>", lambda _e: self._copy_widget(widget))
        copy.bind("<Enter>", lambda _e: copy.configure(fg=self._theme.foreground))
        copy.bind("<Leave>", lambda _e: copy.configure(fg=self._theme.text_dim))

    def _insert_rich(self, widget: tk.Text, text: str) -> None:
        """Insert Markdown-styled text into a Text widget, tags included.

        Newlines are inserted *between* lines only, so the widget's content is the
        message exactly -- no trailing newline -- which keeps a copied message
        byte-for-byte identical to what was shown.
        """
        self._configure_tags(widget)
        for index, styled in enumerate(md.iter_styled_lines(text)):
            if index:
                widget.insert("end", "\n")
            self._insert_styled_line(widget, styled)

    def _configure_tags(self, widget: tk.Text) -> None:
        """Create the tags a rendered message uses, on this widget."""
        theme = self._theme
        widget.tag_configure("heading", font=(self._font, 15, "bold"), foreground=theme.contrast,
                             spacing1=6, spacing3=4)
        widget.tag_configure("heading3", font=(self._font, 12, "bold"), foreground=theme.contrast,
                             spacing1=4, spacing3=3)
        widget.tag_configure("bullet", lmargin1=16, lmargin2=30)
        widget.tag_configure("quote", lmargin1=16, lmargin2=16, foreground=theme.text_secondary,
                             background=theme.surface_raised)
        widget.tag_configure("code", font=(self._mono, 10), background=theme.code_bg,
                             foreground=theme.code_fg, lmargin1=10, lmargin2=10, spacing1=2, spacing3=2)
        widget.tag_configure("table", font=(self._mono, 10), foreground=theme.foreground, lmargin1=8)
        widget.tag_configure("rule", foreground=theme.border)
        widget.tag_configure("bold", font=(self._font, 11, "bold"), foreground=theme.contrast)
        widget.tag_configure("inline_code", font=(self._mono, 10), background=theme.code_bg,
                             foreground=theme.code_fg)
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
            self._highlight_code_line(widget, styled.text, start, end)
        elif styled.kind not in (md.CODE, md.RULE):
            self._inline_tags(widget, styled.text, start)

    def _inline_tags(self, widget: tk.Text, line: str, line_start: str) -> None:
        """Add bold / inline-code tags to the recognised spans of one line."""
        row = line_start.split(".")[0]
        for segment in md.parse_inline(line):
            if segment.kind == "text":
                continue
            # parse_inline returns whole-line offsets; map them onto the widget.
            offset = line.index(segment.text) if segment.kind == "bold" else None
            if segment.kind == "bold":
                token = f"**{segment.text}**"
                idx = line.find(token)
                if idx >= 0:
                    widget.tag_add("bold", f"{row}.{idx}", f"{row}.{idx + len(token)}")
            elif offset is not None:
                widget.tag_add("inline_code", f"{row}.{offset}", f"{row}.{offset + len(segment.text)}")
        # Inline code is handled explicitly: mark `x` spans.
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

    def _highlight_code_line(self, widget: tk.Text, line: str, start: str, end: str) -> None:
        """Colour keywords/strings/comments inside a fenced code line."""
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

    def _avatar(self, parent: tk.Misc) -> None:
        """A small gold disc with the brand letter, beside an assistant message."""
        badge = tk.Canvas(parent, width=24, height=24, bg=self._theme.base, highlightthickness=0)
        badge.create_oval(1, 1, 23, 23, fill=self._theme.primary, outline="")
        badge.create_text(12, 13, text="ف", fill=self._theme.primary_foreground, font=(self._font, 10, "bold"))
        badge.pack(side="left")

    def _reflow(self) -> None:
        """Recentre rows and resize bubbles after the window width changes."""
        pad = self._column_pad()
        for row in self._row_frames:
            row.pack_configure(padx=(pad, pad))
        for entry in self._rows:
            if entry.role == "user":
                box = entry.widget.master.master if entry.widget.master is not None else None
                if isinstance(box, RoundedBox):
                    box.configure(width=self._bubble_card_width(entry.body))
                self._fit_height(entry.widget)
            else:
                entry.widget.configure(width=self._bubble_text_width(entry.body, role=entry.role))
                self._fit_height(entry.widget)

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
        """Keep a Text editable-looking but read-only, yet still selectable/copyable."""
        def block(event: tk.Event) -> str:
            if int(event.state) & 0x0004 or int(event.state) & 0x0008:  # Control / Command
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

    def _fit_height(self, widget: tk.Text) -> None:
        """Size a read-only Text to its wrapped content, up to a scroll cap."""
        widget.update_idletasks()
        measured = widget.count("1.0", "end", "displaylines")
        lines = int(measured[0]) if measured else 1
        widget.configure(height=min(max(lines, 1), _MAX_BUBBLE_LINES))

    def _bubble_texts(self) -> list[str]:
        return [entry.text for entry in self._rows]

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
        widget = self.root.focus_get()
        if isinstance(widget, tk.Text):
            try:
                if widget.tag_ranges("sel"):
                    return widget.get("sel.first", "sel.last")
            except tk.TclError:
                pass
        for entry in self._rows:
            try:
                if entry.widget.tag_ranges("sel"):
                    return entry.widget.get("sel.first", "sel.last")
            except tk.TclError:
                continue
        return ""

    def _on_copy(self, _event: tk.Event | None = None) -> str:
        selection = self._selected_text()
        if not selection:
            return ""
        self._copy_text(selection)
        return "break"

    def _on_copy_conversation(self) -> None:
        self._copy_text("\n\n".join(self._bubble_texts()))

    def _on_select_all(self, _event: tk.Event | None = None) -> str:
        if self.root.focus_get() is self._entry:
            return ""
        for entry in self._rows:
            entry.widget.tag_add("sel", "1.0", "end-1c")
        return "break"

    def _on_clear_transcript(self, _event: tk.Event | None = None) -> str:
        if self.root.focus_get() is self._entry:
            return ""
        self._clear_transcript()
        return "break"

    def _clear_transcript(self) -> None:
        for entry in list(self._rows):
            entry.row.destroy()
        self._rows.clear()
        self._row_frames.clear()
        self._show_hero()

    def _on_transcript_configure(self, _event: tk.Event) -> None:
        self._canvas.configure(scrollregion=self._canvas.bbox("all"))
        self._toggle_scrollbar()

    def _on_canvas_configure(self, event: tk.Event) -> None:
        self._canvas.itemconfigure(self._window, width=event.width)
        self._reflow()
        if self._empty is not None:
            self._empty.pack_configure(padx=self._column_pad())

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

    # ------------------------------------------------------------- composer

    def _on_composer_configure(self, _event: tk.Event | None = None) -> None:
        """Lift the composer by half the input box's height.

        The lift is extra bottom padding under the composer, so it moves the
        input box (and everything above it) up by half the box's own height
        without changing the layout that already pins the composer to the bottom.
        """
        box_height = self._composer_box.winfo_height()
        if box_height <= 1:
            return
        lift = box_height // 2
        if lift != self._composer_lift:
            self._composer_lift = lift
            self._composer_wrap.pack_configure(pady=(8, 14 + lift))

    def _build_composer(self, main: tk.Frame) -> None:
        composer = tk.Frame(main, bg=self._theme.base)
        composer.pack(fill="x", padx=self._column_pad(), pady=(8, 14))
        self._composer_wrap = composer
        # Raise the whole composer (input box, chips, buttons) by half the input
        # box's own height, measured once it has a real size.
        composer.bind("<Configure>", self._on_composer_configure)

        self._confirm_bar = tk.Frame(composer, bg=self._theme.surface_raised)
        self._confirm_label = tk.Label(
            self._confirm_bar, text="", bg=self._theme.surface_raised, fg=self._theme.foreground,
            font=(self._font, 9), anchor="w", justify="left", wraplength=_COLUMN_MAX - 200,
        )
        self._confirm_label.pack(side="left", padx=10, pady=6)
        for label, approve in (("Run", True), ("Cancel", False)):
            button = tk.Button(
                self._confirm_bar, text=label,
                command=partial(self._resolve_confirmation, approve),
                bg=self._theme.primary if approve else self._theme.surface,
                fg=self._theme.primary_foreground if approve else self._theme.foreground,
                relief="flat", borderwidth=0, font=(self._font, 9, "bold"), padx=10, pady=2,
                cursor="hand2",
            )
            button.pack(side="right", padx=(0, 6), pady=4)

        self._attachments_row = tk.Frame(composer, bg=self._theme.base)
        self._attachment_chips: list[tk.Widget] = []
        self._attachments_row.pack(side="top", fill="x", pady=(0, 6))

        self._composer_box = RoundedBox(
            composer, bg=self._theme.surface, fill=self._theme.surface,
            outline=self._theme.border, corner=RADIUS_LARGE, parent_bg=self._theme.base,
        )
        self._composer_box.pack(side="top", fill="x")
        field = self._composer_box.inner

        self._attach_button = tk.Button(
            field, text="＋", command=self._on_attach_clicked,
            bg=self._theme.surface, fg=self._theme.text_dim,
            activebackground=self._theme.surface, activeforeground=self._theme.contrast,
            font=(self._font, 16, "bold"), relief="flat", borderwidth=0,
            padx=12, pady=0, cursor="hand2",
        )
        self._attach_button.pack(side="left", fill="y")
        _Tooltip(self._attach_button, "Attach a Word (.docx) or text file", self._theme, self._font)

        self._entry = tk.Text(
            field, height=2, width=1, bg=self._theme.surface, fg=self._theme.contrast,
            insertbackground=self._theme.primary, font=(self._font, 11),
            relief="flat", wrap="word", padx=6, pady=12,
            highlightthickness=0, borderwidth=0,
        )
        self._entry.pack(side="left", fill="both", expand=True)
        self._entry.bind("<Return>", self._on_enter)
        self._entry.bind("<Shift-Return>", lambda _e: None)

        self._send = CircleButton(
            field, size=SEND_BUTTON_SIZE, bg=self._theme.surface,
            fill=self._theme.primary, ink=self._theme.primary_foreground,
            text="↑", command=self._on_send,
        )
        self._send.pack(side="right", padx=(6, 6), pady=6)
        self._render_attachment_chips()
        self._entry.focus_set()

    # ------------------------------------------------------------- behavior

    def _apply_mode_label(self) -> None:
        """Reflect the current operating mode in the header and the composer hint."""
        spec = self.service.operating_mode
        self._mode_var.set(spec.label)
        self._mode_label.configure(text=f"{spec.label} · {spec.description}")
        self._mode_menu.configure(fg=self._theme.contrast)

    def _source_choice_label(self) -> str:
        return "From trained model" if self.service.source == "model" else "From sources (cited)"

    def _status_text(self) -> str:
        return (
            f"Session  {self.service.session or 'unsaved'}\n"
            f"{self.service.operating_mode.label} · Ctrl+N new · Ctrl+S save"
        )

    def _on_mode_menu(self, label: str) -> None:
        """Switch operating mode from the header control; messages are preserved."""
        spec = resolve_mode(label)
        self._set_mode(spec)

    def _set_mode(self, spec: ModeSpec) -> None:
        previous = self.service.operating_mode
        if spec.key == previous.key:
            return
        self.service.set_operating_mode(spec.key)
        self._apply_mode_label()
        self._render_tools()
        self._add_bubble(
            "note",
            f"Switched to {spec.label}. "
            + (
                "Sandboxed tools are now available for this conversation."
                if spec.allows_tools
                else "Answers stay conversational; tools are off."
            ),
        )
        self._update_status()

    def _on_source_menu(self, label: str) -> None:
        want = "model" if label.startswith("From trained") else "sources"
        if want == self.service.source:
            return
        try:
            if want == "model":
                self.service.use_model()
            else:
                self.service.use_sources()
        except (FileNotFoundError, OSError, ValueError) as error:
            self._source_menu_var_reset()
            self._add_bubble(
                "note",
                f"Could not load the trained model: {error}\n"
                "Train it first with the “Train the language model” action.",
            )
            return
        self._source_menu_var_reset()
        self._add_bubble(
            "note",
            "Now answering from the trained model (no sources)."
            if want == "model"
            else "Now answering from sources, with citations.",
        )

    def _source_menu_var_reset(self) -> None:
        """Keep the model selector's shown value in step with the real mouth."""
        self._source_var_menu.set(self._source_choice_label())

    def _on_source_change(self) -> None:
        """The settings-panel radio, mirroring the header's model selector."""
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
        self._source_menu_var_reset()
        self._add_bubble(
            "note",
            "Now answering from the trained model (no sources)."
            if want == "model"
            else "Now answering from sources, with citations.",
        )

    def _on_enter(self, _event: tk.Event) -> str:
        self._on_send()
        return "break"

    def _on_send(self) -> None:
        if self._busy:
            return
        message = self._entry.get("1.0", "end").strip()
        if not message:
            return
        self._entry.delete("1.0", "end")
        self._add_bubble("user", message)

        # In Jarvics Mode a destructive plan asks first; everything else sends now.
        preview = self.service.preview_execution(message) if self.service.tools_enabled else None
        if preview is not None and preview.kind == "tool" and preview.needs_confirmation:
            self._ask_confirmation(message, preview)
            return
        self._dispatch(message)

    def _ask_confirmation(self, message: str, preview) -> None:
        """Show the plan and wait for the user before a mutating tool runs."""
        self._pending_message = message
        lines = "\n".join(f"• {step['tool']}: {step['arguments']}" for step in preview.steps)
        self._confirm_label.configure(
            text=f"This will change files in the sandbox:\n{lines}\nRun it?"
        )
        if self._confirm_bar is not None:
            self._confirm_bar.pack(side="top", fill="x", pady=(0, 6), before=self._attachments_row)
        self._scroll_to_bottom()

    def _resolve_confirmation(self, approve: bool) -> None:
        message = self._pending_message
        self._pending_message = None
        if self._confirm_bar is not None:
            self._confirm_bar.pack_forget()
        if not message:
            return
        if approve:
            self._dispatch(message)
        else:
            self._add_bubble("note", "Cancelled — the tool was not run.")

    def _dispatch(self, message: str) -> None:
        self._retry_text = message
        self._show_thinking("Thinking" if not self.service.tools_enabled else "Working")
        self._start_work(lambda: self.service.ask(message))

    def _on_retry(self) -> None:
        """Re-send the last message; the previous answer stays in the transcript."""
        if self._busy or not self._retry_text:
            return
        self._add_bubble("note", f"Retrying: {self._retry_text}")
        self._dispatch(self._retry_text)

    def _on_new_chat(self, _event: tk.Event | None = None) -> str:
        if self._busy:
            return "break"
        conversation_id = self.service.new_conversation()
        self._hide_thinking()
        self._reset_attachments()
        self._clear_transcript()
        self._add_bubble("note", f"Started a new conversation ({conversation_id}).")
        self._update_status()
        self.refresh_history()
        return "break"

    def _on_save(self, _event: tk.Event | None = None) -> str:
        if self._busy:
            return "break"
        self._add_bubble("note", self.service.save())
        self._update_status()
        self.refresh_history()
        return "break"

    def _on_resume_session(self, item: HistoryItem) -> None:
        if self._busy or item.is_current:
            return
        if self.service.resume(item.session_id):
            self._reset_attachments()
            self._add_bubble("note", f"Resumed conversation {item.session_id}.")
            self._update_status()
            self.refresh_history()
        else:
            self._add_bubble("note", f"Could not resume conversation {item.session_id}.")
            self.refresh_history()

    # ---------------------------------------------------------- sidebar toggle

    def _is_compact(self) -> bool:
        width = self.root.winfo_width()
        if width <= 1:
            return False  # not realised yet; keep the sidebar visible
        return width < _COMPACT_WIDTH

    def _toggle_sidebar(self) -> None:
        """Show or hide the sidebar (a drawer on a compact window)."""
        if self._sidebar_visible:
            self._sidebar.pack_forget()
            self._sidebar_visible = False
            self._auto_hidden = False
        else:
            self._sidebar.pack(side="left", fill="y", before=self._main_frame)
            self._sidebar_visible = True
            self._auto_hidden = False

    def _on_root_configure(self, event: tk.Event) -> None:
        """Collapse the sidebar to a drawer on a compact window, and restore it."""
        if event.widget is not self.root:
            return
        if self._is_compact() and self._sidebar_visible:
            self._sidebar.pack_forget()
            self._sidebar_visible = False
            self._auto_hidden = True
        elif not self._is_compact() and not self._sidebar_visible and getattr(self, "_auto_hidden", False):
            self._sidebar.pack(side="left", fill="y", before=self._main_frame)
            self._sidebar_visible = True
            self._auto_hidden = False

    # ------------------------------------------------------------ attachments

    def _on_attach_clicked(self) -> None:
        """Open the ＋ menu: attach a file to this chat, or open the Import Center.

        Attaching keeps the existing behaviour (the file's text joins this
        conversation's evidence). The Import Center is the *persistent* path: it
        writes into the knowledge base through the ingestion pipeline, so the
        imported documents are available to every future chat.
        """
        menu = self._build_attach_menu()
        try:
            menu.tk_popup(
                self._attach_button.winfo_rootx(),
                self._attach_button.winfo_rooty() + self._attach_button.winfo_height(),
            )
        finally:
            menu.grab_release()

    def _build_attach_menu(self) -> tk.Menu:
        """Build the ＋ menu (kept separate so tests can read its entries)."""
        theme = self._theme
        menu = tk.Menu(
            self.root, tearoff=0, bg=theme.surface, fg=theme.foreground,
            activebackground=theme.interactive_active, activeforeground=theme.contrast,
            font=(self._font, 10), borderwidth=0,
        )
        menu.add_command(label="Attach a file to this chat…", command=self._attach_one_file)
        menu.add_command(
            label="Attach a scanned PDF (OCR)…", command=lambda: self._attach_one_file(ocr=True)
        )
        menu.add_separator()
        ocr_ready = optional_deps.is_available(optional_deps.FEATURE_OCR)
        menu.add_command(
            label="OCR support is installed" if ocr_ready else "Install OCR support (scanned PDFs)…",
            command=self._on_install_ocr,
            state="disabled" if ocr_ready else "normal",
        )
        menu.add_separator()
        menu.add_command(
            label="Open Knowledge Import Center…", command=self._open_import_center
        )
        menu.add_command(
            label="Import a SQL database (SQLite)…", command=self._open_sql_import
        )
        menu.add_command(
            label="Generate an image from text…", command=self._open_image_gen
        )
        menu.add_command(
            label="Edit an image from text…", command=self._open_image_edit
        )
        return menu

    def _attach_one_file(self, *, ocr: bool = False) -> None:
        path = filedialog.askopenfilename(
            parent=self.root,
            title="Attach a scanned PDF (OCR)" if ocr else "Attach a document",
            filetypes=[
                ("Documents", "*.txt *.md *.markdown *.csv *.json *.pdf *.docx *.docm"),
                ("Text files", "*.txt *.md *.markdown"),
                ("PDF", "*.pdf"),
                ("Word documents", "*.docx *.docm"),
                ("All files", "*.*"),
            ],
        )
        if path:
            self._on_attach(path, ocr=ocr)

    def _open_import_center(self) -> None:
        """Open the Knowledge Import Center as a separate window."""
        from gui.import_dialog import ImportCenterDialog

        # Keep a reference so the window is not garbage-collected while open.
        self._import_dialog = ImportCenterDialog(
            self.root,
            self.service,
            theme=self._theme,
            font=self._font,
            mono=self._mono,
            on_close=lambda: setattr(self, "_import_dialog", None),
        )

    def _open_sql_import(self) -> None:
        """Open the SQL Database Import screen as a separate window."""
        from gui.sql_import_dialog import SqlImportDialog

        self._sql_import_dialog = SqlImportDialog(
            self.root,
            self.service,
            theme=self._theme,
            font=self._font,
            mono=self._mono,
            on_close=lambda: setattr(self, "_sql_import_dialog", None),
        )

    def _open_image_gen(self) -> None:
        """Open the text-to-image screen as a separate window."""
        from gui.image_gen_dialog import ImageGenDialog

        self._image_gen_dialog = ImageGenDialog(
            self.root,
            self.service.image_gen,
            theme=self._theme,
            font=self._font,
            mono=self._mono,
            on_close=lambda: setattr(self, "_image_gen_dialog", None),
        )

    def _open_image_edit(self) -> None:
        """Open the image-editing screen as a separate window."""
        from gui.image_edit_dialog import ImageEditDialog

        self._image_edit_dialog = ImageEditDialog(
            self.root,
            self.service.image_edit,
            theme=self._theme,
            font=self._font,
            mono=self._mono,
            on_close=lambda: setattr(self, "_image_edit_dialog", None),
        )

    def _open_ultron(self) -> None:
        """Open Ultron -- the general, advanced local model -- as a chat window."""
        from core.ultron import UltronEngine
        from gui.ultron_dialog import UltronDialog

        if getattr(self, "_ultron_engine", None) is None:
            self._ultron_engine = UltronEngine()
        self._ultron_dialog = UltronDialog(
            self.root,
            self._ultron_engine,
            theme=self._theme,
            font=self._font,
            mono=self._mono,
            on_close=lambda: setattr(self, "_ultron_dialog", None),
        )

    def _on_install_ocr(self) -> None:
        """Install the optional OCR packages from within the window.

        The window used to fail an attach with a pip command the user could not
        run (the project is not on PyPI). This runs the working command on a
        worker thread and reports the outcome, then re-checks the feature so the
        ＋ menu label reflects the new state.
        """
        if self._busy:
            return
        if optional_deps.is_available(optional_deps.FEATURE_OCR):
            self._add_bubble("note", "OCR support is already installed.")
            return
        if optional_deps.install_command(optional_deps.FEATURE_OCR) is None:
            self._add_bubble("note", optional_deps.install_hint(optional_deps.FEATURE_OCR))
            return
        self._add_bubble("note", "Installing OCR support… this can take a moment.")
        self._start_work(lambda: optional_deps.install(optional_deps.FEATURE_OCR))

    def _render_install_result(self, result: "optional_deps.InstallResult") -> None:
        self._add_bubble("note", result.message)
        if result.ok:
            self._add_bubble(
                "note",
                "You can now attach a scanned PDF with “Attach a scanned PDF (OCR)…”.",
            )

    def _on_attach(self, path: str, *, ocr: bool = False) -> AttachmentNotice:
        notice = self.service.attach_file(path, ocr=ocr)
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
        if not hasattr(self, "_attachments_row"):
            return
        for widget in self._attachment_chips:
            widget.destroy()
        self._attachment_chips = []
        attachments = self.service.attachments
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
        self.service.clear_attachments()
        self._render_attachment_chips()

    def _update_status(self) -> None:
        self._status.configure(text=self._status_text())

    # ---------------------------------------------------------------- actions

    def _on_action(self, action: GuiAction) -> None:
        if self._busy:
            return
        # The Import Center has a real window, so the sidebar opens it instead of
        # running the headless prompt (which has no terminal to read from here).
        if action.key == "import-center":
            self._open_import_center()
            return
        # The text-to-image screen likewise has its own window.
        if action.key == "image":
            self._open_image_gen()
            return
        # Ultron is a general chat window of its own, not a headless command.
        if action.key == "ultron":
            self._open_ultron()
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
        return (simpledialog.askstring(title, "Question:", parent=self.root) or "").strip()

    def _run_action(self, action: GuiAction, argv: tuple[str, ...]) -> str:
        buffer = io.StringIO()
        with contextlib.redirect_stdout(buffer), contextlib.redirect_stderr(LineStream(self._lines)):
            action.command(argv)
        output = buffer.getvalue().strip()
        return output or f"{action.label} finished."

    def _append_progress_line(self, text: str) -> None:
        self._pending.append(text)

    def _flush_progress(self, *, force: bool = False) -> None:
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
        self._send.set_state(
            enabled=enabled,
            fill=self._theme.primary if enabled else self._theme.disabled,
            ink=self._theme.primary_foreground if enabled else self._theme.text_dim,
        )

    def _pump(self) -> None:
        self._drain_lines(cap=500)
        self._flush_progress()
        try:
            kind, payload = self._results.get_nowait()
        except queue.Empty:
            kind = None
        if kind is not None:
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
            self.refresh_history()
        self.root.after(100, self._pump)

    def _drain_lines(self, *, cap: int | None) -> None:
        drained = 0
        while cap is None or drained < cap:
            try:
                self._append_progress_line(self._lines.get_nowait())
                drained += 1
            except queue.Empty:
                break

    def _render_result(self, payload: Any) -> None:
        if isinstance(payload, optional_deps.InstallResult):
            self._render_install_result(payload)
        elif isinstance(payload, ChatReply):
            if payload.steps:
                self._add_bubble("assistant", self._render_steps(payload))
            else:
                self._add_bubble("assistant", payload.text)
            if payload.hint:
                self._add_bubble("note", payload.hint)
        else:
            self._add_bubble("assistant", str(payload))
        self._entry.focus_set()

    @staticmethod
    def _render_steps(payload: ChatReply) -> str:
        """Render a tool run's trace so results and errors are both visible."""
        lines: list[str] = []
        for step in payload.steps:
            status = "ok" if step.get("ok") else "error"
            detail = step.get("output") if step.get("ok") else step.get("error")
            lines.append(f"- **{step.get('tool', '?')}** [{status}]: {detail}")
        return "\n".join(lines) if lines else payload.text


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
        operating_mode=args.mode,
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
