"""A professional, collapsible sidebar section -- the shared accordion row.

The main window's sidebar used to be one long flat column: a chat list, a voice
list, a settings panel, and then every action as its own row, all stacked with
the same weight. Graphics and design programs do not look like that; they group
related tools under a short, collapsible header so the column stays readable and
nothing has to be scrolled past to reach what matters.

This module is the one place that shape is defined: :class:`SidebarSection` is a
header (an icon, a title, and a chevron) over a body frame the caller fills. The
body is created **once** and only packed/unpacked, so the widgets inside it keep
their state across a collapse -- a voice list or a picker is not rebuilt just
because the user closed the section to get it out of the way.

Nothing here imports :mod:`gui.app`, so the window can build on this module
without a circular import; it uses only the duck-typed ``Theme`` tokens and
:class:`gui.typography.Typography` it is handed, exactly like the shared composer.
"""

from __future__ import annotations

import tkinter as tk
from collections.abc import Callable
from pathlib import Path

from gui.typography import Typography

#: The chevron glyphs for the two states (Tk has no native disclosure triangle).
_CHEVRON_OPEN = "▾"
_CHEVRON_CLOSED = "▸"


class SidebarSection:
    """One collapsible group: a clickable header over a body frame.

    ``key`` is a stable identifier the caller uses to remember the open/closed
    choice; ``title`` and ``icon`` are what the header draws. The body frame is
    built once and shown or hidden with :meth:`toggle`/:meth:`set_open`, so a
    widget inside it is never destroyed by collapsing the section. ``on_toggle``
    is called with ``(key, is_open)`` after every change, so the caller can
    persist it.
    """

    def __init__(
        self,
        parent: tk.Misc,
        *,
        key: str,
        title: str,
        icon: str,
        theme: object,
        typography: Typography,
        open: bool = True,
        on_toggle: Callable[[str, bool], None] | None = None,
    ) -> None:
        self.key = key
        self.title = title
        self.icon = icon
        self._theme = theme
        self._type = typography
        self._on_toggle = on_toggle
        self._open = bool(open)

        # The section owns a frame so the caller can pack it into the scrolling
        # panel and every later section lands below it.
        self.frame = tk.Frame(parent, bg=_token(theme, "surface"))
        self.frame.pack(fill="x")

        # The header: a full-width clickable band (icon + title on the left, the
        # chevron on the right). Tk labels have no button styling, so the band is
        # a frame with two labels and the click is bound on all three.
        self.header = tk.Frame(self.frame, bg=_token(theme, "surface"), cursor="hand2")
        self.header.pack(fill="x", pady=(0, 1))
        self._title_label = tk.Label(
            self.header,
            text=f"{icon}  {title}",
            bg=_token(theme, "surface"),
            fg=_token(theme, "text_secondary"),
            font=typography.font("2xs", weight="bold"),
            anchor="w",
            padx=18,
            pady=6,
            cursor="hand2",
        )
        self._title_label.pack(side="left", fill="x", expand=True)
        self._chevron_label = tk.Label(
            self.header,
            text=_CHEVRON_OPEN if self._open else _CHEVRON_CLOSED,
            bg=_token(theme, "surface"),
            fg=_token(theme, "text_dim"),
            font=typography.font("2xs"),
            padx=12,
            cursor="hand2",
        )
        self._chevron_label.pack(side="right")

        # The body: created once, then packed or unpacked. Its background matches
        # the sidebar surface so a nested panel reads as part of the column.
        self.body = tk.Frame(self.frame, bg=_token(theme, "surface"))
        if self._open:
            self.body.pack(fill="x")

        for widget in (self.header, self._title_label, self._chevron_label):
            widget.bind("<Button-1>", self._on_click)
            widget.bind("<Enter>", self._on_enter)
            widget.bind("<Leave>", self._on_leave)

    # ------------------------------------------------------------------ state

    @property
    def is_open(self) -> bool:
        """Whether the body is currently shown."""
        return self._open

    def toggle(self) -> None:
        """Show or hide the body (the header's click handler)."""
        self.set_open(not self._open)

    def set_open(self, open: bool, *, persist: bool = True) -> None:
        """Show or hide the body.

        ``persist`` decides whether the caller's ``on_toggle`` is told about the
        change: a **user's own click** (the header, through :meth:`toggle`)
        persists, while a *programmatic* open (the window showing a panel because
        the program asked for it) passes ``persist=False`` -- so merely displaying
        a section can never rewrite the layout the user arranged.
        """
        open = bool(open)
        if open == self._open:
            return
        self._open = open
        if open:
            self.body.pack(fill="x")
        else:
            self.body.pack_forget()
        self._chevron_label.configure(
            text=_CHEVRON_OPEN if open else _CHEVRON_CLOSED
        )
        if persist and self._on_toggle is not None:
            self._on_toggle(self.key, open)

    # ------------------------------------------------------------------- hover

    def _on_click(self, _event: tk.Event) -> None:
        self.toggle()

    def _on_enter(self, _event: tk.Event) -> None:
        hover = _token(self._theme, "interactive_hover")
        for widget in (self.header, self._title_label, self._chevron_label):
            widget.configure(bg=hover)

    def _on_leave(self, _event: tk.Event) -> None:
        base = _token(self._theme, "surface")
        for widget in (self.header, self._title_label, self._chevron_label):
            widget.configure(bg=base)


def _token(theme: object, name: str) -> str:
    """Read a colour token off the duck-typed theme, with a safe fallback."""
    return str(getattr(theme, name, getattr(theme, "surface", "#21252F")))


class SidebarColumn:
    """A dialog's left column: a scrollable stack of collapsible sections.

    Every window in the project except the main one used to draw its left side as
    a flat list of ``SECTION LABEL`` headers over their contents, all with the same
    weight and none of them collapsible. A graphics or design program does not look
    like that; it groups related controls under short headers the user can fold
    away. This class is the one place that shape is built, so every dialog's column
    reads the same and none of them re-implements the canvas, the scrollbar, the
    wheel bindings, or the section bookkeeping.

    ``shell`` is the outer frame the caller grids or packs; ``body`` is the inner
    frame the sections are stacked in; :meth:`add` returns a section's body frame
    to fill. The scrollbar is **always visible** (not only on overflow), so the
    column reads as one scrollable list rather than a dead end on a short window.

    A ``scope`` and ``state_root`` make the open/closed choice survive a restart
    (stored beside the main window's own section state); without them the sections
    simply start at their declared state and are not persisted.
    """

    def __init__(
        self,
        parent: tk.Misc,
        *,
        theme: object,
        typography: Typography,
        width: int,
        scope: str = "",
        state_root: Path | str | None = None,
        grid: bool = True,
    ) -> None:
        self._theme = theme
        self._type = typography
        self._scope = scope
        self._state_root = state_root
        self._states: dict[str, bool] = (
            load_section_states(scope, state_root) if scope else {}
        )
        self.sections: dict[str, SidebarSection] = {}

        self.shell = tk.Frame(parent, bg=_token(theme, "surface"), width=width)
        if grid:
            self.shell.grid(row=0, column=0, sticky="nsew")
        # Propagation is left **on** (only the grid pass is pinned) so the shell
        # grows to fit the canvas *and* the always-visible scrollbar beside it; a
        # fixed-width shell would squeeze the scrollbar out entirely.
        self.shell.grid_propagate(False)

        self.canvas = tk.Canvas(
            self.shell, bg=_token(theme, "surface"), highlightthickness=0, width=width
        )
        self.canvas.pack(side="left", fill="both", expand=True)
        self.scrollbar = tk.Scrollbar(self.shell, command=self.canvas.yview)
        # Always visible, so the column is never a dead end on a short window.
        self.scrollbar.pack(side="right", fill="y")
        self.canvas.configure(yscrollcommand=self.scrollbar.set)

        self.body = tk.Frame(self.canvas, bg=_token(theme, "surface"))
        self._window = self.canvas.create_window((0, 0), window=self.body, anchor="nw")
        self.body.bind(
            "<Configure>", lambda _e: self.canvas.configure(scrollregion=self.canvas.bbox("all"))
        )
        self.canvas.bind("<Configure>", lambda e: self.canvas.itemconfigure(self._window, width=e.width))
        for widget in (self.canvas, self.body):
            widget.bind("<MouseWheel>", self._on_wheel)
            widget.bind("<Button-4>", self._on_wheel)
            widget.bind("<Button-5>", self._on_wheel)

    def add(
        self,
        key: str,
        title: str,
        *,
        icon: str = "",
        open: bool = True,
    ) -> tk.Frame:
        """Add a collapsible section and return its **body** frame to fill."""
        section = SidebarSection(
            self.body,
            key=key,
            title=title,
            icon=icon,
            theme=self._theme,
            typography=self._type,
            open=self._states.get(key, open),
            on_toggle=self._on_toggle if self._scope else None,
        )
        self.sections[key] = section
        return section.body

    def set_open(self, key: str, open: bool = True) -> None:
        """Open (or close) a section programmatically, without persisting it."""
        section = self.sections.get(key)
        if section is not None:
            section.set_open(open, persist=False)

    def _on_toggle(self, key: str, is_open: bool) -> None:
        self._states[key] = is_open
        if self._scope:
            save_section_states(self._scope, self._states, self._state_root)

    def _on_wheel(self, event: tk.Event) -> None:
        delta = -1 if getattr(event, "delta", 0) > 0 else 1
        if getattr(event, "num", None) == 4:
            delta = -1
        elif getattr(event, "num", None) == 5:
            delta = 1
        self.canvas.yview_scroll(delta, "units")


#: The namespace the per-dialog section states are stored under (beside the main
#: window's own ``sidebar_sections`` record).
_SECTIONS_NAMESPACE = "gui"


def load_section_states(scope: str, root: Path | str | None = None) -> dict[str, bool]:
    """The saved open/closed state of one dialog's sidebar sections.

    A missing or broken record reads as empty and a value that is not a real bool
    is dropped, so a corrupt store can never leave a section undrawable.
    """
    from core.paths import data_dir
    from storage.json_store import JsonStore

    try:
        saved = JsonStore(root if root is not None else data_dir()).load(
            _SECTIONS_NAMESPACE, f"sidebar_sections:{scope}"
        )
    except Exception:
        return {}
    raw = saved.get("sections") if isinstance(saved, dict) else None
    if not isinstance(raw, dict):
        return {}
    return {str(key): value for key, value in raw.items() if isinstance(value, bool)}


def save_section_states(scope: str, states: dict[str, bool], root: Path | str | None = None) -> None:
    """Persist one dialog's sidebar sections' open/closed state (best effort)."""
    from core.best_effort import best_effort
    from core.paths import data_dir
    from storage.json_store import JsonStore

    payload = {str(key): bool(value) for key, value in states.items()}
    with best_effort(f"save the {scope} sidebar sections"):
        JsonStore(root if root is not None else data_dir()).save(
            _SECTIONS_NAMESPACE, f"sidebar_sections:{scope}", {"sections": payload}
        )


__all__ = [
    "SidebarColumn",
    "SidebarSection",
    "load_section_states",
    "save_section_states",
]
