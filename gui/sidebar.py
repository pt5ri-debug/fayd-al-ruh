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


__all__ = ["SidebarSection"]
