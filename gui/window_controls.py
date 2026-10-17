"""Shared minimize / maximize controls for the project's dialog windows.

Every secondary window in the program -- the Code Studio, the Presentation
Studio, the final editor, the Course window, the Import Center, the SQL screen,
and the two image pages -- is a :class:`tkinter.Toplevel` with a slim custom
header drawn in the project's own palette. Tk gives a ``Toplevel`` the window
manager's frame, but on some desktops (and always when the header is the only
chrome the user sees) there is no obvious way to shrink the window to the taskbar
or to grow it to fill the screen. This module adds that, once, so every window
grows the same two controls instead of each one inventing its own.

The buttons are painted with the same theme tokens and type scale the rest of the
program uses, so a window's header reads as one piece. Maximising is honest about
what the platform supports: ``state("zoomed")`` where the window manager
implements it (Windows, some X11 desktops), and a full-screen window otherwise --
exactly the fallback the feature was asked for -- so the control never silently
does nothing.

Tk is imported at module import, so this module is only ever imported by GUI code
that already depends on Tk; ``cli/`` must never import it.
"""

from __future__ import annotations

import tkinter as tk
from collections.abc import Callable
from typing import Any, Literal

#: The glyphs. Chosen from the characters every Tk font on every platform carries,
#: so a control never renders as a missing-glyph box.
_MINIMIZE_GLYPH = "\u2013"  # en dash
_MAXIMIZE_GLYPH = "\u25a1"  # white square
_RESTORE_GLYPH = "\u2750"  # upper right shadowed white square


class WindowControls:
    """The minimize / maximize pair for one window, with an honest toggle state.

    Construct it through :func:`install_window_controls`, which builds and packs
    the buttons into a window's header. The object keeps the small amount of state
    Tk does not expose -- whether *this* code made the window full-screen, so the
    restore path can undo exactly what the maximize path did.
    """

    def __init__(self, window: tk.Toplevel, frame: tk.Frame, maximize_button: tk.Button) -> None:
        self.window = window
        self.frame = frame
        self.maximize_button = maximize_button
        self._fullscreen = False
        self._zoomed = False

    def minimize(self) -> None:
        """Shrink the window to the taskbar / dock."""
        try:
            self.window.iconify()
        except tk.TclError:
            pass

    def maximize(self) -> None:
        """Grow the window: ``state("zoomed")`` where supported, else full-screen."""
        if self._is_open():
            return
        try:
            self.window.state("zoomed")
            self._zoomed = True
        except tk.TclError:
            # The window manager does not implement "zoomed" (many X11 desktops),
            # so fall back to a real full-screen window -- the requested behaviour.
            try:
                self.window.attributes("-fullscreen", True)
                self._fullscreen = True
            except tk.TclError:
                pass
        self._sync_glyph()

    def restore(self) -> None:
        """Undo :meth:`maximize`, whichever way it grew the window."""
        if self._zoomed:
            try:
                self.window.state("normal")
            except tk.TclError:
                pass
            self._zoomed = False
        if self._fullscreen:
            try:
                self.window.attributes("-fullscreen", False)
            except tk.TclError:
                pass
            self._fullscreen = False
        self._sync_glyph()

    def toggle(self) -> None:
        """Maximize when not grown, restore when grown."""
        if self._is_open():
            self.restore()
        else:
            self.maximize()

    def _is_open(self) -> bool:
        return self._zoomed or self._fullscreen

    def _sync_glyph(self) -> None:
        """Point the button's glyph at what the next press will do."""
        try:
            self.maximize_button.configure(
                text=_RESTORE_GLYPH if self._is_open() else _MAXIMIZE_GLYPH
            )
        except tk.TclError:
            pass


def install_window_controls(
    window: tk.Toplevel,
    header: tk.Misc,
    *,
    theme: Any,
    typography: Any,
    on_close: Callable[[], None] | None = None,
    pack_side: Literal["left", "right", "top", "bottom"] = "right",
    padx: int = 4,
) -> WindowControls:
    """Build the minimize / maximize buttons into ``header`` and return them.

    ``header`` is the inner frame of the window's own header (the one the title
    labels live in). The controls are packed as one unit on ``pack_side`` (right by
    default), so they sit with whatever the caller already packs there; pack the
    caller's own buttons *after* calling this to put them further inside, or
    *before* to put them further out.

    ``on_close``, when given, is wired to the window's close protocol as well, so a
    caller that adds the controls to an existing header keeps one close path.
    """
    controls = tk.Frame(header, bg=theme.base)
    controls.pack(side=pack_side, padx=(0, padx))

    button_font = typography.font("sm", weight="bold")
    minimize_button = _control_button(controls, _MINIMIZE_GLYPH, theme, button_font)
    minimize_button.pack(side="left")
    maximize_button = _control_button(controls, _MAXIMIZE_GLYPH, theme, button_font)
    maximize_button.pack(side="left", padx=(2, 0))

    controller = WindowControls(window, controls, maximize_button)
    minimize_button.configure(command=controller.minimize)
    maximize_button.configure(command=controller.toggle)
    if on_close is not None:
        try:
            window.protocol("WM_DELETE_WINDOW", on_close)
        except tk.TclError:
            pass
    return controller


def _control_button(parent: tk.Misc, text: str, theme: Any, font: Any) -> tk.Button:
    return tk.Button(
        parent,
        text=text,
        relief="flat",
        borderwidth=0,
        highlightthickness=0,
        bg=theme.surface_raised,
        fg=theme.foreground,
        activebackground=theme.interactive_active,
        activeforeground=theme.contrast,
        font=font,
        width=2,
        padx=2,
        pady=0,
        cursor="hand2",
    )


__all__ = ["WindowControls", "install_window_controls"]
