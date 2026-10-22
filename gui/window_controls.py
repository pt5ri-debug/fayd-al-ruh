"""Shared minimize / maximize / close controls for every window's title bar.

Every window in the program -- the main window and each secondary one: the Code
Studio, the Presentation Studio, the final editor, the Course window, the Import
Center, the SQL screen, and the two image pages -- draws its title in the
project's own dark-grey bar. This module gives that bar the three controls a
Windows title bar carries, once, so every window grows the same set instead of
each one inventing its own:

* **minimize** (``–``) -- shrink the window to the taskbar / dock;
* **maximize / restore** (``□`` / ``❐``) -- fill the screen, and shrink it back to
  the size it had before;
* **close** (``✕``) -- the window's one close path.

The buttons are painted with the same theme tokens and type scale the rest of the
program uses, so a title bar reads as one piece. Maximising is honest about what
the platform supports: ``state("zoomed")`` where the window manager implements it
(Windows, some X11 desktops), and a full-screen window otherwise -- exactly the
fallback the feature was asked for -- so the control never silently does nothing.

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
_CLOSE_GLYPH = "\u2715"  # multiplication x

#: Hover fills: a quiet grey for minimize/maximize, the standard red for close --
#: the one title-bar affordance every desktop user already recognises.
_BUTTON_HOVER = "hover"
_CLOSE_HOVER = "close"


class WindowControls:
    """The minimize / maximize / close trio for one window.

    Construct it through :func:`install_window_controls`, which builds and packs
    the buttons into a window's title bar. The object keeps the small amount of
    state Tk does not expose -- whether *this* code made the window full-screen,
    so the restore path can undo exactly what the maximize path did.
    """

    def __init__(
        self,
        window: tk.Tk | tk.Toplevel,
        frame: tk.Frame,
        maximize_button: tk.Button,
        *,
        close: Callable[[], None] | None = None,
        close_button: tk.Button | None = None,
    ) -> None:
        self.window = window
        self.frame = frame
        self.maximize_button = maximize_button
        self.close_button = close_button
        self._close = close
        self._fullscreen = False
        self._zoomed = False
        self._bind_destroy()

    def _bind_destroy(self) -> None:
        """Let the bar release the caller when the window goes away.

        Tk's own destroy never drops a ``command`` callable, so a plain
        ``self._close`` would keep the whole window in a reference cycle that only
        the cyclic collector could break -- and that can run on a worker thread,
        where a Tcl call deadlocks. Each button clears its *own* command on its own
        ``<Destroy>`` (while the widget still exists, which a frame-level handler
        cannot promise), and the frame drops the caller's handler, exactly like
        ``gui.chat_composer`` does.
        """
        try:
            self.frame.bind("<Destroy>", self._on_destroy, add="+")
            for button in (self.maximize_button, self.close_button):
                if button is not None:
                    button.bind("<Destroy>", self._on_button_destroy, add="+")
        except (tk.TclError, AttributeError):
            pass

    def _on_destroy(self, event: tk.Event) -> None:
        if event.widget is not self.frame:
            return
        self._close = None

    def _on_button_destroy(self, event: tk.Event) -> None:
        # Bound only to the two buttons, so the widget is a Button; the narrowing
        # is explicit because a Tk event carries its widget as a generic Misc.
        widget = event.widget
        if not isinstance(widget, tk.Button):
            return
        try:
            widget.configure(command="")
        except tk.TclError:  # the widget's interpreter already went away
            pass

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

    def close(self) -> None:
        """Close the window through the one handler it was given.

        With no handler the window is simply destroyed, which is what a window
        that never registered a close path would do anyway.
        """
        if self._close is not None:
            self._close()
            return
        try:
            self.window.destroy()
        except tk.TclError:
            pass

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
    window: tk.Tk | tk.Toplevel,
    header: tk.Misc,
    *,
    theme: Any,
    typography: Any,
    on_close: Callable[[], None] | None = None,
    pack_side: Literal["left", "right", "top", "bottom"] = "right",
    padx: int = 4,
) -> WindowControls:
    """Build the minimize / maximize / close buttons into ``header``.

    ``header`` is the inner frame of the window's own title bar (the one the title
    labels live in). The controls are packed as one unit on ``pack_side`` (right by
    default), so they sit with whatever the caller already packs there; pack the
    caller's own buttons *after* calling this to put them further inside, or
    *before* to put them further out.

    ``on_close``, when given, is wired to the window's close protocol as well, so
    the ✕ button and the window manager's own close both take one path.
    """
    controls = tk.Frame(header, bg=theme.titlebar)
    controls.pack(side=pack_side, padx=(0, padx))

    button_font = typography.font("sm", weight="bold")
    minimize_button = _control_button(controls, _MINIMIZE_GLYPH, theme, button_font)
    minimize_button.pack(side="left")
    maximize_button = _control_button(controls, _MAXIMIZE_GLYPH, theme, button_font)
    maximize_button.pack(side="left", padx=(2, 0))
    close_button = _control_button(
        controls, _CLOSE_GLYPH, theme, button_font, hover=_CLOSE_HOVER
    )
    close_button.pack(side="left", padx=(2, 0))

    controller = WindowControls(
        window, controls, maximize_button, close=on_close, close_button=close_button
    )
    minimize_button.configure(command=controller.minimize)
    maximize_button.configure(command=controller.toggle)
    close_button.configure(command=controller.close)
    if on_close is not None:
        try:
            window.protocol("WM_DELETE_WINDOW", on_close)
        except tk.TclError:
            pass
    return controller


def _control_button(
    parent: tk.Misc, text: str, theme: Any, font: Any, *, hover: str = _BUTTON_HOVER
) -> tk.Button:
    if hover == _CLOSE_HOVER:
        active_bg, active_fg = theme.danger, theme.contrast
    else:
        active_bg, active_fg = theme.interactive_active, theme.contrast
    return tk.Button(
        parent,
        text=text,
        relief="flat",
        borderwidth=0,
        highlightthickness=0,
        bg=theme.surface_raised,
        fg=theme.foreground,
        activebackground=active_bg,
        activeforeground=active_fg,
        font=font,
        width=2,
        padx=2,
        pady=0,
        cursor="hand2",
    )


__all__ = ["WindowControls", "install_window_controls"]
