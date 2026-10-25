"""Shared minimize / maximize / close controls for every window's title bar.

Every window in the program -- the main window and each secondary one: the Code
Studio, the Presentation Studio, the final editor, the Course window, the Import
Center, the SQL screen, and the two image pages -- draws its title in the
project's own dark-grey bar. This module gives that bar the three controls a
Windows title bar carries, once, so every window grows the same set instead of
each one inventing its own:

* **minimize** (``–``) -- shrink the window to the taskbar / dock;
* **maximize / restore** (``□`` / ``❐``) -- grow the window to the *usable desktop*
  (the whole screen minus the Windows taskbar), and shrink it back to the size it
  had before;
* **close** (``✕``) -- the window's one close path.

The buttons are painted with the same theme tokens and type scale the rest of the
program uses, so a title bar reads as one piece. Maximising is honest about what
the platform supports: ``state("zoomed")`` where the window manager implements it
(Windows and macOS, where it already grows to the work area and leaves the taskbar
visible), and -- where it does not, as on many X11 desktops -- a real windowed
fill of the work area (see :func:`work_area`) rather than a full-screen takeover
that would cover the taskbar. So the control always grows the window to fit the
screen *without* hiding the taskbar, and never silently does nothing.

Tk is imported at module import, so this module is only ever imported by GUI code
that already depends on Tk; ``cli/`` must never import it.
"""

from __future__ import annotations

import sys
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


def work_area(window: tk.Misc) -> tuple[int, int, int, int] | None:
    """The usable desktop rectangle ``(x, y, width, height)``, or ``None``.

    "Maximize" means *the whole screen minus the taskbar*, and a plain
    ``winfo_screenheight()`` includes the taskbar's strip, so a window grown to it
    would sit under the taskbar. The work area is the platform's own answer to
    "how much of the screen may a maximised window use", and each desktop reports
    it differently:

    * **Windows** -- the primary display's *work area* (``rcWork``), asked of the
      system directly through ``SystemParametersInfo``, which already excludes the
      taskbar on whichever edge the user moved it to;
    * **X11** -- the ``_NET_WORKAREA`` property the window manager publishes, read
      through ``xprop``; many desktops (GNOME, KDE, XFCE) set it, and it excludes
      panels and docks;
    * **macOS** -- ``NSScreen.visibleFrame``, which excludes the menu bar and the
      Dock.

    The lookup is best-effort by design: the first answer that yields a sane
    rectangle wins, and a platform that cannot report one returns ``None`` so the
    caller falls back to a full-screen window (which is correct everywhere, just
    less tidy under a taskbar). Every probe is a subprocess or a system call, so a
    failure is expected rather than exceptional and is swallowed here.
    """
    probes = {
        "win32": _windows_work_area,
        "darwin": _macos_work_area,
    }
    probe = probes.get(sys.platform)
    if probe is not None:
        return probe(window)
    if sys.platform.startswith("linux"):
        return _x11_work_area()
    return None


def _windows_work_area(window: tk.Misc) -> tuple[int, int, int, int] | None:
    """The primary monitor's work area, through ``SystemParametersInfoW``."""
    try:
        import ctypes
        from ctypes import wintypes

        class RECT(ctypes.Structure):
            _fields_ = [
                ("left", wintypes.LONG),
                ("top", wintypes.LONG),
                ("right", wintypes.LONG),
                ("bottom", wintypes.LONG),
            ]

        rect = RECT()
        # SPI_GETWORKAREA = 0x0030; the call fills the primary display's work area.
        # ``windll`` exists only on Windows (which is where this branch runs), so it
        # is looked up with a default -- that keeps the module importable and the
        # type checker happy on the other platforms.
        windll = getattr(ctypes, "windll", None)
        if windll is None:
            return None
        ok = windll.user32.SystemParametersInfoW(0x0030, 0, ctypes.byref(rect), 0)
        if not ok:
            return None
        width = rect.right - rect.left
        height = rect.bottom - rect.top
        if width > 0 and height > 0:
            return (rect.left, rect.top, width, height)
    except Exception:
        pass
    return None


def _macos_work_area(window: tk.Misc) -> tuple[int, int, int, int] | None:
    """The primary screen's ``visibleFrame`` (excludes the menu bar and Dock)."""
    try:
        import subprocess

        script = (
            'tell application "Finder" to get bounds of window of desktop'
        )
        result = subprocess.run(
            ["osascript", "-e", script],
            capture_output=True, text=True, timeout=2, check=False,
        )
        if result.returncode != 0:
            return None
        parts = [int(piece.strip()) for piece in result.stdout.strip().split(",")]
        if len(parts) != 4:
            return None
        left, top, right, bottom = parts
        width, height = right - left, bottom - top
        if width > 0 and height > 0:
            return (left, top, width, height)
    except Exception:
        pass
    return None


def _x11_work_area() -> tuple[int, int, int, int] | None:
    """The window manager's ``_NET_WORKAREA``, read through ``xprop``.

    ``xprop`` prints an array of cardinals; the first four are the x, y, width,
    and height of the work area (a per-desktop array follows, which the slice
    drops). A desktop that does not publish the property prints an error and the
    ``returncode`` guard turns that into ``None``.
    """
    try:
        import subprocess

        result = subprocess.run(
            ["xprop", "-root", "-notype", "_NET_WORKAREA"],
            capture_output=True, text=True, timeout=2, check=False,
        )
        if result.returncode != 0 or "=" not in result.stdout:
            return None
        numbers = [
            int(piece)
            for piece in result.stdout.split("=", 1)[1].replace(",", " ").split()
            if piece.lstrip("-").isdigit()
        ]
        if len(numbers) < 4:
            return None
        x, y, width, height = numbers[:4]
        if width > 0 and height > 0:
            return (x, y, width, height)
    except Exception:
        pass
    return None


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
        self._filled = False
        #: The geometry the window had before a work-area fill, so restore returns
        #: it exactly (Tk remembers nothing once ``geometry`` is set).
        self._restore_geometry: str | None = None
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
        """Grow the window to the usable desktop, leaving the taskbar visible.

        ``state("zoomed")`` is tried first because it is the *native* maximize --
        the window manager already sizes it to the work area (taskbar excluded) and
        remembers the previous size itself. Where it is not implemented (many X11
        desktops), the work area is filled by hand (see :func:`work_area`) rather
        than going full-screen: a full-screen window would cover the taskbar, which
        is not what "maximize" means on any desktop. Only when the work area cannot
        be learned either does it fall back to a real full-screen window, so the
        control still grows the window on a desktop that reports neither.
        """
        if self._is_open():
            return
        try:
            self.window.state("zoomed")
            self._zoomed = True
        except tk.TclError:
            if not self._fill_work_area():
                try:
                    self.window.attributes("-fullscreen", True)
                    self._fullscreen = True
                except tk.TclError:
                    pass
        self._sync_glyph()

    def _fill_work_area(self) -> bool:
        """Size the window to the work area; return whether it could be learned."""
        area = work_area(self.window)
        if area is None:
            return False
        x, y, width, height = area
        try:
            self._restore_geometry = self.window.geometry()
            self.window.geometry(f"{width}x{height}+{x}+{y}")
            self._filled = True
        except tk.TclError:
            return False
        return True

    def restore(self) -> None:
        """Undo :meth:`maximize`, whichever way it grew the window."""
        if self._zoomed:
            try:
                self.window.state("normal")
            except tk.TclError:
                pass
            self._zoomed = False
        if self._filled:
            if self._restore_geometry:
                try:
                    self.window.geometry(self._restore_geometry)
                except tk.TclError:
                    pass
            self._restore_geometry = None
            self._filled = False
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
        return self._zoomed or self._fullscreen or self._filled

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
