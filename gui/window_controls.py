"""Shared minimize / maximize / close controls for every window's title bar.

Every window in the program -- the main window and all twenty-five secondary
windows (the Code Studio, the Presentation Studio and its finalize/course
windows, the Import Center, the SQL screen, Doc's Studio, the image pages, the
Characters window, the Cognition window, the Training Center, the Workplace, the
Romance Studio and its managers, the Scene Studio, the speech studios, the Model
Manager, the maintenance and personality windows, and the per-mouth chat
windows) -- draws its title in the project's own dark-grey bar. This module gives
that bar the three controls a Windows title bar carries, once, so every window
grows the same set instead of each one inventing its own:

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

It also owns the one **opening-size rule** every window uses
(:func:`screen_metrics`, :func:`window_fit`, :func:`apply_window_geometry`,
:func:`fit_minsize`), so a window opened on a laptop opens inside the usable
desktop -- the whole screen minus the Windows taskbar, minus the window's own
frame -- and its composer and controls stay reachable. The rule is one function
rather than a per-dialog constant, so a window added tomorrow fits the same
screen the same way with no extra work.

Tk is imported at module import, so this module is only ever imported by GUI code
that already depends on Tk; ``cli/`` must never import it.
"""

from __future__ import annotations

import sys
import tkinter as tk
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Literal

from core.best_effort import best_effort

#: The room a *decorated* window needs for its own operating-system frame -- the
#: title bar and the resize borders -- which Tk does **not** count when it is told
#: a geometry. A window asked for the work area's exact height therefore ends up
#: that much taller once its frame is drawn, and its bottom edge (a composer, a
#: row of buttons) slides under the taskbar. These allowances are reserved when a
#: window is *placed*, so the whole window -- frame included -- fits the usable
#: desktop. They are deliberately modest: an under-estimate costs a few pixels, an
#: over-estimate would shrink every window for no reason.
WINDOW_CHROME_WIDTH = 16
WINDOW_CHROME_HEIGHT = 44

#: The glyphs. Chosen from the characters every Tk font on every platform carries,
#: so a control never renders as a missing-glyph box.
_MINIMIZE_GLYPH = "\u2013"  # en dash
_MAXIMIZE_GLYPH = "\u25a1"  # white square
_RESTORE_GLYPH = "\u2750"  # upper right shadowed white square
_CLOSE_GLYPH = "\u2715"  # multiplication x

#: The optional **help** control. A window that can explain itself passes an
#: ``on_help`` callback and this glyph is drawn to the left of the trio; a window
#: with nothing to explain simply does not pass one and the glyph never appears.
_HELP_GLYPH = "?"

#: Hover fills: a quiet grey for minimize/maximize/help, the standard red for close
#: -- the one title-bar affordance every desktop user already recognises.
_BUTTON_HOVER = "hover"
_CLOSE_HOVER = "close"


@dataclass(frozen=True)
class ScreenMetrics:
    """What the desktop reports about the space a window may really use.

    ``screen_width``/``screen_height`` are the display's **whole** size (the
    taskbar included, which is what ``winfo_screenheight`` returns), and
    ``work_x``/``work_y``/``work_width``/``work_height`` are the *usable* desktop
    -- the screen minus the taskbar, minus any dock or panel the desktop reserves.
    ``taskbar`` is the difference in height between the two (0 when the desktop
    reports no work area, or a taskbar along a side), so a caller can *name* what
    it subtracted rather than quietly shrinking a window.

    Every field is a real, measured figure, never a guess: where the desktop
    cannot report its work area the usable rectangle is the screen, and
    :attr:`reported` is ``False`` so an honest caller can say so.
    """

    screen_width: int
    screen_height: int
    work_x: int
    work_y: int
    work_width: int
    work_height: int
    taskbar: int
    reported: bool

    @property
    def usable_width(self) -> int:
        """The width a window may occupy (the work area's)."""
        return self.work_width

    @property
    def usable_height(self) -> int:
        """The height a window may occupy -- the screen *minus the taskbar*."""
        return self.work_height

    def describe(self) -> str:
        """A one-line, honest account of the measured desktop."""
        if self.reported:
            return (
                f"screen {self.screen_width}x{self.screen_height}, "
                f"taskbar {self.taskbar}px, usable {self.work_width}x{self.work_height}"
            )
        return (
            f"screen {self.screen_width}x{self.screen_height}; "
            "the desktop reports no work area, so the screen is the bound"
        )


def work_area() -> tuple[int, int, int, int] | None:
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
        return probe()
    if sys.platform.startswith("linux"):
        return _x11_work_area()
    return None


def screen_metrics(window: tk.Misc | None = None) -> ScreenMetrics:
    """The desktop's whole size and its usable work area, measured once.

    This is the one place a window learns *how much screen it may use*, so every
    window -- the main one and each dialog, now and in future -- is sized against
    the same honest figures: the screen **minus the Windows taskbar** (or the
    desktop's panel), never the raw ``winfo_screenheight`` that includes it.

    ``window`` supplies the screen size when the platform reports no work area
    (and is what makes the answer testable with a stand-in). The work area, when
    the desktop reports one, wins on both size *and* origin -- a taskbar at the
    top moves the usable rectangle's ``y`` down, and a dock on the left moves its
    ``x`` across -- so a placed window sits in the space that is really free.
    """
    screen_width, screen_height = _screen_size(window)
    area = work_area()
    if area is None:
        return ScreenMetrics(
            screen_width=screen_width,
            screen_height=screen_height,
            work_x=0,
            work_y=0,
            work_width=screen_width,
            work_height=screen_height,
            taskbar=0,
            reported=False,
        )
    x, y, width, height = area
    # A work area larger than the screen is a mis-report (a stale property after a
    # resolution change); the screen is the honest bound in that case.
    width = min(width, screen_width) if screen_width > 0 else width
    height = min(height, screen_height) if screen_height > 0 else height
    # The vertical space the desktop reserves (the taskbar, wherever it sits).
    taskbar = max(0, screen_height - height)
    return ScreenMetrics(
        screen_width=screen_width,
        screen_height=screen_height,
        work_x=x,
        work_y=y,
        work_width=width,
        work_height=height,
        taskbar=taskbar,
        reported=True,
    )


def _screen_size(window: tk.Misc | None) -> tuple[int, int]:
    """The display's whole size, from the window when it has one.

    A window that cannot report it (a stand-in, or one whose interpreter has gone
    away) yields ``(0, 0)``, which every caller reads as "no screen bound" and
    falls back to the work area -- so a probe failure narrows what is known
    rather than breaking a window.
    """
    if window is not None:
        try:
            return int(window.winfo_screenwidth()), int(window.winfo_screenheight())
        except (tk.TclError, AttributeError):
            return 0, 0
    return 0, 0



def _windows_work_area() -> tuple[int, int, int, int] | None:
    """The primary monitor's work area, through ``SystemParametersInfoW``."""
    with best_effort("read the Windows work area"):
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
    return None


def _macos_work_area() -> tuple[int, int, int, int] | None:
    """The primary screen's ``visibleFrame`` (excludes the menu bar and Dock)."""
    import subprocess

    with best_effort("read the macOS work area"):
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
    return None


def _x11_work_area() -> tuple[int, int, int, int] | None:
    """The window manager's ``_NET_WORKAREA``, read through ``xprop``.

    ``xprop`` prints an array of cardinals; the first four are the x, y, width,
    and height of the work area (a per-desktop array follows, which the slice
    drops). A desktop that does not publish the property prints an error and the
    ``returncode`` guard turns that into ``None``.
    """
    import subprocess

    with best_effort("read the X11 work area"):
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
        help_button: tk.Button | None = None,
        open_fit: WindowFit | None = None,
    ) -> None:
        self.window = window
        self.frame = frame
        self.maximize_button = maximize_button
        self.close_button = close_button
        self.help_button = help_button
        self._close = close
        self._fullscreen = False
        self._zoomed = False
        self._filled = False
        #: The geometry the window had before a work-area fill, so restore returns
        #: it exactly (Tk remembers nothing once ``geometry`` is set).
        self._restore_geometry: str | None = None
        #: The size the window opened at. A window maximized before it is first
        #: mapped has no live geometry yet (Tk answers ``1x1+0+0``), so restore
        #: returns this instead of the placeholder -- which would collapse the
        #: window. Only set when the window opened through
        #: :func:`apply_window_geometry`, so a plain window is unaffected.
        self._open_fit = open_fit
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
            for button in (self.maximize_button, self.close_button, self.help_button):
                if button is not None:
                    button.bind("<Destroy>", self._on_button_destroy, add="+")
        except (tk.TclError, AttributeError):
            pass

    def _on_destroy(self, event: tk.Event) -> None:
        if event.widget is not self.frame:
            return
        self._close = None

    def _on_button_destroy(self, event: tk.Event) -> None:
        # Bound only to the trio (plus the optional help button), so the widget is
        # a Button; the narrowing is explicit because a Tk event carries its widget
        # as a generic Misc.
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
        if self._try_native_zoom():
            self._sync_glyph()
            return
        if not self._fill_work_area():
            try:
                self.window.attributes("-fullscreen", True)
                self._fullscreen = True
            except tk.TclError:
                pass
        self._sync_glyph()

    def _try_native_zoom(self) -> bool:
        """Use the window manager's own maximize when it really implements one.

        ``state("zoomed")`` is only trustworthy if the window manager *acts* on
        it: on a desktop without the state it raises (handled here), but a desktop
        that accepts the call and ignores it would leave the button offering a
        restore for a window that never grew. So the state is read back, and the
        native path is accepted only when the window really reports itself zoomed;
        otherwise the caller falls back to the work-area fill.
        """
        try:
            self.window.state("zoomed")
        except tk.TclError:
            return False
        try:
            zoomed = str(self.window.state()) == "zoomed"
        except tk.TclError:
            zoomed = True
        if not zoomed:
            # The call was accepted but ignored; undo it so the fill is honest.
            with best_effort("undo an ignored native maximize"):
                self.window.state("normal")
            return False
        self._zoomed = True
        return True

    def _fill_work_area(self) -> bool:
        """Size the window to the work area; return whether it could be learned."""
        area = work_area()
        if area is None:
            return False
        x, y, width, height = area
        # Reserve the operating system's own frame, which Tk's geometry does not
        # count: without this the frame's title bar and borders push the window's
        # bottom edge under the taskbar even though the *client* area fits.
        width = max(MIN_USABLE_WIDTH, width - WINDOW_CHROME_WIDTH)
        height = max(MIN_USABLE_HEIGHT, height - WINDOW_CHROME_HEIGHT)
        try:
            self._restore_geometry = self._current_geometry()
            self.window.geometry(f"{width}x{height}+{x}+{y}")
            self._filled = True
        except tk.TclError:
            return False
        return True

    def _current_geometry(self) -> str:
        """The size restore should return to.

        The live geometry, unless the window is not really drawn yet -- Tk answers
        an unmapped window with ``1x1+0+0`` -- in which case the size the window
        opened at is used, so maximizing before the first draw cannot leave restore
        collapsing the window to a sliver.
        """
        try:
            live = self.window.geometry()
        except tk.TclError:
            live = ""
        size, _, _rest = live.partition("+")
        try:
            width, height = (int(part) for part in size.split("x", 1))
        except ValueError:
            width = height = 0
        if width > 1 and height > 1:
            return live
        if self._open_fit is not None:
            return self._open_fit.placement
        return live

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
    on_help: Callable[[], None] | None = None,
    pack_side: Literal["left", "right", "top", "bottom"] = "right",
    padx: int = 4,
    open_fit: WindowFit | None = None,
) -> WindowControls:
    """Build the minimize / maximize / close buttons into ``header``.

    ``header`` is the inner frame of the window's own title bar (the one the title
    labels live in). The controls are packed as one unit on ``pack_side`` (right by
    default), so they sit with whatever the caller already packs there; pack the
    caller's own buttons *after* calling this to put them further inside, or
    *before* to put them further out.

    ``on_close``, when given, is wired to the window's close protocol as well, so
    the ✕ button and the window manager's own close both take one path.

    ``on_help``, when given, adds a **?** button to the left of the trio; pressing
    it runs the caller's explanation handler. A window with nothing to explain
    passes nothing and keeps exactly the three controls it had, so this is
    additive for every existing caller.
    """
    if open_fit is None:
        # A window that opened through ``apply_window_geometry`` recorded its fit
        # on itself; read it here so no dialog has to thread it through by hand and
        # restore still returns the size the window really opened at.
        candidate = getattr(window, "_window_fit", None)
        if isinstance(candidate, WindowFit):
            open_fit = candidate
    controls = tk.Frame(header, bg=theme.titlebar)
    controls.pack(side=pack_side, padx=(0, padx))

    button_font = typography.font("sm", weight="bold")
    help_button: tk.Button | None = None
    if on_help is not None:
        help_button = _control_button(controls, _HELP_GLYPH, theme, button_font)
        help_button.pack(side="left")
    minimize_button = _control_button(controls, _MINIMIZE_GLYPH, theme, button_font)
    minimize_button.pack(side="left", padx=(2, 0) if help_button is not None else 0)
    maximize_button = _control_button(controls, _MAXIMIZE_GLYPH, theme, button_font)
    maximize_button.pack(side="left", padx=(2, 0))
    close_button = _control_button(
        controls, _CLOSE_GLYPH, theme, button_font, hover=_CLOSE_HOVER
    )
    close_button.pack(side="left", padx=(2, 0))

    controller = WindowControls(
        window, controls, maximize_button, close=on_close, close_button=close_button,
        help_button=help_button, open_fit=open_fit,
    )
    if help_button is not None and on_help is not None:
        help_button.configure(command=on_help)
    minimize_button.configure(command=controller.minimize)
    maximize_button.configure(command=controller.toggle)
    close_button.configure(command=controller.close)
    if on_close is not None:
        with best_effort("register the window's close handler"):
            window.protocol("WM_DELETE_WINDOW", on_close)
    # Escape closes a **dialog**, the standard affordance; the main window is a
    # ``Tk`` root (not a ``Toplevel``), so Escape there is left to the window's own
    # shortcuts rather than quitting the whole application. The binding sits on the
    # toplevel, so it works wherever focus is inside the dialog, and it never fires
    # while a modal grab is in effect -- a nested chooser still owns Escape.
    if isinstance(window, tk.Toplevel):
        with best_effort("bind Escape to close the dialog"):
            window.bind("<Escape>", lambda _event: controller.close())
    return controller


def fit_geometry(
    window: tk.Misc, width: int, height: int, *, min_height: int = 0
) -> str:
    """A ``"WxH"`` opening geometry that never exceeds the usable desktop.

    A window taller than the screen -- minus the taskbar -- would run off the
    bottom, where its composer or its buttons could not be reached, and a window
    wider than the screen would run off the side. Every dialog opens through this,
    so a small screen opens the same window shrunk to fit instead of a window that
    is partly off-screen: the requested size is clamped to :func:`work_area` (the
    platform's own "screen minus taskbar" rectangle) where the window manager
    reports one, and to the screen otherwise.

    ``min_height`` raises the result to a floor (a window must stay usable), but is
    itself capped at the available height, so a very short screen still fits.
    """
    fit = window_fit(window, width, height, min_height=min_height)
    return fit.geometry


@dataclass(frozen=True)
class WindowFit:
    """The size one window should take, measured against the usable desktop.

    Produced by :func:`window_fit`, and the single shape every window's opening
    rule goes through, so the main window and all the dialogs are sized the same
    way and by the same figures.

    ``width``/``height`` are the window's **outer** size -- what Tk's ``geometry``
    wants -- already inside the work area *and* already reserving the room the
    operating system's own frame needs (:data:`WINDOW_CHROME_WIDTH`/
    :data:`WINDOW_CHROME_HEIGHT`), so the whole window, frame included, fits
    beside the taskbar. ``x``/``y`` are where to put it (the work area's origin,
    so a taskbar along the top or a dock along the left is stepped over);
    ``centred`` is true when the window was centred in the work area (the default)
    rather than pinned to its origin.

    ``shrunk`` names whether the request had to be reduced -- the honest signal a
    caller (or a test) can read instead of guessing from the numbers -- and
    ``metrics`` is the measured desktop it was fitted to.
    """

    width: int
    height: int
    x: int
    y: int
    centred: bool
    shrunk: bool
    metrics: ScreenMetrics

    @property
    def geometry(self) -> str:
        """The Tk geometry string, ``"WxH"`` (the caller places the window)."""
        return f"{self.width}x{self.height}"

    @property
    def placement(self) -> str:
        """The Tk geometry string **with** a position, ``"WxH+X+Y"``."""
        return f"{self.width}x{self.height}+{self.x}+{self.y}"


#: The smallest a window may be shrunk to and still be usable. A very small
#: screen gives up height before it gives up width, because a window's controls
#: live in a row and a column of them would be unusable before the height is.
MIN_USABLE_WIDTH = 420
MIN_USABLE_HEIGHT = 360


def window_fit(
    window: tk.Misc,
    width: int,
    height: int,
    *,
    min_width: int = 0,
    min_height: int = 0,
    reserve_chrome: bool = True,
    centre: bool = True,
    metrics: ScreenMetrics | None = None,
) -> WindowFit:
    """Fit ``width`` x ``height`` into the desktop, taskbar and frame reserved.

    The one rule every window -- the main one and each dialog, present and future
    -- sizes itself by, so "fits the laptop screen, minus the Windows taskbar" is
    true of all of them rather than of whichever window happened to be fixed last.

    * the **usable** rectangle is the desktop's work area (screen minus taskbar /
      panel) where it reports one, else the screen (see :func:`screen_metrics`);
    * the operating system's own frame is reserved (``reserve_chrome``), because
      Tk's geometry does not count it and a window asked for the exact work-area
      height would otherwise hang its bottom edge under the taskbar;
    * ``min_width``/``min_height`` raise a too-small request, but are themselves
      capped at what fits, and :data:`MIN_USABLE_WIDTH`/:data:`MIN_USABLE_HEIGHT`
      keep the result usable on a tiny screen;
    * the window is **centred** in the work area (the professional default) rather
      than jammed into its corner -- and centring is safe even for a shrunk window,
      because the frame's room was reserved above, so the whole window still fits.
    """
    if metrics is None:
        metrics = screen_metrics(window)
    chrome_w = WINDOW_CHROME_WIDTH if reserve_chrome else 0
    chrome_h = WINDOW_CHROME_HEIGHT if reserve_chrome else 0
    available_w = max(MIN_USABLE_WIDTH, metrics.usable_width - chrome_w)
    available_h = max(MIN_USABLE_HEIGHT, metrics.usable_height - chrome_h)

    wanted_w = max(0, int(width))
    wanted_h = max(0, int(height))
    shrunk = wanted_w > available_w or wanted_h > available_h

    fitted_w = min(wanted_w, available_w)
    fitted_h = min(wanted_h, available_h)
    if min_width:
        fitted_w = max(fitted_w, min(min_width, available_w))
    if min_height:
        fitted_h = max(fitted_h, min(min_height, available_h))
    fitted_w = max(fitted_w, min(MIN_USABLE_WIDTH, available_w))
    fitted_h = max(fitted_h, min(MIN_USABLE_HEIGHT, available_h))

    if centre:
        x = metrics.work_x + max(0, (metrics.usable_width - fitted_w - chrome_w) // 2)
        y = metrics.work_y + max(0, (metrics.usable_height - fitted_h - chrome_h) // 2)
    else:
        x = metrics.work_x
        y = metrics.work_y
    return WindowFit(
        width=fitted_w,
        height=fitted_h,
        x=x,
        y=y,
        centred=centre,
        shrunk=shrunk,
        metrics=metrics,
    )


def fit_minsize(
    window: tk.Misc,
    width: int,
    height: int,
    *,
    reserve_chrome: bool = True,
    metrics: ScreenMetrics | None = None,
) -> tuple[int, int]:
    """The smallest size a window should allow, capped to the usable desktop.

    A ``minsize`` larger than the work area would make the window impossible to
    shrink to fit -- the window manager would refuse the resize and the window
    would stay partly off-screen -- so the declared minimum is clamped to the
    space that really exists, frame reserved. A screen too small for the minimum
    still gets the whole usable desktop rather than an unreachable window.
    """
    if metrics is None:
        metrics = screen_metrics(window)
    chrome_w = WINDOW_CHROME_WIDTH if reserve_chrome else 0
    chrome_h = WINDOW_CHROME_HEIGHT if reserve_chrome else 0
    available_w = max(MIN_USABLE_WIDTH, metrics.usable_width - chrome_w)
    available_h = max(MIN_USABLE_HEIGHT, metrics.usable_height - chrome_h)
    return min(max(0, int(width)), available_w), min(max(0, int(height)), available_h)


def apply_window_geometry(
    window: tk.Tk | tk.Toplevel,
    width: int,
    height: int,
    *,
    min_width: int = 0,
    min_height: int = 0,
    centre: bool = True,
    place: bool = True,
) -> WindowFit:
    """Size (and optionally place) a window inside the usable desktop.

    The one call a window makes at construction, so every window -- the main one
    and each dialog, now and in future -- opens at a size that fits the laptop
    screen **minus the Windows taskbar**, and its ``minsize`` is capped to the
    same space so it can never be stuck partly off-screen. ``place=False`` sets
    only the size and lets the window manager choose the position (the dialogs'
    ``transient`` behaviour); the default centres a window that fits, which is
    what reads as deliberate rather than as a window dropped in the corner.

    Returns the :class:`WindowFit` that was applied, so a caller (or a test) can
    read what really happened -- including whether the request had to shrink.
    """
    # The desktop is measured **once** and reused, so a window open costs one
    # ``xprop``/system probe rather than one per question asked about the screen.
    metrics = screen_metrics(window)
    fit = window_fit(
        window,
        width,
        height,
        min_width=min_width,
        min_height=min_height,
        centre=centre,
        metrics=metrics,
    )
    with best_effort("size the window to the usable desktop"):
        window.geometry(fit.placement if place else fit.geometry)
    small_w, small_h = fit_minsize(window, min_width, min_height, metrics=metrics)
    if small_w > 0 and small_h > 0:
        with best_effort("cap the window's minimum size to the usable desktop"):
            window.minsize(small_w, small_h)
    # Record the fit on the window itself, so ``install_window_controls`` can give
    # the controller the size restore should return to: a window maximized before
    # it is first mapped has no live geometry, and Tk would answer ``1x1+0+0``.
    with best_effort("record the window's opening fit"):
        window.__dict__["_window_fit"] = fit
    return fit



def _control_button(
    parent: tk.Misc, text: str, theme: Any, font: Any, *, hover: str = _BUTTON_HOVER
) -> tk.Button:
    if hover == _CLOSE_HOVER:
        active_bg, active_fg = theme.danger, theme.contrast
    else:
        active_bg, active_fg = theme.interactive_active, theme.contrast
    button = tk.Button(
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
        padx=6,
        pady=4,
        cursor="hand2",
    )
    # Tk's ``activebackground`` only paints while the button is *held*, so a
    # title-bar control would give no feedback on hover -- which reads as dead.
    # The tint is bound on enter/leave and restored to the button's own resting
    # colour, exactly like the composer's controls.
    resting = theme.surface_raised

    def on_enter(_event: tk.Event) -> None:
        try:
            button.configure(bg=active_bg)
        except tk.TclError:
            pass

    def on_leave(_event: tk.Event) -> None:
        try:
            button.configure(bg=resting)
        except tk.TclError:
            pass

    button.bind("<Enter>", on_enter, add="+")
    button.bind("<Leave>", on_leave, add="+")
    return button


__all__ = [
    "MIN_USABLE_HEIGHT",
    "MIN_USABLE_WIDTH",
    "WINDOW_CHROME_HEIGHT",
    "WINDOW_CHROME_WIDTH",
    "ScreenMetrics",
    "WindowControls",
    "WindowFit",
    "apply_window_geometry",
    "fit_geometry",
    "fit_minsize",
    "install_window_controls",
    "screen_metrics",
    "window_fit",
    "work_area",
]
