"""The shared minimize / maximize / close controls every window's title bar installs.

Two levels are tested, matching how the module is built:

* the branch logic (``state("zoomed")`` where the window manager supports it, a
  real full-screen window otherwise, and the honest restore that undoes exactly
  what was done) through small duck-typed fakes, so it is deterministic on any
  platform and needs no display;
* the real installation into a Tk title bar, which skips when no display is
  available, like the other window tests.
"""

from __future__ import annotations

from typing import Any

import pytest

try:
    import tkinter as tk

    from gui import window_controls
    from gui.window_controls import (
        _CLOSE_GLYPH,
        _HELP_GLYPH,
        _MAXIMIZE_GLYPH,
        _RESTORE_GLYPH,
        WindowControls,
        install_window_controls,
    )
except ImportError:  # pragma: no cover - no libtk on this machine
    tk = None  # type: ignore[assignment]

pytestmark = pytest.mark.skipif(tk is None, reason="Tk is unavailable")


class FakeButton:
    """A minimal stand-in for a Tk button: records the text it is configured with."""

    def __init__(self) -> None:
        self.text = ""

    def configure(self, **kwargs: Any) -> None:
        if "text" in kwargs:
            self.text = kwargs["text"]


class FakeWindow:
    """A Tk-window stand-in whose "zoomed" state is configurable.

    ``supports_zoom=False`` makes ``state("zoomed")`` raise :class:`tk.TclError`,
    exactly as an X11 desktop without the ``zoomed`` state does, so the
    work-area / full-screen fallback can be exercised without a particular window
    manager.
    """

    def __init__(self, *, supports_zoom: bool = True) -> None:
        self.supports_zoom = supports_zoom
        self.iconified = 0
        self.destroyed = 0
        self.states: list[str] = []
        self.fullscreen: dict[str, bool] = {}
        self.geometry_value = "1180x648+0+0"
        self.geometries: list[str] = []

    def iconify(self) -> None:
        self.iconified += 1

    def destroy(self) -> None:
        self.destroyed += 1

    def state(self, value: str | None = None) -> str:
        if value is None:
            return self.states[-1] if self.states else "normal"
        if value == "zoomed" and not self.supports_zoom:
            raise tk.TclError("bad state")
        self.states.append(value)
        return value

    def geometry(self, value: str | None = None) -> str:
        if value is None:
            return self.geometry_value
        self.geometry_value = value
        self.geometries.append(value)
        return value

    def attributes(self, name: str, value: bool | None = None) -> bool:
        if value is None:
            return self.fullscreen.get(name, False)
        self.fullscreen[name] = value
        return value


def _controls(**kwargs: Any) -> tuple[WindowControls, FakeWindow, FakeButton]:
    window = FakeWindow(**kwargs)
    button = FakeButton()
    controls = WindowControls(window, None, button)  # type: ignore[arg-type]
    return controls, window, button


def test_minimize_iconifies_the_window() -> None:
    controls, window, _button = _controls()
    controls.minimize()
    assert window.iconified == 1


def test_maximize_uses_the_zoomed_state_when_supported() -> None:
    controls, window, button = _controls(supports_zoom=True)
    controls.maximize()
    assert window.states == ["zoomed"]
    assert "-fullscreen" not in window.fullscreen
    assert button.text == _RESTORE_GLYPH


def test_maximize_falls_back_to_the_work_area_without_the_zoomed_state(monkeypatch) -> None:
    """Without "zoomed", the work area is filled rather than the whole screen.

    This is the whole point of the change: a full-screen window would cover the
    taskbar, so a desktop that reports its work area gets a windowed fill of it.
    """
    monkeypatch.setattr(
        "gui.window_controls.work_area", lambda: (0, 40, 1920, 1040)
    )
    controls, window, button = _controls(supports_zoom=False)
    controls.maximize()
    assert "-fullscreen" not in window.fullscreen
    # The work area *minus the window's own frame*, which Tk's geometry does not
    # count: without that reservation the frame's title bar and borders would push
    # the window's bottom edge under the taskbar even though the client fits.
    assert window.geometry_value == (
        f"{1920 - window_controls.WINDOW_CHROME_WIDTH}x"
        f"{1040 - window_controls.WINDOW_CHROME_HEIGHT}+0+40"
    )
    assert button.text == _RESTORE_GLYPH


def test_restore_returns_the_geometry_a_work_area_fill_replaced(monkeypatch) -> None:
    monkeypatch.setattr(
        "gui.window_controls.work_area", lambda: (0, 40, 1920, 1040)
    )
    controls, window, button = _controls(supports_zoom=False)
    controls.toggle()
    assert window.geometry_value.endswith("+0+40")
    controls.toggle()
    assert window.geometry_value == "1180x648+0+0"
    assert button.text == _MAXIMIZE_GLYPH


def test_restore_returns_the_opening_size_when_maximized_before_first_draw(
    monkeypatch,
) -> None:
    """A window maximized before it is mapped restores to the size it opened at.

    Tk answers an unmapped window's geometry with ``1x1+0+0``; capturing that as
    the "previous" size would leave restore collapsing the window to a sliver, so
    the opening fit recorded by ``apply_window_geometry`` is used instead.
    """
    monkeypatch.setattr(
        "gui.window_controls.work_area", lambda: (0, 40, 1920, 1040)
    )
    fit = window_controls.WindowFit(
        width=1180,
        height=648,
        x=370,
        y=236,
        centred=True,
        shrunk=False,
        metrics=window_controls.ScreenMetrics(
            screen_width=1920,
            screen_height=1080,
            work_x=0,
            work_y=40,
            work_width=1920,
            work_height=1040,
            taskbar=40,
            reported=True,
        ),
    )
    window = FakeWindow(supports_zoom=False)
    # Not drawn yet: Tk answers an unmapped window with this placeholder.
    window.geometry_value = "1x1+0+0"
    button = FakeButton()
    controls = window_controls.WindowControls(
        window, None, button, open_fit=fit  # type: ignore[arg-type]
    )
    controls.maximize()
    assert window.geometry_value.endswith("+0+40")  # really grew
    controls.restore()
    assert window.geometry_value == fit.placement  # not the 1x1 placeholder


def test_a_real_live_geometry_still_wins_over_the_opening_fit(monkeypatch) -> None:
    """Once the window is really drawn, restore returns *that* size, not the fit."""
    monkeypatch.setattr(
        "gui.window_controls.work_area", lambda: (0, 40, 1920, 1040)
    )
    fit = window_controls.WindowFit(
        width=1180,
        height=648,
        x=370,
        y=236,
        centred=True,
        shrunk=False,
        metrics=window_controls.ScreenMetrics(
            screen_width=1920,
            screen_height=1080,
            work_x=0,
            work_y=40,
            work_width=1920,
            work_height=1040,
            taskbar=40,
            reported=True,
        ),
    )
    window = FakeWindow(supports_zoom=False)
    window.geometry_value = "900x700+120+80"  # the user resized it
    controls = window_controls.WindowControls(
        window, None, FakeButton(), open_fit=fit  # type: ignore[arg-type]
    )
    controls.maximize()
    controls.restore()
    assert window.geometry_value == "900x700+120+80"


def test_maximize_falls_back_to_fullscreen_when_no_work_area_is_known(monkeypatch) -> None:
    """A desktop that reports neither "zoomed" nor a work area still grows."""
    monkeypatch.setattr("gui.window_controls.work_area", lambda: None)
    controls, window, button = _controls(supports_zoom=False)
    controls.maximize()
    assert window.fullscreen["-fullscreen"] is True
    assert button.text == _RESTORE_GLYPH


def test_maximize_prefers_the_native_zoomed_state(monkeypatch) -> None:
    """When "zoomed" works, the work area is not touched -- the WM already fits it."""
    called: list[int] = []
    monkeypatch.setattr(
        "gui.window_controls.work_area", lambda: called.append(1)
    )
    controls, window, _button = _controls(supports_zoom=True)
    controls.maximize()
    assert window.states == ["zoomed"]
    assert called == []


def test_toggle_maximizes_then_restores_the_same_way() -> None:
    controls, window, button = _controls(supports_zoom=True)
    controls.toggle()
    assert window.states == ["zoomed"]
    controls.toggle()
    assert window.states == ["zoomed", "normal"]
    assert button.text == _MAXIMIZE_GLYPH


def test_restore_undoes_a_fullscreen_maximize(monkeypatch) -> None:
    monkeypatch.setattr("gui.window_controls.work_area", lambda: None)
    controls, window, button = _controls(supports_zoom=False)
    controls.toggle()
    assert window.fullscreen["-fullscreen"] is True
    controls.toggle()
    assert window.fullscreen["-fullscreen"] is False
    assert button.text == _MAXIMIZE_GLYPH


def test_maximize_is_idempotent_while_grown() -> None:
    controls, window, _button = _controls(supports_zoom=True)
    controls.maximize()
    controls.maximize()
    assert window.states == ["zoomed"]


# --- close ----------------------------------------------------------------


def test_close_runs_the_callers_handler() -> None:
    """The ✕ takes the window's own close path, not a second one of its own."""
    window = FakeWindow()
    closed: list[int] = []
    controls = WindowControls(
        window, None, FakeButton(), close=lambda: closed.append(1)  # type: ignore[arg-type]
    )
    controls.close()
    assert closed == [1]
    assert window.destroyed == 0


def test_close_destroys_a_window_that_has_no_handler() -> None:
    window = FakeWindow()
    controls = WindowControls(window, None, FakeButton())  # type: ignore[arg-type]
    controls.close()
    assert window.destroyed == 1


# --- the work area lookup --------------------------------------------------


def test_work_area_reads_the_x11_property(monkeypatch) -> None:
    """`xprop`'s array is parsed into x, y, width, height (per-desktop tail dropped)."""
    import subprocess

    from gui import window_controls

    class FakeCompleted:
        returncode = 0
        stdout = "_NET_WORKAREA(CARDINAL) = 0, 40, 1920, 1040, 0, 40, 1920, 1040\n"

    monkeypatch.setattr(window_controls.sys, "platform", "linux")
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: FakeCompleted())
    assert window_controls.work_area() == (0, 40, 1920, 1040)


def test_work_area_returns_none_when_x11_has_no_property(monkeypatch) -> None:
    import subprocess

    from gui import window_controls

    class FakeCompleted:
        returncode = 1
        stdout = '_NET_WORKAREA:  no such atom on any window.\n'

    monkeypatch.setattr(window_controls.sys, "platform", "linux")
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: FakeCompleted())
    assert window_controls.work_area() is None


def test_work_area_is_none_on_an_unknown_platform(monkeypatch) -> None:
    from gui import window_controls

    monkeypatch.setattr(window_controls.sys, "platform", "haiku")
    assert window_controls.work_area() is None


# --- fit_geometry / window_fit: the usable desktop, taskbar and frame reserved


def test_fit_geometry_clamps_to_the_work_area(monkeypatch) -> None:
    """A small work area wins over the requested size, so nothing runs off-screen."""
    from gui import window_controls

    monkeypatch.setattr(window_controls, "work_area", lambda: (0, 30, 1000, 600))
    assert window_controls.fit_geometry(object(), 1180, 700) == (  # type: ignore[arg-type]
        f"{1000 - window_controls.WINDOW_CHROME_WIDTH}x"
        f"{600 - window_controls.WINDOW_CHROME_HEIGHT}"
    )


def test_fit_geometry_keeps_a_size_that_already_fits(monkeypatch) -> None:
    from gui import window_controls

    monkeypatch.setattr(window_controls, "work_area", lambda: (0, 0, 1920, 1080))
    assert window_controls.fit_geometry(object(), 1180, 700) == "1180x700"  # type: ignore[arg-type]


def test_fit_geometry_raises_to_a_floor_but_never_past_the_screen(monkeypatch) -> None:
    """``min_height`` lifts a too-short request, but is itself capped to fit."""
    from gui import window_controls

    monkeypatch.setattr(window_controls, "work_area", lambda: (0, 0, 1920, 1080))
    assert window_controls.fit_geometry(object(), 1180, 500, min_height=620) == "1180x620"  # type: ignore[arg-type]
    # A short work area caps the floor, the frame reserved.
    monkeypatch.setattr(window_controls, "work_area", lambda: (0, 0, 1920, 560))
    assert window_controls.fit_geometry(object(), 1180, 500, min_height=620) == (  # type: ignore[arg-type]
        f"1180x{560 - window_controls.WINDOW_CHROME_HEIGHT}"
    )


def test_fit_geometry_falls_back_to_the_screen_without_a_work_area(monkeypatch) -> None:
    """With no work-area rectangle, the screen size is the bound (a small screen shrinks)."""
    from gui import window_controls

    class FakeWindow:
        def winfo_screenwidth(self) -> int:
            return 1024

        def winfo_screenheight(self) -> int:
            return 768

    monkeypatch.setattr(window_controls, "work_area", lambda: None)
    assert window_controls.fit_geometry(FakeWindow(), 1180, 900) == (  # type: ignore[arg-type]
        f"{1024 - window_controls.WINDOW_CHROME_WIDTH}x"
        f"{768 - window_controls.WINDOW_CHROME_HEIGHT}"
    )


# --- screen_metrics: the screen, and the taskbar subtracted -----------------


def test_screen_metrics_subtracts_the_taskbar(monkeypatch) -> None:
    """The usable height is the screen *minus the taskbar* (the reported work area)."""
    from gui import window_controls

    class FakeWindow:
        def winfo_screenwidth(self) -> int:
            return 1920

        def winfo_screenheight(self) -> int:
            return 1080

    # A 1080-tall screen whose work area is 1040 tall: a 40 px taskbar at the bottom.
    monkeypatch.setattr(window_controls, "work_area", lambda: (0, 0, 1920, 1040))
    metrics = window_controls.screen_metrics(FakeWindow())  # type: ignore[arg-type]
    assert (metrics.screen_width, metrics.screen_height) == (1920, 1080)
    assert (metrics.usable_width, metrics.usable_height) == (1920, 1040)
    assert metrics.taskbar == 40
    assert metrics.reported is True
    assert "taskbar 40px" in metrics.describe()


def test_screen_metrics_names_a_taskbar_on_the_top(monkeypatch) -> None:
    """A taskbar along the top moves the work area's origin down, not its height alone."""
    from gui import window_controls

    class FakeWindow:
        def winfo_screenwidth(self) -> int:
            return 1920

        def winfo_screenheight(self) -> int:
            return 1080

    monkeypatch.setattr(window_controls, "work_area", lambda: (0, 40, 1920, 1040))
    metrics = window_controls.screen_metrics(FakeWindow())  # type: ignore[arg-type]
    assert metrics.work_y == 40
    assert metrics.taskbar == 40


def test_screen_metrics_is_honest_without_a_work_area(monkeypatch) -> None:
    """No work area reported: the screen is the bound and the report says so."""
    from gui import window_controls

    class FakeWindow:
        def winfo_screenwidth(self) -> int:
            return 1366

        def winfo_screenheight(self) -> int:
            return 768

    monkeypatch.setattr(window_controls, "work_area", lambda: None)
    metrics = window_controls.screen_metrics(FakeWindow())  # type: ignore[arg-type]
    assert (metrics.usable_width, metrics.usable_height) == (1366, 768)
    assert metrics.taskbar == 0
    assert metrics.reported is False
    assert "no work area" in metrics.describe()


# --- window_fit: one rule for every window ---------------------------------


def test_window_fit_centres_a_window_that_fits(monkeypatch) -> None:
    from gui import window_controls

    monkeypatch.setattr(window_controls, "work_area", lambda: (0, 0, 1920, 1040))
    fit = window_controls.window_fit(object(), 800, 600)  # type: ignore[arg-type]
    assert (fit.width, fit.height) == (800, 600)
    assert fit.shrunk is False
    assert fit.centred is True
    # Centred inside the work area (the frame's room accounted for).
    assert fit.x > 0 and fit.y > 0


def test_window_fit_shrinks_a_too_large_window_and_still_centres_it(monkeypatch) -> None:
    from gui import window_controls

    monkeypatch.setattr(window_controls, "work_area", lambda: (0, 0, 1000, 700))
    fit = window_controls.window_fit(object(), 1400, 900)  # type: ignore[arg-type]
    assert fit.shrunk is True
    # Shrinking never leaves the window off-screen, so it may still be centred.
    assert fit.centred is True
    assert (fit.x, fit.y) == (0, 0)  # it fills the work area, so the centre is its origin
    assert fit.width == 1000 - window_controls.WINDOW_CHROME_WIDTH
    assert fit.height == 700 - window_controls.WINDOW_CHROME_HEIGHT


def test_window_fit_never_goes_below_a_usable_minimum(monkeypatch) -> None:
    """A very small work area still yields a usable window rather than a sliver."""
    from gui import window_controls

    monkeypatch.setattr(window_controls, "work_area", lambda: (0, 0, 300, 200))
    fit = window_controls.window_fit(object(), 1180, 700)  # type: ignore[arg-type]
    assert fit.width >= window_controls.MIN_USABLE_WIDTH
    assert fit.height >= window_controls.MIN_USABLE_HEIGHT


def test_fit_minsize_caps_a_too_large_minimum(monkeypatch) -> None:
    """A declared minimum larger than the screen is clamped, so the window can shrink."""
    from gui import window_controls

    monkeypatch.setattr(window_controls, "work_area", lambda: (0, 0, 1024, 640))
    width, height = window_controls.fit_minsize(object(), 1400, 900)  # type: ignore[arg-type]
    assert width == 1024 - window_controls.WINDOW_CHROME_WIDTH
    assert height == 640 - window_controls.WINDOW_CHROME_HEIGHT


# --- the real application of a fit (needs a display) -----------------------


def _display_available() -> bool:
    try:
        root = tk.Tk()
    except Exception:
        return False
    root.destroy()
    return True


@pytest.mark.skipif(not _display_available(), reason="no display available")
def test_apply_window_geometry_sizes_places_and_caps_the_minimum(monkeypatch) -> None:
    """One call sizes, places, and caps ``minsize`` inside the usable desktop."""
    from gui import window_controls

    root = tk.Tk()
    try:
        root.withdraw()
        window = tk.Toplevel(root)
        monkeypatch.setattr(window_controls, "work_area", lambda: (0, 40, 1000, 600))
        fit = window_controls.apply_window_geometry(
            window, 1400, 900, min_width=1100, min_height=700
        )
        root.update()
        # The applied size is inside the work area, frame reserved.
        assert fit.shrunk is True
        assert fit.width == 1000 - window_controls.WINDOW_CHROME_WIDTH
        assert fit.height == 600 - window_controls.WINDOW_CHROME_HEIGHT
        # The minsize is capped to the same space, so the window can always fit.
        assert window.minsize() == (fit.width, fit.height)
        # The geometry really reached Tk.
        assert window.geometry().startswith(f"{fit.width}x{fit.height}")
    finally:
        root.destroy()


# --- the real installation (needs a display) -------------------------------


@pytest.mark.skipif(not _display_available(), reason="no display available")
def test_install_builds_the_buttons_and_wires_close() -> None:
    from gui.app import DARK_THEME
    from gui.typography import for_root

    root = tk.Tk()
    try:
        root.withdraw()
        window = tk.Toplevel(root)
        header = tk.Frame(window)
        header.pack(fill="x")
        closed: list[int] = []
        controls = install_window_controls(
            window,
            header,
            theme=DARK_THEME,
            typography=for_root(root),
            on_close=lambda: closed.append(1),
        )
        root.update()
        # The three controls live in one frame in the title bar: minimize,
        # maximize, and the ✕ that takes the caller's own close path.
        assert controls.frame.master is header
        assert len(controls.frame.winfo_children()) == 3
        assert controls.maximize_button.cget("text") == _MAXIMIZE_GLYPH
        assert controls.close_button is not None
        assert controls.close_button.cget("text") == _CLOSE_GLYPH
        controls.close()
        assert closed == [1]
        # The close protocol now carries the caller's handler rather than Tk's
        # default: querying it returns the registered command string.
        assert window.protocol("WM_DELETE_WINDOW")
        # Escape closes the window, the standard keyboard affordance.
        assert window.bind("<Escape>")
    finally:
        root.destroy()


@pytest.mark.skipif(not _display_available(), reason="no display available")
def test_install_adds_a_help_button_only_when_asked() -> None:
    from gui.app import DARK_THEME
    from gui.typography import for_root

    root = tk.Tk()
    try:
        root.withdraw()
        window = tk.Toplevel(root)
        header = tk.Frame(window)
        header.pack(fill="x")
        # Without a handler there is no fourth button, so a window with nothing to
        # explain keeps exactly the trio it had.
        plain = install_window_controls(
            window, header, theme=DARK_THEME, typography=for_root(root)
        )
        root.update()
        assert plain.help_button is None
        assert len(plain.frame.winfo_children()) == 3

        # With a handler the ? control is added *left of* the trio and runs it.
        pressed: list[int] = []
        with_help = install_window_controls(
            window, header, theme=DARK_THEME, typography=for_root(root),
            on_help=lambda: pressed.append(1),
        )
        root.update()
        assert with_help.help_button is not None
        assert with_help.help_button.cget("text") == _HELP_GLYPH
        assert len(with_help.frame.winfo_children()) == 4
        with_help.help_button.invoke()
        assert pressed == [1]
    finally:
        root.destroy()


@pytest.mark.skipif(not _display_available(), reason="no display available")
def test_escape_closes_the_window() -> None:
    from gui.app import DARK_THEME
    from gui.typography import for_root

    root = tk.Tk()
    try:
        root.withdraw()
        window = tk.Toplevel(root)
        header = tk.Frame(window)
        header.pack(fill="x")
        closed: list[int] = []
        install_window_controls(
            window,
            header,
            theme=DARK_THEME,
            typography=for_root(root),
            on_close=lambda: closed.append(1),
        )
        root.update()
        # Escape closes the window, the standard keyboard affordance.
        assert window.bind("<Escape>")
        window.focus_force()
        window.event_generate("<Escape>", when="now")
        root.update()
        assert closed == [1]
    finally:
        root.destroy()
