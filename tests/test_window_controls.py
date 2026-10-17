"""The shared minimize / maximize controls every dialog window installs.

Two levels are tested, matching how the module is built:

* the branch logic (``state("zoomed")`` where the window manager supports it, a
  real full-screen window otherwise, and the honest restore that undoes exactly
  what was done) through small duck-typed fakes, so it is deterministic on any
  platform and needs no display;
* the real installation into a Tk header, which skips when no display is
  available, like the other window tests.
"""

from __future__ import annotations

import tkinter as tk
from typing import Any

import pytest

from gui.window_controls import (
    _MAXIMIZE_GLYPH,
    _RESTORE_GLYPH,
    WindowControls,
    install_window_controls,
)


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
    full-screen fallback can be exercised without a particular window manager.
    """

    def __init__(self, *, supports_zoom: bool = True) -> None:
        self.supports_zoom = supports_zoom
        self.iconified = 0
        self.states: list[str] = []
        self.fullscreen: dict[str, bool] = {}

    def iconify(self) -> None:
        self.iconified += 1

    def state(self, value: str | None = None) -> str:
        if value is None:
            return self.states[-1] if self.states else "normal"
        if value == "zoomed" and not self.supports_zoom:
            raise tk.TclError("bad state")
        self.states.append(value)
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


def test_maximize_falls_back_to_fullscreen_without_the_zoomed_state() -> None:
    """The requested fallback: a window manager with no "zoomed" still maximises."""
    controls, window, button = _controls(supports_zoom=False)
    controls.maximize()
    assert window.fullscreen["-fullscreen"] is True
    assert button.text == _RESTORE_GLYPH


def test_toggle_maximizes_then_restores_the_same_way() -> None:
    controls, window, button = _controls(supports_zoom=True)
    controls.toggle()
    assert window.states == ["zoomed"]
    controls.toggle()
    assert window.states == ["zoomed", "normal"]
    assert button.text == _MAXIMIZE_GLYPH


def test_restore_undoes_a_fullscreen_maximize() -> None:
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


# --- the real installation (needs a display) -------------------------------


def _display_available() -> bool:
    try:
        root = tk.Tk()
    except Exception:
        return False
    root.destroy()
    return True


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
        # The two buttons live in one frame in the header.
        assert controls.frame.master is header
        assert len(controls.frame.winfo_children()) == 2
        assert controls.maximize_button.cget("text") == _MAXIMIZE_GLYPH
        # The close protocol now carries the caller's handler rather than Tk's
        # default: querying it returns the registered command string.
        assert window.protocol("WM_DELETE_WINDOW")
    finally:
        root.destroy()
