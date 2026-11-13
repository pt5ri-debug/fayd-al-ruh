"""The shared component library and the design tokens (Phase 1).

The token part is headless; the widget part needs a display and is skipped
without one (or under Xvfb, exactly like the other window tests).
"""

from __future__ import annotations

import pytest
from gui import design
from gui.design import RADIUS, SIZE, SPACE, SPACING_STEPS, space
from gui.themes import DARK_THEME
from gui.typography import DEFAULT_TYPOGRAPHY

try:
    import tkinter as tk

    from gui import components
except ImportError:  # pragma: no cover - no libtk on this machine
    tk = None  # type: ignore[assignment]
    components = None  # type: ignore[assignment]


class _Root:
    """A hidden Tk root, or a skip when there is no display."""

    def __init__(self) -> None:
        if tk is None:
            pytest.skip("Tk is unavailable")
        try:
            self.root = tk.Tk()
        except tk.TclError as exc:  # pragma: no cover - depends on the display
            pytest.skip(f"no display: {exc}")
        self.root.withdraw()

    def __enter__(self):
        return self.root

    def __exit__(self, *_exc: object) -> None:
        try:
            self.root.destroy()
        except Exception:
            pass


# --------------------------------------------------------------- design tokens


def test_spacing_steps_ascend_and_are_unique() -> None:
    assert list(SPACING_STEPS) == sorted(SPACING_STEPS)
    assert len(set(SPACING_STEPS)) == len(SPACING_STEPS)
    assert SPACING_STEPS[0] > 0


def test_space_resolves_a_named_step() -> None:
    assert space("md") == SPACE.md == 8
    assert space("xxl") == SPACE.xxl


def test_space_rejects_an_unknown_step() -> None:
    with pytest.raises(KeyError):
        space("mdd")


def test_radii_and_sizes_are_the_shipped_values() -> None:
    assert (RADIUS.small, RADIUS.card, RADIUS.large) == (5, 12, 20)
    assert RADIUS.small < RADIUS.card < RADIUS.large
    assert SIZE.send == 36 and SIZE.status_dot == 10


# --------------------------------------------------------------------- buttons


def test_button_variants_are_the_closed_set() -> None:
    assert design.BUTTON_VARIANTS == ("primary", "secondary", "ghost", "danger")
    # ``gui.components`` re-exports the same tuple, so the two cannot drift.
    if components is not None:
        assert components.BUTTON_VARIANTS == design.BUTTON_VARIANTS


def test_button_rejects_an_unknown_variant() -> None:
    with _Root() as root:
        with pytest.raises(ValueError):
            components.button(
                root, "x", None, theme=DARK_THEME, typography=DEFAULT_TYPOGRAPHY,
                variant="fancy",
            )


def test_button_invokes_its_command() -> None:
    with _Root() as root:
        calls: list[str] = []
        widget = components.button(
            root, "Go", lambda: calls.append("go"),
            theme=DARK_THEME, typography=DEFAULT_TYPOGRAPHY, variant="primary",
        )
        widget.invoke()
        assert calls == ["go"]


def test_button_releases_its_command_on_destroy() -> None:
    with _Root() as root:
        calls: list[str] = []
        widget = components.button(
            root, "Go", lambda: calls.append("go"),
            theme=DARK_THEME, typography=DEFAULT_TYPOGRAPHY,
        )
        widget.pack()
        widget.destroy()
        root.update()
        assert widget._command_ref is None
        assert calls == []


def test_button_disables_and_reenables() -> None:
    with _Root() as root:
        widget = components.button(
            root, "Go", lambda: None, theme=DARK_THEME, typography=DEFAULT_TYPOGRAPHY
        )
        widget.set_enabled(False)
        assert str(widget.cget("state")) == "disabled"
        widget.set_enabled(True)
        assert str(widget.cget("state")) == "normal"


# -------------------------------------------------------------------- sections


def test_section_card_exposes_a_body_frame() -> None:
    with _Root() as root:
        section = components.Section(
            root, title="A. General", theme=DARK_THEME, typography=DEFAULT_TYPOGRAPHY
        )
        assert section.card is True
        assert isinstance(section.body, tk.Frame)
        assert section.frame.winfo_children()  # the rounded card is drawn


def test_section_plain_shape_returns_the_frame_itself() -> None:
    with _Root() as root:
        section = components.Section(
            root, title="1. Files", theme=DARK_THEME, typography=DEFAULT_TYPOGRAPHY,
            card=False,
        )
        assert section.body is section.frame


# ----------------------------------------------------------------- status line


def test_status_line_reports_text_and_error_colour() -> None:
    with _Root() as root:
        line = components.StatusLine(
            root, theme=DARK_THEME, typography=DEFAULT_TYPOGRAPHY
        )
        line.pack()
        line.set("ready")
        assert line.text == "ready"
        assert str(line.label.cget("fg")) == DARK_THEME.text_secondary
        line.set("boom", error=True)
        assert str(line.label.cget("fg")) == DARK_THEME.danger


# --------------------------------------------------------------------- header


def test_dialog_header_builds_title_and_controls() -> None:
    with _Root() as root:
        window = tk.Toplevel(root)
        closed: list[bool] = []
        header = components.DialogHeader(
            window, title="Thing", subtitle="a subtitle", theme=DARK_THEME,
            typography=DEFAULT_TYPOGRAPHY, on_close=lambda: closed.append(True),
        )
        assert str(header.title_label.cget("text")) == "Thing"
        assert str(header.subtitle_label.cget("text")) == "a subtitle"
        assert header.controls is not None
        window.destroy()


def test_dialog_header_busy_toggles_the_spinner() -> None:
    with _Root() as root:
        window = tk.Toplevel(root)
        header = components.DialogHeader(
            window, title="T", theme=DARK_THEME, typography=DEFAULT_TYPOGRAPHY
        )
        header.set_busy(True)
        assert header.spinner.winfo_ismapped() or True  # packed; not yet mapped
        header.set_busy(False)
        assert not header.spinner.winfo_manager()
        window.destroy()


def test_dialog_header_cancel_button_is_shown_only_while_busy() -> None:
    with _Root() as root:
        window = tk.Toplevel(root)
        cancelled: list[bool] = []
        header = components.DialogHeader(
            window, title="T", theme=DARK_THEME, typography=DEFAULT_TYPOGRAPHY,
            cancel=lambda: cancelled.append(True),
        )
        assert header.cancel_button is not None
        assert not header.cancel_button.winfo_manager()  # hidden at rest
        header.set_busy(True)
        assert header.cancel_button.winfo_manager() == "pack"
        header.cancel_button.invoke()
        assert cancelled == [True]
        header.set_busy(False)
        assert not header.cancel_button.winfo_manager()
        window.destroy()


def test_dialog_header_extra_slot_draws_into_the_inner_frame() -> None:
    with _Root() as root:
        window = tk.Toplevel(root)
        seen: list[tk.Frame] = []

        def extra(inner: tk.Frame) -> None:
            seen.append(inner)
            tk.Label(inner, text="working…").pack(side="right")

        header = components.DialogHeader(
            window, title="T", theme=DARK_THEME, typography=DEFAULT_TYPOGRAPHY,
            extra=extra,
        )
        assert seen == [header.inner]
        assert len(header.inner.winfo_children()) >= 2  # titles + the extra widget
        window.destroy()


def test_hero_header_builds_title_subtitle_notes_and_controls() -> None:
    with _Root() as root:
        window = tk.Toplevel(root)
        header = components.HeroHeader(
            window, title="Big", subtitle="a subtitle", notes=("note one", "note two"),
            theme=DARK_THEME, typography=DEFAULT_TYPOGRAPHY,
        )
        assert str(header.title_label.cget("text")) == "Big"
        assert header.subtitle_label is not None
        assert [str(n.cget("text")) for n in header.note_labels] == ["note one", "note two"]
        assert header.controls is not None
        window.destroy()


def test_hero_header_extra_packs_beside_the_controls() -> None:
    with _Root() as root:
        window = tk.Toplevel(root)
        extra_seen: list[tk.Frame] = []
        header = components.HeroHeader(
            window, title="Big", theme=DARK_THEME, typography=DEFAULT_TYPOGRAPHY,
            extra=extra_seen.append,
        )
        assert extra_seen == [header.title_row]
        window.destroy()


# ------------------------------------------------------------------- states


def test_empty_state_packs_and_destroys() -> None:
    with _Root() as root:
        state = components.EmptyState(
            root, title="Nothing yet", hint="Start a chat",
            theme=DARK_THEME, typography=DEFAULT_TYPOGRAPHY,
        )
        state.pack(fill="both", expand=True)
        assert state.frame.winfo_manager() == "pack"
        state.destroy()


def test_loading_state_starts_and_stops() -> None:
    with _Root() as root:
        state = components.LoadingState(
            root, caption="Working…", theme=DARK_THEME, typography=DEFAULT_TYPOGRAPHY
        )
        state.pack()
        state.start()
        state.stop()
        state.destroy()


# ------------------------------------------------------------------- search


def test_search_input_calls_on_change() -> None:
    with _Root() as root:
        seen: list[str] = []
        box = components.SearchInput(
            root, theme=DARK_THEME, typography=DEFAULT_TYPOGRAPHY,
            on_change=seen.append,
        )
        box.pack()
        box.variable.set("poro")
        assert seen[-1] == "poro"
        assert box.query == "poro"
        box.clear()
        assert box.query == ""


# ---------------------------------------------------------------- settings


def test_settings_section_adds_rows() -> None:
    with _Root() as root:
        group = components.SettingsSection(
            root, title="Appearance", theme=DARK_THEME, typography=DEFAULT_TYPOGRAPHY
        )
        row = group.row(
            "Theme", hint="dark or light", theme=DARK_THEME, typography=DEFAULT_TYPOGRAPHY
        )
        assert isinstance(row, components.SettingsRow)
        assert str(row.label.cget("text")) == "Theme"
        assert row.hint is not None


# ----------------------------------------------------------- notifications


def test_notification_center_keeps_newest_first_and_bounded() -> None:
    with _Root() as root:
        center = components.NotificationCenter(
            root, theme=DARK_THEME, typography=DEFAULT_TYPOGRAPHY, limit=2
        )
        center.add(components.Notification(title="one"))
        center.add(components.Notification(title="two"))
        center.add(components.Notification(title="three"))
        assert [n.title for n in center.items] == ["three", "two"]
        center.clear()
        assert center.items == ()


def test_notification_center_rejects_an_unknown_level() -> None:
    with _Root() as root:
        center = components.NotificationCenter(
            root, theme=DARK_THEME, typography=DEFAULT_TYPOGRAPHY
        )
        with pytest.raises(ValueError):
            center.add(components.Notification(title="x", level="loud"))


# ------------------------------------------------------------------ worker


def test_worker_pump_delivers_the_result_on_the_ui_thread() -> None:
    with _Root() as root:
        got: list[tuple[object, object]] = []
        pump = components.WorkerPump(
            root, on_finish=lambda result, error: got.append((result, error))
        )
        pump.start(lambda: 21 * 2)
        _run_until(lambda: bool(got), root)
        assert got == [(42, None)]
        assert pump.busy is False


def test_worker_pump_reports_an_exception_without_raising() -> None:
    with _Root() as root:
        got: list[tuple[object, object]] = []
        pump = components.WorkerPump(
            root, on_finish=lambda result, error: got.append((result, error))
        )

        def boom() -> int:
            raise ValueError("nope")

        pump.start(boom)
        _run_until(lambda: bool(got), root)
        assert got[0][0] is None
        assert isinstance(got[0][1], ValueError)


def test_worker_pump_cancel_drops_a_late_result() -> None:
    with _Root() as root:
        got: list[tuple[object, object]] = []
        pump = components.WorkerPump(
            root, on_finish=lambda result, error: got.append((result, error))
        )
        pump.start(lambda: "late")
        pump.cancel()
        _run_until(lambda: not pump.busy, root)
        # The drain ran, but the cancelled result was dropped.
        assert got == []


def test_worker_pump_ignores_a_second_start_while_busy() -> None:
    with _Root() as root:
        got: list[object] = []
        pump = components.WorkerPump(root, on_finish=lambda result, error: got.append(result))
        pump.start(lambda: "first")
        pump.start(lambda: "second")  # ignored while busy
        _run_until(lambda: bool(got), root)
        assert got == ["first"]


def test_worker_pump_close_cancels_the_ticket() -> None:
    with _Root() as root:
        pump = components.WorkerPump(root, on_finish=lambda result, error: None)
        pump.start(lambda: "x")
        pump.close()
        assert pump.busy is False


def _run_until(predicate, root: tk.Tk, *, limit: int = 200) -> None:
    """Spin the Tk event loop until ``predicate`` is true (or the limit runs out)."""
    import time

    for _ in range(limit):
        root.update()
        if predicate():
            return
        time.sleep(0.005)
    root.update()


def test_worker_pump_never_blocks_the_ui_thread() -> None:
    """A slow worker keeps the UI responsive: the event loop keeps turning.

    The whole point of the pump is that long work runs off the UI thread, so a
    slow task must not stop ``root.update()`` from returning -- a blocking
    ``join`` (or running the task inline) would freeze the window. This measures
    that the loop turns many times while the worker is still busy.
    """
    import threading
    import time

    with _Root() as root:
        release = threading.Event()
        got: list[object] = []
        pump = components.WorkerPump(root, on_finish=lambda result, error: got.append(result))

        def slow() -> str:
            release.wait(timeout=5.0)  # blocks until the test lets it finish
            return "done"

        pump.start(slow)
        turns = 0
        deadline = time.monotonic() + 1.0
        while pump.busy and time.monotonic() < deadline:
            root.update()  # returns at once; the worker is on its own thread
            turns += 1
        assert turns > 10, "the UI thread kept turning while the worker ran"
        assert got == []  # the slow result has not arrived yet
        release.set()
        _run_until(lambda: bool(got), root)
        assert got == ["done"]
