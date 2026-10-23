"""The one type scale, and the shared Markdown renderer built on it.

The scale is pure arithmetic, so most of this runs headless; the renderer is
exercised only when a display (or Xvfb) is available, exactly like the other
window tests.
"""

from __future__ import annotations

import tkinter as tk

import pytest
from gui import rich_text
from gui import typography as ty


def _display_available() -> bool:
    try:
        root = tk.Tk()
    except tk.TclError:
        return False
    root.destroy()
    return True


# --- the scale ------------------------------------------------------------


def test_to_points_converts_a_pixel_size_with_the_root_scaling() -> None:
    # 96 dpi: 1 px == 0.75 pt, so the 14 px body is 10.5 pt -> 10 (Python's
    # round-half-to-even); Tk wants a whole number either way.
    assert ty.to_points(14, 96.0 / 72.0) == 10
    # A denser display asks for *fewer* points for the same pixel size.
    assert ty.to_points(14, 2.0) == 7
    assert ty.to_points(14, 1.0) == 14


def test_to_points_never_returns_zero_or_overflow() -> None:
    assert ty.to_points(0, 96.0 / 72.0) == 1
    assert ty.to_points(-5, 1.0) == 1
    assert ty.to_points(10_000, 1.0) == ty._MAX_POINTS


def test_pick_family_prefers_the_first_available_candidate() -> None:
    assert ty.pick_family(("Outfit", "DejaVu Sans"), {"DejaVu Sans"}, "TkDefaultFont") == "DejaVu Sans"
    assert ty.pick_family(("Outfit",), set(), "TkDefaultFont") == "TkDefaultFont"


def test_headless_typography_uses_the_reference_density() -> None:
    scale = ty.for_root(None)
    assert scale.scaling == ty._BASE_SCALING
    # No display means no families are visible, so Tk's own fallbacks are used.
    assert scale.ui_family == ty._TK_UI_FALLBACK
    assert scale.mono_family == ty._TK_MONO_FALLBACK


def test_resolve_honours_the_windows_chosen_families() -> None:
    """A dialog receives the window's families, so it must keep them exactly."""
    scale = ty.resolve(None, ui_family="DejaVu Sans", mono_family="DejaVu Sans Mono")
    assert scale.ui_family == "DejaVu Sans"
    assert scale.mono_family == "DejaVu Sans Mono"


def test_font_tuple_defaults_weight_by_role() -> None:
    scale = ty.resolve(None, ui_family="DejaVu Sans", mono_family="DejaVu Sans Mono")
    assert scale.font("body") == ("DejaVu Sans", scale.points("body"))
    # A heading is bold by its role, without the caller asking.
    assert scale.font("heading1") == ("DejaVu Sans", scale.points("heading1"), "bold")
    # An explicit weight overrides the role's own.
    assert scale.font("heading1", weight="normal") == ("DejaVu Sans", scale.points("heading1"))
    assert scale.font("body", weight="bold") == ("DejaVu Sans", scale.points("body"), "bold")


def test_font_tuple_carries_an_italic_slant() -> None:
    scale = ty.resolve(None, ui_family="DejaVu Sans", mono_family="DejaVu Sans Mono")
    assert scale.font("body", slant="italic") == ("DejaVu Sans", scale.points("body"), "italic")
    assert scale.font("heading1", slant="italic") == (
        "DejaVu Sans", scale.points("heading1"), "bold", "italic",
    )


def test_mono_font_uses_the_monospace_family() -> None:
    scale = ty.resolve(None, ui_family="DejaVu Sans", mono_family="DejaVu Sans Mono")
    assert scale.mono("code") == ("DejaVu Sans Mono", scale.points("code"))
    assert scale.mono("code", weight="bold") == (
        "DejaVu Sans Mono", scale.points("code"), "bold",
    )


def test_unknown_role_falls_back_to_body() -> None:
    scale = ty.for_root(None)
    assert scale.points("no-such-role") == scale.points("body")
    assert scale.size_px("no-such-role") == scale.size_px("body")


def test_block_metrics_realise_the_web_leading_and_gap() -> None:
    scale = ty.for_root(None)
    # body is text-sm leading-6: 14 px text, 24 px line-height.
    assert scale.leading_px("body") == 24
    metrics = scale.block_metrics("body", gap_px=1)
    # The gap is the paragraph's py-1.5 (rounded up to a whole point).
    assert metrics.spacing1 == metrics.spacing3 == scale.px(1)
    # spacing2 is what is *left* of the line-height after the family's own line,
    # and never negative.
    assert metrics.spacing2 >= 0
    assert metrics.spacing2 == max(0, round(24 - scale.natural_line_px("body")))


def test_block_metrics_never_asks_for_a_negative_leading() -> None:
    # A role whose natural line already exceeds its web leading must not produce
    # a negative spacing2 -- Tk would reject it.
    scale = ty.for_root(None)
    for role in ty.SCALE:
        assert scale.block_metrics(role, gap_px=4).spacing2 >= 0


# --- the shared renderer --------------------------------------------------


@pytest.mark.skipif(not _display_available(), reason="no display available")
def test_insert_markdown_round_trips_and_tags_every_block() -> None:
    root = tk.Tk()
    try:
        widget = tk.Text(root)
        scale = ty.for_root(root)
        text = (
            "# H1\n\n## H2\n\n### H3\n\nA paragraph.\n\n- one\n- two\n\n1. first\n\n"
            "> quote\n\n```python\nx = 1\n```\n\n| a | b |\n|---|---|\n| 1 | 2 |\n\n---"
        )
        rich_text.insert_markdown(widget, text, theme=_theme(), typography=scale)
        assert widget.get("1.0", "end-1c") == text  # exact round-trip
        for tag in (
            rich_text.HEADING1_TAG, rich_text.HEADING2_TAG, rich_text.HEADING3_TAG,
            rich_text.PARAGRAPH_TAG, rich_text.BULLET_TAG, rich_text.QUOTE_TAG,
            rich_text.CODE_TAG, rich_text.TABLE_TAG, rich_text.RULE_TAG,
        ):
            assert widget.tag_ranges(tag), f"missing tag {tag}"
    finally:
        root.destroy()


@pytest.mark.skipif(not _display_available(), reason="no display available")
def test_apply_body_spacing_gives_a_blank_line_its_height() -> None:
    """The blank line between two paragraphs must carry the body rhythm."""
    root = tk.Tk()
    try:
        widget = tk.Text(root)
        rich_text.apply_body_spacing(widget, ty.for_root(root))
        assert int(widget.cget("spacing1")) > 0
        assert int(widget.cget("spacing3")) > 0
    finally:
        root.destroy()


@pytest.mark.skipif(not _display_available(), reason="no display available")
def test_every_dialog_shares_the_windows_scale() -> None:
    """A dialog built with the window's families draws at the same point sizes."""
    from gui.app import _pick_font, _pick_mono

    window_scale = ty.resolve(None, ui_family=_pick_font(), mono_family=_pick_mono())
    dialog_scale = ty.resolve(None, ui_family=_pick_font(), mono_family=_pick_mono())
    assert window_scale.font("body") == dialog_scale.font("body")
    assert window_scale.mono("code") == dialog_scale.mono("code")


class _Theme:
    """The handful of colour tokens the renderer reads (a Theme stand-in)."""

    contrast = "#ffffff"
    text_secondary = "#cccccc"
    text_dim = "#999999"
    surface_raised = "#222222"
    code_bg = "#111111"
    code_fg = "#eeeeee"
    foreground = "#ffffff"
    border = "#333333"
    primary = "#ffcc00"
    success = "#33cc33"


def _theme() -> _Theme:
    return _Theme()
