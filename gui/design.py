"""The one place the design system's non-colour tokens live.

A :class:`~gui.themes.Theme` already owns every **colour** the program paints
with, and :mod:`gui.typography` owns the **type scale**. What was still a literal
copied into every window was everything in between: the corner radii, the panel
padding, the vertical gaps, the icon sizes. Two windows that both "look right"
could still disagree by a few pixels -- which is exactly how a set of windows
stops reading as one program.

This module is the single source of truth for those numbers, so a window asks for
``SPACE.lg`` instead of typing ``18``. It is deliberately **tkinter-free**: the
values are data, so they can be inspected and tested with no display, and a
window imports them without pulling Tk in.

The values themselves are the ones the shipped windows already used (HeroUI's
small/large radii, a 4/6/8/12/18/20 spacing rhythm), so adopting this module is a
*renaming*, never a visual change -- a window that reads ``RADIUS.card`` gets the
same ``12`` it typed before.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Spacing:
    """The vertical/horizontal rhythm, named by role rather than by number.

    ``xxs``..``xxl`` are the steps the shipped windows already use; naming them
    lets a window say *how much room* it wants (a hairline gap, a section gap)
    instead of repeating a pixel figure that can drift.
    """

    xxs: int = 2
    xs: int = 4
    sm: int = 6
    md: int = 8
    lg: int = 12
    xl: int = 18
    xxl: int = 20
    #: The gap above a card and the inner padding of a card.
    card_gap: int = 10
    card_pad: int = 14

    def steps(self) -> tuple[int, ...]:
        """Every step, smallest first -- for a test that proves they ascend."""
        return (self.xxs, self.xs, self.sm, self.md, self.lg, self.xl, self.xxl)


#: The shared spacing rhythm.
SPACE = Spacing()


@dataclass(frozen=True)
class Radii:
    """The corner radii. HeroUI's small/large pair plus the card radius."""

    small: int = 5      # buttons, chips, inputs (HeroUI ``radius.small``)
    card: int = 12      # a section card
    large: int = 20     # the composer and other big surfaces (HeroUI ``radius.large``)
    pill: int = 999     # a fully rounded chip (drawn as half the height by Tk)


RADIUS = Radii()


@dataclass(frozen=True)
class Sizes:
    """Fixed control sizes shared across windows."""

    send: int = 36          # the Send disc's diameter
    icon: int = 16          # an inline icon/glyph box
    status_dot: int = 10    # the composer's status dot
    spinner: int = 120      # the busy progress bar's length


SIZE = Sizes()


@dataclass(frozen=True)
class TitleBar:
    """The proportions of every window's title bar.

    The bar is the one band a user sees first in every window, so its rhythm is
    defined once here rather than nudged per dialog: the height of the band, the
    icon badge's size and radius, the gap between the badge and the titles, and
    the size of the window-control buttons. A dialog asks for these instead of
    typing pixel figures, which is what keeps twenty-five title bars looking like
    one program.
    """

    #: The minimum height of the band (a title and a subtitle need the room).
    min_height: int = 56
    #: The icon badge: a rounded square holding the window's own glyph.
    badge: int = 34
    badge_radius: int = 10
    #: The gap between the badge and the title block.
    gap: int = 12
    #: The window-control buttons' square size (minimize / maximize / close).
    control: int = 26
    #: The gap between two window controls.
    control_gap: int = 3


TITLEBAR = TitleBar()


#: The named spacing steps, so a test can assert the rhythm is monotonic and a
#: caller can iterate them (e.g. a "spacing scale" section in a style guide).
SPACING_STEPS: tuple[int, ...] = SPACE.steps()


def space(step: str) -> int:
    """One spacing step by name (``"md"`` -> 8), raising on an unknown name.

    A typo must be reported, not silently treated as zero: a window that asks for
    ``SPACE.mdd`` and gets ``0`` would draw a gap-less layout that looks broken.
    """
    if not hasattr(SPACE, step):
        raise KeyError(step)
    return int(getattr(SPACE, step))


#: The button variants. ``primary`` is the one call to action on a screen (a gold
#: fill); ``secondary`` is the ordinary raised-surface button; ``ghost`` is a
#: borderless button for a toolbar; ``danger`` is the destructive one. They live
#: here, beside the other non-colour tokens, so a headless caller can read them
#: without importing Tk -- ``gui.components`` re-exports them unchanged.
VARIANT_PRIMARY = "primary"
VARIANT_SECONDARY = "secondary"
VARIANT_GHOST = "ghost"
VARIANT_DANGER = "danger"
BUTTON_VARIANTS: tuple[str, ...] = (
    VARIANT_PRIMARY,
    VARIANT_SECONDARY,
    VARIANT_GHOST,
    VARIANT_DANGER,
)


__all__ = [
    "BUTTON_VARIANTS",
    "RADIUS",
    "SIZE",
    "SPACE",
    "SPACING_STEPS",
    "TITLEBAR",
    "VARIANT_DANGER",
    "VARIANT_GHOST",
    "VARIANT_PRIMARY",
    "VARIANT_SECONDARY",
    "Radii",
    "Sizes",
    "Spacing",
    "TitleBar",
    "space",
]
