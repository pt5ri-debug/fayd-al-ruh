"""The window's typography: OpenHands' type scale, in one auditable place.

Tk has no stylesheet, so every ``font=(family, size)`` and every
``spacing1``/``spacing3`` the window draws with used to be a literal scattered
across ``gui/app.py`` and the dialogs. This module is the single source of
truth for *type* -- exactly as :class:`gui.app.Theme` is for colour.

The scale is copied from the OpenHands frontend, so the project's windows read
like the OpenHands chat rather than a generic form:

* the body/chat text is ``text-sm`` (14 px) at ``leading-6`` (24 px), the same
  size and line-height OpenHands' ``chat-message.tsx`` uses;
* Markdown headings follow OpenHands' ``markdown/headings.tsx`` (h1 ``text-xl``
  … h4 ``text-sm``);
* a paragraph carries ``py-1.5`` (6 px above and below) and ``leading-6``,
  OpenHands' ``markdown/paragraph.tsx``;
* lists are ``my-2`` with ``leading-6`` and ``li`` ``py-0.5``; a quote is
  ``my-2`` italic with a 4 px accent bar; a table is ``my-4`` ``text-sm`` with
  ``px-3 py-2`` cells; a rule is ``my-4`` -- all straight from OpenHands'
  ``markdown/`` components.

Three Tk facts shape the numbers:

* a font size is in *points*, so a pixel size from the web scale is converted
  with the root's own ``tk scaling`` (:func:`to_points`), which keeps the scale
  honest on a high-DPI display instead of shrinking to a hair;
* Tk's weight is only ``normal`` or ``bold`` -- it has no ``font-medium``, so
  OpenHands' medium headings are drawn bold; the *size and line-height* match
  the web exactly, and only the weight is approximated;
* a ``Text`` *tag* has no ``-linespacing`` option, so line-height is emulated
  with the tag's ``spacing1``/``spacing2``/``spacing3`` (leading above a line,
  between the wrapped lines of one paragraph, and below it). A block that wants
  a 24 px line-height at a ~17 px natural line height sets ``spacing2 = 7``;
  the paragraph gap (``py-1.5``) becomes ``spacing1``/``spacing3``.
  :meth:`Typography.block_metrics` computes all three.

Nothing here draws anything, so it is testable without a display.
"""

from __future__ import annotations

from dataclasses import dataclass

# ------------------------------------------------------------ the font stacks
#
# OpenHands' ``src/index.css``: the body stack is the system UI stack (its
# ``Outfit`` import is used by the marketing shell, not the chat), and code
# uses ``source-code-pro, Menlo, Monaco, Consolas, "Courier New", monospace``.
# Tk resolves the first family it actually has, so the same list works on
# macOS, Windows, and Linux.

#: The proportional (UI + prose) stack, in OpenHands' order.
UI_FONT_STACK: tuple[str, ...] = (
    "SF Pro Text", "SF Pro", ".AppleSystemUIFont", "Helvetica Neue",
    "Segoe UI", "Roboto", "Oxygen", "Ubuntu", "Cantarell", "Fira Sans",
    "Droid Sans", "DejaVu Sans", "Liberation Sans", "Noto Sans", "Arial",
    "Helvetica",
)

#: The monospace (code) stack, in OpenHands' order.
MONO_FONT_STACK: tuple[str, ...] = (
    "SF Mono", "Menlo", "Monaco", "Consolas", "Roboto Mono", "Ubuntu Mono",
    "DejaVu Sans Mono", "Liberation Mono", "Noto Sans Mono", "Courier New",
    "Courier",
)

#: Tk's own fallbacks, used only when no family above exists.
_TK_UI_FALLBACK = "TkDefaultFont"
_TK_MONO_FALLBACK = "TkFixedFont"

#: The reference display density (96 dpi) the web scale assumes.
_BASE_SCALING = 96.0 / 72.0

#: Tk caps a font size at this many points; a larger request is clamped.
_MAX_POINTS = 256


def _tk_scaling(root: object | None) -> float:
    """The root's ``tk scaling`` (pixels per point), or 96-dpi when unknowable.

    Called with no display it returns the value Tk uses on a standard 96-dpi
    screen, so the module stays importable and testable headless.
    """
    interpreter = getattr(root, "tk", None)
    if interpreter is None:
        return _BASE_SCALING
    try:
        value = float(interpreter.call("tk", "scaling"))
    except Exception:
        return _BASE_SCALING
    if value <= 0:
        return _BASE_SCALING
    return value


def to_points(pixels: float, scaling: float) -> int:
    """Convert a web pixel size to the integer point size Tk accepts.

    Tk rejects a fractional size, so the result is rounded -- and never below
    ``1``, so a tiny caption is still legible rather than invisible.
    """
    return max(1, min(_MAX_POINTS, round(pixels / scaling)))


def pick_family(candidates: tuple[str, ...], available: set[str], fallback: str) -> str:
    """The first candidate the platform actually has, else ``fallback``.

    Kept as a pure function so the choice is testable without a display.
    """
    for name in candidates:
        if name in available:
            return name
    return fallback


# -------------------------------------------------------------- the type scale
#
# Each entry is a web pixel size from OpenHands' Tailwind classes. The keys are
# semantic *roles*, so a caller asks for ``"body"`` or ``"heading1"`` and the
# number lives here rather than at the call site.


@dataclass(frozen=True)
class TypeStep:
    """One step of the scale: a pixel size and a pixel line-height."""

    size: float     # px, from Tailwind's ``text-*``
    leading: float  # px, from Tailwind's ``leading-*``


#: The scale, keyed by role. Sizes/leadings are OpenHands' own.
SCALE: dict[str, TypeStep] = {
    # Tailwind's default ramp, for the places OpenHands uses a bare size.
    "2xs": TypeStep(10, 14),   # ``text-[10px]`` — file size, micro labels
    "xs": TypeStep(12, 16),    # ``text-xs``     — captions, chips
    "sm": TypeStep(14, 20),    # ``text-sm``     — buttons, table cells
    "base": TypeStep(16, 24),  # ``text-base``   — the composer input
    "lg": TypeStep(18, 28),    # ``text-lg``
    "xl": TypeStep(20, 28),    # ``text-xl``
    "2xl": TypeStep(24, 32),   # ``text-2xl``
    # The chat and its Markdown, from OpenHands' own components.
    "body": TypeStep(14, 24),       # chat-message.tsx: text-sm leading-6
    "chatInput": TypeStep(16, 20),  # chat-input placeholder: 16px/20px
    "heading1": TypeStep(20, 28),   # headings.tsx h1: text-xl leading-7
    "heading2": TypeStep(18, 24),   # h2: text-lg leading-6
    "heading3": TypeStep(16, 24),   # h3: text-base leading-6
    "heading4": TypeStep(14, 20),   # h4/h5/h6: text-sm
    "code": TypeStep(12, 20),       # a fenced block reads a step down
    "table": TypeStep(14, 20),      # table.tsx: text-sm
    "hero": TypeStep(24, 32),       # the empty-state title
    "heroSubtitle": TypeStep(16, 24),
    "brand": TypeStep(18, 24),
    "brandMark": TypeStep(20, 24),
}

#: Roles drawn bold. OpenHands uses ``font-medium`` for headings; Tk has no
#: medium, so the approximation is bold (see the module docstring).
_BOLD_ROLES: frozenset[str] = frozenset(
    {
        "heading1", "heading2", "heading3", "heading4", "brand", "brandMark",
        "hero",
    }
)


@dataclass(frozen=True)
class BlockMetrics:
    """The ``spacing1``/``spacing2``/``spacing3`` that realise a block's rhythm.

    ``spacing1`` is the leading above a line, ``spacing2`` the leading between
    the wrapped lines of the *same* paragraph (how a line-height is emulated),
    and ``spacing3`` the leading below it -- all in pixels, as Tk wants.
    """

    spacing1: int
    spacing2: int
    spacing3: int


@dataclass(frozen=True)
class Typography:
    """A resolved type scale: real Tk point sizes for one root's DPI.

    Build it once per window with :func:`for_root` and hand it to every widget
    and dialog, so the whole application draws from one scale.
    """

    ui_family: str
    mono_family: str
    scaling: float

    # --- sizes ------------------------------------------------------------

    def points(self, role: str) -> int:
        """The Tk point size for ``role`` (falls back to ``body``)."""
        step = SCALE.get(role, SCALE["body"])
        return to_points(step.size, self.scaling)

    def size_px(self, role: str) -> float:
        """The web pixel size for ``role`` (falls back to ``body``)."""
        return SCALE.get(role, SCALE["body"]).size

    def leading_px(self, role: str) -> float:
        """The web pixel line-height for ``role`` (falls back to ``body``)."""
        return SCALE.get(role, SCALE["body"]).leading

    def px(self, pixels: float) -> int:
        """A raw pixel size (e.g. a one-off) as a Tk point size."""
        return to_points(pixels, self.scaling)

    # --- font tuples ------------------------------------------------------

    def font(
        self, role: str = "body", *, weight: str | None = None, slant: str | None = None
    ) -> tuple:
        """A Tk font tuple for ``role``.

        The weight defaults to the role's own (bold for a heading or the brand,
        normal elsewhere); pass ``weight="bold"`` for an emphasis label or
        ``weight="normal"`` to override a role's default. ``slant="italic"``
        adds the italic style Tk takes as a further tuple element.
        """
        resolved = weight
        if resolved is None:
            resolved = "bold" if role in _BOLD_ROLES else "normal"
        styles = [resolved] if resolved != "normal" else []
        if slant is not None:
            styles.append(slant)
        return (self.ui_family, self.points(role), *styles)

    def mono(self, role: str = "code", *, weight: str | None = None) -> tuple:
        """A Tk monospace font tuple for ``role`` (default the code step)."""
        if weight in (None, "normal"):
            return (self.mono_family, self.points(role))
        return (self.mono_family, self.points(role), weight)

    # --- line and block spacing ------------------------------------------

    def block_metrics(self, role: str, *, gap_px: float = 0.0) -> BlockMetrics:
        """The Tk spacing that realises ``role``'s line-height and gap.

        ``gap_px`` is extra space above and below the block (OpenHands'
        ``mt-*``/``mb-*``/``py-*``); the natural line height is whatever the
        family draws at that point size, so ``spacing2`` is the *difference*
        the web leading asks for and never negative.
        """
        step = SCALE.get(role, SCALE["body"])
        leading = max(0, round(step.leading - self.natural_line_px(role)))
        gap = self.px(gap_px)
        return BlockMetrics(spacing1=gap, spacing2=leading, spacing3=gap)

    def natural_line_px(self, role: str) -> float:
        """The family's own line height, in pixels, for ``role``.

        Approximated as 1.2x the em -- the standard ratio for the system stacks
        -- which is all ``spacing2`` needs to top up.
        """
        return 1.2 * self.points(role) * self.scaling


def resolve(
    root: object | None = None,
    *,
    ui_family: str | None = None,
    mono_family: str | None = None,
) -> Typography:
    """Build a :class:`Typography`, honouring explicit family names.

    A dialog receives the window's chosen families as strings, so passing them
    here keeps a dialog on the very same families as the window that opened it,
    while the *sizes* still come from the one shared scale.
    """
    available = _families(root)
    return Typography(
        ui_family=ui_family or pick_family(UI_FONT_STACK, available, _TK_UI_FALLBACK),
        mono_family=mono_family or pick_family(MONO_FONT_STACK, available, _TK_MONO_FALLBACK),
        scaling=_tk_scaling(root),
    )


def for_root(root: object | None) -> Typography:
    """Build the :class:`Typography` for ``root``'s display.

    With no display (a headless test, a build machine) it returns the standard
    96-dpi scale so callers never have to guard the call.
    """
    return resolve(root)


def _families(root: object | None) -> set[str]:
    """Every font family Tk can see, or an empty set with no display."""
    try:
        from tkinter import font as tkfont

        return set(tkfont.families(root))  # type: ignore[arg-type]
    except Exception:
        return set()


#: A typography built for the default scale, for a caller with no root yet
#: (a headless import, or a test asserting the mapping).
DEFAULT_TYPOGRAPHY = Typography(
    ui_family=_TK_UI_FALLBACK,
    mono_family=_TK_MONO_FALLBACK,
    scaling=_BASE_SCALING,
)
