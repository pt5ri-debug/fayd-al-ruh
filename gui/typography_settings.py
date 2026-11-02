"""Word-like font and paragraph settings, in one place, per *scope*.

The window's type comes from :mod:`gui.typography` (OpenHands' own scale). This
module is the **user's own** layer over it: a small, persisted profile that can
change the font family, the font size, bold/italic, the paragraph alignment,
the line spacing, and the space before/after a paragraph -- the properties a
writer reaches for in Microsoft Word.

Two **scopes** are kept apart on purpose, because the request is that the
windows' lists and the chat area are tuned separately:

* ``windows`` -- every dialog and the main window's own chrome (sidebar, menus,
  buttons, the composer of a studio): the *interface* font;
* ``chat`` -- the main window's conversation (the message bubbles and their
  Markdown) and its composer: the *reading* font.

Each scope has its own :class:`TypographyProfile`, stored separately through
:class:`storage.json_store.JsonStore` under ``gui``/``typography``, so a change
to one never disturbs the other, and a corrupt or missing store reads as the
project's own defaults rather than crashing a window.

The module is deliberately **tkinter-free** so the profiles can be built,
validated, and tested with no display; the only display-aware helper,
:func:`available_families`, is called by the window when it opens the panel.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace

from gui.typography import MONO_FONT_STACK, UI_FONT_STACK

#: The namespace and key every profile is stored under.
TYPOGRAPHY_NAMESPACE = "gui"
TYPOGRAPHY_KEY = "typography"

#: The three scopes, in the order the panel lists them.
SCOPE_WINDOWS = "windows"
SCOPE_LISTS = "lists"
SCOPE_CHAT = "chat"
SCOPES: tuple[str, ...] = (SCOPE_WINDOWS, SCOPE_LISTS, SCOPE_CHAT)
SCOPE_LABELS: dict[str, str] = {
    SCOPE_WINDOWS: "Windows & menus",
    SCOPE_LISTS: "Window lists",
    SCOPE_CHAT: "Chat area",
}
SCOPE_SUMMARIES: dict[str, str] = {
    SCOPE_WINDOWS: "Every dialog and the sidebar — the interface font",
    SCOPE_LISTS: "The list boxes inside the dialogs (projects, files, models, slides)",
    SCOPE_CHAT: "The conversation bubbles and the message box — the reading font",
}
#: Which controls each scope really honours. A ``Listbox`` draws a font but has no
#: paragraph alignment, line spacing, or space-before/after, so the *lists* scope
#: offers only the font, the size, and the bold/italic styles -- a control that
#: would do nothing is not shown.
SCOPE_CONTROLS: dict[str, tuple[str, ...]] = {
    SCOPE_WINDOWS: ("font", "size", "styles", "align", "line_spacing", "space"),
    SCOPE_LISTS: ("font", "size", "styles"),
    SCOPE_CHAT: ("font", "size", "styles", "align", "line_spacing", "space"),
}


def scope_shows(scope: str, feature: str) -> bool:
    """Whether ``scope``'s block offers ``feature`` (an unknown scope shows none)."""
    return feature in SCOPE_CONTROLS.get(scope, ())


#: Word's paragraph alignments that Tk's ``Text`` can actually draw. Tk has no
#: full-justify option (``justify`` accepts only left/right/center), so the
#: panel offers the three it can honour rather than a control that would lie.
ALIGNMENTS: tuple[str, ...] = ("left", "center", "right")
ALIGNMENT_LABELS: dict[str, str] = {
    "left": "Left",
    "center": "Center",
    "right": "Right",
}
#: The glyph each alignment is drawn with in the panel.
ALIGNMENT_GLYPHS: dict[str, str] = {
    "left": "⬅", "center": "⬌", "right": "➡",
}
#: Shown under the alignment row, so the missing Word option is stated, not hidden.
ALIGNMENT_NOTE = "Tk draws left, center, or right; full justification is not available."
#: Shown under the *Window lists* block: a ``Listbox`` draws a font but has no
#: paragraph controls, so the block says why it is shorter rather than looking
#: broken.
LISTS_NOTE = "A list box draws a font and its size; paragraph controls do not apply to it."

#: Word's line-spacing presets, as a multiplier of the type scale's own leading.
LINE_SPACINGS: tuple[float, ...] = (1.0, 1.15, 1.5, 2.0)
LINE_SPACING_LABELS: dict[float, str] = {
    1.0: "Single", 1.15: "1.15", 1.5: "1.5", 2.0: "Double",
}

#: Word's font-size presets, in points (the body text size).
FONT_SIZES: tuple[int, ...] = (8, 9, 10, 11, 12, 14, 16, 18, 20, 24, 28, 32)

#: Word's "space before/after paragraph" presets, in pixels.
SPACE_STEPS: tuple[int, ...] = (0, 2, 4, 6, 10, 14)

#: Bounds a profile is clamped to, so a hand-edited file cannot break a window.
MIN_SIZE_PT = 6
MAX_SIZE_PT = 72
MIN_LINE_SPACING = 1.0
MAX_LINE_SPACING = 3.0
MAX_SPACE_PX = 40


@dataclass(frozen=True)
class ParagraphStyle:
    """A Word-like paragraph: alignment, line spacing, and space before/after."""

    alignment: str = "left"
    line_spacing: float = 1.0
    space_before: int = 0
    space_after: int = 0

    def validated(self) -> ParagraphStyle:
        """A copy clamped to the allowed ranges (a bad field falls to default)."""
        alignment = self.alignment if self.alignment in ALIGNMENTS else "left"
        spacing = _clamp_float(self.line_spacing, MIN_LINE_SPACING, MAX_LINE_SPACING, 1.0)
        return ParagraphStyle(
            alignment=alignment,
            line_spacing=spacing,
            space_before=_clamp_int(self.space_before, 0, MAX_SPACE_PX, 0),
            space_after=_clamp_int(self.space_after, 0, MAX_SPACE_PX, 0),
        )

    def is_default(self) -> bool:
        return self.validated() == ParagraphStyle()

    def describe(self) -> str:
        """A short readable summary for the panel's status line."""
        parts = [ALIGNMENT_LABELS.get(self.alignment, self.alignment)]
        parts.append(f"line {LINE_SPACING_LABELS.get(self.line_spacing, f'{self.line_spacing:g}')}")
        if self.space_before or self.space_after:
            parts.append(f"space {self.space_before}/{self.space_after}")
        return " · ".join(parts)

    def to_primitive(self) -> dict:
        return {
            "alignment": self.alignment,
            "line_spacing": self.line_spacing,
            "space_before": self.space_before,
            "space_after": self.space_after,
        }

    @classmethod
    def from_primitive(cls, data: object) -> ParagraphStyle:
        if not isinstance(data, dict):
            return cls()
        return cls(
            alignment=str(data.get("alignment", "left")),
            line_spacing=_as_float(data.get("line_spacing"), 1.0),
            space_before=_as_int(data.get("space_before"), 0),
            space_after=_as_int(data.get("space_after"), 0),
        ).validated()


@dataclass(frozen=True)
class TypographyProfile:
    """One scope's own font and paragraph choices.

    An empty ``family``/``mono_family`` and a ``size_pt`` of ``0`` mean "keep the
    project's own", so a profile that was never edited leaves every window
    exactly as it shipped.
    """

    family: str = ""
    mono_family: str = ""
    size_pt: int = 0
    bold: bool = False
    italic: bool = False
    paragraph: ParagraphStyle = field(default_factory=ParagraphStyle)

    def validated(self) -> TypographyProfile:
        """A copy clamped to the allowed ranges, with an unknown family dropped."""
        return TypographyProfile(
            family=str(self.family or ""),
            mono_family=str(self.mono_family or ""),
            size_pt=_clamp_int(self.size_pt, 0, MAX_SIZE_PT, 0),
            bold=bool(self.bold),
            italic=bool(self.italic),
            paragraph=self.paragraph.validated(),
        )

    def is_default(self) -> bool:
        return self.validated() == TypographyProfile()

    def describe(self) -> str:
        """A short readable summary for the panel's status line."""
        parts: list[str] = []
        parts.append(self.family or "Project font")
        if self.size_pt:
            parts.append(f"{self.size_pt} pt")
        styles = [name for name, on in (("Bold", self.bold), ("Italic", self.italic)) if on]
        parts.extend(styles)
        summary = " · ".join(parts)
        return f"{summary} — {self.paragraph.describe()}"

    def to_primitive(self) -> dict:
        return {
            "family": self.family,
            "mono_family": self.mono_family,
            "size_pt": self.size_pt,
            "bold": self.bold,
            "italic": self.italic,
            "paragraph": self.paragraph.to_primitive(),
        }

    @classmethod
    def from_primitive(cls, data: object) -> TypographyProfile:
        if not isinstance(data, dict):
            return cls()
        return cls(
            family=str(data.get("family", "") or ""),
            mono_family=str(data.get("mono_family", "") or ""),
            size_pt=_as_int(data.get("size_pt"), 0),
            bold=bool(data.get("bold", False)),
            italic=bool(data.get("italic", False)),
            paragraph=ParagraphStyle.from_primitive(data.get("paragraph")),
        ).validated()

    # -- Word-like setters (each returns a new, validated profile) ----------

    def with_family(self, family: str) -> TypographyProfile:
        return replace(self, family=family).validated()

    def with_mono_family(self, family: str) -> TypographyProfile:
        return replace(self, mono_family=family).validated()

    def with_size(self, size_pt: int) -> TypographyProfile:
        return replace(self, size_pt=size_pt).validated()

    def with_bold(self, bold: bool) -> TypographyProfile:
        return replace(self, bold=bold).validated()

    def with_italic(self, italic: bool) -> TypographyProfile:
        return replace(self, italic=italic).validated()

    def with_paragraph(self, paragraph: ParagraphStyle) -> TypographyProfile:
        return replace(self, paragraph=paragraph).validated()

    def reset(self) -> TypographyProfile:
        """The project's own defaults for this scope."""
        return TypographyProfile()


def default_profile(scope: str) -> TypographyProfile:
    """The shipped profile for ``scope`` (an unknown scope reads as *windows*)."""
    return TypographyProfile()


def profiles() -> dict[str, TypographyProfile]:
    """The default profile of every scope, keyed by scope name."""
    return {scope: default_profile(scope) for scope in SCOPES}


# --------------------------------------------------------------- persistence


def load_profiles() -> dict[str, TypographyProfile]:
    """Every saved profile, keyed by scope; missing or broken reads as default.

    A corrupt store, an unknown scope, or a bad field falls back to the project's
    own settings rather than raising, so a window always opens.
    """
    result = profiles()
    try:
        from core.paths import data_dir
        from storage.json_store import JsonStore

        saved = JsonStore(data_dir()).load(TYPOGRAPHY_NAMESPACE, TYPOGRAPHY_KEY)
    except Exception:
        return result
    if not isinstance(saved, dict):
        return result
    for scope in SCOPES:
        value = saved.get(scope)
        if value is not None:
            result[scope] = TypographyProfile.from_primitive(value)
    return result


def save_profiles(all_profiles: dict[str, TypographyProfile]) -> None:
    """Persist every scope's profile; a store error is recorded, not fatal."""
    from core.best_effort import best_effort
    from core.paths import data_dir
    from storage.json_store import JsonStore

    payload = {
        scope: all_profiles.get(scope, default_profile(scope)).validated().to_primitive()
        for scope in SCOPES
    }
    with best_effort("save the typography settings"):
        JsonStore(data_dir()).save(TYPOGRAPHY_NAMESPACE, TYPOGRAPHY_KEY, payload)


def load_profile(scope: str) -> TypographyProfile:
    """One scope's saved profile (a shortcut over :func:`load_profiles`)."""
    return load_profiles().get(scope, default_profile(scope))


def save_profile(scope: str, profile: TypographyProfile) -> None:
    """Save one scope's profile, keeping the other scope's own choice."""
    current = load_profiles()
    current[scope] = profile
    save_profiles(current)


# ----------------------------------------------------------- family choices


def available_families(root: object | None = None) -> list[str]:
    """The font families a user may choose: the project stacks first, then Tk's.

    The project's own stacks come first (so the shipped look is one click away),
    then every family Tk actually reports, de-duplicated and sorted. With no
    display only the stacks are offered, so the panel still builds.
    """
    ordered: list[str] = []
    for name in (*UI_FONT_STACK, *MONO_FONT_STACK):
        if name not in ordered:
            ordered.append(name)
    try:
        from tkinter import font as tkfont

        for name in sorted(set(tkfont.families(root))):  # type: ignore[arg-type]
            if name not in ordered:
                ordered.append(name)
    except Exception:
        pass
    return ordered


# ------------------------------------------------------------------ helpers


def _clamp_int(value: object, low: int, high: int, default: int) -> int:
    number = _as_int(value, default)
    return max(low, min(high, number))


def _clamp_float(value: object, low: float, high: float, default: float) -> float:
    number = _as_float(value, default)
    return max(low, min(high, number))


def _as_int(value: object, default: int) -> int:
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, (int, float, str)):
        try:
            return int(value)
        except (TypeError, ValueError):
            return default
    return default


def _as_float(value: object, default: float) -> float:
    if isinstance(value, (int, float, str)):
        try:
            return float(value)
        except (TypeError, ValueError):
            return default
    return default


__all__ = [
    "ALIGNMENTS",
    "ALIGNMENT_GLYPHS",
    "ALIGNMENT_LABELS",
    "FONT_SIZES",
    "LINE_SPACINGS",
    "LINE_SPACING_LABELS",
    "LISTS_NOTE",
    "MAX_SIZE_PT",
    "MIN_SIZE_PT",
    "SCOPES",
    "SCOPE_CHAT",
    "SCOPE_CONTROLS",
    "SCOPE_LABELS",
    "SCOPE_LISTS",
    "SCOPE_SUMMARIES",
    "SCOPE_WINDOWS",
    "SPACE_STEPS",
    "TYPOGRAPHY_KEY",
    "TYPOGRAPHY_NAMESPACE",
    "ParagraphStyle",
    "TypographyProfile",
    "available_families",
    "default_profile",
    "load_profile",
    "load_profiles",
    "profiles",
    "save_profile",
    "save_profiles",
    "scope_shows",
]
