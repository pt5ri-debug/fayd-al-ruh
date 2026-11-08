"""The project's appearance themes, in one place every window reads from.

A :class:`Theme` is one complete palette: every colour a window paints with,
named after OpenHands' semantic tokens (``surface``, ``foreground``,
``border_subtle`` …) so the mapping back to the web UI stays obvious. The window
and all fifteen dialogs draw from the same object, so one theme switch restyles
every open window at once.

The module is deliberately **tkinter-free**: a theme is data, so the catalogue
can be inspected, validated, and tested with no display, and a window imports it
without pulling Tk in. :mod:`gui.app` re-exports :class:`Theme` and the shipped
themes for backward compatibility, so existing callers and tests keep working.

Eight themes ship: the two OpenHands appearances (*dark*, *light*) plus six
distinct looks -- *midnight* (near-black, cool), *nord* (the Nord palette),
*solarized* (Solarized Dark), *sepia* (warm paper), *ocean* (deep blue), and
*high-contrast* (maximum legibility, WCAG-friendly). Each is a full palette, so
switching never leaves a widget painted with a stale token.
"""

from __future__ import annotations

from dataclasses import dataclass, fields

#: The namespace and key the chosen theme is saved under in the user's data dir.
THEME_NAMESPACE = "gui"
THEME_KEY = "appearance"


@dataclass(frozen=True)
class Theme:
    """One appearance: every colour the window draws with.

    Field names follow OpenHands' semantic tokens (``surface``, ``tertiary``,
    ``border_subtle`` …) so the mapping back to the web UI stays obvious.
    """

    name: str
    # Surfaces, from deepest to most raised.
    surface_deep: str      # context menus, deep insets
    base: str              # app shell background
    surface: str           # cards, composer, sidebar
    surface_raised: str    # hover/selected tint inside a panel
    # Text, from most to least prominent.
    contrast: str          # strongest contrast (assistant text, titles)
    foreground: str        # primary readable text
    text_secondary: str    # secondary text
    text_dim: str          # placeholders, very muted labels
    # Lines and state.
    border: str            # standard 1px stroke
    border_subtle: str     # within-panel separators, list dividers
    primary: str           # gold accent — send button, brand, focus
    primary_foreground: str  # text/icon on the primary fill
    interactive_hover: str   # hover on the base surface
    interactive_active: str  # selected/active on the base surface
    user_bubble: str       # user message fill
    user_bubble_fg: str    # user message text
    code_bg: str           # fenced/inline code background
    code_fg: str           # fenced code text
    quote_bar: str         # blockquote accent
    titlebar: str          # every window's title bar (a dark grey band)
    titlebar_edge: str     # the hairline under a window's title bar
    success: str
    danger: str
    disabled: str


DARK_THEME = Theme(
    name="dark",
    surface_deep="#05070A",     # cool-grey-975
    base="#0B0E14",             # cool-grey-950
    surface="#21252F",          # cool-grey-925
    surface_raised="#2C313F",   # cool-grey-900
    contrast="#F7F9FC",         # cool-grey-50
    foreground="#C3CDDC",       # cool-grey-300
    text_secondary="#A3B0C4",   # cool-grey-400
    text_dim="#7E8A9E",         # cool-grey-500
    border="#4B5468",           # cool-grey-700
    border_subtle="#383F50",    # cool-grey-800
    primary="#C9B974",          # gold
    primary_foreground="#0B0E14",
    interactive_hover="#2C313F",  # cool-grey-900
    interactive_active="#383F50",  # cool-grey-800
    user_bubble="#2C313F",        # cool-grey-900, like OpenHands' bg-tertiary
    user_bubble_fg="#C3CDDC",
    code_bg="#151A24",            # a touch deeper than the base
    code_fg="#DCE3EE",
    quote_bar="#4B5468",
    titlebar="#181C25",         # a dark grey band, a step above the base
    titlebar_edge="#2C313F",    # cool-grey-900, the hairline under it
    success="#A5E75E",
    danger="#E76A5E",
    disabled="#383F50",
)

LIGHT_THEME = Theme(
    name="light",
    surface_deep="#EEF2F7",     # cool-grey-100
    base="#F7F9FC",             # cool-grey-50
    surface="#FFFFFF",
    surface_raised="#EEF2F7",   # cool-grey-100
    contrast="#0B0E14",         # cool-grey-950
    foreground="#2C313F",       # cool-grey-900
    text_secondary="#4B5468",   # cool-grey-700
    text_dim="#7E8A9E",         # cool-grey-500
    border="#DCE3EE",           # cool-grey-200
    border_subtle="#EEF2F7",    # cool-grey-100
    primary="#8A6E1F",          # darkened gold, AA on the light base
    primary_foreground="#FFFFFF",
    interactive_hover="#EEF2F7",  # cool-grey-100
    interactive_active="#DCE3EE",  # cool-grey-200
    user_bubble="#EEF2F7",        # cool-grey-100
    user_bubble_fg="#0B0E14",
    code_bg="#F0F3F8",
    code_fg="#1B2029",
    quote_bar="#C4CEDD",
    titlebar="#DCE3EE",         # a light grey band in the light appearance
    titlebar_edge="#C4CEDD",    # the hairline under it
    success="#3F7D1E",
    danger="#C0473C",
    disabled="#DCE3EE",
)

MIDNIGHT_THEME = Theme(
    name="midnight",
    surface_deep="#000000",
    base="#050608",
    surface="#0E1116",
    surface_raised="#171B22",
    contrast="#FFFFFF",
    foreground="#D4DAE4",
    text_secondary="#9AA4B4",
    text_dim="#69717F",
    border="#2A2F39",
    border_subtle="#1C2028",
    primary="#7AA2F7",          # a cool blue accent, like Tokyo Night
    primary_foreground="#050608",
    interactive_hover="#171B22",
    interactive_active="#20252E",
    user_bubble="#171B22",
    user_bubble_fg="#D4DAE4",
    code_bg="#0A0D12",
    code_fg="#E2E8F0",
    quote_bar="#2A2F39",
    titlebar="#0A0C11",
    titlebar_edge="#1C2028",
    success="#9ECE6A",
    danger="#F7768E",
    disabled="#20252E",
)

NORD_THEME = Theme(
    name="nord",
    surface_deep="#242933",     # polar night, darkest
    base="#2E3440",             # nord0
    surface="#3B4252",          # nord1
    surface_raised="#434C5E",   # nord2
    contrast="#ECEFF4",         # nord6
    foreground="#D8DEE9",       # nord4
    text_secondary="#C0C8D6",
    text_dim="#8C97A8",
    border="#4C566A",           # nord3
    border_subtle="#434C5E",
    primary="#88C0D0",          # nord8, a frost cyan
    primary_foreground="#2E3440",
    interactive_hover="#434C5E",
    interactive_active="#4C566A",
    user_bubble="#434C5E",
    user_bubble_fg="#ECEFF4",
    code_bg="#2B313C",
    code_fg="#E5E9F0",          # nord5
    quote_bar="#4C566A",
    titlebar="#292E38",
    titlebar_edge="#434C5E",
    success="#A3BE8C",          # nord14
    danger="#BF616A",           # nord11
    disabled="#434C5E",
)

SOLARIZED_THEME = Theme(
    name="solarized",
    surface_deep="#00212B",
    base="#002B36",             # base03
    surface="#073642",          # base02
    surface_raised="#0A4553",
    contrast="#FDF6E3",         # base3
    foreground="#93A1A1",       # base1
    text_secondary="#839496",   # base0
    text_dim="#657B83",         # base00
    border="#0F4C5C",
    border_subtle="#0A4553",
    primary="#B58900",          # yellow
    primary_foreground="#002B36",
    interactive_hover="#0A4553",
    interactive_active="#0F4C5C",
    user_bubble="#0A4553",
    user_bubble_fg="#EEE8D5",   # base2
    code_bg="#00212B",
    code_fg="#EEE8D5",
    quote_bar="#586E75",        # base01
    titlebar="#00252E",
    titlebar_edge="#0A4553",
    success="#859900",          # green
    danger="#DC322F",           # red
    disabled="#0F4C5C",
)

SEPIA_THEME = Theme(
    name="sepia",
    surface_deep="#E7DBC5",
    base="#F4ECD8",             # warm paper
    surface="#FBF5E7",
    surface_raised="#EFE4CC",
    contrast="#2B2118",
    foreground="#4A3F32",
    text_secondary="#6B5D4B",
    text_dim="#93856F",
    border="#DCCDB0",
    border_subtle="#E7DBC5",
    primary="#8F5F26",          # burnt sienna, AA on the paper fill
    primary_foreground="#FBF5E7",
    interactive_hover="#EFE4CC",
    interactive_active="#E4D6BC",
    user_bubble="#EFE4CC",
    user_bubble_fg="#2B2118",
    code_bg="#EDE2CB",
    code_fg="#3A2F24",
    quote_bar="#C9B896",
    titlebar="#E7DBC5",
    titlebar_edge="#D6C7A8",
    success="#5C7A29",
    danger="#A63D2F",
    disabled="#E4D6BC",
)

OCEAN_THEME = Theme(
    name="ocean",
    surface_deep="#031B2E",
    base="#052A44",             # deep sea
    surface="#0A3A5A",
    surface_raised="#0F4A70",
    contrast="#EAF6FF",
    foreground="#C3E0F2",
    text_secondary="#9CC4DD",
    text_dim="#6E93AC",
    border="#155A82",
    border_subtle="#0F4A70",
    primary="#4FC3D9",          # bright aqua
    primary_foreground="#052A44",
    interactive_hover="#0F4A70",
    interactive_active="#155A82",
    user_bubble="#0F4A70",
    user_bubble_fg="#EAF6FF",
    code_bg="#042138",
    code_fg="#D6ECFA",
    quote_bar="#155A82",
    titlebar="#04253C",
    titlebar_edge="#0F4A70",
    success="#5BD6A0",
    danger="#FF7A6B",
    disabled="#155A82",
)

HIGH_CONTRAST_THEME = Theme(
    name="high-contrast",
    surface_deep="#000000",
    base="#000000",
    surface="#101010",
    surface_raised="#1E1E1E",
    contrast="#FFFFFF",
    foreground="#FFFFFF",
    text_secondary="#E6E6E6",
    text_dim="#B8B8B8",
    border="#FFFFFF",
    border_subtle="#4D4D4D",
    primary="#FFD400",          # bright yellow, high contrast on black
    primary_foreground="#000000",
    interactive_hover="#1E1E1E",
    interactive_active="#333333",
    user_bubble="#1E1E1E",
    user_bubble_fg="#FFFFFF",
    code_bg="#000000",
    code_fg="#FFFFFF",
    quote_bar="#FFFFFF",
    titlebar="#000000",
    titlebar_edge="#FFFFFF",
    success="#00FF88",
    danger="#FF5A5A",
    disabled="#4D4D4D",
)

#: Every shipped theme, in the order the menus list them (the two OpenHands
#: appearances first, then the six extra looks).
_THEME_ORDER = (
    DARK_THEME,
    LIGHT_THEME,
    MIDNIGHT_THEME,
    NORD_THEME,
    SOLARIZED_THEME,
    SEPIA_THEME,
    OCEAN_THEME,
    HIGH_CONTRAST_THEME,
)

THEMES: dict[str, Theme] = {theme.name: theme for theme in _THEME_ORDER}

#: A short, honest description of each theme, for the sidebar and the CLI.
THEME_SUMMARIES: dict[str, str] = {
    "dark": "OpenHands dark — the project's default cool-grey appearance.",
    "light": "OpenHands light — the same tokens on a bright base.",
    "midnight": "Near-black and cool, with a blue accent.",
    "nord": "The Nord palette — muted polar blues and frost.",
    "solarized": "Solarized Dark — warm, low-contrast, easy on the eyes.",
    "sepia": "Warm paper — a soft, book-like light look.",
    "ocean": "Deep sea blues with an aqua accent.",
    "high-contrast": "Maximum legibility — white on black, bright accent.",
}


def theme_names() -> tuple[str, ...]:
    """Every shipped theme's key, in menu order."""
    return tuple(THEMES)


def theme_summaries() -> tuple[tuple[str, str], ...]:
    """``(name, summary)`` for every shipped theme, in menu order."""
    return tuple((name, THEME_SUMMARIES.get(name, "")) for name in THEMES)


def find_theme(name: str) -> Theme | None:
    """The theme named ``name``, or ``None`` for an unknown key (never raises)."""
    return THEMES.get(name)


def default_theme() -> Theme:
    """The theme the window opens with when nothing is saved."""
    return DARK_THEME


def validate_theme(theme: Theme) -> tuple[str, ...]:
    """Every problem with a theme, as readable lines (``()`` when it is fine).

    A theme is one complete palette, so a missing token would leave a widget
    painted with a stale colour and a malformed colour would make Tk reject the
    widget outright; this checks every field is a non-empty ``#RRGGBB`` string.
    It is used by the catalogue's own test, not on the hot path.
    """
    problems: list[str] = []
    if not theme.name:
        problems.append("the theme has no name")
    for field in fields(theme):
        if field.name == "name":
            continue
        value = getattr(theme, field.name)
        if not isinstance(value, str) or not value.strip():
            problems.append(f"{theme.name}.{field.name} is empty")
        elif _colour(value) is None:
            problems.append(f"{theme.name}.{field.name} is not a #RRGGBB colour: {value!r}")
    return tuple(problems)


def _colour(value: str) -> tuple[int, int, int] | None:
    """A ``#RRGGBB`` string as an ``(r, g, b)`` triple, else ``None``."""
    text = value.strip()
    if not text.startswith("#") or len(text) != 7:
        return None
    try:
        return (int(text[1:3], 16), int(text[3:5], 16), int(text[5:7], 16))
    except ValueError:
        return None


def relative_luminance(value: str) -> float | None:
    """The WCAG relative luminance of a ``#RRGGBB`` colour, or ``None``.

    Used by the catalogue's test to prove each theme keeps its body text
    readable against its own base -- an honest check that a new palette was not
    shipped unreadable.
    """
    rgb = _colour(value)
    if rgb is None:
        return None

    def channel(component: int) -> float:
        scaled = component / 255.0
        return scaled / 12.92 if scaled <= 0.03928 else ((scaled + 0.055) / 1.055) ** 2.4

    r, g, b = (channel(component) for component in rgb)
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


def contrast_ratio(first: str, second: str) -> float | None:
    """The WCAG contrast ratio between two ``#RRGGBB`` colours, or ``None``."""
    a = relative_luminance(first)
    b = relative_luminance(second)
    if a is None or b is None:
        return None
    lighter, darker = max(a, b), min(a, b)
    return (lighter + 0.05) / (darker + 0.05)


def load_theme_name() -> str:
    """The saved theme name, defaulting to dark; never raises on a bad store."""
    from core.best_effort import best_effort
    from core.paths import data_dir
    from storage.json_store import JsonStore

    saved: object = None
    with best_effort("read the saved appearance"):
        saved = JsonStore(data_dir()).load(THEME_NAMESPACE, THEME_KEY)
    name = saved.get("theme") if isinstance(saved, dict) else None
    return name if name in THEMES else default_theme().name


def save_theme_name(name: str) -> None:
    """Persist the chosen theme; a store error is recorded, not fatal."""
    from core.best_effort import best_effort
    from core.paths import data_dir
    from storage.json_store import JsonStore

    if name not in THEMES:
        return
    with best_effort("save the theme"):
        JsonStore(data_dir()).save(THEME_NAMESPACE, THEME_KEY, {"theme": name})


__all__ = [
    "DARK_THEME",
    "HIGH_CONTRAST_THEME",
    "LIGHT_THEME",
    "MIDNIGHT_THEME",
    "NORD_THEME",
    "OCEAN_THEME",
    "SEPIA_THEME",
    "SOLARIZED_THEME",
    "THEMES",
    "THEME_KEY",
    "THEME_NAMESPACE",
    "THEME_SUMMARIES",
    "Theme",
    "contrast_ratio",
    "default_theme",
    "find_theme",
    "load_theme_name",
    "relative_luminance",
    "save_theme_name",
    "theme_names",
    "theme_summaries",
    "validate_theme",
]
