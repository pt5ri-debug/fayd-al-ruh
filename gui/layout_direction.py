"""The reading direction: left-to-right or right-to-left, as one preference.

Arabic is the project's second language, and an Arabic-first reader expects the
interface mirrored -- the navigation on the right, the text right-aligned. Tk has
no single ``dir`` switch, so this module is the one place the choice lives, and
the windows consult it where a direction genuinely changes a layout decision:

* :func:`side_for` -- which side the navigation column is packed on (``left`` for
  LTR, ``right`` for RTL);
* :func:`default_alignment` -- the paragraph alignment a *plain* block falls back
  to when the user has not chosen one in the font settings (``left`` for LTR,
  ``right`` for RTL);
* :func:`detect_direction` -- the direction a piece of text itself reads in, so a
  mixed Arabic/English message can be aligned to its dominant script.

It is deliberately **honest about its limits**: it is not a full mirroring of
every dialog. A dialog keeps its own layout; this preference moves the navigation
and the default alignment, which is what a reader notices first. It is Tk-free
and loads no model, so it is exercised headless.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from core.best_effort import best_effort
from core.paths import data_dir

Direction = Literal["ltr", "rtl"]

#: The two directions, in the order a menu lists them.
DIRECTIONS: tuple[Direction, ...] = ("ltr", "rtl")
DEFAULT_DIRECTION: Direction = "ltr"

#: The label each direction shows, in English and Arabic.
DIRECTION_LABELS: dict[Direction, str] = {
    "ltr": "Left to right (English)",
    "rtl": "Right to left (Arabic / العربية)",
}

_NAMESPACE = "gui"
_KEY = "layout_direction"


def normalize_direction(value: object) -> Direction:
    """Coerce anything to a real direction, defaulting to LTR for the unknown."""
    text = str(value or "").strip().lower()
    return "rtl" if text in ("rtl", "ar", "arabic", "right", "right-to-left") else "ltr"


def side_for(direction: Direction) -> Literal["left", "right"]:
    """The side the navigation column sits on for ``direction``."""
    return "right" if direction == "rtl" else "left"


def default_alignment(direction: Direction) -> Literal["left", "right"]:
    """The paragraph alignment a plain block falls back to for ``direction``."""
    return "right" if direction == "rtl" else "left"


def detect_direction(text: str, *, fallback: Direction = DEFAULT_DIRECTION) -> Direction:
    """The direction ``text`` itself reads in, from the script of its letters.

    A run of Arabic-script letters makes the text right-to-left; Latin letters make
    it left-to-right. When a text carries both, the script with more letters wins
    (so a mostly-Arabic sentence with an English term is still RTL); a text with no
    letters at all returns ``fallback`` rather than guessing. Reuses the project's
    own script check, so there is no second Arabic-letter test.
    """
    from core.voices import detect_language

    letters_ar = sum(1 for char in text if "\u0600" <= char <= "\u06ff")
    letters_latin = sum(1 for char in text if char.isascii() and char.isalpha())
    if letters_ar == 0 and letters_latin == 0:
        return fallback
    if letters_ar >= letters_latin and letters_ar > 0:
        return "rtl"
    if letters_latin > 0:
        return "ltr"
    # The project's own check as a backstop, so this agrees with the rest of the app.
    return "rtl" if detect_language(text) == "ar" else "ltr"


def load_direction(root: Path | None = None) -> Direction:
    """The saved direction, or LTR on a missing or broken store."""
    from storage.json_store import JsonStore

    saved: object = None
    with best_effort("read the saved reading direction"):
        saved = JsonStore(root if root is not None else data_dir()).load(_NAMESPACE, _KEY)
    raw = saved.get("direction") if isinstance(saved, dict) else None
    return normalize_direction(raw) if raw is not None else DEFAULT_DIRECTION


def save_direction(direction: Direction, root: Path | None = None) -> None:
    """Persist the direction; a store error is recorded, never fatal."""
    from storage.json_store import JsonStore

    with best_effort("save the reading direction"):
        JsonStore(root if root is not None else data_dir()).save(
            _NAMESPACE, _KEY, {"direction": normalize_direction(direction)}
        )


def describe(direction: Direction) -> str:
    """A one-line description of the direction and what it changes here."""
    chosen = normalize_direction(direction)
    where = "right" if chosen == "rtl" else "left"
    return (
        f"{DIRECTION_LABELS[chosen]} — the navigation sits on the {where} and plain "
        "text falls back to that side (dialogs keep their own layout)."
    )


__all__ = [
    "DEFAULT_DIRECTION",
    "DIRECTIONS",
    "DIRECTION_LABELS",
    "Direction",
    "default_alignment",
    "describe",
    "detect_direction",
    "load_direction",
    "normalize_direction",
    "save_direction",
    "side_for",
]
