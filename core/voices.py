"""A diverse text-to-speech voice catalog: men, women, boys, and girls.

The chat can read its own text aloud (text-to-speech). Which *voice* reads it is a
user choice, and this module is the one place that choice is declared: a
:class:`Voice` names a timbre and the engine settings that produce it, and
:data:`VOICE_CATALOG` lists several voices in each of four categories -- ``man``,
``woman``, ``boy``, ``girl`` -- so the reader is varied rather than one monotone.

Honesty is a hard rule, as everywhere in this project:

* A :class:`Voice` is *not* a recorded human. It is a set of settings for the
  installed speech engine (``espeak-ng``/``espeak``): a gender variant, a pitch,
  and a speed. The class says so in its ``description``.
* ``espeak-ng`` has no true child voice, so the ``boy``/``girl`` categories are
  the male/female variants at a raised pitch, and each description states that
  plainly instead of pretending a real child is speaking.
* The catalog is dependency-free data; resolving a voice to audio happens in
  :mod:`core.speech`, which needs the engine installed and reports honestly when
  it is not.

The language a voice speaks is chosen from the *text* (Arabic script -> ``ar``,
otherwise ``en``), so the same voice reads Arabic and English.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

#: The four voice categories, in the order the menus show them.
CATEGORY_MAN = "man"
CATEGORY_WOMAN = "woman"
CATEGORY_BOY = "boy"
CATEGORY_GIRL = "girl"

#: ``(key, English label, Arabic label)`` per category.
CATEGORY_LABELS: tuple[tuple[str, str, str], ...] = (
    (CATEGORY_MAN, "Men", "رجال"),
    (CATEGORY_WOMAN, "Women", "نساء"),
    (CATEGORY_BOY, "Boys", "أولاد"),
    (CATEGORY_GIRL, "Girls", "بنات"),
)

_CATEGORY_KEYS = tuple(key for key, _en, _ar in CATEGORY_LABELS)

#: The lowest and highest pitch and speed an engine accepts, so a bad catalog
#: entry can never be handed to the engine.
MIN_PITCH = 0
MAX_PITCH = 99
MIN_SPEED = 80
MAX_SPEED = 450

#: Arabic letters (the base block plus the presentation forms), used to pick the
#: language a piece of text should be spoken in.
_ARABIC_RE = re.compile(r"[\u0600-\u06ff\u0750-\u077f\ufb50-\ufdff\ufe70-\ufeff]")


def detect_language(text: str, *, default: str = "en") -> str:
    """Return ``"ar"`` when ``text`` carries Arabic letters, else ``"en"``.

    A cheap, dependency-free guess: it looks for Arabic script only, which is the
    one distinction the project's two languages need. Mixed text (an English term
    inside an Arabic sentence) is read as Arabic, since the sentence around it is.
    """
    if not text.strip():
        return default
    return "ar" if _ARABIC_RE.search(text) else "en"


@dataclass(frozen=True)
class Voice:
    """One selectable voice: a category, a label, and the engine settings.

    ``variant`` is the engine's own timbre code (``espeak-ng`` uses ``m1``..``m7``
    for male and ``f1``..``f5`` for female); ``pitch`` and ``speed`` shape it
    further. Nothing here is a recorded person -- the settings are applied to
    whatever engine is installed, which is why :meth:`description` says so.
    """

    key: str
    category: str
    label_en: str
    label_ar: str
    variant: str
    pitch: int
    speed: int

    def __post_init__(self) -> None:
        if not self.key.strip():
            raise ValueError("A voice needs a key.")
        if self.category not in _CATEGORY_KEYS:
            raise ValueError(
                f"Unknown voice category {self.category!r}; "
                f"choose one of {', '.join(_CATEGORY_KEYS)}."
            )
        if not self.variant.strip():
            raise ValueError(f"Voice {self.key!r} needs an engine variant.")
        if not MIN_PITCH <= self.pitch <= MAX_PITCH:
            raise ValueError(f"Voice {self.key!r} pitch must be in [{MIN_PITCH}, {MAX_PITCH}].")
        if not MIN_SPEED <= self.speed <= MAX_SPEED:
            raise ValueError(f"Voice {self.key!r} speed must be in [{MIN_SPEED}, {MAX_SPEED}].")

    def label(self, language: str = "en") -> str:
        """The voice's name in ``language`` (falling back to English)."""
        return self.label_ar if language == "ar" else self.label_en

    def category_label(self, language: str = "en") -> str:
        """The voice's category name in ``language``."""
        for key, english, arabic in CATEGORY_LABELS:
            if key == self.category:
                return arabic if language == "ar" else english
        return self.category

    def espeak_voice(self, language: str = "ar") -> str:
        """The ``espeak-ng`` voice spec, e.g. ``"ar+m3"`` or ``"en+f2"``."""
        return f"{language}+{self.variant}"

    def espeak_options(self, language: str = "ar") -> tuple[str, ...]:
        """The extra argv an ``espeak-ng``/``espeak`` invocation needs for this voice."""
        return (
            "-v",
            self.espeak_voice(language),
            "-p",
            str(self.pitch),
            "-s",
            str(self.speed),
        )

    def description(self, language: str = "en") -> str:
        """An honest note about what this voice is."""
        kind = self.category_label(language).lower() if language != "ar" else self.category_label("ar")
        if language == "ar":
            return (
                f"صوت {kind} ({self.label_ar}) من إعدادات محرّك النطق: "
                f"النمط {self.variant}، النغمة {self.pitch}، السرعة {self.speed}. "
                "ليس تسجيلاً لشخص حقيقي، ولا صوت طفل حقيقي في فئتي الأولاد والبنات."
            )
        return (
            f"A {kind} voice ({self.label_en}) built from the speech engine's settings: "
            f"variant {self.variant}, pitch {self.pitch}, speed {self.speed}. It is not a "
            "recording of a real person, and the boys'/girls' categories are a raised "
            "pitch on a male/female voice, not a real child."
        )

    def to_primitive(self) -> dict[str, object]:
        return {
            "key": self.key,
            "category": self.category,
            "label_en": self.label_en,
            "label_ar": self.label_ar,
            "variant": self.variant,
            "pitch": self.pitch,
            "speed": self.speed,
        }


#: Every selectable voice, grouped by category (four per category). The variants
#: and pitches are chosen so the four voices in a category sound clearly distinct.
VOICE_CATALOG: tuple[Voice, ...] = (
    # --- men ---------------------------------------------------------------
    Voice("man-1", CATEGORY_MAN, "Adam", "آدم", "m1", 45, 165),
    Voice("man-2", CATEGORY_MAN, "Basil", "باسل", "m2", 32, 150),
    Voice("man-3", CATEGORY_MAN, "Faris", "فارس", "m4", 58, 178),
    Voice("man-4", CATEGORY_MAN, "Tariq", "طارق", "m6", 26, 142),
    # --- women -------------------------------------------------------------
    Voice("woman-1", CATEGORY_WOMAN, "Layla", "ليلى", "f1", 60, 165),
    Voice("woman-2", CATEGORY_WOMAN, "Mariam", "مريم", "f2", 72, 172),
    Voice("woman-3", CATEGORY_WOMAN, "Salma", "سلمى", "f3", 54, 155),
    Voice("woman-4", CATEGORY_WOMAN, "Noor", "نور", "f4", 66, 182),
    # --- boys (a male voice at a raised pitch: espeak-ng has no child voice) --
    Voice("boy-1", CATEGORY_BOY, "Omar", "عمر", "m3", 76, 180),
    Voice("boy-2", CATEGORY_BOY, "Zaid", "زيد", "m5", 82, 186),
    Voice("boy-3", CATEGORY_BOY, "Yusuf", "يوسف", "m2", 72, 174),
    Voice("boy-4", CATEGORY_BOY, "Hamza", "حمزة", "m1", 80, 192),
    # --- girls (a female voice at a raised pitch) --------------------------
    Voice("girl-1", CATEGORY_GIRL, "Hala", "هالة", "f1", 86, 180),
    Voice("girl-2", CATEGORY_GIRL, "Lina", "لينا", "f2", 92, 186),
    Voice("girl-3", CATEGORY_GIRL, "Reem", "ريم", "f3", 84, 174),
    Voice("girl-4", CATEGORY_GIRL, "Dana", "دانة", "f4", 90, 192),
)

_BY_KEY = {voice.key: voice for voice in VOICE_CATALOG}

#: The voice used when the user has not chosen one.
DEFAULT_VOICE_KEY = "woman-1"


def voice_categories() -> tuple[tuple[str, str, str], ...]:
    """The categories as ``(key, English label, Arabic label)``, in menu order."""
    return CATEGORY_LABELS


def voices_for(category: str) -> tuple[Voice, ...]:
    """The voices in one category, in catalog order."""
    if category not in _CATEGORY_KEYS:
        raise ValueError(
            f"Unknown voice category {category!r}; choose one of {', '.join(_CATEGORY_KEYS)}."
        )
    return tuple(voice for voice in VOICE_CATALOG if voice.category == category)


def voice_keys() -> tuple[str, ...]:
    """Every voice key, in catalog order."""
    return tuple(voice.key for voice in VOICE_CATALOG)


def find_voice(key: str) -> Voice:
    """Return the voice named ``key``; raise ``ValueError`` for an unknown one."""
    try:
        return _BY_KEY[key.strip()]
    except KeyError:
        raise ValueError(
            f"Unknown voice {key!r}; choose one of {', '.join(_BY_KEY)}."
        ) from None


def default_voice() -> Voice:
    """The voice used when none was chosen."""
    return _BY_KEY[DEFAULT_VOICE_KEY]


def voice_summaries() -> tuple[tuple[str, str, str, str], ...]:
    """``(key, category, label_en, label_ar)`` per voice, for a menu or a table."""
    return tuple(
        (voice.key, voice.category, voice.label_en, voice.label_ar) for voice in VOICE_CATALOG
    )


__all__ = [
    "CATEGORY_BOY",
    "CATEGORY_GIRL",
    "CATEGORY_LABELS",
    "CATEGORY_MAN",
    "CATEGORY_WOMAN",
    "DEFAULT_VOICE_KEY",
    "MAX_PITCH",
    "MAX_SPEED",
    "MIN_PITCH",
    "MIN_SPEED",
    "VOICE_CATALOG",
    "Voice",
    "default_voice",
    "detect_language",
    "find_voice",
    "voice_categories",
    "voice_keys",
    "voice_summaries",
    "voices_for",
]
