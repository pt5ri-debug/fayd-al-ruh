"""Prepare Arabic text so a speech engine *says* it instead of spelling it out.

This is the "read the numbers and units, not the symbols" layer for Arabic
text-to-speech. It is a deterministic, dependency-free, **presentation-only**
pass: it never adds a claim, never drops a word the reader needs, and never
touches English text.

Why it exists
-------------

The project has a real neural Arabic voice (Piper's ``ar_JO-kareem``), and it is
preferred whenever it is installed. But the *fallback* is ``espeak-ng``, whose
Arabic voice is rule-based formant synthesis: it reads Arabic **words** well and
Arabic **symbols** badly. Measured on this project's own output:

===============  ==========================================
input            what ``espeak-ng -v ar`` actually says
===============  ==========================================
``25%``          ``خمسة وعشرون`` -- the unit silently vanishes
``1500 psi``     ``ألف وخمسمئة`` then a switch to an *English* voice
``2000 ft``      ``ألفان`` then an English voice
``3.5 mD``       ``ثلاثة فاصلة خمسة`` then an English voice
``10-20``        ``عشرة عشرون`` -- the range collapses to gibberish
``المسامـــية``   ``المسامية`` but with the letters spelled out (tatweel)
===============  ==========================================

Spelled out, the same text is read correctly (``خمسة وعشرين بالمئة``). So the
gap is not the engine's Arabic -- it is that nobody converted the numbers, the
units, and the ranges into the words the engine can say. That is this module.

What it does
------------

* **Numbers -> Arabic words.** ``1500`` becomes ``ألف وخمسمئة``; ``3.5`` becomes
  ``ثلاثة فاصلة خمسة``. Uses the Arabic counting rules: the dual for ``2``, the
  plural for ``3..10`` (``ثلاثة آلاف``), and the singular for ``11`` and above
  (``أحد عشر ألف``).
* **Units -> Arabic.** ``psi`` becomes ``باوند لكل بوصة مربعة``, ``ft`` becomes
  ``قدم``, ``%`` becomes ``بالمئة``. A unit is *translated*, never guessed, so an
  unknown unit is left exactly as it was.
* **Ranges -> words.** ``10-20`` and ``10 -- 20`` become ``عشرة إلى عشرين``.
* **Decimals -> words.** ``3.5`` becomes ``ثلاثة فاصلة خمسة``.
* **Punctuation -> Arabic.** ``, ; ?`` inside Arabic text become ``، ؛ ؟``, so
  the engine's clause detection matches the script it is reading.
* **Tatweel removed.** ``ـــ`` (the kashida) is decoration; it makes the engine
  spell a word out letter by letter, so it is dropped.

What it deliberately does **not** do
------------------------------------

* It does not touch English text. ``detect_language`` gates the whole pass, and
  an English sentence is returned unchanged.
* It does not add diacritics (tashkeel). Guessing the right vowel marks is a
  linguistic decision this module cannot make correctly, and a *wrong* diacritic
  sounds worse than none -- ``espeak-ng`` already infers them from the word.
* It does not change a Latin identifier (``mD``, ``API``, ``SPE``) that has no
  Arabic name, and it does not touch a version string or a code span.
* It never invents a unit: an unknown symbol is left as the engine's own
  business rather than read as something it is not.

The pass is **idempotent** -- running it twice changes nothing the second time,
because the second run sees only Arabic words and Arabic punctuation.
"""

from __future__ import annotations

import re

from core.voices import detect_language

#: Arabic-Indic digits (``٠١٢٣٤٥٦٧٨٩``) mapped onto ASCII, so one number reader
#: handles both scripts. The reverse is never done: the output is words.
_ARABIC_INDIC = str.maketrans("٠١٢٣٤٥٦٧٨٩", "0123456789")

#: The kashida/tatweel: a decorative letter-stretcher with no phonetic value.
_TATWEEL = "\u0640"

#: One Arabic letter, used to decide whether a symbol sits inside Arabic text.
_ARABIC_LETTER_RE = re.compile(r"[\u0621-\u064a]")

#: A decimal number, e.g. ``3.5`` or ``1,500.25``.
_DECIMAL_RE = re.compile(r"\d[\d,]*(?:\.\d+)?")

#: A numeric range, e.g. ``10-20``, ``10 -- 20``, ``10–20``. The en/em dashes
#: and the Arabic tatweel-like dash are all accepted, because a document may use
#: any of them; a hyphen between two digits is always a range, never a minus.
_RANGE_RE = re.compile(r"(\d[\d,]*)\s*(?:-|--|–|—|−|\u2013|\u2014)\s*(\d[\d,]*)")

#: A percentage written with the sign, e.g. ``25%`` or ``25 %``.
_PERCENT_RE = re.compile(r"(\d[\d,]*(?:\.\d+)?)\s*%")

#: Unit spellings the engine cannot say, mapped to the Arabic it should say.
#: Only *known* units are listed: an unrecognised token is left alone rather than
#: read as a guess. Keys are matched case-insensitively and must be a whole
#: token (``psi`` matches, ``psig`` does not unless listed).
_UNITS: dict[str, str] = {
    # Pressure
    "psi": "باوند لكل بوصة مربعة",
    "psia": "باوند لكل بوصة مربعة مطلقة",
    "psig": "باوند لكل بوصة مربعة قياسية",
    "bar": "بار",
    "kpa": "كيلوباسكال",
    "mpa": "ميغاباسكال",
    "pa": "باسكال",
    "atm": "ضغط جوي",
    # Length / depth
    "ft": "قدم",
    "feet": "قدم",
    "foot": "قدم",
    "m": "متر",
    "km": "كيلومتر",
    "mm": "مليمتر",
    "cm": "سنتيمتر",
    "in": "بوصة",
    "inch": "بوصة",
    "inches": "بوصة",
    "yd": "ياردة",
    "mi": "ميل",
    # Volume / rate
    "bbl": "برميل",
    "bpd": "برميل في اليوم",
    "bopd": "برميل نفط في اليوم",
    "mcf": "ألف قدم مكعبة",
    "mmcf": "مليون قدم مكعبة",
    "l": "لتر",
    "gal": "غالون",
    "m3": "متر مكعب",
    "scf": "قدم مكعبة قياسية",
    # Mass / density
    "kg": "كيلوغرام",
    "g": "غرام",
    "lb": "باوند",
    "lbs": "باوند",
    "ton": "طن",
    "tonne": "طن",
    "ppg": "باوند لكل غالون",
    # Permeability / porosity
    "md": "ملي دارسي",
    "d": "دارسي",
    # Temperature
    "c": "درجة مئوية",
    "°c": "درجة مئوية",
    "f": "درجة فهرنهايت",
    "°f": "درجة فهرنهايت",
    "k": "كلفن",
    # Time
    "s": "ثانية",
    "sec": "ثانية",
    "min": "دقيقة",
    "h": "ساعة",
    "hr": "ساعة",
    "day": "يوم",
    "yr": "سنة",
    # Electrical / power
    "v": "فولت",
    "kv": "كيلوفولت",
    "a": "أمبير",
    "kw": "كيلوواط",
    "mw": "ميغاواط",
    "hp": "حصان",
    "hz": "هرتز",
    # Dimensionless / other
    "ppm": "جزء في المليون",
    "wt": "وزن",
    "vol": "حجم",
    "api": "درجة أي بي آي",
}

#: The largest unit a token may be, so a long English word is never mistaken for
#: a unit and translated. Long enough for the plural spellings (``inches``).
_MAX_UNIT_LEN = 8

#: A run of one or more Latin/unit characters, the candidate a unit lookup uses.
_TOKEN_RE = re.compile(r"[A-Za-z°][A-Za-z0-9°]*")

#: ASCII punctuation an Arabic sentence should carry in its own form, and the
#: Arabic mark for each. Mirrors :mod:`companion.polish`, so the written polish
#: and the spoken one agree.
_PUNCTUATION: dict[str, str] = {",": "،", ";": "؛", "?": "؟"}

#: The Arabic connective that turns two numbers into a spoken range.
_RANGE_WORD = "إلى"

#: The Arabic word for the decimal point when a number is spoken.
_DECIMAL_WORD = "فاصلة"

#: The Arabic word for a percentage when the sign is spoken.
_PERCENT_WORD = "بالمئة"

#: Units/words that must never be translated, because they are Latin identifiers
#: the reader should hear as letters (or leave alone) rather than as a unit.
_KEEP_VERBATIM = frozenset({"spe", "api", "iso", "astm", "ogs", "eia", "usgs"})

# --------------------------------------------------------------- number words

#: Ones, by value. ``0`` is empty: Arabic says ``عشرون``, not ``صفر عشرون``.
_ONES = (
    "", "واحد", "اثنان", "ثلاثة", "أربعة", "خمسة", "ستة", "سبعة", "ثمانية", "تسعة",
    "عشرة", "أحد عشر", "اثنا عشر", "ثلاثة عشر", "أربعة عشر", "خمسة عشر",
    "ستة عشر", "سبعة عشر", "ثمانية عشر", "تسعة عشر",
)

#: The tens, by value/10 (index 2 == 20), in the nominative (``عشرون``).
_TENS = (
    "", "", "عشرون", "ثلاثون", "أربعون", "خمسون", "ستون", "سبعون", "ثمانون", "تسعون",
)

#: The same tens in the genitive/accusative (``عشرين``), which Arabic requires
#: after a preposition. A range is read "``X`` إلى ``Y``", and ``إلى`` governs the
#: second number, so the range end must take this form or it sounds wrong.
_TENS_OBLIQUE = (
    "", "", "عشرين", "ثلاثين", "أربعين", "خمسين", "ستين", "سبعين", "ثمانين", "تسعين",
)

#: The hundreds, by value/100 (index 1 == 100). Index 2 is the dual, which has a
#: separate genitive/accusative form (``مئتين``) used after a preposition.
_HUNDREDS = (
    "", "مئة", "مئتان", "ثلاثمئة", "أربعمئة", "خمسمئة", "ستمئة", "سبعمئة",
    "ثمانمئة", "تسعمئة",
)
_HUNDREDS_OBLIQUE = (
    "", "مئة", "مئتين", "ثلاثمئة", "أربعمئة", "خمسمئة", "ستمئة", "سبعمئة",
    "ثمانمئة", "تسعمئة",
)

#: The scale names for the group of three at each position, and the singular,
#: dual, and plural form of each. Arabic counts groups with all three:
#: ``ألف`` (1), ``ألفان`` (2), ``ثلاثة آلاف`` (3-10), ``أحد عشر ألف`` (11+).
_SCALES: tuple[tuple[str, str, str], ...] = (
    ("", "", ""),
    ("ألف", "ألفان", "آلاف"),
    ("مليون", "مليونان", "ملايين"),
    ("مليار", "ملياران", "مليارات"),
    ("تريليون", "تريليونان", "تريليونات"),
)

#: The dual scale names in the genitive/accusative (``ألفين``, ``مليونين``), used
#: when a preposition governs the number -- as in a spoken range.
_SCALES_DUAL_OBLIQUE: tuple[str, ...] = ("", "ألفين", "مليونين", "مليارين", "تريليونين")


def _under_1000(value: int, *, oblique: bool = False) -> str:
    """The Arabic words for ``0 <= value < 1000`` (no scale name).

    ``oblique`` asks for the genitive/accusative form (``عشرين`` rather than
    ``عشرون``), which a governing preposition requires.
    """
    if value == 0:
        return ""
    parts: list[str] = []
    hundreds, rest = divmod(value, 100)
    if hundreds:
        parts.append((_HUNDREDS_OBLIQUE if oblique else _HUNDREDS)[hundreds])
    if rest:
        if rest < 20:
            parts.append(_ONES[rest])
        else:
            tens, ones = divmod(rest, 10)
            tens_word = (_TENS_OBLIQUE if oblique else _TENS)[tens]
            if ones:
                # Arabic says the unit before the ten: ``خمسة وعشرون``.
                parts.append(f"{_ONES[ones]} و{tens_word}")
            else:
                parts.append(tens_word)
    return " و".join(part for part in parts if part)


def _group_words(value: int, scale: int, *, oblique: bool = False) -> str:
    """The Arabic words for one three-digit ``group`` at ``scale`` position."""
    if value == 0:
        return ""
    singular, dual, plural = _SCALES[scale]
    if scale == 0:
        return _under_1000(value, oblique=oblique)
    if value == 1:
        return singular
    if value == 2:
        return _SCALES_DUAL_OBLIQUE[scale] if oblique else dual
    if 3 <= value <= 10:
        # ``ثلاثة آلاف``: the plural scale name after a small count.
        return f"{_under_1000(value)} {plural}"
    # 11 and above take the singular: ``أحد عشر ألف``.
    return f"{_under_1000(value)} {singular}"


def number_to_arabic_words(value: int, *, oblique: bool = False) -> str:
    """Spell a non-negative integer in Arabic words.

    ``1500`` -> ``ألف وخمسمئة``, ``2000`` -> ``ألفان``, ``25000`` ->
    ``خمسة وعشرون ألف``, ``0`` -> ``صفر``. ``oblique`` gives the
    genitive/accusative form a preposition requires (``2000`` -> ``ألفين``).
    """
    if value < 0:
        return f"سالب {number_to_arabic_words(-value, oblique=oblique)}"
    if value == 0:
        return "صفر"
    groups: list[str] = []
    scale = 0
    while value > 0 and scale < len(_SCALES):
        value, group = divmod(value, 1000)
        if group:
            groups.append(_group_words(group, scale, oblique=oblique))
        scale += 1
    if value:
        # Beyond the named scales: speak the remaining digits so nothing is lost.
        groups.append(str(value))
    return " و".join(reversed(groups))


def _spell_number(token: str, *, oblique: bool = False) -> str:
    """Spell one numeric token (an integer, a decimal, or a thousands group)."""
    cleaned = token.replace(",", "").translate(_ARABIC_INDIC)
    if "." in cleaned:
        whole, _, fraction = cleaned.partition(".")
        whole_words = number_to_arabic_words(int(whole or "0"), oblique=oblique)
        # Read the fraction digit by digit: ``3.5`` is ``ثلاثة فاصلة خمسة``,
        # not ``ثلاثة فاصلة خمسة أعشار``, which is what a reader expects.
        digits = " و".join(number_to_arabic_words(int(d)) for d in fraction)
        return f"{whole_words} {_DECIMAL_WORD} {digits}"
    return number_to_arabic_words(int(cleaned or "0"), oblique=oblique)


# ------------------------------------------------------------------ the pass


def _has_arabic(text: str) -> bool:
    return bool(_ARABIC_LETTER_RE.search(text))


def _translate_units(text: str) -> str:
    """Replace each known unit token with its Arabic name, leaving the rest."""

    def replace(match: re.Match[str]) -> str:
        token = match.group(0)
        lowered = token.lower()
        if lowered in _KEEP_VERBATIM or len(lowered) > _MAX_UNIT_LEN:
            return token
        return _UNITS.get(lowered, token)

    return _TOKEN_RE.sub(replace, text)


def _convert_numbers(text: str) -> str:
    """Turn ranges, percentages, and bare numbers into Arabic words."""
    # Ranges first: a range's two halves are numbers, so it must win over the
    # single-number rule or ``10-20`` would be read as ``عشرة`` then ``عشرين``
    # with the dash lost. ``إلى`` governs the end, so it takes the oblique form.
    text = _RANGE_RE.sub(
        lambda m: (
            f"{_spell_number(m.group(1))} {_RANGE_WORD} "
            f"{_spell_number(m.group(2), oblique=True)}"
        ),
        text,
    )
    text = _PERCENT_RE.sub(lambda m: f"{_spell_number(m.group(1))} {_PERCENT_WORD}", text)
    return _DECIMAL_RE.sub(lambda m: _spell_number(m.group(0)), text)


def _arabic_punctuation(text: str) -> str:
    """Use the Arabic ``, ; ?`` inside Arabic text, so clause detection matches."""
    if not _has_arabic(text):
        return text
    return "".join(_PUNCTUATION.get(char, char) for char in text)


def prepare_for_speech(text: str, *, language: str | None = None) -> str:
    """Prepare ``text`` for Arabic speech synthesis; return it unchanged otherwise.

    ``language`` pins the language; when it is ``None`` the text decides (Arabic
    script -> Arabic), so one voice can read both and English is never touched.

    The pass is idempotent and presentation-only: it spells numbers and units as
    the words a listener hears, normalises Arabic punctuation, and drops the
    decorative tatweel. It never adds a claim, never drops information, and never
    rewrites a Latin identifier that has no Arabic name.
    """
    if not text.strip():
        return text
    code = language if language in ("ar", "en") else detect_language(text)
    if code != "ar":
        return text
    prepared = text.replace(_TATWEEL, "")
    prepared = _convert_numbers(prepared)
    prepared = _translate_units(prepared)
    return _arabic_punctuation(prepared)


__all__ = [
    "number_to_arabic_words",
    "prepare_for_speech",
]
