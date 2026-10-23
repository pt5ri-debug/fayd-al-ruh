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

Each :class:`Voice` carries, besides its timbre, pitch, and speed, the two settings
that keep the reading from sounding mechanical: an **intonation range** (a living
sentence melody, ``espeak-ng``'s ``-P``) and a small **word gap** (a natural pause
between words, ``-g``). A flat pitch and a rushed, gapless delivery are the two
things that make a synthetic voice sound like a machine; giving every voice a wide
melody and a conversational pace is what makes men's and women's voices alike read
like a person. Every catalog entry is tuned that way -- see :data:`VOICE_CATALOG`.

Two kinds of voice live here, and the distinction matters:

* :data:`VOICE_CATALOG` -- *synthetic* voices: named settings for ``espeak-ng``
  (a gender variant, a pitch, a speed). They are always available and portable,
  but they are not the voices a Windows or macOS user actually installed.
* :class:`SystemVoice` -- *real* voices: the voices the OS itself reports by name
  (``Microsoft Hoda``, ``Majed``, ...). They are what the machine really speaks
  with, so the menus prefer them when they exist and the engine is asked for that
  exact voice. They are listed through :func:`core.speech.system_voices`.

A third kind, :class:`ModelVoice`, names a voice *inside a local model* the project
runs itself (VoiceTut-TTS's studio speakers over the audio.cpp engine, for example).
Its key is ``model:<engine>:<name>``, so a saved choice is routed to the engine that
can actually speak it.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from core.best_effort import best_effort

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

#: Prefix that marks a key as a *real installed* voice (see :class:`SystemVoice`)
#: rather than an entry of the synthetic :data:`VOICE_CATALOG`.
SYSTEM_VOICE_PREFIX = "system:"

#: The categories a *real* system voice is grouped under in a menu, keyed by its
#: two-letter language code, so the voices a user can actually hear are listed
#: under the language they speak instead of a synthetic man/woman/child split.
LANGUAGE_LABELS: dict[str, tuple[str, str]] = {
    "ar": ("Arabic", "العربية"),
    "en": ("English", "الإنجليزية"),
}

#: Where a voice comes from. A menu groups its voices by this, so a user reads
#: each set under the name of the source it actually came from -- the voices the
#: OS installed, the local VoiceTut model, Piper, Qwen3-TTS, or the synthetic
#: catalog the system engine speaks. The value is the *engine* that owns the
#: voices (``"system"`` is the OS itself, not an engine name).
SOURCE_SYSTEM = "system"
SOURCE_AUDIOCPP = "audio.cpp"
SOURCE_PIPER = "piper"
SOURCE_QWEN = "qwen3-tts"
SOURCE_SYNTHETIC = "synthetic"

#: ``(source, title)`` per source, in the order a menu shows them. The titles are
#: the ones the user asked for: each set of voices sits under the name of the
#: source it came from, with the language(s) it can really speak.
SOURCE_TITLES: tuple[tuple[str, str], ...] = (
    (SOURCE_SYSTEM, "Installed Voices (Arabic + English)"),
    (SOURCE_AUDIOCPP, "VoiceTut-TTS (Local Arabic)"),
    (SOURCE_PIPER, "Piper (Arabic + English)"),
    (SOURCE_QWEN, "Qwen3-TTS (English)"),
    (SOURCE_SYNTHETIC, "Synthetic Voices (Arabic + English)"),
)

#: The lowest and highest pitch and speed an engine accepts, so a bad catalog
#: entry can never be handed to the engine.
MIN_PITCH = 0
MAX_PITCH = 99
MIN_SPEED = 80
MAX_SPEED = 450

#: Intonation range (``espeak-ng``'s ``-P``): how far the pitch is allowed to move
#: across a sentence. A wider range gives the natural sentence melody that a flat
#: monotone lacks; the default is 50, and the catalog keeps every voice above it so
#: none reads as a machine. Clamped so a bad entry is never handed to the engine.
MIN_PITCH_RANGE = 0
MAX_PITCH_RANGE = 99

#: Word gap (``espeak-ng``'s ``-g``): a short pause between words in units of 10 ms
#: at the default speed. A small value restores the natural word rhythm and breath
#: that a rushed, robotic reading loses; ``0`` is the engine's own (rushed) default.
MIN_WORD_GAP = 0
MAX_WORD_GAP = 20

#: The intonation range and word gap a voice gets when it does not name its own.
#: They are the two settings that make a synthetic reading sound less mechanical:
#: a living sentence melody and a natural pause between words.
DEFAULT_PITCH_RANGE = 62
DEFAULT_WORD_GAP = 2

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
    further, and the two prosody settings below are what keep the reading from
    sounding mechanical:

    * ``pitch_range`` (``espeak-ng``'s ``-P``) is how far the pitch rises and falls
      across a sentence. A flat, fixed pitch is the single biggest giveaway of a
      synthetic voice; a living sentence melody is what a human speaker has.
    * ``word_gap`` (``espeak-ng``'s ``-g``) is a short pause between words, which
      restores the natural rhythm and breath a rushed reading loses.

    Nothing here is a recorded person -- the settings are applied to whatever
    engine is installed, which is why :meth:`description` says so.
    """

    key: str
    category: str
    label_en: str
    label_ar: str
    variant: str
    pitch: int
    speed: int
    pitch_range: int = DEFAULT_PITCH_RANGE
    word_gap: int = DEFAULT_WORD_GAP

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
        if not MIN_PITCH_RANGE <= self.pitch_range <= MAX_PITCH_RANGE:
            raise ValueError(
                f"Voice {self.key!r} pitch range must be in "
                f"[{MIN_PITCH_RANGE}, {MAX_PITCH_RANGE}]."
            )
        if not MIN_WORD_GAP <= self.word_gap <= MAX_WORD_GAP:
            raise ValueError(
                f"Voice {self.key!r} word gap must be in [{MIN_WORD_GAP}, {MAX_WORD_GAP}]."
            )

    def label(self, language: str = "en") -> str:
        """The voice's name in ``language`` (falling back to English)."""
        return self.label_ar if language == "ar" else self.label_en

    @property
    def source(self) -> str:
        """The source this voice came from: the synthetic catalog."""
        return SOURCE_SYNTHETIC

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
        """The extra argv an ``espeak-ng``/``espeak`` invocation needs for this voice.

        Alongside the timbre, pitch, and speed, the intonation range (``-P``) and
        word gap (``-g``) are passed so the reading has a natural sentence melody
        and word rhythm instead of the engine's flat, rushed default.
        """
        return (
            "-v",
            self.espeak_voice(language),
            "-p",
            str(self.pitch),
            "-s",
            str(self.speed),
            "-P",
            str(self.pitch_range),
            "-g",
            str(self.word_gap),
        )

    def description(self, language: str = "en") -> str:
        """An honest note about what this voice is."""
        kind = self.category_label(language).lower() if language != "ar" else self.category_label("ar")
        if language == "ar":
            return (
                f"صوت {kind} ({self.label_ar}) من إعدادات محرّك النطق: "
                f"النمط {self.variant}، النغمة {self.pitch}، السرعة {self.speed}، "
                f"مدى النغمة {self.pitch_range}، فاصل الكلمات {self.word_gap}. "
                "الإيقاع والتلوين مضبوطان ليقرأ قراءة طبيعية لا آلية، "
                "وليس تسجيلاً لشخص حقيقي، ولا صوت طفل حقيقي في فئتي الأولاد والبنات."
            )
        return (
            f"A {kind} voice ({self.label_en}) built from the speech engine's settings: "
            f"variant {self.variant}, pitch {self.pitch}, speed {self.speed}, "
            f"pitch range {self.pitch_range}, word gap {self.word_gap}. The melody and "
            "rhythm are tuned so it reads naturally rather than mechanically. It is not "
            "a recording of a real person, and the boys'/girls' categories are a raised "
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
            "pitch_range": self.pitch_range,
            "word_gap": self.word_gap,
        }


@dataclass(frozen=True)
class SystemVoice:
    """A voice that is **really installed** on this machine, by its own name.

    The synthetic :data:`VOICE_CATALOG` is a set of *settings* -- an espeak-ng
    variant, a pitch, and a speed -- which is honest but cannot name the voice a
    Windows or macOS user actually has (Microsoft Hoda, Majed, ...). This class is
    the other half: it wraps a voice the OS reports by name, so the menus can list
    what the machine will really speak with, and the engine can be asked for that
    exact voice instead of an approximation.

    Honesty rules, as everywhere:

    * ``name`` is the engine's own voice name (e.g. ``"Microsoft Hoda"``), the
      exact string the engine is given.
    * ``language`` is the engine's language tag (``ar-SA``), so Arabic voices are
      grouped and chosen by the language they truly speak.
    * ``key`` is a stable ``system:<engine>:<name>`` string, so a saved choice
      survives a restart and cannot be confused with a catalog key.
    """

    name: str
    language: str
    gender: str = ""
    engine: str = ""

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("A system voice needs a name.")

    @property
    def source(self) -> str:
        """The source this voice came from: the OS itself."""
        return SOURCE_SYSTEM

    @property
    def language_code(self) -> str:
        """The two-letter language code (``ar-SA`` -> ``ar``; ``en-US`` -> ``en``)."""
        return self.language.strip().lower().split("-")[0][:2] or "en"

    @property
    def arabic(self) -> bool:
        """True when this voice speaks Arabic."""
        return self.language_code == "ar"

    @property
    def key(self) -> str:
        """A stable ``system:<engine>:<name>`` key for menus and persistence."""
        engine = self.engine or "system"
        return f"{SYSTEM_VOICE_PREFIX}{engine}:{self.name}"

    def language_label(self, language: str = "en") -> str:
        """The voice's language group name (``Arabic`` / ``العربية``)."""
        english, arabic = LANGUAGE_LABELS.get(self.language_code, (self.language, self.language))
        return arabic if language == "ar" else english

    def label(self, language: str = "en") -> str:
        """The voice's name as the engine reports it (never translated)."""
        return self.name

    @property
    def label_en(self) -> str:
        """The voice's name (English), for interface parity with :class:`Voice`."""
        return self.name

    @property
    def label_ar(self) -> str:
        """The voice's name (unchanged: an installed voice is not translated)."""
        return self.name

    def category_label(self, language: str = "en") -> str:
        """The voice's language group, so a menu can group by language."""
        return self.language_label(language)

    def description(self, language: str = "en") -> str:
        """An honest note about what this voice is."""
        gender = f" ({self.gender})" if self.gender else ""
        if language == "ar":
            return (
                f"صوت حقيقي مثبّت في النظام: {self.name}{gender}، اللغة {self.language}. "
                "يُنطق مباشرة من محرّك النظام، وليس من إعدادات مختصرة."
            )
        return (
            f"A real voice installed on this machine: {self.name}{gender}, "
            f"language {self.language}. It is spoken directly by the system engine, "
            "not approximated from settings."
        )

    def to_primitive(self) -> dict[str, object]:
        return {
            "key": self.key,
            "name": self.name,
            "language": self.language,
            "gender": self.gender,
            "engine": self.engine,
        }


#: A voice the caller can choose: a synthetic catalog :class:`Voice`, a
#: :class:`SystemVoice` naming a voice the OS really reports, or a
#: :class:`ModelVoice` naming a voice of a **local model** the project can run
#: (VoiceTut-TTS over audio.cpp, Piper, Qwen3-TTS). The engine treats a
#: ``SystemVoice`` as an exact name to select and a ``Voice`` as settings to map;
#: a ``ModelVoice`` is handed to the synthesizer that owns its ``engine``.
MODEL_VOICE_PREFIX = "model:"


@dataclass(frozen=True)
class ModelVoice:
    """A voice of a **local speech model**, by the name that model gives it.

    The synthetic :data:`VOICE_CATALOG` is a set of *settings* for a formant
    engine, and :class:`SystemVoice` names a voice the operating system installed.
    This class is the third kind: a named voice *inside a model the project runs
    itself* -- VoiceTut-TTS's built-in studio speakers (``Ahmed``, ``Sarah``, ...)
    over the audio.cpp engine, for example.

    Honesty rules, as everywhere:

    * ``name`` is the model's own speaker name (``"Ahmed"``), the exact string the
      engine is asked for.
    * ``engine`` is the speech engine that can speak it (``"audio.cpp"``), so the
      reader routes the choice to the right synthesizer instead of the default.
    * ``key`` is a stable ``model:<engine>:<name>`` string, so a saved choice
      survives a restart and can never be confused with a catalog or system key.
    * ``reference``/``reference_text`` are what a *cloning* engine needs for this
      voice: a short reference clip and its transcript. ``None`` means the engine
      can speak the name without them.
    """

    name: str
    engine: str
    language: str = "ar"
    gender: str = ""
    model: str = ""
    reference: Path | None = None
    reference_text: str = ""
    #: A human-readable name for a menu, when the engine's own ``name`` is a model
    #: key (Piper's ``ar_JO-kareem-medium`` reads as "Kareem (Arabic, male)"). Empty
    #: means ``name`` is already the label.
    title: str = ""

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("A model voice needs a name.")
        if not self.engine.strip():
            raise ValueError("A model voice needs the engine that speaks it.")

    @property
    def source(self) -> str:
        """The source this voice came from (its owning engine)."""
        return self.engine

    @property
    def language_code(self) -> str:
        """The two-letter language code (``arz`` -> ``ar``; ``en-US`` -> ``en``)."""
        code = self.language.strip().lower().split("-")[0]
        # ``arz`` is Egyptian Arabic: an Arabic language id written with three
        # letters, so it must not be mistaken for a non-Arabic one.
        if code in ("arz", "arb", "ara"):
            return "ar"
        return code[:2] or "en"

    @property
    def arabic(self) -> bool:
        """True when this voice speaks Arabic."""
        return self.language_code == "ar"

    @property
    def key(self) -> str:
        """A stable ``model:<engine>:<name>`` key for menus and persistence."""
        return f"{MODEL_VOICE_PREFIX}{self.engine}:{self.name}"

    def language_label(self, language: str = "en") -> str:
        """The voice's language group name (``Arabic`` / ``العربية``)."""
        english, arabic = LANGUAGE_LABELS.get(self.language_code, (self.language, self.language))
        return arabic if language == "ar" else english

    def label(self, language: str = "en") -> str:
        """The voice's display name: its readable title, else the model's own name."""
        return self.title or self.name

    @property
    def label_en(self) -> str:
        """The voice's display name (English), for parity with :class:`Voice`."""
        return self.label()

    @property
    def label_ar(self) -> str:
        """The voice's display name (unchanged: a model's speaker is not translated)."""
        return self.label()

    def category_label(self, language: str = "en") -> str:
        """The voice's language group, so a menu can group by language."""
        return self.language_label(language)

    def description(self, language: str = "en") -> str:
        """An honest note about what this voice is."""
        gender = f" ({self.gender})" if self.gender else ""
        if language == "ar":
            return (
                f"صوت من موديل محلي: {self.name}{gender} من {self.model or self.engine}. "
                f"يُنطق عبر محرّك {self.engine} على هذا الجهاز، ولا يخرج من الجهاز."
            )
        return (
            f"A voice of a local model: {self.name}{gender} from "
            f"{self.model or self.engine}. It is spoken by the {self.engine} engine "
            "on this machine; nothing leaves it."
        )

    def to_primitive(self) -> dict[str, object]:
        return {
            "key": self.key,
            "name": self.name,
            "engine": self.engine,
            "language": self.language,
            "gender": self.gender,
            "model": self.model,
            "title": self.title,
        }


#: Every selectable voice, grouped by category (four per category). The variants
#: and pitches are chosen so the four voices in a category sound clearly distinct,
#: and the speeds are kept near a human conversational rate rather than a rushed,
#: machine-gun clip. The last two numbers per entry are the prosody that keeps the
#: reading from sounding synthetic: ``pitch_range`` (a wide, living sentence melody
#: -- every entry is above the flat engine default of 50) and ``word_gap`` (a short
#: pause between words, restoring natural rhythm and breath).
VOICE_CATALOG: tuple[Voice, ...] = (
    # --- men ---------------------------------------------------------------
    Voice("man-1", CATEGORY_MAN, "Adam", "آدم", "m1", 42, 160, 60, 3),
    Voice("man-2", CATEGORY_MAN, "Basil", "باسل", "m2", 32, 152, 66, 2),
    Voice("man-3", CATEGORY_MAN, "Faris", "فارس", "m4", 54, 168, 58, 2),
    Voice("man-4", CATEGORY_MAN, "Tariq", "طارق", "m6", 28, 148, 68, 3),
    # --- women -------------------------------------------------------------
    Voice("woman-1", CATEGORY_WOMAN, "Layla", "ليلى", "f1", 60, 162, 70, 2),
    Voice("woman-2", CATEGORY_WOMAN, "Mariam", "مريم", "f2", 70, 168, 68, 2),
    Voice("woman-3", CATEGORY_WOMAN, "Salma", "سلمى", "f3", 56, 156, 74, 3),
    Voice("woman-4", CATEGORY_WOMAN, "Noor", "نور", "f4", 66, 172, 66, 2),
    # --- boys (a male voice at a raised pitch: espeak-ng has no child voice) --
    Voice("boy-1", CATEGORY_BOY, "Omar", "عمر", "m3", 70, 172, 72, 3),
    Voice("boy-2", CATEGORY_BOY, "Zaid", "زيد", "m5", 76, 178, 70, 2),
    Voice("boy-3", CATEGORY_BOY, "Yusuf", "يوسف", "m2", 68, 168, 74, 3),
    Voice("boy-4", CATEGORY_BOY, "Hamza", "حمزة", "m1", 74, 180, 66, 2),
    # --- girls (a female voice at a raised pitch) --------------------------
    Voice("girl-1", CATEGORY_GIRL, "Hala", "هالة", "f1", 80, 172, 76, 3),
    Voice("girl-2", CATEGORY_GIRL, "Lina", "لينا", "f2", 86, 178, 72, 2),
    Voice("girl-3", CATEGORY_GIRL, "Reem", "ريم", "f3", 78, 168, 74, 3),
    Voice("girl-4", CATEGORY_GIRL, "Dana", "دانة", "f4", 84, 180, 70, 2),
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


def is_system_voice_key(key: str) -> bool:
    """True when ``key`` names a *real installed* voice rather than a catalog one."""
    return key.strip().startswith(SYSTEM_VOICE_PREFIX)


def find_system_voice(key: str) -> SystemVoice | None:
    """The installed voice a ``system:<engine>:<name>`` key names, or ``None``.

    Resolves against the voices the engine really reports (see
    :func:`core.speech.system_voices`), so a saved choice points at a voice that
    exists. ``None`` means the voice is no longer installed here, so the caller can
    fall back honestly instead of asking the engine for a name it would reject.
    """
    if not is_system_voice_key(key):
        return None
    for voice in system_voices():
        if voice.key == key.strip():
            return voice
    return None


def system_voices() -> tuple[SystemVoice, ...]:
    """The real voices installed on this machine, or ``()`` when none are known.

    A thin wrapper over :func:`core.speech.system_voices` so callers that only need
    the catalog module do not have to import the engine layer; the import is local
    because :mod:`core.speech` itself imports this module.
    """
    from core.speech import system_voices as _system_voices

    return _system_voices()


def is_model_voice_key(key: str) -> bool:
    """True when ``key`` names a voice *inside a local model* rather than a catalog one."""
    return key.strip().startswith(MODEL_VOICE_PREFIX)


def model_voices(engine: str | None = None) -> tuple[ModelVoice, ...]:
    """The voices of the local speech models this machine can actually run.

    Every engine that owns named voices is asked, and each one returns ``()`` when
    it is not installed -- so the list is exactly what can be spoken *here*, never a
    promise of a model that is absent. ``engine`` narrows the answer to one engine.

    The three engines are **VoiceTut-TTS** over audio.cpp (its studio speakers),
    **Piper** (its per-language ONNX voices), and **Qwen3-TTS** (its built-in
    English speakers). The import is local because each engine module imports this
    one, so a top-level import would be circular.
    """
    readers: list[tuple[str, Callable[[], tuple[ModelVoice, ...]]]] = []

    from core.audiocpp_speech import AUDIOCPP_ENGINE_NAME
    from core.audiocpp_speech import voice_list as _audiocpp_voices

    readers.append((AUDIOCPP_ENGINE_NAME, _audiocpp_voices))
    with best_effort("list Piper voices"):
        from core.piper_speech import PIPER_ENGINE_NAME
        from core.piper_speech import voice_list as _piper_voices

        readers.append((PIPER_ENGINE_NAME, _piper_voices))
    with best_effort("list Qwen3-TTS voices"):
        from core.qwen_speech import TTS_ENGINE_NAME
        from core.qwen_speech import voice_list as _qwen_voices

        readers.append((TTS_ENGINE_NAME, _qwen_voices))

    collected: list[ModelVoice] = []
    for name, reader in readers:
        if engine is not None and engine != name:
            continue
        try:
            collected.extend(reader())
        except Exception:  # a broken optional engine must never break a menu
            continue
    return tuple(collected)


@dataclass(frozen=True)
class VoiceSection:
    """One source's voices, as a menu shows them: a title and its voices.

    ``voices`` is a flat tuple so a caller can list or select them directly, and
    ``groups`` pairs a readable header with each subset of them: a language label
    (Arabic first) for the real and model voices, and the category label
    (Men/Women/Boys/Girls) for the synthetic catalog.
    """

    source: str
    title: str
    voices: tuple[Any, ...]
    groups: tuple[tuple[str, tuple[Any, ...]], ...]

    @property
    def empty(self) -> bool:
        """True when this source contributed no voice (so a menu can skip it)."""
        return not self.voices


def _labeled_language_groups(
    voices: tuple[Any, ...],
) -> tuple[tuple[str, tuple[Any, ...]], ...]:
    """The language groups with each group's display label (Arabic first)."""
    return tuple(
        (group[0].language_label(), group) for _code, group in _group_by_language(voices)
    )


def _catalog_groups(voices: tuple[Any, ...]) -> tuple[tuple[str, tuple[Any, ...]], ...]:
    """The synthetic catalog grouped by category (Men/Women/Boys/Girls)."""
    return tuple(
        (label_en, tuple(voice for voice in voices if voice.category == category))
        for category, label_en, _label_ar in CATEGORY_LABELS
        if any(voice.category == category for voice in voices)
    )


def voice_sections(
    *,
    system: tuple[SystemVoice, ...] | None = None,
    model: tuple[ModelVoice, ...] | None = None,
    catalog: bool | None = None,
) -> tuple[VoiceSection, ...]:
    """The voices grouped under the source each came from, in menu order.

    One section per source -- *Installed Voices*, *VoiceTut-TTS*, *Piper*,
    *Qwen3-TTS*, and the synthetic catalog -- each carrying only the voices that
    engine actually reports here, so a menu built from this can never list a voice
    that would fail to speak. Empty sections are dropped, and the synthetic catalog
    is dropped as a whole when no engine can speak it here, because those settings
    would otherwise come out silent.

    ``system``/``model`` default to what this machine really reports and
    ``catalog`` to whether any engine can speak the catalog here; a caller (or a
    test) may pass fixed values instead.
    """
    system_voices_ = system_voices() if system is None else system
    model_voices_ = model_voices() if model is None else model
    if catalog is None:
        # A catalog voice is spoken by whichever engine discovery picks -- the
        # system engine, Piper, or Qwen3-TTS -- so it is offered exactly when one of
        # them can really speak it. A named system voice is proof on its own.
        from core.speech import speech_available

        catalog = bool(system_voices_) or speech_available()

    by_source: dict[str, list[Any]] = {}
    for system_voice in system_voices_:
        by_source.setdefault(SOURCE_SYSTEM, []).append(system_voice)
    for model_voice in model_voices_:
        by_source.setdefault(model_voice.source, []).append(model_voice)
    if catalog:
        for catalog_voice in VOICE_CATALOG:
            by_source.setdefault(SOURCE_SYNTHETIC, []).append(catalog_voice)

    sections: list[VoiceSection] = []
    for source, title in SOURCE_TITLES:
        voices = tuple(by_source.get(source, ()))
        if not voices:
            continue
        groups = (
            _catalog_groups(voices)
            if source == SOURCE_SYNTHETIC
            else _labeled_language_groups(voices)
        )
        sections.append(VoiceSection(source, title, voices, groups))
    return tuple(sections)


def find_model_voice(key: str) -> ModelVoice | None:
    """The model voice a ``model:<engine>:<name>`` key names, or ``None``.

    Resolves against the voices the installed models really report (see
    :func:`model_voices`), so a saved choice points at a voice that exists. ``None``
    means the model (or that speaker) is no longer installed here, so the caller can
    fall back honestly instead of asking an engine for a name it would reject.
    """
    if not is_model_voice_key(key):
        return None
    for voice in model_voices():
        if voice.key == key.strip():
            return voice
    return None


def default_voice() -> Voice:
    """The voice used when none was chosen."""
    return _BY_KEY[DEFAULT_VOICE_KEY]


def voice_summaries() -> tuple[tuple[str, str, str, str], ...]:
    """``(key, category, label_en, label_ar)`` per voice, for a menu or a table."""
    return tuple(
        (voice.key, voice.category, voice.label_en, voice.label_ar) for voice in VOICE_CATALOG
    )


def _group_by_language(voices: tuple[Any, ...]) -> tuple[tuple[str, tuple[Any, ...]], ...]:
    """``(language_code, voices)`` groups, Arabic first then alphabetical.

    One ordering rule for every voice menu: the language a voice truly speaks is
    what groups it, and Arabic comes first because it is the language this project
    is read in most. The voices inside a group keep the order they were given in,
    so a caller's own ordering (catalog order, the OS's listing) is preserved.
    """
    grouped: dict[str, list[Any]] = {}
    for voice in voices:
        grouped.setdefault(voice.language_code, []).append(voice)
    return tuple(
        (code, tuple(grouped[code]))
        for code in sorted(grouped, key=lambda code: (code != "ar", code))
    )


def group_system_voices(
    voices: tuple[SystemVoice, ...] | None = None,
) -> tuple[tuple[str, tuple[SystemVoice, ...]], ...]:
    """The installed voices grouped by language (Arabic first).

    ``voices`` defaults to :func:`system_voices`, so a menu can group the real
    installed voices with one call and a test can group a fixed list.
    """
    return _group_by_language(voices if voices is not None else system_voices())


def group_model_voices(
    voices: tuple[ModelVoice, ...] | None = None, *, engine: str | None = None
) -> tuple[tuple[str, tuple[ModelVoice, ...]], ...]:
    """The local model voices grouped by language (Arabic first).

    ``voices`` defaults to :func:`model_voices` for the engine, so a menu can group
    the voices of the local models with one call and a test can group a fixed list.
    """
    return _group_by_language(voices if voices is not None else model_voices(engine))


__all__ = [
    "CATEGORY_BOY",
    "CATEGORY_GIRL",
    "CATEGORY_LABELS",
    "CATEGORY_MAN",
    "CATEGORY_WOMAN",
    "DEFAULT_PITCH_RANGE",
    "DEFAULT_VOICE_KEY",
    "DEFAULT_WORD_GAP",
    "LANGUAGE_LABELS",
    "MAX_PITCH",
    "MAX_PITCH_RANGE",
    "MAX_SPEED",
    "MAX_WORD_GAP",
    "MIN_PITCH",
    "MIN_PITCH_RANGE",
    "MIN_SPEED",
    "MIN_WORD_GAP",
    "MODEL_VOICE_PREFIX",
    "SOURCE_AUDIOCPP",
    "SOURCE_PIPER",
    "SOURCE_QWEN",
    "SOURCE_SYNTHETIC",
    "SOURCE_SYSTEM",
    "SOURCE_TITLES",
    "SYSTEM_VOICE_PREFIX",
    "VOICE_CATALOG",
    "ModelVoice",
    "SystemVoice",
    "Voice",
    "VoiceSection",
    "default_voice",
    "detect_language",
    "find_model_voice",
    "find_system_voice",
    "find_voice",
    "group_model_voices",
    "group_system_voices",
    "is_model_voice_key",
    "is_system_voice_key",
    "model_voices",
    "system_voices",
    "voice_categories",
    "voice_keys",
    "voice_sections",
    "voice_summaries",
    "voices_for",
]
