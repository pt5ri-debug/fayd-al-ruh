"""The emotional states a reading voice can be put into, and their prosody.

A reading is not only *which* voice speaks but *how* it speaks. This module is the
one declarative place that second choice lives: a :class:`Mood` names a well-known
emotional state (happy, sad, excited, rapturous, in love, angry, calm, ...) and
carries the **prosody** that produces it -- how far the pitch moves, how fast the
delivery is, how much the melody swings, and, for an instructable engine, a short
natural-language style line.

Honesty is a hard rule, as everywhere in this project:

* A mood is **settings applied to a synthesizer**, not a recorded actor. The class
  says so in its ``description``.
* Every engine applies as much of the prosody as it really supports, and no more.
  A formant engine (``espeak-ng``) takes pitch, speed, intonation range, and word
  gap; Piper takes pace and expressiveness; Qwen3-TTS takes a style instruction;
  the VoiceTut cloning engine takes only the reference clip, so a mood changes
  little there and that is stated rather than hidden. :func:`support_note` is the
  one place that per-engine truth is written down.
* A mood never invents a feeling in the *text*. It changes the delivery of the
  words that are already there.

The prosody is deliberately small and bounded: a pitch offset in engine points, a
speed multiplier, an intonation-range offset, a word-gap offset, an energy
multiplier, and an instruction string. :func:`apply_mood` folds the numeric part
onto a catalog :class:`~core.voices.Voice` (clamped to the catalog's own limits),
and the engines read the rest through :class:`~core.speech.load_synthesizer`'s
``prosody`` option, so one mood behaves the same wherever it is chosen.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

#: The mood used when the user has not chosen one: the voice's own neutral reading.
DEFAULT_MOOD_KEY = "neutral"

#: Bounds for the numeric prosody, so a bad entry can never ask an engine for an
#: unreadable delivery. They are generous enough for the strongest mood and tight
#: enough that no single setting can turn speech into noise.
MIN_PITCH_DELTA = -40
MAX_PITCH_DELTA = 40
MIN_SPEED_SCALE = 0.6
MAX_SPEED_SCALE = 1.6
MIN_RANGE_DELTA = -30
MAX_RANGE_DELTA = 30
MIN_GAP_DELTA = -2
MAX_GAP_DELTA = 8
MIN_ENERGY = 0.6
MAX_ENERGY = 1.8


@dataclass(frozen=True)
class Prosody:
    """How a mood shapes a delivery, in engine-neutral terms.

    Every field is a *delta* or a *multiplier* from the chosen voice's own settings,
    so a mood layers onto whatever timbre the user picked rather than replacing it:

    * ``pitch_delta`` -- added to the voice's pitch register (a raised, brighter
      delivery for joy; a lowered, heavier one for sadness).
    * ``speed_scale`` -- multiplied into the voice's speed (a hurried, excited
      delivery; a slow, weighted one).
    * ``range_delta`` -- added to the voice's intonation range (a wide, singing
      melody for rapture; a narrow, flat one for a subdued mood).
    * ``gap_delta`` -- added to the voice's word gap (a breathless, close delivery
      versus a measured, spaced one).
    * ``energy`` -- a loudness/emphasis multiplier, clamped so it stays audible.
    * ``instruct`` -- a short style line for an engine that takes natural-language
      direction (Qwen3-TTS). Empty means "no instruction".
    """

    pitch_delta: int = 0
    speed_scale: float = 1.0
    range_delta: int = 0
    gap_delta: int = 0
    energy: float = 1.0
    instruct: str = ""

    def __post_init__(self) -> None:
        for name, value, low, high in (
            ("pitch_delta", self.pitch_delta, MIN_PITCH_DELTA, MAX_PITCH_DELTA),
            ("speed_scale", self.speed_scale, MIN_SPEED_SCALE, MAX_SPEED_SCALE),
            ("range_delta", self.range_delta, MIN_RANGE_DELTA, MAX_RANGE_DELTA),
            ("gap_delta", self.gap_delta, MIN_GAP_DELTA, MAX_GAP_DELTA),
            ("energy", self.energy, MIN_ENERGY, MAX_ENERGY),
        ):
            if not low <= value <= high:
                raise ValueError(f"Prosody {name} must be in [{low}, {high}] (got {value}).")

    @property
    def neutral(self) -> bool:
        """True when this prosody leaves the voice exactly as it is."""
        return (
            self.pitch_delta == 0
            and self.speed_scale == 1.0
            and self.range_delta == 0
            and self.gap_delta == 0
            and self.energy == 1.0
            and not self.instruct
        )

    def to_primitive(self) -> dict[str, object]:
        return {
            "pitch_delta": self.pitch_delta,
            "speed_scale": self.speed_scale,
            "range_delta": self.range_delta,
            "gap_delta": self.gap_delta,
            "energy": self.energy,
            "instruct": self.instruct,
        }


@dataclass(frozen=True)
class Mood:
    """One selectable emotional state: a name, an honest note, and its prosody."""

    key: str
    label_en: str
    label_ar: str
    description_en: str
    description_ar: str
    prosody: Prosody

    def __post_init__(self) -> None:
        if not self.key.strip():
            raise ValueError("A mood needs a key.")
        if not self.label_en.strip():
            raise ValueError(f"Mood {self.key!r} needs an English label.")

    def label(self, language: str = "en") -> str:
        """The mood's name in ``language`` (falling back to English)."""
        return self.label_ar if language == "ar" else self.label_en

    def description(self, language: str = "en") -> str:
        """An honest note about what this mood does to a delivery."""
        return self.description_ar if language == "ar" else self.description_en

    @property
    def neutral(self) -> bool:
        """True when the mood leaves the chosen voice exactly as it is."""
        return self.prosody.neutral

    def to_primitive(self) -> dict[str, object]:
        return {
            "key": self.key,
            "label_en": self.label_en,
            "label_ar": self.label_ar,
            "prosody": self.prosody.to_primitive(),
        }


#: Every mood a reading can be put into, in menu order. The prosody is the whole
#: point of the table: each entry is tuned so the *delivery* changes while the
#: words do not. They are the states people actually ask for -- joy and sorrow,
#: excitement and calm, rapture and longing, anger and tenderness, and the neutral
#: reading the voice already has.
MOODS: tuple[Mood, ...] = (
    Mood(
        "neutral", "Neutral", "محايد",
        "The voice's own reading, with no change at all.",
        "قراءة الصوت كما هي، دون أي تغيير.",
        Prosody(),
    ),
    Mood(
        "happy", "Happy", "فرحان",
        "Brighter and a little quicker, with a lifted melody.",
        "أكثر إشراقاً وأسرع قليلاً، مع نغمة مرتفعة.",
        Prosody(pitch_delta=12, speed_scale=1.08, range_delta=8, gap_delta=-1, energy=1.15,
                instruct="Speak in a cheerful, smiling, upbeat tone."),
    ),
    Mood(
        "sad", "Sad", "حزين",
        "Lower, slower, and narrower, with longer pauses.",
        "أخفض وأبطأ وأضيق، مع وقفات أطول.",
        Prosody(pitch_delta=-14, speed_scale=0.85, range_delta=-14, gap_delta=2, energy=0.85,
                instruct="Speak in a sorrowful, heavy, subdued tone."),
    ),
    Mood(
        "excited", "Excited", "متحمس",
        "Fast, bright, and energetic, with a wide swing.",
        "سريع ومشرق ومفعم بالطاقة، مع تموّج واسع.",
        Prosody(pitch_delta=18, speed_scale=1.22, range_delta=16, gap_delta=-1, energy=1.4,
                instruct="Speak in an excited, energetic, enthusiastic tone."),
    ),
    Mood(
        "rapturous", "Rapturous (هيام)", "هيام",
        "A slow, wide, soaring delivery, as if carried away.",
        "أداء بطيء واسع ومحلّق، كأنه منساق.",
        Prosody(pitch_delta=14, speed_scale=0.88, range_delta=26, gap_delta=2, energy=1.2,
                instruct="Speak in a rapturous, dreamy, carried-away tone."),
    ),
    Mood(
        "in_love", "In love", "في حالة حب",
        "Warm, close, and tender, with a soft, rising melody.",
        "دافئ وقريب وحنون، مع نغمة ناعمة صاعدة.",
        Prosody(pitch_delta=8, speed_scale=0.9, range_delta=18, gap_delta=1, energy=1.0,
                instruct="Speak in a warm, loving, tender tone."),
    ),
    Mood(
        "angry", "Angry", "هياج",
        "Loud, fast, and sharp, with a hard, narrow melody.",
        "عالٍ وسريع وحاد، مع نغمة حادة ضيقة.",
        Prosody(pitch_delta=6, speed_scale=1.3, range_delta=-8, gap_delta=-2, energy=1.8,
                instruct="Speak in an angry, forceful, agitated tone."),
    ),
    Mood(
        "calm", "Calm", "هادئ",
        "Slow and even, with a steady, unhurried melody.",
        "بطيء ومتّزن، مع نغمة ثابتة غير متعجلة.",
        Prosody(pitch_delta=-4, speed_scale=0.82, range_delta=-6, gap_delta=3, energy=0.9,
                instruct="Speak in a calm, steady, unhurried tone."),
    ),
    Mood(
        "tender", "Tender", "حنان",
        "Gentle and low, with a soft, rounded melody.",
        "لطيف ومنخفض، مع نغمة ناعمة مستديرة.",
        Prosody(pitch_delta=-6, speed_scale=0.86, range_delta=6, gap_delta=2, energy=0.85,
                instruct="Speak in a gentle, tender, soothing tone."),
    ),
    Mood(
        "serious", "Serious", "جدي",
        "Measured and firm, with a level, deliberate melody.",
        "متّزن ورصين، مع نغمة مستوية متأنية.",
        Prosody(pitch_delta=-8, speed_scale=0.92, range_delta=-16, gap_delta=1, energy=1.05,
                instruct="Speak in a serious, grave, deliberate tone."),
    ),
    Mood(
        "surprised", "Surprised", "مندهش",
        "Quick and lifted, with a sharp rise at the start.",
        "سريع ومرتفع، مع صعود حاد في البداية.",
        Prosody(pitch_delta=20, speed_scale=1.1, range_delta=20, gap_delta=0, energy=1.25,
                instruct="Speak in a surprised, astonished tone."),
    ),
    Mood(
        "fearful", "Fearful", "خائف",
        "Hushed and quick, with a trembling, narrow melody.",
        "خفيض وسريع، مع نغمة مرتجفة ضيقة.",
        Prosody(pitch_delta=6, speed_scale=1.15, range_delta=-4, gap_delta=-1, energy=0.8,
                instruct="Speak in a fearful, anxious, hesitant tone."),
    ),
    Mood(
        "hopeful", "Hopeful", "متفائل",
        "Warm and rising, with an open, forward melody.",
        "دافئ وصاعد، مع نغمة منفتحة متقدمة.",
        Prosody(pitch_delta=10, speed_scale=1.0, range_delta=12, gap_delta=0, energy=1.1,
                instruct="Speak in a hopeful, optimistic, encouraging tone."),
    ),
    Mood(
        "nostalgic", "Nostalgic", "حنين",
        "Slow and soft, with a lingering, wistful melody.",
        "بطيء وناعم، مع نغمة متلهفة حزينة الشجن.",
        Prosody(pitch_delta=-8, speed_scale=0.84, range_delta=10, gap_delta=3, energy=0.85,
                instruct="Speak in a nostalgic, wistful, remembering tone."),
    ),
    Mood(
        "proud", "Proud", "فخور",
        "Strong and deliberate, with a firm, lifted melody.",
        "قوي ومتأنٍّ، مع نغمة راسخة مرتفعة.",
        Prosody(pitch_delta=8, speed_scale=0.94, range_delta=4, gap_delta=1, energy=1.3,
                instruct="Speak in a proud, confident, dignified tone."),
    ),
    Mood(
        "playful", "Playful", "مرح",
        "Lively and bouncy, with a teasing, skipping melody.",
        "حيوي وخفيف، مع نغمة مداعبة متقافزة.",
        Prosody(pitch_delta=14, speed_scale=1.12, range_delta=22, gap_delta=-1, energy=1.2,
                instruct="Speak in a playful, teasing, light-hearted tone."),
    ),
    Mood(
        "whisper", "Whisper", "همس",
        "Very soft and close, with a quiet, intimate melody.",
        "خفيض جداً وقريب، مع نغمة هادئة حميمية.",
        Prosody(pitch_delta=-4, speed_scale=0.88, range_delta=-2, gap_delta=2, energy=0.6,
                instruct="Speak in a soft, close whisper."),
    ),
    Mood(
        "storytelling", "Storytelling", "سرد",
        "Measured and expressive, with a drawing, unfolding melody.",
        "متّزن ومعبّر، مع نغمة جاذبة متدرجة.",
        Prosody(pitch_delta=4, speed_scale=0.94, range_delta=18, gap_delta=2, energy=1.1,
                instruct="Speak in a warm, expressive storytelling tone."),
    ),
    Mood(
        "formal", "Formal", "رسمي",
        "Even and precise, with a restrained, level melody.",
        "متّزن ودقيق، مع نغمة متحفّظة مستوية.",
        Prosody(pitch_delta=-6, speed_scale=0.96, range_delta=-18, gap_delta=1, energy=1.0,
                instruct="Speak in a formal, precise, professional tone."),
    ),
    Mood(
        "empathetic", "Empathetic", "متعاطف",
        "Warm and unhurried, with a caring, reassuring melody.",
        "دافئ وغير متعجل، مع نغمة حانية مطمئنة.",
        Prosody(pitch_delta=-2, speed_scale=0.9, range_delta=8, gap_delta=2, energy=0.95,
                instruct="Speak in a warm, caring, empathetic tone."),
    ),
)

_BY_KEY: dict[str, Mood] = {mood.key: mood for mood in MOODS}


def mood_keys() -> tuple[str, ...]:
    """Every mood key, in menu order."""
    return tuple(mood.key for mood in MOODS)


def default_mood() -> Mood:
    """The mood used when none was chosen."""
    return _BY_KEY[DEFAULT_MOOD_KEY]


def find_mood(key: str) -> Mood:
    """Return the mood named ``key``; raise ``ValueError`` for an unknown one."""
    try:
        return _BY_KEY[key.strip()]
    except KeyError:
        raise ValueError(
            f"Unknown mood {key!r}; choose one of {', '.join(_BY_KEY)}."
        ) from None


def resolve_mood(key: str | None) -> Mood:
    """A saved mood key, falling back to the neutral mood when it is gone.

    A mood that left the catalog must never break a saved choice, so an unknown
    key yields the neutral mood rather than raising.
    """
    if not key:
        return default_mood()
    return _BY_KEY.get(key.strip(), default_mood())


def mood_summaries() -> tuple[tuple[str, str, str, str], ...]:
    """``(key, label_en, label_ar, description_en)`` per mood, for a menu or a table."""
    return tuple(
        (mood.key, mood.label_en, mood.label_ar, mood.description_en) for mood in MOODS
    )


def apply_mood(voice: Any, mood: Mood | Prosody) -> Any:
    """Fold a mood's numeric prosody onto a catalog :class:`~core.voices.Voice`.

    Accepts a :class:`Mood` or a bare :class:`Prosody` (the engine layer carries the
    latter). A catalog voice *is* a set of engine settings, so a mood can be folded
    straight into it: the pitch, speed, intonation range, and word gap are shifted
    and clamped to the catalog's own limits, and the result is a new ``Voice`` the
    engines already know how to read. A named voice (an installed
    :class:`~core.voices.SystemVoice`, or a :class:`~core.voices.ModelVoice` inside
    a local model) keeps its own timbre -- the engine picks it by name -- so it is
    returned unchanged; those engines take the prosody directly instead (see
    :func:`~core.speech.load_synthesizer`'s ``prosody`` option).
    """
    from core.voices import MAX_PITCH, MAX_PITCH_RANGE, MAX_SPEED, MAX_WORD_GAP, MIN_WORD_GAP, Voice

    prosody = mood.prosody if isinstance(mood, Mood) else mood
    if prosody.neutral or not isinstance(voice, Voice):
        return voice
    return replace(
        voice,
        pitch=max(0, min(MAX_PITCH, voice.pitch + prosody.pitch_delta)),
        speed=max(1, min(MAX_SPEED, round(voice.speed * prosody.speed_scale))),
        pitch_range=max(0, min(MAX_PITCH_RANGE, voice.pitch_range + prosody.range_delta)),
        word_gap=max(MIN_WORD_GAP, min(MAX_WORD_GAP, voice.word_gap + prosody.gap_delta)),
    )


#: How much of a mood each engine can really apply, in one honest line per engine.
#: This is the truth the window shows beside the mood list, so a user is never told
#: a cloning engine will change its tone when it cannot.
ENGINE_MOOD_SUPPORT: dict[str, str] = {
    "espeak": "full (pitch, speed, melody, and pauses)",
    "espeak-ng": "full (pitch, speed, melody, and pauses)",
    "windows-sapi": "partial (rate and register)",
    "windows-onecore": "partial (rate, register, and emphasis)",
    "piper": "partial (pace and expressiveness)",
    "qwen3-tts": "full (a natural-language style instruction)",
    "audio.cpp": "limited (a cloning engine keeps its reference voice; mood changes little)",
    "say": "partial (rate)",
    "flite": "none (this engine takes no prosody)",
    "silent": "none (no engine is installed, so nothing is spoken)",
}


def support_note(engine: str) -> str:
    """An honest note about how much of a mood ``engine`` can apply."""
    return ENGINE_MOOD_SUPPORT.get(
        engine, "unknown (this engine does not report what it can change)"
    )


__all__ = [
    "DEFAULT_MOOD_KEY",
    "ENGINE_MOOD_SUPPORT",
    "MOODS",
    "Mood",
    "Prosody",
    "apply_mood",
    "default_mood",
    "find_mood",
    "mood_keys",
    "mood_summaries",
    "resolve_mood",
    "support_note",
]
