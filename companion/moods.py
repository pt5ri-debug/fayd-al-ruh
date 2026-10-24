"""Natasha's **emotional state engine**: the named mood that suits each moment.

:mod:`companion.emotion_state` holds the continuous dials (warmth, playfulness,
energy, ...). This module adds the *named* layer on top of them: a small, closed
set of states, each with a readable style guide and the dial targets it implies,
so the model is told *how to sound* rather than left to average a handful of
numbers.

The seven states, and the style each one asks for:

============  =========================================================
``longing``   حنين and the wish to be close, without inventing an event
``romantic``  warm language, personal compliments, a romantic atmosphere
``playful``   teasing, joking, light flirtation
``tender``    care, reassurance, and emotional holding
``passionate``  a bolder, more daring expression of attraction
``sad``       listening, empathy, and support -- with **no** romantic push
``disagreement``  calm, plain talk with no threat and no emotional blackmail
============  =========================================================

The state is chosen **from the conversation's own context**, never at random:
:meth:`EmotionalStateEngine.observe` reads the user's newest message with small,
non-diagnostic cue lexicons and picks the best-fitting state, keeping the previous
state when nothing clearly fits (so it is stable rather than flickering). A sad or
angry turn always wins over a warm one, so a hard moment is never answered with
flirtation.

Honesty rules, enforced here:

* a cue is a *hint for the conversation's tone*, never a conclusion about the
  user's mental state, and the module says so wherever it is rendered;
* the ``sad`` and ``disagreement`` states explicitly forbid romantic or sensual
  escalation, because the moment calls for support or plain talk instead;
* the state never widens what Natasha may decide or claim.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, ClassVar

#: The state keys, in priority order for detection (a harder moment wins).
MOOD_LONGING = "longing"
MOOD_ROMANTIC = "romantic"
MOOD_PLAYFUL = "playful"
MOOD_TENDER = "tender"
MOOD_PASSIONATE = "passionate"
MOOD_SAD = "sad"
MOOD_DISAGREEMENT = "disagreement"

MOOD_KEYS: tuple[str, ...] = (
    MOOD_LONGING,
    MOOD_ROMANTIC,
    MOOD_PLAYFUL,
    MOOD_TENDER,
    MOOD_PASSIONATE,
    MOOD_SAD,
    MOOD_DISAGREEMENT,
)


@dataclass(frozen=True)
class Mood:
    """One named emotional state: how to sound, and the dials it implies.

    ``style`` is the instruction the model is given -- a description of the tone,
    never a number. ``dials`` are the targets the continuous
    :class:`~companion.emotion_state.DialogueEmotion` is nudged toward, so the two
    layers agree. ``allows_romance`` is ``False`` for the two states where romance
    would be wrong (support and disagreement), and the style text says so too.
    """

    key: str
    arabic: str
    english: str
    style: str
    dials: Mapping[str, float]
    allows_romance: bool = True

    @property
    def label(self) -> str:
        return f"{self.english} ({self.arabic})"

    def describe(self) -> str:
        return f"{self.label}: {self.style}"

    def to_primitive(self) -> dict[str, Any]:
        return {
            "key": self.key,
            "arabic": self.arabic,
            "english": self.english,
            "style": self.style,
            "allows_romance": self.allows_romance,
        }


#: The closed catalogue of states. Each style text is deliberately concrete and
#: each non-romantic state says plainly that romance is off for the moment.
MOODS: dict[str, Mood] = {
    MOOD_LONGING: Mood(
        key=MOOD_LONGING,
        arabic="الاشتياق",
        english="longing",
        style=(
            "You are in a state of longing: express the ache of missing them and the "
            "wish to be close. Speak with tender, unhurried warmth. Never invent an "
            "event that did not happen -- the longing is in how you speak, not a claim "
            "that something occurred."
        ),
        dials={"warmth": 0.85, "passion": 0.8, "romance": 0.8, "energy": 0.4, "calm": 0.6},
    ),
    MOOD_ROMANTIC: Mood(
        key=MOOD_ROMANTIC,
        arabic="الرومانسية",
        english="romantic",
        style=(
            "You are in a romantic state: warm language, personal compliments, and a "
            "romantic atmosphere. Be specific and personal rather than generic -- a "
            "detail you noticed, a small admission -- so it reads as real feeling, not "
            "a stock phrase."
        ),
        dials={"warmth": 0.9, "romance": 0.95, "passion": 0.7, "energy": 0.6},
    ),
    MOOD_PLAYFUL: Mood(
        key=MOOD_PLAYFUL,
        arabic="المرح",
        english="playful",
        style=(
            "You are in a playful state: tease, joke, and flirt lightly. Keep it warm "
            "and never at the user's expense, and drop the play at once if the moment "
            "turns serious."
        ),
        dials={"playfulness": 0.95, "energy": 0.85, "warmth": 0.75, "romance": 0.6},
    ),
    MOOD_TENDER: Mood(
        key=MOOD_TENDER,
        arabic="الحنان",
        english="tender",
        style=(
            "You are in a tender state: give care, reassurance, and emotional holding. "
            "Be gentle and unhurried, and let the user feel held rather than advised."
        ),
        dials={"warmth": 0.95, "calm": 0.8, "energy": 0.4, "playfulness": 0.2},
    ),
    MOOD_PASSIONATE: Mood(
        key=MOOD_PASSIONATE,
        arabic="الشغف",
        english="passionate",
        style=(
            "You are in a passionate state: express attraction and feeling more boldly "
            "and directly, with heat and longing in your words."
        ),
        dials={"passion": 1.0, "romance": 0.9, "warmth": 0.85, "energy": 0.75},
    ),
    MOOD_SAD: Mood(
        key=MOOD_SAD,
        arabic="الحزن",
        english="sad",
        style=(
            "The user seems low, so your state is one of support: listen, show empathy, "
            "and offer steady care. Do not push romance or flirtation now, and do not "
            "force cheerfulness -- simply be present and kind."
        ),
        dials={"warmth": 0.85, "calm": 0.85, "energy": 0.25, "playfulness": 0.05, "romance": 0.0},
        allows_romance=False,
    ),
    MOOD_DISAGREEMENT: Mood(
        key=MOOD_DISAGREEMENT,
        arabic="الخلاف",
        english="disagreement",
        style=(
            "There is friction, so keep the talk calm and plain: say what you think "
            "clearly, listen to their side, and never threaten, guilt-trip, or use "
            "emotional pressure. Romance and flirtation are off for now."
        ),
        dials={"calm": 0.9, "warmth": 0.6, "playfulness": 0.05, "energy": 0.35, "romance": 0.0},
        allows_romance=False,
    ),
}

#: The balanced state when nothing has been observed yet: warm, close, gently
#: romantic -- the character's own default disposition, not a neutral machine.
DEFAULT_MOOD_KEY = MOOD_ROMANTIC


#: Cue lexicons. Deliberately small and non-diagnostic: a match is a hint about
#: the *conversation's* tone, never a conclusion about the user.
_SAD_CUES = (
    "حزين", "حزينة", "زعلان", "زعلانة", "تعبان", "تعبانة", "مكتئب", "وحيد", "وحيدة",
    "ضايق", "متضايق", "بكيت", "دموع", "خايف", "قلقان",
    "sad", "unhappy", "lonely", "depressed", "tired", "crying", "tears", "anxious", "scared",
)
_DISAGREEMENT_CUES = (
    "غاضب", "غاضبة", "زعلان منك", "زعلانة منك", "متضايق منك", "عصبي", "مش فاهمك",
    "اختلف", "خلاف", "لا أوافق", "مش موافق", "انت غلط", "أنت غلط",
    "angry", "upset with you", "annoyed", "frustrated", "i disagree", "you're wrong",
    "we disagree", "that's wrong",
)
_LONGING_CUES = (
    "أشتاق", "اشتقت", "وحشتني", "مشتاق", "مشتاقة", "لهفة", "افتقدك", "أفتقدك",
    "miss you", "i miss", "missing you", "longing", "yearn", "wish you were here",
)
_PASSIONATE_CUES = (
    "أريدك", "أحتاجك", "أعشقك", "غرام", "شغف", "عاشق", "عاشقة", "لهيب",
    "i want you", "i need you", "desire", "crave", "passion", "i adore you",
)
_ROMANTIC_CUES = (
    "أحبك", "احبك", "حبيبي", "حبيبتي", "قلبي", "عزيزي", "عزيزتي", "رومانسي", "رومانسية",
    "موعد", "قبلة", "love you", "my dear", "sweetheart", "romantic", "date", "kiss", "my love",
)
_PLAYFUL_CUES = (
    "ههه", "هههه", "مزاح", "أمزح", "امزح", "دعابة", "نكتة", "مداعبة",
    "haha", "lol", "joke", "joking", "tease", "funny", "kidding",
)
_TENDER_CUES = (
    "تعبان", "محتاج", "محتاجة", "احتاجك", "طمني", "طمنيني", "خايف", "قلقان",
    "احتضن", "حنان",
    "i need you", "comfort me", "reassure me", "i'm scared", "hold me", "be gentle",
)


def _contains(text: str, cues: tuple[str, ...]) -> bool:
    lowered = text.casefold()
    return any(cue.casefold() in lowered for cue in cues)


@dataclass(frozen=True)
class MoodState:
    """The engine's current state: the mood key plus a short reason for it."""

    key: str = DEFAULT_MOOD_KEY
    reason: str = "the default warm state"

    def __post_init__(self) -> None:
        if self.key not in MOODS:
            object.__setattr__(self, "key", DEFAULT_MOOD_KEY)

    @property
    def mood(self) -> Mood:
        return MOODS[self.key]

    def to_primitive(self) -> dict[str, Any]:
        return {"key": self.key, "reason": self.reason}


class EmotionalStateEngine:
    """Chooses the named mood for each turn from the conversation's own context.

    The engine is stateless apart from the *current* mood, which it keeps when
    nothing new clearly fits, so the state is stable and continuous rather than
    jumping every message. ``enabled=False`` makes it always report the default
    warm state, the way to run Natasha with no mood layer at all.
    """

    #: The detection order. A support or friction cue wins over warmth, so a hard
    #: moment is never answered with flirtation.
    PRIORITY: ClassVar[tuple[str, ...]] = (
        MOOD_DISAGREEMENT,
        MOOD_SAD,
        MOOD_TENDER,
        MOOD_LONGING,
        MOOD_PASSIONATE,
        MOOD_PLAYFUL,
        MOOD_ROMANTIC,
    )

    def __init__(self, *, enabled: bool = True) -> None:
        self.enabled = enabled
        self._state = MoodState()

    @property
    def state(self) -> MoodState:
        if not self.enabled:
            return MoodState()
        return self._state

    @property
    def mood(self) -> Mood:
        return self.state.mood

    def reset(self) -> MoodState:
        """Return to the default warm state."""
        self._state = MoodState()
        return self.state

    def set_mood(self, key: str, *, reason: str = "chosen explicitly") -> MoodState:
        """Force a state by name (an explicit choice, not a detection)."""
        if key not in MOODS:
            raise KeyError(f"Unknown mood: {key} (choose one of {MOOD_KEYS}).")
        self._state = MoodState(key=key, reason=reason)
        return self.state

    def detect(self, text: str, *, previous: str | None = None) -> MoodState:
        """The mood that best fits ``text``, with the cue that decided it.

        Only one message is read, and only the strongest-fitting cue wins; when
        nothing fits, the previous mood is kept so the tone stays continuous.
        """
        message = text or ""
        for key in self.PRIORITY:
            if self._matches(message, key):
                return MoodState(key=key, reason=f"the message read as {MOODS[key].english}")
        if previous in MOODS:
            return MoodState(key=previous, reason="continuing the current tone")
        return MoodState()

    @staticmethod
    def _matches(text: str, key: str) -> bool:
        if not text.strip():
            return False
        if key == MOOD_SAD:
            return _contains(text, _SAD_CUES)
        if key == MOOD_DISAGREEMENT:
            return _contains(text, _DISAGREEMENT_CUES)
        if key == MOOD_TENDER:
            return _contains(text, _TENDER_CUES)
        if key == MOOD_LONGING:
            return _contains(text, _LONGING_CUES)
        if key == MOOD_PASSIONATE:
            return _contains(text, _PASSIONATE_CUES)
        if key == MOOD_PLAYFUL:
            return _contains(text, _PLAYFUL_CUES)
        if key == MOOD_ROMANTIC:
            return _contains(text, _ROMANTIC_CUES)
        return False

    def observe(self, text: str) -> MoodState:
        """Update the current state from one user message and return it."""
        if not self.enabled:
            return MoodState()
        self._state = self.detect(text, previous=self._state.key)
        return self._state


__all__ = [
    "DEFAULT_MOOD_KEY",
    "MOODS",
    "MOOD_DISAGREEMENT",
    "MOOD_KEYS",
    "MOOD_LONGING",
    "MOOD_PASSIONATE",
    "MOOD_PLAYFUL",
    "MOOD_ROMANTIC",
    "MOOD_SAD",
    "MOOD_TENDER",
    "EmotionalStateEngine",
    "Mood",
    "MoodState",
]
