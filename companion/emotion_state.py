"""Natasha's *dialogue emotion*: a small, bounded, simulated tone.

This is not an attempt to measure the user's feelings, and it is not a claim that
Natasha has real emotions. It is a handful of 0..1 dials that shape *how she
speaks* -- a narrative colour, like a character's mood in a story. The rules that
keep it honest:

* every value is a number in ``[0, 1]`` with a stated meaning;
* it moves **gradually** -- one message can nudge a dial by at most
  ``MAX_STEP``, never teleport it;
* it drifts back toward a balanced baseline over turns, so a single sharp
  message does not colour the whole conversation;
* a single sad or angry message is treated as a *cue for the conversation*, not
  as proof of the user's psychological state;
* it can be disabled, reset, and persisted, and it is stored separately from the
  permanent personality (see :mod:`companion.personality`).

The values are saved per conversation so a resumed session keeps its tone.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass, replace
from typing import Any, ClassVar

from companion.database import CompanionDatabase, _utc_now

#: The most any single dial may move on one message, so changes stay gradual.
MAX_STEP = 0.15

#: How fast a dial drifts back toward the baseline each turn (0 = never, 1 = at once).
DEFAULT_DECAY = 0.12


def _clamp(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


#: The dials and, in the docstring, exactly what each one means.
@dataclass(frozen=True)
class DialogueEmotion:
    """The simulated tone, each field a 0..1 dial.

    * ``warmth``            -- how affectionate her phrasing is (0 reserved .. 1 very warm).
    * ``playfulness``       -- how light and humorous (0 serious .. 1 playful).
    * ``energy``            -- how lively the exchange feels (0 subdued .. 1 energetic).
    * ``calm``              -- how unhurried she is (0 quick .. 1 very calm).
    * ``context_sensitivity`` -- how carefully she is tracking the mood (0 steady .. 1 attentive).
    * ``romance``           -- how much romantic colour is appropriate (0 none .. 1 open).
    """

    warmth: float = 0.5
    playfulness: float = 0.5
    energy: float = 0.5
    calm: float = 0.5
    context_sensitivity: float = 0.5
    romance: float = 0.2

    FIELDS: ClassVar[tuple[str, ...]] = (
        "warmth",
        "playfulness",
        "energy",
        "calm",
        "context_sensitivity",
        "romance",
    )

    def __post_init__(self) -> None:
        for name in self.FIELDS:
            object.__setattr__(self, name, _clamp(getattr(self, name)))

    def to_primitive(self) -> dict[str, float]:
        return {name: float(getattr(self, name)) for name in self.FIELDS}

    @classmethod
    def from_primitive(cls, data: Mapping[str, Any]) -> DialogueEmotion:
        base = cls()
        values = base.to_primitive()
        for name in cls.FIELDS:
            if name in data:
                try:
                    values[name] = float(data[name])
                except (TypeError, ValueError):
                    continue
        return cls(**values)

    @classmethod
    def baseline(cls) -> DialogueEmotion:
        """The balanced state the dials drift back toward."""
        return cls()

    def nudged(self, deltas: Mapping[str, float], *, max_step: float = MAX_STEP) -> DialogueEmotion:
        """A copy moved toward ``deltas``, with each dial clamped to ``max_step``."""
        changes: dict[str, float] = {}
        for name in self.FIELDS:
            delta = deltas.get(name, 0.0)
            if delta == 0.0:
                continue
            step = max(-max_step, min(max_step, float(delta)))
            changes[name] = _clamp(getattr(self, name) + step)
        return replace(self, **changes) if changes else self

    def decayed(self, baseline: DialogueEmotion | None = None, *, rate: float = DEFAULT_DECAY) -> DialogueEmotion:
        """A copy drifted one step toward ``baseline`` (a balanced tone by default)."""
        target = baseline or DialogueEmotion.baseline()
        rate = max(0.0, min(1.0, rate))
        return replace(
            self,
            **{
                name: _clamp(getattr(self, name) + (getattr(target, name) - getattr(self, name)) * rate)
                for name in self.FIELDS
            },
        )

    def describe(self) -> str:
        """A short, human-readable summary of the tone."""
        def word(value: float, low: str, mid: str, high: str) -> str:
            return low if value < 0.34 else mid if value < 0.67 else high

        return (
            f"warmth={word(self.warmth, 'reserved', 'warm', 'very warm')}, "
            f"playfulness={word(self.playfulness, 'serious', 'light', 'playful')}, "
            f"energy={word(self.energy, 'subdued', 'even', 'lively')}, "
            f"calm={word(self.calm, 'brisk', 'steady', 'very calm')}, "
            f"attention={word(self.context_sensitivity, 'steady', 'attentive', 'very attentive')}, "
            f"romance={word(self.romance, 'none', 'subtle', 'open')}"
        )


#: Mild cue lexicons. These are deliberately small and non-diagnostic: they only
#: shape the conversation's tone, and a match is a hint, never a conclusion.
_SAD_CUES = ("حزين", "حزينة", "زعلان", "تعبان", "مكتئب", "وحيد", "sad", "unhappy", "lonely", "tired")
_ANGRY_CUES = ("غاضب", "زعلان منك", "متضايق", "عصبي", "angry", "upset", "annoyed", "frustrated")
_HAPPY_CUES = ("سعيد", "مبسوط", "فرحان", "ممتاز", "رائع", "happy", "great", "excited", "wonderful")
_AFFECTION_CUES = ("أحبك", "احبك", "حبيبتي", "عزيزي", "قلبي", "love you", "my dear", "sweetheart")
_ROMANCE_CUES = ("رومانسي", "رومانسية", "موعد", "قبلة", "romantic", "date", "kiss")
_QUESTION_CUES = ("؟", "?", "لماذا", "كيف", "why", "how", "what do you think")


def _contains(text: str, cues: tuple[str, ...]) -> bool:
    lowered = text.casefold()
    return any(cue.casefold() in lowered for cue in cues)


class DialogueEmotionTracker:
    """Tracks and persists the dialogue tone for one conversation at a time.

    The tracker holds the current state in memory and mirrors it to the database
    after each change, so a resumed session keeps its tone. ``enabled=False``
    makes every operation a no-op that returns a neutral baseline -- the way to
    run Natasha with no emotion layer at all.
    """

    def __init__(self, database: CompanionDatabase | None = None, *, enabled: bool = True) -> None:
        self._db = database
        self.enabled = enabled
        self._state = DialogueEmotion.baseline()

    @property
    def state(self) -> DialogueEmotion:
        return self._state if self.enabled else DialogueEmotion.baseline()

    def reset(self, conversation_id: str | None = None) -> DialogueEmotion:
        """Return the tone to its balanced baseline (and persist that)."""
        self._state = DialogueEmotion.baseline()
        if self._db is not None and conversation_id:
            self._persist(conversation_id)
        return self._state

    def load(self, conversation_id: str) -> DialogueEmotion:
        """Restore the tone last saved for ``conversation_id``."""
        if not self.enabled or self._db is None:
            self._state = DialogueEmotion.baseline()
            return self._state
        row = self._db.query_one(
            "SELECT state FROM emotion_states WHERE conversation_id = ? ORDER BY id DESC LIMIT 1",
            (conversation_id,),
        )
        if row is None:
            self._state = DialogueEmotion.baseline()
            return self._state
        try:
            data = json.loads(row["state"])
        except (ValueError, TypeError):
            self._state = DialogueEmotion.baseline()
            return self._state
        self._state = DialogueEmotion.from_primitive(data if isinstance(data, dict) else {})
        return self._state

    def observe(self, text: str, *, conversation_id: str | None = None) -> DialogueEmotion:
        """Move the tone gradually in response to one user message.

        A sad or angry message lowers energy and raises calm and attentiveness --
        it never sets a "the user is sad" flag, and it never moves a dial more
        than :data:`MAX_STEP`. A neutral message drifts back toward the baseline.
        """
        if not self.enabled:
            return DialogueEmotion.baseline()

        deltas: dict[str, float] = {}
        matched = False
        if _contains(text, _SAD_CUES) or _contains(text, _ANGRY_CUES):
            deltas.update(energy=-1.0, calm=0.8, warmth=0.6, playfulness=-0.6, context_sensitivity=0.8)
            matched = True
        if _contains(text, _HAPPY_CUES):
            deltas.update(energy=0.8, playfulness=0.7, warmth=0.5, calm=-0.3)
            matched = True
        if _contains(text, _AFFECTION_CUES):
            deltas.update(warmth=0.9, romance=0.4)
            matched = True
        if _contains(text, _ROMANCE_CUES):
            deltas.update(romance=1.0, warmth=0.4)
            matched = True
        if _contains(text, _QUESTION_CUES):
            deltas.update(context_sensitivity=0.4, energy=0.2)

        if matched:
            self._state = self._state.nudged(deltas)
        else:
            self._state = self._state.decayed()
        if self._db is not None and conversation_id:
            self._persist(conversation_id)
        return self._state

    def _persist(self, conversation_id: str) -> None:
        assert self._db is not None
        self._db.execute(
            "INSERT INTO emotion_states(conversation_id, state, created_at) VALUES(?, ?, ?)",
            (conversation_id, json.dumps(self._state.to_primitive(), ensure_ascii=False), _utc_now()),
        )


__all__ = [
    "DEFAULT_DECAY",
    "MAX_STEP",
    "DialogueEmotion",
    "DialogueEmotionTracker",
]
