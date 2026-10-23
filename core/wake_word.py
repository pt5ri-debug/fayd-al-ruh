"""Wake-word gating for the always-listening Jarvis mode.

Jarvis listens all the time, but it must not *answer* everything it hears. This
module adds the small gate between "audio arrived" and "the assistant wakes up":

* :class:`WakeWordDetector` matches a phrase against an utterance;
* :class:`AudioSpeechGate` decides, from honest signal features (duration and
  loudness), whether a clip even contains speech worth transcribing.

Both are deliberately dependency-free and text-first, so the whole gate is
testable without a microphone or a speech engine. The detector accepts Arabic
and English phrases, since the user may address the assistant in either language.
"""

from __future__ import annotations

from dataclasses import dataclass

DEFAULT_WAKE_WORDS = (
    "jarvis",
    "hey jarvis",
    "ok jarvis",
    "يا جارفيس",
    "مرحبا جارفيس",
)


def _normalize(text: str) -> str:
    """Casefold and strip punctuation so 'Jarvis!' matches 'jarvis'."""
    return "".join(character if character.isalnum() or character.isspace() else " " for character in text).casefold()


@dataclass(frozen=True)
class WakeWordDetector:
    """Matches an utterance against a set of wake phrases."""

    phrases: tuple[str, ...] = DEFAULT_WAKE_WORDS

    def __post_init__(self) -> None:
        if not self.phrases:
            raise ValueError("A wake-word detector needs at least one phrase.")

    @property
    def normalized_phrases(self) -> tuple[str, ...]:
        return tuple(_normalize(phrase) for phrase in self.phrases)

    def detect(self, utterance: str) -> WakeWordMatch:
        """Return the match for ``utterance`` (``woke`` is False when it does not).

        Longer phrases win, so "hey jarvis" is consumed whole rather than leaving
        "hey" behind when the bare word "jarvis" would match first.
        """
        normalized = _normalize(utterance)
        padded = f" {normalized} "
        ordered = sorted(zip(self.phrases, self.normalized_phrases, strict=False), key=lambda pair: len(pair[1]), reverse=True)
        for phrase, normalized_phrase in ordered:
            if f" {normalized_phrase} " in padded:
                remainder = padded.replace(f" {normalized_phrase} ", " ", 1).strip()
                return WakeWordMatch(woke=True, phrase=phrase, remainder=remainder)
        return WakeWordMatch(woke=False, phrase=None, remainder=utterance.strip())


@dataclass(frozen=True)
class WakeWordMatch:
    """The outcome of a wake-word check."""

    woke: bool
    phrase: str | None
    remainder: str


@dataclass(frozen=True)
class AudioSpeechGate:
    """Decides whether a clip is worth sending to a recognizer.

    The gate is honest about what it can measure: a clip is treated as speech
    when it lasts long enough and carries enough energy. It never pretends to
    understand content -- that is the recognizer's job.
    """

    min_duration_seconds: float = 0.2
    min_peak: int = 64

    def has_speech(self, clip) -> bool:
        if clip.duration_seconds < self.min_duration_seconds:
            return False
        peak = max((abs(sample) for sample in clip.samples), default=0)
        return peak >= self.min_peak


__all__ = [
    "DEFAULT_WAKE_WORDS",
    "AudioSpeechGate",
    "WakeWordDetector",
    "WakeWordMatch",
]
