"""Natasha's **anti-repetition engine**: she must not say the same thing twice.

A small model drifts into repeating itself -- the same opener, the same compliment,
the same sentence -- and over a long conversation that is what makes her sound like
a machine. This module is the deterministic guard against it. It does not rewrite
the model's words; it *measures* the repetition and turns it into guidance for the
next turn, and it can tell a caller honestly when a reply repeated her own earlier
wording.

Four mechanisms, all dependency-free:

* **repeated-phrase detection** -- it normalises text (case, whitespace, Arabic
  diacritics, a trailing stop) and finds the n-grams the new reply shares with
  what she already said, longest first;
* **a short rolling summary of the current conversation** -- what has been said, in
  a bounded form, so the prompt can say "you have already covered this" instead of
  re-deriving it;
* **variation hints** -- a rotating list of ways to re-open and re-express, so the
  guidance itself does not become the same sentence every turn;
* **answer-length adaptation** -- a long, heavy question earns room; a short,
  light one gets a short answer, so she does not answer "صباح الخير" with a speech.

Honesty: this is a *heuristic*, and it says so. It flags a phrase it has seen, it
never claims to know what the user knows, and it never edits the model's output.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

#: The shortest n-gram (in words) worth calling a repeated phrase. Two words is
#: enough to catch a repeated compliment ("يا قلبي") without flagging a single
#: common word.
MIN_PHRASE_WORDS = 3

#: At most this many repeated phrases are reported, longest first.
MAX_PHRASES = 5

#: The default number of recent assistant turns kept for comparison.
DEFAULT_WINDOW = 12

_TOKEN_RE = re.compile(r"[\w\u0600-\u06FF]+")
_DIACRITICS_RE = re.compile(r"[\u064B-\u0652\u0670\u0640]")
_ALEF = str.maketrans({"أ": "ا", "إ": "ا", "آ": "ا", "ى": "ي", "ة": "ه", "ؤ": "و", "ئ": "ي"})
#: Sentence boundaries, so the comparison works sentence by sentence.
_SENTENCE_SPLIT = re.compile(r"[.!؟?؛;\n]+")

#: Rotating variation hints. They cycle, so the *guidance* varies too.
_VARIATION_HINTS: tuple[str, ...] = (
    "Open differently this time -- do not reuse your last opener.",
    "Say it in a fresh way; if you praised them before, find a new, specific detail.",
    "Change the shape of the sentence, not only the words in it.",
    "Lead with something new rather than restating what you already said.",
    "Let this turn be a little quieter and shorter than the last one.",
    "Answer the moment itself; do not re-introduce what is already understood.",
)


def _normalize(text: str) -> str:
    """Lower-case, strip Arabic diacritics/tatweel, unify letters, collapse spaces."""
    value = _DIACRITICS_RE.sub("", text).translate(_ALEF).casefold()
    value = re.sub(r"[^\w\u0600-\u06FF\s]", " ", value)
    return re.sub(r"\s+", " ", value).strip()


def _words(text: str) -> list[str]:
    return _TOKEN_RE.findall(_normalize(text))


def _sentences(text: str) -> list[str]:
    return [part.strip() for part in _SENTENCE_SPLIT.split(text or "") if part.strip()]


def _ngrams(words: Sequence[str], size: int) -> list[tuple[str, ...]]:
    if size < 1 or len(words) < size:
        return []
    return [tuple(words[i : i + size]) for i in range(len(words) - size + 1)]


@dataclass(frozen=True)
class RepeatedPhrase:
    """A phrase the new reply shares with something she already said."""

    phrase: str
    words: int
    occurrences: int

    def describe(self) -> str:
        return f'"{self.phrase}" (said {self.occurrences} time(s) before)'


@dataclass(frozen=True)
class RepetitionReport:
    """The honest result of checking one candidate reply against her history."""

    phrases: tuple[RepeatedPhrase, ...] = ()
    opener_repeated: bool = False
    checked_against: int = 0

    @property
    def is_repetitive(self) -> bool:
        return bool(self.phrases) or self.opener_repeated

    @property
    def score(self) -> float:
        """A rough 0..1 measure of how much was repeated (longest phrase first)."""
        if not self.phrases and not self.opener_repeated:
            return 0.0
        longest = max((p.words for p in self.phrases), default=0)
        base = min(1.0, longest / 10.0)
        if self.opener_repeated:
            base = max(base, 0.5)
        return round(base, 3)

    def describe(self) -> str:
        if not self.is_repetitive:
            return "No repeated phrasing was detected."
        parts: list[str] = []
        if self.opener_repeated:
            parts.append("it opens the same way as an earlier reply")
        for phrase in self.phrases:
            parts.append(phrase.describe())
        return "; ".join(parts) + "."

    def to_primitive(self) -> dict[str, object]:
        return {
            "phrases": [p.describe() for p in self.phrases],
            "opener_repeated": self.opener_repeated,
            "checked_against": self.checked_against,
            "score": self.score,
        }


class AntiRepetitionEngine:
    """Tracks what Natasha has said and measures a new reply against it.

    The engine keeps a bounded window of her recent replies (and, optionally, a
    rolling summary of the conversation). It never changes a reply; it reports
    repetition and produces guidance the prompt builder can include.
    """

    def __init__(self, *, window: int = DEFAULT_WINDOW, enabled: bool = True) -> None:
        if window < 1:
            raise ValueError("window must be at least 1.")
        self.window = window
        self.enabled = enabled
        self._replies: list[str] = []
        self._hint_index = 0
        self._summary_lines: list[str] = []

    # -- state ------------------------------------------------------------
    @property
    def replies(self) -> tuple[str, ...]:
        return tuple(self._replies)

    @property
    def summary(self) -> str:
        """A short rolling summary of the conversation so far."""
        return " ".join(self._summary_lines)

    def reset(self) -> None:
        self._replies.clear()
        self._summary_lines.clear()
        self._hint_index = 0

    def load_replies(self, replies: Iterable[str]) -> None:
        """Seed the window (e.g. from a resumed session)."""
        self._replies = [r for r in replies if r and r.strip()][-self.window :]

    def observe(self, reply: str) -> None:
        """Record one reply Natasha produced, trimming the window."""
        text = (reply or "").strip()
        if not text:
            return
        self._replies.append(text)
        if len(self._replies) > self.window:
            self._replies = self._replies[-self.window :]
        self._record_summary(text)

    def observe_user(self, text: str) -> None:
        """Record a user turn in the rolling summary (never compared for phrasing)."""
        self._record_summary((text or "").strip())

    def _record_summary(self, text: str) -> None:
        if not text:
            return
        first = _sentences(text)
        line = (first[0] if first else text)[:160]
        if line and (not self._summary_lines or self._summary_lines[-1] != line):
            self._summary_lines.append(line)
        if len(self._summary_lines) > self.window:
            self._summary_lines = self._summary_lines[-self.window :]

    # -- detection --------------------------------------------------------
    def check(self, candidate: str, *, known: Sequence[str] = ()) -> RepetitionReport:
        """Measure ``candidate`` against her recent replies (and any ``known`` text).

        ``known`` lets a caller also compare against what both sides already know
        (e.g. the user's own recent messages), so a reply that merely restates the
        question is flagged too.
        """
        if not self.enabled:
            return RepetitionReport()
        text = (candidate or "").strip()
        if not text:
            return RepetitionReport()
        history = [*self._replies, *[k for k in known if k and k.strip()]]
        if not history:
            return RepetitionReport()

        history_words = [_words(item) for item in history]
        candidate_words = _words(text)

        # Longest shared n-grams first, de-duplicated by phrase.
        found: list[RepeatedPhrase] = []
        seen: set[str] = set()
        max_size = min(len(candidate_words), 12)
        for size in range(max_size, MIN_PHRASE_WORDS - 1, -1):
            for gram in _ngrams(candidate_words, size):
                phrase = " ".join(gram)
                if phrase in seen:
                    continue
                occurrences = sum(1 for words in history_words if _contains_gram(words, gram))
                if occurrences:
                    seen.add(phrase)
                    found.append(RepeatedPhrase(phrase=phrase, words=size, occurrences=occurrences))
            if len(found) >= MAX_PHRASES:
                break

        # Trim overlapping shorter phrases that are inside a longer reported one.
        found = _prune_overlaps(found)[:MAX_PHRASES]
        opener_repeated = self._opener_repeated(candidate_words, history_words)
        return RepetitionReport(
            phrases=tuple(found),
            opener_repeated=opener_repeated,
            checked_against=len(history),
        )

    def _opener_repeated(self, candidate_words: Sequence[str], history_words: Sequence[Sequence[str]]) -> bool:
        if len(candidate_words) < 2:
            return False
        opener = tuple(candidate_words[:2])
        return any(tuple(words[:2]) == opener for words in history_words if len(words) >= 2)

    # -- guidance ---------------------------------------------------------
    def variation_hint(self) -> str:
        """The next rotating variation hint (cycles, so the guidance varies too)."""
        hint = _VARIATION_HINTS[self._hint_index % len(_VARIATION_HINTS)]
        self._hint_index += 1
        return hint

    def guidance(self, *, repetitiveness: float = 0.05, report: RepetitionReport | None = None) -> str:
        """The standing instruction that keeps her from repeating herself.

        A high ``repetitiveness`` trait relaxes it; the default (low) makes it
        strict. When a ``report`` found repetition, its findings are named so the
        model is told exactly what to avoid.
        """
        if not self.enabled:
            return ""
        if repetitiveness >= 0.67:
            base = (
                "You may return to a familiar refrain, but keep any repetition "
                "deliberate rather than accidental."
            )
        elif repetitiveness >= 0.34:
            base = "Vary your openers and compliments; do not reuse a whole sentence."
        else:
            base = (
                "Never reuse an opener, a compliment, or a sentence you have already "
                "used in this conversation, and do not restate what both of you "
                "already know."
            )
        parts = [base, self.variation_hint()]
        if report is not None and report.is_repetitive:
            parts.append(f"Your draft repeats earlier wording: {report.describe()}")
        return " ".join(parts)

    def length_hint(self, question: str) -> str:
        """How long the answer should be, adapted to the question at hand.

        A short, light message gets a short answer; a long or many-part question
        earns more room. The rule is stated as a ceiling, never a quota.
        """
        words = len(_words(question))
        heavy = _looks_heavy(question)
        if words <= 4 and not heavy:
            return "Keep this one short -- a sentence or two is enough."
        if words >= 40 or heavy:
            return "This deserves a full answer; take the room you need, without padding."
        return "Keep it to a few sentences unless the moment asks for more."


def _contains_gram(words: Sequence[str], gram: Sequence[str]) -> bool:
    size = len(gram)
    if size == 0 or len(words) < size:
        return False
    for i in range(len(words) - size + 1):
        if tuple(words[i : i + size]) == tuple(gram):
            return True
    return False


def _prune_overlaps(phrases: Sequence[RepeatedPhrase]) -> list[RepeatedPhrase]:
    """Drop a shorter phrase that is contained in a longer reported one."""
    kept: list[RepeatedPhrase] = []
    for phrase in sorted(phrases, key=lambda p: (-p.words, p.phrase)):
        if any(phrase.phrase in longer.phrase for longer in kept):
            continue
        kept.append(phrase)
    return kept


#: Words that mark a question as substantial even when it is short.
_HEAVY_CUES = (
    "why", "how", "explain", "tell me about", "what do you think", "advice",
    "لماذا", "كيف", "اشرح", "حدثني عن", "ما رأيك", "نصيحة", "ماذا أفعل",
)


def _looks_heavy(text: str) -> bool:
    lowered = (text or "").casefold()
    return any(cue in lowered for cue in _HEAVY_CUES)


def repeated_phrases(
    candidate: str, previous: Sequence[str], *, min_words: int = MIN_PHRASE_WORDS
) -> list[RepeatedPhrase]:
    """The phrases ``candidate`` shares with ``previous`` (a convenience wrapper)."""
    engine = AntiRepetitionEngine()
    engine.load_replies(previous)
    report = engine.check(candidate)
    return [phrase for phrase in report.phrases if phrase.words >= min_words]


__all__ = [
    "DEFAULT_WINDOW",
    "MAX_PHRASES",
    "MIN_PHRASE_WORDS",
    "AntiRepetitionEngine",
    "RepeatedPhrase",
    "RepetitionReport",
    "repeated_phrases",
]
