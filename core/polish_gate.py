"""The held-out gate for the universal linguistic polish and the Arabic vocalizer.

The project answers through seven characters and its own model, and every reply now
passes through one shared polisher (:mod:`core.text_polish`) that can also fully
vocalize an Arabic reply (tashkeel). This module is the gate that keeps that
behaviour honest: a catalogue of hand-written judgements, run against the **real**
entry points, each with its own negative control, so a regression that vocalized
English, corrupted a formula, changed a word, or ran a *second* pass that differed
from the first fails the evaluation instead of passing silently.

Every judgement is measured, not asserted: the checks call
:func:`core.text_polish.polish_answer` (and, for the wiring judgement, a real
:class:`core.jarvis.JarvisMode`) and read the result. The share of judgements the
shipped behaviour satisfies is the metric ``polish_accuracy``.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from core.arabic_tashkeel import MODE_COLLOQUIAL, MODE_FORMAL, strip_diacritics
from core.text_polish import TASHKEEL_FORMAL, polish_answer

#: Arabic vowel marks (the harakat) -- what "vocalized" means.
_HARAKAT = "\u064b\u064c\u064d\u064e\u064f\u0650\u0651\u0652\u0670\u0640"

_ARABIC_LETTER = re.compile(r"[\u0600-\u06ff]")


def _has_vowel_marks(text: str) -> bool:
    return any(char in _HARAKAT for char in text)


def _words(text: str) -> list[str]:
    stripped = [word.strip(".,!?؛،:").casefold() for word in text.split()]
    return [word for word in stripped if word]


@dataclass(frozen=True)
class PolishJudgement:
    """One held-out expectation about the polish, and the way it is checked."""

    name: str
    check: str
    text: str
    language: str
    tashkeel: bool
    expected: bool
    mode: str = TASHKEEL_FORMAL
    note: str = ""


#: The held-out catalogue. Each ``check`` is one of the named rules below, and each
#: judgement states what the *shipped* behaviour must do; several are negative
#: controls (a case that must **not** change).
POLISH_JUDGEMENTS: tuple[PolishJudgement, ...] = (
    PolishJudgement(
        "Arabic answer is vocalized when asked",
        "vocalized",
        "المسامية هي نسبة الفراغات في الصخر.",
        "ar",
        True,
        True,
        note="tashkeel on, an Arabic reply",
    ),
    PolishJudgement(
        "Arabic answer is left bare by default",
        "vocalized",
        "المسامية هي نسبة الفراغات في الصخر.",
        "ar",
        False,
        False,
        note="tashkeel off, the project's long-standing output (negative control)",
    ),
    PolishJudgement(
        "English reply is never vocalized",
        "vocalized",
        "The porosity is the fraction of void space.",
        "en",
        True,
        False,
        note="tashkeel on, but an English reply (negative control)",
    ),
    PolishJudgement(
        "A Latin identifier survives untouched",
        "preserves",
        "The unit is SPE and the value is 3.14 within 10--20.",
        "en",
        True,
        True,
        note="a formula, a decimal, and a hyphenated range keep their form",
    ),
    PolishJudgement(
        "Arabic typography is repaired",
        "punctuation",
        "المسامية في الصخر , وهي مهمة ؟",
        "ar",
        False,
        True,
        note="a space before punctuation is closed and ASCII marks become Arabic",
    ),
    PolishJudgement(
        "English typography is repaired",
        "punctuation",
        "hello there ,how are you ?  i am fine .",
        "en",
        False,
        True,
        note="spacing, capitalisation, and the standalone pronoun",
    ),
    PolishJudgement(
        "The polish is idempotent (Arabic)",
        "idempotent",
        "المسامية في الصخر , وهي مهمة ؟",
        "ar",
        True,
        True,
        note="a second pass must change nothing",
    ),
    PolishJudgement(
        "The polish is idempotent (English)",
        "idempotent",
        "hello there ,how are you ?",
        "en",
        False,
        True,
        note="a second pass must change nothing",
    ),
    PolishJudgement(
        "The polish never adds or removes a word",
        "preserves_words",
        "the reservoir pressure is 3000 psi , and the porosity is 0.2",
        "en",
        False,
        True,
        note="presentation only",
    ),
    PolishJudgement(
        "The colloquial mode drops the case endings",
        "colloquial",
        "ضغط المكمن",
        "ar",
        True,
        True,
        mode=MODE_COLLOQUIAL,
        note="the spoken form keeps the lexical vowels and drops the إعراب",
    ),
    PolishJudgement(
        "An already-vocalized Arabic reply is not double-marked",
        "idempotent",
        "الْمَسَامِيَّةُ فِي الصَّخْرِ.",
        "ar",
        True,
        True,
        note="vocalizing a vocalized reply is a no-op (negative control)",
    ),
    PolishJudgement(
        "Every answer is polished on the real Jarvis path",
        "wiring",
        "hello there ,how are you ?",
        "en",
        False,
        True,
        note="the connected path, not the polisher in isolation",
    ),
    PolishJudgement(
        "A clean sentence is left exactly as it is",
        "punctuation",
        "The porosity is the fraction of void space.",
        "en",
        False,
        False,
        note="nothing to repair, so the polish must change nothing (negative control)",
    ),
)


def _vocalized(judgement: PolishJudgement) -> bool:
    report = polish_answer(
        judgement.text,
        language=judgement.language,
        tashkeel=judgement.tashkeel,
        tashkeel_mode=judgement.mode,
    )
    return _has_vowel_marks(report.text)


def _preserves(judgement: PolishJudgement) -> bool:
    report = polish_answer(
        judgement.text,
        language=judgement.language,
        tashkeel=judgement.tashkeel,
        tashkeel_mode=judgement.mode,
    )
    for token in ("SPE", "3.14", "10--20"):
        if token not in report.text:
            return False
    return True


def _punctuation(judgement: PolishJudgement) -> bool:
    report = polish_answer(
        judgement.text,
        language=judgement.language,
        tashkeel=judgement.tashkeel,
        tashkeel_mode=judgement.mode,
    )
    # Nothing changed means the typography was not repaired.
    return report.text != judgement.text and " ," not in report.text and " ?" not in report.text


def _idempotent(judgement: PolishJudgement) -> bool:
    once = polish_answer(
        judgement.text,
        language=judgement.language,
        tashkeel=judgement.tashkeel,
        tashkeel_mode=judgement.mode,
    )
    twice = polish_answer(
        once.text,
        language=once.language,
        tashkeel=judgement.tashkeel,
        tashkeel_mode=judgement.mode,
    )
    return twice.text == once.text


def _preserves_words(judgement: PolishJudgement) -> bool:
    report = polish_answer(
        judgement.text,
        language=judgement.language,
        tashkeel=judgement.tashkeel,
        tashkeel_mode=judgement.mode,
    )
    return _words(report.text) == _words(judgement.text)


def _colloquial(judgement: PolishJudgement) -> bool:
    formal = polish_answer(
        judgement.text, language="ar", tashkeel=True, tashkeel_mode=MODE_FORMAL
    ).text
    spoken = polish_answer(
        judgement.text, language="ar", tashkeel=True, tashkeel_mode=MODE_COLLOQUIAL
    ).text
    # The spoken form is vocalized too, but it drops the final case endings, so the
    # two forms differ and both keep the bare letters.
    return (
        _has_vowel_marks(spoken)
        and spoken != formal
        and strip_diacritics(spoken) == strip_diacritics(formal)
    )


def polish_accuracy(
    judgements: tuple[PolishJudgement, ...] = POLISH_JUDGEMENTS,
) -> float:
    """The share of held-out judgements the shipped polish satisfies.

    A judgement counts only when the measured behaviour matches ``expected``
    exactly, so vocalizing an English reply, corrupting a formula, dropping a word,
    or making a second pass differ from the first fails as surely as failing to
    vocalize an Arabic one. Returns ``0.0`` for an empty catalogue.
    """
    catalogue = tuple(judgements)
    if not catalogue:
        return 0.0
    correct = 0
    for judgement in catalogue:
        check = _CHECKS.get(judgement.check)
        if check is None:
            continue
        if bool(check(judgement)) == judgement.expected:
            correct += 1
    return correct / len(catalogue)


class _StubAssistant:
    """A minimal assistant returning a fixed answer, for the wiring check."""

    def __init__(self, text: str) -> None:
        self._text = text

    def ask(self, question: str, **kwargs):
        from types import SimpleNamespace

        return SimpleNamespace(text=self._text, citations=(), confidence=0.0)

    def describe(self) -> str:
        return "the stub mouth"


def answer_is_polished(question: str, answer_text: str) -> str:
    """Run one answer through a **real** ``JarvisMode`` and return its response.

    The gate uses this so it measures the *connected* path -- the wiring in
    ``core.jarvis`` that every chat surface goes through -- and not only the
    polisher in isolation. A regression that removed the wiring call would leave
    the answer exactly as the mouth emitted it.
    """
    import tempfile

    from core.command_router import NaturalLanguageRouter
    from core.jarvis import JarvisMode
    from core.tools import build_sandbox

    with tempfile.TemporaryDirectory() as root:
        mode = JarvisMode(
            _StubAssistant(answer_text),
            router=NaturalLanguageRouter(build_sandbox(root)),
            polish=True,
        )
        return mode.handle(question, require_wake=False, speak=False).response


def _wiring(judgement: PolishJudgement) -> bool:
    """The response from a real mode is the polished form, not the raw mouth text."""
    response = answer_is_polished(judgement.text, judgement.text)
    if response == judgement.text:
        return False
    # The raw mouth text's scars (a space before punctuation) are gone.
    return " ," not in response and " ?" not in response


_CHECKS = {
    "vocalized": _vocalized,
    "preserves": _preserves,
    "punctuation": _punctuation,
    "idempotent": _idempotent,
    "preserves_words": _preserves_words,
    "colloquial": _colloquial,
    "wiring": _wiring,
}


__all__ = [
    "POLISH_JUDGEMENTS",
    "PolishJudgement",
    "answer_is_polished",
    "polish_accuracy",
]
