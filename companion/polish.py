"""Natasha's linguistic polisher: make a reply read cleanly, in either language.

A small model leaves small typographic scars -- a space before a full stop, a
missing space after a comma, a lowercase sentence opening, doubled punctuation, an
Arabic sentence punctuated with ASCII ``?`` and ``,``. This module repairs exactly
those, and **nothing else**.

What it is:

* **deterministic** -- plain rules, no model, no network, no dependency;
* **presentation only** -- it never adds, removes, or reorders a *word*, and it
  never touches a number, a formula, or a hyphenated range (``3.14``, ``10--20``,
  ``q / A`` all survive untouched). It changes only spacing, punctuation marks, and
  the capitalisation of a sentence's first letter;
* **idempotent** -- polishing an already-polished reply changes nothing, which a
  test asserts;
* **explainable** -- :func:`polish_report` returns the same text *plus* the list of
  changes it made, so a caller can show what was touched instead of guessing.

Two languages, two rule sets: :func:`polish_arabic` and :func:`polish_english`.
:func:`polish` picks one from the text itself (or takes the language the caller
already decided), so the agent never polishes Arabic with English rules.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from companion.language import ARABIC, ENGLISH, detect_language_of

#: Characters that must never be preceded by a space.
_NO_SPACE_BEFORE = ",.!?;:\u060c\u061b\u061f"

#: Closing brackets that a space must not be inserted before.
_CLOSERS = ")]}\u00bb\u201d\u2019"

#: Arabic sentence-final / separator marks, for the collapse rule.
_ARABIC_MARKS = "\u060c\u061b\u061f"

_RUNS_OF_SPACE = re.compile(r"[ \t]{2,}")
_SPACE_BEFORE = re.compile(r"[ \t]+([" + re.escape(_NO_SPACE_BEFORE) + r"])")
_SPACE_BEFORE_CLOSER = re.compile(r"[ \t]+([" + re.escape(_CLOSERS) + r"])")
_SPACE_AFTER_OPENER = re.compile(r"([(\[\u00ab\u201c])[ \t]+")
#: A run of two or more of the *same* separator collapses to one. A run of dots
#: is left alone, because ``...`` is a deliberate ellipsis.
_REPEATED_MARK = re.compile(r"([!?\u061f\u060c\u061b,;:])\1+")
#: An English standalone "i" (the pronoun) is capitalised.
_LONE_I = re.compile(r"\bi\b")
#: An Arabic letter, for deciding whether ASCII punctuation sits in Arabic text.
_ARABIC_LETTER = re.compile(r"[\u0600-\u06ff]")
#: ASCII punctuation an Arabic sentence should carry in its own form.
_ARABIC_MARKS_MAP = {",": "\u060c", ";": "\u061b", "?": "\u061f"}


@dataclass(frozen=True)
class PolishResult:
    """The polished text, the language used, and the changes that were made."""

    text: str
    language: str
    changes: tuple[str, ...] = field(default_factory=tuple)

    @property
    def changed(self) -> bool:
        return bool(self.changes)


def _collapse_spaces(text: str) -> str:
    """Collapse runs of spaces/tabs, but never a newline or a line indent."""
    lines = []
    for line in text.split("\n"):
        stripped = line.strip(" \t")
        indent = line[: len(line) - len(line.lstrip(" \t"))]
        lines.append(indent + _RUNS_OF_SPACE.sub(" ", stripped))
    return "\n".join(lines)


#: An abbreviation whose dots are part of the token: ``U.S.A``, ``e.g.``, ``i.e.``,
#: ``Ph.D``. These are shielded before *any* rule runs and restored at the end, so
#: no rule can split one into ``U. S.A`` or capitalise its inner letter.
_ABBREVIATION = re.compile(r"\b(?:[A-Za-z]\.){2,}")

#: The sentinel that stands in for a shielded token. ``\x00`` never occurs in
#: model text, so it cannot collide with real content.
_SENTINEL = "\x00{}\x00"


class _Shield:
    """Hides abbreviation tokens from every rule, then puts them back verbatim.

    Spacing, sentence capitalisation, and the lone-``i`` rule all treat a dot as a
    boundary; an abbreviation's internal dots are not boundaries, so the whole token
    is swapped for an opaque sentinel first and restored once every rule has run.
    """

    def __init__(self, text: str) -> None:
        self._tokens: list[str] = []
        self.text = _ABBREVIATION.sub(self._stash, text)

    def _stash(self, match: re.Match[str]) -> str:
        self._tokens.append(match.group(0))
        return _SENTINEL.format(len(self._tokens) - 1)

    def restore(self, text: str) -> str:
        for position, token in enumerate(self._tokens):
            text = text.replace(_SENTINEL.format(position), token)
        return text


def _space_after_separators(text: str) -> str:
    """Put exactly one space after a separator when a word follows it.

    Guards keep the model's own numbers intact: a decimal (``3.14``) and a digit
    group are left as they are. Abbreviations are already shielded by the caller.
    """
    out: list[str] = []
    length = len(text)
    for index, char in enumerate(text):
        out.append(char)
        if char not in _NO_SPACE_BEFORE:
            continue
        following = text[index + 1] if index + 1 < length else ""
        if not following or following.isspace() or following in _CLOSERS:
            continue
        previous = text[index - 1] if index > 0 else ""
        if char == "." and previous.isdigit() and following.isdigit():
            continue  # a decimal number, e.g. 3.14
        out.append(" ")
    return "".join(out)


def _to_arabic_marks(text: str) -> str:
    """Replace ASCII ``, ; ?`` with the Arabic ``، ؛ ؟`` inside Arabic text.

    A mark is converted when an Arabic letter sits on either side of it, skipping
    any spaces -- so ``3.14, وأعتقد`` converts the comma (the next word is Arabic)
    while ``3,14`` keeps it (a digit is on both sides, so it is a number).
    """
    out: list[str] = []
    length = len(text)

    def _letter_around(index: int) -> str:
        before = index - 1
        while before >= 0 and text[before].isspace():
            before -= 1
        after = index + 1
        while after < length and text[after].isspace():
            after += 1
        return (text[before] if before >= 0 else "") + (text[after] if after < length else "")

    for index, char in enumerate(text):
        replacement = _ARABIC_MARKS_MAP.get(char)
        if replacement is None:
            out.append(char)
            continue
        if _ARABIC_LETTER.search(_letter_around(index)):
            out.append(replacement)
        else:
            out.append(char)
    return "".join(out)


def _ensure_terminal(text: str) -> tuple[str, bool]:
    """Give the reply a closing mark when it ends on a bare word."""
    stripped = text.rstrip()
    if not stripped:
        return text, False
    if stripped[-1] in _NO_SPACE_BEFORE or stripped[-1] in _CLOSERS:
        return text, False
    if not stripped[-1].isalnum():
        return text, False
    return stripped + ".", True


def _capitalise_sentences(text: str) -> tuple[str, bool]:
    """Capitalise the first letter of each English sentence and the pronoun 'I'.

    Operates on already-shielded text: a shielded token counts as an ordinary word,
    so the sentence-start flag is consumed by it and a following lowercase word is
    not wrongly capitalised (``Use e.g. this`` keeps ``this`` lower-case).
    """
    changed = False
    capitalised = _LONE_I.sub("I", text)
    if capitalised != text:
        changed = True
    out: list[str] = []
    start_of_sentence = True
    for char in capitalised:
        if start_of_sentence and char.isalpha():
            upper = char.upper()
            if upper != char:
                changed = True
            out.append(upper)
            start_of_sentence = False
            continue
        out.append(char)
        if char in ".!?\n":
            start_of_sentence = True
        elif not char.isspace():
            start_of_sentence = False
    return "".join(out), changed


def _spacing_cleanup(text: str) -> tuple[str, list[str]]:
    """The spacing repairs both languages share, on already-shielded text."""
    changes: list[str] = []
    cleaned = text.strip()
    if cleaned != text:
        changes.append("trimmed surrounding whitespace")
    collapsed = _collapse_spaces(cleaned)
    if collapsed != cleaned:
        changes.append("collapsed repeated spaces")
    if _SPACE_BEFORE.search(collapsed):
        changes.append("removed a space before punctuation")
    cleaned = _SPACE_BEFORE.sub(r"\1", collapsed)
    cleaned = _SPACE_BEFORE_CLOSER.sub(r"\1", cleaned)
    cleaned = _SPACE_AFTER_OPENER.sub(r"\1", cleaned)
    if _REPEATED_MARK.search(cleaned):
        changes.append("collapsed repeated punctuation")
    cleaned = _REPEATED_MARK.sub(r"\1", cleaned)
    spaced = _space_after_separators(cleaned)
    if spaced != cleaned:
        changes.append("added a space after punctuation")
    return spaced, changes


def polish_english(text: str) -> PolishResult:
    """Polish an English reply: spacing, punctuation, and capitalisation."""
    shield = _Shield(text)
    cleaned, changes = _spacing_cleanup(shield.text)
    capitalised, did_capitalise = _capitalise_sentences(cleaned)
    if did_capitalise:
        changes.append("capitalised sentence openings")
    finished, added = _ensure_terminal(capitalised)
    if added:
        changes.append("closed the final sentence")
    return PolishResult(
        text=shield.restore(finished), language=ENGLISH, changes=tuple(changes)
    )


def polish_arabic(text: str) -> PolishResult:
    """Polish an Arabic reply: spacing, punctuation, and Arabic punctuation marks."""
    shield = _Shield(text)
    cleaned, changes = _spacing_cleanup(shield.text)
    if _ARABIC_LETTER.search(cleaned) and any(c in _ARABIC_MARKS_MAP for c in cleaned):
        marked = _to_arabic_marks(cleaned)
        if marked != cleaned:
            changes.append("used Arabic punctuation marks")
        cleaned = marked
    finished, added = _ensure_terminal(cleaned)
    if added:
        changes.append("closed the final sentence")
    return PolishResult(
        text=shield.restore(finished), language=ARABIC, changes=tuple(changes)
    )


def polish_report(text: str, *, language: str | None = None) -> PolishResult:
    """Polish ``text`` and report the language and every change that was made.

    ``language`` may be supplied when the caller has already decided it (the agent
    does, from the user's own message); otherwise it is detected from the text.
    """
    if not text or not text.strip():
        return PolishResult(text=text, language=language or ARABIC)
    chosen = language or detect_language_of(text, default=ARABIC)
    if chosen == ENGLISH:
        return polish_english(text)
    return polish_arabic(text)


def polish(text: str, *, language: str | None = None) -> str:
    """The polished text alone, for callers that do not need the report."""
    return polish_report(text, language=language).text


__all__ = [
    "PolishResult",
    "polish",
    "polish_arabic",
    "polish_english",
    "polish_report",
]
