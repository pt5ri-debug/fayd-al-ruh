"""Content normalization: one deterministic pass over perceived text (section 3).

Normalization is applied at the perception boundary because it changes the text
and hence its content hash, so it is a deliberate, opt-in step rather than a
hidden one. This module reuses the project's own passes rather than inventing new
ones:

* :func:`core.text_normalize.normalize_text` does the Unicode NFC, line-ending,
  control-character, and whitespace work (never touching punctuation or Arabic
  letter distinctions);
* :func:`core.voices.detect_language` is the one Arabic-script check, reused so
  there is no second language guess.

Two things are added on top, both about *reporting* rather than rewriting:

* :func:`normalize_content` returns the text plus a :class:`NormalizationReport`
  naming exactly what changed (collapsed whitespace, removed control characters,
  the detected language), so an import can record *how* a file was read;
* :func:`truncate` caps a text at a character budget on a word boundary and says
  so, so a huge document is bounded without silently cutting a sentence in half.

Nothing here reorders, translates, or summarises content: a normalization that
changed meaning would be a silent lie, which the project forbids.
"""

from __future__ import annotations

import unicodedata
from dataclasses import dataclass, field

from core.text_normalize import normalize_text
from core.voices import detect_language


@dataclass(frozen=True)
class NormalizationReport:
    """What a normalization pass changed, so it can be recorded honestly."""

    changed: bool
    language: str
    original_chars: int
    normalized_chars: int
    folded_arabic: bool = False
    truncated: bool = False
    notes: tuple[str, ...] = field(default_factory=tuple)

    def describe(self) -> str:
        bits = [f"language {self.language}"]
        bits.append(f"{self.original_chars} -> {self.normalized_chars} chars")
        if self.changed:
            bits.append("normalised")
        if self.folded_arabic:
            bits.append("Arabic letter forms unified")
        if self.truncated:
            bits.append("truncated to the character budget")
        return "; ".join(bits)


def _has_control_characters(text: str) -> bool:
    return any(
        unicodedata.category(ch) == "Cc" and ch not in "\t\n\r" for ch in text
    )


def normalize_content(
    text: str,
    *,
    fold_arabic: bool = False,
    language: str | None = None,
) -> tuple[str, NormalizationReport]:
    """Normalize ``text`` and report what changed.

    ``fold_arabic=True`` additionally unifies Arabic letter forms (matching the
    ingestion flag); it is off by default because it changes the text and can lose
    a distinction a query relies on. ``language`` pins the reported language when
    the caller already knows it (a transcript's audio language), else it is
    detected from the text.
    """
    original_chars = len(text)
    had_control = _has_control_characters(text)
    collapsed = " ".join(text.split()) != text
    normalized = normalize_text(text, fold_arabic_letters=fold_arabic)
    resolved_language = language or detect_language(normalized)
    notes: list[str] = []
    if had_control:
        notes.append("control characters removed")
    if collapsed:
        notes.append("whitespace collapsed")
    return normalized, NormalizationReport(
        changed=normalized != text,
        language=resolved_language,
        original_chars=original_chars,
        normalized_chars=len(normalized),
        folded_arabic=fold_arabic,
        notes=tuple(notes),
    )


def truncate(text: str, *, max_chars: int) -> tuple[str, bool]:
    """Cap ``text`` at ``max_chars`` on a word boundary; returns (text, truncated).

    A cut is made at the last whitespace before the budget so a word is not split,
    and the boolean says whether anything was dropped -- so a caller can record a
    capped read rather than presenting it as the whole document.
    """
    if max_chars < 1:
        raise ValueError("max_chars must be at least 1.")
    if len(text) <= max_chars:
        return text, False
    window = text[:max_chars]
    cut = window.rfind(" ")
    if cut <= 0:
        cut = max_chars
    return window[:cut].rstrip(), True


__all__ = [
    "NormalizationReport",
    "normalize_content",
    "truncate",
]
