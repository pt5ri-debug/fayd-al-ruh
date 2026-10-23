"""The project's one Arabic-aware tokenizer.

Every text signal in the project must agree on what a "word" is, or a stored
document and a later query would normalise to different tokens. This module is
that single rule: light Arabic normalisation (diacritics and tatweel removed,
letter variants unified, common prefixes stripped) plus a content-term filter
that drops the bilingual stopword list.

It lives on its own so both :mod:`core.rag` (retrieval) and
:mod:`core.embeddings` (the learned semantic space) share it without importing
each other. :mod:`core.rag` re-exports these names, because several modules
import ``_query_terms`` from there.
"""

from __future__ import annotations

import re

_TOKEN_RE = re.compile(r"\w+", flags=re.UNICODE)
_DIACRITICS_RE = re.compile(r"[\u0610-\u061a\u064b-\u065f\u0670\u06d6-\u06ed]")
_STOPWORDS = frozenset(
    {
        # English (primary)
        "a", "an", "the", "what", "how", "is", "are", "of", "to", "in", "on",
        "for", "and", "or", "with", "between", "does", "do", "why", "when",
        "which", "that", "this", "these", "those", "it", "its", "as", "by",
        "from", "at", "be", "can", "will", "would", "should", "there", "their",
        "about", "into", "than", "then", "not", "no", "yes", "if", "so", "such",
        # Arabic (retained so bilingual aliases still normalise cleanly)
        "ما", "ماذا", "كيف", "هل", "من", "في", "على", "إلى", "عن", "هو", "هي",
        "التي", "الذي", "و", "أو", "علاقة", "بين", "مع", "ثم", "قد", "لا", "إن",
        "هذا", "هذه", "ذلك", "تلك", "أي", "كل", "بعض",
    }
)
_ALEF = str.maketrans({"أ": "ا", "إ": "ا", "آ": "ا", "ى": "ي", "ة": "ه", "ؤ": "و", "ئ": "ي"})
_PREFIXES = ("وال", "بال", "فال", "كال", "ال", "و", "ب", "ل", "ك", "ف")


def normalize_token(token: str) -> str:
    """Light Arabic normalisation: strip diacritics, unify letters, drop affixes.

    A *multi-letter* prefix (``ال``, ``وال``, ...) may be dropped once at least
    three letters remain. A single-letter prefix is only dropped when at least
    four remain, because ``ك`` and ``ب`` are real prefixes but also the first
    letter of countless ordinary words -- without the longer floor ``كتاب``
    would be reduced to ``تاب``.
    """
    value = _DIACRITICS_RE.sub("", token).replace("ـ", "").translate(_ALEF).casefold()
    changed = True
    while changed and len(value) > 3:
        changed = False
        for prefix in _PREFIXES:
            floor = 4 if len(prefix) == 1 else 3
            if value.startswith(prefix) and len(value) - len(prefix) >= floor:
                value = value[len(prefix):]
                changed = True
                break
    return value


def tokenize(text: str) -> set[str]:
    """Every normalised token in ``text`` (stopwords kept)."""
    return {normalize_token(token) for token in _TOKEN_RE.findall(text)}


def content_terms(text: str) -> set[str]:
    """The content terms of ``text``: normalised tokens minus the stopwords."""
    return tokenize(text) - _STOPWORDS


def token_sequence(text: str) -> list[str]:
    """The normalised tokens of ``text`` in order, for a co-occurrence window."""
    return [normalize_token(token) for token in _TOKEN_RE.findall(text)]


__all__ = [
    "content_terms",
    "normalize_token",
    "token_sequence",
    "tokenize",
]
