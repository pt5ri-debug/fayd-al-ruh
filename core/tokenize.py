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
from functools import lru_cache

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

#: How many distinct tokens :func:`normalize_token` remembers. The token vocabulary
#: of a session is small and highly repetitive (every retrieval re-normalises the
#: same document tokens, and the corpus words recur across documents), so a modest
#: bound covers a whole session's working set while keeping memory flat.
_TOKEN_CACHE_SIZE = 20_000


def normalize_token(token: str) -> str:
    """Light Arabic normalisation: strip diacritics, unify letters, drop affixes.

    A *multi-letter* prefix (``ال``, ``وال``, ...) may be dropped once at least
    three letters remain. A single-letter prefix is only dropped when at least
    four remain, because ``ك`` and ``ب`` are real prefixes but also the first
    letter of countless ordinary words -- without the longer floor ``كتاب``
    would be reduced to ``تاب``.

    The function is pure and its inputs repeat heavily (a retrieval normalises the
    same document tokens on every query, and the same corpus words recur across
    documents), so the result is memoised. The bound keeps a long session with
    unusual input from growing without limit; the transform itself is a handful of
    string operations, so a cache miss stays cheap.
    """
    return _normalize_token_cached(token)


@lru_cache(maxsize=_TOKEN_CACHE_SIZE)
def _normalize_token_cached(token: str) -> str:
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


def normalize_phrase(text: str) -> str:
    """Normalise every token of a phrase, keeping the spaces between them.

    This is the comparison form for a *concept term*: ``"well log"`` stays
    ``"well log"`` and ``"المسامية"`` becomes ``"مساميه"``, so a term and a text
    are compared in the same normalised space. Used by the ontology lookup and
    the term matchers, which must agree on what a word is or an Arabic alias
    would never resolve.
    """
    return " ".join(normalize_token(part) for part in _TOKEN_RE.findall(text))


def token_spans(text: str) -> list[tuple[int, int, str]]:
    """Each token's ``(start, end, normalised)`` in ``text``, in order.

    The span matcher needs both *where* a token is and its normalised form, so a
    multi-word term can be matched across a run of whole tokens. Matching on
    tokens (not a regex on the raw text) is what makes it work for Arabic, where
    ``\\b`` misbehaves next to the script's non-word marks.
    """
    return [
        (match.start(), match.end(), normalize_token(match.group()))
        for match in _TOKEN_RE.finditer(text)
    ]


def token_sequence(text: str) -> list[str]:
    """The normalised tokens of ``text`` in order, for a co-occurrence window."""
    return [normalize_token(token) for token in _TOKEN_RE.findall(text)]


__all__ = [
    "content_terms",
    "normalize_phrase",
    "normalize_token",
    "token_sequence",
    "token_spans",
    "tokenize",
]
