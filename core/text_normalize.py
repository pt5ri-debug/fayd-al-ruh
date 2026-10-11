"""Safe, deterministic text normalization for imported documents.

Ingested files arrive in whatever encoding and dialect their author used. Before
they are chunked and stored, the text is put through one predictable pass so two
files that mean the same thing produce the same knowledge:

* **Unicode** -- the text is normalized to NFC so visually identical strings share
  one code-point sequence.
* **Structure** -- line endings are unified, control characters (other than tab
  and newline) are dropped, and runs of trailing spaces on a line are trimmed.
  This is safe: it never removes a word.
* **Whitespace** -- spaces and tabs are collapsed; three or more newlines become
  two, so paragraph breaks survive but a runaway export does not bloat a chunk.
* **Arabic (opt-in)** -- diacritics (tashkeel), tatweel, and the common letter
  variants are unified when ``fold_arabic=True``. This is the *same* light
  normalization :func:`core.rag._normalize` applies to query tokens, so a stored
  document and a later query agree on letter forms.

Determinism matters more than aggressiveness: the pass is pure, idempotent, and
depends only on the input string, so the same bytes always normalize to the same
bytes. It is deliberately *opt-in* at the ingestion boundary (see
``IngestConfig.normalize``) because normalization changes the text and therefore
its content hash; leaving it off keeps an existing dedup ledger valid.

Nothing here executes the document: it is a pure string transformation.
"""

from __future__ import annotations

import re
import unicodedata

#: Arabic combining marks: tashkeel, superscript alef, and Quranic annotation.
_ARABIC_DIACRITICS_RE = re.compile(r"[\u0610-\u061a\u064b-\u065f\u0670\u06d6-\u06ed]")
#: Tatweel (kashida), used only for justification.
_TATWEEL = "\u0640"
#: Letter variants folded to their common form, mirroring ``core.rag``.
_ARABIC_LETTERS = str.maketrans(
    {"أ": "ا", "إ": "ا", "آ": "ا", "ى": "ي", "ة": "ه", "ؤ": "و", "ئ": "ي"}
)
_CONTROL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
_SPACES_RE = re.compile(r"[ \t\u00a0\u2000-\u200a\u202f\u205f\u3000]+")
_MANY_NEWLINES_RE = re.compile(r"\n{3,}")


def fold_arabic(text: str) -> str:
    """Unify Arabic letter forms and drop diacritics/tatweel (idempotent)."""
    value = _ARABIC_DIACRITICS_RE.sub("", text).replace(_TATWEEL, "")
    return value.translate(_ARABIC_LETTERS)


def normalize_text(text: str, *, fold_arabic_letters: bool = False) -> str:
    """Normalize ``text`` for storage; see the module docstring for the rules.

    ``fold_arabic_letters`` defaults to ``False`` so the plain pass is
    conservative (Unicode + structure only). Pass ``True`` to also unify Arabic
    letter forms, matching the retriever's token normalization.
    """
    if not text:
        return ""
    # ``\r\n`` and ``\r`` become ``\n`` before anything else so line structure is
    # stable regardless of the source platform.
    value = text.replace("\r\n", "\n").replace("\r", "\n")
    value = unicodedata.normalize("NFC", value)
    value = _CONTROL_RE.sub("", value)
    value = _SPACES_RE.sub(" ", value)
    # Trim trailing spaces per line (leading spaces are meaningful for indentation).
    value = "\n".join(line.rstrip() for line in value.split("\n"))
    value = _MANY_NEWLINES_RE.sub("\n\n", value)
    if fold_arabic_letters:
        value = fold_arabic(value)
    return value.strip()


def collapse_whitespace(text: str) -> str:
    """Collapse *all* whitespace to single spaces, exactly as ``chunk_text`` does.

    Kept here so a caller can reproduce the chunker's whitespace behavior
    without importing the knowledge base, and so the two stay in step.
    """
    return " ".join(text.split())


__all__ = ["collapse_whitespace", "fold_arabic", "normalize_text"]
