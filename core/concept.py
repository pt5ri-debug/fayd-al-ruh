from __future__ import annotations

from dataclasses import dataclass, field
from uuid import uuid4

from core.tokenize import normalize_phrase


def is_arabic(text: str) -> bool:
    """Whether ``text`` contains Arabic-script letters.

    The one script test the concept layer uses, so an Arabic alias and an English
    label are told apart the same way everywhere (``core.voices.detect_language``
    is the speech-layer sibling).
    """
    return any("\u0600" <= char <= "\u06ff" for char in text)


@dataclass(frozen=True)
class Concept:
    label: str
    aliases: frozenset[str] = field(default_factory=frozenset)
    id: str = field(default_factory=lambda: f"CON-{uuid4().hex[:8]}")

    def __post_init__(self) -> None:
        if not self.label.strip():
            raise ValueError("Concept label cannot be empty.")

    @property
    def terms(self) -> frozenset[str]:
        return frozenset({self.label.casefold(), *(alias.casefold() for alias in self.aliases)})

    @property
    def normalized_terms(self) -> frozenset[str]:
        """The terms in the shared normalised space (``core.tokenize``).

        A term and a text are compared through this, so ``المسامية`` (with the
        definite article) matches a stored ``مسامية`` alias, and an English term
        matches across case. Without it the Arabic half of every alias is dead,
        because a bare ``casefold`` never strips the article.
        """
        return frozenset(normalize_phrase(term) for term in self.terms)

    def terms_for(self, language: str) -> frozenset[str]:
        """The terms written in ``language`` (``"ar"`` or ``"en"``).

        ``"en"`` keeps the terms with no Arabic letters (the label and any Latin
        alias); ``"ar"`` keeps only the Arabic ones. Used by the bilingual
        lookup, so an Arabic question and an English one each reach the concept
        through the words the user actually typed.
        """
        if language == "ar":
            return frozenset(term for term in self.terms if is_arabic(term))
        return frozenset(term for term in self.terms if not is_arabic(term))
