"""Which language Natasha replies in, decided from the user's own words.

A bilingual companion must answer in the language it was asked in, and switch the
moment the user switches -- without the user having to change a setting. This
module owns that one decision, so the prompt builder, the agent, and the CLI all
agree about it.

The rule is deliberately simple and explainable:

* the language of the **newest user message** decides the reply language;
* when that message carries no letters at all (a number, an emoji, "?"), the
  previous user turn is consulted, so a bare follow-up does not flip the language;
* when nothing can be decided, the user's configured ``language`` setting is the
  fallback.

Detection itself reuses :func:`core.voices.detect_language`, the project's one
dependency-free Arabic-script check, so there is no second implementation.
"""

from __future__ import annotations

from collections.abc import Iterable

from core.voices import detect_language

#: The two languages the companion speaks, matching ``personality.LANGUAGES``.
ARABIC = "ar"
ENGLISH = "en"

#: The label shown in the prompt for each language.
LANGUAGE_LABELS = {ARABIC: "Arabic", ENGLISH: "English"}


def _has_letters(text: str) -> bool:
    """True when ``text`` carries any letter (Latin or Arabic)."""
    return any(character.isalpha() for character in text)


def detect_language_of(text: str, *, default: str = ENGLISH) -> str:
    """The language of one message, with an explicit fallback.

    Wraps :func:`core.voices.detect_language` so the companion never depends on
    that helper's own default, which is chosen for speech playback, not for a
    reply.
    """
    return detect_language(text, default=default)


def resolve_reply_language(
    question: str,
    *,
    previous_user: str | None = None,
    configured: str = ARABIC,
) -> str:
    """The language Natasha should answer ``question`` in.

    ``question`` decides it when it carries letters; otherwise the previous user
    turn is used; otherwise ``configured`` (the user's setting) is the fallback.
    So an English question gets an English answer, an Arabic question an Arabic
    one, and a bare "?" keeps the language of the exchange so far.
    """
    if _has_letters(question):
        return detect_language_of(question, default=configured)
    if previous_user and _has_letters(previous_user):
        return detect_language_of(previous_user, default=configured)
    return configured if configured in LANGUAGE_LABELS else ARABIC


def last_user_text(messages: Iterable[object]) -> str | None:
    """The most recent user turn's text, or ``None`` when there is none.

    Accepts anything with ``role`` and ``content`` attributes (a
    ``conversation_manager.Message``), so callers do not re-implement the scan.
    """
    latest: str | None = None
    for message in messages:
        if getattr(message, "role", None) == "user":
            latest = str(getattr(message, "content", "") or "")
    return latest


__all__ = [
    "ARABIC",
    "ENGLISH",
    "LANGUAGE_LABELS",
    "detect_language_of",
    "last_user_text",
    "resolve_reply_language",
]
