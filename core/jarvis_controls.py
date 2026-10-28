"""Conversation-level controls for Jarvis mode.

A voice assistant needs a few commands that are *not* questions and *not* tool
calls: stop talking, repeat that, start over, help. This module recognises them
so :class:`core.jarvis.JarvisMode` can act on them directly.

The matcher is deliberately conservative:

* a control must be the **whole** utterance, once normalised -- so "how do I stop
  the pump?" is a question, not a mute command;
* leading fillers ("please restart", "jarvis stop") are removed
  before matching, so everyday phrasing still works;
* it never touches the knowledge graph, the sandbox, or any external capability,
  so a control can never widen the model's authority.

English and Arabic phrases are both accepted, matching the project's convention
of English-primary content with Arabic kept as aliases.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum


class ControlIntent(str, Enum):
    """The conversation-level command a phrase maps to."""

    MUTE = "mute"
    SPEAK = "speak"
    REPEAT = "repeat"
    RESET = "reset"
    SAVE = "save"
    SESSIONS = "sessions"
    WEB = "web"
    CONFIRM = "confirm"
    CORRECT = "correct"
    WHY = "why"
    HELP = "help"


# Phrases are exact matches after normalisation. Longer phrases are tried first
# so a specific command is never shadowed by a shorter one.
_PHRASES: dict[ControlIntent, tuple[str, ...]] = {
    ControlIntent.MUTE: (
        "stop", "stop it", "stop talking", "be quiet", "silence", "quiet", "hush",
        "mute", "shut up",
        "توقف", "اسكت", "اصمت", "صمت", "اهدأ",
    ),
    ControlIntent.SPEAK: (
        "speak", "speak up", "talk", "unmute", "voice on", "turn on voice", "sound on",
        "تكلم", "انطق", "شغل الصوت", "شغّل الصوت", "فعّل الصوت",
    ),
    ControlIntent.REPEAT: (
        "repeat", "repeat that", "say that again", "say again", "again", "once more",
        "أعد", "اعد", "كرر", "مرة أخرى", "مرة اخرى", "قل مرة أخرى",
    ),
    ControlIntent.RESET: (
        "new conversation", "reset", "reset conversation", "start over", "start again",
        "forget everything", "clear conversation", "forget this",
        "محادثة جديدة", "ابدأ من جديد", "ابدأ من جديد", "نسيت كل شيء", "مسح المحادثة",
    ),
    ControlIntent.SAVE: (
        "save", "save conversation", "save session", "save this",
        "احفظ", "احفظ المحادثة", "احفظ الجلسة",
    ),
    ControlIntent.SESSIONS: (
        "sessions", "list sessions", "saved conversations", "show sessions",
        "الجلسات", "اعرض الجلسات", "قائمة الجلسات", "المحادثات المحفوظة",
    ),
    ControlIntent.WEB: (
        "web", "web on", "web off", "use the web", "use the internet", "go online",
        "online", "offline", "search the web", "internet on", "internet off",
        "البحث في الإنترنت", "استخدم الإنترنت", "الإنترنت", "اتصل بالإنترنت",
    ),
    ControlIntent.CONFIRM: (
        "that's right", "thats right", "that is right", "that's correct", "thats correct",
        "that is correct", "correct", "right", "yes exactly", "exactly", "confirmed",
        "confirm", "that's true", "thats true", "you are right", "spot on", "well done",
        "صحيح", "هذا صحيح", "بالضبط", "أجل", "نعم صحيح", "مضبوط",
    ),
    ControlIntent.WHY: (
        "why", "why?", "why that", "why so", "how come", "explain", "explain that",
        "explain your answer", "explain the answer", "justify", "justify that",
        "show your reasoning", "show the reasoning", "reasoning", "your reasoning",
        "how did you reach that", "how did you get that", "why did you say that",
        "لماذا", "لماذا؟", "اشرح", "اشرح ذلك", "اشرح إجابتك", "وضّح", "وضح",
        "ما سبب ذلك", "كيف وصلت إلى ذلك", "أرني الاستدلال",
    ),
    ControlIntent.HELP: (
        "help", "commands", "what can you do", "what can you do?", "list commands",
        "مساعدة", "ساعدني", "الأوامر", "الاوامر", "ماذا تستطيع",
    ),
}

# Leading fillers are dropped before matching, so "please repeat" and
# "jarvis stop" are understood without accepting arbitrary sentences.
_FILLERS = (
    "hey jarvis", "ok jarvis", "hey", "ok", "okay", "jarvis",
    "please", "just", "now",
    "يا جارفيس", "من فضلك", "الآن", "الان",
)

_NON_WORD = re.compile(r"[^\w\s]", flags=re.UNICODE)


def normalize_utterance(text: str) -> str:
    """Case-fold, strip punctuation, and collapse whitespace for exact matching."""
    cleaned = _NON_WORD.sub(" ", text.strip().casefold())
    return " ".join(cleaned.split())


@dataclass(frozen=True)
class JarvisControl:
    """A recognised conversation control and the phrase that matched it."""

    intent: ControlIntent
    phrase: str
    utterance: str
    correction: str = ""


class JarvisControls:
    """Matches an utterance against the control phrases (exact, normalised)."""

    def __init__(self, phrases: dict[ControlIntent, tuple[str, ...]] | None = None) -> None:
        source = phrases or _PHRASES
        # Store normalised phrases, longest first, so the most specific wins.
        self._phrases: tuple[tuple[ControlIntent, str], ...] = tuple(
            (intent, normalize_utterance(phrase))
            for intent in source
            for phrase in sorted(source[intent], key=len, reverse=True)
        )

    def match(self, utterance: str) -> JarvisControl | None:
        """Return the control an utterance names, or ``None`` if it is not one."""
        normalized = normalize_utterance(utterance)
        if not normalized:
            return None
        stripped = _strip_leading_filler(normalized)
        for intent, phrase in self._phrases:
            if stripped == phrase:
                return JarvisControl(intent=intent, phrase=phrase, utterance=utterance)
        correction = _extract_correction(stripped)
        if correction:
            return JarvisControl(
                intent=ControlIntent.CORRECT,
                phrase="correction",
                utterance=utterance,
                correction=correction,
            )
        return None


def _extract_correction(normalized: str) -> str:
    """Return the corrected wording when an utterance clearly supplies one.

    Only a tight, explicit pattern matches ("that's wrong, it's X" / "no, it's X"
    / "actually X"), so an ordinary disagreement without new content stays an
    answer rather than being treated as a control.
    """
    for pattern in (
        r"^(?:that s |thats |that is |it s |its |it is )?(?:wrong|incorrect|not right)"
        r"[ ,]+(?:it s |its |it is )?(.+)$",
        r"^no[ ,]+(?:it s |its |it is )(.+)$",
        r"^actually[ ,]+(?:it s |its |it is )(.+)$",
        r"^(?:لا|خطأ)[ ،]+بل (.+)$",
        r"^بل (.+)$",
    ):
        found = re.search(pattern, normalized)
        if found and found.group(1).strip():
            return found.group(1).strip()
    return ""


def _strip_leading_filler(normalized: str) -> str:
    """Drop leading filler words ("please", "hey jarvis", ...) if any are present."""
    changed = True
    while changed:
        changed = False
        for filler in sorted(_FILLERS, key=len, reverse=True):
            if normalized.startswith(filler + " "):
                normalized = normalized[len(filler) + 1:].strip()
                changed = True
                break
    return normalized


__all__ = [
    "ControlIntent",
    "JarvisControl",
    "JarvisControls",
    "normalize_utterance",
]
