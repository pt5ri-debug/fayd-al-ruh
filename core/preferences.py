"""Persistent user preferences for Jarvis mode.

Jarvis should remember how the user likes to be served: their name, their
language, and whether to speak answers aloud. Preferences are a small, honest
record persisted with the project's :class:`storage.json_store.JsonStore`, so
they survive restarts without a database.

Nothing here is inferred behind the user's back: preferences change only when
:meth:`UserPreferences.update` is called with an explicit value.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from storage.json_store import JsonStore

_NAMESPACE = "jarvis"
_KEY = "preferences"

SUPPORTED_LANGUAGES = ("en", "ar")

#: The vocalization forms a preference may name, mirroring
#: :data:`core.text_polish.TASHKEEL_MODES` (imported lazily to avoid a cycle).
TASHKEEL_MODES = ("formal", "colloquial")


@dataclass(frozen=True)
class PreferenceRecord:
    """The stored preferences, with conservative defaults."""

    user_name: str = ""
    language: str = "en"
    speak: bool = True
    wake_word: str = "jarvis"
    #: Whether Jarvis may search the internet. Off by default: a search leaves the
    #: machine and costs a network round trip, so it is an explicit opt-in, and the
    #: ``web on``/``web off`` control flips this and persists it.
    web: bool = False
    #: Whether every answer's Arabic is fully vocalized (tashkeel). Off by default,
    #: so a bare, undiacritized answer stays the project's output until the user
    #: asks otherwise with the ``tashkeel on`` control.
    tashkeel: bool = False
    #: The vocalization form: ``"formal"`` (فصحى, case endings included) or
    #: ``"colloquial"`` (عامية, lexical vowels only).
    tashkeel_mode: str = "formal"

    def __post_init__(self) -> None:
        if self.language not in SUPPORTED_LANGUAGES:
            raise ValueError(f"Unsupported language: {self.language!r}")
        if not self.wake_word.strip():
            raise ValueError("Wake word cannot be empty.")
        if self.tashkeel_mode not in TASHKEEL_MODES:
            raise ValueError(f"Unsupported tashkeel mode: {self.tashkeel_mode!r}")

    def to_primitive(self) -> dict[str, object]:
        return {
            "user_name": self.user_name,
            "language": self.language,
            "speak": self.speak,
            "wake_word": self.wake_word,
            "web": self.web,
            "tashkeel": self.tashkeel,
            "tashkeel_mode": self.tashkeel_mode,
        }

    @classmethod
    def from_primitive(cls, data: dict[str, object]) -> PreferenceRecord:
        return cls(
            user_name=str(data.get("user_name", "")),
            language=str(data.get("language", "en")),
            speak=bool(data.get("speak", True)),
            wake_word=str(data.get("wake_word", "jarvis")),
            web=bool(data.get("web", False)),
            tashkeel=bool(data.get("tashkeel", False)),
            tashkeel_mode=str(data.get("tashkeel_mode", "formal")),
        )


class UserPreferences:
    """Loads, updates, and saves :class:`PreferenceRecord` via ``JsonStore``."""

    def __init__(self, root: str | Path) -> None:
        self.store = JsonStore(root)
        self._record = self._load()

    def _load(self) -> PreferenceRecord:
        data = self.store.load(_NAMESPACE, _KEY)
        if not isinstance(data, dict):
            return PreferenceRecord()
        try:
            return PreferenceRecord.from_primitive(data)
        except (ValueError, TypeError):
            return PreferenceRecord()

    @property
    def record(self) -> PreferenceRecord:
        return self._record

    def save(self) -> None:
        self.store.save(_NAMESPACE, _KEY, self._record.to_primitive())

    def update(self, **changes: object) -> PreferenceRecord:
        """Apply explicit changes, persist them, and return the new record."""
        unknown = set(changes) - {
            "user_name", "language", "speak", "wake_word", "web", "tashkeel", "tashkeel_mode",
        }
        if unknown:
            raise ValueError(f"Unknown preference(s): {', '.join(sorted(unknown))}")
        current = self._record
        self._record = PreferenceRecord(
            user_name=str(changes.get("user_name", current.user_name)),
            language=str(changes.get("language", current.language)),
            speak=bool(changes.get("speak", current.speak)),
            wake_word=str(changes.get("wake_word", current.wake_word)),
            web=bool(changes.get("web", current.web)),
            tashkeel=bool(changes.get("tashkeel", current.tashkeel)),
            tashkeel_mode=str(changes.get("tashkeel_mode", current.tashkeel_mode)),
        )
        self.save()
        return self._record


__all__ = ["SUPPORTED_LANGUAGES", "PreferenceRecord", "UserPreferences"]
