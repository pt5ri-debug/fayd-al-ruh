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


@dataclass(frozen=True)
class PreferenceRecord:
    """The stored preferences, with conservative defaults."""

    user_name: str = ""
    language: str = "en"
    speak: bool = True
    wake_word: str = "jarvis"

    def __post_init__(self) -> None:
        if self.language not in SUPPORTED_LANGUAGES:
            raise ValueError(f"Unsupported language: {self.language!r}")
        if not self.wake_word.strip():
            raise ValueError("Wake word cannot be empty.")

    def to_primitive(self) -> dict[str, object]:
        return {
            "user_name": self.user_name,
            "language": self.language,
            "speak": self.speak,
            "wake_word": self.wake_word,
        }

    @classmethod
    def from_primitive(cls, data: dict[str, object]) -> "PreferenceRecord":
        return cls(
            user_name=str(data.get("user_name", "")),
            language=str(data.get("language", "en")),
            speak=bool(data.get("speak", True)),
            wake_word=str(data.get("wake_word", "jarvis")),
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
        unknown = set(changes) - {"user_name", "language", "speak", "wake_word"}
        if unknown:
            raise ValueError(f"Unknown preference(s): {', '.join(sorted(unknown))}")
        current = self._record
        self._record = PreferenceRecord(
            user_name=str(changes.get("user_name", current.user_name)),
            language=str(changes.get("language", current.language)),
            speak=bool(changes.get("speak", current.speak)),
            wake_word=str(changes.get("wake_word", current.wake_word)),
        )
        self.save()
        return self._record


__all__ = ["PreferenceRecord", "SUPPORTED_LANGUAGES", "UserPreferences"]
