"""The Workplace settings: user preferences the module reads and writes.

Everything a *user* configures about the Workplace lives here as one small
``JsonStore`` document, so a preference survives a restart and a program update
can change a shipped default without overwriting what the user chose -- exactly
the pattern the project's other settings use (``gui.typography_settings``,
``core.character_registry``).

The settings are deliberately few and honest:

* ``language`` -- the interface language the Workplace opens in (``en`` / ``ar``).
* ``rtl`` -- whether the interface is laid out right-to-left.
* ``default_institution`` -- the institution a new unit/division defaults into.
* ``show_proposed`` -- whether proposed records are marked in the lists.

An unknown value falls back to the shipped default rather than raising, and a
corrupt or unreadable store reads as the defaults, so a window always opens.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from storage.json_store import JsonStore

#: The namespace/record the settings live under (one document).
NAMESPACE = "workplace"
KEY = "settings"

#: The interface languages the Workplace offers.
LANGUAGES = ("en", "ar")


@dataclass(frozen=True)
class WorkplaceSettings:
    """The Workplace's user preferences."""

    language: str = "en"
    rtl: bool = False
    default_institution: str = ""
    show_proposed: bool = True

    def with_language(self, language: str) -> WorkplaceSettings:
        chosen = language if language in LANGUAGES else self.language
        # Choosing Arabic implies RTL unless the user explicitly turned it off;
        # the layout direction is a real choice, so it is kept, not forced.
        return replace(self, language=chosen, rtl=(chosen == "ar" or self.rtl))

    def with_rtl(self, rtl: bool) -> WorkplaceSettings:
        return replace(self, rtl=bool(rtl))

    def with_default_institution(self, key: str) -> WorkplaceSettings:
        return replace(self, default_institution=key.strip())

    def with_show_proposed(self, show: bool) -> WorkplaceSettings:
        return replace(self, show_proposed=bool(show))

    def to_primitive(self) -> dict[str, Any]:
        return {
            "language": self.language,
            "rtl": self.rtl,
            "default_institution": self.default_institution,
            "show_proposed": self.show_proposed,
        }

    @classmethod
    def from_primitive(cls, data: Any) -> WorkplaceSettings:
        if not isinstance(data, dict):
            return cls()
        language = data.get("language")
        return cls(
            language=language if language in LANGUAGES else "en",
            rtl=bool(data.get("rtl", False)),
            default_institution=str(data.get("default_institution", "") or ""),
            show_proposed=bool(data.get("show_proposed", True)),
        )


def _store(root: Path | None) -> JsonStore:
    from core.paths import data_dir

    return JsonStore(root if root is not None else data_dir())


def load_settings(root: Path | None = None) -> WorkplaceSettings:
    """Read the saved settings, or the defaults when none are stored.

    A corrupt or unreadable store reads as the defaults rather than crashing a
    window, exactly like the project's typography and sidebar settings.
    """
    try:
        data = _store(root).load(NAMESPACE, KEY)
    except Exception:  # a broken store must not stop the Workplace opening
        return WorkplaceSettings()
    return WorkplaceSettings.from_primitive(data)


def save_settings(settings: WorkplaceSettings, root: Path | None = None) -> bool:
    """Persist the settings, returning whether the write succeeded."""
    try:
        _store(root).save(NAMESPACE, KEY, settings.to_primitive())
        return True
    except Exception:  # a read-only folder is reported by the caller, not fatal here
        return False


__all__ = [
    "KEY",
    "LANGUAGES",
    "NAMESPACE",
    "WorkplaceSettings",
    "load_settings",
    "save_settings",
]
