"""The one central registry: load, validate, save, APPLY, and roll back.

Every surface -- the GUI's Characters window, the CLI, and the code that wires a
mouth -- reads and writes configuration through **this** module, so there is one
source of truth and no character's behaviour is hard-coded anywhere else. It
replaces nothing: it *wraps* the existing defaults
(:func:`core.character_config.default_config_for`) and persists an edit through
:class:`storage.json_store.JsonStore`'s atomic write.

Reading follows a two-layer rule that keeps an update safe:

1. the **shipped** default in the repository's ``config/`` folder (tracked, so an
   update refreshes it);
2. the **user's applied** configuration under ``data/config`` (git-ignored, so an
   update never overwrites it), which wins when it exists.

A missing, corrupt, or partial file falls back to the shipped default rather than
raising, so a character is never unusable.

:meth:`CharacterRegistry.apply` is the APPLY the task asks for: it validates the
draft, snapshots the current configuration, saves the new one, tries to *activate*
the affected components, and **rolls back** to the snapshot if activation fails --
so a character is never left partially configured. A setting that cannot be
hot-applied is reported as restart-required rather than silently ignored.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from core.best_effort import best_effort
from core.character_config import (
    CharacterConfig,
    default_config_for,
    owner_names,
)
from core.learning_isolation import MAIN_MODEL_OWNER, owner_for
from core.paths import bundled_config_dir, character_config_dir

#: The JsonStore namespace every applied configuration lives under.
NAMESPACE = "characters"


class RegistryError(RuntimeError):
    """Raised when a configuration cannot be loaded, saved, or applied."""


@dataclass
class ApplyReport:
    """What an APPLY did: what was saved, what activated, what needs a restart."""

    key: str
    saved: bool = False
    activated: tuple[str, ...] = ()
    restart_required: tuple[str, ...] = ()
    rolled_back: bool = False
    message: str = ""

    @property
    def ok(self) -> bool:
        return self.saved and not self.rolled_back

    def describe(self) -> str:
        """One honest report of the APPLY, naming anything that needs a restart."""
        if self.rolled_back:
            return f"Could not apply {self.key}: {self.message} (the previous configuration was restored)."
        if not self.saved:
            return f"Nothing was applied for {self.key}: {self.message}"
        lines = [f"Applied {self.key}'s configuration."]
        if self.activated:
            lines.append("Activated now: " + ", ".join(self.activated) + ".")
        if self.restart_required:
            lines.append(
                "Needs a restart to take effect: " + ", ".join(self.restart_required) + "."
            )
        return " ".join(lines)


def _read_json(path: Path) -> dict[str, Any] | None:
    """Read a JSON object, or ``None`` for a missing or unreadable file."""
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return data if isinstance(data, dict) else None


class CharacterRegistry:
    """The central configuration manager for every character and the main model.

    ``data_root`` lets a caller keep the applied configuration under its own data
    folder (a test, or a portable build); the layout beneath it is the same
    ``config/characters`` tree.
    """

    def __init__(
        self,
        *,
        data_root: Path | None = None,
        bundled_root: Path | None = None,
    ) -> None:
        self._data_root = data_root
        self._bundled_root = bundled_root

    # --------------------------------------------------------------- paths

    def applied_dir(self) -> Path:
        """Where the user's applied configuration lives (``data/config``)."""
        if self._data_root is not None:
            return Path(self._data_root) / "config"
        return character_config_dir().parent

    def shipped_dir(self) -> Path:
        """Where the shipped defaults live (read-only, ``config/``)."""
        if self._bundled_root is not None:
            return Path(self._bundled_root) / "config"
        return bundled_config_dir()

    def applied_path(self, key: str) -> Path:
        """``data/config/characters/<key>.json``, or ``data/config/fayd_al_ruh.json``.

        The main model's file sits beside ``characters/`` rather than inside it,
        exactly as the task's suggested layout shows, so the project's own model is
        visibly *not* one of the seven characters.
        """
        return self._path_in(self.applied_dir(), key)

    def shipped_path(self, key: str) -> Path:
        """``config/characters/<key>.json``, or ``config/fayd_al_ruh.json``."""
        return self._path_in(self.shipped_dir(), key)

    @staticmethod
    def _path_in(folder: Path, key: str) -> Path:
        owner = owner_for(key)
        if owner == MAIN_MODEL_OWNER:
            return folder / f"{owner}.json"
        return folder / "characters" / f"{owner}.json"

    # -------------------------------------------------------------- reading

    def keys(self) -> tuple[str, ...]:
        """Every owner this registry manages, in the project's order."""
        return owner_names()

    def load(self, key: str) -> CharacterConfig:
        """The effective configuration for ``key``: the user's, else the shipped one.

        Falls back to the built-in default when neither file exists, and tolerates a
        corrupt file by moving on to the next layer, so a character always loads.
        """
        owner = owner_for(key)
        for path in (self.applied_path(owner), self.shipped_path(owner)):
            data = _read_json(path)
            if data is not None:
                try:
                    config = CharacterConfig.from_primitive({**data, "key": owner, "owner": owner})
                except (ValueError, TypeError):
                    continue
                return config
        return default_config_for(owner)

    def load_all(self) -> dict[str, CharacterConfig]:
        """Every owner's effective configuration, keyed by owner name."""
        return {owner: self.load(owner) for owner in self.keys()}

    def is_customised(self, key: str) -> bool:
        """True when the user has applied a configuration for ``key``."""
        return self.applied_path(key).is_file()

    # -------------------------------------------------------------- writing

    def save(self, config: CharacterConfig) -> CharacterConfig:
        """Persist an **already validated** configuration atomically.

        Validation is the caller's (or :meth:`apply`'s) responsibility so a caller
        that only wants to seed a default does not pay for it twice; a store error
        is best-effort so a read-only folder never crashes a window, but it is
        recorded rather than swallowed.
        """
        payload = config.to_primitive()
        with best_effort(f"save {config.owner}'s configuration"):
            target = self.applied_path(config.owner)
            target.parent.mkdir(parents=True, exist_ok=True)
            temporary = target.with_suffix(".tmp")
            temporary.write_text(
                json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True),
                encoding="utf-8",
            )
            temporary.replace(target)
        return config

    def reset(self, key: str) -> CharacterConfig:
        """Drop the user's applied configuration, back to the shipped default.

        Returns the restored default, so a caller can report what is now in force.
        """
        path = self.applied_path(key)
        if path.is_file():
            with best_effort(f"reset {key}'s configuration"):
                path.unlink()
        return self.load(key)

    # ----------------------------------------------------------------- APPLY

    def apply(
        self,
        config: CharacterConfig,
        *,
        activate: Callable[[CharacterConfig], tuple[str, ...]] | None = None,
    ) -> ApplyReport:
        """Validate, save, activate, and roll back on failure (the APPLY button).

        ``activate`` is the caller's own activation hook: it receives the validated
        configuration, applies what it can *now*, and returns the names of the parts
        it activated. Anything the hook does not name is reported as
        restart-required, so a setting that could not be hot-applied is stated
        rather than silently lost.

        The order is what makes it safe:

        1. validate the draft (a bad edit raises :class:`ConfigError` and nothing is
           written);
        2. snapshot the current configuration;
        3. save the new one;
        4. run the activation hook;
        5. if the hook raises, restore the snapshot and report the rollback.

        The return value distinguishes "saved but needs a restart" from "activated
        now" from "failed and rolled back", so a caller can show exactly what
        happened.
        """
        config.validate()
        previous = self.load(config.owner)
        self.save(config)
        activated: tuple[str, ...] = ()
        if activate is not None:
            try:
                activated = tuple(activate(config))
            except Exception as error:  # the hook failed: restore the old config
                self.save(previous)
                report = ApplyReport(
                    key=config.owner,
                    saved=False,
                    rolled_back=True,
                    message=f"{type(error).__name__}: {error}",
                )
                return report
        everything = ("general", "personality", "model", "voice", "learning")
        restart = tuple(name for name in everything if name not in activated)
        return ApplyReport(
            key=config.owner,
            saved=True,
            activated=activated,
            restart_required=restart,
        )

    # ------------------------------------------------------------- shipping

    def write_shipped_defaults(self, folder: Path | None = None) -> tuple[Path, ...]:
        """Write the shipped default files for every owner.

        Used once to populate the repository's ``config/`` folder; kept in the
        module so the shipped files are always regenerable from the same defaults
        the code uses, and can never drift from them.
        """
        target_dir = Path(folder) if folder is not None else self.shipped_dir()
        written: list[Path] = []
        for owner in self.keys():
            config = default_config_for(owner)
            path = self._path_in(target_dir, owner)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(
                json.dumps(config.to_primitive(), ensure_ascii=False, indent=2, sort_keys=True)
                + "\n",
                encoding="utf-8",
            )
            written.append(path)
        return tuple(written)


#: The process-wide registry, built lazily so importing this module touches no disk.
_default: CharacterRegistry | None = None


def default_registry() -> CharacterRegistry:
    """The shared registry (built on first use)."""
    global _default
    if _default is None:
        _default = CharacterRegistry()
    return _default


def load_config(key: str) -> CharacterConfig:
    """The effective configuration for ``key`` (the shared registry)."""
    return default_registry().load(key)


def apply_config(
    config: CharacterConfig,
    *,
    activate: Callable[[CharacterConfig], tuple[str, ...]] | None = None,
) -> ApplyReport:
    """APPLY ``config`` through the shared registry."""
    return default_registry().apply(config, activate=activate)


__all__ = [
    "NAMESPACE",
    "ApplyReport",
    "CharacterRegistry",
    "RegistryError",
    "apply_config",
    "default_registry",
    "load_config",
]
