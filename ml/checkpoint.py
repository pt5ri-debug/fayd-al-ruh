"""Durable training-progress state, so a long run continues instead of restarting.

Every trainer in the project already keeps its learned models on disk -- the
Q-policies (``policy_<task>.npz``), the predictive-coding brains
(``brain_<task>/``), the language-model checkpoint and its optimiser sidecar --
but until now only the language model could actually *continue*: the policy and
brain loops re-ran from episode 0 on every invocation, so a run interrupted at
hour two restarted at hour zero.

This module is the missing bookkeeping: one small JSON ledger, written beside the
models it describes, that records how far each *unit of work* has come. Resuming
is then a read of the ledger rather than a re-run of the training, and the
recorded figure is the **cumulative** one, so raising the episode budget later
tops a task up instead of starting it over.

Three rules keep it honest:

* **A changed configuration invalidates the record.** The ledger stores a
  fingerprint of the work it describes (the task list, the seed, the replay
  batch); if that changes, the recorded progress is stale and is discarded rather
  than resumed into a run it no longer matches. The episode *budget* is
  deliberately **not** part of the fingerprint -- that is what makes a top-up
  possible.
* **A corrupt ledger reads as empty, never fatal.** A truncated or malformed file
  reads as "nothing recorded yet", so a crash mid-write cannot brick a trainer.
* **It is bounded.** A long-lived ledger cannot grow without limit; only the most
  recent ``DEFAULT_MAX_UNITS`` units are kept.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

#: The ledger's file name, written into whatever folder a trainer saves to.
LEDGER_NAME = "training_progress.json"

#: The ledger schema; a file from a newer build is read leniently (its unknown
#: keys are ignored) rather than refused, because progress is not precious enough
#: to fail a run over.
SCHEMA_VERSION = 1

#: A bounded ledger: the most recent units only, so a long-lived file cannot grow
#: without limit.
DEFAULT_MAX_UNITS = 4096


def fingerprint_of(config: Any) -> str:
    """A stable digest of the work a scope describes.

    Two runs with the same task list, seed, and replay settings share a
    fingerprint; any change to those yields a different one, which is how a
    changed run is told apart from a continued one.
    """
    payload = json.dumps(config, sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.blake2b(payload.encode("utf-8"), digest_size=16).hexdigest()


@dataclass(frozen=True)
class ProgressEntry:
    """How far one unit of work has come, and the artifact it left behind."""

    completed: int = 0
    artifact: str = ""


class TrainingProgress:
    """The durable, bounded record of how far one training scope has come.

    A *scope* is one trainer's work in one folder -- ``petroleum`` in
    ``artifacts/petroleum``, ``ml`` in ``artifacts`` -- so several trainers can
    share a ledger file without their units colliding.
    """

    def __init__(
        self,
        directory: str | Path,
        *,
        scope: str,
        config: Any = None,
        max_units: int = DEFAULT_MAX_UNITS,
    ) -> None:
        self.directory = Path(directory)
        self.path = self.directory / LEDGER_NAME
        self.scope = scope
        self.fingerprint = fingerprint_of(config) if config is not None else ""
        self.max_units = max(1, int(max_units))
        self.stale = False
        self._document = self._read()
        scopes = self._document.setdefault("scopes", {})
        recorded = scopes.get(scope)
        if isinstance(recorded, dict) and self.fingerprint:
            if str(recorded.get("fingerprint", "")) != self.fingerprint:
                # The work this scope describes has changed, so the recorded
                # progress no longer applies; start clean rather than resuming
                # into a run the record does not match.
                scopes.pop(scope, None)
                recorded = None
                self.stale = True
        self._units: dict[str, ProgressEntry] = {}
        if isinstance(recorded, dict):
            raw = recorded.get("units")
            if isinstance(raw, dict):
                for unit, value in raw.items():
                    entry = self._parse_entry(value)
                    if entry is not None:
                        self._units[str(unit)] = entry

    # -- reading -----------------------------------------------------------
    @staticmethod
    def _parse_entry(value: Any) -> ProgressEntry | None:
        if isinstance(value, dict):
            try:
                return ProgressEntry(
                    completed=int(value.get("completed", 0)),
                    artifact=str(value.get("artifact", "")),
                )
            except (TypeError, ValueError):
                return None
        if isinstance(value, int):
            # A bare integer is accepted as the completed count, so a hand-written
            # ledger (or an older shape) still reads.
            return ProgressEntry(completed=value)
        return None

    def _read(self) -> dict[str, Any]:
        """Read the ledger, treating anything unreadable as empty."""
        if not self.path.exists():
            return {"version": SCHEMA_VERSION, "scopes": {}}
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return {"version": SCHEMA_VERSION, "scopes": {}}
        if not isinstance(data, dict):
            return {"version": SCHEMA_VERSION, "scopes": {}}
        if not isinstance(data.get("scopes"), dict):
            data["scopes"] = {}
        return data

    @property
    def available(self) -> bool:
        """Whether anything is recorded for this scope."""
        return bool(self._units)

    @property
    def units(self) -> dict[str, ProgressEntry]:
        return dict(self._units)

    def completed(self, unit: str) -> int:
        """The cumulative count recorded for a unit (0 when nothing is)."""
        entry = self._units.get(unit)
        return entry.completed if entry is not None else 0

    def remaining(self, unit: str, target: int) -> int:
        """How much more of a unit is needed to reach ``target`` (never negative)."""
        return max(0, int(target) - self.completed(unit))

    def artifact_path(self, unit: str) -> Path | None:
        """The saved artifact a unit recorded, resolved against the ledger folder."""
        entry = self._units.get(unit)
        if entry is None or not entry.artifact:
            return None
        candidate = Path(entry.artifact)
        return candidate if candidate.is_absolute() else self.directory / candidate

    # -- writing -----------------------------------------------------------
    def record(self, unit: str, completed: int, *, artifact: str | Path | None = None) -> None:
        """Record a unit's cumulative progress (and the artifact it produced)."""
        previous = self._units.get(unit)
        artifact_name = previous.artifact if previous is not None else ""
        if artifact is not None:
            path = Path(artifact)
            try:
                artifact_name = str(path.relative_to(self.directory))
            except ValueError:
                artifact_name = str(path)
        self._units[unit] = ProgressEntry(completed=max(0, int(completed)), artifact=artifact_name)
        if len(self._units) > self.max_units:
            # Keep the most recent units; a ledger is a cache of progress, not an
            # archive, so trimming the oldest is safe.
            for stale in list(self._units)[: len(self._units) - self.max_units]:
                self._units.pop(stale, None)

    def reset(self) -> None:
        """Forget this scope's recorded progress."""
        self._units.clear()
        self.stale = True

    def save(self) -> Path:
        """Write the ledger atomically, so a crash cannot leave it half-written."""
        scopes = self._document.setdefault("scopes", {})
        scopes[self.scope] = {
            "fingerprint": self.fingerprint,
            "units": {
                unit: {"completed": entry.completed, "artifact": entry.artifact}
                for unit, entry in self._units.items()
            },
        }
        self._document["version"] = SCHEMA_VERSION
        self.directory.mkdir(parents=True, exist_ok=True)
        payload = json.dumps(self._document, ensure_ascii=False, indent=2, sort_keys=True)
        handle, temp_name = tempfile.mkstemp(dir=str(self.directory), prefix=".progress-", suffix=".tmp")
        try:
            with os.fdopen(handle, "w", encoding="utf-8") as sink:
                sink.write(payload)
            os.replace(temp_name, self.path)
        except BaseException:
            Path(temp_name).unlink(missing_ok=True)
            raise
        return self.path

    def describe(self) -> str:
        """A readable one-line summary of what is recorded."""
        if not self._units:
            note = " (the recorded work no longer matches this run)" if self.stale else ""
            return f"{self.scope}: nothing recorded yet{note}"
        done = sum(1 for entry in self._units.values() if entry.completed > 0)
        total = sum(entry.completed for entry in self._units.values())
        return f"{self.scope}: {done} unit(s), {total} cumulative step(s) recorded"


__all__ = [
    "DEFAULT_MAX_UNITS",
    "LEDGER_NAME",
    "SCHEMA_VERSION",
    "ProgressEntry",
    "TrainingProgress",
    "fingerprint_of",
]
