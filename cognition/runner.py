"""Background autonomous operation: the mind runs, but always stoppable.

:class:`CognitiveRunner` is a thin, *deterministic* scheduler loop that a caller
drives -- either from a background thread (the GUI) or from a loop (the CLI). It
owns no thread of its own, so it is testable with no sleeping and cannot outlive
its owner.

The task's resource rules live here in one place, as three **modes**:

* ``LIGHT`` -- cycles rarely, one idea at a time, for a busy machine;
* ``BALANCED`` -- the default: a cycle every few minutes;
* ``INTENSIVE`` -- cycles often, for a machine that is otherwise idle.

Every mode is still bounded by :class:`~cognition.scheduler.SchedulerSettings`, so
"intensive" means *more*, never *unbounded*. The runner checks the machine's own
figures before each cycle (through :mod:`core.model_resources`, which reports an
honest ``None`` where a platform cannot say) and defers a cycle when the machine
is busy, so the mind never competes with the user.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum

from cognition.proactive_mind import ProactiveMind
from cognition.records import ThoughtCycle
from cognition.scheduler import SchedulerSettings


class ResourceMode(str, Enum):
    """How hard the background mind may work on this machine."""

    LIGHT = "light"
    BALANCED = "balanced"
    INTENSIVE = "intensive"


#: The scheduler settings each mode applies.
MODE_SETTINGS: dict[ResourceMode, dict[str, object]] = {
    ResourceMode.LIGHT: {
        "interval_seconds": 900,
        "cycle_budget_seconds": 30,
        "ideas_per_cycle": 1,
        "model_calls_per_cycle": 1,
        "max_cycles_per_day": 32,
        "max_cpu_load": 0.6,
        "min_free_memory_fraction": 0.25,
    },
    ResourceMode.BALANCED: {
        "interval_seconds": 300,
        "cycle_budget_seconds": 60,
        "ideas_per_cycle": 3,
        "model_calls_per_cycle": 2,
        "max_cycles_per_day": 96,
        "max_cpu_load": 0.85,
        "min_free_memory_fraction": 0.15,
    },
    ResourceMode.INTENSIVE: {
        "interval_seconds": 120,
        "cycle_budget_seconds": 120,
        "ideas_per_cycle": 5,
        "model_calls_per_cycle": 4,
        "max_cycles_per_day": 240,
        "max_cpu_load": 0.95,
        "min_free_memory_fraction": 0.08,
    },
}


def apply_mode(mind: ProactiveMind, mode: ResourceMode) -> SchedulerSettings:
    """Set the mind's scheduler to a mode's settings, keeping the day's counters."""
    settings = mind.scheduler.settings
    for field, value in MODE_SETTINGS[mode].items():
        setattr(settings, field, value)
    return settings


#: Where the chosen mode is remembered, under the cognition meta namespace.
_MODE_KEY = "resource_mode"


def load_mode() -> ResourceMode:
    """The saved resource mode, or ``BALANCED`` -- never an invented value."""
    try:
        from cognition.storage import meta_get

        saved = meta_get(_MODE_KEY)
        if isinstance(saved, str):
            return ResourceMode(saved)
    except Exception:
        pass
    return ResourceMode.BALANCED


def save_mode(mode: ResourceMode) -> None:
    """Remember the chosen mode best-effort (a read-only data folder is fine)."""
    try:
        from cognition.storage import meta_set

        meta_set(_MODE_KEY, mode.value)
    except Exception:
        pass


@dataclass
class MachineLoad:
    """An honest reading of how busy the machine is."""

    cpu_load: float = 0.0
    free_memory_fraction: float = 1.0
    measured: bool = False

    def describe(self) -> str:
        if not self.measured:
            return "machine load not measured"
        return f"CPU {self.cpu_load:.2f}, free memory {self.free_memory_fraction:.2f}"


def measure_load() -> MachineLoad:
    """Read the machine's own figures, honestly reporting what cannot be read.

    CPU load is the recent *idle* fraction from the OS load average when it is
    available (falling back to 0.0, "assume free", rather than inventing a
    number); free memory comes from the project's own resource reader.
    """
    cpu_load = 0.0
    measured = False
    try:
        import os

        if hasattr(os, "getloadavg"):
            one, _, _ = os.getloadavg()
            cpus = os.cpu_count() or 1
            cpu_load = round(min(1.0, one / cpus), 6)
            measured = True
    except Exception:
        cpu_load = 0.0
    free_fraction = 1.0
    try:
        from core.model_resources import system_resources

        resources = system_resources()
        total = getattr(resources, "total_memory_bytes", None)
        available = getattr(resources, "available_memory_bytes", None)
        if total and available and total > 0:
            free_fraction = round(min(1.0, available / total), 6)
            measured = True
    except Exception:
        free_fraction = 1.0
    return MachineLoad(cpu_load=cpu_load, free_memory_fraction=free_fraction, measured=measured)


class CognitiveRunner:
    """Drive the proactive mind on demand, honouring the resource rules."""

    def __init__(
        self,
        mind: ProactiveMind,
        *,
        mode: ResourceMode | None = None,
        load_reader: Callable[[], MachineLoad] | None = None,
    ) -> None:
        self.mind = mind
        self.mode = mode or load_mode()
        self._load_reader = load_reader or measure_load
        self._last_decision: object | None = None
        apply_mode(self.mind, self.mode)

    def set_mode(self, mode: ResourceMode) -> None:
        self.mode = mode
        apply_mode(self.mind, mode)
        save_mode(mode)

    def tick(self) -> ThoughtCycle | None:
        """Run one cycle if it is due; return it, or ``None`` when deferred.

        This is the one method a background thread calls. It never sleeps: the
        caller decides the cadence, and the scheduler decides whether this tick
        may do work. A deferred tick returns ``None`` and a readable reason is
        available from :meth:`decision`.
        """
        load = self._load_reader()
        decision = self.mind.due(
            cpu_load=load.cpu_load,
            free_memory_fraction=load.free_memory_fraction,
        )
        self._last_decision = decision
        if not decision.allowed:
            return None
        return self.mind.cycle()

    def decision(self):
        """The last scheduling decision (for the UI's status line)."""
        return getattr(self, "_last_decision", None)

    def stop(self) -> None:
        """Pause the mind (the same as the UI's stop control)."""
        self.mind.pause()

    def start(self) -> None:
        self.mind.resume()

    def status(self) -> dict[str, object]:
        data = self.mind.status()
        load = self._load_reader()
        data.update(
            {
                "mode": self.mode.value,
                "load": load.describe(),
                "next_due": self.mind.scheduler.next_due_at().isoformat(),
            }
        )
        return data


__all__ = [
    "MODE_SETTINGS",
    "CognitiveRunner",
    "MachineLoad",
    "ResourceMode",
    "apply_mode",
    "load_mode",
    "measure_load",
    "save_mode",
]
