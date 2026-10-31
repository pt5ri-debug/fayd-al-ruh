"""The Thought Scheduler: when the mind may think, and for how long.

The scheduler is *not* a loop. It is a pure decision function
(:meth:`ThoughtScheduler.due`) plus a small state record, so the background
runner (or a "Think now" button) can ask "is a cycle due?" without any thread of
its own. This is what makes the mind stoppable, resumable, and testable with no
sleeping.

It enforces the task's resource rules in one place:

* a **minimum interval** between cycles (never a busy loop);
* a **daily budget** of cycles and of model calls;
* a **quiet window** when the user is busy (cycles are deferred, not skipped);
* **deferral** of a cycle that would exceed the CPU/RAM ceiling.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta


def _as_int(value: object, fallback: int = 0) -> int:
    """Read a stored integer, falling back rather than raising on bad user data."""
    if isinstance(value, bool) or not isinstance(value, int):
        return fallback
    return int(value)



def _now() -> datetime:
    return datetime.now(UTC)


@dataclass
class SchedulerSettings:
    """The tunable knobs the UI exposes, with safe defaults for a laptop."""

    #: Minimum seconds between two cycles (the anti-busy-loop guarantee).
    interval_seconds: int = 300
    #: Longest a single cycle may run before it must stop.
    cycle_budget_seconds: int = 60
    #: How many ideas a single cycle may produce.
    ideas_per_cycle: int = 3
    #: How many model calls a single cycle may make.
    model_calls_per_cycle: int = 2
    #: Daily caps, so a long-running mind stays bounded.
    max_cycles_per_day: int = 96
    max_model_calls_per_day: int = 200
    #: CPU load (0..1) above which a cycle is deferred while the user is busy.
    max_cpu_load: float = 0.85
    #: Fraction of RAM free below which a cycle is deferred.
    min_free_memory_fraction: float = 0.15
    #: The hours (local) during which the mind may run; empty means always.
    quiet_start_hour: int | None = 1
    quiet_end_hour: int | None = 6
    #: Master switch.
    enabled: bool = True

    def __post_init__(self) -> None:
        if self.interval_seconds < 0:
            raise ValueError("interval_seconds cannot be negative.")
        if self.cycle_budget_seconds < 1:
            raise ValueError("cycle_budget_seconds must be positive.")
        if not 1 <= self.ideas_per_cycle <= 20:
            raise ValueError("ideas_per_cycle must be between 1 and 20.")
        if not 0.0 <= self.max_cpu_load <= 1.0:
            raise ValueError("max_cpu_load must be between 0 and 1.")
        if not 0.0 <= self.min_free_memory_fraction <= 1.0:
            raise ValueError("min_free_memory_fraction must be between 0 and 1.")


@dataclass
class SchedulerState:
    """What the scheduler remembers between cycles (persisted as JSON)."""

    last_run: datetime | None = None
    cycles_today: int = 0
    model_calls_today: int = 0
    day: str = ""
    paused: bool = False

    def roll_day(self, today: str) -> None:
        """Reset the daily counters when the calendar day changes."""
        if self.day != today:
            self.day = today
            self.cycles_today = 0
            self.model_calls_today = 0

    def to_primitive(self) -> dict[str, object]:
        return {
            "last_run": None if self.last_run is None else self.last_run.isoformat(),
            "cycles_today": self.cycles_today,
            "model_calls_today": self.model_calls_today,
            "day": self.day,
            "paused": self.paused,
        }

    @classmethod
    def from_primitive(cls, data: dict[str, object]) -> SchedulerState:
        last = data.get("last_run")
        try:
            last_run = datetime.fromisoformat(str(last)) if last else None
        except ValueError:
            last_run = None
        return cls(
            last_run=last_run,
            cycles_today=_as_int(data.get("cycles_today")),
            model_calls_today=_as_int(data.get("model_calls_today")),
            day=str(data.get("day", "")),
            paused=bool(data.get("paused", False)),
        )


@dataclass(frozen=True)
class Decision:
    """Whether a cycle may run now, and if not, why."""

    allowed: bool
    reason: str = ""

    def describe(self) -> str:
        return "allowed" if self.allowed else f"deferred: {self.reason}"


class ThoughtScheduler:
    """Decide when a proactive cycle may run, honouring the resource rules."""

    def __init__(
        self,
        settings: SchedulerSettings | None = None,
        state: SchedulerState | None = None,
    ) -> None:
        self.settings = settings or SchedulerSettings()
        self.state = state or SchedulerState()

    def due(
        self,
        *,
        now: datetime | None = None,
        cpu_load: float = 0.0,
        free_memory_fraction: float = 1.0,
    ) -> Decision:
        """Return whether a cycle is due, checking every rule in a fixed order."""
        now = now or _now()
        self.state.roll_day(now.date().isoformat())
        if not self.settings.enabled:
            return Decision(False, "the proactive mind is disabled")
        if self.state.paused:
            return Decision(False, "the proactive mind is paused")
        if self._in_quiet_window(now):
            return Decision(False, "inside the quiet window")
        if self.state.cycles_today >= self.settings.max_cycles_per_day:
            return Decision(False, "the daily cycle budget is spent")
        if self.state.model_calls_today >= self.settings.max_model_calls_per_day:
            return Decision(False, "the daily model-call budget is spent")
        if self.state.last_run is not None:
            elapsed = (now - self.state.last_run).total_seconds()
            if elapsed < self.settings.interval_seconds:
                remaining = self.settings.interval_seconds - elapsed
                return Decision(False, f"only {elapsed:.0f}s since the last cycle ({remaining:.0f}s to go)")
        if cpu_load > self.settings.max_cpu_load:
            return Decision(False, f"CPU load {cpu_load:.2f} exceeds {self.settings.max_cpu_load:.2f}")
        if free_memory_fraction < self.settings.min_free_memory_fraction:
            return Decision(
                False,
                f"free memory {free_memory_fraction:.2f} is below "
                f"{self.settings.min_free_memory_fraction:.2f}",
            )
        return Decision(True)

    def record_cycle(self, *, now: datetime | None = None, model_calls: int = 0) -> None:
        """Record that a cycle ran, advancing the counters."""
        now = now or _now()
        self.state.roll_day(now.date().isoformat())
        self.state.last_run = now
        self.state.cycles_today += 1
        self.state.model_calls_today += max(0, model_calls)

    def pause(self) -> None:
        self.state.paused = True

    def resume(self) -> None:
        self.state.paused = False

    def set_enabled(self, enabled: bool) -> None:
        self.settings.enabled = enabled

    def _in_quiet_window(self, now: datetime) -> bool:
        start = self.settings.quiet_start_hour
        end = self.settings.quiet_end_hour
        if start is None or end is None:
            return False
        hour = now.astimezone().hour
        if start <= end:
            return start <= hour < end
        # A window that wraps midnight (e.g. 23 -> 6).
        return hour >= start or hour < end

    def next_due_at(self, *, now: datetime | None = None) -> datetime:
        """When the next cycle could earliest run (for the UI)."""
        now = now or _now()
        if self.state.last_run is None:
            return now
        return self.state.last_run + timedelta(seconds=self.settings.interval_seconds)


__all__ = [
    "Decision",
    "SchedulerSettings",
    "SchedulerState",
    "ThoughtScheduler",
]
