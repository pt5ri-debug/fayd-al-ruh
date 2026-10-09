"""Live progress reporting for long training runs.

A run that prints nothing for minutes looks hung, and the operator cannot tell
how much is left. This module turns a loop counter into a readable line with the
percentage, elapsed time, and an estimate of the time remaining, so every
training entry point can show movement without pulling in a dependency.

The reporter is deliberately small and dependency-free (stdlib only). It writes
to ``sys.stderr`` by default, so a caller's stdout stays clean for reports and
machine-readable JSON while progress is still visible on a terminal.
"""

from __future__ import annotations

import sys
import time
from typing import IO, Any, Protocol


class ProgressCallback(Protocol):
    """The shape a training loop calls to report progress."""

    def __call__(self, completed: int, total: int, *, detail: str = "") -> None: ...


def format_duration(seconds: float) -> str:
    """A compact, human-readable duration: ``45s``, ``2m 05s``, ``1h 03m``."""
    seconds = max(0.0, seconds)
    if seconds < 60:
        return f"{seconds:.0f}s"
    minutes, remainder = divmod(int(round(seconds)), 60)
    if minutes < 60:
        return f"{minutes}m {remainder:02d}s"
    hours, minutes = divmod(minutes, 60)
    return f"{hours}h {minutes:02d}m"


class ProgressReporter:
    """Print a progress line every so often as a loop advances.

    ``every`` controls the cadence: a value of ``0`` picks a sensible default of
    about twenty updates across the whole run. The reporter is callable, so it
    can be handed straight to a training loop as a
    :data:`ProgressCallback`.
    """

    def __init__(
        self,
        label: str,
        total: int,
        *,
        every: int = 0,
        stream: IO[str] | None = None,
        enabled: bool = True,
    ) -> None:
        self.label = label
        self.total = max(1, int(total))
        self.every = every if every > 0 else max(1, self.total // 20)
        self.stream = stream
        self.enabled = enabled
        self.started = time.monotonic()
        self._last_printed = 0

    def __call__(self, completed: int, total: int | None = None, *, detail: str = "") -> None:
        """Record progress; print when the cadence boundary or the end is reached."""
        if total is not None:
            self.total = max(1, int(total))
        completed = max(0, int(completed))
        finished = completed >= self.total
        if not finished and completed - self._last_printed < self.every:
            return
        self._last_printed = completed
        self._emit(completed, detail=detail)

    def finish(self, *, detail: str = "") -> None:
        """Force a final 100% line unless one was already printed."""
        if self._last_printed >= self.total:
            return
        self._last_printed = self.total
        self._emit(self.total, detail=detail)

    def _emit(self, completed: int, *, detail: str) -> None:
        if not self.enabled:
            return
        elapsed = time.monotonic() - self.started
        fraction = min(1.0, completed / self.total)
        percent = f"{fraction * 100:5.1f}%"
        parts = [
            f"  {self.label:<28.28} {completed:>6}/{self.total:<6} {percent}",
            f"elapsed {format_duration(elapsed)}",
        ]
        if completed > 0 and completed < self.total:
            remaining = elapsed * (self.total - completed) / completed
            parts.append(f"eta {format_duration(remaining)}")
        elif completed >= self.total:
            parts.append("done")
        if detail:
            parts.append(detail)
        print("  ".join(parts), file=self.stream or sys.stderr, flush=True)


def null_reporter(*_args: Any, **_kwargs: Any) -> None:
    """A reporter that prints nothing, for callers that want to stay silent."""
    return None


__all__ = ["ProgressCallback", "ProgressReporter", "format_duration", "null_reporter"]
