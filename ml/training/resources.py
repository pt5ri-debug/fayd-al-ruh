"""The training resource monitor: the guards a run checks before and during work.

A training run on the stated hardware (16 GB RAM, no usable CUDA GPU) must not be
allowed to exhaust the machine. This module is the honest guard:

* :func:`check_disk` refuses a run when the free space is below the configured
  floor, naming the real figures;
* :func:`check_memory` refuses a run whose *estimated* need exceeds the
  configured ceiling **or** the machine's available memory, and is explicit that
  the estimate is an estimate;
* :class:`ResourceMonitor` samples the process's own resident memory during a run
  and records the peak, so a run's record can report what it actually used.

It reads the machine through :mod:`core.model_resources` (the project's one
place the machine's figures are read), so a training report and a Model Manager
report never disagree about the same machine.
"""

from __future__ import annotations

import os
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

from core.model_resources import SystemResources, system_resources


class ResourceError(RuntimeError):
    """A resource guard refused the run."""


@dataclass(frozen=True)
class ResourceCheck:
    """One guard's verdict, with the figures it used."""

    name: str
    ok: bool
    detail: str = ""
    figures: tuple[tuple[str, str], ...] = ()

    def describe(self) -> str:
        state = "ok" if self.ok else "refused"
        text = f"{self.name}: {state}"
        if self.detail:
            text += f" -- {self.detail}"
        for label, value in self.figures:
            text += f"\n    {label}: {value}"
        return text


def _human(size: int | None) -> str:
    if size is None:
        return "unknown"
    value = float(size)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if value < 1024 or unit == "TB":
            return f"{value:.1f} {unit}" if unit != "B" else f"{value:.0f} B"
        value /= 1024
    return f"{value:.0f} B"


def check_disk(
    path: Path | str | None = None, *, floor_bytes: int
) -> ResourceCheck:
    """Refuse when the free disk at ``path`` is below ``floor_bytes``."""
    import shutil

    target = Path(path) if path is not None else Path.cwd()
    try:
        usage = shutil.disk_usage(target if target.exists() else target.parent)
        free = usage.free
    except OSError as exc:
        return ResourceCheck(
            "disk", False, f"could not read the free space at {target}: {exc}"
        )
    ok = free >= floor_bytes
    detail = (
        "enough free space"
        if ok
        else f"needs at least {_human(floor_bytes)} free, found {_human(free)}"
    )
    return ResourceCheck(
        "disk",
        ok,
        detail,
        figures=(("free", _human(free)), ("required", _human(floor_bytes))),
    )


def check_memory(
    estimated_bytes: int,
    *,
    limit_bytes: int,
    resources: SystemResources | None = None,
) -> ResourceCheck:
    """Refuse when an estimate exceeds the configured ceiling or free memory.

    The estimate is **not** a measurement, and the check says so: it compares the
    caller's estimate against the ceiling and against what the machine reports as
    available, and refuses only when it cannot fit.
    """
    system = resources or system_resources()
    figures: list[tuple[str, str]] = [
        ("estimated need", _human(estimated_bytes)),
        ("configured ceiling", _human(limit_bytes)),
    ]
    ok = True
    detail = "the estimate fits the configured ceiling"
    if estimated_bytes > limit_bytes:
        ok = False
        detail = (
            f"the estimated need {_human(estimated_bytes)} exceeds the configured "
            f"ceiling {_human(limit_bytes)}"
        )
    elif system.available_memory_bytes is not None:
        figures.append(("available memory", _human(system.available_memory_bytes)))
        if estimated_bytes > system.available_memory_bytes:
            ok = False
            detail = (
                f"the estimated need {_human(estimated_bytes)} exceeds the "
                f"{_human(system.available_memory_bytes)} the machine reports free"
            )
    else:
        figures.append(("available memory", "unknown on this platform"))
        detail += " (the machine's free memory could not be read)"
    return ResourceCheck("memory", ok, detail, figures=tuple(figures))


def estimate_training_memory(
    *,
    parameter_count: int | None,
    rank: int,
    batch_size: int,
    grad_accum: int,
    optimizer: str = "adam",
    bytes_per_parameter: int = 4,
) -> int:
    """A transparent **estimate** of a LoRA run's memory need, in bytes.

    The estimate is built from named terms and is deliberately conservative:

    * the **frozen base** weights (``parameters × bytes_per_parameter``);
    * the **trainable adapter** (a small multiple of ``rank`` per layer, bounded);
    * the **optimizer state** for the adapter only (LoRA's whole point: the base
      is frozen, so no optimizer state is kept for it);
    * the **activations** for one batch (a bounded multiple of the batch size).

    It is an estimate, never a guarantee, and a caller must treat it as one.
    """
    params = max(1, int(parameter_count or 1))
    base = params * bytes_per_parameter
    # The adapter is tiny next to the base: rank * a per-layer constant, capped.
    adapter = min(base, max(64 * 1024, rank * 64 * 1024))
    optimizer_factor = 3 if optimizer.lower().startswith("adam") else 1
    optimizer_state = adapter * optimizer_factor
    effective_batch = max(1, batch_size) * max(1, grad_accum)
    activations = effective_batch * 512 * 1024
    overhead = int((base + adapter + optimizer_state + activations) * 0.1)
    return base + adapter + optimizer_state + activations + overhead


@dataclass
class ResourceMonitor:
    """Samples the process's resident memory and records the peak.

    ``sampler`` is an injection seam, so the monitor is tested without a real
    process. The default reads ``/proc/self/status`` (Linux), the OS on Windows
    through :mod:`core.model_resources`, or returns ``None`` when the platform
    cannot say -- in which case the peak is honestly unknown.
    """

    sampler: Callable[[], int | None] | None = None
    samples: list[int] = field(default_factory=list)
    peak_bytes: int | None = None
    start_bytes: int | None = None

    def sample(self) -> int | None:
        """Take one reading and fold it into the peak."""
        value = (self.sampler or _process_memory_bytes)()
        if value is None:
            return None
        self.samples.append(value)
        self.peak_bytes = value if self.peak_bytes is None else max(self.peak_bytes, value)
        return value

    def start(self) -> int | None:
        value = self.sample()
        self.start_bytes = value
        return value

    @property
    def grew_by(self) -> int | None:
        if self.peak_bytes is None or self.start_bytes is None:
            return None
        return max(0, self.peak_bytes - self.start_bytes)

    def describe(self) -> str:
        if self.peak_bytes is None:
            return "resource monitor: the process memory could not be read"
        return (
            f"resource monitor: peak {_human(self.peak_bytes)} resident, "
            f"grew {_human(self.grew_by)} over {len(self.samples)} sample(s)"
        )


def _process_memory_bytes() -> int | None:
    """The process's own resident memory, or ``None`` when unavailable."""
    try:
        import resource  # POSIX only

        usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        # Linux reports kilobytes, macOS reports bytes.
        return int(usage * 1024) if os.name == "posix" and usage < 10**12 else int(usage)
    except Exception:
        pass
    try:
        system = system_resources()
        return system.process_memory_bytes
    except Exception:
        return None


__all__ = [
    "ResourceCheck",
    "ResourceError",
    "ResourceMonitor",
    "check_disk",
    "check_memory",
    "estimate_training_memory",
]
