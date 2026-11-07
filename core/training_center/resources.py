"""Honest resource figures and training estimates for the Training Center.

The project already reads this machine's own figures (:mod:`core.model_resources`)
and estimates a training run's memory need (:mod:`ml.training.resources`). This
module is a *thin composition* of the two, plus one honest timing note, so the
Resource Monitor section and the pre-training estimate show the same numbers and
never invent one:

* a figure the platform cannot report stays ``None`` and is drawn as "unknown";
* a model's memory need is an *estimate*, labelled as one, with its unknowns
  named;
* a training *time* is only ever given when a measured rate exists; otherwise the
  estimate says plainly that the time is unknown rather than guessing.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ml.training.resources import estimate_training_memory

from core.model_resources import SystemResources, system_resources

#: The one honest sentence shown beside every estimate.
ESTIMATE_LINE = (
    "A model's memory need and a run's time are estimates derived from the model "
    "and this machine's own figures, never measurements."
)


def human_size(value: int | None) -> str:
    """A readable byte size, or ``unknown`` -- never a zero standing in for one."""
    if value is None:
        return "unknown"
    amount = float(value)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if amount < 1024 or unit == "TB":
            return f"{amount:.1f} {unit}"
        amount /= 1024
    return f"{amount:.1f} TB"


@dataclass(frozen=True)
class ResourceSnapshot:
    """This machine's own figures, read from :mod:`core.model_resources`."""

    resources: SystemResources

    @property
    def cpu_count(self) -> int | None:
        return self.resources.cpu_count

    @property
    def total_memory_bytes(self) -> int | None:
        return self.resources.total_memory_bytes

    @property
    def available_memory_bytes(self) -> int | None:
        return self.resources.available_memory_bytes

    @property
    def disk_free_bytes(self) -> int | None:
        return self.resources.disk_free_bytes

    @property
    def gpu_available(self) -> bool:
        return bool(self.resources.gpu.available)

    def describe(self) -> str:
        return self.resources.describe()

    def to_primitive(self) -> dict[str, Any]:
        return self.resources.to_primitive()


def read_resources() -> ResourceSnapshot:
    """Read this machine's figures (never raises; unknowns stay ``None``)."""
    return ResourceSnapshot(resources=system_resources())


@dataclass(frozen=True)
class TrainingEstimate:
    """An *estimate* of what one training run would need, with its unknowns."""

    parameter_count: int | None
    estimated_memory_bytes: int
    fits: bool | None
    memory_detail: str
    disk_free_bytes: int | None
    disk_ok: bool
    notes: tuple[str, ...] = ()

    def describe(self) -> str:
        lines = [
            "Training estimate (an estimate, not a measurement)",
            f"  parameters:        {self.parameter_count if self.parameter_count else 'unknown'}",
            f"  estimated memory:  {human_size(self.estimated_memory_bytes)}",
            f"  memory:            {self.memory_detail}",
            f"  disk free:         {human_size(self.disk_free_bytes)}"
            + ("" if self.disk_ok else "  (below the safety floor)"),
        ]
        lines.extend(f"  note: {note}" for note in self.notes)
        lines.append(f"  {ESTIMATE_LINE}")
        return "\n".join(lines)


def estimate_training(
    *,
    parameter_count: int | None,
    rank: int = 8,
    batch_size: int = 4,
    grad_accum: int = 2,
    root: Path | str | None = None,
    disk_floor_bytes: int = 0,
    memory_limit_bytes: int = 0,
) -> TrainingEstimate:
    """Estimate one run's memory need and check it against this machine.

    ``parameter_count`` may be ``None`` for a model whose size the project cannot
    read; the estimate then says so rather than guessing. The memory figure is the
    project's own :func:`ml.training.resources.estimate_training_memory`, so the
    Training Center and the controlled trainer agree.
    """
    estimated = estimate_training_memory(
        parameter_count=parameter_count, rank=rank, batch_size=batch_size, grad_accum=grad_accum
    )
    resources = system_resources()
    available = resources.available_memory_bytes
    notes: list[str] = []
    fits: bool | None
    if available is None:
        fits = None
        memory_detail = "this platform does not report available memory"
        notes.append("available memory is unknown, so whether the run fits is unknown too")
    else:
        fits = estimated <= available
        memory_detail = (
            f"needs {human_size(estimated)} of {human_size(available)} available "
            + ("(fits)" if fits else "(may not fit)")
        )
        if not fits:
            notes.append("the estimate exceeds available memory; a smaller rank or batch is safer")
    if memory_limit_bytes and estimated > memory_limit_bytes:
        notes.append(
            f"the estimate exceeds the configured memory limit ({human_size(memory_limit_bytes)})"
        )
    disk_free = resources.disk_free_bytes
    disk_ok = True
    if disk_free is not None and disk_floor_bytes and disk_free < disk_floor_bytes:
        disk_ok = False
        notes.append("free disk is below the configured safety floor")
    if parameter_count is None:
        notes.append("the model's parameter count could not be read from its file")
    return TrainingEstimate(
        parameter_count=parameter_count,
        estimated_memory_bytes=estimated,
        fits=fits,
        memory_detail=memory_detail,
        disk_free_bytes=disk_free,
        disk_ok=disk_ok,
        notes=tuple(notes),
    )


__all__ = [
    "ESTIMATE_LINE",
    "ResourceSnapshot",
    "TrainingEstimate",
    "estimate_training",
    "human_size",
    "read_resources",
]
