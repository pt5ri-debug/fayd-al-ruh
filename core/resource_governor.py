"""One arbiter over the machine's real figures, so nothing over-commits it.

Several subsystems want the same scarce resources: a language model needs
gigabytes of RAM and may want the GPU, a thought cycle wants CPU, an OCR or
vision pass wants memory *and* a model, and a training run wants all of it for a
long time. Today each of them decides on its own, from its own reading -- so two
of them can both decide "yes" and the machine swaps.

``ResourceGovernor`` is the one place that answers *"may I do this now?"*. It is
a **view** over the facts that already exist -- :mod:`core.model_resources`
(the machine's figures, each an honest ``None`` where the platform cannot say)
and :mod:`core.model_preflight` (an *estimated* memory need, with a band and its
unknowns named) -- and it never invents a figure. It returns a
:class:`GovernorDecision` that is either a grant or a deferral **with a reason**,
so a caller can say why it waited instead of silently slowing down.

Two rules keep it honest:

* an **unknown** is never treated as plenty. When the machine cannot report
  available memory, a request that *needs* a figure to be judged is **deferred**,
  not granted (``UNKNOWN_MEMORY``); a request that names no memory need is
  unaffected.
* the governor is **advisory**: it grants or defers, it never blocks a caller
  that chooses to proceed, and it changes no knowledge and loads no model. It is
  how the core *arbitrates*, not how it *acts*.

Dependency-free, CPU-first, and it imports nothing heavy -- the model preflight
is imported lazily so a governor built on a machine with no model catalog still
works.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from core.best_effort import best_effort
from core.model_resources import SystemResources, system_resources

#: Headroom kept free after a grant, so the machine is not filled to the brim.
#: A quarter of available memory is reserved for the OS, the GUI, and the
#: unexpected -- a model that exactly fits "available" does not really fit.
DEFAULT_HEADROOM = 0.25

#: How many concurrent *model* loads the governor allows by default. On the
#: stated 16 GB CPU machine one large model at a time is the realistic ceiling,
#: so the default is one; a stronger machine raises it in configuration.
DEFAULT_MAX_CONCURRENT_MODELS = 1


class GovernorReason(str, Enum):
    """Why a decision came out the way it did -- one closed vocabulary."""

    GRANTED = "granted"
    LOW_MEMORY = "low_memory"
    UNKNOWN_MEMORY = "unknown_memory"
    TOO_MANY_MODELS = "too_many_models"
    TOO_MANY_TASKS = "too_many_tasks"
    DISABLED = "disabled"


@dataclass(frozen=True)
class ResourceRequest:
    """What a caller wants to do, and how much of the machine it needs.

    ``memory_bytes`` is the *high* end of an estimate (``None`` when the caller
    cannot estimate); ``model`` marks a request that occupies a model slot;
    ``label`` names the work in the reason so a deferral is readable.
    """

    label: str
    memory_bytes: int | None = None
    model: bool = False
    cpu_bound: bool = False
    gpu: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.label.strip():
            raise ValueError("A resource request needs a label.")
        if self.memory_bytes is not None and self.memory_bytes < 0:
            raise ValueError("A memory estimate cannot be negative.")


@dataclass(frozen=True)
class GovernorDecision:
    """The governor's answer: a grant, or a deferral with a readable reason."""

    granted: bool
    reason: GovernorReason
    label: str
    detail: str = ""
    available_memory_bytes: int | None = None
    required_memory_bytes: int | None = None

    def describe(self) -> str:
        if self.granted:
            return f"granted: {self.label}"
        return f"deferred: {self.label} ({self.detail})"

    def to_primitive(self) -> dict[str, Any]:
        return {
            "granted": self.granted,
            "reason": self.reason.value,
            "label": self.label,
            "detail": self.detail,
            "available_memory_bytes": self.available_memory_bytes,
            "required_memory_bytes": self.required_memory_bytes,
        }


class ResourceGovernor:
    """The one arbiter. Advisory, honest about unknowns, and cheap to ask.

    ``resources`` is the reading function (injectable, so a test can fix the
    machine's figures). ``on_decision`` observes every decision -- the event bus
    or an audit log is the natural sink -- and is best-effort: an observer that
    raises never breaks the governor.
    """

    def __init__(
        self,
        *,
        resources: Callable[[], SystemResources] = system_resources,
        headroom: float = DEFAULT_HEADROOM,
        max_concurrent_models: int = DEFAULT_MAX_CONCURRENT_MODELS,
        max_concurrent_tasks: int = 8,
        enabled: bool = True,
        on_decision: Callable[[GovernorDecision], None] | None = None,
    ) -> None:
        if not 0.0 <= headroom < 1.0:
            raise ValueError("Headroom must be in [0, 1).")
        if max_concurrent_models < 1:
            raise ValueError("At least one model slot is required.")
        if max_concurrent_tasks < 1:
            raise ValueError("At least one task slot is required.")
        self._resources = resources
        self.headroom = headroom
        self.max_concurrent_models = max_concurrent_models
        self.max_concurrent_tasks = max_concurrent_tasks
        self.enabled = enabled
        self._on_decision = on_decision
        self._active_models: set[str] = set()
        self._active_tasks: set[str] = set()
        self._decisions: list[GovernorDecision] = []

    # -- asking ------------------------------------------------------------

    def request(self, request: ResourceRequest) -> GovernorDecision:
        """Decide whether ``request`` may proceed now, and record the decision."""
        decision = self._decide(request)
        self._decisions.append(decision)
        if self._on_decision is not None:
            with best_effort("resource governor observer"):
                self._on_decision(decision)
        return decision

    def _decide(self, request: ResourceRequest) -> GovernorDecision:
        if not self.enabled:
            return self._defer(request, GovernorReason.DISABLED, "the governor is disabled")

        if request.model and len(self._active_models) >= self.max_concurrent_models:
            return self._defer(
                request,
                GovernorReason.TOO_MANY_MODELS,
                f"{len(self._active_models)}/{self.max_concurrent_models} model slot(s) in use",
            )

        if len(self._active_tasks) >= self.max_concurrent_tasks:
            return self._defer(
                request,
                GovernorReason.TOO_MANY_TASKS,
                f"{len(self._active_tasks)}/{self.max_concurrent_tasks} task slot(s) in use",
            )

        resources = self._resources()
        available = resources.available_memory_bytes

        if request.memory_bytes is not None:
            if available is None:
                # The machine cannot report free memory, so a need cannot be
                # judged. An unknown is not plenty -- defer, honestly.
                return self._defer(
                    request,
                    GovernorReason.UNKNOWN_MEMORY,
                    "available memory is unknown on this platform",
                    available=available,
                )
            budget = int(available * (1.0 - self.headroom))
            if request.memory_bytes > budget:
                return self._defer(
                    request,
                    GovernorReason.LOW_MEMORY,
                    f"needs {_size(request.memory_bytes)}, budget {_size(budget)} "
                    f"of {_size(available)} available",
                    available=available,
                )

        return GovernorDecision(
            granted=True,
            reason=GovernorReason.GRANTED,
            label=request.label,
            available_memory_bytes=available,
            required_memory_bytes=request.memory_bytes,
        )

    @staticmethod
    def _defer(
        request: ResourceRequest,
        reason: GovernorReason,
        detail: str,
        *,
        available: int | None = None,
    ) -> GovernorDecision:
        return GovernorDecision(
            granted=False,
            reason=reason,
            label=request.label,
            detail=detail,
            available_memory_bytes=available,
            required_memory_bytes=request.memory_bytes,
        )

    # -- occupancy ---------------------------------------------------------

    def acquire(self, request: ResourceRequest) -> GovernorDecision:
        """Ask, and on a grant **reserve** the slot so the next caller sees it.

        This is the form a long-lived consumer uses (loading a model, starting a
        training run): the grant is not merely advisory, it takes the slot. A
        deferral reserves nothing.
        """
        decision = self.request(request)
        if decision.granted:
            if request.model:
                self._active_models.add(request.label)
            self._active_tasks.add(request.label)
        return decision

    def release(self, label: str) -> None:
        """Free a slot taken by :meth:`acquire` (idempotent)."""
        self._active_models.discard(label)
        self._active_tasks.discard(label)

    @property
    def active_models(self) -> tuple[str, ...]:
        return tuple(sorted(self._active_models))

    @property
    def active_tasks(self) -> tuple[str, ...]:
        return tuple(sorted(self._active_tasks))

    # -- reading -----------------------------------------------------------

    def snapshot(self) -> dict[str, Any]:
        """The machine's figures plus what is currently reserved."""
        resources = self._resources()
        return {
            "resources": resources.to_primitive(),
            "headroom": self.headroom,
            "max_concurrent_models": self.max_concurrent_models,
            "max_concurrent_tasks": self.max_concurrent_tasks,
            "active_models": list(self.active_models),
            "active_tasks": list(self.active_tasks),
            "enabled": self.enabled,
        }

    def describe(self) -> str:
        resources = self._resources()
        return "\n".join(
            [
                "Resource governor",
                f"  memory available: {_size(resources.available_memory_bytes)}"
                f" of {_size(resources.total_memory_bytes)}",
                f"  headroom:         {self.headroom:.0%}",
                f"  models in use:    {len(self._active_models)}/{self.max_concurrent_models}",
                f"  tasks in use:     {len(self._active_tasks)}/{self.max_concurrent_tasks}",
                f"  enabled:          {self.enabled}",
            ]
        )

    @property
    def decisions(self) -> tuple[GovernorDecision, ...]:
        return tuple(self._decisions)

    def can_load_model(self, memory_bytes: int | None, *, label: str = "model") -> GovernorDecision:
        """Convenience: may one more model of this size be loaded now?"""
        return self.request(ResourceRequest(label=label, memory_bytes=memory_bytes, model=True))

    def can_run_task(self, *, label: str = "task", memory_bytes: int | None = None) -> GovernorDecision:
        """Convenience: may one more (non-model) task run now?"""
        return self.request(ResourceRequest(label=label, memory_bytes=memory_bytes))


def _size(value: int | None) -> str:
    """A readable byte size, or ``unknown`` -- never a zero standing in for one."""
    if value is None:
        return "unknown"
    amount = float(value)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if amount < 1024 or unit == "TB":
            return f"{amount:.1f} {unit}"
        amount /= 1024
    return f"{amount:.1f} TB"


def estimate_model_memory(path: Any, *, context: int | None = None) -> int | None:
    """The *high* end of a model file's memory estimate, or ``None`` if unknown.

    Delegates to :mod:`core.model_preflight` (imported lazily) so the governor
    carries no dependency on the model catalog. A file that cannot be inspected
    yields ``None`` -- the governor then treats the need as unknown, not as zero.
    """
    try:
        from core.model_files import probe_file
        from core.model_preflight import estimate_memory

        probe = probe_file(path)
        if not probe.usable:
            # A missing or unrecognised file is an *unknown* need, not a zero one.
            return None
        estimate = estimate_memory(probe, context_length=context)
    except Exception:
        return None
    high = getattr(estimate, "high", None)
    return high or None


__all__ = [
    "DEFAULT_HEADROOM",
    "DEFAULT_MAX_CONCURRENT_MODELS",
    "GovernorDecision",
    "GovernorReason",
    "ResourceGovernor",
    "ResourceRequest",
    "estimate_model_memory",
]
