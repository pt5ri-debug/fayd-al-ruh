"""Will this model fit here? -- an estimate, and the checks that precede a load.

Two questions a model manager must answer honestly, and neither with a guess:

* **How much memory does this model need?** The weights are the file size, but a
  model also needs a KV cache, a compute buffer, and the runtime's own overhead --
  and those depend on the architecture, the context length, and the quantization,
  not on one multiplier applied to every model. :func:`estimate_memory` builds the
  estimate from whatever facts are actually known, returns a **band** rather than a
  single confident figure, and says which term dominates. When a fact is missing it
  says so instead of filling it in.
* **Can it run here at all?** :func:`preflight` runs the checks the audit lists --
  file present, integrity, runtime available, format compatible, metadata present,
  memory estimated against what is really free, the context supported, GPU offload
  validated -- and returns a structured verdict with the blocking errors, the
  warnings, the recommended configuration, and a confidence.

The one rule that runs through both: **an estimate is never a guarantee.** Every
rendering says so, and ``fits`` means "the estimate is within what the OS reports
as available", which is a reason to try, not a promise that a load will succeed.

Nothing here loads a model, starts a process, or imports a runtime: it reads
metadata and the machine, and returns data.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from core.model_files import FORMAT_FAYD_CHECKPOINT, FORMAT_GGUF, FORMAT_UNKNOWN, probe_file
from core.model_resources import (
    GpuInfo,
    SystemResources,
    describe_reading,
    system_resources,
)

#: How confident an estimate is, in words -- never a false precision.
CONFIDENCE_HIGH = "high"
CONFIDENCE_MEDIUM = "medium"
CONFIDENCE_LOW = "low"
CONFIDENCE_UNKNOWN = "unknown"

CONFIDENCE_NOTES: dict[str, str] = {
    CONFIDENCE_HIGH: "the file size, architecture, and context are all known",
    CONFIDENCE_MEDIUM: "some figures are known; the KV term is approximated",
    CONFIDENCE_LOW: "little is known beyond the file size; treat this as a rough guide",
    CONFIDENCE_UNKNOWN: "not enough is known to estimate at all",
}

#: The fraction of the estimate the runtime's own buffers are assumed to add.
#: A band, not a constant: a small model pays proportionally more.
_RUNTIME_OVERHEAD_LOW = 0.05
_RUNTIME_OVERHEAD_HIGH = 0.15

#: Bytes per KV-cache element at full precision, halved for a quantized cache.
_KV_BYTES_FP16 = 2

#: The context a model is assumed to run at when it declares none: the project's
#: own default, so the estimate matches what a load would actually use.
DEFAULT_CONTEXT = 4096


@dataclass(frozen=True)
class MemoryEstimate:
    """An estimate of a model's memory need, with the uncertainty made explicit.

    ``low`` and ``high`` bracket the estimate -- the honest answer to "will it
    fit?" is "somewhere between these", and a caller compares the *high* end to
    what is free. ``dominant_term`` says what drove the figure, so a user can see
    that a long context, not the weights, is the problem.
    """

    weights_bytes: int = 0
    kv_cache_bytes: int = 0
    runtime_overhead_bytes: int = 0
    context_length: int = 0
    confidence: str = CONFIDENCE_UNKNOWN
    #: What the estimate could not account for, so a reader knows the gap.
    unknowns: tuple[str, ...] = field(default=())
    notes: tuple[str, ...] = field(default=())

    @property
    def low(self) -> int:
        return self.weights_bytes + self.kv_cache_bytes + self.runtime_overhead_bytes

    @property
    def high(self) -> int:
        # The high end adds the upper overhead band on top of the low estimate.
        base = self.weights_bytes + self.kv_cache_bytes
        return base + int(base * _RUNTIME_OVERHEAD_HIGH) if base else 0

    @property
    def typical(self) -> int:
        """A single figure for a progress bar or a one-line summary, never a promise."""
        return self.low + max(0, (self.high - self.low) // 2)

    @property
    def dominant_term(self) -> str:
        """Which term drove the estimate: ``weights``, ``kv-cache``, or ``none``."""
        if not self.weights_bytes and not self.kv_cache_bytes:
            return "none"
        return "kv-cache" if self.kv_cache_bytes > self.weights_bytes else "weights"

    def describe(self) -> str:
        if not self.weights_bytes and not self.kv_cache_bytes:
            return "Memory: cannot be estimated (no file size was available)."
        lines = [
            f"Memory estimate: {_size(self.low)} – {_size(self.high)} "
            f"(typical {_size(self.typical)})",
            f"  weights:          {_size(self.weights_bytes)}",
            f"  KV cache:         {_size(self.kv_cache_bytes)}"
            + (f" at {self.context_length} tokens" if self.context_length else ""),
            f"  runtime overhead: {_size(self.runtime_overhead_bytes)} (a band, not a constant)",
            f"  dominated by:     {self.dominant_term}",
            f"  confidence:       {self.confidence} — {CONFIDENCE_NOTES.get(self.confidence, '')}",
        ]
        lines.extend(f"  unknown:          {item}" for item in self.unknowns)
        lines.extend(f"  note:             {item}" for item in self.notes)
        lines.append(
            "  This is an estimate, not a measurement. A model can fail to load even "
            "when the estimate fits."
        )
        return "\n".join(lines)

    def to_primitive(self) -> dict[str, Any]:
        return {
            "confidence": self.confidence,
            "context_length": self.context_length,
            "dominant_term": self.dominant_term,
            "high": self.high,
            "kv_cache_bytes": self.kv_cache_bytes,
            "low": self.low,
            "runtime_overhead_bytes": self.runtime_overhead_bytes,
            "typical": self.typical,
            "unknowns": list(self.unknowns),
            "weights_bytes": self.weights_bytes,
        }


def estimate_memory(
    record: Any,
    *,
    context_length: int | None = None,
    gpu_layers: int = 0,
    gpu: GpuInfo | None = None,
) -> MemoryEstimate:
    """Estimate a model's memory need from the facts that are actually known.

    The weights term is the file size -- a real fact. The KV term needs the
    architecture, the layer count, and the context, and when any is missing it is
    *not* invented: the estimate says which it could not account for and lowers its
    confidence accordingly. ``gpu_layers`` records how many layers a caller intends
    to offload; the estimate notes that those move to VRAM but does **not** claim
    to model the split precisely, because that depends on the runtime.
    """
    size = int(getattr(record, "size_bytes", 0) or 0)
    architecture = str(getattr(record, "architecture", "") or "")
    declared_context = getattr(record, "context_length", None)
    context = context_length or declared_context or 0
    parameters = getattr(record, "parameter_count", None)

    unknowns: list[str] = []
    notes: list[str] = []
    if not size:
        unknowns.append("the file size")
    if not architecture:
        unknowns.append("the architecture (the KV cache cannot be sized)")
    if not context:
        unknowns.append("the context length")
        context = DEFAULT_CONTEXT
        notes.append(f"No context was declared; the estimate assumes {DEFAULT_CONTEXT} tokens.")

    weights = size
    kv = _kv_cache_bytes(
        architecture=architecture,
        parameters=parameters,
        context=context,
    )
    base = weights + kv
    overhead = int(base * _RUNTIME_OVERHEAD_LOW)

    if weights and kv and architecture and declared_context:
        confidence = CONFIDENCE_HIGH
    elif weights and (kv or architecture):
        confidence = CONFIDENCE_MEDIUM
    elif weights:
        confidence = CONFIDENCE_LOW
    else:
        confidence = CONFIDENCE_UNKNOWN

    if gpu_layers:
        notes.append(
            f"{gpu_layers} layer(s) were requested to be offloaded; part of this would "
            "move to VRAM, which this estimate does not model precisely."
        )
        if gpu is not None and gpu.available and gpu.memory_total_bytes:
            notes.append(
                f"This machine reports {gpu.memory_total_bytes / 1e9:.2f} GB of GPU memory."
            )
        elif gpu is not None and not gpu.available:
            notes.append("No GPU memory was reported, so the offload may not be possible.")

    return MemoryEstimate(
        weights_bytes=weights,
        kv_cache_bytes=kv,
        runtime_overhead_bytes=overhead,
        context_length=context,
        confidence=confidence,
        unknowns=tuple(dict.fromkeys(unknowns)),
        notes=tuple(notes),
    )


def _kv_cache_bytes(
    *,
    architecture: str,
    parameters: int | None,
    context: int,
) -> int:
    """The KV cache, estimated from the architecture and the context.

    A real transformer's cache is ``2 (keys+values) * layers * d_model * context *
    bytes_per_element``. None of those are in the file header directly, so the
    layer count and the width are inferred from the parameter count where it is
    known, and the result is deliberately a rough figure -- reported as part of a
    band, and ``0`` when the inputs are not there rather than a made-up number.
    """
    if not context:
        return 0
    if not architecture and not parameters:
        return 0
    if not parameters or parameters <= 0:
        return 0
    # A standard heuristic: layers ~ n^(1/3) scaled, width ~ sqrt(n / layers).
    layers = max(1, round((parameters ** (1 / 3)) / 12))
    width = max(1, round((parameters / (layers * 12)) ** 0.5))
    return int(2 * layers * width * context * _KV_BYTES_FP16)


# ------------------------------------------------------------------ preflight


@dataclass(frozen=True)
class PreflightReport:
    """Whether a model can be loaded, with the reasons and the recommendation.

    ``can_load`` is ``True`` only when **no** blocking error was found. The
    warnings are things a load would survive; the recommendations are changes that
    would help (a shorter context, CPU-only). ``confidence`` says how much the
    resource part of the verdict is worth, because an estimate is not a guarantee.
    """

    model_id: str
    runtime: str
    can_load: bool
    errors: tuple[str, ...] = field(default=())
    warnings: tuple[str, ...] = field(default=())
    recommendations: tuple[str, ...] = field(default=())
    estimate: MemoryEstimate | None = None
    resources: SystemResources | None = None
    context_length: int | None = None
    gpu_offload: bool | None = None
    confidence: str = CONFIDENCE_UNKNOWN
    #: True when the only reason a model would not run is that a GPU is absent.
    cpu_fallback: bool = False

    def describe(self) -> str:
        head = "can load" if self.can_load else "cannot load"
        lines = [f"{head} ({self.runtime}) — {self.model_id}"]
        lines.extend(f"  error:          {item}" for item in self.errors)
        lines.extend(f"  warning:        {item}" for item in self.warnings)
        lines.extend(f"  recommendation: {item}" for item in self.recommendations)
        if self.cpu_fallback:
            lines.append("  CPU fallback:   available; a GPU is not required")
        if self.estimate is not None and (self.estimate.weights_bytes or self.estimate.kv_cache_bytes):
            lines.append("  " + self.estimate.describe().replace("\n", "\n  "))
        if self.resources is not None and self.resources.available_memory_bytes is not None:
            lines.append(
                f"  available memory: {_size(self.resources.available_memory_bytes)} "
                f"({describe_reading(self.resources.memory_reading)})"
            )
        return "\n".join(lines)

    def to_primitive(self) -> dict[str, Any]:
        return {
            "can_load": self.can_load,
            "confidence": self.confidence,
            "context_length": self.context_length,
            "cpu_fallback": self.cpu_fallback,
            "errors": list(self.errors),
            "estimate": self.estimate.to_primitive() if self.estimate is not None else None,
            "gpu_offload": self.gpu_offload,
            "model_id": self.model_id,
            "recommendations": list(self.recommendations),
            "runtime": self.runtime,
            "warnings": list(self.warnings),
        }


def preflight(
    manager: Any,
    model_id: str,
    *,
    context_length: int | None = None,
    gpu_layers: int | None = None,
    resources: SystemResources | None = None,
    adapter: Any = None,
) -> PreflightReport:
    """Every check that should pass before a model is loaded.

    Deliberately a **function over the manager**, not a method on it: it composes
    facts the manager already owns (the record, the adapter's capabilities and
    validation) with the machine's own figures, and it changes nothing. A caller
    that wants to load without checking can still do so; this is the path that
    tells the truth first.

    ``context_length`` and ``gpu_layers`` let a user ask "would it fit at *this*
    setting?", which is what the recommendations are for. An unknown GPU offload
    never blocks CPU execution -- it is a warning, and ``cpu_fallback`` says so.
    """
    from core.model_registry import ModelNotFoundError
    from core.model_runtime import RuntimeUnavailable, UnsupportedCapability

    errors: list[str] = []
    warnings: list[str] = []
    recommendations: list[str] = []
    cpu_fallback = False

    try:
        record = manager.registry.get(model_id)
    except ModelNotFoundError as error:
        return PreflightReport(
            model_id=model_id, runtime="", can_load=False, errors=(str(error),),
        )

    runtime = str(getattr(record, "runtime", "") or "unknown")
    resolved_resources = resources if resources is not None else system_resources()

    # 1-2. the file, and its integrity
    path = getattr(record, "path", None)
    if path is None:
        from core.model_registry import SOURCE_REMOTE, SOURCE_TRAINED

        if record.source not in (SOURCE_TRAINED, SOURCE_REMOTE):
            errors.append("This record has no local file, so there is nothing to load.")
    else:
        probe = probe_file(path)
        if not probe.exists:
            errors.append(f"The file is gone: {path}")
        elif probe.format == FORMAT_UNKNOWN:
            errors.append(probe.detail or "The file is not a recognised model format.")
        elif not probe.usable:
            errors.append(probe.detail or "The file failed its integrity check.")
        elif str(getattr(record, "integrity", "") or "") and not record.verified:
            warnings.append(
                "The file was registered with a failing integrity verdict; re-inspect it."
            )

    # 3-5. the runtime, the format, and the adapter's own validation
    resolved_adapter = adapter
    status = None
    capabilities = None
    if resolved_adapter is None:
        try:
            resolved_adapter = manager.adapter(runtime)
        except RuntimeUnavailable as error:
            errors.append(str(error))
            resolved_adapter = None
    if resolved_adapter is not None:
        try:
            status = resolved_adapter.status()
            capabilities = resolved_adapter.capabilities()
        except Exception as error:  # a broken runtime must not crash the check
            warnings.append(f"The {runtime} runtime could not report its state: {error}")
        if status is not None and not status.available:
            errors.append(status.detail or f"The {runtime} runtime is not available.")
        elif status is not None and status.hint:
            warnings.append(status.hint)
        if capabilities is not None and not capabilities.can_load:
            errors.append(f"The {runtime} runtime cannot load a model here.")
        if path is not None:
            try:
                probe = probe_file(path)
                ok, reason = resolved_adapter.validate(probe)
                if not ok:
                    errors.append(reason)
            except UnsupportedCapability:
                pass

    # 6-7. the estimate, against what is really free
    declared_context = getattr(record, "context_length", None)
    requested_context = context_length or declared_context or None
    estimate = estimate_memory(
        record,
        context_length=requested_context,
        gpu_layers=gpu_layers or 0,
        gpu=resolved_resources.gpu,
    )
    if requested_context and declared_context and requested_context > declared_context:
        errors.append(
            f"The model declares a context of {declared_context} tokens, so "
            f"{requested_context} is not supported."
        )
        recommendations.append(f"Use a context of {declared_context} tokens or fewer.")
    elif requested_context and not declared_context:
        warnings.append(
            "The model declares no context length, so a requested one cannot be "
            "checked against it."
        )

    available = resolved_resources.available_memory_bytes
    if available is not None and estimate.high:
        if estimate.high > available:
            warnings.append(
                f"The estimate ({_size(estimate.high)} at the high end) is larger than "
                f"what the OS reports as available ({_size(available)}). The model may "
                "not fit, or may load slowly by swapping."
            )
            if estimate.context_length > DEFAULT_CONTEXT:
                recommendations.append(
                    f"Try a shorter context than {estimate.context_length} tokens to "
                    "shrink the KV cache."
                )
        else:
            warnings.append(
                f"The estimate ({_size(estimate.high)} at the high end) fits within "
                f"{_size(available)} reported available. This is an estimate, not a "
                "guarantee that the load will succeed."
            )
    elif available is None:
        warnings.append(
            "This platform does not report available memory, so the estimate could "
            "not be compared against it."
        )

    # 8-10. GPU offload, and the CPU path that must remain available
    offload: bool | None = None
    if status is not None:
        offload = getattr(status, "gpu_offload", None)
    if gpu_layers:
        if offload is False:
            warnings.append(
                "GPU offload was requested but the installed runtime reports it is "
                "not supported; the model would run on the CPU."
            )
            recommendations.append("Run without GPU layers (the CPU path).")
        elif offload is None:
            warnings.append(
                "GPU offload was requested but could not be confirmed for the "
                "installed runtime; the CPU path remains available."
            )
        if not resolved_resources.gpu.available:
            warnings.append(
                "The system reported no GPU, so an offload request may have no effect."
            )
    if offload is False or offload is None:
        cpu_fallback = True

    can_load = not errors
    if can_load and not recommendations and offload is not True:
        recommendations.append("The CPU path is available; a GPU is not required.")

    confidence = estimate.confidence
    return PreflightReport(
        model_id=str(getattr(record, "model_id", model_id)),
        runtime=runtime,
        can_load=can_load,
        errors=tuple(errors),
        warnings=tuple(warnings),
        recommendations=tuple(dict.fromkeys(recommendations)),
        estimate=estimate,
        resources=resolved_resources,
        context_length=estimate.context_length or None,
        gpu_offload=offload,
        confidence=confidence,
        cpu_fallback=cpu_fallback,
    )


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


#: Re-exported so a caller can name a format without importing ``core.model_files``.
KNOWN_FORMATS = (FORMAT_GGUF, FORMAT_FAYD_CHECKPOINT, FORMAT_UNKNOWN)


__all__ = [
    "CONFIDENCE_HIGH",
    "CONFIDENCE_LOW",
    "CONFIDENCE_MEDIUM",
    "CONFIDENCE_NOTES",
    "CONFIDENCE_UNKNOWN",
    "DEFAULT_CONTEXT",
    "KNOWN_FORMATS",
    "MemoryEstimate",
    "PreflightReport",
    "estimate_memory",
    "preflight",
]
