"""Model compatibility for Phase 6: can this base model actually be fine-tuned?

The task is explicit that a GGUF inference file is **not** a training base and that
CUDA must not be assumed. This module is the honest answer: it inspects a model
record, decides which training methods are *possible*, and reports what is missing
rather than promising a run that cannot start.

A training method is a closed set:

* ``lora_inprocess`` -- a LoRA adapter over the **in-project** numpy transformer.
  This is the one method that needs no extra dependency and runs on a CPU, so it
  is the default and the fallback everywhere.
* ``lora_transformers`` -- a LoRA adapter over a Hugging Face ``transformers``
  model, when the ``[training]`` extra (torch/transformers/peft) is installed. It
  needs the **unquantised** safetensors weights, not a GGUF.
* ``qlora`` -- a 4-bit quantised LoRA. It needs a CUDA GPU (``bitsandbytes``), and
  this module refuses it on the stated hardware rather than trying.

Nothing here loads a model. It reads the facts a :class:`~core.model_registry.ModelRecord`
already carries (format, runtime, size, architecture) and, when a file is present,
its header.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from core.model_registry import FORMAT_FAYD_CHECKPOINT, FORMAT_GGUF
from core.optional_deps import FEATURE_TRAINING
from core.optional_deps import status as feature_status

# ---------------------------------------------------------------- the methods

METHOD_LORA_INPROCESS = "lora_inprocess"
METHOD_LORA_TRANSFORMERS = "lora_transformers"
METHOD_QLORA = "qlora"

METHODS: tuple[str, ...] = (
    METHOD_LORA_INPROCESS,
    METHOD_LORA_TRANSFORMERS,
    METHOD_QLORA,
)

METHOD_LABELS: dict[str, str] = {
    METHOD_LORA_INPROCESS: "LoRA over the in-project transformer (CPU, no extra)",
    METHOD_LORA_TRANSFORMERS: "LoRA over a transformers model (needs the [training] extra)",
    METHOD_QLORA: "QLoRA 4-bit (needs a CUDA GPU)",
}

#: A model must fit in RAM with room for the optimizer state to train in-process.
#: Below this many parameters the in-project path is comfortable on 16 GB.
INPROCESS_MAX_PARAMETERS = 60_000_000

#: The unquantised transformers path is realistic on 16 GB up to roughly this size.
TRANSFORMERS_MAX_PARAMETERS = 3_000_000_000


@dataclass(frozen=True)
class MethodSupport:
    """Whether one training method is possible, and the real next step if not."""

    method: str
    supported: bool
    reason: str = ""
    next_step: str = ""

    @property
    def label(self) -> str:
        return METHOD_LABELS.get(self.method, self.method)

    def describe(self) -> str:
        state = "supported" if self.supported else "unsupported"
        text = f"{self.label}: {state}"
        if self.reason:
            text += f" -- {self.reason}"
        if not self.supported and self.next_step:
            text += f" (next step: {self.next_step})"
        return text


@dataclass(frozen=True)
class CompatibilityVerdict:
    """The compatibility validator's answer for one base model."""

    model_id: str
    model_format: str
    trainable: bool
    methods: tuple[MethodSupport, ...] = ()
    preferred: str = ""
    warnings: tuple[str, ...] = ()
    unknowns: tuple[str, ...] = ()
    parameter_count: int | None = None

    def support_for(self, method: str) -> MethodSupport | None:
        for item in self.methods:
            if item.method == method:
                return item
        return None

    @property
    def supported_methods(self) -> tuple[str, ...]:
        return tuple(m.method for m in self.methods if m.supported)

    def describe(self) -> str:
        lines = [
            f"Compatibility for {self.model_id} [{self.model_format}]",
            f"  trainable: {'yes' if self.trainable else 'no'}"
            + (f" · preferred method: {self.preferred}" if self.preferred else ""),
        ]
        lines.extend(f"  {item.describe()}" for item in self.methods)
        lines.extend(f"  warning: {w}" for w in self.warnings)
        lines.extend(f"  unknown: {u}" for u in self.unknowns)
        return "\n".join(lines)


def _parameter_count(record: Any) -> int | None:
    value = getattr(record, "parameter_count", None)
    if value:
        return int(value)
    # A live in-project transformer exposes its parameters directly.
    parameters = getattr(record, "parameters", None)
    if callable(parameters):
        try:
            return int(sum(int(p.size) for p in parameters()))
        except Exception:  # pragma: no cover - defensive
            return None
    return None


def _is_live_transformer(record: Any) -> bool:
    """True when ``record`` is a live in-project transformer rather than a file record."""
    return callable(getattr(record, "parameters", None)) and hasattr(record, "config")


def _training_extra_available() -> bool:
    try:
        return feature_status(FEATURE_TRAINING).python_ok
    except Exception:  # pragma: no cover - defensive
        return False


def _gpu_available() -> bool:
    try:
        from core.model_resources import system_resources

        gpu = system_resources().gpu
        return bool(getattr(gpu, "available", False))
    except Exception:  # pragma: no cover - defensive
        return False


def validate_compatibility(record: Any, *, method: str = "") -> CompatibilityVerdict:
    """Decide which training methods are possible for one model record.

    The rules, in order:

    * the **in-project** method is always offered -- it needs no extra and runs on
      a CPU -- but only when the record is the project's own checkpoint or a model
      small enough to train in process; a GGUF file is *not* an in-project base;
    * the **transformers** method needs the extra *and* unquantised weights, so a
      GGUF is refused with the conversion note rather than accepted;
    * **QLoRA** needs a CUDA GPU; on this hardware it is refused by name.
    """
    model_format = str(getattr(record, "model_format", "") or "")
    parameters = _parameter_count(record)
    model_id = str(
        getattr(record, "model_id", "")
        or getattr(record, "identity", "")
        or ("fayd-inprocess-model" if _is_live_transformer(record) else "model")
    )
    quantization = str(getattr(record, "quantization", "") or "")
    warnings: list[str] = []
    unknowns: list[str] = []

    is_checkpoint = model_format == FORMAT_FAYD_CHECKPOINT
    is_gguf = model_format == FORMAT_GGUF
    live_transformer = _is_live_transformer(record)

    # -- method 1: in-process LoRA ---------------------------------------
    inprocess_ok = False
    inprocess_reason = ""
    inprocess_next = ""
    if live_transformer:
        inprocess_ok = True
        inprocess_reason = (
            "a live in-project transformer trains in process"
            + (f" ({parameters:,} parameters)" if parameters is not None else "")
        )
    elif is_checkpoint:
        inprocess_ok = True
        inprocess_reason = "the project's own checkpoint trains in process"
    elif is_gguf:
        inprocess_reason = "a GGUF is a quantised inference file, not a training base"
        inprocess_next = "train the project checkpoint, or install the [training] extra"
    elif parameters is None:
        inprocess_reason = "the parameter count is unknown, so the size cannot be checked"
        inprocess_next = "inspect the model file first (python -m cli models inspect)"
        unknowns.append("the parameter count")
    elif parameters <= INPROCESS_MAX_PARAMETERS:
        inprocess_ok = True
        inprocess_reason = f"{parameters:,} parameters fit the in-process trainer"
    else:
        inprocess_reason = (
            f"{parameters:,} parameters exceed the in-process trainer's ceiling "
            f"({INPROCESS_MAX_PARAMETERS:,})"
        )
        inprocess_next = "use the [training] extra, or a smaller base model"

    # -- method 2: transformers LoRA -------------------------------------
    extra = _training_extra_available()
    transformers_ok = False
    transformers_reason = ""
    transformers_next = ""
    if not extra:
        transformers_reason = "the [training] extra (torch/transformers/peft) is not installed"
        transformers_next = "python -m pip install -r requirements-training.txt"
    elif is_gguf:
        transformers_reason = (
            "a GGUF is quantised; transformers LoRA needs the unquantised safetensors"
        )
        transformers_next = (
            "download the base model's safetensors weights (a conversion is required)"
        )
    elif parameters is not None and parameters > TRANSFORMERS_MAX_PARAMETERS:
        transformers_reason = (
            f"{parameters:,} parameters are above the {TRANSFORMERS_MAX_PARAMETERS:,} "
            "ceiling for 16 GB of RAM"
        )
        transformers_next = "choose a smaller base model or a machine with more RAM"
    else:
        transformers_ok = True
        transformers_reason = "the extra is installed and the weights are usable"

    # -- method 3: QLoRA --------------------------------------------------
    gpu = _gpu_available()
    qlora_ok = False
    qlora_reason = ""
    qlora_next = ""
    if not gpu:
        qlora_reason = (
            "no CUDA-capable GPU is reported; the stated AMD/Intel GPUs do not "
            "support modern CUDA/ROCm acceleration"
        )
        qlora_next = "train on a machine with a CUDA GPU, or use CPU LoRA"
    elif not extra:
        qlora_reason = "QLoRA needs the [training] extra (torch/transformers/peft/bitsandbytes)"
        qlora_next = "python -m pip install -r requirements-training.txt"
    else:
        qlora_ok = True
        qlora_reason = "a GPU and the extra are present"

    methods = (
        MethodSupport(METHOD_LORA_INPROCESS, inprocess_ok, inprocess_reason, inprocess_next),
        MethodSupport(
            METHOD_LORA_TRANSFORMERS, transformers_ok, transformers_reason, transformers_next
        ),
        MethodSupport(METHOD_QLORA, qlora_ok, qlora_reason, qlora_next),
    )
    supported = tuple(m.method for m in methods if m.supported)
    preferred = ""
    if method:
        chosen = next((m for m in methods if m.method == method), None)
        if chosen is None:
            warnings.append(f"unknown training method {method!r}")
        elif not chosen.supported:
            warnings.append(f"the requested method {method!r} is not supported here")
        else:
            preferred = method
    if not preferred and supported:
        preferred = supported[0]

    if is_gguf:
        warnings.append(
            "A GGUF file is an inference artifact; the original file is never "
            "modified and no adapter is written into it."
        )
    if quantization and not is_gguf:
        warnings.append(f"the record reports quantisation {quantization!r}")

    return CompatibilityVerdict(
        model_id=model_id,
        model_format=model_format or "unknown",
        trainable=bool(supported),
        methods=methods,
        preferred=preferred,
        warnings=tuple(warnings),
        unknowns=tuple(unknowns),
        parameter_count=parameters,
    )


__all__ = [
    "INPROCESS_MAX_PARAMETERS",
    "METHODS",
    "METHOD_LABELS",
    "METHOD_LORA_INPROCESS",
    "METHOD_LORA_TRANSFORMERS",
    "METHOD_QLORA",
    "TRANSFORMERS_MAX_PARAMETERS",
    "CompatibilityVerdict",
    "MethodSupport",
    "validate_compatibility",
]
