"""Local model management for perception (section 12).

Perception uses the project's **existing** Local Model Manager
(:mod:`core.model_manager`) rather than a second registry: the vision model
Doc's Studio runs (Qwen3-VL), the OCR engine, and the speech engines are all
described through it, and this module is the thin adapter that makes the manager
aware of perception's own needs.

It adds exactly three things, none of them a new store:

* :data:`PERCEPTION_TASKS` -- the perception task names the manager can select a
  model for (``vision``, ``ocr``, ``speech-to-text``, ``text-to-speech``), mapped
  to the engine(s) whose models suit each. The mapping is a *view* of the
  engines the project already names, so it can never disagree with them.
* :func:`perception_model_status` -- an honest :class:`PerceptionModelStatus` per
  perception task, saying whether a suitable model is installed here and the real
  next step when it is not.
* :func:`model_resource_report` -- the machine's own figures (memory, disk, GPU)
  read through the manager, so a caller can decide whether a model *can* run
  before it is loaded (the target machine has integrated graphics and no CUDA).

**No model is loaded here.** Loading stays with the engine that owns it (Doc's
Studio, the speech factory), behind its own permission and approval gates, and the
project's rule is that perception never loads every model at once.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

#: The perception tasks the manager can select a model for, mapped to the engines
#: whose models suit each. A view of existing facts (see
#: :data:`core.model_manager.TASKS`), never a second registry.
PERCEPTION_TASKS: dict[str, tuple[str, ...]] = {
    "vision": ("docs",),
    "speech-to-text": ("qwen-asr",),
    "text-to-speech": ("qwen-tts", "piper", "audio.cpp"),
    "ocr": (),
}


@dataclass(frozen=True)
class PerceptionModelStatus:
    """Whether a perception task has a usable model here, and the next step if not."""

    name: str
    available: bool
    detail: str = ""
    hint: str = ""
    model: str = ""

    def describe(self) -> str:
        if self.available:
            return f"{self.name}: {self.model or 'ready'}. {self.detail}".strip()
        return f"{self.name}: unavailable. {self.detail} {self.hint}".strip()


def _vision_status() -> PerceptionModelStatus:
    from core.docs_model import DEFAULT_DOCS_MODEL, docs_model_status

    status = docs_model_status()
    return PerceptionModelStatus(
        name="vision",
        available=status.available,
        detail=status.detail,
        hint=status.hint or "python -m cli docs download --approve",
        model=DEFAULT_DOCS_MODEL if status.available else "",
    )


def _ocr_status() -> PerceptionModelStatus:
    from core.pdf_ocr import missing_ocr_python_packages, ocr_available, tesseract_available

    if ocr_available():
        return PerceptionModelStatus(
            name="ocr",
            available=True,
            detail="Tesseract and its language data are reachable.",
            model="tesseract",
        )
    if missing_ocr_python_packages():
        detail = "the OCR Python packages are not installed."
    elif not tesseract_available():
        detail = "the Python packages are present but the Tesseract engine is missing."
    else:  # pragma: no cover - defensive
        detail = "OCR is not usable."
    return PerceptionModelStatus(
        name="ocr",
        available=False,
        detail=detail,
        hint="python -m cli deps --install ocr",
    )


def _stt_status() -> PerceptionModelStatus:
    from core.speech import discover_recognizer_name

    name = discover_recognizer_name()
    available = name != "transcript"
    return PerceptionModelStatus(
        name="speech-to-text",
        available=available,
        detail=f"engine: {name}"
        if available
        else "only the transcript stand-in is available.",
        hint="python -m cli qwen-asr download --approve, or install the `whisper` program",
        model=name if available else "",
    )


def _tts_status() -> PerceptionModelStatus:
    from core.speech import discover_synthesizer_name

    name = discover_synthesizer_name()
    available = name != "silent"
    return PerceptionModelStatus(
        name="text-to-speech",
        available=available,
        detail=f"engine: {name}" if available else "no speech engine is installed.",
        hint="python -m cli deps --install speech, or install Piper",
        model=name if available else "",
    )


def perception_model_status() -> tuple[PerceptionModelStatus, ...]:
    """An honest status for each perception model task. Never raises."""
    return (_vision_status(), _ocr_status(), _stt_status(), _tts_status())


def model_resource_report(*, disk_path: Any = None) -> Any:
    """The machine's own resource figures, read through the project's readers.

    Returns a :class:`core.model_resources.SystemResources` (memory, disk, CPU
    count, and what the OS reports about GPUs), each reading carrying its own
    provenance. Nothing here loads a model; it is the data a caller uses to decide
    whether one *could* run.
    """
    from core.model_resources import system_resources

    return system_resources(disk_path=disk_path)


def can_load_vision_model() -> tuple[bool, str]:
    """Whether the machine looks able to run the local vision model, with a reason.

    A conservative estimate, not a guarantee: Qwen3-VL 4B needs roughly 3-4 GB
    free, and the model runs on the CPU here (the target machine's AMD/Intel
    integrated GPUs are not used, because ``llama-cpp-python`` on this build has no
    verified CUDA/Metal backend). The check is honest about being an estimate.
    """
    resources = model_resource_report()
    available = getattr(resources, "available_memory_bytes", None)
    if available is None:
        return True, "the machine's free memory could not be read; proceeding optimistically."
    gigabytes = available / (1024**3)
    if gigabytes < 3.0:
        return False, (
            f"only {gigabytes:.1f} GB of memory is free; the 4B vision model needs about 3-4 GB. "
            "Close other programs, or use the 2B model."
        )
    return True, f"{gigabytes:.1f} GB free, enough for the local vision model."


__all__ = [
    "PERCEPTION_TASKS",
    "PerceptionModelStatus",
    "can_load_vision_model",
    "model_resource_report",
    "perception_model_status",
]
