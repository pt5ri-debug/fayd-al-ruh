"""Install the optional extras (PDF, OCR) from within the application.

The base install stays dependency-light: reading a *digital* PDF and reading a
*scanned* PDF with OCR are separate, optional paths, and the reader and the OCR
engine report clearly when their packages are absent. Those messages used to
point only at ``pip install fayd-al-ruh[ocr]``, which cannot work for a source
checkout -- the project is not published on PyPI, so that command fails with
"no matching distribution". A user who reached the message from the window had
no way to act on it.

This module gives the application a real next step. It knows, for each optional
feature, the bundled requirements file, the import names to check, and the pip
command that actually works from a source checkout (``python -m pip install -r
requirements-<feature>.txt``). :func:`install` runs that command, and the window
and the CLI offer it when a feature is missing.

Design points, matching the rest of the project:

* **Dependency-free and import-safe.** Only the standard library is imported at
  module load, so ``cli`` never pulls in an optional package.
* **Honest about two kinds of missing piece.** A feature can be missing its
  *Python packages* (pip can fix that) or its *system* package (pip cannot --
  Tesseract and its language data are OS packages). :func:`status` separates the
  two so the app never promises an install it cannot perform.
* **Injection for tests.** :func:`install` takes an optional ``runner``, so the
  command construction and the result handling are tested without touching the
  network or the real pip.
* **A frozen build cannot self-install.** A PyInstaller bundle has no pip and no
  source requirements file; :func:`install` reports that plainly instead of
  running a command that would fail.
"""

from __future__ import annotations

import importlib.util
import subprocess
import sys
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

from core.paths import bundle_root, is_frozen

#: Feature keys used across the app.
FEATURE_PDF = "pdf"
FEATURE_OCR = "ocr"
FEATURE_IMAGE = "image"
FEATURE_ULTRON = "ultron"
FEATURE_CODER = "coder"
FEATURE_VISION = "vision"
FEATURE_PPTX = "pptx"
FEATURE_PRESENTATION = "presentation"
FEATURE_BANNER = "banner"
FEATURE_QWEN_ASR = "qwen-asr"
FEATURE_QWEN_TTS = "qwen-tts"


@dataclass(frozen=True)
class OptionalFeature:
    """One optional capability, and how to install its Python packages."""

    key: str
    title: str
    summary: str
    #: The bundled requirements file that installs the Python packages.
    requirements_file: str
    #: Import names checked to decide whether the Python packages are present.
    import_names: tuple[str, ...]
    #: The system package(s) pip cannot install ("" when there are none).
    system_note: str = ""
    #: How to phrase "installed, but one more step is needed" for this feature.
    system_missing_note: str = "the system engine is not available."


FEATURES: dict[str, OptionalFeature] = {
    FEATURE_PDF: OptionalFeature(
        key=FEATURE_PDF,
        title="PDF support",
        summary="Read the text layer of a digital PDF.",
        requirements_file="requirements-pdf.txt",
        import_names=("pypdf",),
    ),
    FEATURE_OCR: OptionalFeature(
        key=FEATURE_OCR,
        title="OCR support",
        summary="Read a scanned PDF (an image with no text layer).",
        requirements_file="requirements-ocr.txt",
        import_names=("pypdfium2", "pytesseract", "PIL"),
        system_note=(
            "The Tesseract engine and its language data are separate system packages "
            "(e.g. `apt install tesseract-ocr tesseract-ocr-eng tesseract-ocr-ara`)."
        ),
    ),
    FEATURE_IMAGE: OptionalFeature(
        key=FEATURE_IMAGE,
        title="Learned image backend",
        summary="Generate a photorealistic image with a real diffusion model (instead of the renderer).",
        requirements_file="requirements-image.txt",
        import_names=("torch", "diffusers"),
        system_note=(
            "The model weights are a separate one-time, human-approved download: "
            "run `python -m cli image download --approve` (~4 GB)."
        ),
        system_missing_note="the model weights are not cached yet.",
    ),
    FEATURE_ULTRON: OptionalFeature(
        key=FEATURE_ULTRON,
        title="Ultron (general local model)",
        summary="Talk with a general, advanced local model on any subject (Ultron).",
        requirements_file="requirements-ultron.txt",
        import_names=("llama_cpp", "huggingface_hub"),
        system_note=(
            "The model weights are a separate one-time, human-approved download: "
            "run `python -m cli ultron download --approve` (~5 GB), or drop a `.gguf` "
            "file into the data/ultron folder."
        ),
        system_missing_note="no `.gguf` model file is present yet.",
    ),
    FEATURE_CODER: OptionalFeature(
        key=FEATURE_CODER,
        title="Code Studio coding model (in-process)",
        summary="Let the Code Studio write code with a local model, with no Ollama server running.",
        requirements_file="requirements-coder.txt",
        import_names=("llama_cpp", "huggingface_hub"),
        system_note=(
            "The model weights are a separate one-time, human-approved download: "
            "run `python -m cli studio download --approve` (~5 GB), or drop a `.gguf` "
            "file into the data/coder folder."
        ),
        system_missing_note="no `.gguf` coding model file is present yet.",
    ),
    FEATURE_VISION: OptionalFeature(
        key=FEATURE_VISION,
        title="Vision (petroleum local model)",
        summary="Ask a petroleum-engineering specialist local model (Vision).",
        requirements_file="requirements-vision.txt",
        import_names=("llama_cpp", "huggingface_hub"),
        system_note=(
            "The model weights are a separate one-time, human-approved download: "
            "run `python -m cli vision download --approve` (~5 GB), or drop a `.gguf` "
            "file into the data/vision folder."
        ),
        system_missing_note="no `.gguf` model file is present yet.",
    ),
    FEATURE_PPTX: OptionalFeature(
        key=FEATURE_PPTX,
        title="PowerPoint export engine (python-pptx)",
        summary=(
            "Export a Presentation Studio deck as a real, editable .pptx with "
            "tables, embedded images, charts, notes, and right-to-left text."
        ),
        requirements_file="requirements-pptx.txt",
        import_names=("pptx",),
    ),
    FEATURE_PRESENTATION: OptionalFeature(
        key=FEATURE_PRESENTATION,
        title="Presentation Studio authoring model (in-process)",
        summary=(
            "Draft a presentation's outline and slide content with a local Qwen3 "
            "model, with no Ollama server running."
        ),
        requirements_file="requirements-presentation.txt",
        import_names=("llama_cpp", "huggingface_hub"),
        system_note=(
            "The model weights are a separate one-time, human-approved download: "
            "run `python -m cli presentation model-download --approve` (~5 GB), or "
            "drop a `.gguf` file into the data/presentation folder."
        ),
        system_missing_note="no `.gguf` presentation model file is present yet.",
    ),
    FEATURE_BANNER: OptionalFeature(
        key=FEATURE_BANNER,
        title="Banner, the presentation local model (in-process)",
        summary=(
            "Answer from Banner, the Presentation Studio's own local Qwen3 model, "
            "with no Ollama server running. It shares the data/presentation folder "
            "and the [presentation] runtime with the Studio's authoring model."
        ),
        requirements_file="requirements-banner.txt",
        import_names=("llama_cpp", "huggingface_hub"),
        system_note=(
            "The model weights are a separate one-time, human-approved download: "
            "run `python -m cli banner download --approve` (~5 GB), or drop a "
            "`.gguf` file into the data/presentation folder."
        ),
        system_missing_note="no `.gguf` presentation model file is present yet.",
    ),
    FEATURE_QWEN_ASR: OptionalFeature(
        key=FEATURE_QWEN_ASR,
        title="Qwen3-ASR speech recognition",
        summary=(
            "Transcribe speech with the local Qwen3-ASR model (52 languages, "
            "including Arabic) instead of the system `whisper` engine."
        ),
        requirements_file="requirements-qwen-speech.txt",
        import_names=("torch", "qwen_asr"),
        system_note=(
            "The model weights are a separate one-time, human-approved download: "
            "run `python -m cli qwen-asr download --approve` (~3.4 GB), or drop a "
            "model folder into the data/qwen-speech folder."
        ),
        system_missing_note="no Qwen3-ASR model folder is present yet.",
    ),
    FEATURE_QWEN_TTS: OptionalFeature(
        key=FEATURE_QWEN_TTS,
        title="Qwen3-TTS speech synthesis",
        summary=(
            "Read chat text aloud with the local Qwen3-TTS model (expressive, ten "
            "languages; it does not speak Arabic)."
        ),
        requirements_file="requirements-qwen-speech.txt",
        import_names=("torch", "qwen_tts"),
        system_note=(
            "The model weights are a separate one-time, human-approved download: "
            "run `python -m cli qwen-tts download --approve` (~3.4 GB), or drop a "
            "model folder into the data/qwen-speech folder. Qwen3-TTS does not "
            "speak Arabic; the system engine still reads Arabic aloud."
        ),
        system_missing_note="no Qwen3-TTS model folder is present yet.",
    ),
}


def feature(key: str) -> OptionalFeature:
    """The :class:`OptionalFeature` for ``key``; raises ``KeyError`` if unknown."""
    return FEATURES[key]


@dataclass(frozen=True)
class FeatureStatus:
    """What is present and what is missing for one optional feature."""

    key: str
    python_ok: bool
    """The Python packages import (pip can install these when ``False``)."""

    system_ok: bool
    """The system package is usable (pip cannot install this when ``False``)."""

    missing_python: tuple[str, ...] = ()
    """The import names that are absent."""

    def __str__(self) -> str:
        spec = FEATURES[self.key]
        if self.python_ok and self.system_ok:
            return f"{spec.title}: available."
        if not self.python_ok:
            return (
                f"{spec.title}: the Python packages {', '.join(self.missing_python)} "
                "are not installed."
            )
        return f"{spec.title}: installed, but {spec.system_missing_note}"

    @property
    def available(self) -> bool:
        return self.python_ok and self.system_ok


def missing_imports(key: str) -> tuple[str, ...]:
    """The feature's import names that cannot be found."""
    spec = feature(key)
    return tuple(name for name in spec.import_names if importlib.util.find_spec(name) is None)


def _ocr_system_ok() -> bool:
    """Whether the Tesseract engine is reachable (only meaningful with the packages)."""
    try:
        import pytesseract  # type: ignore[import-not-found]

        pytesseract.get_tesseract_version()
    except Exception:
        return False
    return True


def _image_weights_cached() -> bool:
    """Whether the learned image model's weights are already on disk.

    The extra installs the packages; the weights are a separate one-time
    download, so this is the second half of "usable right now".
    """
    from core.imagegen_diffusion import DEFAULT_MODEL, model_is_cached

    return model_is_cached(DEFAULT_MODEL)


def _ultron_model_cached() -> bool:
    """Whether at least one Ultron model file (a ``.gguf``) is already on disk.

    As with the image backend, the extra installs the runtime; the model file is
    a separate one-time download (or a file the user drops in), so this is the
    second half of "usable right now".
    """
    from core.ultron import model_is_cached

    return model_is_cached()


def _coder_model_cached() -> bool:
    """Whether at least one coding model file (a ``.gguf``) is already on disk.

    The extra installs the runtime; the model file is a separate one-time
    download (or a file the user drops in), so this is the second half of
    "usable right now".
    """
    from core.coder_llama import model_is_cached

    return model_is_cached()


def _vision_model_cached() -> bool:
    """Whether at least one Vision model file (a ``.gguf``) is already on disk.

    Vision shares Ultron's ``[ultron]`` runtime extra; the model file is a
    separate one-time download (or a file the user drops in), so this is the
    second half of "usable right now".
    """
    from core.vision_model import model_is_cached

    return model_is_cached()


def _banner_model_cached() -> bool:
    """Whether at least one Banner model file (a ``.gguf``) is already on disk.

    Banner shares the Presentation Studio's ``data/presentation`` folder, so a
    model the Studio already fetched counts here too; the extra installs the
    runtime and the file is the second half of "usable right now".
    """
    from core.banner import model_is_cached

    return model_is_cached()


def _qwen_speech_model_cached(kind: str) -> bool:
    """Whether a Qwen3 speech model (``"asr"`` or ``"tts"``) is already on disk.

    Both families live under ``data/qwen-speech`` and are keyed by a registry; the
    extra installs the runtime and the model folder is the second half of "usable
    right now".
    """
    from core.qwen_speech import model_is_cached

    return model_is_cached(kind)


def status(key: str) -> FeatureStatus:
    """Check one optional feature without importing its heavy packages eagerly."""
    missing = missing_imports(key)
    python_ok = not missing
    if key == FEATURE_OCR:
        system_ok = _ocr_system_ok() if python_ok else False
    elif key == FEATURE_IMAGE:
        system_ok = _image_weights_cached() if python_ok else False
    elif key == FEATURE_ULTRON:
        system_ok = _ultron_model_cached() if python_ok else False
    elif key == FEATURE_CODER:
        system_ok = _coder_model_cached() if python_ok else False
    elif key == FEATURE_VISION:
        system_ok = _vision_model_cached() if python_ok else False
    elif key == FEATURE_BANNER:
        system_ok = _banner_model_cached() if python_ok else False
    elif key == FEATURE_QWEN_ASR:
        system_ok = _qwen_speech_model_cached("asr") if python_ok else False
    elif key == FEATURE_QWEN_TTS:
        system_ok = _qwen_speech_model_cached("tts") if python_ok else False
    else:
        system_ok = True
    return FeatureStatus(key=key, python_ok=python_ok, system_ok=system_ok, missing_python=missing)


def is_available(key: str) -> bool:
    """Whether an optional feature is fully usable right now."""
    return status(key).available


def requirements_path(key: str) -> Path:
    """The bundled requirements file for ``key`` (may not exist in a frozen build)."""
    return bundle_root() / feature(key).requirements_file


def pip_install_command(key: str) -> str:
    """A short, copy-pasteable pip command for ``key`` (not the full interpreter path)."""
    return f"python -m pip install -r {feature(key).requirements_file}"


def install_command(key: str) -> tuple[str, ...] | None:
    """The pip command that installs ``key``'s packages, or ``None`` if unavailable.

    ``None`` means the app cannot install the feature itself: either this is a
    frozen bundle (no pip, no source requirements file) or the requirements file
    is missing. The caller should show :func:`install_hint` instead.
    """
    if is_frozen():
        return None
    path = requirements_path(key)
    if not path.exists():
        return None
    return (sys.executable, "-m", "pip", "install", "-r", str(path))


def install_hint(key: str) -> str:
    """A short, *working* instruction for installing ``key`` from this checkout."""
    spec = feature(key)
    if is_frozen():
        return (
            f"{spec.title} is not bundled. Install it with "
            f"`python -m pip install -r {spec.requirements_file}` in a source checkout"
            + (f". {spec.system_note}" if spec.system_note else ".")
        )
    lines = [
        f"Install {spec.title.lower()} with:",
        f"    python -m pip install -r {spec.requirements_file}",
    ]
    if spec.system_note:
        lines.append(spec.system_note)
    return "\n".join(lines)


@dataclass
class InstallResult:
    """The outcome of an attempted install, ready to show to the user."""

    key: str
    ok: bool
    command: tuple[str, ...] = ()
    returncode: int | None = None
    output: str = ""
    message: str = ""
    ran: bool = True
    """False when no command was run (a frozen build, or a missing requirements file)."""

    @property
    def title(self) -> str:
        return feature(self.key).title


#: Runs one command and returns ``(returncode, combined_output)``.
CommandRunner = Callable[[Sequence[str]], tuple[int, str]]


def _default_runner(argv: Sequence[str]) -> tuple[int, str]:
    completed = subprocess.run(
        list(argv),
        capture_output=True,
        text=True,
        check=False,
    )
    combined = (completed.stdout or "") + (completed.stderr or "")
    return completed.returncode, combined


def install(
    key: str,
    *,
    runner: CommandRunner | None = None,
    progress: Callable[[str], None] | None = None,
) -> InstallResult:
    """Install ``key``'s Python packages with pip. Never raises for a pip failure.

    Returns an :class:`InstallResult`; ``ok`` is True only when pip succeeded and
    the feature's Python packages are now importable. When the app cannot install
    (a frozen build, or no requirements file) ``ran`` is False and ``message``
    explains what to do instead.
    """
    spec = feature(key)
    command = install_command(key)
    if command is None:
        return InstallResult(
            key=key,
            ok=False,
            ran=False,
            message=install_hint(key),
        )
    if progress is not None:
        progress(f"Installing {spec.title.lower()}…")
    run = runner or _default_runner
    try:
        returncode, output = run(command)
    except OSError as error:  # pip could not be launched at all
        return InstallResult(
            key=key,
            ok=False,
            command=command,
            message=f"Could not run pip: {error}. {install_hint(key)}",
        )
    remaining = missing_imports(key)
    ok = returncode == 0 and not remaining
    if ok:
        message = f"{spec.title} installed."
        if key == FEATURE_OCR and not _ocr_system_ok():
            message += " " + spec.system_note
        elif key == FEATURE_IMAGE and not _image_weights_cached():
            message += " " + spec.system_note
        elif key == FEATURE_ULTRON and not _ultron_model_cached():
            message += " " + spec.system_note
        elif key == FEATURE_VISION and not _vision_model_cached():
            message += " " + spec.system_note
    elif returncode != 0:
        tail = output.strip().splitlines()[-1] if output.strip() else "pip failed."
        message = f"Could not install {spec.title.lower()}: {tail}"
    else:
        message = (
            f"pip finished but {', '.join(remaining)} still cannot be imported. "
            + install_hint(key)
        )
    return InstallResult(
        key=key,
        ok=ok,
        command=command,
        returncode=returncode,
        output=output,
        message=message,
    )


__all__ = [
    "FEATURE_BANNER",
    "FEATURE_IMAGE",
    "FEATURE_OCR",
    "FEATURE_PDF",
    "FEATURE_ULTRON",
    "FEATURE_VISION",
    "FEATURES",
    "FeatureStatus",
    "InstallResult",
    "OptionalFeature",
    "feature",
    "install",
    "install_command",
    "install_hint",
    "is_available",
    "missing_imports",
    "pip_install_command",
    "requirements_path",
    "status",
]
