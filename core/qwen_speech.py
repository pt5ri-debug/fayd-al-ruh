"""Qwen3 speech engines: local ASR and TTS for every chat interface.

The project already speaks through *system* programs -- ``espeak-ng`` for
text-to-speech, ``whisper`` for speech-to-text (see :mod:`core.speech`). Those
need no Python package and no download, which is why they are the default, but
they are also generic: ``espeak-ng`` is robotic and ``whisper`` is English-first.

This module adds the two **Qwen3** speech models as a real, local alternative,
the way Ultron and Vision add a real local *language* model:

* :class:`QwenAsrRecognizer` -- speech recognition with **Qwen3-ASR** (1.7B or
  0.6B), which covers 52 languages **including Arabic**, so an Arabic question is
  transcribed correctly where ``whisper`` would struggle.
* :class:`QwenTtsSynthesizer` -- speech synthesis with **Qwen3-TTS** (12 Hz,
  1.7B or 0.6B), expressive and natural across its ten languages.

Both are **local**: the weights are a ``transformers``/``qwen-*`` model loaded
into this process (not a GGUF through ``llama.cpp``), nothing is sent off the
machine, and the runtime is an optional extra so the base install imports neither
``torch`` nor a model file. The models are Qwen's own, Apache-2.0.

Three things stay honest, exactly as in :mod:`core.ultron` and
:mod:`core.presentation_model`:

* Each engine is a *seam*: a small runner protocol the engine delegates to, so a
  test (and a user with a different runtime) can inject a replacement and the
  engine is exercised with no ``torch`` and no weights.
* :func:`qwen_asr_status` / :func:`qwen_tts_status` report *exactly* what is
  installed -- whether the extra imports, whether the weights are on disk, and
  the real next step when they are not.
* **Qwen3-TTS does not speak Arabic.** Its ten languages are Chinese, English,
  Japanese, Korean, German, French, Russian, Portuguese, Spanish, and Italian.
  :data:`QWEN_TTS_LANGUAGES` states this, and :class:`QwenTtsSynthesizer` refuses
  Arabic with a clear message (and can be given a *fallback* engine -- the
  system one -- so Arabic still gets read aloud). Qwen3-ASR, by contrast, does
  cover Arabic. The asymmetry is reported, never hidden.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from core.human_approval import HumanApprovalGate
from core.optional_deps import FEATURE_QWEN_ASR, FEATURE_QWEN_TTS, pip_install_command
from core.paths import qwen_speech_models_dir
from core.permissions import Capability, PermissionPolicy
from core.speech import AudioClip, SpeechSynthesizer
from core.voices import detect_language

#: The registry file written after a download, naming the model to load.
REGISTRY_FILENAME = "model.json"

#: The tool name the download gate is keyed on (one per family).
DOWNLOAD_TOOL_ASR = "qwen-asr-download"
DOWNLOAD_TOOL_TTS = "qwen-tts-download"

#: The engine names the speech factories use (``--asr qwen3-asr``, ``--tts qwen3-tts``).
ASR_ENGINE_NAME = "qwen3-asr"
TTS_ENGINE_NAME = "qwen3-tts"

#: The ten languages Qwen3-TTS speaks, as its own language names. **Arabic is not
#: among them** -- that is a property of the released model, not a choice here.
QWEN_TTS_LANGUAGES: tuple[str, ...] = (
    "Chinese",
    "English",
    "Japanese",
    "Korean",
    "German",
    "French",
    "Russian",
    "Portuguese",
    "Spanish",
    "Italian",
)

#: The default CustomVoice speaker (a warm, gentle female) and a second verified
#: English one (a sunny American male); ``--tts-speaker`` may name any speaker the
#: installed model ships.
DEFAULT_TTS_SPEAKER = "Serena"
KNOWN_TTS_SPEAKERS: tuple[str, ...] = ("Serena", "Aiden")

#: Project language code -> the model's own language name.
_LANGUAGE_NAMES: dict[str, str] = {"en": "English", "ar": "Arabic"}


class QwenSpeechUnavailable(RuntimeError):
    """Raised when a Qwen3 speech engine cannot run (no extra, no weights)."""


@dataclass(frozen=True)
class QwenAsrSpec:
    """A fetchable Qwen3-ASR model."""

    key: str
    title: str
    repo: str
    params: str
    approx_size_gb: float
    summary: str


@dataclass(frozen=True)
class QwenTtsSpec:
    """A fetchable Qwen3-TTS model."""

    key: str
    title: str
    repo: str
    params: str
    approx_size_gb: float
    summary: str


#: The ASR catalog. ``qwen3-asr-1.7b`` is the default: state-of-the-art accuracy
#: among open ASR models, and it covers Arabic.
ASR_CATALOG: dict[str, QwenAsrSpec] = {
    "qwen3-asr-1.7b": QwenAsrSpec(
        key="qwen3-asr-1.7b",
        title="Qwen3-ASR 1.7B (default)",
        repo="Qwen/Qwen3-ASR-1.7B",
        params="1.7B",
        approx_size_gb=3.4,
        summary="Best accuracy, 52 languages including Arabic; ~3.4 GB.",
    ),
    "qwen3-asr-0.6b": QwenAsrSpec(
        key="qwen3-asr-0.6b",
        title="Qwen3-ASR 0.6B (light)",
        repo="Qwen/Qwen3-ASR-0.6B",
        params="0.6B",
        approx_size_gb=1.2,
        summary="Lighter and faster, still 52 languages including Arabic; ~1.2 GB.",
    ),
}

#: The TTS catalog. ``qwen3-tts-1.7b`` is the default: expressive CustomVoice.
TTS_CATALOG: dict[str, QwenTtsSpec] = {
    "qwen3-tts-1.7b": QwenTtsSpec(
        key="qwen3-tts-1.7b",
        title="Qwen3-TTS 1.7B CustomVoice (default)",
        repo="Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice",
        params="1.7B",
        approx_size_gb=3.4,
        summary="Expressive 10-language speech with built-in speakers; no Arabic.",
    ),
    "qwen3-tts-0.6b": QwenTtsSpec(
        key="qwen3-tts-0.6b",
        title="Qwen3-TTS 0.6B CustomVoice (light)",
        repo="Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice",
        params="0.6B",
        approx_size_gb=1.2,
        summary="Lighter 10-language speech with built-in speakers; no Arabic.",
    ),
}

DEFAULT_ASR_SPEC_KEY = "qwen3-asr-1.7b"
DEFAULT_TTS_SPEC_KEY = "qwen3-tts-1.7b"


def resolve_asr_spec(name: str | None) -> QwenAsrSpec:
    """The :class:`QwenAsrSpec` for ``name`` (or the default). Raises on a bad name."""
    key = (name or DEFAULT_ASR_SPEC_KEY).strip().lower()
    if key in ASR_CATALOG:
        return ASR_CATALOG[key]
    raise ValueError(f"Unknown Qwen3-ASR model {name!r}; choose one of {', '.join(ASR_CATALOG)}.")


def resolve_tts_spec(name: str | None) -> QwenTtsSpec:
    """The :class:`QwenTtsSpec` for ``name`` (or the default). Raises on a bad name."""
    key = (name or DEFAULT_TTS_SPEC_KEY).strip().lower()
    if key in TTS_CATALOG:
        return TTS_CATALOG[key]
    raise ValueError(f"Unknown Qwen3-TTS model {name!r}; choose one of {', '.join(TTS_CATALOG)}.")


def asr_spec_names() -> tuple[str, ...]:
    return tuple(ASR_CATALOG)


def tts_spec_names() -> tuple[str, ...]:
    return tuple(TTS_CATALOG)


def asr_spec_summaries() -> tuple[tuple[str, str, float, str], ...]:
    """``(key, title, approx_size_gb, summary)`` for every ASR catalog entry."""
    return tuple((s.key, s.title, s.approx_size_gb, s.summary) for s in ASR_CATALOG.values())


def tts_spec_summaries() -> tuple[tuple[str, str, float, str], ...]:
    """``(key, title, approx_size_gb, summary)`` for every TTS catalog entry."""
    return tuple((s.key, s.title, s.approx_size_gb, s.summary) for s in TTS_CATALOG.values())


def qwen_tts_supports(language: str) -> bool:
    """Whether Qwen3-TTS can speak ``language`` (a project code ``en``/``ar``).

    Arabic is deliberately **not** supported by the released Qwen3-TTS models;
    this returns ``False`` for it so a caller can fall back to a system engine
    rather than sending Arabic text to a model that cannot voice it.
    """
    return _LANGUAGE_NAMES.get(language, language) in QWEN_TTS_LANGUAGES


# --------------------------------------------------------------------- files


def _speech_dir(root: Path | str | None) -> Path:
    return Path(root) if root is not None else qwen_speech_models_dir()


def _registry_file(root: Path | str | None) -> Path:
    return _speech_dir(root) / REGISTRY_FILENAME


def _read_registry(root: Path | str | None) -> dict[str, str]:
    registry = _registry_file(root)
    if not registry.is_file():
        return {}
    try:
        data = json.loads(registry.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return {str(k): str(v) for k, v in data.items()} if isinstance(data, dict) else {}


def _write_registry(directory: Path, kind: str, spec_key: str, repo: str) -> None:
    data = _read_registry(directory)
    data[kind] = spec_key
    data[f"{kind}_repo"] = repo
    try:
        directory.mkdir(parents=True, exist_ok=True)
        _registry_file(directory).write_text(json.dumps(data, indent=2), encoding="utf-8")
    except OSError:
        pass


def find_model_dir(kind: str, root: Path | str | None = None) -> Path | None:
    """The downloaded model folder for ``kind`` (``"asr"`` or ``"tts"``), or ``None``.

    Prefers the folder the registry names (``model.json``, written by
    :func:`download_model`); otherwise a ``<kind>`` subfolder, so a user who drops
    a model in by hand is still found.
    """
    directory = _speech_dir(root)
    key = _read_registry(root).get(kind)
    if key:
        candidate = directory / key
        if (candidate / "config.json").is_file():
            return candidate
    fallback = directory / kind
    if (fallback / "config.json").is_file():
        return fallback
    return None


def model_is_cached(kind: str, root: Path | str | None = None) -> bool:
    """True when a ``kind`` model is present on disk."""
    return find_model_dir(kind, root) is not None


# ------------------------------------------------------------------- extras


def qwen_asr_extra_installed() -> bool:
    """Whether the Python runtime for Qwen3-ASR is importable."""
    import importlib.util

    return importlib.util.find_spec("torch") is not None and (
        importlib.util.find_spec("qwen_asr") is not None
        or importlib.util.find_spec("transformers") is not None
    )


def qwen_tts_extra_installed() -> bool:
    """Whether the Python runtime for Qwen3-TTS is importable."""
    import importlib.util

    return importlib.util.find_spec("torch") is not None and (
        importlib.util.find_spec("qwen_tts") is not None
    )


@dataclass(frozen=True)
class QwenSpeechStatus:
    """Exactly what is present for one Qwen3 speech engine."""

    kind: str
    extra_installed: bool
    model_cached: bool
    model_dir: Path | None
    hint: str

    @property
    def available(self) -> bool:
        return self.extra_installed and self.model_cached


def qwen_asr_status(root: Path | str | None = None) -> QwenSpeechStatus:
    """Report whether Qwen3-ASR can run here, and the real next step if not."""
    installed = qwen_asr_extra_installed()
    directory = find_model_dir("asr", root)
    if not installed:
        hint = (
            "Install the Qwen3 speech runtime with: "
            f"{pip_install_command(FEATURE_QWEN_ASR)}"
        )
    elif directory is None:
        hint = "Download the model weights with: python -m cli qwen-asr download --approve"
    else:
        hint = "Qwen3-ASR is ready."
    return QwenSpeechStatus("asr", installed, directory is not None, directory, hint)


def qwen_tts_status(root: Path | str | None = None) -> QwenSpeechStatus:
    """Report whether Qwen3-TTS can run here, and the real next step if not."""
    installed = qwen_tts_extra_installed()
    directory = find_model_dir("tts", root)
    if not installed:
        hint = (
            "Install the Qwen3 speech runtime with: "
            f"{pip_install_command(FEATURE_QWEN_TTS)}"
        )
    elif directory is None:
        hint = "Download the model weights with: python -m cli qwen-tts download --approve"
    else:
        hint = "Qwen3-TTS is ready."
    # The Arabic gap is a property of the model, not of this install, so it is
    # stated on every status -- including "not ready" -- rather than only when ready.
    return QwenSpeechStatus(
        "tts",
        installed,
        directory is not None,
        directory,
        f"{hint} Note: Qwen3-TTS does not speak Arabic; the system engine reads it aloud.",
    )


# -------------------------------------------------------------------- runners


class AsrRunner(Protocol):
    """The seam that runs ASR: raw samples in, transcript out.

    The default runner is a ``qwen-asr``/``transformers`` wrapper; a test injects
    its own, so the recognizer is exercised with no extra and no weights.
    """

    def transcribe(self, samples: Any, sample_rate: int, *, language: str | None) -> str: ...


class TtsRunner(Protocol):
    """The seam that runs TTS: text in, ``(samples, sample_rate)`` out.

    ``samples`` is a float waveform in ``[-1, 1]``; the synthesizer converts and
    resamples it into the project's 16 kHz mono :class:`~core.speech.AudioClip`.
    """

    def synthesize(
        self, text: str, *, language: str | None, speaker: str | None, instruct: str | None
    ) -> tuple[Any, int]: ...


class TransformersAsrRunner:
    """A ``Qwen3-ASR`` runner built on the ``qwen-asr`` package (lazy imports)."""

    def __init__(
        self,
        model_dir: Path | str,
        *,
        device: str = "cpu",
        max_new_tokens: int = 256,
    ) -> None:
        self.model_dir = Path(model_dir)
        self.device = device
        self.max_new_tokens = max_new_tokens
        self._model: Any = None

    def _load(self) -> Any:
        if self._model is None:
            import torch  # noqa: F401 - imported for the dtype/device side effect
            from qwen_asr import Qwen3ASRModel

            self._model = Qwen3ASRModel.from_pretrained(
                str(self.model_dir),
                device_map=self.device,
                max_new_tokens=self.max_new_tokens,
            )
        return self._model

    def transcribe(self, samples: Any, sample_rate: int, *, language: str | None) -> str:
        model = self._load()
        results = model.transcribe(audio=(samples, sample_rate), language=language)
        if not results:
            raise ValueError("Qwen3-ASR returned no result.")
        text = getattr(results[0], "text", None)
        if text is None:
            raise ValueError("Qwen3-ASR returned a result with no text.")
        return str(text)


class QwenTtsRunner:
    """A ``Qwen3-TTS`` runner built on the ``qwen-tts`` package (lazy imports)."""

    def __init__(self, model_dir: Path | str, *, device: str = "cpu") -> None:
        self.model_dir = Path(model_dir)
        self.device = device
        self._model: Any = None

    def _load(self) -> Any:
        if self._model is None:
            from qwen_tts import Qwen3TTSModel

            self._model = Qwen3TTSModel.from_pretrained(str(self.model_dir), device_map=self.device)
        return self._model

    def synthesize(
        self, text: str, *, language: str | None, speaker: str | None, instruct: str | None
    ) -> tuple[Any, int]:
        model = self._load()
        wavs, sample_rate = model.generate_custom_voice(
            text=text,
            language=language,
            speaker=speaker or DEFAULT_TTS_SPEAKER,
            instruct=instruct or "",
        )
        return wavs, int(sample_rate)


# ------------------------------------------------------------------- engines


def _to_int16(samples: Any, sample_rate: int, *, target_rate: int = 16_000) -> tuple[int, ...]:
    """Convert a float waveform to int16 PCM, resampled to ``target_rate``."""
    import numpy as np

    array = np.asarray(samples, dtype=np.float32).reshape(-1)
    if array.size == 0:
        return ()
    if sample_rate != target_rate:
        duration = array.size / sample_rate
        count = max(1, int(round(duration * target_rate)))
        source = np.arange(array.size, dtype=np.float32)
        target = np.linspace(0.0, float(array.size - 1), count, dtype=np.float32)
        array = np.interp(target, source, array).astype(np.float32)
    array = np.clip(array, -1.0, 1.0)
    return tuple(int(value) for value in (array * 32767.0).astype(np.int16))


def _to_float(samples: tuple[int, ...]) -> Any:
    """Convert int16 PCM into a float waveform in ``[-1, 1]`` for the model."""
    import numpy as np

    return np.asarray(samples, dtype=np.float32) / 32768.0


class QwenAsrRecognizer:
    """Speech recognition through Qwen3-ASR, satisfying :class:`SpeechRecognizer`.

    ``runner`` is the injection seam; when it is ``None`` the engine loads the
    model folder itself (the ``qwen-asr``/``transformers`` path). ``language``
    pins the spoken language (``"Arabic"``/``"English"``); ``None`` lets the model
    identify it, which is the model's own strength.
    """

    name = ASR_ENGINE_NAME

    def __init__(
        self,
        *,
        model_dir: Path | str | None = None,
        runner: AsrRunner | None = None,
        language: str | None = None,
        device: str = "cpu",
        root: Path | str | None = None,
    ) -> None:
        self.language = language
        self.device = device
        self._root = root
        if runner is not None:
            self._runner: AsrRunner = runner
        else:
            directory = Path(model_dir) if model_dir is not None else find_model_dir("asr", root)
            if directory is None:
                raise QwenSpeechUnavailable(
                    "No Qwen3-ASR model is present. " + qwen_asr_status(root).hint
                )
            self._runner = TransformersAsrRunner(directory, device=device)

    def transcribe(self, clip: AudioClip) -> str:
        text = self._runner.transcribe(
            _to_float(clip.samples), clip.sample_rate, language=self.language
        )
        text = text.strip()
        if not text:
            raise ValueError("Qwen3-ASR produced an empty transcript.")
        return text


class QwenTtsSynthesizer:
    """Speech synthesis through Qwen3-TTS, satisfying :class:`SpeechSynthesizer`.

    ``runner`` is the injection seam; when it is ``None`` the engine loads the
    model folder itself. ``voice``/``speaker`` pick a built-in speaker. Arabic is
    **not** a Qwen3-TTS language: when the text is Arabic this engine refuses,
    unless a ``fallback`` synthesizer (the system engine) was supplied, in which
    case the fallback reads it -- so Arabic is never silently mispronounced.
    """

    name = TTS_ENGINE_NAME

    def __init__(
        self,
        *,
        model_dir: Path | str | None = None,
        runner: TtsRunner | None = None,
        speaker: str | None = None,
        language: str | None = None,
        instruct: str | None = None,
        fallback: SpeechSynthesizer | None = None,
        device: str = "cpu",
        root: Path | str | None = None,
    ) -> None:
        self.speaker = speaker or DEFAULT_TTS_SPEAKER
        self.language = language
        self.instruct = instruct
        self.fallback = fallback
        self.device = device
        self._root = root
        if runner is not None:
            self._runner: TtsRunner = runner
        else:
            directory = Path(model_dir) if model_dir is not None else find_model_dir("tts", root)
            if directory is None:
                raise QwenSpeechUnavailable(
                    "No Qwen3-TTS model is present. " + qwen_tts_status(root).hint
                )
            self._runner = QwenTtsRunner(directory, device=device)

    def synthesize(self, text: str) -> AudioClip:
        if not text.strip():
            raise ValueError("Cannot synthesize empty text.")
        code = self.language or detect_language(text)
        if not qwen_tts_supports(code):
            if self.fallback is not None:
                return self.fallback.synthesize(text)
            raise ValueError(
                f"Qwen3-TTS cannot speak {code!r}; it speaks "
                f"{', '.join(QWEN_TTS_LANGUAGES)} (no Arabic). Use a system engine "
                "for Arabic, or pass a fallback synthesizer."
            )
        wavs, sample_rate = self._runner.synthesize(
            text,
            language=_LANGUAGE_NAMES.get(code, code),
            speaker=self.speaker,
            instruct=self.instruct,
        )
        samples = _to_int16(wavs, sample_rate)
        if not samples:
            raise ValueError("Qwen3-TTS produced no audio.")
        return AudioClip(samples)


# ------------------------------------------------------------- availability


def available_recognizer() -> bool:
    """True when Qwen3-ASR is fully usable (extra installed and weights cached)."""
    return qwen_asr_status().available


def available_synthesizer() -> bool:
    """True when Qwen3-TTS is fully usable (extra installed and weights cached)."""
    return qwen_tts_status().available


# ----------------------------------------------------------------- download


def _approve(tool: str) -> HumanApprovalGate:
    """A gate that records a human approval (the CLI's ``--approve`` convenience)."""
    gate = HumanApprovalGate()
    request = gate.request(tool, requested_by="cli-user")
    request.approve("cli-reviewer")
    return gate


def download_model(
    kind: str,
    spec_key: str | None = None,
    *,
    root: Path | str | None = None,
    approved: bool = False,
    permissions: PermissionPolicy | None = None,
    approvals: HumanApprovalGate | None = None,
) -> Path:
    """Download a Qwen3 speech model's weights into the data folder.

    ``kind`` is ``"asr"`` or ``"tts"``. The download leaves the machine, so it is
    treated as an external tool: denied by default, and it runs only when an
    approving :class:`PermissionPolicy` *and* a live :class:`HumanApprovalGate`
    approval exist (or ``approved=True``, the CLI's convenience that records a
    human approval). Returns the model folder.
    """
    if kind == "asr":
        spec: QwenAsrSpec | QwenTtsSpec = resolve_asr_spec(spec_key)
        tool = DOWNLOAD_TOOL_ASR
        feature = FEATURE_QWEN_ASR
    elif kind == "tts":
        spec = resolve_tts_spec(spec_key)
        tool = DOWNLOAD_TOOL_TTS
        feature = FEATURE_QWEN_TTS
    else:
        raise ValueError(f"Unknown Qwen speech kind {kind!r}; choose 'asr' or 'tts'.")

    if approved:
        permissions = permissions or PermissionPolicy(
            allowed=frozenset(
                {
                    Capability.SYMBOLIC_SIMULATION,
                    Capability.SANDBOX_TASK,
                    Capability.EXTERNAL_TOOL,
                }
            ),
            allowed_external_tools=frozenset({tool}),
        )
        approvals = approvals or _approve(tool)
    policy = permissions or PermissionPolicy()
    gate = approvals or HumanApprovalGate()
    decision = policy.decide(Capability.EXTERNAL_TOOL, tool)
    if not decision.allowed:
        raise QwenSpeechUnavailable(f"Downloading {spec.key!r} is denied by policy ({decision.reason}).")
    approval = gate.decide(tool)
    if not approval.approved:
        raise QwenSpeechUnavailable(
            f"Downloading {spec.key!r} needs human approval ({approval.reason})."
        )

    directory = _speech_dir(root) / spec.key
    if (directory / "config.json").is_file():
        _write_registry(_speech_dir(root), kind, spec.key, spec.repo)
        return directory

    try:
        from huggingface_hub import snapshot_download
    except ImportError as error:
        raise QwenSpeechUnavailable(
            "Downloading a model needs `huggingface_hub`. Install it with: "
            f"{pip_install_command(feature)}"
        ) from error

    directory.mkdir(parents=True, exist_ok=True)
    snapshot_download(repo_id=spec.repo, local_dir=str(directory))
    _write_registry(_speech_dir(root), kind, spec.key, spec.repo)
    return directory


__all__ = [
    "ASR_CATALOG",
    "ASR_ENGINE_NAME",
    "DEFAULT_ASR_SPEC_KEY",
    "DEFAULT_TTS_SPEAKER",
    "DEFAULT_TTS_SPEC_KEY",
    "KNOWN_TTS_SPEAKERS",
    "QWEN_TTS_LANGUAGES",
    "TTS_CATALOG",
    "TTS_ENGINE_NAME",
    "AsrRunner",
    "QwenAsrRecognizer",
    "QwenAsrSpec",
    "QwenSpeechStatus",
    "QwenSpeechUnavailable",
    "QwenTtsRunner",
    "QwenTtsSpec",
    "QwenTtsSynthesizer",
    "TransformersAsrRunner",
    "TtsRunner",
    "asr_spec_names",
    "asr_spec_summaries",
    "available_recognizer",
    "available_synthesizer",
    "download_model",
    "find_model_dir",
    "model_is_cached",
    "qwen_asr_extra_installed",
    "qwen_asr_status",
    "qwen_tts_extra_installed",
    "qwen_tts_status",
    "qwen_tts_supports",
    "resolve_asr_spec",
    "resolve_tts_spec",
    "tts_spec_names",
    "tts_spec_summaries",
]
