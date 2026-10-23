"""Piper: a real, local neural text-to-speech engine.

**Piper** (``rhasspy/piper``, now maintained by the Open Home Foundation as
``OHF-Voice/piper1-gpl``) renders speech with a small VITS model exported to ONNX,
so a voice sounds like a person instead of the robotic formant synthesis of
``espeak-ng`` -- and it runs **entirely on this machine**, on the CPU, with
nothing sent off it. Unlike the system engines it needs a *model file* per voice,
and unlike Qwen3-TTS it **speaks Arabic** (``ar_JO-kareem``). That combination is
why it is the honest way to give the project a natural Arabic *and* English voice
without leaving the machine.

Design points, matching the rest of the project:

* **A real engine behind the same protocol.** :class:`PiperSynthesizer` satisfies
  :class:`core.speech.SpeechSynthesizer`, so the window, Jarvis, the voice loop,
  and every mouth use it with no change.
* **One voice = one model.** :data:`PIPER_VOICES` names the fetchable voices, and
  :func:`spec_for_voice` maps a project catalog voice (``man-1`` ... ``girl-4``)
  onto one, so the existing picker keeps working and each entry stays distinct.
  Arabic has a *single* Piper voice (male), so every Arabic reading uses it -- a
  limitation of the released voices, stated plainly rather than hidden.
* **Prosody from the catalog.** A project voice's ``speed``/``pitch_range`` become
  Piper's ``length_scale``/``noise_scale``/``noise_w_scale``, so the two voice
  systems agree on how a voice should read.
* **Lazy and optional.** The ``[piper]`` extra (``piper-tts``) and the weights are
  separate; the import is lazy, the app reports the real next step when either is
  missing, and a download runs only through the permission + human-approval gates.
* **Injection for tests.** :class:`PiperRunner` is the seam; a test injects its
  own, so the engine is exercised with no extra and no weights.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from core.arabic_speech import prepare_for_speech
from core.human_approval import HumanApprovalGate
from core.optional_deps import FEATURE_PIPER, pip_install_command
from core.paths import piper_models_dir
from core.permissions import Capability, PermissionPolicy
from core.speech import AudioClip, SelectableVoice, SpeechSynthesizer
from core.voices import (
    CATEGORY_BOY,
    CATEGORY_GIRL,
    CATEGORY_MAN,
    CATEGORY_WOMAN,
    ModelVoice,
    Voice,
    detect_language,
    voices_for,
)

#: The engine name the speech factory uses (``--tts piper``).
PIPER_ENGINE_NAME = "piper"

#: The tool name the download gate is keyed on.
DOWNLOAD_TOOL = "piper-download"

#: The registry file written after a download, naming the voice to load.
REGISTRY_FILENAME = "model.json"

#: The Hugging Face repository the official voices live in.
PIPER_REPO = "rhasspy/piper-voices"

#: The rate Piper renders at (its models are 22.05 kHz); the clip is resampled to
#: the project's 16 kHz in :func:`_float_to_pcm16`.
PIPER_SAMPLE_RATE = 22_050

#: The one Arabic voice Piper ships. Arabic has no second (female) voice in the
#: released catalog, so an Arabic reading always uses this one.
ARABIC_VOICE_KEY = "ar_JO-kareem-medium"


class PiperUnavailable(RuntimeError):
    """Raised when Piper cannot run (no extra, no model file)."""


@dataclass(frozen=True)
class PiperVoiceSpec:
    """A fetchable Piper voice: one ONNX model, one language, one timbre."""

    key: str
    title: str
    language: str
    category: str
    quality: str
    approx_size_mb: int
    summary: str


#: A curated set of the official Piper voices that covers Arabic and English,
#: men and women (and the boys'/girls' light, higher voices). ``key`` is Piper's
#: own model name (``<lang>_<region>-<name>-<quality>``); the ONNX file is fetched
#: from :data:`PIPER_REPO`. Arabic has a single voice; English has many.
PIPER_VOICES: tuple[PiperVoiceSpec, ...] = (
    PiperVoiceSpec(
        key="ar_JO-kareem-medium",
        title="Kareem (Arabic, male)",
        language="ar",
        category=CATEGORY_MAN,
        quality="medium",
        approx_size_mb=63,
        summary="The only official Arabic Piper voice; natural Jordanian MSA.",
    ),
    PiperVoiceSpec(
        key="en_US-ryan-high",
        title="Ryan (English US, male, high)",
        language="en",
        category=CATEGORY_MAN,
        quality="high",
        approx_size_mb=110,
        summary="Clear, warm American male; the best male English voice.",
    ),
    PiperVoiceSpec(
        key="en_US-joe-medium",
        title="Joe (English US, male)",
        language="en",
        category=CATEGORY_MAN,
        quality="medium",
        approx_size_mb=63,
        summary="Relaxed American male, conversational.",
    ),
    PiperVoiceSpec(
        key="en_US-hfc_male-medium",
        title="HFC male (English US, male)",
        language="en",
        category=CATEGORY_MAN,
        quality="medium",
        approx_size_mb=63,
        summary="Clean American male, slightly regional.",
    ),
    PiperVoiceSpec(
        key="en_GB-alan-medium",
        title="Alan (English GB, male)",
        language="en",
        category=CATEGORY_MAN,
        quality="medium",
        approx_size_mb=63,
        summary="Polished British male.",
    ),
    PiperVoiceSpec(
        key="en_US-amy-medium",
        title="Amy (English US, female)",
        language="en",
        category=CATEGORY_WOMAN,
        quality="medium",
        approx_size_mb=63,
        summary="Clear, friendly American female.",
    ),
    PiperVoiceSpec(
        key="en_US-lessac-high",
        title="Lessac (English US, female, high)",
        language="en",
        category=CATEGORY_WOMAN,
        quality="high",
        approx_size_mb=110,
        summary="Articulate American female; a classic Piper voice.",
    ),
    PiperVoiceSpec(
        key="en_US-kristin-medium",
        title="Kristin (English US, female)",
        language="en",
        category=CATEGORY_WOMAN,
        quality="medium",
        approx_size_mb=63,
        summary="Natural American female.",
    ),
    PiperVoiceSpec(
        key="en_US-ljspeech-high",
        title="LJSpeech (English US, female, high)",
        language="en",
        category=CATEGORY_WOMAN,
        quality="high",
        approx_size_mb=110,
        summary="Bright American female, audiobook-like.",
    ),
    PiperVoiceSpec(
        key="en_US-hfc_female-medium",
        title="HFC female (English US, female)",
        language="en",
        category=CATEGORY_WOMAN,
        quality="medium",
        approx_size_mb=63,
        summary="Light American female; used for the boys'/girls' voices.",
    ),
)

_BY_KEY: dict[str, PiperVoiceSpec] = {spec.key: spec for spec in PIPER_VOICES}

#: English voice per project category, in the order the catalog lists them, so
#: ``man-1`` .. ``man-4`` pick four *different* real voices. Arabic ignores this
#: (there is one Arabic voice) and uses :data:`ARABIC_VOICE_KEY`.
_ENGLISH_BY_CATEGORY: dict[str, tuple[str, ...]] = {
    CATEGORY_MAN: ("en_US-ryan-high", "en_US-joe-medium", "en_US-hfc_male-medium", "en_GB-alan-medium"),
    CATEGORY_WOMAN: ("en_US-amy-medium", "en_US-lessac-high", "en_US-kristin-medium", "en_US-ljspeech-high"),
    CATEGORY_BOY: ("en_US-hfc_female-medium", "en_US-ljspeech-high", "en_US-amy-medium", "en_US-kristin-medium"),
    CATEGORY_GIRL: ("en_US-ljspeech-high", "en_US-hfc_female-medium", "en_US-kristin-medium", "en_US-amy-medium"),
}

#: The voice used for a language when none was chosen.
_DEFAULT_BY_LANGUAGE: dict[str, str] = {"ar": ARABIC_VOICE_KEY, "en": "en_US-amy-medium"}


def spec_names() -> tuple[str, ...]:
    """The keys of the fetchable Piper voices."""
    return tuple(spec.key for spec in PIPER_VOICES)


def resolve_spec(name: str | None) -> PiperVoiceSpec:
    """The :class:`PiperVoiceSpec` for ``name`` (or the default). Raises on a bad name."""
    key = (name or ARABIC_VOICE_KEY).strip()
    if key in _BY_KEY:
        return _BY_KEY[key]
    raise ValueError(f"Unknown Piper voice {name!r}; choose one of {', '.join(_BY_KEY)}.")


def spec_summaries() -> tuple[tuple[str, str, str, int, str], ...]:
    """One row per fetchable voice: ``(key, title, language, size_mb, summary)``."""
    return tuple(
        (spec.key, spec.title, spec.language, spec.approx_size_mb, spec.summary)
        for spec in PIPER_VOICES
    )


def spec_for_voice(voice: SelectableVoice | None, language: str) -> PiperVoiceSpec:
    """The Piper voice a project catalog voice maps to, for ``language``.

    Arabic has one official voice, so every Arabic reading uses it; English maps a
    voice's category and its place in that category onto a distinct real voice, so
    ``man-1`` and ``man-2`` do not come out identical. A :class:`~core.voices.SystemVoice`
    (an OS voice Piper cannot see) falls back to the language default.
    """
    code = language if language in ("ar", "en") else detect_language(language)
    if code == "ar":
        return _BY_KEY[ARABIC_VOICE_KEY]
    if not isinstance(voice, Voice):
        return _BY_KEY[_DEFAULT_BY_LANGUAGE["en"]]
    pool = _ENGLISH_BY_CATEGORY.get(voice.category)
    if not pool:
        return _BY_KEY[_DEFAULT_BY_LANGUAGE["en"]]
    siblings = voices_for(voice.category)
    index = siblings.index(voice) if voice in siblings else 0
    return _BY_KEY[pool[min(index, len(pool) - 1)]]


# ------------------------------------------------------------------ prosody


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def length_scale(voice: SelectableVoice | None) -> float:
    """Piper's ``length_scale`` (higher is slower) from a catalog voice's speed.

    175 wpm is Piper's neutral pace, so the catalog's conversational speeds land
    near 1.0 and the reading stays natural rather than rushed.
    """
    if not isinstance(voice, Voice):
        return 1.0
    return round(_clamp(175.0 / voice.speed, 0.7, 1.4), 3)


def noise_scale(voice: SelectableVoice | None) -> float:
    """Piper's ``noise_scale`` from a catalog voice's intonation range.

    A wider range (a more expressive voice) means a little more variation, which
    is what makes the delivery sound alive rather than flat.
    """
    if not isinstance(voice, Voice):
        return 0.667
    return round(_clamp(0.667 * (voice.pitch_range / 62.0), 0.4, 1.2), 3)


def noise_w_scale(voice: SelectableVoice | None) -> float:
    """Piper's ``noise_w_scale`` (phoneme-width variation) from the intonation range."""
    if not isinstance(voice, Voice):
        return 0.8
    return round(_clamp(0.8 * (voice.pitch_range / 62.0), 0.5, 1.4), 3)


def _speaker_id(voice: SelectableVoice | None) -> int | None:
    """The multi-speaker id for a voice, or ``None`` for the single-speaker models.

    Every voice in :data:`PIPER_VOICES` is single-speaker, so this is ``None``; the
    hook is kept explicit so a multi-speaker model can be added later without a
    change to the runner contract.
    """
    return None


# --------------------------------------------------------------- conversion


def _float_to_pcm16(samples: Any, sample_rate: int, *, target_rate: int = 16_000) -> tuple[int, ...]:
    """Convert a float waveform in ``[-1, 1]`` into int16 PCM at ``target_rate``."""
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


# -------------------------------------------------------------------- runner


class PiperRunner(Protocol):
    """The seam that runs Piper: text in, a float waveform and rate out.

    The default runner loads a Piper model with the ``piper-tts`` package; a test
    injects its own, so the engine is exercised with no extra and no weights.
    """

    def synthesize(
        self,
        text: str,
        *,
        speaker: int | None,
        length_scale: float,
        noise_scale: float,
        noise_w_scale: float,
    ) -> tuple[Any, int]: ...


class PiperOnnxRunner:
    """A :class:`PiperRunner` built on the ``piper-tts`` package (lazy imports)."""

    def __init__(self, model_path: Path | str) -> None:
        self.model_path = Path(model_path)
        self._voice: Any = None

    def _load(self) -> Any:
        if self._voice is None:
            from piper.voice import PiperVoice

            self._voice = PiperVoice.load(str(self.model_path))
        return self._voice

    def synthesize(
        self,
        text: str,
        *,
        speaker: int | None,
        length_scale: float,
        noise_scale: float,
        noise_w_scale: float,
    ) -> tuple[Any, int]:
        import numpy as np
        from piper.config import SynthesisConfig

        voice = self._load()
        config = SynthesisConfig(
            speaker_id=speaker,
            length_scale=length_scale,
            noise_scale=noise_scale,
            noise_w_scale=noise_w_scale,
        )
        chunks = [chunk.audio_float_array for chunk in voice.synthesize(text, syn_config=config)]
        if not chunks:
            return (), int(voice.config.sample_rate)
        samples = np.concatenate(chunks) if len(chunks) > 1 else chunks[0]
        return samples, int(voice.config.sample_rate)


# ------------------------------------------------------------------- engine


class PiperSynthesizer:
    """Speech synthesis through Piper, satisfying :class:`SpeechSynthesizer`.

    ``runner`` is the injection seam; when it is ``None`` the engine loads the
    ONNX model itself. ``voice`` is a project catalog voice, and ``spec_key`` pins
    a Piper voice explicitly. Arabic always uses Piper's one Arabic voice; English
    picks a real voice from the catalog. When the voice a language needs is not
    downloaded yet and a ``fallback`` synthesizer (the system engine) was supplied,
    the fallback reads that language instead -- so one installed voice does not
    leave the other language silent.
    """

    name = PIPER_ENGINE_NAME

    def __init__(
        self,
        *,
        voice: SelectableVoice | None = None,
        language: str | None = None,
        spec_key: str | None = None,
        model_dir: Path | str | None = None,
        runner: PiperRunner | None = None,
        fallback: SpeechSynthesizer | None = None,
        root: Path | str | None = None,
    ) -> None:
        self.language = language
        self._voice = voice
        self._spec_key = spec_key
        self._root = root
        self._fallback = fallback
        self._runner: PiperRunner | None = None
        self._loaded_key: str | None = None
        self._fixed_model = model_dir is not None or runner is not None
        if runner is not None:
            # An injected runner is language-agnostic: the test controls the audio.
            self._runner = runner
            self._loaded_key = spec_key
            return
        directory = Path(model_dir) if model_dir is not None else find_model_file(spec_key, root)
        if directory is None:
            raise PiperUnavailable("No Piper voice is present. " + piper_status(root).hint)
        self._runner = PiperOnnxRunner(directory)
        # The file is named ``<voice>.onnx``, so its stem is the voice key: that
        # lets a language switch still find a *different* model while an explicit
        # path is kept for the language it was given for.
        self._loaded_key = spec_key or Path(directory).stem

    def _spec(self, text: str) -> PiperVoiceSpec:
        code = self.language or detect_language(text)
        if self._spec_key:
            return resolve_spec(self._spec_key)
        if isinstance(self._voice, ModelVoice) and self._voice.engine == PIPER_ENGINE_NAME:
            # A chosen Piper voice names its own model file, so load exactly it.
            return resolve_spec(self._voice.name)
        return spec_for_voice(self._voice, code)

    def _ensure_model(self, spec: PiperVoiceSpec) -> bool:
        """Make sure ``spec``'s model is loaded; return False when it is absent.

        A language switch loads the other language's model on demand. ``False``
        (rather than raising) lets :meth:`synthesize` hand the text to the fallback
        engine when that voice is simply not downloaded yet.
        """
        if self._runner is None or self._loaded_key == spec.key:
            return True
        if self._fixed_model:
            # An explicit model path was given, so it is the only one to use; the
            # loaded key is kept in step so the check above does not retry forever.
            self._loaded_key = spec.key
            return True
        path = find_model_file(spec.key, self._root)
        if path is None:
            return False
        self._runner = PiperOnnxRunner(path)
        self._loaded_key = spec.key
        return True

    def synthesize(self, text: str) -> AudioClip:
        if not text.strip():
            raise ValueError("Cannot synthesize empty text.")
        spec = self._spec(text)
        if not self._ensure_model(spec):
            if self._fallback is not None:
                return self._fallback.synthesize(text)
            raise PiperUnavailable(
                f"No Piper voice {spec.key!r} is present. Fetch it with: "
                f"python -m cli piper download {spec.key} --approve"
            )
        assert self._runner is not None  # guaranteed by __init__ / _ensure_model
        # Piper phonemizes through espeak-ng, so its Arabic voice has the same
        # blind spot for ``25%``/``1500 psi``/``10-20``: spell them as words.
        text = prepare_for_speech(text, language=self.language or detect_language(text))
        samples, sample_rate = self._runner.synthesize(
            text,
            speaker=_speaker_id(self._voice),
            length_scale=length_scale(self._voice),
            noise_scale=noise_scale(self._voice),
            noise_w_scale=noise_w_scale(self._voice),
        )
        pcm = _float_to_pcm16(samples, sample_rate)
        if not pcm:
            raise ValueError("Piper produced no audio.")
        return AudioClip(pcm)


# ------------------------------------------------------------- availability


def piper_extra_installed() -> bool:
    """Whether the Python runtime for Piper is importable."""
    import importlib.util

    return importlib.util.find_spec("piper") is not None


@dataclass(frozen=True)
class PiperStatus:
    """Exactly what is present for Piper."""

    extra_installed: bool
    model_cached: bool
    model_file: Path | None
    hint: str

    @property
    def available(self) -> bool:
        return self.extra_installed and self.model_cached


def piper_status(root: Path | str | None = None) -> PiperStatus:
    """Report whether Piper can run here, and the real next step if not."""
    installed = piper_extra_installed()
    model = find_model_file(None, root)
    if not installed:
        hint = f"Install the Piper runtime with: {pip_install_command(FEATURE_PIPER)}"
    elif model is None:
        hint = "Download a voice with: python -m cli piper download --approve"
    else:
        hint = "Piper is ready."
    return PiperStatus(installed, model is not None, model, hint)


def available_synthesizer(root: Path | str | None = None) -> bool:
    """True when Piper is fully usable (extra installed and a voice cached)."""
    return piper_status(root).available


# --------------------------------------------------------------------- files


def _piper_dir(root: Path | str | None) -> Path:
    return Path(root) if root is not None else piper_models_dir()


def _registry_file(root: Path | str | None) -> Path:
    return _piper_dir(root) / REGISTRY_FILENAME


def _read_registry(root: Path | str | None) -> dict[str, str]:
    import json

    registry = _registry_file(root)
    if not registry.is_file():
        return {}
    try:
        data = json.loads(registry.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return {str(k): str(v) for k, v in data.items()} if isinstance(data, dict) else {}


def _write_registry(directory: Path, key: str) -> None:
    import json

    try:
        directory.mkdir(parents=True, exist_ok=True)
        _registry_file(directory).write_text(
            json.dumps({"voice": key}, indent=2), encoding="utf-8"
        )
    except OSError:
        pass


def find_model_file(key: str | None = None, root: Path | str | None = None) -> Path | None:
    """The ONNX file for a Piper voice, or the first installed one when ``key`` is ``None``.

    Prefers the voice the registry names (``model.json``, written by
    :func:`download_model`); a file the user dropped in by hand is still found, so
    the engine never pretends nothing is installed.
    """
    base = _piper_dir(root)
    if key:
        return _locate(base, key)
    registered = _read_registry(root).get("voice")
    if registered:
        found = _locate(base, registered)
        if found is not None:
            return found
    hits = sorted(base.rglob("*.onnx")) if base.is_dir() else []
    return hits[0] if hits else None


def _locate(base: Path, key: str) -> Path | None:
    for candidate in (base / f"{key}.onnx", base / key / f"{key}.onnx"):
        if candidate.is_file():
            return candidate
    hits = sorted(base.rglob(f"{key}.onnx")) if base.is_dir() else []
    return hits[0] if hits else None


def model_is_cached(key: str | None = None, root: Path | str | None = None) -> bool:
    """True when a Piper voice file is present on disk."""
    return find_model_file(key, root) is not None


def _gender_for_category(category: str) -> str:
    """The gender a catalog category reports (Piper's voices are one or the other)."""
    return "Female" if category in (CATEGORY_WOMAN, CATEGORY_GIRL) else "Male"


def cached_keys(root: Path | str | None = None) -> tuple[str, ...]:
    """The Piper voice keys whose ONNX model is present on disk here.

    A key is reported only when *its* file exists (``find_model_file(key)``), so a
    menu built from this can never offer a voice that is not downloaded yet.
    """
    return tuple(spec.key for spec in PIPER_VOICES if find_model_file(spec.key, root) is not None)


def voice_list(root: Path | str | None = None) -> tuple[ModelVoice, ...]:
    """The Piper voices downloaded here, as :class:`~core.voices.ModelVoice` objects.

    Returns ``()`` unless the ONNX runtime is installed *and* at least one voice
    file is present, so the reading-voice menu lists exactly the Piper voices that
    can really be spoken here. Each voice is keyed ``model:piper:<key>`` and carries
    its readable title, so choosing one routes the reading to Piper (see
    :func:`core.speech._engine_for_voice`) with the right voice file loaded.
    """
    if not piper_extra_installed():
        return ()
    voices: list[ModelVoice] = []
    for key in cached_keys(root):
        spec = _BY_KEY[key]
        voices.append(
            ModelVoice(
                name=spec.key,
                engine=PIPER_ENGINE_NAME,
                language=spec.language,
                gender=_gender_for_category(spec.category),
                model="Piper",
                title=spec.title,
            )
        )
    return tuple(voices)


# ------------------------------------------------------------------ download


def _approve(tool: str) -> HumanApprovalGate:
    """A gate that records a human approval (the CLI's ``--approve`` convenience)."""
    gate = HumanApprovalGate()
    request = gate.request(tool, requested_by="cli-user")
    request.approve("cli-reviewer")
    return gate


def download_model(
    spec_key: str | None = None,
    *,
    root: Path | str | None = None,
    approved: bool = False,
    permissions: PermissionPolicy | None = None,
    approvals: HumanApprovalGate | None = None,
) -> Path:
    """Download a Piper voice's ONNX model and config into the data folder.

    The download leaves the machine, so it is treated as an external tool: denied
    by default, and it runs only when an approving :class:`PermissionPolicy` *and*
    a live :class:`HumanApprovalGate` approval exist (or ``approved=True``, the
    CLI's convenience that records a human approval). Returns the voice folder.
    """
    spec = resolve_spec(spec_key)

    if approved:
        permissions = permissions or PermissionPolicy(
            allowed=frozenset(
                {
                    Capability.SYMBOLIC_SIMULATION,
                    Capability.SANDBOX_TASK,
                    Capability.EXTERNAL_TOOL,
                }
            ),
            allowed_external_tools=frozenset({DOWNLOAD_TOOL}),
        )
        approvals = approvals or _approve(DOWNLOAD_TOOL)
    policy = permissions or PermissionPolicy()
    gate = approvals or HumanApprovalGate()
    decision = policy.decide(Capability.EXTERNAL_TOOL, DOWNLOAD_TOOL)
    if not decision.allowed:
        raise PiperUnavailable(
            f"Downloading Piper voice {spec.key!r} is denied by policy ({decision.reason})."
        )
    approval = gate.decide(DOWNLOAD_TOOL)
    if not approval.approved:
        raise PiperUnavailable(
            f"Downloading Piper voice {spec.key!r} needs human approval ({approval.reason})."
        )

    directory = _piper_dir(root) / spec.key
    if _locate(directory.parent, spec.key) is not None:
        _write_registry(_piper_dir(root), spec.key)
        return directory

    try:
        from huggingface_hub import snapshot_download
    except ImportError as error:
        raise PiperUnavailable(
            "Downloading a Piper voice needs `huggingface_hub`. Install it with: "
            f"{pip_install_command(FEATURE_PIPER)}"
        ) from error

    directory.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id=PIPER_REPO,
        local_dir=str(directory),
        allow_patterns=[f"**/{spec.key}.onnx", f"**/{spec.key}.onnx.json"],
    )
    _write_registry(_piper_dir(root), spec.key)
    return directory


__all__ = [
    "ARABIC_VOICE_KEY",
    "DOWNLOAD_TOOL",
    "PIPER_ENGINE_NAME",
    "PIPER_REPO",
    "PIPER_SAMPLE_RATE",
    "PIPER_VOICES",
    "PiperOnnxRunner",
    "PiperRunner",
    "PiperStatus",
    "PiperSynthesizer",
    "PiperUnavailable",
    "PiperVoiceSpec",
    "available_synthesizer",
    "cached_keys",
    "download_model",
    "find_model_file",
    "length_scale",
    "model_is_cached",
    "noise_scale",
    "noise_w_scale",
    "piper_extra_installed",
    "piper_status",
    "resolve_spec",
    "spec_for_voice",
    "spec_names",
    "spec_summaries",
    "voice_list",
]
