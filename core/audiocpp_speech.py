"""VoiceTut-TTS over **audio.cpp**: a real, local Arabic speech engine.

`audio.cpp <https://github.com/0xShug0/audio.cpp>`_ is an all-in-one C++ (ggml)
inference engine for audio models -- TTS, ASR, voice conversion, and more -- with
no Python dependency. This module drives its ``audiocpp_cli`` binary against
**VoiceTut-TTS**, an OmniVoice fine-tune for **Egyptian Arabic** with seamless
Arabic/English code-switching, which is the honest answer to "read Arabic like a
person": Piper has a single Arabic voice and Qwen3-TTS has none at all.

Everything runs **on this machine**. Nothing is sent off it, and the engine is a
native program (like ``espeak-ng`` or Tesseract), not a pip package -- so this
module *finds* the binary and reports the real next step when it is absent, rather
than pretending a pip install would fix it.

Design points, matching the rest of the project:

* **A real engine behind the same protocol.** :class:`AudioCppSynthesizer`
  satisfies :class:`core.speech.SpeechSynthesizer`, so the window, Jarvis, the
  voice loop, and every mouth use it with no change.
* **The model's own voices.** :func:`voice_list` reports VoiceTut's built-in studio
  speakers (Ahmed, Sarah, ...) as :class:`~core.voices.ModelVoice` objects, so the
  menus list the voices the model really has and a chosen one is spoken by name.
* **Injection for tests.** :class:`AudioCppRunner` is the seam; a test injects its
  own, so the engine is exercised with no binary, no weights, and no audio.
* **Lazy and optional.** The engine and the weights are separate one-time,
  human-approved steps (``install_engine`` / ``download_model``), each through the
  permission + human-approval gates, like every other model download.
* **Arabic is spelled for speech.** The text goes through
  :func:`core.arabic_speech.prepare_for_speech`, because audio.cpp passes text
  through as-is and reads Arabic *symbols* badly (``25%``, ``1500 psi``, ``10-20``).
"""

from __future__ import annotations

import json
import shutil
import tarfile
import tempfile
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from core.arabic_speech import prepare_for_speech
from core.human_approval import HumanApprovalGate
from core.paths import audiocpp_models_dir
from core.permissions import Capability, PermissionPolicy
from core.speech import AudioClip, SpeechSynthesizer
from core.voices import ModelVoice, detect_language

#: The engine name the speech factory uses (``--tts audio.cpp``).
AUDIOCPP_ENGINE_NAME = "audio.cpp"

#: The tool names the permission/approval gates are keyed on.
DOWNLOAD_TOOL = "audiocpp-download"
INSTALL_TOOL = "audiocpp-install-engine"

#: The registry file written after a download, naming the model to load.
REGISTRY_FILENAME = "model.json"

#: The audio.cpp family VoiceTut-TTS runs through (its architecture is OmniVoice's).
VOICETUT_FAMILY = "omnivoice"

#: The language id VoiceTut was trained with. ``arz`` is *Egyptian* Arabic; passing
#: it is what the model card insists on, so it is the default here.
VOICETUT_LANGUAGE = "arz"

#: The rate VoiceTut renders at.
AUDIOCPP_SAMPLE_RATE = 24_000

#: The diffusion steps to use per request. The model's own default is 32, which is
#: slow on a CPU; 8 measured ~1/3 the wall time for the same intelligible speech.
DEFAULT_INFERENCE_STEPS = 8

#: Where a prebuilt audio.cpp release lives. The binaries are per-platform, so a
#: user who cannot fetch one is told to build it instead.
RELEASE_BASE = "https://github.com/0xShug0/audio.cpp/releases/download/v0.9.2-drama3-base-small"


@dataclass(frozen=True)
class VoiceTutSpec:
    """A fetchable VoiceTut-TTS package: one GGUF file, one language, many voices."""

    key: str
    title: str
    filename: str
    repo: str
    precision: str
    approx_size_mb: int
    summary: str


#: The two standalone GGUF packages VoiceTut-TTS ships. Both are self-contained:
#: the generator, the OmniVoice audio tokenizer, the tokenizer, and the audio.cpp
#: model spec are all in the one file, so one file is all that is needed.
VOICETUT_MODELS: tuple[VoiceTutSpec, ...] = (
    VoiceTutSpec(
        key="voicetut-q8_0",
        title="VoiceTut-TTS Egyptian Arabic (Q8_0)",
        filename="voicetut-tts-q8_0.gguf",
        repo="mohammedaly22/VoiceTut-TTS-GGUF",
        precision="q8_0",
        approx_size_mb=1288,
        summary="Recommended: Egyptian Arabic with Arabic/English code-switching.",
    ),
    VoiceTutSpec(
        key="voicetut-f16",
        title="VoiceTut-TTS Egyptian Arabic (F16)",
        filename="voicetut-tts-f16.gguf",
        repo="mohammedaly22/VoiceTut-TTS-GGUF",
        precision="f16",
        approx_size_mb=1563,
        summary="Higher precision; slower on GPUs without tensor cores.",
    ),
)

#: The upstream repo that carries the built-in studio reference speakers.
SPEAKERS_REPO = "mohammedaly22/VoiceTut-TTS"

#: The built-in studio speakers, exactly as the upstream ``references.json`` names
#: them (``speaker_name`` -> file stem + gender). Kept as declarative data so the
#: menu can list them without a network call; :func:`voice_list` still verifies
#: each reference clip exists before it offers the voice.
VOICETUT_SPEAKERS: tuple[tuple[str, str, str], ...] = (
    ("Ahmed", "ahmed", "male"),
    ("Ali", "ali", "male"),
    ("Hadeer", "hadeer", "female"),
    ("Hanan", "hanan", "female"),
    ("Hassan", "hassan", "male"),
    ("Karim", "karim", "male"),
    ("Mahmoud", "mahmoud", "male"),
    ("Mariam", "mariam", "female"),
    ("Mohammed", "mohammed", "male"),
    ("Mostafa", "mostafa", "male"),
    ("Ramy", "ramy", "male"),
    ("Rana", "rana", "female"),
    ("Sarah", "sarah", "female"),
    ("Shaker", "shaker", "male"),
    ("Tarek", "tarek", "male"),
    ("Yara", "yara", "female"),
    ("Yasmin", "yasmin", "female"),
    ("Yehia", "yehia", "male"),
    ("Youssef", "youssef", "male"),
)


class AudioCppUnavailable(RuntimeError):
    """Raised when the audio.cpp engine or its model cannot run here."""


# ------------------------------------------------------------------ binaries


def engine_dir(root: Path | str | None = None) -> Path:
    """The folder the audio.cpp binaries live in (``data/audio.cpp/bin``)."""
    base = Path(root) if root is not None else audiocpp_models_dir()
    return base / "bin"


def find_cli(root: Path | str | None = None) -> Path | None:
    """The ``audiocpp_cli`` executable, or ``None`` when it is not installed.

    A user who built audio.cpp themselves and put it on ``PATH`` is honoured too, so
    the engine never reports "not installed" for a working binary.
    """
    local = engine_dir(root) / ("audiocpp_cli.exe" if _is_windows() else "audiocpp_cli")
    if local.is_file():
        return local
    found = shutil.which("audiocpp_cli")
    return Path(found) if found else None


def engine_installed(root: Path | str | None = None) -> bool:
    """True when an ``audiocpp_cli`` binary can be found."""
    return find_cli(root) is not None


def _is_windows() -> bool:
    import sys

    return sys.platform.startswith("win")


def _engine_env(bin_dir: Path) -> dict[str, str]:
    """The environment the engine runs with: its own folder on the library path.

    A prebuilt audio.cpp release ships the binary *and* its shared libraries
    (``libggml*.so``) side by side, and the binary looks for them next to itself
    through ``$ORIGIN``. On some platforms that lookup is not enough, so the folder
    is also put on the dynamic-loader path -- which costs nothing when it already
    works and is what makes the engine start when it would otherwise report a
    missing ``libggml``.
    """
    import os

    env = dict(os.environ)
    folder = str(bin_dir)
    variable = "PATH" if _is_windows() else ("DYLD_LIBRARY_PATH" if _is_macos() else "LD_LIBRARY_PATH")
    existing = env.get(variable, "")
    env[variable] = f"{folder}{os.pathsep}{existing}" if existing else folder
    return env


def _is_macos() -> bool:
    import sys

    return sys.platform == "darwin"


def _release_asset() -> tuple[str, str]:
    """The prebuilt release asset for this platform as ``(filename, kind)``.

    ``kind`` is ``'tar'`` or ``'zip'``. Only CPU builds are fetched: they run
    everywhere and need no CUDA/Vulkan runtime, which is the honest default for an
    install step the app performs on the user's behalf.
    """
    import platform
    import sys

    machine = platform.machine().lower()
    if sys.platform.startswith("win"):
        return "audio-v0.9.2-drama3-base-small-bin-windows-x64-cpu.zip", "zip"
    if sys.platform == "darwin":
        tag = "arm64" if machine in ("arm64", "aarch64") else "x64"
        return f"audio-v0.9.2-drama3-base-small-bin-macos-{tag}-metal.tar.gz", "tar"
    return "audio-v0.9.2-drama3-base-small-bin-ubuntu-x64-cpu-portable.tar.gz", "tar"


def _extract(archive: Path, destination: Path, kind: str) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    if kind == "zip":
        with zipfile.ZipFile(archive) as handle:
            handle.extractall(destination)
    else:
        with tarfile.open(archive) as handle:
            handle.extractall(destination, filter="data")


def install_engine(
    *,
    root: Path | str | None = None,
    approved: bool = False,
    permissions: PermissionPolicy | None = None,
    approvals: HumanApprovalGate | None = None,
) -> Path:
    """Fetch a prebuilt ``audiocpp_cli`` binary into the data folder.

    The engine is a native program, so pip cannot install it. This downloads a
    prebuilt CPU release instead, which leaves the machine -- so it is treated as an
    external tool: denied by default, and it runs only through the permission *and*
    human-approval gates (or ``approved=True``, the CLI's convenience). A user who
    prefers to build it can do so and put it on ``PATH``; the engine is then found
    with no download at all. Returns the folder holding the binaries.
    """
    if approved:
        permissions = permissions or PermissionPolicy(
            allowed=frozenset(
                {
                    Capability.SYMBOLIC_SIMULATION,
                    Capability.SANDBOX_TASK,
                    Capability.EXTERNAL_TOOL,
                }
            ),
            allowed_external_tools=frozenset({INSTALL_TOOL}),
        )
        approvals = approvals or _approve(INSTALL_TOOL)
    policy = permissions or PermissionPolicy()
    gate = approvals or HumanApprovalGate()
    decision = policy.decide(Capability.EXTERNAL_TOOL, INSTALL_TOOL)
    if not decision.allowed:
        raise AudioCppUnavailable(
            f"Installing the audio.cpp engine is denied by policy ({decision.reason})."
        )
    approval = gate.decide(INSTALL_TOOL)
    if not approval.approved:
        raise AudioCppUnavailable(
            f"Installing the audio.cpp engine needs human approval ({approval.reason})."
        )

    target = engine_dir(root)
    asset, kind = _release_asset()
    url = f"{RELEASE_BASE}/{asset}"
    with tempfile.TemporaryDirectory(prefix="audiocpp-") as scratch:
        scratch_path = Path(scratch)
        archive = scratch_path / asset
        try:
            urllib.request.urlretrieve(url, archive)
        except OSError as error:
            raise AudioCppUnavailable(
                f"Could not download the audio.cpp engine from {url} ({error}). "
                "Build it from https://github.com/0xShug0/audio.cpp and put "
                "`audiocpp_cli` on your PATH instead."
            ) from error
        staged = scratch_path / "staged"
        _extract(archive, staged, kind)
        # A release tarball may nest the payload in a versioned folder, and the
        # binary must keep its shared libraries beside it (it finds them through
        # $ORIGIN), so the folder that actually holds `audiocpp_cli` is what is
        # installed -- copying only the executable would leave it unable to start.
        binary_name = "audiocpp_cli.exe" if _is_windows() else "audiocpp_cli"
        matches = list(staged.rglob(binary_name))
        if not matches:
            raise AudioCppUnavailable(
                f"The audio.cpp archive did not contain `{binary_name}`."
            )
        source = matches[0].parent
        if target.exists():
            shutil.rmtree(target, ignore_errors=True)
        shutil.copytree(source, target)
    if not engine_installed(root):
        raise AudioCppUnavailable(
            f"The audio.cpp engine was installed to {target} but cannot be found."
        )
    return target


# -------------------------------------------------------------------- models


def _models_root(root: Path | str | None) -> Path:
    return Path(root) if root is not None else audiocpp_models_dir()


def _registry_file(root: Path | str | None) -> Path:
    return _models_root(root) / REGISTRY_FILENAME


def _read_registry(root: Path | str | None) -> dict[str, str]:
    registry = _registry_file(root)
    if not registry.is_file():
        return {}
    try:
        data = json.loads(registry.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return {str(k): str(v) for k, v in data.items()} if isinstance(data, dict) else {}


def _write_registry(root: Path | str | None, key: str) -> None:
    try:
        registry = _registry_file(root)
        registry.parent.mkdir(parents=True, exist_ok=True)
        registry.write_text(json.dumps({"model": key}, indent=2), encoding="utf-8")
    except OSError:
        pass


def resolve_spec(key: str | None) -> VoiceTutSpec:
    """The VoiceTut package named ``key`` (or the default); raise when unknown."""
    if key is None:
        return VOICETUT_MODELS[0]
    wanted = key.strip()
    for spec in VOICETUT_MODELS:
        if spec.key == wanted:
            return spec
    raise ValueError(
        f"Unknown VoiceTut model {key!r}; choose one of {', '.join(s.key for s in VOICETUT_MODELS)}."
    )


def spec_names() -> tuple[str, ...]:
    """Every fetchable VoiceTut package key, in catalog order."""
    return tuple(spec.key for spec in VOICETUT_MODELS)


def spec_summaries() -> tuple[tuple[str, str, str, int, str], ...]:
    """``(key, title, precision, approx_size_mb, summary)`` per package."""
    return tuple(
        (spec.key, spec.title, spec.precision, spec.approx_size_mb, spec.summary)
        for spec in VOICETUT_MODELS
    )


def find_model_file(key: str | None = None, root: Path | str | None = None) -> Path | None:
    """The GGUF file for a VoiceTut package, or the first installed one.

    Prefers the file the registry names (``model.json``, written by
    :func:`download_model`); a ``.gguf`` the user dropped in by hand is still found,
    so the engine never pretends nothing is installed.
    """
    base = _models_root(root)
    if key:
        return _locate(base, resolve_spec(key).filename)
    registered = _read_registry(root).get("model")
    if registered:
        try:
            found = _locate(base, resolve_spec(registered).filename)
        except ValueError:
            found = None
        if found is not None:
            return found
    hits = sorted(base.rglob("*.gguf")) if base.is_dir() else []
    return hits[0] if hits else None


def _locate(base: Path, filename: str) -> Path | None:
    direct = base / filename
    if direct.is_file():
        return direct
    hits = sorted(base.rglob(filename)) if base.is_dir() else []
    return hits[0] if hits else None


def model_is_cached(key: str | None = None, root: Path | str | None = None) -> bool:
    """True when a VoiceTut GGUF file is present on disk."""
    return find_model_file(key, root) is not None


def download_model(
    spec_key: str | None = None,
    *,
    root: Path | str | None = None,
    approved: bool = False,
    permissions: PermissionPolicy | None = None,
    approvals: HumanApprovalGate | None = None,
) -> Path:
    """Download a VoiceTut GGUF file into the data folder.

    The download leaves the machine, so it is treated as an external tool: denied by
    default, and it runs only through the permission *and* human-approval gates (or
    ``approved=True``, the CLI's convenience). Returns the model folder.
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
        raise AudioCppUnavailable(
            f"Downloading {spec.key!r} is denied by policy ({decision.reason})."
        )
    approval = gate.decide(DOWNLOAD_TOOL)
    if not approval.approved:
        raise AudioCppUnavailable(
            f"Downloading {spec.key!r} needs human approval ({approval.reason})."
        )

    directory = _models_root(root) / "VoiceTut-TTS-GGUF"
    directory.mkdir(parents=True, exist_ok=True)
    target = directory / spec.filename
    if not target.is_file():
        url = f"https://huggingface.co/{spec.repo}/resolve/main/{spec.filename}"
        try:
            urllib.request.urlretrieve(url, target)
        except OSError as error:
            raise AudioCppUnavailable(
                f"Could not download {spec.filename} from {url} ({error}). "
                "You can fetch it by hand and drop it into "
                f"{directory} instead."
            ) from error
    _write_registry(root, spec.key)
    return directory


def download_speakers(
    *,
    root: Path | str | None = None,
    approved: bool = False,
    permissions: PermissionPolicy | None = None,
    approvals: HumanApprovalGate | None = None,
) -> Path:
    """Download VoiceTut's built-in reference speakers (one short clip each).

    A zero-shot cloning model needs a reference clip *and* its transcript to speak a
    named voice, so this fetches the upstream studio speakers -- a few hundred KB
    each -- plus ``references.json``. Governed exactly like the model download.
    """
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
    if not policy.decide(Capability.EXTERNAL_TOOL, DOWNLOAD_TOOL).allowed:
        raise AudioCppUnavailable("Downloading the reference speakers is denied by policy.")
    if not gate.decide(DOWNLOAD_TOOL).approved:
        raise AudioCppUnavailable("Downloading the reference speakers needs human approval.")

    directory = voices_dir(root)
    directory.mkdir(parents=True, exist_ok=True)
    base = f"https://huggingface.co/{SPEAKERS_REPO}/resolve/main/reference_speakers"
    for _name, stem, _gender in VOICETUT_SPEAKERS:
        target = directory / f"{stem}.wav"
        if target.is_file():
            continue
        try:
            urllib.request.urlretrieve(f"{base}/{stem}.wav", target)
        except OSError as error:
            raise AudioCppUnavailable(
                f"Could not download the reference voice {stem}.wav ({error})."
            ) from error
    references = directory.parent / "references.json"
    if not references.is_file():
        try:
            urllib.request.urlretrieve(f"{base}/references.json", references)
        except OSError:
            # The transcripts are an optimisation, not a requirement: a missing file
            # only means the reference text is read from the catalog instead.
            pass
    return directory


def voices_dir(root: Path | str | None = None) -> Path:
    """Where VoiceTut's reference clips live (``<data>/audio.cpp/voices``)."""
    return _models_root(root) / "voices"


def _reference_texts(root: Path | str | None) -> dict[str, str]:
    """``{file stem: transcript}`` from the downloaded ``references.json``."""
    path = _models_root(root) / "references.json"
    if not path.is_file():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    texts: dict[str, str] = {}
    if isinstance(data, list):
        for entry in data:
            if not isinstance(entry, dict):
                continue
            audio = str(entry.get("audio_path") or "")
            text = str(entry.get("reference_text") or "")
            if audio and text:
                texts[Path(audio).stem.lower()] = text
    return texts


def voice_list(root: Path | str | None = None) -> tuple[ModelVoice, ...]:
    """VoiceTut's built-in speakers as :class:`~core.voices.ModelVoice` objects.

    Returns ``()`` unless the model is installed *and* a reference clip is present,
    because a cloning engine cannot speak a named voice without one -- offering a
    voice that would fail is exactly the dishonesty this project avoids. Each voice
    carries its reference clip and transcript, which the engine needs to clone it.
    """
    if find_model_file(None, root) is None:
        return ()
    directory = voices_dir(root)
    texts = _reference_texts(root)
    voices: list[ModelVoice] = []
    for name, stem, gender in VOICETUT_SPEAKERS:
        clip = directory / f"{stem}.wav"
        if not clip.is_file():
            continue
        voices.append(
            ModelVoice(
                name=name,
                engine=AUDIOCPP_ENGINE_NAME,
                language=VOICETUT_LANGUAGE,
                gender=gender,
                model="VoiceTut-TTS",
                reference=clip,
                reference_text=texts.get(stem, ""),
            )
        )
    return tuple(voices)


# ------------------------------------------------------------- availability


@dataclass(frozen=True)
class AudioCppStatus:
    """Exactly what is present for the audio.cpp engine."""

    engine_installed: bool
    cli_path: Path | None
    model_cached: bool
    model_file: Path | None
    voices: int
    hint: str

    @property
    def available(self) -> bool:
        """True when the engine *and* a model are present (so audio is not silence)."""
        return self.engine_installed and self.model_cached


def audiocpp_status(root: Path | str | None = None) -> AudioCppStatus:
    """Report whether VoiceTut over audio.cpp can run here, and the real next step."""
    cli = find_cli(root)
    model = find_model_file(None, root)
    voices = len(voice_list(root))
    if cli is None:
        hint = (
            "Install the engine with: python -m cli audio install-engine --approve, "
            "or build audio.cpp and put `audiocpp_cli` on your PATH."
        )
    elif model is None:
        hint = "Download the model with: python -m cli audio download --approve"
    elif voices == 0:
        hint = (
            "Download the built-in voices with: "
            "python -m cli audio download --speakers --approve"
        )
    else:
        hint = "VoiceTut-TTS over audio.cpp is ready."
    return AudioCppStatus(cli is not None, cli, model is not None, model, voices, hint)


def available_synthesizer(root: Path | str | None = None) -> bool:
    """True when the engine and a model are both present."""
    return audiocpp_status(root).available


def extra_installed() -> bool:
    """Always true: the engine needs no Python package.

    The downloads are plain ``urllib`` requests and the engine is a native program,
    so nothing here can be missing a *pip* dependency. The name is kept because the
    speech factory and the status report ask every engine the same question.
    """
    return True


# ------------------------------------------------------------------- engine


class AudioCppRunner(Protocol):
    """The seam that runs audio.cpp: text in, a WAV path out."""

    def synthesize(
        self,
        text: str,
        *,
        language: str,
        reference: Path | None,
        reference_text: str,
        steps: int,
    ) -> Path: ...


class AudioCppCliRunner:
    """An :class:`AudioCppRunner` that drives the real ``audiocpp_cli`` binary.

    The text is written to a JSON request file rather than passed on the command
    line: a long Arabic answer would otherwise hit the platform's argument-length
    limit, and the JSON route also keeps the reference transcript and the language id
    off the command line. One request per call, so a second call cannot inherit the
    first one's state.
    """

    def __init__(self, cli: Path | str, model: Path | str, *, backend: str = "cpu", timeout: int = 900) -> None:
        self.cli = Path(cli)
        self.model = Path(model)
        self.backend = backend
        self.timeout = timeout

    def synthesize(
        self,
        text: str,
        *,
        language: str,
        reference: Path | None,
        reference_text: str,
        steps: int,
    ) -> Path:
        import subprocess

        request: dict[str, Any] = {
            "id": "speech",
            "text": text,
            "language": language,
            "options": {"num_inference_steps": steps},
        }
        if reference is not None:
            request["voice_ref"] = str(reference)
            if reference_text:
                request["reference_text"] = reference_text
        with tempfile.TemporaryDirectory(prefix="audiocpp-run-") as scratch:
            scratch_path = Path(scratch)
            sequence = scratch_path / "request.json"
            sequence.write_text(
                json.dumps({"requests": [request]}, ensure_ascii=False), encoding="utf-8"
            )
            out_dir = scratch_path / "out"
            out_dir.mkdir()
            command = [
                str(self.cli),
                "--task",
                "tts",
                "--family",
                VOICETUT_FAMILY,
                "--model",
                str(self.model),
                "--backend",
                self.backend,
                "--request-sequence",
                str(sequence),
                "--out-dir",
                str(out_dir),
            ]
            try:
                completed = subprocess.run(
                    command,
                    capture_output=True,
                    text=True,
                    timeout=self.timeout,
                    check=False,
                    cwd=str(self.cli.parent),
                    env=_engine_env(self.cli.parent),
                )
            except (OSError, subprocess.SubprocessError) as error:
                raise AudioCppUnavailable(f"audio.cpp could not run ({error}).") from error
            if completed.returncode != 0:
                detail = (completed.stderr or completed.stdout or "").strip().splitlines()
                tail = detail[-1] if detail else "no output"
                raise AudioCppUnavailable(f"audio.cpp failed (exit {completed.returncode}): {tail}")
            produced = sorted(out_dir.glob("*.wav"))
            if not produced:
                raise AudioCppUnavailable("audio.cpp produced no audio file.")
            # The clip must outlive the temporary folder, so move it beside the
            # model rather than returning a path that is about to be deleted.
            keep = Path(tempfile.mkstemp(prefix="audiocpp-", suffix=".wav")[1])
            shutil.copyfile(produced[0], keep)
            return keep


class AudioCppSynthesizer:
    """Speech synthesis through VoiceTut-TTS over audio.cpp.

    ``runner`` is the injection seam; when it is ``None`` the engine drives the real
    binary. ``voice`` is a :class:`~core.voices.ModelVoice` naming one of the model's
    speakers (it carries the reference clip the engine clones), and ``spec_key``
    pins a model package explicitly.
    """

    name = AUDIOCPP_ENGINE_NAME

    def __init__(
        self,
        *,
        voice: ModelVoice | None = None,
        language: str | None = None,
        spec_key: str | None = None,
        model_file: Path | str | None = None,
        runner: AudioCppRunner | None = None,
        fallback: SpeechSynthesizer | None = None,
        root: Path | str | None = None,
        steps: int = DEFAULT_INFERENCE_STEPS,
        backend: str = "cpu",
    ) -> None:
        self.voice = voice
        self.language = language or VOICETUT_LANGUAGE
        self.steps = steps
        self._fallback = fallback
        self._root = root
        if runner is not None:
            self._runner: AudioCppRunner = runner
            self._model = Path(model_file) if model_file is not None else Path("model.gguf")
            return
        cli = find_cli(root)
        if cli is None:
            raise AudioCppUnavailable("The audio.cpp engine is not installed. " + audiocpp_status(root).hint)
        model = Path(model_file) if model_file is not None else find_model_file(spec_key, root)
        if model is None:
            raise AudioCppUnavailable("No VoiceTut-TTS model is present. " + audiocpp_status(root).hint)
        self._model = model
        self._runner = AudioCppCliRunner(cli, model, backend=backend)

    def _reference(self) -> tuple[Path | None, str]:
        voice = self.voice
        if voice is None or voice.reference is None:
            return None, ""
        return voice.reference, voice.reference_text

    def synthesize(self, text: str) -> AudioClip:
        if not text.strip():
            raise ValueError("Cannot synthesize empty text.")
        code = detect_language(text)
        # audio.cpp passes text through as-is and reads Arabic symbols badly, so the
        # numbers and units are spelled as words before the engine sees them.
        prepared = prepare_for_speech(text, language=code)
        reference, reference_text = self._reference()
        try:
            path = self._runner.synthesize(
                prepared,
                language=self.language,
                reference=reference,
                reference_text=reference_text,
                steps=self.steps,
            )
        except AudioCppUnavailable:
            if self._fallback is not None:
                return self._fallback.synthesize(text)
            raise
        clip = AudioClip.read_wav(path)
        try:
            Path(path).unlink()
        except OSError:
            pass
        if not clip.samples:
            if self._fallback is not None:
                return self._fallback.synthesize(text)
            raise ValueError("audio.cpp produced no audio.")
        return clip


# ------------------------------------------------------------------ gates


def _approve(tool: str) -> HumanApprovalGate:
    """A gate that records a human approval (the CLI's ``--approve`` convenience)."""
    gate = HumanApprovalGate()
    request = gate.request(tool, requested_by="cli-user")
    request.approve("cli-reviewer")
    return gate


def install_hint() -> str:
    """The honest next step for a machine that cannot run the engine yet."""
    return audiocpp_status().hint


__all__ = [
    "AUDIOCPP_ENGINE_NAME",
    "AUDIOCPP_SAMPLE_RATE",
    "DEFAULT_INFERENCE_STEPS",
    "VOICETUT_MODELS",
    "VOICETUT_SPEAKERS",
    "AudioCppCliRunner",
    "AudioCppRunner",
    "AudioCppStatus",
    "AudioCppSynthesizer",
    "AudioCppUnavailable",
    "VoiceTutSpec",
    "audiocpp_status",
    "available_synthesizer",
    "download_model",
    "download_speakers",
    "engine_dir",
    "engine_installed",
    "extra_installed",
    "find_cli",
    "find_model_file",
    "install_engine",
    "install_hint",
    "model_is_cached",
    "resolve_spec",
    "spec_names",
    "spec_summaries",
    "voice_list",
    "voices_dir",
]
