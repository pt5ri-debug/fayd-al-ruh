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
from core.voice_moods import Prosody
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

#: The language ids the audio.cpp OmniVoice family accepts. Anything else is
#: mapped onto one of these before the engine sees it: the engine rejects a bare
#: ``ar`` with "unsupported OmniVoice language", so a caller that passes the
#: project's own two-letter code must still be spoken rather than fail.
AUDIOCPP_LANGUAGES: tuple[str, ...] = ("ar", "arz", "en")


def normalize_language(language: str | None) -> str:
    """The audio.cpp language id for a project language code.

    The project speaks in two-letter codes (``ar``/``en``), but VoiceTut was
    trained with the Egyptian-Arabic id ``arz`` and the engine *rejects* a bare
    ``ar``. So an Arabic hint becomes ``arz`` -- the model's own language -- and an
    English hint stays ``en``; anything else falls back to the model's ``arz``
    default rather than an id the engine would refuse.
    """
    code = (language or "").strip().lower()
    if not code:
        return VOICETUT_LANGUAGE
    if code in AUDIOCPP_LANGUAGES:
        return "arz" if code == "ar" else code
    return VOICETUT_LANGUAGE


#: The rate VoiceTut renders at.
AUDIOCPP_SAMPLE_RATE = 24_000

#: The diffusion steps to use per request. The model's own default is 32, which is
#: slow on a CPU. Measured on this machine (a resident server, so load is out of
#: the picture): 2 steps -> 2.4 s, 4 steps -> 4.3 s, 8 steps -> 8.6 s for the same
#: short sentence, i.e. wall time is very nearly linear in the step count. Four
#: steps halves the wait against eight and keeps the speech intelligible, so it is
#: the default; ``steps`` is a parameter, so a caller who wants the finer detail of
#: eight (or the very fast two) can ask for it.
DEFAULT_INFERENCE_STEPS = 4

#: The GGUF magic every real model file starts with. A truncated download or a
#: renamed non-model file does not carry it, so it is checked before a file is
#: handed to the engine -- otherwise the engine is left to load a bad file, which
#: is exactly the "it hangs and never answers" failure this guards against.
GGUF_MAGIC = b"GGUF"

#: How many backend threads the engine may use on the CPU path. The engine's own
#: default is 4, which leaves most of a modern machine idle and makes a CPU read
#: far slower than it needs to be; the cap keeps a big machine from oversubscribing.
DEFAULT_MAX_THREADS = 8

#: The generator performance mode the engine runs with. ``flash_attention`` is the
#: engine's own documented performance path (its default, ``off``, is the
#: "exact-safe" one), and it is the single biggest lever on CPU read speed. It is
#: named here rather than left implicit so a caller can ask for ``off`` when the
#: exact-safe numerics matter more than the speed.
DEFAULT_PERF_MODE = "flash_attention"

#: The ``--session-option`` values the performance mode may be set to.
PERF_MODES: tuple[str, ...] = ("off", "flash_attention")


def _default_threads() -> int:
    """The engine's worker-thread count: the machine's cores, capped."""
    import os

    return max(1, min(os.cpu_count() or 4, DEFAULT_MAX_THREADS))


def is_valid_gguf(path: Path | str) -> bool:
    """True when ``path`` really is a GGUF model file (not truncated or a stray file).

    Only the 4-byte magic is checked -- cheap, and it needs no model load -- but it
    is enough to refuse a partial download or a renamed file *before* the engine
    tries to load it, so the caller reports the real next step instead of hanging.
    """
    try:
        with Path(path).open("rb") as handle:
            return handle.read(len(GGUF_MAGIC)) == GGUF_MAGIC
    except OSError:
        return False


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
#: model spec are all in the one file, so one file is all that is needed. Whether
#: the engine also needs a ``--family`` is decided from the file itself
#: (:func:`_model_family`), so a hand-built package works without a code change.
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

#: The file the user's own imported/cloned voices are registered in, beside the
#: reference clips (``<data>/audio.cpp/imported_voices.json``).
IMPORTED_VOICES_FILE = "imported_voices.json"

#: The audio formats an imported voice may arrive in. A ``.wav`` is read directly;
#: another format is converted with ``ffmpeg`` or ``sox`` when one is installed, and
#: refused honestly when none is -- audio.cpp itself clones from a WAV reference.
VOICE_CLIP_SUFFIXES: tuple[str, ...] = (".wav", ".mp3", ".m4a", ".flac", ".ogg", ".opus", ".aac")

#: What the import picker offers, including the ``.onnx`` the user asked to accept
#: (an ONNX voice belongs to the *Piper* studio; VoiceTut clones from a clip, so an
#: ONNX file is reported with that next step rather than stored as a dead asset).
IMPORT_PICKER_SUFFIXES: tuple[str, ...] = (*VOICE_CLIP_SUFFIXES, ".onnx")

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


def find_server(root: Path | str | None = None) -> Path | None:
    """The ``audiocpp_server`` executable, or ``None`` when this build lacks it.

    The server is the *speed* path: it keeps the model resident, so a document is
    read without re-loading the weights for every paragraph. A build that ships only
    the CLI still works through the single-process path, so a missing server is not
    an error -- it is simply the slower, honest fallback.
    """
    local = engine_dir(root) / (
        f"{SERVER_BINARY_NAME}.exe" if _is_windows() else SERVER_BINARY_NAME
    )
    if local.is_file():
        return local
    found = shutil.which(SERVER_BINARY_NAME)
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


def _read_gguf_family(path: Path) -> str | None:
    """The model family a GGUF file declares, or ``None`` when it declares none.

    A standalone package (the ones VoiceTut ships) embeds its own package spec, so
    the engine can load it with no ``--family``. A GGUF that names no family -- or
    that cannot be read -- needs the family spelled out instead, and getting that
    wrong is exactly what makes the engine fail to load the model at all. Only the
    header is read: the metadata key/value block, never the tensor data.
    """
    import struct

    try:
        with Path(path).open("rb") as handle:
            header = handle.read(24)
            if len(header) < 24 or header[:4] != GGUF_MAGIC:
                return None
            version = struct.unpack_from("<I", header, 4)[0]
            if version < 2:
                return None
            tensor_count, metadata_count = struct.unpack_from("<QQ", header, 8)
            if tensor_count > 1_000_000 or metadata_count > 100_000:
                return None
            return _scan_gguf_family(handle, metadata_count)
    except (OSError, ValueError, struct.error):
        return None


def _scan_gguf_family(handle: Any, metadata_count: int) -> str | None:
    """Walk the GGUF metadata block for the model family the engine must be told.

    A ``*.model_spec.family`` key wins over everything: that is the value the engine
    itself reads from the embedded spec, and the real VoiceTut packages carry it
    (``audiocpp.model_spec.family = omnivoice``) while ``general.architecture`` says
    the *container* format (``audiocpp``), which the engine rejects as a family hint
    -- naming that one is exactly what made loading fail. A ``family`` key is the
    next best; ``general.architecture`` is used only when it names a known model
    family and not the generic ``audiocpp`` container. Stops quietly at the first
    entry it cannot decode, so an unexpected file yields ``None`` rather than an
    exception.
    """
    import struct

    def read_string() -> str:
        raw = handle.read(8)
        if len(raw) < 8:
            raise ValueError("truncated GGUF string length")
        length = struct.unpack("<Q", raw)[0]
        if length > 1 << 20:
            raise ValueError("implausible GGUF string length")
        return handle.read(length).decode("utf-8", errors="replace")

    def skip_value(value_type: int) -> None:
        sizes = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}
        if value_type in sizes:
            handle.read(sizes[value_type])
            return
        if value_type == 8:  # string
            read_string()
            return
        if value_type == 9:  # array
            header = handle.read(12)
            if len(header) < 12:
                raise ValueError("truncated GGUF array")
            element_type, count = struct.unpack("<IQ", header)
            for _ in range(count):
                skip_value(element_type)
            return
        raise ValueError("unknown GGUF value type")

    spec_family: str | None = None
    plain_family: str | None = None
    architecture: str | None = None
    for _ in range(metadata_count):
        try:
            key = read_string()
            raw = handle.read(4)
            if len(raw) < 4:
                break
            value_type = struct.unpack("<I", raw)[0]
            if value_type != 8:
                skip_value(value_type)
                continue
            value = read_string().strip()
            lowered = key.lower()
            if not value:
                continue
            if lowered.endswith("model_spec.family") or lowered.endswith(".spec.family"):
                spec_family = spec_family or value
            elif "family" in lowered:
                plain_family = plain_family or value
            elif "architecture" in lowered:
                architecture = architecture or value
        except (ValueError, OSError):
            break
    if spec_family is not None:
        return spec_family
    if plain_family is not None:
        return plain_family
    if architecture is not None and architecture.lower() not in _CONTAINER_ARCHITECTURES:
        return architecture
    return None


#: ``general.architecture`` values that name audio.cpp's *container* format rather
#: than a model family: the engine rejects them as a family hint, so they are never
#: returned as one (the embedded ``*.model_spec.family`` is the real answer).
_CONTAINER_ARCHITECTURES: frozenset[str] = frozenset({"audiocpp"})


def _model_family(path: Path) -> str | None:
    """The ``--family`` a model file needs, or ``None`` when it needs none.

    A standalone package -- the ones VoiceTut ships -- embeds its own package spec
    (``audiocpp.model_spec.family = omnivoice``), so naming a family is redundant; the
    file is left to describe itself. A file that declares some *other* family is
    loaded through that family, and a file that declares no family at all keeps the
    VoiceTut default, so a hand-built package behaves as it did before the file could
    be inspected.
    """
    declared = _read_gguf_family(path)
    if declared is None:
        return VOICETUT_FAMILY
    lowered = declared.lower()
    if "omnivoice" in lowered or "voicetut" in lowered:
        return None
    if lowered in _CONTAINER_ARCHITECTURES:
        return None
    return declared


def find_model_file(key: str | None = None, root: Path | str | None = None) -> Path | None:
    """The GGUF file for a VoiceTut package, or the first installed one.

    Prefers the file the registry names (``model.json``, written by
    :func:`download_model`); a ``.gguf`` the user dropped in by hand is still found,
    so the engine never pretends nothing is installed. A file that is not really a
    GGUF (a truncated download, or a renamed file) is skipped rather than handed to
    the engine, because loading one is what leaves a read hanging with no error.
    """
    base = _models_root(root)
    if key:
        return _locate_valid(base, resolve_spec(key).filename)
    registered = _read_registry(root).get("model")
    if registered:
        try:
            found = _locate_valid(base, resolve_spec(registered).filename)
        except ValueError:
            found = None
        if found is not None:
            return found
    hits = sorted(base.rglob("*.gguf")) if base.is_dir() else []
    for hit in hits:
        if is_valid_gguf(hit):
            return hit
    return None


def _locate(base: Path, filename: str) -> Path | None:
    direct = base / filename
    if direct.is_file():
        return direct
    hits = sorted(base.rglob(filename)) if base.is_dir() else []
    return hits[0] if hits else None


def _locate_valid(base: Path, filename: str) -> Path | None:
    """Like :func:`_locate`, but only when the file is a real GGUF."""
    found = _locate(base, filename)
    return found if found is not None and is_valid_gguf(found) else None


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
    # A partial file left by an interrupted download must be fetched again rather
    # than trusted: an incomplete GGUF is what leaves the engine loading forever.
    if target.is_file() and not is_valid_gguf(target):
        _unlink_quietly(target)
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
        if not is_valid_gguf(target):
            _unlink_quietly(target)
            raise AudioCppUnavailable(
                f"{spec.filename} downloaded but is not a usable GGUF file (the download "
                "was interrupted). Try again, or fetch it by hand and drop it into "
                f"{directory}."
            )
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

    A zero-shot cloning model needs a reference clip **and its transcript** to speak
    a named voice: audio.cpp's OmniVoice prompt builder refuses a clone request with
    no ``reference_text`` ("native voice clone currently requires reference_text when
    reference audio is provided"). So this fetches the upstream studio speakers -- a
    few hundred KB each -- plus the ``references.json`` that carries their
    transcripts, and a missing transcript file is a hard error rather than a silent
    skip: without it every built-in voice is listed but cannot be spoken, which is
    exactly the "the voices do not work" report this fixes.

    The transcripts live *inside* ``reference_speakers/`` upstream (not at the repo
    root), so that is the URL used. A previously downloaded copy in the model folder
    is still honoured, so a machine that fetched it by hand is not asked again.
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
    _download_reference_texts(root)
    return directory


def _download_reference_texts(root: Path | str | None = None) -> Path:
    """Fetch ``references.json`` -- the transcripts a cloning voice needs.

    Tries the location upstream really uses (``reference_speakers/references.json``)
    and then the repo root, and refuses honestly when neither answers: a voice with a
    clip but no transcript cannot be spoken by the engine, so a missing file must be
    reported rather than quietly skipped. A copy already in the model folder counts,
    so a hand-placed file is never overwritten.
    """
    target = _models_root(root) / "references.json"
    if target.is_file() and _reference_texts(root):
        return target
    candidates = (
        f"https://huggingface.co/{SPEAKERS_REPO}/resolve/main/reference_speakers/references.json",
        f"https://huggingface.co/{SPEAKERS_REPO}/resolve/main/references.json",
    )
    last_error: Exception | None = None
    for url in candidates:
        try:
            urllib.request.urlretrieve(url, target)
        except OSError as error:
            last_error = error
            continue
        if _reference_texts(root):
            return target
        _unlink_quietly(target)
    raise AudioCppUnavailable(
        "Could not download references.json, the transcripts the built-in voices need "
        f"({last_error}). Without it a cloning voice cannot be spoken; fetch "
        f"{candidates[0]} by hand and drop it into {_models_root(root)}."
    )



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

    Returns ``()`` unless the model is installed *and* a reference clip **with its
    transcript** is present, because a cloning engine cannot speak a named voice
    without both: audio.cpp's OmniVoice builder refuses a clone request that has a
    reference clip but no ``reference_text``. Offering a voice the engine would
    refuse is exactly the dishonesty this project avoids -- and was the "the voices
    do not work" report, because a speaker was listed from its clip alone.
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
        transcript = texts.get(stem, "")
        if not transcript:
            # A clip with no transcript cannot be cloned, so it is not offered.
            continue
        voices.append(
            ModelVoice(
                name=name,
                engine=AUDIOCPP_ENGINE_NAME,
                language=VOICETUT_LANGUAGE,
                gender=gender,
                model="VoiceTut-TTS",
                reference=clip,
                reference_text=transcript,
            )
        )
    voices.extend(imported_voice_models(root))
    return tuple(voices)


# --------------------------------------------------- the user's own voices


@dataclass(frozen=True)
class ImportedVoice:
    """A voice the user added to VoiceTut: a reference clip the engine clones.

    VoiceTut is a *zero-shot cloning* model: it speaks in the voice of a short
    reference clip. So a user's own voice is a clip (recorded here, or imported
    from a file) plus, optionally, its transcript -- exactly the shape the model's
    built-in speakers take. The clip is copied into the model's own voices folder
    and registered here, so it survives a restart and shows in the menu beside the
    built-in speakers.
    """

    name: str
    clip: str
    transcript: str = ""
    source: str = ""

    def to_primitive(self) -> dict[str, str]:
        return {
            "name": self.name,
            "clip": self.clip,
            "transcript": self.transcript,
            "source": self.source,
        }

    @classmethod
    def from_primitive(cls, data: object) -> ImportedVoice | None:
        """Read one registry row, or ``None`` when it is not usable."""
        if not isinstance(data, dict):
            return None
        name = str(data.get("name") or "").strip()
        clip = str(data.get("clip") or "").strip()
        if not name or not clip:
            return None
        return cls(
            name=name,
            clip=clip,
            transcript=str(data.get("transcript") or "").strip(),
            source=str(data.get("source") or "").strip(),
        )


def imported_voices_path(root: Path | str | None = None) -> Path:
    """The registry file holding the user's imported/cloned voices."""
    return _models_root(root) / IMPORTED_VOICES_FILE


def load_imported_voices(root: Path | str | None = None) -> tuple[ImportedVoice, ...]:
    """Every voice the user added, in the order they were added.

    A missing or unreadable registry yields ``()`` rather than raising, so a stale
    file can never make a menu unusable; a row whose clip has been deleted is
    dropped, so the menu never offers a voice that cannot be spoken.
    """
    path = imported_voices_path(root)
    if not path.is_file():
        return ()
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return ()
    rows = data.get("voices") if isinstance(data, dict) else data
    if not isinstance(rows, list):
        return ()
    directory = voices_dir(root)
    voices: list[ImportedVoice] = []
    for row in rows:
        voice = ImportedVoice.from_primitive(row)
        if voice is None:
            continue
        if not (directory / voice.clip).is_file():
            continue
        voices.append(voice)
    return tuple(voices)


def _save_imported_voices(
    voices: list[ImportedVoice], root: Path | str | None = None
) -> None:
    """Write the registry (best-effort: a read-only data folder must not crash)."""
    from core.best_effort import best_effort

    path = imported_voices_path(root)
    payload = {"voices": [voice.to_primitive() for voice in voices]}
    with best_effort("save the imported VoiceTut voices"):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _to_wav(source: Path, target: Path) -> None:
    """Ensure ``source`` is a WAV at ``target``, converting with ffmpeg/sox if needed.

    A WAV is copied as-is. Any other audio format needs a converter; ``ffmpeg`` is
    preferred and ``sox`` is the fallback, and neither being installed is reported
    as the real next step rather than a decode error. A failure here is a
    :class:`ValueError` -- a bad *choice* of clip -- so the CLI and the window
    report it as such instead of as an engine fault.
    """
    import shutil as _shutil
    import subprocess

    if source.suffix.lower() == ".wav":
        _shutil.copyfile(source, target)
        return
    converters = (
        ("ffmpeg", ["ffmpeg", "-y", "-i", str(source), str(target)]),
        ("sox", ["sox", str(source), str(target)]),
    )
    for executable, argv in converters:
        if _shutil.which(executable):
            completed = subprocess.run(argv, capture_output=True, text=True, check=False)
            if completed.returncode == 0 and target.is_file():
                return
            raise ValueError(
                f"{executable} could not convert {source.name} to WAV: "
                f"{(completed.stderr or '').strip().splitlines()[-1:] or 'no output'}"
            )
    raise ValueError(
        f"{source.name} is not a WAV and no converter is installed. Install ffmpeg "
        "(or sox), or save the clip as a .wav first."
    )


def import_voice(
    source: Path | str | AudioClip,
    *,
    name: str = "",
    transcript: str = "",
    root: Path | str | None = None,
) -> ImportedVoice:
    """Copy an audio file (or a recorded clip) in as a new VoiceTut voice.

    Accepts any common audio format (a WAV directly; another format through
    ``ffmpeg``/``sox``), and also a live :class:`~core.speech.AudioClip` -- the
    microphone's own output -- so the window's *Clone my voice* recording reaches
    the same registry without a round trip through a file. An ``.onnx`` file -- the
    user asked to accept one -- is refused with the honest next step: an ONNX voice
    belongs to the *Piper* studio, because VoiceTut clones from a reference clip and
    cannot load one. Raises :class:`ValueError` for a missing file, an unusable
    format, or audio that does not decode, so a bad choice is reported rather than
    stored.
    """
    if isinstance(source, AudioClip):
        if source.is_silent:
            raise ValueError("The recording decoded to silence; record an audible clip.")
        return _register_clip(source, name=name or "My voice", transcript=transcript, root=root)

    path = Path(source).expanduser()
    if not path.is_file():
        raise ValueError(f"There is no file at {path}.")
    suffix = path.suffix.lower()
    if suffix == ".onnx":
        raise ValueError(
            "VoiceTut clones a voice from an audio clip, so an .onnx voice cannot be "
            "loaded here. Use the Piper studio for an .onnx voice, or import a "
            "recording (.wav, .mp3, .m4a, .flac, .ogg)."
        )
    if suffix not in VOICE_CLIP_SUFFIXES:
        raise ValueError(
            "Choose an audio clip ("
            + ", ".join(VOICE_CLIP_SUFFIXES)
            + f"); got {suffix or 'no extension'}."
        )

    label = (name or path.stem).strip() or path.stem
    target = voices_dir(root) / f"user-{_voice_slug(label)}.wav"
    target.parent.mkdir(parents=True, exist_ok=True)
    _to_wav(path, target)
    # Prove the clip really decodes, so a corrupt file is refused now rather than
    # failing on the first read.
    try:
        clip = AudioClip.read_wav(target)
    except (OSError, ValueError) as error:
        _unlink_quietly(target)
        raise ValueError(f"{path.name} is not usable audio: {error}") from error
    if clip.is_silent:
        _unlink_quietly(target)
        raise ValueError(f"{path.name} decoded to silence; choose an audible clip.")
    return _register_clip(clip, name=label, transcript=transcript, root=root, source=str(path))


def _register_clip(
    clip: AudioClip,
    *,
    name: str,
    transcript: str = "",
    root: Path | str | None = None,
    source: str = "",
) -> ImportedVoice:
    """Write ``clip`` into the voices folder and register it under ``name``."""
    directory = voices_dir(root)
    directory.mkdir(parents=True, exist_ok=True)
    label = (name or "My voice").strip() or "My voice"
    target = directory / f"user-{_voice_slug(label)}.wav"
    clip.write_wav(target)
    existing = [v for v in load_imported_voices(root) if v.name.lower() != label.lower()]
    voice = ImportedVoice(name=label, clip=target.name, transcript=transcript.strip(), source=source)
    existing.append(voice)
    _save_imported_voices(existing, root)
    return voice


def _voice_slug(label: str) -> str:
    """A filesystem-safe slug for an imported voice's clip filename."""
    slug = "".join(ch if ch.isalnum() or ch in "-_" else "-" for ch in label.strip().lower())
    slug = "-".join(part for part in slug.split("-") if part)
    return slug or "voice"


def clone_voice(
    recording: Path | str | AudioClip,
    *,
    name: str = "",
    transcript: str = "",
    root: Path | str | None = None,
) -> ImportedVoice:
    """Register a recording of the user's own voice as a VoiceTut voice.

    This is the "merge my voice into the model" step the user asked for, done the
    way a zero-shot cloning model actually works: the recording becomes a reference
    clip, and every later read is spoken in that voice. Accepts a file path or the
    microphone's own :class:`~core.speech.AudioClip`. Thin wrapper over
    :func:`import_voice` so the window and the CLI share one path.
    """
    return import_voice(recording, name=name or "My voice", transcript=transcript, root=root)


def set_imported_voice_transcript(
    name: str, transcript: str, root: Path | str | None = None
) -> ImportedVoice | None:
    """Give an imported voice the transcript of its clip, and return it.

    A zero-shot cloning voice needs the clip **and** what the clip says; a voice
    recorded or imported without a transcript is registered but cannot be spoken,
    so this is the one call the studio and the CLI use to fill the gap -- normally
    with the speech-to-text engine's own reading of the clip. An empty transcript,
    or a name that is not registered, changes nothing and returns ``None``.
    """
    value = transcript.strip()
    if not value:
        return None
    wanted = name.strip().lower()
    voices = list(load_imported_voices(root))
    for index, voice in enumerate(voices):
        if voice.name.lower() != wanted:
            continue
        if voice.transcript == value:
            return voice
        updated = ImportedVoice(
            name=voice.name, clip=voice.clip, transcript=value, source=voice.source
        )
        voices[index] = updated
        _save_imported_voices(voices, root)
        return updated
    return None


def remove_imported_voice(name: str, root: Path | str | None = None) -> bool:
    """Remove an imported voice (and its clip); ``True`` when one was removed."""
    wanted = name.strip().lower()
    kept: list[ImportedVoice] = []
    removed = False
    for voice in load_imported_voices(root):
        if voice.name.lower() == wanted:
            removed = True
            _unlink_quietly(voices_dir(root) / voice.clip)
            continue
        kept.append(voice)
    if removed:
        _save_imported_voices(kept, root)
    return removed


def imported_voice_models(root: Path | str | None = None) -> tuple[ModelVoice, ...]:
    """The user's own voices as :class:`~core.voices.ModelVoice` objects."""
    directory = voices_dir(root)
    return tuple(
        ModelVoice(
            name=voice.name,
            engine=AUDIOCPP_ENGINE_NAME,
            language=VOICETUT_LANGUAGE,
            gender="",
            model="VoiceTut-TTS",
            reference=directory / voice.clip,
            reference_text=voice.transcript,
            title=voice.name,
        )
        for voice in load_imported_voices(root)
    )


def _unlink_quietly(path: Path | None) -> None:
    """Remove ``path`` if it is there, ignoring an ``OSError``."""
    import contextlib

    if path is None:
        return
    with contextlib.suppress(OSError):
        path.unlink()


def _clip_sort_key(path: Path) -> tuple[int, str]:
    """Order engine output clips by their trailing number, not lexically.

    ``speech-2.wav`` must come before ``speech-10.wav``: a plain string sort would
    put ``10`` first, so a batch of ten or more chunks would be spoken out of order.
    """
    stem = path.stem
    digits = ""
    for character in reversed(stem):
        if not character.isdigit():
            break
        digits = character + digits
    return (int(digits) if digits else 0, stem)


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
        # Distinguish "nothing there" from "a file is there but unusable": a partial
        # GGUF that the engine cannot load is what made a read look frozen.
        broken = _first_gguf_file(root)
        if broken is not None:
            hint = (
                f"The model file {broken.name} is not a usable GGUF (an interrupted "
                "download, or the wrong file). Remove it and re-download with: "
                "python -m cli audio download --approve"
            )
        else:
            hint = "Download the model with: python -m cli audio download --approve"
    elif voices == 0:
        # Distinguish "no voices at all" from "clips are there but their transcripts
        # are not": a cloning voice needs both, so the second case has a different
        # next step (re-fetch the speakers, which brings references.json).
        clips = list(voices_dir(root).glob("*.wav")) if voices_dir(root).is_dir() else []
        if clips:
            hint = (
                "The reference clips are present but their transcripts "
                "(references.json) are not, and a cloning voice needs both. "
                "Re-download them with: python -m cli audio download --speakers --approve"
            )
        else:
            hint = (
                "Download the built-in voices with: "
                "python -m cli audio download --speakers --approve"
            )
    else:
        hint = "VoiceTut-TTS over audio.cpp is ready."
    return AudioCppStatus(cli is not None, cli, model is not None, model, voices, hint)


def _first_gguf_file(root: Path | str | None = None) -> Path | None:
    """Any ``.gguf`` on disk, valid or not (so a corrupt one can be named)."""
    base = _models_root(root)
    hits = sorted(base.rglob("*.gguf")) if base.is_dir() else []
    return hits[0] if hits else None


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


#: The port the local audio.cpp server is started on when the project runs one
#: itself. Chosen away from the common 8080 so it does not collide with a server
#: the user already runs.
DEFAULT_SERVER_PORT = 8199

#: The engine binary that serves an OpenAI-compatible API with the model kept
#: resident. It ships beside ``audiocpp_cli`` in the same release.
SERVER_BINARY_NAME = "audiocpp_server"

#: The OpenAI-compatible path the resident server speaks on.
SERVER_SPEECH_PATH = "/v1/audio/speech"

#: The resident servers this process started, so they can be stopped when it exits.
#: A server the *user* was already running is deliberately not in here.
_STARTED_SERVERS: list[Any] = []


def _register_server(runner: Any) -> None:
    """Remember a server this process started, and reap it at interpreter exit."""
    import atexit

    if runner in _STARTED_SERVERS:
        return
    _STARTED_SERVERS.append(runner)
    if len(_STARTED_SERVERS) == 1:
        atexit.register(_stop_started_servers)


def _stop_started_servers() -> None:
    """Terminate every server this process started (idempotent)."""
    while _STARTED_SERVERS:
        runner = _STARTED_SERVERS.pop()
        try:
            runner.stop()
        except Exception:
            # Exit cleanup must never raise: a stuck process has already been killed
            # by ``stop`` itself, and anything left is the OS's to reap.
            pass


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
    """An :class:`AudioCppCliRunner` that drives the real ``audiocpp_cli`` binary.

    The text is written to a JSON request file rather than passed on the command
    line: a long Arabic answer would otherwise hit the platform's argument-length
    limit, and the JSON route also keeps the reference transcript and the language id
    off the command line.

    **One invocation per batch, not per chunk.** audio.cpp's own ``--request-sequence``
    is built to "run multiple JSON requests through one long-lived offline model
    session", and that is the difference between a document being read and a document
    looking frozen: every separate process reloads the whole GGUF (1.35 GB for the
    recommended Q8_0 package) before it can speak, so a paragraph-per-process read
    spent all its time loading weights. :meth:`synthesize_many` therefore sends a
    whole batch of chunks as one sequence and returns the clips in order, so the model
    is loaded once for the batch.
    """

    def __init__(
        self,
        cli: Path | str,
        model: Path | str,
        *,
        backend: str = "cpu",
        timeout: int = 900,
        threads: int | None = None,
        family: str | None = None,
        perf_mode: str = DEFAULT_PERF_MODE,
    ) -> None:
        self.cli = Path(cli)
        self.model = Path(model)
        self.backend = backend
        self.timeout = timeout
        self.threads = threads if threads is not None else _default_threads()
        self.perf_mode = perf_mode if perf_mode in PERF_MODES else DEFAULT_PERF_MODE
        # A standalone GGUF carries its own package spec, so no ``--family`` is
        # named for it; a package that does not is loaded through the family
        # implementation, and naming the wrong one (or none) is what leaves the
        # engine unable to load the model.
        self.family = family if family is not None else _model_family(self.model)

    def synthesize(
        self,
        text: str,
        *,
        language: str,
        reference: Path | None,
        reference_text: str,
        steps: int,
    ) -> Path:
        return self.synthesize_many(
            [text],
            language=language,
            reference=reference,
            reference_text=reference_text,
            steps=steps,
        )[0]

    def session_options(self) -> tuple[str, ...]:
        """The engine ``--session-option`` values this engine should run with.

        The generator's performance attention path is the engine's own documented
        speed lever (its default keeps the "exact-safe" one), so it is turned on
        here; the weight storage types are left at the package's own precision,
        because the recommended Q8_0 GGUF is already quantized and re-quantizing
        SafeTensors at load time is a documented quality risk on some paths.
        """
        return (f"omnivoice.perf_mode={self.perf_mode}",)

    def synthesize_many(
        self,
        texts: list[str],
        *,
        language: str,
        reference: Path | None,
        reference_text: str,
        steps: int,
    ) -> list[Path]:
        """Render several texts in **one** engine session, returning their WAV paths.

        The texts keep their order (each is a numbered request, and the outputs are
        matched back by id), so a batch is spoken as the sequence the caller sent.
        An empty list is answered without launching the engine at all.
        """
        import subprocess

        if not texts:
            return []
        requests: list[dict[str, Any]] = []
        for index, text in enumerate(texts):
            request: dict[str, Any] = {
                "id": f"speech-{index}",
                "text": text,
                "language": language,
                "options": {"num_inference_steps": steps},
            }
            if reference is not None:
                request["voice_ref"] = str(reference)
                if reference_text:
                    request["reference_text"] = reference_text
            requests.append(request)
        with tempfile.TemporaryDirectory(prefix="audiocpp-run-") as scratch:
            scratch_path = Path(scratch)
            sequence = scratch_path / "request.json"
            sequence.write_text(
                json.dumps({"requests": requests}, ensure_ascii=False), encoding="utf-8"
            )
            out_dir = scratch_path / "out"
            out_dir.mkdir()
            command = [
                str(self.cli),
                "--task",
                "tts",
                "--model",
                str(self.model),
                "--backend",
                self.backend,
                "--threads",
                str(self.threads),
                "--request-sequence",
                str(sequence),
                "--out-dir",
                str(out_dir),
            ]
            if self.family:
                command[3:3] = ["--family", self.family]
            for option in self.session_options():
                command.extend(["--session-option", option])
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
            except subprocess.TimeoutExpired as error:
                raise AudioCppUnavailable(
                    f"audio.cpp did not finish within {self.timeout}s. A first read loads "
                    "the whole model, so give it longer, use a smaller package "
                    "(voicetut-f16 is larger; the Q8_0 one is smaller), or free some "
                    "memory."
                ) from error
            except (OSError, subprocess.SubprocessError) as error:
                raise AudioCppUnavailable(f"audio.cpp could not run ({error}).") from error
            if completed.returncode != 0:
                detail = (completed.stderr or completed.stdout or "").strip().splitlines()
                tail = detail[-1] if detail else "no output"
                raise AudioCppUnavailable(f"audio.cpp failed (exit {completed.returncode}): {tail}")
            kept: list[Path] = []
            for index in range(len(texts)):
                produced = out_dir / f"speech-{index}.wav"
                if not produced.is_file():
                    # An engine that ignores the ids still writes the clips; fall back
                    # to the files it did produce rather than losing the whole batch.
                    # Sorted by their trailing number, not lexically, so a batch of 10+
                    # chunks keeps its order (speech-2 before speech-10).
                    fallback = sorted(out_dir.glob("*.wav"), key=_clip_sort_key)
                    if index < len(fallback):
                        produced = fallback[index]
                if not produced.is_file():
                    raise AudioCppUnavailable(
                        f"audio.cpp produced no audio file for request {index}."
                    )
                # The clip must outlive the temporary folder, so move it beside the
                # model rather than returning a path that is about to be deleted.
                keep = Path(tempfile.mkstemp(prefix="audiocpp-", suffix=".wav")[1])
                shutil.copyfile(produced, keep)
                kept.append(keep)
            return kept
class AudioCppServerRunner:
    """An :class:`AudioCppRunner` that keeps the model resident in a local server.

    Every ``audiocpp_cli`` process reloads the whole GGUF before it can speak, and a
    document is read paragraph by paragraph, so a read spends real time re-loading
    1.35 GB of weights. audio.cpp ships ``audiocpp_server``, which loads a model
    **once** and answers OpenAI-compatible ``POST /v1/audio/speech`` requests from
    memory -- the same lever the ``--request-sequence`` batching pulls, but across
    the whole session rather than one batch.

    The server is started lazily on first use and left running with
    ``--max-loaded-models 1`` so the weights stay warm; the object is safe to reuse
    for a whole read (and a whole session). It is **opt-in** -- the CLI's default
    runner is unchanged -- so a machine that never asked for it starts no daemon.
    The server only listens on the loopback address, so nothing leaves the machine.
    """

    def __init__(
        self,
        server: Path | str,
        model: Path | str,
        *,
        backend: str = "cpu",
        timeout: int = 900,
        threads: int | None = None,
        family: str | None = None,
        perf_mode: str = DEFAULT_PERF_MODE,
        host: str = "127.0.0.1",
        port: int = DEFAULT_SERVER_PORT,
        model_id: str = "voicetut",
    ) -> None:
        self.server = Path(server)
        self.model = Path(model)
        self.backend = backend
        self.timeout = timeout
        self.threads = threads if threads is not None else _default_threads()
        self.perf_mode = perf_mode if perf_mode in PERF_MODES else DEFAULT_PERF_MODE
        self.family = family if family is not None else _model_family(self.model)
        self.host = host
        self.port = port
        self.model_id = model_id
        self._process: Any = None
        self._config: Path | None = None

    # ------------------------------------------------------------ lifecycle

    def base_url(self) -> str:
        return f"http://{self.host}:{self.port}"

    def is_running(self) -> bool:
        """Whether a server answers ``/health`` on this port right now."""
        import urllib.request

        try:
            with urllib.request.urlopen(f"{self.base_url()}/health", timeout=3) as response:
                return response.status == 200
        except (OSError, ValueError):
            return False

    def start(self) -> None:
        """Start the server with this model preloaded, unless one is already up."""
        import subprocess
        import tempfile

        if self.is_running():
            return
        config = {
            "host": self.host,
            "port": self.port,
            "backend": self.backend,
            "threads": self.threads,
            "max_loaded_models": 1,
            "models": [
                {
                    "id": self.model_id,
                    "path": str(self.model),
                    # The server config requires a family key even for a standalone
                    # package that resolves its own spec, so the VoiceTut family is the
                    # default here (the engine then reads the embedded spec anyway).
                    "family": self.family or VOICETUT_FAMILY,
                }
            ],
        }
        handle = tempfile.NamedTemporaryFile(
            "w", prefix="audiocpp-server-", suffix=".json", delete=False, encoding="utf-8"
        )
        try:
            json.dump(config, handle, ensure_ascii=False)
            handle.close()
            self._config = Path(handle.name)
            self._process = subprocess.Popen(
                [str(self.server), "--config", str(self._config)],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                cwd=str(self.server.parent),
                env=_engine_env(self.server.parent),
            )
        except (OSError, ValueError) as error:
            _unlink_quietly(self._config)
            self._config = None
            raise AudioCppUnavailable(
                f"Could not start the audio.cpp server ({error})."
            ) from error
        # Loading 1.35 GB of weights takes a few seconds; poll until it answers.
        import time

        deadline = time.monotonic() + min(self.timeout, 180)
        while time.monotonic() < deadline:
            if self.is_running():
                _register_server(self)
                return
            if self._process.poll() is not None:
                raise AudioCppUnavailable(
                    "The audio.cpp server exited while loading the model. Use the "
                    "single-process engine instead (python -m cli audio say ...)."
                )
            time.sleep(0.5)
        self.stop()
        raise AudioCppUnavailable("The audio.cpp server did not become ready in time.")

    def stop(self) -> None:
        """Stop a server this runner started; leave one the user was already running."""
        import subprocess

        process = self._process
        self._process = None
        if process is not None and process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=10)
            except (OSError, subprocess.SubprocessError):
                process.kill()
        if self._config is not None:
            _unlink_quietly(self._config)
            self._config = None

    # -------------------------------------------------------------- rendering

    def synthesize(
        self,
        text: str,
        *,
        language: str,
        reference: Path | None,
        reference_text: str,
        steps: int,
    ) -> Path:
        return self.synthesize_many(
            [text],
            language=language,
            reference=reference,
            reference_text=reference_text,
            steps=steps,
        )[0]

    def synthesize_many(
        self,
        texts: list[str],
        *,
        language: str,
        reference: Path | None,
        reference_text: str,
        steps: int,
    ) -> list[Path]:
        """Render each text through the resident server, in order."""
        import urllib.error
        import urllib.request

        if not texts:
            return []
        self.start()
        kept: list[Path] = []
        for text in texts:
            payload: dict[str, Any] = {
                "model": self.model_id,
                "input": text,
                "language": language,
                "response_format": "wav",
                "num_inference_steps": steps,
            }
            if reference is not None:
                payload["voice_ref"] = str(reference)
                if reference_text:
                    payload["reference_text"] = reference_text
            request = urllib.request.Request(
                f"{self.base_url()}{SERVER_SPEECH_PATH}",
                data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            try:
                with urllib.request.urlopen(request, timeout=self.timeout) as response:
                    audio = response.read()
            except urllib.error.HTTPError as error:
                detail = error.read().decode("utf-8", errors="replace").strip()
                raise AudioCppUnavailable(
                    f"audio.cpp server refused the request ({error.code}): {detail[:200]}"
                ) from error
            except (OSError, ValueError) as error:
                raise AudioCppUnavailable(
                    f"The audio.cpp server could not be reached ({error})."
                ) from error
            target = Path(tempfile.mkstemp(prefix="audiocpp-", suffix=".wav")[1])
            target.write_bytes(audio)
            kept.append(target)
        return kept


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
        threads: int | None = None,
        prosody: Prosody | None = None,
        perf_mode: str = DEFAULT_PERF_MODE,
        use_server: bool = False,
    ) -> None:
        self.voice = voice
        # The engine only accepts the model's own language ids (``arz``/``en``), and
        # rejects a bare ``ar``; the project's two-letter codes are mapped here once.
        self.language = normalize_language(language)
        self.steps = steps
        self._fallback = fallback
        self._root = root
        self._server_runner: AudioCppServerRunner | None = None
        # A zero-shot cloning engine reproduces the reference clip's voice, so a mood
        # has almost nothing to change here; it is carried so the *fallback* (the
        # system engine) can still read the mood when this engine cannot.
        self.prosody = prosody
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
        # The resident server keeps the weights loaded across a whole read, which is
        # the real fix for a slow read (a per-paragraph process reloads 1.35 GB each
        # time). It is used when the build ships the server binary and the caller
        # asked for it; otherwise the single-process engine is the honest fallback.
        server = find_server(root) if use_server else None
        if server is not None:
            self._server_runner = AudioCppServerRunner(
                server, model, backend=backend, threads=threads, perf_mode=perf_mode
            )
            self._runner = self._server_runner
            return
        self._runner = AudioCppCliRunner(
            cli, model, backend=backend, threads=threads, perf_mode=perf_mode
        )

    def stop(self) -> None:
        """Stop a resident server this synthesizer started, releasing the model.

        A no-op when the single-process engine is in use or the server was already
        running for someone else, so it is always safe to call when a read ends.
        """
        if self._server_runner is not None:
            self._server_runner.stop()

    def _reference(self) -> tuple[Path | None, str]:
        voice = self.voice
        if voice is None or voice.reference is None:
            return None, ""
        return voice.reference, voice.reference_text

    def synthesize(self, text: str) -> AudioClip:
        if not text.strip():
            raise ValueError("Cannot synthesize empty text.")
        return self.synthesize_many([text])[0]

    def synthesize_many(self, texts: list[str]) -> list[AudioClip]:
        """Render several texts, loading the model **once** for the whole batch.

        A document is read paragraph by paragraph, and each separate engine process
        reloads the whole GGUF before it can speak -- which is why a long document
        looked frozen. Sending the whole batch as one ``--request-sequence`` keeps a
        single model load. Each text is prepared for speech on its own (so Arabic
        numbers and units are spelled per chunk), and the clips come back in order.
        """
        bodies = [text for text in texts if text.strip()]
        if not bodies:
            raise ValueError("Cannot synthesize empty text.")
        prepared = [prepare_for_speech(text, language=detect_language(text)) for text in bodies]
        reference, reference_text = self._reference()
        try:
            if reference is not None and not reference_text.strip():
                # A clone request with a clip but no transcript is refused by the
                # engine ("requires reference_text when reference audio is provided").
                # Raised inside the try so a supplied fallback still reads the text.
                raise AudioCppUnavailable(
                    f"The voice {getattr(self.voice, 'name', '')!r} has a reference clip but "
                    "no transcript, which a cloning voice needs. Re-download the built-in "
                    "voices with: python -m cli audio download --speakers --approve"
                )
            paths = self._render_paths(prepared, reference, reference_text)
        except AudioCppUnavailable:
            if self._fallback is not None:
                return [self._fallback.synthesize(text) for text in bodies]
            raise
        clips: list[AudioClip] = []
        for path in paths:
            clip = AudioClip.read_wav(path)
            try:
                Path(path).unlink()
            except OSError:
                pass
            clips.append(clip)
        if any(not clip.samples for clip in clips):
            if self._fallback is not None:
                return [self._fallback.synthesize(text) for text in bodies]
            raise ValueError("audio.cpp produced no audio.")
        return clips

    def _render_paths(
        self, prepared: list[str], reference: Path | None, reference_text: str
    ) -> list[Path]:
        """Drive the runner for a batch, batching when it supports it.

        The default runner renders a whole batch in one engine session
        (:meth:`AudioCppCliRunner.synthesize_many`); a custom runner that only
        implements the single-text seam is called once per text, so the injection
        point stays small.
        """
        batch = getattr(self._runner, "synthesize_many", None)
        if callable(batch):
            return batch(
                prepared,
                language=self.language,
                reference=reference,
                reference_text=reference_text,
                steps=self.steps,
            )
        return [
            self._runner.synthesize(
                text,
                language=self.language,
                reference=reference,
                reference_text=reference_text,
                steps=self.steps,
            )
            for text in prepared
        ]


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
    "DEFAULT_MAX_THREADS",
    "DEFAULT_SERVER_PORT",
    "GGUF_MAGIC",
    "IMPORTED_VOICES_FILE",
    "IMPORT_PICKER_SUFFIXES",
    "VOICETUT_MODELS",
    "VOICETUT_SPEAKERS",
    "VOICE_CLIP_SUFFIXES",
    "AudioCppCliRunner",
    "AudioCppRunner",
    "AudioCppServerRunner",
    "AudioCppStatus",
    "AudioCppSynthesizer",
    "AudioCppUnavailable",
    "ImportedVoice",
    "VoiceTutSpec",
    "audiocpp_status",
    "available_synthesizer",
    "clone_voice",
    "download_model",
    "download_speakers",
    "engine_dir",
    "engine_installed",
    "extra_installed",
    "find_cli",
    "find_model_file",
    "find_server",
    "import_voice",
    "imported_voice_models",
    "imported_voices_path",
    "install_engine",
    "install_hint",
    "is_valid_gguf",
    "load_imported_voices",
    "model_is_cached",
    "remove_imported_voice",
    "resolve_spec",
    "spec_names",
    "spec_summaries",
    "voice_list",
    "voices_dir",
]
