"""Speech I/O for the assistant: Arabic ASR and TTS behind a small interface.

Voice is a *front end* to the grounded assistant, not a new reasoning layer. This
module defines the contracts and the plumbing so a real speech engine can be
dropped in without touching the cognitive core:

* :class:`SpeechRecognizer` (ASR) turns audio into text;
* :class:`SpeechSynthesizer` (TTS) turns text into audio;
* :class:`AudioClip` is a tiny, dependency-free PCM buffer, so the whole pipeline
  can be tested and demoed without any audio hardware or third-party package.

Engines are discovered lazily by name (``piper``, ``whisper``, ...). When none is
installed the project still exposes a working *text* voice loop -- the recognizer
returns a supplied transcript and the synthesizer produces a silent clip -- so the
conversation flow is exercised end to end on a fresh clone.
"""

from __future__ import annotations

import platform
import shutil
import subprocess
import tempfile
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol, runtime_checkable

from core.model_interface import CognitiveModel, ModelRequest
from core.voices import Voice, default_voice, detect_language, find_voice

SAMPLE_RATE = 16_000


@dataclass(frozen=True)
class AudioClip:
    """A mono PCM buffer, 16-bit samples at :data:`SAMPLE_RATE`."""

    samples: tuple[int, ...]
    sample_rate: int = SAMPLE_RATE

    def __post_init__(self) -> None:
        if self.sample_rate < 1:
            raise ValueError("Sample rate must be positive.")
        if any(not -32768 <= sample <= 32767 for sample in self.samples):
            raise ValueError("PCM samples must be signed 16-bit values.")

    @property
    def duration_seconds(self) -> float:
        return len(self.samples) / self.sample_rate

    @property
    def is_silent(self) -> bool:
        return not any(self.samples)

    def to_bytes(self) -> bytes:
        import array

        buffer = array.array("h", self.samples)
        return buffer.tobytes()

    @classmethod
    def silence(cls, duration_seconds: float, *, sample_rate: int = SAMPLE_RATE) -> "AudioClip":
        if duration_seconds < 0:
            raise ValueError("Silence duration cannot be negative.")
        return cls(tuple([0] * int(duration_seconds * sample_rate)), sample_rate)

    @classmethod
    def from_bytes(cls, payload: bytes, *, sample_rate: int = SAMPLE_RATE) -> "AudioClip":
        import array

        samples = array.array("h")
        samples.frombytes(payload[: len(payload) - (len(payload) % 2)])
        return cls(tuple(samples), sample_rate)

    def write_wav(self, path) -> None:
        """Write a 16-bit mono WAV file (no third-party dependency)."""
        import wave

        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        with wave.open(str(target), "wb") as handle:
            handle.setnchannels(1)
            handle.setsampwidth(2)
            handle.setframerate(self.sample_rate)
            handle.writeframes(self.to_bytes())

    @classmethod
    def read_wav(cls, path) -> "AudioClip":
        import wave

        with wave.open(str(Path(path)), "rb") as handle:
            if handle.getnchannels() != 1 or handle.getsampwidth() != 2:
                raise ValueError("Only 16-bit mono WAV files are supported.")
            return cls.from_bytes(handle.readframes(handle.getnframes()), sample_rate=handle.getframerate())


@runtime_checkable
class SpeechRecognizer(Protocol):
    """Turns an audio clip into text (Arabic ASR)."""

    name: str

    def transcribe(self, clip: AudioClip) -> str: ...


@runtime_checkable
class SpeechSynthesizer(Protocol):
    """Turns text into audio (Arabic TTS)."""

    name: str

    def synthesize(self, text: str) -> AudioClip: ...


class SilentSynthesizer:
    """A TTS stand-in that produces a silent clip sized to the text length.

    Keeps the voice loop runnable and testable when no engine is installed; the
    duration estimate is deliberately crude (about 12 characters per second).
    """

    name = "silent"

    def synthesize(self, text: str) -> AudioClip:
        if not text.strip():
            raise ValueError("Cannot synthesize empty text.")
        seconds = max(0.2, len(text) / 12)
        return AudioClip.silence(seconds)


# System TTS engines tried in order: (name, executable, argv builder, voice-option
# builder). Each argv builder returns the argv that renders ``text`` to the WAV
# file ``path``; the voice-option builder returns the extra engine flags for a
# chosen :class:`~core.voices.Voice` (empty when the engine has no such control).
# These are ordinary programs already present on many desktops, so no pip package
# and no network access is needed; when none exists the silent synthesizer is used.
def _espeak_argv(executable: str, text: str, path: Path) -> list[str]:
    return [executable, "-w", str(path), text]


def _flite_argv(executable: str, text: str, path: Path) -> list[str]:
    return [executable, "-t", text, "-o", str(path)]


def _say_argv(executable: str, text: str, path: Path) -> list[str]:
    # macOS: ``say -o file.wav`` renders a WAVE file.
    return [executable, "-o", str(path), text]


def _espeak_voice_options(voice: Voice | None, language: str) -> tuple[str, ...]:
    """The ``espeak-ng`` flags that select a catalog voice (gender variant, pitch, speed)."""
    if voice is None:
        return ()
    return voice.espeak_options(language)


def _say_voice_options(voice: Voice | None, language: str) -> tuple[str, ...]:
    """``say`` has no gender variant, so only the speaking rate carries over."""
    if voice is None:
        return ()
    return ("-r", str(voice.speed))


def _flite_voice_options(_voice: Voice | None, _language: str) -> tuple[str, ...]:
    """``flite`` exposes no voice controls here, so a chosen voice is engine-default."""
    return ()


TTS_ENGINES: tuple[tuple[str, str, "object", "object"], ...] = (
    ("espeak-ng", "espeak-ng", _espeak_argv, _espeak_voice_options),
    ("espeak", "espeak", _espeak_argv, _espeak_voice_options),
    ("flite", "flite", _flite_argv, _flite_voice_options),
    ("say", "say", _say_argv, _say_voice_options),
)


#: The system package a Debian/Ubuntu machine needs for a real voice engine.
SPEECH_SYSTEM_PACKAGES: dict[str, str] = {
    "debian": "espeak-ng",
    "fedora": "espeak-ng",
    "arch": "espeak-ng",
    "alpine": "espeak-ng",
    "macos": "espeak-ng",  # `brew install espeak-ng`; macOS also has its own `say`
    "windows": "espeak-ng",  # download the Windows build, or use `pyttsx3`
}


@dataclass(frozen=True)
class SpeechInstallPlan:
    """How to get a real speech engine, honest about what pip cannot do.

    A speech engine (``espeak-ng`` and friends) is an ordinary *system* program,
    not a Python package, so ``pip`` cannot install it -- the same distinction
    the OCR path makes for Tesseract. ``command`` is a ready-to-run package-manager
    command on this platform, or empty when the app does not know one (so the
    caller shows ``hint`` instead of pretending it can install).
    """

    platform: str
    command: tuple[str, ...]
    hint: str

    @property
    def can_install(self) -> bool:
        return bool(self.command)


def _package_manager_command() -> tuple[str, ...]:
    """A ``sudo`` package-manager command for this OS, or empty when unknown."""
    system = platform.system().lower()
    if system == "linux":
        if shutil.which("apt-get"):
            return ("sudo", "apt-get", "install", "-y", SPEECH_SYSTEM_PACKAGES["debian"])
        if shutil.which("dnf"):
            return ("sudo", "dnf", "install", "-y", SPEECH_SYSTEM_PACKAGES["fedora"])
        if shutil.which("pacman"):
            return ("sudo", "pacman", "-S", "--noconfirm", SPEECH_SYSTEM_PACKAGES["arch"])
        if shutil.which("apk"):
            return ("sudo", "apk", "add", SPEECH_SYSTEM_PACKAGES["alpine"])
        return ()
    if system == "darwin" and shutil.which("brew"):
        return ("brew", "install", SPEECH_SYSTEM_PACKAGES["macos"])
    return ()


def speech_install_plan() -> SpeechInstallPlan:
    """The platform-aware way to install a real speech engine (usually ``espeak-ng``)."""
    system = platform.system().lower()
    command = _package_manager_command()
    if command:
        return SpeechInstallPlan(system, command, " ".join(command))
    if system == "darwin":
        return SpeechInstallPlan(
            system,
            (),
            "Install espeak-ng with Homebrew (`brew install espeak-ng`); macOS `say` "
            "already provides a voice.",
        )
    if system == "windows":
        return SpeechInstallPlan(
            system,
            (),
            "Download the espeak-ng Windows build and put `espeak-ng.exe` on PATH; "
            "the project also speaks through macOS `say` and Linux espeak-ng/espeak/flite.",
        )
    return SpeechInstallPlan(
        system,
        (),
        "Install espeak-ng with your system package manager "
        "(e.g. `sudo apt install espeak-ng`).",
    )


@dataclass(frozen=True)
class SpeechInstallResult:
    """The outcome of an attempted speech-engine install, ready to show the user."""

    ok: bool
    command: tuple[str, ...] = ()
    returncode: int | None = None
    output: str = ""
    message: str = ""
    ran: bool = True


def speech_available() -> bool:
    """Whether any real TTS engine is installed (the same check the reader uses)."""
    return discover_synthesizer_name() != "silent"


def install_speech_engine(
    *,
    runner: "SpeechCommandRunner | None" = None,
) -> SpeechInstallResult:
    """Try to install a real speech engine as a system package. Never raises.

    ``pip`` cannot install ``espeak-ng`` (it is an OS package, like Tesseract), so
    this runs the platform's package manager through ``sudo`` when the app knows
    one. If it succeeds and the engine is now on PATH, ``ok`` is True. When the app
    does not know an install command (or the runner fails), ``ran``/``ok`` are
    False and ``message`` carries the manual hint instead of a false promise.
    """
    if speech_available():
        return SpeechInstallResult(
            ok=True, ran=False, message="A speech engine is already installed."
        )
    plan = speech_install_plan()
    if not plan.can_install:
        return SpeechInstallResult(ok=False, ran=False, message=plan.hint)
    run = runner or _default_speech_runner
    returncode, output = run(plan.command)
    ok = returncode == 0 and speech_available()
    if ok:
        message = f"Installed a speech engine ({SPEECH_SYSTEM_PACKAGES['debian']})."
    elif returncode == 0:
        message = (
            "The package manager reported success, but no speech engine is on PATH yet. "
            + plan.hint
        )
    else:
        message = f"The install command failed.\n{plan.hint}"
    return SpeechInstallResult(
        ok=ok,
        command=plan.command,
        returncode=returncode,
        output=output,
        message=message,
    )


def _default_speech_runner(argv: Sequence[str]) -> tuple[int, str]:
    completed = subprocess.run(  # noqa: S603 - fixed argv, no shell
        list(argv),
        capture_output=True,
        text=True,
        check=False,
    )
    combined = (completed.stdout or "") + (completed.stderr or "")
    return completed.returncode, combined


#: Runs one install command and returns ``(returncode, combined_output)``.
SpeechCommandRunner = Callable[[Sequence[str]], tuple[int, str]]


def discover_synthesizer_name() -> str:
    """Return the first system TTS engine available, or ``'silent'``."""
    for name, executable, _builder, _voice_options in TTS_ENGINES:
        if shutil.which(executable):
            return name
    return "silent"


class CommandSynthesizer:
    """A real TTS engine driven through a system command.

    The engine is a normal program (``espeak``, ``say``, ...). We render to a
    temporary WAV, read it back as an :class:`AudioClip`, and never require a pip
    dependency. If the engine fails for any reason, synthesis falls back to a
    silent clip so a voice turn can still complete.

    ``voice`` is an optional :class:`~core.voices.Voice` from the catalog; its
    engine flags are inserted after the executable. ``language`` pins the spoken
    language, and when it is ``None`` the language is taken from the text (Arabic
    script -> ``ar``, otherwise ``en``), so one voice reads both.
    """

    def __init__(
        self,
        name: str,
        executable: str,
        builder,
        *,
        voice: Voice | None = None,
        language: str | None = None,
        voice_options=None,
    ) -> None:
        self.name = name
        self.executable = executable
        self._builder = builder
        self.voice = voice
        self.language = language
        self._voice_options = voice_options

    def _options_for(self, text: str) -> tuple[str, ...]:
        if self.voice is None or self._voice_options is None:
            return ()
        language = self.language or detect_language(text)
        return tuple(self._voice_options(self.voice, language))

    def synthesize(self, text: str) -> AudioClip:
        if not text.strip():
            raise ValueError("Cannot synthesize empty text.")
        with tempfile.TemporaryDirectory(prefix="fayd-tts-") as folder:
            target = Path(folder) / "speech.wav"
            argv = self._builder(self.executable, text, target)
            options = self._options_for(text)
            if options:
                # Options sit between the executable and its own arguments, which
                # every engine here accepts.
                argv = [argv[0], *options, *argv[1:]]
            try:
                completed = subprocess.run(  # noqa: S603 - fixed argv, no shell
                    argv,
                    capture_output=True,
                    timeout=30,
                    check=False,
                )
            except (OSError, subprocess.SubprocessError):
                return SilentSynthesizer().synthesize(text)
            if completed.returncode != 0 or not target.exists():
                return SilentSynthesizer().synthesize(text)
            try:
                return AudioClip.read_wav(target)
            except (OSError, ValueError):
                return SilentSynthesizer().synthesize(text)


class WhisperRecognizer:
    """A real ASR engine driven through the ``whisper`` command-line program.

    ``whisper`` writes its transcript next to the audio file, so we render the
    clip to a temporary WAV, run the command, and read the ``.txt`` it produces.
    When the program is missing or fails, transcription raises ``ValueError`` so
    the caller can fall back rather than inventing speech.
    """

    name = "whisper"

    def __init__(self, executable: str = "whisper", *, model: str = "tiny") -> None:
        self.executable = executable
        self.model = model

    def transcribe(self, clip: AudioClip) -> str:
        with tempfile.TemporaryDirectory(prefix="fayd-asr-") as folder:
            audio = Path(folder) / "speech.wav"
            clip.write_wav(audio)
            argv = [
                self.executable,
                str(audio),
                "--model",
                self.model,
                "--output_format",
                "txt",
                "--output_dir",
                folder,
                "--fp16",
                "False",
            ]
            try:
                completed = subprocess.run(  # noqa: S603 - fixed argv, no shell
                    argv,
                    capture_output=True,
                    timeout=120,
                    check=False,
                )
            except (OSError, subprocess.SubprocessError) as error:
                raise ValueError(f"ASR engine {self.name!r} failed: {error}") from error
            transcript = audio.with_suffix(".txt")
            if completed.returncode != 0 or not transcript.exists():
                raise ValueError(f"ASR engine {self.name!r} produced no transcript.")
            text = transcript.read_text(encoding="utf-8", errors="replace").strip()
            if not text:
                raise ValueError(f"ASR engine {self.name!r} produced an empty transcript.")
            return text


def discover_recognizer_name() -> str:
    """Return ``'whisper'`` when the program is installed, else ``'transcript'``."""
    return "whisper" if shutil.which("whisper") else "transcript"


class TranscriptRecognizer:
    """An ASR stand-in that echoes a supplied transcript.

    Used by the text voice loop and by tests, so the pipeline can run without
    audio hardware. It deliberately refuses to invent speech from silence.
    """

    name = "transcript"

    def __init__(self, transcript: str = "") -> None:
        self.transcript = transcript

    def transcribe(self, clip: AudioClip) -> str:
        if not self.transcript.strip():
            raise ValueError("No transcript was supplied to the recognizer.")
        return self.transcript


def load_recognizer(name: str = "transcript", **options) -> SpeechRecognizer:
    """Load a named ASR engine.

    ``name='auto'`` picks ``whisper`` when it is installed, else the transcript
    recognizer. An explicitly named engine that is unavailable raises, so a user
    who asks for it gets a clear error instead of a silent substitute.
    """
    if name == "auto":
        name = discover_recognizer_name()
    if name == "transcript":
        return TranscriptRecognizer(options.get("transcript", ""))
    if name == "whisper":
        executable = options.get("executable", "whisper")
        if shutil.which(executable):
            return WhisperRecognizer(executable, model=options.get("model", "tiny"))
        raise ValueError(
            "ASR engine 'whisper' is not installed. Use --asr auto or --asr transcript."
        )
    raise ValueError(
        f"ASR engine {name!r} is not available in this build. "
        "Install a speech backend or use the transcript recognizer."
    )


def load_synthesizer(name: str = "silent", **options) -> SpeechSynthesizer:
    """Load a named TTS engine, optionally with a chosen catalog voice.

    ``name='auto'`` picks the first installed system engine (``espeak``,
    ``say``, ...), else the silent synthesizer. An explicitly named engine that
    is unavailable raises, so the caller never mistakes silence for speech.

    ``voice`` selects a :class:`~core.voices.Voice` from the catalog, by key or
    object; ``language`` pins the spoken language (``None`` lets the engine's text
    decide, so Arabic is read as Arabic). A voice can only be *applied* by an
    engine that supports one -- the silent synthesizer ignores it, and its
    ``name`` stays ``'silent'`` so a caller can tell it apart from real speech.
    """
    if name == "auto":
        name = discover_synthesizer_name()
    if name == "silent":
        return SilentSynthesizer()
    voice = _resolve_voice(options.get("voice"))
    language = options.get("language")
    for engine_name, executable, builder, voice_options in TTS_ENGINES:
        if name == engine_name:
            if shutil.which(executable):
                return CommandSynthesizer(
                    engine_name,
                    executable,
                    builder,
                    voice=voice,
                    language=language,
                    voice_options=voice_options,
                )
            raise ValueError(
                f"TTS engine {name!r} is not installed. Use --tts auto or --tts silent."
            )
    raise ValueError(
        f"TTS engine {name!r} is not available in this build. "
        "Install a speech backend or use the silent synthesizer."
    )


def _resolve_voice(value) -> Voice | None:
    """Coerce a voice key or object into a :class:`~core.voices.Voice`.

    ``None`` returns the default catalog voice so a plain ``--tts espeak-ng``
    still speaks with a real, named timbre; an unknown key raises, so a typo is
    reported rather than silently replaced.
    """
    if value is None:
        return default_voice()
    if isinstance(value, Voice):
        return value
    return find_voice(str(value))


class MicrophoneRecorder:
    """Records a short clip from the system microphone, when a recorder exists.

    Uses ``arecord`` (Linux/ALSA) or ``sox`` (cross-platform) if installed; when
    neither exists, :meth:`available` is False and the caller should fall back to
    the text loop. No pip package and no network access is involved.
    """

    def __init__(self, *, seconds: float = 4.0, sample_rate: int = SAMPLE_RATE) -> None:
        self.seconds = seconds
        self.sample_rate = sample_rate

    @property
    def recorder(self) -> str | None:
        if shutil.which("arecord"):
            return "arecord"
        if shutil.which("sox"):
            return "sox"
        return None

    def available(self) -> bool:
        return self.recorder is not None

    def record(self) -> AudioClip:
        recorder = self.recorder
        if recorder is None:
            raise ValueError("No microphone recorder is available (install arecord or sox).")
        with tempfile.TemporaryDirectory(prefix="fayd-mic-") as folder:
            target = Path(folder) / "input.wav"
            if recorder == "arecord":
                argv = ["arecord", "-q", "-f", "S16_LE", "-r", str(self.sample_rate), "-d", str(int(self.seconds)), str(target)]
            else:  # sox
                argv = ["sox", "-d", "-r", str(self.sample_rate), "-c", "1", "-b", "16", str(target), "trim", "0", str(self.seconds)]
            try:
                completed = subprocess.run(  # noqa: S603 - fixed argv, no shell
                    argv, capture_output=True, timeout=int(self.seconds) + 10, check=False
                )
            except (OSError, subprocess.SubprocessError) as error:
                raise ValueError(f"Microphone recorder {recorder!r} failed: {error}") from error
            if completed.returncode != 0 or not target.exists():
                raise ValueError(f"Microphone recorder {recorder!r} produced no audio.")
            return AudioClip.read_wav(target)


class Speaker:
    """Plays a clip through the system speakers, when a player exists.

    Uses ``aplay`` (Linux/ALSA), ``afplay`` (macOS), or ``sox``/``play`` if
    installed. :meth:`available` reports honestly, and :meth:`play` raises when
    no player exists so the caller can fall back to text. No pip package needed.
    """

    def __init__(self, *, sample_rate: int = SAMPLE_RATE) -> None:
        self.sample_rate = sample_rate

    @property
    def player(self) -> str | None:
        for executable in ("aplay", "afplay", "play", "sox"):
            if shutil.which(executable):
                return executable
        return None

    def available(self) -> bool:
        return self.player is not None

    def play(self, clip: AudioClip) -> None:
        player = self.player
        if player is None:
            raise ValueError("No audio player is available (install aplay, afplay, or sox).")
        with tempfile.TemporaryDirectory(prefix="fayd-play-") as folder:
            target = Path(folder) / "speech.wav"
            clip.write_wav(target)
            if player == "aplay":
                argv = ["aplay", "-q", str(target)]
            elif player == "afplay":
                argv = ["afplay", str(target)]
            elif player == "play":
                argv = ["play", "-q", str(target)]
            else:  # sox
                argv = ["sox", str(target), "-d"]
            try:
                completed = subprocess.run(  # noqa: S603 - fixed argv, no shell
                    argv, capture_output=True, timeout=120, check=False
                )
            except (OSError, subprocess.SubprocessError) as error:
                raise ValueError(f"Audio player {player!r} failed: {error}") from error
            if completed.returncode != 0:
                raise ValueError(f"Audio player {player!r} failed to play the clip.")


@dataclass
class VoiceTurn:
    """One voice exchange: what was heard, what was answered, what was spoken."""

    heard: str
    answer_text: str
    spoken: AudioClip
    used_model: bool = False
    citations: tuple[str, ...] = field(default_factory=tuple)


class VoiceAssistant:
    """Wraps a grounded assistant with a speech recognizer and synthesizer."""

    def __init__(
        self,
        assistant,
        *,
        recognizer: SpeechRecognizer | None = None,
        synthesizer: SpeechSynthesizer | None = None,
    ) -> None:
        self.assistant = assistant
        self.recognizer = recognizer or TranscriptRecognizer()
        self.synthesizer = synthesizer or SilentSynthesizer()

    def respond_to_speech(self, clip: AudioClip, *, conversation=None, limit: int = 4) -> VoiceTurn:
        heard = self.recognizer.transcribe(clip).strip()
        if not heard:
            raise ValueError("Speech recognizer returned no text.")
        answer = self.assistant.ask(heard, conversation=conversation, limit=limit)
        spoken = self.synthesizer.synthesize(answer.text)
        return VoiceTurn(
            heard=heard,
            answer_text=answer.text,
            spoken=spoken,
            used_model=answer.used_model,
            citations=answer.citations,
        )


class AudioModel:
    """Adapter that exposes an audio clip to a :class:`CognitiveModel`.

    The clip is described as text metadata (duration, sample rate, silence) so a
    model that cannot process raw audio still receives an honest description
    rather than fabricated content.
    """

    def __init__(self, model: CognitiveModel) -> None:
        self.model = model

    def describe(self, clip: AudioClip, *, instruction: str = "Describe the audio signal.") -> str:
        request = ModelRequest(
            inputs=(),
            system_instruction=instruction,
            context=(
                f"Audio clip: duration {clip.duration_seconds:.2f} s, "
                f"sample rate {clip.sample_rate} Hz, "
                f"{'silent' if clip.is_silent else 'contains a signal'}.",
            ),
        )
        return self.model.generate(request).text


__all__ = [
    "SAMPLE_RATE",
    "SPEECH_SYSTEM_PACKAGES",
    "AudioClip",
    "AudioModel",
    "CommandSynthesizer",
    "MicrophoneRecorder",
    "SilentSynthesizer",
    "Speaker",
    "SpeechCommandRunner",
    "SpeechInstallPlan",
    "SpeechInstallResult",
    "SpeechRecognizer",
    "SpeechSynthesizer",
    "TTS_ENGINES",
    "TranscriptRecognizer",
    "VoiceAssistant",
    "VoiceTurn",
    "WhisperRecognizer",
    "discover_recognizer_name",
    "discover_synthesizer_name",
    "install_speech_engine",
    "load_recognizer",
    "load_synthesizer",
    "speech_available",
    "speech_install_plan",
]
