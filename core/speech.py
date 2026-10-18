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

This module is the **one** place every chat interface gets its speech from: the
GUI, Jarvis, ``ai.voice``, ``ai.speak``, and every future mouth call
:func:`load_recognizer` / :func:`load_synthesizer` (or ``'auto'``). The two
**Qwen3** models -- :mod:`core.qwen_speech`'s Qwen3-ASR and Qwen3-TTS -- are wired
in here as named engines (``'qwen3-asr'`` / ``'qwen3-tts'``), so adding them once
makes them available to every interface, present and future, with no per-window
work. ``auto`` prefers the Qwen3 model for ASR (it covers Arabic) and a system
engine for TTS (tiny, installed, and Arabic-capable); choosing ``qwen3-tts``
explicitly still reads Arabic through the system engine rather than failing.
"""

from __future__ import annotations

import platform
import shutil
import signal
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


#: Windows ships its own speech engine (SAPI, reached through ``System.Speech``);
#: this is the name the project gives it, and the shell that drives it.
_WINDOWS_SAPI = "windows-sapi"
_WINDOWS_SHELL = "powershell"


def _ps_quote(value: str) -> str:
    """Escape ``value`` for a single-quoted Windows PowerShell string literal."""
    return value.replace("'", "''")


def _sapi_rate(voice: Voice | None) -> int:
    """Map a catalog voice's speed onto SAPI's ``-10..10`` rate (175 wpm is neutral)."""
    if voice is None:
        return 0
    return max(-10, min(10, round((voice.speed - 175) / 20)))


def _sapi_select_voice(voice: Voice | None) -> str:
    """A PowerShell fragment that picks a SAPI voice by the catalog voice's gender.

    SAPI voices are named and installed by the OS, so a catalog entry cannot choose
    an exact timbre. It maps as far as SAPI allows: the variant's letter picks a
    male or female installed voice, and the trailing number picks among them. When
    no voice of that gender exists the engine keeps its default rather than failing.
    """
    if voice is None:
        return ""
    gender = "Male" if voice.variant[:1].lower() == "m" else "Female"
    digits = "".join(character for character in voice.variant if character.isdigit())
    index = max(0, int(digits) - 1) if digits else 0
    return (
        "$v=@($s.GetInstalledVoices()|?{$_.Enabled});"
        f"$g=@($v|?{{$_.VoiceInfo.Gender -eq '{gender}'}});"
        "if($g.Count -eq 0){$g=$v};"
        f"if($g.Count -gt 0){{$s.SelectVoice($g[{index} % $g.Count].VoiceInfo.Name)}};"
    )


def _windows_sapi_script(text: str, target: Path, voice: Voice | None) -> str:
    """The one-line PowerShell that renders ``text`` to the WAV file ``target``."""
    script = [
        "$ErrorActionPreference='Stop';",
        "Add-Type -AssemblyName System.Speech;",
        "$s=New-Object System.Speech.Synthesis.SpeechSynthesizer;",
        f"$s.Rate={_sapi_rate(voice)};",
    ]
    selection = _sapi_select_voice(voice)
    if selection:
        script.append(selection)
    script.append(f"$s.SetOutputToWaveFile('{_ps_quote(str(target))}');")
    script.append(f"$s.Speak('{_ps_quote(text)}');")
    script.append("$s.Dispose()")
    return " ".join(script)


def windows_speech_available() -> bool:
    """True when Windows' built-in speech engine can be driven here (SAPI/PowerShell)."""
    return platform.system().lower() == "windows" and shutil.which(_WINDOWS_SHELL) is not None


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
    """A package-manager command for this OS, or empty when the app knows none."""
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
    if system == "windows":
        if shutil.which("winget"):
            return ("winget", "install", "-e", "--id", "eSpeak-NG.eSpeak-NG")
        if shutil.which("choco"):
            return ("choco", "install", "-y", SPEECH_SYSTEM_PACKAGES["windows"])
    return ()


def speech_install_plan() -> SpeechInstallPlan:
    """The platform-aware way to install a real speech engine (usually ``espeak-ng``).

    Windows is the special case: it already ships a speech engine (SAPI), so the
    plan reports that there is nothing to install and ``can_install`` stays False,
    with a hint that says so instead of sending the user off to download a build.
    """
    system = platform.system().lower()
    if system == "windows" and windows_speech_available():
        return SpeechInstallPlan(
            system,
            (),
            "Windows already provides a speech engine (SAPI), so read-aloud works "
            "without installing anything.",
        )
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
            "Install espeak-ng with `winget install -e --id eSpeak-NG.eSpeak-NG` (or "
            "`choco install espeak-ng`), then open a new terminal so it is on PATH; "
            "Windows' built-in speech engine is used automatically when PowerShell "
            "is available.",
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
    """Return the best installed TTS engine, or ``'silent'``.

    The system engines (``espeak``, ``say``, Windows SAPI) come first because they
    are tiny, already installed, and speak Arabic. When none exists but the local
    Qwen3-TTS model is fully present, that model is used instead, so a machine
    with the Qwen3 extra still gets a natural voice rather than silence.
    """
    for name, executable, _builder, _voice_options in TTS_ENGINES:
        if shutil.which(executable):
            return name
    if windows_speech_available():
        return _WINDOWS_SAPI
    from core.qwen_speech import available_synthesizer

    if available_synthesizer():
        return "qwen3-tts"
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


class WindowsSapiSynthesizer:
    """A real TTS engine using the speech engine Windows already ships (SAPI).

    Windows bundles SAPI, reachable from ``System.Speech`` through PowerShell, so a
    Windows user does **not** need to install ``espeak-ng`` at all: this is the
    engine the project uses when no command-line engine is on PATH. It renders
    ``text`` to a WAV with ``SetOutputToWaveFile``/``Speak`` and reads it back, so
    the same :class:`~core.reading.TextReader` works unchanged. A chosen catalog
    voice maps as far as SAPI allows (gender variant + rate); Arabic may be voiced
    only when an Arabic SAPI voice is installed, which the engine itself decides.
    """

    name = _WINDOWS_SAPI
    executable = _WINDOWS_SHELL

    def __init__(self, *, voice: Voice | None = None, language: str | None = None) -> None:
        self.voice = voice
        self.language = language

    def synthesize(self, text: str) -> AudioClip:
        if not text.strip():
            raise ValueError("Cannot synthesize empty text.")
        with tempfile.TemporaryDirectory(prefix="fayd-tts-") as folder:
            target = Path(folder) / "speech.wav"
            script = _windows_sapi_script(text, target, self.voice)
            argv = [
                self.executable,
                "-NoProfile",
                "-NonInteractive",
                "-Command",
                script,
            ]
            try:
                completed = subprocess.run(  # noqa: S603 - fixed argv, no shell
                    argv,
                    capture_output=True,
                    timeout=60,
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
    """Return the best installed ASR engine: ``'qwen3-asr'``, ``'whisper'``, else ``'transcript'``.

    The local Qwen3-ASR model is preferred when its extra and weights are present
    (it covers Arabic, where ``whisper`` is English-first), then the ``whisper``
    program, then the transcript stand-in -- so ``auto`` never fails and always
    picks the strongest engine that is actually installed.
    """
    from core.qwen_speech import available_recognizer

    if available_recognizer():
        return "qwen3-asr"
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
    if name in ("qwen3-asr", "qwen"):
        from core.qwen_speech import QwenAsrRecognizer, QwenSpeechUnavailable, available_recognizer

        if not available_recognizer() and not options.get("runner"):
            from core.qwen_speech import qwen_asr_status

            raise ValueError(
                "ASR engine 'qwen3-asr' is not ready. " + qwen_asr_status().hint
            )
        try:
            return QwenAsrRecognizer(
                model_dir=options.get("model_dir"),
                runner=options.get("runner"),
                language=options.get("language"),
                device=options.get("device", "cpu"),
                root=options.get("root"),
            )
        except QwenSpeechUnavailable as error:
            raise ValueError(str(error)) from error
    raise ValueError(
        f"ASR engine {name!r} is not available in this build. "
        "Install a speech backend or use the transcript recognizer."
    )


def _system_synthesizer(
    *, voice: Voice | None = None, language: str | None = None
) -> SpeechSynthesizer | None:
    """The installed system TTS engine, or ``None`` when there is none.

    Used as the Arabic fallback for Qwen3-TTS (which has no Arabic). ``None`` --
    not a silent stand-in -- so the caller can decide, rather than being handed an
    engine that would produce nothing.
    """
    for engine_name, executable, builder, voice_options in TTS_ENGINES:
        if shutil.which(executable):
            return CommandSynthesizer(
                engine_name,
                executable,
                builder,
                voice=voice,
                language=language,
                voice_options=voice_options,
            )
    if windows_speech_available():
        return WindowsSapiSynthesizer(voice=voice, language=language)
    return None


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
    if name in ("qwen3-tts", "qwen"):
        from core.qwen_speech import QwenSpeechUnavailable, QwenTtsSynthesizer, available_synthesizer

        if not available_synthesizer() and not options.get("runner"):
            from core.qwen_speech import qwen_tts_status

            raise ValueError("TTS engine 'qwen3-tts' is not ready. " + qwen_tts_status().hint)
        fallback = options.get("fallback")
        if fallback is None:
            # Qwen3-TTS does not speak Arabic, so a system engine reads it instead
            # of the model silently mispronouncing it. Chosen once here, so every
            # caller (window, CLI, voice loop) gets the same honest behaviour.
            system = _system_synthesizer(voice=voice, language=language)
            if system is not None:
                fallback = system
        try:
            return QwenTtsSynthesizer(
                model_dir=options.get("model_dir"),
                runner=options.get("runner"),
                speaker=options.get("speaker"),
                language=language,
                instruct=options.get("instruct"),
                fallback=fallback,
                device=options.get("device", "cpu"),
                root=options.get("root"),
            )
        except QwenSpeechUnavailable as error:
            raise ValueError(str(error)) from error
    if name == _WINDOWS_SAPI:
        if windows_speech_available():
            return WindowsSapiSynthesizer(voice=voice, language=language)
        raise ValueError(
            "The Windows speech engine (SAPI) is not available here. "
            "Use --tts auto or --tts silent."
        )
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


class MicRecording:
    """A microphone recording in progress, stoppable before the time limit.

    :meth:`MicrophoneRecorder.start` returns one of these instead of blocking for
    a fixed number of seconds: the caller :meth:`stop` s it when the user is done
    talking (a second click on the microphone icon) and :meth:`result` returns the
    clip. ``arecord`` stops cleanly on ``SIGINT`` (it finalises the WAV header);
    ``sox`` is asked for a long capture and terminated the same way. The handle
    also works with an injected process, so the flow is testable with no audio
    hardware.
    """

    def __init__(
        self,
        process: subprocess.Popen[bytes],
        *,
        path: Path,
        recorder: str,
        folder: Path,
    ) -> None:
        self._process = process
        self._path = path
        self._recorder = recorder
        self._folder = folder

    @property
    def recorder(self) -> str:
        return self._recorder

    @property
    def recording(self) -> bool:
        """True while the recorder is still capturing."""
        return self._process.poll() is None

    def stop(self) -> None:
        """Ask the recorder to stop and finalise the WAV (idempotent).

        ``SIGINT`` is used rather than ``terminate()`` because ``arecord`` flushes
        its WAV header on interrupt; a recorder that ignores it is killed after a
        short grace period, so ``stop`` never hangs.
        """
        if self._process.poll() is None:
            try:
                self._process.send_signal(signal.SIGINT)
            except (OSError, ValueError):
                self._process.terminate()
            try:
                self._process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self._process.kill()
                self._process.wait(timeout=5)

    def result(self, *, timeout: float | None = 30.0) -> AudioClip:
        """Stop if needed and return the recorded clip.

        Raises :class:`ValueError` when no audio was captured (an unavailable
        microphone, or a clip too short to hold a word) rather than returning a
        silent clip the recognizer would reject anyway.
        """
        if self._process.poll() is None:
            self.stop()
        try:
            self._process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            self._process.kill()
            self._process.wait(timeout=5)
        try:
            if not self._path.exists() or self._path.stat().st_size <= 44:
                raise ValueError(
                    f"Microphone recorder {self._recorder!r} captured no audio."
                )
            return AudioClip.read_wav(self._path)
        finally:
            shutil.rmtree(self._folder, ignore_errors=True)


class MicrophoneRecorder:
    """Records a clip from the system microphone, when a recorder exists.

    Uses ``arecord`` (Linux/ALSA) or ``sox`` (cross-platform) if installed; when
    neither exists, :meth:`available` is False and the caller should fall back to
    the text loop. No pip package and no network access is involved.
    :meth:`record` blocks for a fixed number of seconds (the CLI's behaviour);
    :meth:`start` returns a :class:`MicRecording` the caller stops when the user
    is done -- which is how the window's microphone icon toggles.
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

    @staticmethod
    def _argv(recorder: str, target: Path, sample_rate: int, seconds: float) -> list[str]:
        """The command that captures ``seconds`` of mono 16-bit audio to ``target``."""
        if recorder == "arecord":
            return [
                "arecord", "-q", "-f", "S16_LE", "-r", str(sample_rate),
                "-d", str(int(seconds)), str(target),
            ]
        return [
            "sox", "-d", "-r", str(sample_rate), "-c", "1", "-b", "16",
            str(target), "trim", "0", str(seconds),
        ]

    def start(self) -> MicRecording:
        """Begin recording in the background; return a stoppable MicRecording.

        ``seconds`` bounds the capture (a safety limit); the caller normally stops
        it sooner. A recorder that cannot start raises, so the caller can report
        it instead of hanging.
        """
        recorder = self.recorder
        if recorder is None:
            raise ValueError("No microphone recorder is available (install arecord or sox).")
        folder = Path(tempfile.mkdtemp(prefix="fayd-mic-"))
        target = folder / "input.wav"
        argv = self._argv(recorder, target, self.sample_rate, self.seconds)
        try:
            process = subprocess.Popen(  # noqa: S603 - fixed argv, no shell
                argv, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
            )
        except OSError as error:
            shutil.rmtree(folder, ignore_errors=True)
            raise ValueError(f"Microphone recorder {recorder!r} failed: {error}") from error
        return MicRecording(
            process, path=target, recorder=recorder, folder=folder
        )

    def record(self) -> AudioClip:
        """Record for a fixed ``seconds`` and return the clip (blocking)."""
        return self.start().result()


class Playback:
    """A clip playing in the background, stoppable before it ends.

    :meth:`Speaker.start` returns one of these instead of blocking: the audio runs
    in a child process the caller can :meth:`stop` (so a second click on "Read
    aloud" stops the reading) while :meth:`wait` blocks until it finishes. It is
    honest about what happened -- :attr:`stopped` says whether it was cut short,
    and a real player failure is raised from :meth:`wait` rather than swallowed.
    """

    def __init__(self, process: subprocess.Popen[bytes], *, player: str, folder: Path) -> None:
        self._process = process
        self._player = player
        self._folder: Path | None = folder
        self.stopped = False

    @property
    def player(self) -> str:
        return self._player

    @property
    def playing(self) -> bool:
        """True while the audio is still being played."""
        return self._process.poll() is None

    def stop(self) -> None:
        """Stop the audio now, if it is still playing (idempotent).

        Also removes the temporary WAV, so a stopped reading leaves nothing
        behind -- the caller does not have to remember to clean up.
        """
        if self._process.poll() is None:
            self.stopped = True
            self._process.terminate()
            try:
                self._process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self._process.kill()
                self._process.wait(timeout=5)
        self._cleanup()

    def wait(self, *, timeout: float | None = 120.0) -> None:
        """Block until the audio ends; raise if the player failed.

        A clip the caller stopped is *not* a failure: it returns quietly, so the
        GUI's ``stop()`` then ``wait()`` flow never reports a spurious error.
        """
        try:
            self._process.wait(timeout=timeout)
        except subprocess.TimeoutExpired as error:
            self.stop()
            raise ValueError(f"Audio player {self._player!r} timed out.") from error
        finally:
            self._cleanup()
        if self.stopped:
            return
        if self._process.returncode != 0:
            raise ValueError(f"Audio player {self._player!r} failed to play the clip.")

    def _cleanup(self) -> None:
        if self._folder is not None:
            shutil.rmtree(self._folder, ignore_errors=True)
            self._folder = None


class Speaker:
    """Plays a clip through the system speakers, when a player exists.

    Uses ``aplay`` (Linux/ALSA), ``afplay`` (macOS), or ``sox``/``play`` if
    installed. On Windows, where none of those exist, it falls back to the
    built-in ``SoundPlayer`` through PowerShell so read-aloud is actually audible.
    :meth:`available` reports honestly, and :meth:`play` raises when no player
    exists so the caller can fall back to text. No pip package needed.

    :meth:`play` blocks until the clip ends (the CLI's behaviour); :meth:`start`
    returns a :class:`Playback` the caller can stop -- which is how a chat window
    lets a second click end the reading.
    """

    def __init__(self, *, sample_rate: int = SAMPLE_RATE) -> None:
        self.sample_rate = sample_rate

    @property
    def player(self) -> str | None:
        for executable in ("aplay", "afplay", "play", "sox"):
            if shutil.which(executable):
                return executable
        if windows_speech_available():
            return _WINDOWS_SHELL
        return None

    def available(self) -> bool:
        return self.player is not None

    @staticmethod
    def _argv(player: str, target: Path) -> list[str]:
        """The command that plays the WAV ``target`` through ``player``."""
        if player == "aplay":
            return ["aplay", "-q", str(target)]
        if player == "afplay":
            return ["afplay", str(target)]
        if player == "play":
            return ["play", "-q", str(target)]
        if player == _WINDOWS_SHELL:
            # Windows: play the WAV through the built-in SoundPlayer.
            script = (
                f"(New-Object System.Media.SoundPlayer '{_ps_quote(str(target))}').PlaySync()"
            )
            return [player, "-NoProfile", "-NonInteractive", "-Command", script]
        return ["sox", str(target), "-d"]  # sox

    def start(self, clip: AudioClip) -> Playback:
        """Begin playing ``clip`` in the background; return a stoppable Playback.

        The WAV is written under a temporary folder that outlives this call -- a
        background player reads the file after we return -- and :class:`Playback`
        removes it when the clip finishes or is stopped.
        """
        player = self.player
        if player is None:
            raise ValueError("No audio player is available (install aplay, afplay, or sox).")
        folder = Path(tempfile.mkdtemp(prefix="fayd-play-"))
        target = folder / "speech.wav"
        clip.write_wav(target)
        argv = self._argv(player, target)
        try:
            process = subprocess.Popen(  # noqa: S603 - fixed argv, no shell
                argv, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
            )
        except OSError as error:
            shutil.rmtree(folder, ignore_errors=True)
            raise ValueError(f"Audio player {player!r} failed: {error}") from error
        return Playback(process, player=player, folder=folder)

    def play(self, clip: AudioClip) -> None:
        """Play ``clip`` and block until it ends; raise if no player or it fails."""
        self.start(clip).wait()


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
    "MicRecording",
    "Playback",
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
    "WindowsSapiSynthesizer",
    "discover_recognizer_name",
    "discover_synthesizer_name",
    "install_speech_engine",
    "load_recognizer",
    "load_synthesizer",
    "speech_available",
    "speech_install_plan",
    "windows_speech_available",
]
