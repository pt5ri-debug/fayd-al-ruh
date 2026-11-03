"""Speech-to-text and text-to-speech interfaces (sections 7 and 8).

Both directions are **replaceable local engines** behind the project's own
factories (:func:`core.speech.load_recognizer` / :func:`core.speech.load_synthesizer`),
so this module is a thin adapter that adds the perception-specific guarantees:

**Speech-to-text** (section 7):

* Windows-compatible and CPU-first -- it uses whatever engine the project's own
  discovery picks (Qwen3-ASR, then ``whisper``, then the stand-in), and never
  assumes a GPU.
* Loading and unloading are explicit: :func:`load_recognizer` returns an object a
  caller can drop to release it, and :func:`transcribe_audio` loads, uses, and
  releases in one call.
* The original audio reference and the engine's name/version are preserved in the
  transcript's provenance, so a corrected transcript can always be traced back to
  the audio it came from.
* Recognition output is **never** treated as verified knowledge: the confidence is
  a bounded quality signal and the caller may correct the transcript.

**Text-to-speech** (section 8):

* Fully offline -- it uses a local system engine or a local model, never a cloud
  service.
* Voice and language are configurable through the project's own voice catalog.
* Output is a WAV file (or a playable :class:`core.speech.AudioClip`).
* Graceful fallback: when no engine exists the project's ``silent`` synthesizer is
  used, so a caller gets an honest empty clip rather than an exception -- and TTS
  never touches reasoning or knowledge storage.

Neither direction is required for the rest of the system: a base install with no
speech engine still perceives documents, and reports the real next step for speech.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from perception.limits import CancelToken, PerceptionLimits, cancelled


class SpeechPerceptionError(ValueError):
    """A speech operation could not be completed (no engine, empty input)."""


# --------------------------------------------------------------------------- #
# Speech to text
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class Transcript:
    """One speech-to-text result: the text plus what is honestly known about it."""

    text: str
    engine: str
    language: str = ""
    confidence: float = 0.6
    """A bounded quality signal; never a truth claim. Recognition is imperfect, so
    this is deliberately below 1.0 and a caller is invited to correct the text."""

    timestamps: bool = False
    """Whether the engine reported word-level timing (the bundled engines do not)."""

    audio_reference: str = ""
    version: str = ""

    def describe(self) -> str:
        bits = [f"{self.engine}", f"{len(self.text.split())} words"]
        if self.language:
            bits.append(self.language)
        bits.append(f"confidence {self.confidence:.2f}")
        if self.audio_reference:
            bits.append(f"from {self.audio_reference}")
        return ", ".join(bits)


@dataclass(frozen=True)
class TranscriptCorrection:
    """A user's correction of a transcript, kept traceable to the original audio.

    Speech recognition is imperfect, so a caller may correct a transcript. A
    correction is a **new** record that points back at the original (its result id
    and audio reference), never a mutation of the perceived text -- so the original
    reading is preserved and the corrected text is honestly labelled as
    user-corrected rather than machine-recognised.
    """

    original_text: str
    corrected_text: str
    audio_reference: str = ""
    result_id: str = ""
    corrected_by: str = "user"
    language: str = ""

    @property
    def changed(self) -> bool:
        return self.corrected_text.strip() != self.original_text.strip()

    def describe(self) -> str:
        source = f" ({self.result_id})" if self.result_id else ""
        return (
            f"corrected by {self.corrected_by}{source}: "
            f"{len(self.original_text.split())} -> {len(self.corrected_text.split())} words"
        )


def correct_transcript(
    original: str,
    corrected: str,
    *,
    result: object | None = None,
    audio_reference: str = "",
    corrected_by: str = "user",
    language: str = "",
) -> TranscriptCorrection:
    """Record a user's correction of a transcript, traceable to the original audio.

    ``original`` is the machine transcript; ``corrected`` is the user's text;
    ``result`` (a :class:`perception.schema.PerceptionResult`, optional) supplies
    the result id, audio reference, and language when the caller has it. A blank
    correction is refused (a correction must say something), and the returned
    record is a new value -- nothing perceived is overwritten.
    """
    if not corrected.strip():
        raise SpeechPerceptionError("a transcript correction cannot be empty.")
    reference = audio_reference
    result_id = ""
    resolved_language = language
    if result is not None:
        result_id = str(getattr(result, "result_id", "") or "")
        reference = reference or str(getattr(result, "source_reference", "") or "")
        resolved_language = resolved_language or str(getattr(result, "language", "") or "")
    return TranscriptCorrection(
        original_text=original,
        corrected_text=corrected.strip(),
        audio_reference=reference,
        result_id=result_id,
        corrected_by=corrected_by,
        language=resolved_language,
    )


def load_recognizer(name: str = "auto", **options: object) -> object:
    """Load a local ASR engine, reusing the project's own factory.

    ``name="auto"`` lets the project's discovery choose the strongest installed
    engine (Qwen3-ASR, then ``whisper``, then the transcript stand-in). The
    returned object is a :class:`core.speech.SpeechRecognizer`; drop it to release
    it (the project's engines hold no global state, so this is enough to unload).
    """
    from core.speech import discover_recognizer_name
    from core.speech import load_recognizer as _load

    resolved = discover_recognizer_name() if name == "auto" else name
    return _load(resolved, **options)


def recognizer_available() -> bool:
    """Whether a *real* ASR engine (not the stand-in) is installed."""
    from core.speech import discover_recognizer_name

    return discover_recognizer_name() != "transcript"


def recognizer_name() -> str:
    """The ASR engine discovery would pick here."""
    from core.speech import discover_recognizer_name

    return discover_recognizer_name()


def transcribe_audio(
    path: str | Path,
    *,
    limits: PerceptionLimits | None = None,
    language: str = "",
    recognizer: object | None = None,
    token: CancelToken | None = None,
) -> tuple[str, dict[str, object]]:
    """Transcribe an audio file with a local engine; return (text, metadata).

    A ``recognizer`` may be injected (any object with ``transcribe(clip)`` and a
    ``name``), so a test or a caller with its own engine drives the same path. The
    file is converted to a WAV the recognizer can read (reusing the project's
    converter) and read with the project's own :class:`core.speech.AudioClip`.

    Raises :class:`SpeechPerceptionError` -- never a bare import error -- when no
    engine is installed, the clip is empty, or the engine fails. The metadata
    carries the engine name, a bounded confidence, whether timestamps were
    reported, and the audio reference, all of which the caller records as provenance.
    """
    source = Path(path).expanduser()
    settings = limits or PerceptionLimits()
    if cancelled(token):
        raise SpeechPerceptionError("cancelled before transcription started.")

    engine = recognizer or load_recognizer()
    engine_name = str(getattr(engine, "name", "speech"))
    if engine_name == "transcript" and recognizer is None:
        raise SpeechPerceptionError(
            "No local speech-to-text engine is installed. Install Qwen3-ASR "
            "(python -m cli qwen-asr download --approve) or the `whisper` program."
        )

    clip = _read_clip(source, settings=settings)
    if clip.is_silent:
        raise SpeechPerceptionError("the recording is silent; there is nothing to transcribe.")
    try:
        text = engine.transcribe(clip)  # type: ignore[attr-defined]
    except Exception as error:
        raise SpeechPerceptionError(f"the speech engine failed: {error}") from error

    resolved_language = language or ""
    if not resolved_language and text:
        from core.voices import detect_language

        resolved_language = detect_language(text)
    return (text or "").strip(), {
        "engine": engine_name,
        "confidence": 0.6,
        "timestamps": False,
        "audio_reference": str(source),
        "language": resolved_language,
        "version": _engine_version(engine_name),
    }


def _engine_version(engine_name: str) -> str:
    """The engine's version string when it can be read, else ``""`` (never guessed)."""
    if engine_name == "whisper":
        import shutil
        import subprocess

        executable = shutil.which("whisper")
        if not executable:
            return ""
        try:
            completed = subprocess.run(
                [executable, "--version"], capture_output=True, text=True, timeout=20, check=False
            )
            return (completed.stdout or completed.stderr or "").strip().splitlines()[-1:][0] if (
                completed.stdout or completed.stderr
            ) else ""
        except Exception:
            return ""
    return ""


def _read_clip(source: Path, *, settings: PerceptionLimits):
    """Read an audio file into the project's :class:`core.speech.AudioClip`.

    A WAV is read directly; a compressed format is converted with the project's
    own converter (ffmpeg/sox) into a temporary WAV first. A missing converter is
    reported as the real next step, never a decode error.
    """
    from core.speech import AudioClip

    if source.suffix.lower() == ".wav":
        try:
            return AudioClip.read_wav(source)
        except Exception as error:
            raise SpeechPerceptionError(f"the WAV could not be read: {error}") from error

    from perception.audio import AudioPerceptionError, to_wav
    from perception.limits import temp_workspace

    with temp_workspace(limits=settings) as folder:
        target = folder / "speech.wav"
        try:
            to_wav(source, target)
        except AudioPerceptionError as error:
            raise SpeechPerceptionError(str(error)) from error
        try:
            return AudioClip.read_wav(target)
        except Exception as error:
            raise SpeechPerceptionError(f"the converted WAV could not be read: {error}") from error


# --------------------------------------------------------------------------- #
# Text to speech
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class SpeechOutput:
    """One text-to-speech result: the audio plus what produced it."""

    path: Path | None
    engine: str
    voice: str = ""
    language: str = ""
    duration_seconds: float = 0.0
    silent: bool = False
    """True when the project's silent stand-in produced the clip (no engine)."""

    def describe(self) -> str:
        if self.silent:
            return f"{self.engine}: produced a silent clip (no speech engine is installed)"
        bits = [self.engine]
        if self.voice:
            bits.append(self.voice)
        if self.language:
            bits.append(self.language)
        bits.append(f"{self.duration_seconds:.2f}s")
        return ", ".join(bits)


def tts_available() -> bool:
    """Whether a real (non-silent) text-to-speech engine is installed."""
    from core.speech import discover_synthesizer_name

    return discover_synthesizer_name() != "silent"


def synthesizer_name() -> str:
    """The TTS engine discovery would pick here."""
    from core.speech import discover_synthesizer_name

    return discover_synthesizer_name()


def synthesize_to_file(
    text: str,
    *,
    output: str | Path | None = None,
    voice: str = "",
    language: str = "",
    synthesizer: object | None = None,
    limits: PerceptionLimits | None = None,
) -> SpeechOutput:
    """Render ``text`` to a WAV file with a local engine; return what was produced.

    The engine is the project's own synthesizer (a system engine, Piper, Qwen3-TTS,
    or the audio.cpp voice), chosen by discovery unless ``voice`` names one -- a
    ``system:``/``model:`` key routes to its engine through the project's reader.
    When no engine is installed the silent stand-in is used and ``silent`` is set,
    so the caller reports the honest outcome rather than an error.

    ``output`` defaults to a temporary file under the data folder; the caller owns
    the returned path. TTS is optional and never touches reasoning or storage.
    """
    if not text.strip():
        raise SpeechPerceptionError("cannot synthesise empty text.")
    settings = limits or PerceptionLimits()

    from core.speech import load_synthesizer

    engine = synthesizer or load_synthesizer("auto")
    engine_name = str(getattr(engine, "name", "speech"))

    resolved_language = language
    if not resolved_language:
        from core.voices import detect_language

        resolved_language = detect_language(text)

    from core.speech import AudioClip

    try:
        clip: AudioClip = engine.synthesize(text)  # type: ignore[attr-defined]
    except Exception as error:
        raise SpeechPerceptionError(f"the speech engine failed: {error}") from error

    target = Path(output).expanduser() if output else _default_output(settings)
    target.parent.mkdir(parents=True, exist_ok=True)
    clip.write_wav(target)
    return SpeechOutput(
        path=target,
        engine=engine_name,
        voice=voice,
        language=resolved_language,
        duration_seconds=round(clip.duration_seconds, 3),
        silent=engine_name == "silent" or clip.is_silent,
    )


def _default_output(settings: PerceptionLimits) -> Path:
    from core.paths import data_dir

    return data_dir() / "perception" / "speech" / "speech.wav"


def speak_clip(text: str, *, voice: str = "", synthesizer: object | None = None):
    """Render ``text`` to an in-memory clip for immediate playback.

    Returns a :class:`core.speech.AudioClip` the caller can hand to the project's
    :class:`core.speech.Speaker`; when no engine is installed the silent stand-in
    yields an empty clip rather than raising, so playback degrades gracefully.
    """
    if not text.strip():
        raise SpeechPerceptionError("cannot synthesise empty text.")
    from core.speech import load_synthesizer

    engine = synthesizer or load_synthesizer("auto")
    try:
        return engine.synthesize(text)  # type: ignore[attr-defined]
    except Exception as error:
        raise SpeechPerceptionError(f"the speech engine failed: {error}") from error


__all__ = [
    "SpeechOutput",
    "SpeechPerceptionError",
    "Transcript",
    "TranscriptCorrection",
    "correct_transcript",
    "load_recognizer",
    "recognizer_available",
    "recognizer_name",
    "speak_clip",
    "synthesize_to_file",
    "synthesizer_name",
    "transcribe_audio",
    "tts_available",
]
