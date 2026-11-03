"""The audio processing pipeline (section 6).

An audio file is perceived in three steps, each optional and each honest:

1. **Validation and metadata** -- is it a readable audio file, how long is it,
   what sample rate? A WAV header is read directly with the standard library; a
   compressed format (MP3) needs a decoder to measure, and this module says so
   rather than guessing a duration.
2. **Conversion to a form the recognizer can read** -- the project's own
   :func:`core.audiocpp_speech._to_wav` converts with ``ffmpeg``/``sox`` when
   installed, and reports the real next step when neither is present.
3. **Speech-to-text** -- delegated to :mod:`perception.speech`, which wraps the
   project's STT factory.

This module never claims speaker identity recognition: the pipeline does not do
speaker diarisation (no compatible local implementation is bundled), and it says
so rather than pretending. Transcript timestamps are recorded only where the
engine reports them; the bundled engines do not, so a segment carries the clip's
own span and the honest note that word-level timing is unavailable.

Nothing here requires an online service: every path is a local program or a local
model.
"""

from __future__ import annotations

import wave
from dataclasses import dataclass
from pathlib import Path

from perception.detector import file_digest
from perception.limits import CancelToken, PerceptionLimits, cancelled
from perception.metadata import extract_metadata
from perception.schema import (
    ContentType,
    ExtractionMethod,
    PerceptionResult,
    PerceptionStatus,
    PerceptionWarning,
    Segment,
    SourceType,
)


class AudioPerceptionError(ValueError):
    """An audio file could not be perceived (unreadable, too long, no decoder)."""


@dataclass(frozen=True)
class AudioInfo:
    """What is known about an audio file, and what could not be determined."""

    path: Path
    readable: bool
    duration_seconds: float | None = None
    sample_rate: int | None = None
    channels: int | None = None
    format: str = ""
    note: str = ""

    def describe(self) -> str:
        if not self.readable:
            return f"{self.path.name}: unreadable ({self.note or 'unknown reason'})"
        bits = [f"{self.path.name}: {self.format or 'audio'}"]
        if self.duration_seconds is not None:
            bits.append(f"{self.duration_seconds:.2f}s")
        if self.sample_rate:
            bits.append(f"{self.sample_rate} Hz")
        if self.channels:
            bits.append(f"{self.channels} ch")
        if self.note:
            bits.append(self.note)
        return ", ".join(bits)


def audio_info(path: str | Path) -> AudioInfo:
    """Read what can be determined about an audio file without decoding it whole.

    A WAV's duration/rate/channels come from its own header (standard library). A
    compressed format is reported as readable only if a converter exists to turn
    it into a WAV; its duration is left ``None`` with an honest note, because
    guessing it would be a lie.
    """
    source = Path(path).expanduser()
    if not source.exists() or not source.is_file():
        return AudioInfo(source, False, note="the file does not exist.")
    if source.suffix.lower() == ".wav":
        try:
            with wave.open(str(source), "rb") as handle:
                frames = handle.getnframes()
                rate = handle.getframerate()
                channels = handle.getnchannels()
            duration = frames / rate if rate else 0.0
            return AudioInfo(
                source,
                True,
                duration_seconds=round(duration, 3),
                sample_rate=rate,
                channels=channels,
                format="wav",
            )
        except (wave.Error, OSError, EOFError) as error:
            return AudioInfo(source, False, note=f"not a readable WAV: {error}")

    import shutil

    if shutil.which("ffmpeg") or shutil.which("sox"):
        return AudioInfo(
            source,
            True,
            format=source.suffix.lower().lstrip("."),
            note="duration needs decoding (a converter is available)",
        )
    return AudioInfo(
        source,
        False,
        format=source.suffix.lower().lstrip("."),
        note="a compressed audio file needs ffmpeg or sox to read; neither is installed.",
    )


def validate_audio(
    path: str | Path,
    *,
    limits: PerceptionLimits | None = None,
) -> tuple[bool, str]:
    """Whether an audio file can be processed, and why not when it cannot.

    Checks existence, readability (a WAV header, or a converter for a compressed
    format), and the duration limit. Returns a ``(ok, reason)`` pair rather than
    raising, so a caller turns a failure into a ``FAILED`` result.
    """
    settings = limits or PerceptionLimits()
    info = audio_info(path)
    if not info.readable:
        return False, info.note or "the audio file is not readable."
    if info.duration_seconds is not None and info.duration_seconds > settings.max_audio_seconds:
        return (
            False,
            f"the recording is {info.duration_seconds:.0f}s, over the "
            f"{settings.max_audio_seconds:.0f}s limit.",
        )
    return True, info.describe()


def to_wav(path: str | Path, target: Path) -> Path:
    """Convert ``path`` to a WAV at ``target``, reusing the project's converter.

    Raises :class:`AudioPerceptionError` with the real next step when the file is
    a compressed format and no converter is installed.
    """
    from core.audiocpp_speech import _to_wav

    source = Path(path).expanduser()
    try:
        _to_wav(source, target)
    except ValueError as error:
        raise AudioPerceptionError(str(error)) from error
    return target


def perceive_audio(
    path: str | Path,
    *,
    limits: PerceptionLimits | None = None,
    transcribe: bool = True,
    language: str = "",
    recognizer: object | None = None,
    token: CancelToken | None = None,
) -> PerceptionResult:
    """Perceive an audio file: validate it, measure it, and (optionally) transcribe it.

    ``transcribe=False`` reads only the metadata (a cheap probe); ``transcribe=True``
    runs local speech-to-text through :func:`perception.speech.transcribe_audio`.
    A ``recognizer`` may be injected (a test's, or a caller's own engine) so the
    pipeline is exercised with no model and no audio hardware.

    The transcript is stored with ``content_type=ASR`` and a confidence below 1.0:
    speech recognition output is **never** treated as verified factual knowledge,
    and the caller can correct it (the correction keeps a link to this result).
    """
    source = Path(path).expanduser()
    settings = limits or PerceptionLimits()
    if not source.exists() or not source.is_file():
        return _failed(source, "the file does not exist.")

    ok, reason = validate_audio(source, limits=settings)
    if not ok:
        return _failed(source, reason)
    if cancelled(token):
        return _failed(source, "cancelled before transcription started.")

    info = audio_info(source)
    digest = ""
    try:
        digest = file_digest(source)
    except OSError:
        pass
    metadata = extract_metadata(source)
    metadata.update(
        {
            "duration_seconds": info.duration_seconds,
            "sample_rate": info.sample_rate,
            "channels": info.channels,
        }
    )

    if not transcribe:
        return PerceptionResult(
            source_type=SourceType.AUDIO,
            source_reference=str(source),
            extraction_method=ExtractionMethod.AUDIO,
            content="",
            content_type=ContentType.METADATA,
            segments=(),
            confidence=1.0,
            status=PerceptionStatus.OK,
            provenance={"source_digest": digest},
            metadata=metadata,
        )

    from perception.speech import SpeechPerceptionError, transcribe_audio

    try:
        text, speech_meta = transcribe_audio(
            source, limits=settings, language=language, recognizer=recognizer
        )
    except SpeechPerceptionError as error:
        return _failed(source, str(error))

    text = (text or "").strip()
    warnings: list[PerceptionWarning] = [
        PerceptionWarning(
            "speaker separation is not performed: no compatible local diarisation is bundled"
        )
    ]
    if not speech_meta.get("timestamps"):
        warnings.append(
            PerceptionWarning("the engine reports no word-level timestamps for this clip")
        )

    duration = info.duration_seconds
    segments: tuple[Segment, ...]
    if text:
        segments = (
            Segment(
                index=1,
                text=text,
                label=_time_label(0.0, duration),
                content_type=ContentType.ASR,
                start_seconds=0.0,
                end_seconds=duration,
                confidence=_as_float(speech_meta.get("confidence"), 0.6),
            ),
        )
    else:
        segments = ()
    status = PerceptionStatus.OK if text else PerceptionStatus.EMPTY
    if text and warnings:
        status = PerceptionStatus.PARTIAL

    from core.voices import detect_language

    resolved_language = language or detect_language(text)
    metadata["engine"] = speech_meta.get("engine", "")
    return PerceptionResult(
        source_type=SourceType.AUDIO,
        source_reference=str(source),
        extraction_method=ExtractionMethod.SPEECH,
        content=text,
        content_type=ContentType.ASR,
        language=resolved_language if text else "",
        segments=segments,
        confidence=_as_float(speech_meta.get("confidence"), 0.6) if text else 0.0,
        status=status,
        warnings=tuple(warnings),
        model_name=str(speech_meta.get("engine", "")),
        model_version=str(speech_meta.get("version", "")),
        provenance={
            "source_digest": digest,
            "engine": speech_meta.get("engine", ""),
            "timestamps": bool(speech_meta.get("timestamps")),
        },
        metadata=metadata,
    )


def _time_label(start: float, end: float | None) -> str:
    """A readable ``mm:ss`` label for an audio segment (or a start-only label)."""

    def stamp(seconds: float) -> str:
        total = int(seconds)
        return f"{total // 60:02d}:{total % 60:02d}"

    if end is None:
        return stamp(start)
    return f"{stamp(start)}-{stamp(end)}"


def _as_float(value: object, default: float) -> float:
    """Read a metadata number, falling back to ``default`` rather than guessing."""
    try:
        return float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return default


def _failed(source: Path, reason: str) -> PerceptionResult:
    return PerceptionResult(
        source_type=SourceType.AUDIO,
        source_reference=str(source),
        extraction_method=ExtractionMethod.AUDIO,
        status=PerceptionStatus.FAILED,
        errors=(reason,),
    )


__all__ = [
    "AudioInfo",
    "AudioPerceptionError",
    "audio_info",
    "perceive_audio",
    "to_wav",
    "validate_audio",
]
