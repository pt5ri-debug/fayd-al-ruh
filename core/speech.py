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

from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol, runtime_checkable

from core.model_interface import CognitiveModel, ModelRequest

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


def load_recognizer(name: str, **options) -> SpeechRecognizer:
    """Load a named ASR engine, or fall back to the transcript recognizer."""
    if name == "transcript":
        return TranscriptRecognizer(options.get("transcript", ""))
    raise ValueError(
        f"ASR engine {name!r} is not available in this build. "
        "Install a speech backend or use the transcript recognizer."
    )


def load_synthesizer(name: str, **options) -> SpeechSynthesizer:
    """Load a named TTS engine, or fall back to the silent synthesizer."""
    if name == "silent":
        return SilentSynthesizer()
    raise ValueError(
        f"TTS engine {name!r} is not available in this build. "
        "Install a speech backend or use the silent synthesizer."
    )


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

    def describe(self, clip: AudioClip, *, instruction: str = "صف الإشارة الصوتية.") -> str:
        request = ModelRequest(
            inputs=(),
            system_instruction=instruction,
            context=(
                f"مقطع صوتي: المدة {clip.duration_seconds:.2f} ثانية، "
                f"معدل {clip.sample_rate} هرتز، "
                f"{'صامت' if clip.is_silent else 'يحتوي إشارة'}.",
            ),
        )
        return self.model.generate(request).text


__all__ = [
    "SAMPLE_RATE",
    "AudioClip",
    "AudioModel",
    "SilentSynthesizer",
    "SpeechRecognizer",
    "SpeechSynthesizer",
    "TranscriptRecognizer",
    "VoiceAssistant",
    "VoiceTurn",
    "load_recognizer",
    "load_synthesizer",
]
