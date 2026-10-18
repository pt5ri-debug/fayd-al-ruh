"""Tests for the speech layer: audio clips, recognizer/synthesizer, voice loop."""

from __future__ import annotations

import pytest

from core.assistant import GroundedAssistant
from core.documents import build_document_store
from core.ontology import Ontology
from core.domain_knowledge import seed_ontology
from core.curriculum_knowledge import seed_curriculum
from core.rag import RagRetriever
from core.speech import (
    AudioClip,
    AudioModel,
    CommandSynthesizer,
    MicrophoneRecorder,
    MicRecording,
    Playback,
    SilentSynthesizer,
    Speaker,
    TranscriptRecognizer,
    VoiceAssistant,
    WhisperRecognizer,
    discover_recognizer_name,
    discover_synthesizer_name,
    load_recognizer,
    load_synthesizer,
)


def _assistant():
    ontology = Ontology()
    seed_ontology(ontology)
    seed_curriculum(ontology)
    return GroundedAssistant(RagRetriever(build_document_store(), ontology=ontology))


def test_audio_clip_reports_duration_and_silence():
    clip = AudioClip.silence(1.0)
    assert clip.duration_seconds == pytest.approx(1.0)
    assert clip.is_silent
    assert AudioClip((1, -1, 2)).is_silent is False


def test_audio_clip_validates_samples_and_rate():
    with pytest.raises(ValueError):
        AudioClip((40000,))
    with pytest.raises(ValueError):
        AudioClip((0,), sample_rate=0)
    with pytest.raises(ValueError):
        AudioClip.silence(-1)


def test_audio_clip_bytes_round_trip():
    clip = AudioClip((100, -200, 300, -400))
    assert AudioClip.from_bytes(clip.to_bytes()) == clip


def test_audio_clip_wav_round_trip(tmp_path):
    clip = AudioClip(tuple(range(-100, 100)))
    path = tmp_path / "clip.wav"
    clip.write_wav(path)
    restored = AudioClip.read_wav(path)
    assert restored.samples == clip.samples
    assert restored.sample_rate == clip.sample_rate


def test_silent_synthesizer_sizes_to_text():
    clip = SilentSynthesizer().synthesize("some text")
    assert clip.is_silent
    assert clip.duration_seconds > 0
    with pytest.raises(ValueError):
        SilentSynthesizer().synthesize("  ")


def test_transcript_recognizer_returns_supplied_text():
    assert TranscriptRecognizer("hello").transcribe(AudioClip.silence(0.1)) == "hello"
    with pytest.raises(ValueError):
        TranscriptRecognizer().transcribe(AudioClip.silence(0.1))


def test_loaders_fall_back_and_reject_unknown_engines():
    assert isinstance(load_recognizer("transcript", transcript="text"), TranscriptRecognizer)
    assert isinstance(load_synthesizer("silent"), SilentSynthesizer)
    with pytest.raises(ValueError):
        load_recognizer("whisper")
    with pytest.raises(ValueError):
        load_synthesizer("piper")


def test_voice_assistant_runs_the_full_loop():
    voice = VoiceAssistant(
        _assistant(),
        recognizer=TranscriptRecognizer("What is the relationship between porosity and permeability?"),
        synthesizer=SilentSynthesizer(),
    )
    turn = voice.respond_to_speech(AudioClip.silence(0.2))
    assert turn.heard == "What is the relationship between porosity and permeability?"
    assert turn.answer_text
    assert turn.citations
    assert turn.spoken.is_silent


def test_voice_assistant_keeps_conversation():
    voice = VoiceAssistant(_assistant(), recognizer=TranscriptRecognizer("permeability"))
    conversation = voice.assistant.memory.start(title="voice")
    voice.respond_to_speech(AudioClip.silence(0.1), conversation=conversation)
    assert len(conversation.messages) == 2


def test_voice_assistant_rejects_empty_speech():
    voice = VoiceAssistant(_assistant(), recognizer=TranscriptRecognizer("   "))
    with pytest.raises(ValueError):
        voice.respond_to_speech(AudioClip.silence(0.1))


def test_audio_model_describes_a_clip():
    class _Echo:
        def generate(self, request):
            from core.model_interface import ModelResponse

            return ModelResponse(text="description", confidence=0.5, model_name="echo")

    description = AudioModel(_Echo()).describe(AudioClip.silence(0.5))
    assert description == "description"


# --- system engines -------------------------------------------------------


def test_discovery_returns_a_known_engine_name():
    assert discover_synthesizer_name() in {"silent", "espeak-ng", "espeak", "flite", "say"}
    assert discover_recognizer_name() in {"transcript", "whisper"}


def test_auto_loader_always_returns_a_usable_engine():
    synth = load_synthesizer("auto")
    recog = load_recognizer("auto", transcript="hello")
    assert synth.name in {"silent", "espeak-ng", "espeak", "flite", "say"}
    assert recog.name in {"transcript", "whisper"}
    assert synth.synthesize("hello").duration_seconds > 0


def test_auto_loader_falls_back_and_named_engine_raises():
    # 'auto' never fails: it picks the best available engine or a safe stand-in.
    assert load_synthesizer("auto").name in {"silent", "espeak-ng", "espeak", "flite", "say"}
    assert load_recognizer("auto", transcript="hi").name in {"transcript", "whisper"}
    # an explicitly named, uninstalled engine raises rather than faking audio.
    with pytest.raises(ValueError):
        load_synthesizer("definitely-not-installed")
    with pytest.raises(ValueError):
        load_recognizer("definitely-not-installed")


def test_command_synthesizer_falls_back_when_engine_fails():
    def _builder(executable, text, path):
        return ["definitely-not-a-real-program", text]

    synth = CommandSynthesizer("fake", "definitely-not-a-real-program", _builder)
    clip = synth.synthesize("hello world")
    assert clip.is_silent  # honest fallback, never fabricated audio


def test_command_synthesizer_rejects_empty_text():
    synth = CommandSynthesizer("fake", "espeak", lambda e, t, p: [e, t])
    with pytest.raises(ValueError):
        synth.synthesize("   ")


def test_whisper_recognizer_reports_failure_without_inventing_speech():
    recognizer = WhisperRecognizer("definitely-not-a-real-program")
    with pytest.raises(ValueError):
        recognizer.transcribe(AudioClip.silence(0.5))


def test_microphone_recorder_reports_availability_honestly():
    recorder = MicrophoneRecorder()
    assert recorder.available() == (recorder.recorder is not None)


def test_speaker_reports_availability_honestly():
    speaker = Speaker()
    assert speaker.available() == (speaker.player is not None)


def test_speaker_raises_when_no_player_is_available(monkeypatch):
    import core.speech as speech

    monkeypatch.setattr(speech.shutil, "which", lambda name: None)
    with pytest.raises(ValueError):
        Speaker().play(AudioClip.silence(0.1))


class _FakeProcess:
    """A stand-in for the player/recorder child process (no real audio)."""

    def __init__(self) -> None:
        self.returncode: int | None = None  # None while running, like Popen
        self.signals: list[int] = []
        self.killed = False

    def poll(self):
        return self.returncode

    def send_signal(self, sig) -> None:
        self.signals.append(int(sig))
        self.returncode = 0

    def terminate(self) -> None:
        self.returncode = 0

    def kill(self) -> None:
        self.killed = True
        self.returncode = 0

    def wait(self, timeout=None) -> int:
        assert self.returncode is not None
        return self.returncode


def test_playback_stops_a_running_clip():
    """A Playback can stop the child process and reports it was stopped."""
    from pathlib import Path

    process = _FakeProcess()
    playback = Playback(process, player="aplay", folder=Path("/tmp/does-not-exist"))
    assert playback.playing is True
    playback.stop()
    assert playback.stopped is True
    assert process.returncode == 0  # terminated, not left running


def test_playback_wait_is_quiet_after_a_stop():
    """Stopping then waiting never reports a spurious failure."""
    from pathlib import Path

    process = _FakeProcess()
    playback = Playback(process, player="aplay", folder=Path("/tmp/does-not-exist"))
    playback.stop()
    playback.wait()  # must not raise


def test_microphone_recording_stops_and_finalises():
    """A MicRecording sends SIGINT so the recorder flushes its WAV header."""
    from pathlib import Path

    process = _FakeProcess()
    recording = MicRecording(
        process, path=Path("/tmp/nope.wav"), recorder="arecord", folder=Path("/tmp/nope")
    )
    assert recording.recording is True
    recording.stop()
    assert recording.recording is False
    assert process.signals  # SIGINT was delivered
