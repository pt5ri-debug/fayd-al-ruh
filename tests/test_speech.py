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
    SilentSynthesizer,
    TranscriptRecognizer,
    VoiceAssistant,
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
