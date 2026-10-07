"""Tests for Jarvis mode: the unified listen/route/answer/act/speak loop."""

from __future__ import annotations

import pytest

from core.assistant import GroundedAssistant
from core.command_router import NaturalLanguageRouter
from core.curriculum_knowledge import seed_curriculum
from core.documents import build_document_store
from core.domain_knowledge import seed_ontology
from core.jarvis import JarvisMode
from core.ontology import Ontology
from core.rag import RagRetriever
from core.speech import SilentSynthesizer, TranscriptRecognizer, VoiceAssistant, AudioClip
from core.tool_agent import ToolAgent
from core.tools import build_sandbox
from core.wake_word import AudioSpeechGate, WakeWordDetector


def _assistant():
    ontology = Ontology()
    seed_ontology(ontology)
    seed_curriculum(ontology)
    return GroundedAssistant(RagRetriever(build_document_store(), ontology=ontology))


def _mode(tmp_path, **sandbox_kwargs):
    sandbox = build_sandbox(tmp_path, **sandbox_kwargs)
    assistant = _assistant()
    return JarvisMode(
        assistant,
        router=NaturalLanguageRouter(sandbox),
        agent=ToolAgent(sandbox),
        voice=VoiceAssistant(assistant, recognizer=TranscriptRecognizer(), synthesizer=SilentSynthesizer()),
    )


def test_question_is_answered_with_citations(tmp_path):
    turn = _mode(tmp_path).handle("what is porosity?")
    assert turn.routed == "answer"
    assert turn.answer_text
    assert turn.citations
    assert turn.spoken is not None


def test_tool_request_runs_in_the_sandbox(tmp_path):
    turn = _mode(tmp_path).handle("calculate 2 * (3 + 4)")
    assert turn.routed == "tool"
    assert turn.agent_run is not None
    assert turn.agent_run.succeeded
    assert "14" in turn.response


def test_write_then_read_round_trip(tmp_path):
    mode = _mode(tmp_path, read_only=False)
    write = mode.handle("write to file notes/a.txt: permeability is high")
    assert write.agent_run is not None and write.agent_run.succeeded
    read = mode.handle("read the file notes/a.txt")
    assert read.agent_run is not None and read.agent_run.succeeded
    assert "permeability is high" in read.response


def test_wake_word_gate_ignores_unaddressed_speech(tmp_path):
    mode = _mode(tmp_path)
    turn = mode.handle("what is porosity?", require_wake=True)
    assert turn.woke is False
    assert turn.routed == "ignored"
    assert turn.response == ""


def test_wake_word_then_answers(tmp_path):
    mode = _mode(tmp_path)
    turn = mode.handle("hey jarvis what is porosity?", require_wake=True)
    assert turn.woke is True
    assert turn.routed == "answer"
    assert turn.answer_text


def test_speak_false_produces_no_audio(tmp_path):
    turn = _mode(tmp_path).handle("what is porosity?", speak=False)
    assert turn.spoken is None


def test_audio_below_the_gate_is_not_transcribed(tmp_path):
    mode = _mode(tmp_path)
    turn = mode.handle_audio(AudioClip.silence(1.0), require_wake=True)
    assert turn.routed == "silence"


def test_custom_wake_word(tmp_path):
    mode = _mode(tmp_path)
    mode.wake = WakeWordDetector(phrases=("computer",))
    assert mode.handle("computer what is porosity?", require_wake=True).woke is True
    assert mode.handle("hey jarvis", require_wake=True).woke is False


def test_empty_utterance_is_rejected(tmp_path):
    with pytest.raises(ValueError):
        _mode(tmp_path).handle("   ")


def test_gate_defaults_are_constructible(tmp_path):
    assert AudioSpeechGate().min_duration_seconds > 0
    assert _mode(tmp_path).gate is not None


class _RecordingSpeaker:
    def __init__(self):
        self.played = []

    def play(self, clip):
        self.played.append(clip)


class _FailingSpeaker:
    def play(self, clip):
        raise ValueError("no player")


def test_play_sends_the_clip_to_the_speaker(tmp_path):
    mode = _mode(tmp_path)
    speaker = _RecordingSpeaker()
    mode.speaker = speaker
    turn = mode.handle("what is porosity?")
    assert mode.play(turn) == ""
    assert speaker.played == [turn.spoken]


def test_play_is_best_effort_and_reports_failure(tmp_path):
    mode = _mode(tmp_path)
    mode.speaker = _FailingSpeaker()
    turn = mode.handle("what is porosity?")
    assert mode.play(turn) == "no player"
    assert turn.playback_error == "no player"


def test_play_without_speaker_or_audio_is_a_noop(tmp_path):
    mode = _mode(tmp_path)
    assert mode.play(mode.handle("what is porosity?")) == ""  # no speaker attached
    mode.speaker = _RecordingSpeaker()
    assert mode.play(mode.handle("what is porosity?", speak=False)) == ""  # no clip
