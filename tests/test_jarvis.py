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
from core.speech import AudioClip, SilentSynthesizer, TranscriptRecognizer, VoiceAssistant
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


def _mode_with_memory(tmp_path, memory):
    sandbox = build_sandbox(tmp_path, read_only=False)
    assistant = _assistant()
    return JarvisMode(
        assistant,
        router=NaturalLanguageRouter(sandbox),
        agent=ToolAgent(sandbox),
        voice=VoiceAssistant(assistant, recognizer=TranscriptRecognizer(), synthesizer=SilentSynthesizer()),
        memory=memory,
    )


def _learning_mode(tmp_path):
    """A mode whose assistant and learner share one document store."""
    from core.continuous_learning import ContinuousLearner
    from core.knowledge_update import KnowledgeUpdater

    documents = build_document_store()
    ontology = Ontology()
    seed_ontology(ontology)
    seed_curriculum(ontology)
    assistant = GroundedAssistant(RagRetriever(documents, ontology=ontology))
    learner = ContinuousLearner(KnowledgeUpdater(documents))
    sandbox = build_sandbox(tmp_path)
    return JarvisMode(
        assistant,
        router=NaturalLanguageRouter(sandbox),
        learner=learner,
    )


def test_question_is_answered_with_citations(tmp_path):
    turn = _mode(tmp_path).handle("what is porosity?")
    assert turn.routed == "answer"
    assert turn.answer_text
    assert turn.citations
    assert turn.spoken is not None


def test_sources_are_listed_at_the_end_not_inline(tmp_path):
    from core.composition import sources_block

    turn = _mode(tmp_path).handle("what is porosity?")
    # The prose must not interrupt itself with a citation marker...
    assert "[" not in turn.answer_text
    # ...and the display helper appends the source list after the body.
    block = sources_block(turn.citations)
    assert block.startswith("Sources:")
    displayed = f"{turn.answer_text}\n{block}"
    assert displayed.index("Sources:") > displayed.index(turn.answer_text.splitlines()[0])


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


def test_follow_up_turns_share_one_conversation(tmp_path):
    mode = _mode(tmp_path)
    first = mode.handle("what is porosity?")
    second = mode.handle("what is permeability?")
    assert first.conversation_id
    assert first.conversation_id == second.conversation_id
    assert first.conversation_id == mode.conversation.id
    # user+assistant for each of the two turns
    assert len(mode.conversation.messages) == 4


def test_tool_turns_are_recorded_in_the_conversation(tmp_path):
    mode = _mode(tmp_path)
    mode.handle("calculate 2 * (3 + 4)")
    roles = [message.role.value for message in mode.conversation.messages]
    assert roles == ["user", "assistant"]
    assert "14" in mode.conversation.last_assistant_message().text


def test_stop_mutes_the_voice(tmp_path):
    mode = _mode(tmp_path)
    turn = mode.handle("stop")
    assert turn.routed == "control"
    assert turn.intent == "mute"
    assert mode.mute is True
    assert turn.spoken is None
    assert mode.handle("what is porosity?", speak=True).spoken is None


def test_speak_turns_the_voice_back_on(tmp_path):
    mode = _mode(tmp_path)
    mode.handle("stop")
    turn = mode.handle("speak")
    assert turn.intent == "speak"
    assert mode.mute is False
    assert turn.spoken is not None


def test_repeat_replays_the_last_answer_with_citations(tmp_path):
    mode = _mode(tmp_path)
    answer = mode.handle("what is porosity?")
    turn = mode.handle("repeat that")
    assert turn.intent == "repeat"
    assert turn.response == answer.answer_text
    assert turn.citations == answer.citations


def test_repeat_before_any_answer_is_honest(tmp_path):
    mode = _mode(tmp_path)
    turn = mode.handle("repeat")
    assert turn.intent == "repeat"
    assert "nothing to repeat" in turn.response


def test_new_conversation_starts_over(tmp_path):
    mode = _mode(tmp_path)
    first = mode.handle("what is porosity?").conversation_id
    turn = mode.handle("new conversation")
    assert turn.intent == "reset"
    assert mode.conversation.id != first
    assert len(mode.conversation.messages) == 0


def test_help_lists_the_controls(tmp_path):
    mode = _mode(tmp_path)
    turn = mode.handle("help")
    assert turn.intent == "help"
    assert "stop" in turn.response
    assert "repeat" in turn.response
    assert turn.spoken is not None


def test_controls_do_not_run_in_the_sandbox(tmp_path):
    mode = _mode(tmp_path)
    turn = mode.handle("stop")
    assert turn.agent_run is None


def test_save_without_a_store_is_honest(tmp_path):
    mode = _mode(tmp_path)
    mode.handle("what is porosity?")
    turn = mode.handle("save")
    assert turn.intent == "save"
    assert "not persistent" in turn.response


def test_sessions_lists_nothing_without_a_store(tmp_path):
    mode = _mode(tmp_path)
    assert mode.list_sessions() == ()
    assert "No saved sessions" in mode.handle("sessions").response


def test_session_round_trip_with_a_persistent_store(tmp_path):
    from core.conversation import ConversationMemory
    from storage.json_store import JsonStore

    store_root = tmp_path / "state"
    memory = ConversationMemory(JsonStore(store_root / "jarvis"))
    mode = _mode_with_memory(tmp_path, memory)
    mode.handle("what is porosity?")
    saved_id = mode.save_session()
    assert mode.memory.persistent
    assert saved_id.endswith(mode.conversation.id + ".")
    assert mode.list_sessions() == (mode.conversation.id,)

    # A fresh mode over the same store resumes the saved session and context.
    resumed = _mode_with_memory(tmp_path, ConversationMemory(JsonStore(store_root / "jarvis")))
    assert resumed.conversation.id != mode.conversation.id
    assert resumed.resume_session(mode.conversation.id) is True
    assert resumed.conversation.last_assistant_message() is not None
    assert resumed.handle("repeat").response == mode.conversation.last_assistant_message().text


def test_resume_unknown_session_reports_false(tmp_path):
    mode = _mode(tmp_path)
    assert mode.resume_session("CNV-doesnotexist") is False


def test_confirm_reports_added_and_records_feedback(tmp_path):
    mode = _learning_mode(tmp_path)
    mode.handle("what is porosity?")
    turn = mode.handle("that's right")
    assert turn.intent == "confirm"
    assert "recorded" in turn.response
    assert mode.learner is not None
    assert mode.learner.steps and mode.learner.steps[-1].changed_corpus


def test_correction_changes_the_next_answer(tmp_path):
    mode = _learning_mode(tmp_path)
    mode.handle("what is porosity?")
    turn = mode.handle("that's wrong, it's porosity is the void fraction of a rock")
    assert turn.intent == "correct"
    assert "recorded" in turn.response
    learned = mode.handle("porosity is the void fraction of a rock").response
    assert "void fraction" in learned


def test_confirm_without_a_learner_is_honest(tmp_path):
    mode = _mode(tmp_path)
    mode.handle("what is porosity?")
    turn = mode.handle("that's right")
    assert turn.intent == "confirm"
    assert "not enabled" in turn.response


def test_feedback_without_a_prior_answer_says_so(tmp_path):
    mode = _learning_mode(tmp_path)
    turn = mode.handle("that's right")
    assert turn.intent == "confirm"
    assert "no recent answer" in turn.response


def test_corrected_answer_uses_a_rejected_statement_honestly(tmp_path):
    mode = _learning_mode(tmp_path)
    mode.handle("what is porosity?")
    turn = mode.handle("no, it's xyzzy")
    assert turn.intent == "correct"
    assert "recorded" in turn.response


def test_last_exchange_tracks_the_answered_pair():
    from core.conversation import Conversation

    conversation = Conversation("t")
    assert conversation.last_exchange() is None
    conversation.add_user("what is porosity?")
    assert conversation.last_exchange() is None
    conversation.add_assistant("Porosity is the void fraction.")
    assert conversation.last_exchange() == ("what is porosity?", "Porosity is the void fraction.")
    conversation.add_user("and permeability?")
    assert conversation.last_exchange() is None


# --- internet controls ----------------------------------------------------


def _web_mode(tmp_path, *, documents=None, weak_evidence=0.9, on_learn=None):
    """A mode whose assistant has a (static) web verifier wired in."""
    from core.web_source import StaticSearchProvider, WebResult
    from core.web_verifier import approved_gate, build_web_verifier, granted_permissions

    documents = documents or build_document_store()
    ontology = Ontology()
    seed_ontology(ontology)
    seed_curriculum(ontology)
    assistant = GroundedAssistant(RagRetriever(documents, ontology=ontology))
    web = build_web_verifier(
        provider=StaticSearchProvider(
            (
                WebResult(
                    "https://www.spe.org/archie",
                    "Archie water saturation",
                    "The Archie equation estimates water saturation from resistivity in the "
                    "reservoir rock using porosity and the formation water resistivity.",
                    0.9,
                ),
            )
        ),
        permissions=granted_permissions(),
        approvals=approved_gate(),
        retriever=assistant.retriever,
        trust_threshold=0.5,
    )
    assistant.web = web
    assistant.web_enabled = True
    assistant.weak_evidence = weak_evidence
    if on_learn is not None:
        assistant._on_learn = on_learn
    sandbox = build_sandbox(tmp_path)
    return JarvisMode(assistant, router=NaturalLanguageRouter(sandbox))


def test_web_control_toggles_the_assistant(tmp_path):
    mode = _web_mode(tmp_path)
    off = mode.handle("web off")
    assert off.intent == "web"
    assert mode.assistant.web_enabled is False
    assert "off" in off.response.casefold()

    on = mode.handle("web on")
    assert mode.assistant.web_enabled is True
    assert "on" in on.response.casefold()


def test_web_control_is_honest_without_a_verifier(tmp_path):
    mode = _mode(tmp_path)
    turn = mode.handle("web on")
    assert turn.intent == "web"
    assert "not configured" in turn.response


def test_jarvis_reports_a_web_learned_answer(tmp_path):
    documents = build_document_store()
    mode = _web_mode(tmp_path, documents=documents, on_learn=documents.save)
    turn = mode.handle("Explain the Archie equation for water saturation in reservoir rock.")

    assert turn.used_web
    assert turn.learned
    assert "web" in turn.response.casefold()
