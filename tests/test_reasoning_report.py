"""Tests for the user-visible reasoning trace (``core.reasoning_report``)."""

from __future__ import annotations

import pytest
from ai.chat import format_turn, turn_payload
from core.answer_confidence import answer_confidence
from core.assistant import GroundedAssistant
from core.cognitive_controller import CognitiveController
from core.command_router import NaturalLanguageRouter
from core.documents import build_document_store
from core.jarvis import JarvisMode
from core.jarvis_controls import ControlIntent, JarvisControls
from core.rag import RagRetriever, RetrievedPassage
from core.reasoning_report import (
    SOURCE_CAUSAL,
    SOURCE_CONFIDENCE,
    SOURCE_CONNECTIONS,
    SOURCE_EVIDENCE,
    SOURCE_INFERRED,
    SOURCE_STORED,
    STEP_ORDER,
    TRACE_JUDGEMENTS,
    ReasoningReport,
    ReasoningStep,
    TraceJudgement,
    reasoning_report,
    reasoning_trace_accuracy,
)
from core.tools import build_sandbox


def _ontology():
    return CognitiveController().ontology


def _passage(text: str, citation: str = "doc#1", score: float = 0.8) -> RetrievedPassage:
    return RetrievedPassage(
        doc_id=citation,
        title=citation,
        text=text,
        source="src",
        score=score,
        signal="lexical",
    )


def _assistant() -> GroundedAssistant:
    return GroundedAssistant(
        RagRetriever(build_document_store(), ontology=_ontology())
    )


# --- the step object -------------------------------------------------------


def test_step_describe_lists_its_lines():
    step = ReasoningStep(SOURCE_EVIDENCE, ("one", "two"))
    described = step.describe()
    assert described.startswith(SOURCE_EVIDENCE)
    assert "  - one" in described
    assert "  - two" in described


def test_step_describe_names_an_empty_step():
    assert "(nothing)" in ReasoningStep(SOURCE_INFERRED, ()).describe()


def test_step_derived_flag_defaults_false():
    assert ReasoningStep(SOURCE_STORED, ("x",)).derived is False


# --- the report object -----------------------------------------------------


def test_report_keeps_the_declared_step_order():
    report = reasoning_report("q")
    assert tuple(step.label for step in report.steps) == STEP_ORDER


def test_report_is_empty_when_nothing_contributed():
    assert reasoning_report("q").is_empty is True


def test_report_is_not_empty_when_evidence_contributed():
    report = reasoning_report("q", passages=[_passage("Permeability measures flow.")])
    assert report.is_empty is False


def test_report_describe_names_the_question():
    report = reasoning_report("what is permeability?", passages=[_passage("x")])
    assert "what is permeability?" in report.describe()


def test_report_empty_describe_is_honest():
    assert "no evidence" in reasoning_report("q").describe().lower()


def test_report_derived_steps_selects_only_derivations():
    report = reasoning_report("q", passages=[_passage("x")])
    assert report.derived_steps == ()


def test_report_marks_derived_steps():
    assistant = _assistant()
    answer = assistant.ask("how does waterflood affect production?")
    report = answer.reasoning
    assert report is not None
    derived = {step.label for step in report.derived_steps}
    # Whatever the graph derived, it must be labelled as derived.
    for label in derived:
        assert label in (SOURCE_INFERRED, SOURCE_CONNECTIONS, SOURCE_CAUSAL)


def test_report_never_invents_a_step():
    report = reasoning_report("q")
    assert len(report.steps) == len(STEP_ORDER)


def test_every_step_label_is_a_known_source():
    for step in reasoning_report("q").steps:
        assert step.label in STEP_ORDER


# --- the confidence step ---------------------------------------------------


def test_confidence_step_renders_the_breakdown():
    passages = [_passage("Permeability measures flow.")]
    report = reasoning_report("q", passages=passages, confidence=answer_confidence(passages))
    step = next(s for s in report.steps if s.label == SOURCE_CONFIDENCE)
    assert any("final" in line for line in step.lines)
    assert any("evidence" in line for line in step.lines)


def test_confidence_step_is_empty_without_a_report():
    report = reasoning_report("q")
    step = next(s for s in report.steps if s.label == SOURCE_CONFIDENCE)
    assert step.lines == ()


# --- the assistant carries a trace -----------------------------------------


def test_assistant_answer_carries_a_reasoning_report():
    answer = _assistant().ask("what is porosity?")
    assert isinstance(answer.reasoning, ReasoningReport)
    assert answer.reasoning.question == "what is porosity?"


def test_assistant_trace_includes_evidence():
    answer = _assistant().ask("what is porosity?")
    step = next(s for s in answer.reasoning.steps if s.label == SOURCE_EVIDENCE)
    assert step.lines


def test_assistant_trace_describe_is_readable():
    answer = _assistant().ask("how does waterflood affect production?")
    described = answer.reasoning.describe()
    assert "How this answer was reached" in described


# --- the why control -------------------------------------------------------


def _jarvis(tmp_path, assistant: GroundedAssistant) -> JarvisMode:
    return JarvisMode(assistant, router=NaturalLanguageRouter(build_sandbox(tmp_path)))


def test_why_is_recognised_as_a_control():
    assert JarvisControls().match("why").intent is ControlIntent.WHY


def test_why_arabic_is_recognised():
    assert JarvisControls().match("لماذا").intent is ControlIntent.WHY


def test_a_why_question_is_not_a_control():
    # A real question that begins with "why" must still reach the assistant.
    assert JarvisControls().match("why does waterflood increase production?") is None


def test_why_without_an_answer_says_so(tmp_path):
    jarvis = _jarvis(tmp_path, _assistant())
    turn = jarvis.handle("why")
    assert turn.routed == "control"
    assert turn.intent == ControlIntent.WHY.value
    assert "no recent answer" in turn.response


def test_why_repeats_the_last_reasoning_trace(tmp_path):
    jarvis = _jarvis(tmp_path, _assistant())
    jarvis.handle("how does waterflood affect production?")
    turn = jarvis.handle("why")
    assert turn.intent == ControlIntent.WHY.value
    assert "How this answer was reached" in turn.response


def test_new_conversation_clears_the_reasoning_trace(tmp_path):
    jarvis = _jarvis(tmp_path, _assistant())
    jarvis.handle("what is porosity?")
    jarvis.handle("new conversation")
    turn = jarvis.handle("why")
    assert "no recent answer" in turn.response


def test_answer_turn_carries_the_reasoning_trace(tmp_path):
    jarvis = _jarvis(tmp_path, _assistant())
    turn = jarvis.handle("how does waterflood affect production?")
    assert "How this answer was reached" in turn.reasoning_trace


# --- surfaces --------------------------------------------------------------


def test_format_turn_renders_the_reasoning_trace(tmp_path):
    jarvis = _jarvis(tmp_path, _assistant())
    turn = jarvis.handle("how does waterflood affect production?")
    assert "How this answer was reached" in format_turn(turn)


def test_turn_payload_carries_the_reasoning_trace(tmp_path):
    jarvis = _jarvis(tmp_path, _assistant())
    turn = jarvis.handle("how does waterflood affect production?")
    assert "How this answer was reached" in turn_payload(turn)["reasoning_trace"]


def test_help_lists_the_why_control(tmp_path):
    jarvis = _jarvis(tmp_path, _assistant())
    assert "why" in jarvis.handle("help").response


# --- the held-out gate -----------------------------------------------------


def test_trace_accuracy_is_perfect_on_the_catalogue():
    assert reasoning_trace_accuracy() == 1.0


def test_trace_accuracy_is_zero_for_an_empty_catalogue():
    assert reasoning_trace_accuracy(()) == 0.0


def test_trace_judgement_rejects_an_unknown_label():
    with pytest.raises(ValueError):
        TraceJudgement("bad", ("Not a real step",), 0, 0)


def test_trace_judgement_rejects_an_empty_name():
    with pytest.raises(ValueError):
        TraceJudgement("  ", (), 0, 0)


def test_trace_judgement_rejects_negative_counts():
    with pytest.raises(ValueError):
        TraceJudgement("bad", (), -1, 0)


def test_the_catalogue_has_a_negative_control():
    # At least one judgement must expect *nothing* to have contributed.
    assert any(not judgement.filled for judgement in TRACE_JUDGEMENTS)


def test_a_mislabelled_derivation_fails_the_gate():
    # If a walked connection were counted as a stored claim, the gate must drop.
    mislabelled = TraceJudgement("wrong split", (SOURCE_CONNECTIONS,), 1, 0)
    assert reasoning_trace_accuracy((mislabelled,)) == 0.0


def test_a_dropped_empty_step_would_fail_the_gate():
    # "evidence only" with an extra invented stored step must not pass.
    invented = TraceJudgement("invented", (SOURCE_EVIDENCE, SOURCE_STORED), 1, 0)
    assert reasoning_trace_accuracy((invented,)) == 0.0


def test_every_step_label_appears_in_the_catalogue():
    covered = {label for judgement in TRACE_JUDGEMENTS for label in judgement.filled}
    assert covered == set(STEP_ORDER)

