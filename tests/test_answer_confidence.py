"""Tests for folding graph reasoning into the answer's confidence."""

from __future__ import annotations

import pytest
from core.answer_confidence import (
    CONFIDENCE_JUDGEMENTS,
    CONFLICT_PENALTY_WEIGHT,
    COVERAGE_TARGET,
    GRAPH_SUPPORT_WEIGHT,
    AnswerConfidence,
    ConfidenceConflict,
    ConfidenceJudgement,
    ConfidenceSignal,
    answer_confidence,
    answer_confidence_accuracy,
    evidence_confidence,
)
from core.rag import RetrievedPassage


def _passage(score: float, text: str = "text") -> RetrievedPassage:
    return RetrievedPassage(
        doc_id="d", title="t", text=text, source="c", score=score, signal="lexical"
    )


_Signal = ConfidenceSignal
_Conflict = ConfidenceConflict


# --- the evidence term ------------------------------------------------------


def test_no_passages_means_zero_evidence():
    assert evidence_confidence(()) == 0.0


def test_evidence_is_the_project_formula():
    passages = [_passage(0.8), _passage(0.4)]
    # 0.7 * top + 0.3 * (covered / target)
    expected = round(min(1.0, 0.7 * 0.8 + 0.3 * (2 / COVERAGE_TARGET)), 6)
    assert evidence_confidence(passages) == expected


def test_evidence_coverage_is_capped():
    many = [_passage(0.5) for _ in range(COVERAGE_TARGET + 5)]
    capped = [_passage(0.5) for _ in range(COVERAGE_TARGET)]
    assert evidence_confidence(many) == evidence_confidence(capped)


# --- the graph support term -------------------------------------------------


def test_no_evidence_means_no_confidence_even_with_support():
    # A graph fact cannot manufacture confidence in an answer the sources do not
    # support: with no retrieved evidence the confidence stays zero.
    report = answer_confidence((), causal_paths=[_Signal(1.0)])
    assert report.final == 0.0
    assert report.graph_support == 1.0


def test_support_is_the_strongest_signal_not_the_sum():
    report = answer_confidence(
        [_passage(0.5)],
        inferences=[_Signal(0.2)],
        connections=[_Signal(0.9)],
        causal_paths=[_Signal(0.3)],
    )
    assert report.graph_support == 0.9


def test_support_fills_only_a_bounded_share_of_the_room():
    passages = [_passage(0.5)]
    evidence = evidence_confidence(passages)
    report = answer_confidence(passages, causal_paths=[_Signal(1.0)])
    expected = evidence + (1.0 - evidence) * GRAPH_SUPPORT_WEIGHT
    assert abs(report.final - round(expected, 6)) < 1e-6


def test_support_is_modest_and_never_reaches_certainty():
    report = answer_confidence([_passage(0.5)], causal_paths=[_Signal(1.0)])
    assert report.final < 1.0


def test_a_signal_without_a_usable_confidence_is_skipped():
    class _NoConfidence:
        pass

    report = answer_confidence([_passage(0.5)], causal_paths=[_NoConfidence()])
    assert report.graph_support == 0.0
    assert report.final == evidence_confidence([_passage(0.5)])


# --- the conflict penalty ---------------------------------------------------


def test_a_contradiction_lowers_the_confidence():
    passages = [_passage(0.9)]
    plain = answer_confidence(passages)
    contradicted = answer_confidence(passages, conflicts=[_Conflict(0.5)])
    assert contradicted.final < plain.final
    assert contradicted.conflict_penalty > 0.0


def test_conflict_penalty_matches_the_weight():
    passages = [_passage(0.9)]
    evidence = evidence_confidence(passages)
    report = answer_confidence(passages, conflicts=[_Conflict(1.0)])
    expected = evidence * (1.0 - CONFLICT_PENALTY_WEIGHT)
    assert abs(report.final - round(expected, 6)) < 1e-6


def test_conflict_severity_is_summed_and_capped():
    # Two contradictions are worse than one, but the total is capped at 1.0.
    one = answer_confidence([_passage(0.9)], conflicts=[_Conflict(0.6)])
    two = answer_confidence([_passage(0.9)], conflicts=[_Conflict(0.6), _Conflict(0.6)])
    assert two.final <= one.final
    capped = answer_confidence(
        [_passage(0.9)], conflicts=[_Conflict(0.6), _Conflict(0.6), _Conflict(0.6)]
    )
    assert capped.final == two.final


def test_conflict_is_not_outvoted_by_support():
    # Even with the strongest possible graph support, a full contradiction must
    # leave the answer less confident than the bare evidence was.
    passages = [_passage(0.8)]
    evidence = evidence_confidence(passages)
    report = answer_confidence(
        passages, causal_paths=[_Signal(1.0)], conflicts=[_Conflict(1.0)]
    )
    assert report.final < evidence


def test_confidence_never_goes_negative():
    report = answer_confidence([_passage(0.01)], conflicts=[_Conflict(1.0)])
    assert report.final >= 0.0


# --- the report -------------------------------------------------------------


def test_report_is_bounded_and_validated():
    with pytest.raises(ValueError):
        AnswerConfidence(evidence=1.5, graph_support=0.0, conflict_penalty=0.0, final=0.5)
    with pytest.raises(ValueError):
        AnswerConfidence(evidence=0.5, graph_support=0.0, conflict_penalty=0.0, final=-0.1)


def test_report_describes_the_evidence_alone_when_nothing_else_moved():
    report = answer_confidence([_passage(0.5)])
    assert "retrieved evidence alone" in report.describe()


def test_report_names_the_support_when_it_moved():
    report = answer_confidence([_passage(0.5)], causal_paths=[_Signal(1.0)])
    assert "graph support" in report.describe()


def test_report_names_the_conflict_when_it_moved():
    report = answer_confidence([_passage(0.5)], conflicts=[_Conflict(0.8)])
    assert "conflict" in report.describe()


def test_a_negligible_support_term_is_not_named():
    # Support that rounds to 0.00 is noise, not a reason.
    report = answer_confidence([_passage(0.999)], causal_paths=[_Signal(0.001)])
    assert "graph support" not in report.describe()


def test_answer_confidence_is_deterministic():
    passages = [_passage(0.7)]
    first = answer_confidence(passages, causal_paths=[_Signal(0.6)])
    second = answer_confidence(passages, causal_paths=[_Signal(0.6)])
    assert first == second


# --- the answer path --------------------------------------------------------


def test_assistant_carries_a_confidence_report():
    from core.assistant import build_default_assistant

    assistant = build_default_assistant()
    answer = assistant.ask("why does waterflood increase production?")
    assert answer.confidence_report is not None
    assert answer.confidence_report.final == answer.confidence
    assert 0.0 <= answer.confidence <= 1.0


def test_assistant_confidence_gains_support_from_the_graph():
    from core.assistant import build_default_assistant

    assistant = build_default_assistant()
    answer = assistant.ask("why does waterflood increase production?")
    report = answer.confidence_report
    assert report is not None
    # The causal path corroborates the answer, so the final exceeds the bare evidence.
    assert report.graph_support > 0.0
    assert report.final >= report.evidence


def test_assistant_with_no_evidence_has_zero_confidence():
    from core.assistant import GroundedAssistant

    class _EmptyRetriever:
        ontology = None

        def retrieve(self, question, limit=4):
            return []

    assistant = GroundedAssistant(_EmptyRetriever())  # type: ignore[arg-type]
    answer = assistant.ask("zzz-not-a-concept-xyzzy")
    assert answer.confidence == 0.0
    assert answer.confidence_report is not None
    assert answer.confidence_report.final == 0.0


def test_jarvis_turn_carries_the_confidence_note(tmp_path, monkeypatch):
    monkeypatch.setenv("FAYD_DATA_DIR", str(tmp_path))
    from ai.jarvis import build_session

    mode = build_session(text_only=True)
    turn = mode.handle("why does waterflood increase production?", speak=False)
    assert turn.routed == "answer"
    assert turn.confidence > 0.0
    assert "evidence" in turn.confidence_note


def test_format_turn_renders_the_confidence_line():
    from ai.chat import format_turn
    from core.jarvis import JarvisTurn

    turn = JarvisTurn(
        heard="q",
        response="an answer",
        woke=False,
        routed="answer",
        confidence=0.83,
        confidence_note="0.83 (evidence 0.81 + graph support 0.02)",
    )
    rendered = format_turn(turn)
    assert "Confidence in this answer" in rendered
    assert "evidence 0.81" in rendered


def test_turn_payload_exposes_the_confidence():
    from ai.chat import turn_payload
    from core.jarvis import JarvisTurn

    turn = JarvisTurn(
        heard="q",
        response="a",
        woke=False,
        routed="answer",
        confidence=0.5,
        confidence_note="0.50 (evidence 0.50)",
    )
    payload = turn_payload(turn)
    assert payload["confidence"] == 0.5
    assert payload["confidence_note"] == "0.50 (evidence 0.50)"


# --- the gate ---------------------------------------------------------------


def test_confidence_gate_is_perfect_on_the_catalogue():
    assert answer_confidence_accuracy() == 1.0


def test_confidence_gate_empty_catalogue_is_zero():
    assert answer_confidence_accuracy(()) == 0.0


def test_confidence_gate_catches_manufactured_confidence():
    # A model that gave confidence with no evidence would fail this judgement.
    judgement = ConfidenceJudgement("no-evidence-zero", (), support=1.0, expect="no-evidence-zero")
    assert answer_confidence_accuracy((judgement,)) == 1.0
    report = answer_confidence((), causal_paths=[ConfidenceSignal(1.0)])
    assert report.final == 0.0


def test_confidence_judgement_validates_its_expectation():
    with pytest.raises(ValueError):
        ConfidenceJudgement("bad", (0.5,), expect="not-a-real-expectation")


def test_confidence_judgement_validates_scores():
    with pytest.raises(ValueError):
        ConfidenceJudgement("bad", (1.5,))


def test_confidence_signal_validates_range():
    with pytest.raises(ValueError):
        ConfidenceSignal(1.5)


def test_confidence_conflict_validates_range():
    with pytest.raises(ValueError):
        ConfidenceConflict(1.5)


def test_catalogue_carries_a_negative_control():
    expectations = {judgement.expect for judgement in CONFIDENCE_JUDGEMENTS}
    assert "no-evidence-zero" in expectations
    assert "conflict-lowers" in expectations
