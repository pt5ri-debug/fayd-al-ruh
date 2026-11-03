"""The explainable confidence evaluation and the multi-hop confidence fold."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest
from core.answer_confidence import (
    AnswerConfidence,
    ConfidenceConflict,
    answer_confidence,
    evidence_confidence,
)
from core.confidence_engine import (
    CONFIDENCE_NOTE,
    MAX_INDEPENDENT_SOURCES,
    ConfidenceEvaluation,
    evaluate_confidence,
)
from core.evidence import Evidence


def _evidence(source="SPE", supports=True, strength=1.0) -> Evidence:
    return Evidence(description="note", source=source, supports=supports, strength=strength)


# -- the evaluation -------------------------------------------------------


def test_evaluate_confidence_is_explainable():
    evaluation = evaluate_confidence([_evidence("SPE"), _evidence("OnePetro")], prior=0.5)
    assert 0.0 <= evaluation.confidence <= 1.0
    assert evaluation.term("prior") is not None
    assert evaluation.term("support") is not None
    assert evaluation.term("posterior") is not None
    assert evaluation.evidence_count == 2
    assert evaluation.source_count == 2


def test_a_confidence_must_be_in_range():
    with pytest.raises(ValueError):
        ConfidenceEvaluation(confidence=1.5)


def test_an_out_of_range_prior_is_refused():
    with pytest.raises(ValueError):
        evaluate_confidence([], prior=2.0)


def test_independent_sources_are_counted_by_source_not_by_item():
    """Three snippets from one page are one witness, not three."""
    evaluation = evaluate_confidence([_evidence("SPE"), _evidence("SPE"), _evidence("SPE")])
    assert evaluation.evidence_count == 3
    assert evaluation.source_count == 1


def test_diversity_saturates():
    items = [_evidence(f"source-{index}") for index in range(MAX_INDEPENDENT_SOURCES + 5)]
    evaluation = evaluate_confidence(items)
    assert evaluation.term("source_diversity").value == pytest.approx(1.0)


def test_contradicting_evidence_is_weighed_separately():
    evaluation = evaluate_confidence([_evidence("SPE", supports=True), _evidence("blog", supports=False)])
    assert evaluation.support_weight > 0
    assert evaluation.contradict_weight > 0


def test_the_note_says_confidence_is_not_proof():
    evaluation = evaluate_confidence([_evidence()])
    assert evaluation.note == CONFIDENCE_NOTE
    assert "not proof of truth" in evaluation.note


def test_verification_status_is_reported_not_set():
    evaluation = evaluate_confidence([_evidence()], verified="provisional")
    assert evaluation.verified == "provisional"


def test_recency_discounts_an_old_item():
    now = datetime.now(UTC)
    old = _evidence()
    old.created_at = now - timedelta(days=30)
    fresh = evaluate_confidence([_evidence()], now=now)
    aged = evaluate_confidence([old], now=now, recency_half_life_days=10)
    assert aged.support_weight < fresh.support_weight


def test_recency_is_ignored_when_no_half_life_is_given():
    now = datetime.now(UTC)
    old = _evidence()
    old.created_at = now - timedelta(days=1000)
    fresh = evaluate_confidence([_evidence()], now=now)
    aged = evaluate_confidence([old], now=now)
    # With no half-life the age makes no difference: the two weights are equal.
    assert aged.support_weight == pytest.approx(fresh.support_weight)


def test_describe_and_primitive_are_readable():
    evaluation = evaluate_confidence([_evidence("SPE")])
    assert "confidence:" in evaluation.describe()
    payload = evaluation.to_primitive()
    assert payload["evidence_count"] == 1


# -- the answer confidence fold -------------------------------------------


class _Passage:
    def __init__(self, score: float) -> None:
        self.score = score


class _Walked:
    def __init__(self, confidence: float, hops: int) -> None:
        self.confidence = confidence
        self.hops = hops


class _Signal:
    def __init__(self, confidence: float) -> None:
        self.confidence = confidence


def test_no_evidence_pins_confidence_at_zero():
    assert answer_confidence(()).final == 0.0
    assert answer_confidence((), connections=[_Signal(1.0)]).final == 0.0


def test_graph_support_raises_the_confidence():
    plain = answer_confidence([_Passage(0.6)])
    supported = answer_confidence([_Passage(0.6)], connections=[_Signal(1.0)])
    assert supported.final > plain.final


def test_a_single_edge_carries_no_path_uncertainty():
    report = answer_confidence([_Passage(0.6)], connections=[_Walked(0.9, 1)])
    assert report.path_uncertainty == 0.0


def test_a_multi_hop_chain_is_reported_as_uncertain():
    report = answer_confidence([_Passage(0.6)], connections=[_Walked(0.49, 2)])
    assert report.path_uncertainty > 0.0
    assert any("walked path uncertainty" in reason for reason in report.reasons)


def test_a_multi_hop_chain_corroborates_less_than_a_single_edge():
    single = answer_confidence([_Passage(0.6)], connections=[_Walked(0.9, 1)])
    walked = answer_confidence([_Passage(0.6)], connections=[_Walked(0.49, 2)])
    assert walked.final < single.final


def test_a_contradiction_lowers_the_confidence():
    plain = answer_confidence([_Passage(0.8)])
    contradicted = answer_confidence([_Passage(0.8)], conflicts=[ConfidenceConflict(0.6)])
    assert contradicted.final < plain.final
    assert contradicted.conflict_penalty > 0


def test_the_confidence_is_always_bounded():
    for score in (0.0, 0.5, 1.0):
        report = answer_confidence([_Passage(score)], conflicts=[ConfidenceConflict(1.0)])
        assert 0.0 <= report.final <= 1.0


def test_the_report_validates_its_own_range():
    with pytest.raises(ValueError):
        AnswerConfidence(evidence=0.5, graph_support=0.0, conflict_penalty=0.0, final=1.5)
    with pytest.raises(ValueError):
        AnswerConfidence(evidence=0.5, graph_support=0.0, conflict_penalty=0.0, final=0.5, path_uncertainty=2.0)


def test_evidence_confidence_is_zero_without_passages():
    assert evidence_confidence(()) == 0.0


def test_describe_names_the_terms_that_moved():
    report = answer_confidence([_Passage(0.6)], connections=[_Signal(0.8)])
    assert "evidence" in report.describe()
