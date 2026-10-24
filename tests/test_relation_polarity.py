"""Tests for the relation-polarity model and its use in consistency checking."""

from __future__ import annotations

from core.belief import Belief
from core.cognitive_controller import CognitiveController
from core.consistency_checker import ConsistencyChecker
from core.contradiction_detector import ContradictionDetector
from core.evidence import Evidence
from core.relation_polarity import (
    OPPOSING_PAIRS,
    conflicts,
    find_polarity_conflicts,
    is_symmetric,
    opposes,
    polarity_accuracy,
    polarity_of,
)
from core.semantic_relation import SemanticRelation


def test_polarity_of_declares_the_antonymous_types():
    assert polarity_of("increases") == "positive"
    assert polarity_of("reduces") == "negative"
    assert polarity_of("INCREASES") == "positive"
    assert polarity_of("measures") is None
    assert polarity_of("unknown_relation") is None


def test_opposes_only_for_a_declared_antonymous_pair():
    assert opposes("increases", "reduces")
    assert opposes("reduces", "increases")
    assert opposes("causes", "prevents")
    assert not opposes("increases", "measures")
    assert not opposes("increases", "increases")
    assert not opposes("reduces", "reduces")


def test_a_symmetric_relation_is_never_on_a_polarity_axis():
    assert is_symmetric("associated_with")
    assert is_symmetric("related_to")
    assert not is_symmetric("increases")
    assert not opposes("associated_with", "associated_with")


def test_conflicts_is_none_for_a_non_antonymous_pair():
    assert conflicts("A", "B", "increases", "measures") is None


def test_conflicts_severity_is_the_weaker_confidence():
    conflict = conflicts("A", "B", "increases", "reduces", confidence_a=0.8, confidence_b=0.5)
    assert conflict is not None
    assert conflict.severity == 0.5
    assert conflict.pair == ("A", "B")


def test_find_polarity_conflicts_needs_the_same_direction():
    forward = [
        SemanticRelation("A", "B", "increases", 0.8),
        SemanticRelation("A", "B", "reduces", 0.6),
    ]
    assert len(find_polarity_conflicts(forward)) == 1
    reversed_pair = [
        SemanticRelation("A", "B", "increases", 0.8),
        SemanticRelation("B", "A", "reduces", 0.6),
    ]
    assert find_polarity_conflicts(reversed_pair) == []


def test_find_polarity_conflicts_is_deterministic():
    relations = [
        SemanticRelation("B", "A", "increases", 0.5),
        SemanticRelation("A", "B", "reduces", 0.5),
        SemanticRelation("A", "B", "increases", 0.5),
    ]
    first = find_polarity_conflicts(relations)
    second = find_polarity_conflicts(list(reversed(relations)))
    assert first == second
    assert len(first) == 1
    assert first[0].pair == ("A", "B")


def test_opposing_pairs_are_all_two_element_antonym_sets():
    assert OPPOSING_PAIRS
    for pair in OPPOSING_PAIRS:
        assert len(pair) == 2


def test_polarity_accuracy_is_perfect_on_the_held_out_catalogue():
    assert polarity_accuracy() == 1.0


def test_polarity_accuracy_is_zero_for_an_empty_catalogue():
    assert polarity_accuracy(()) == 0.0


def test_consistency_checker_reads_opposing_relation_metadata():
    increasing = Belief(
        "waterflood increases recovery",
        0.9,
        metadata={"relation_type": "increases", "source_id": "waterflood", "target_id": "recovery_factor"},
    )
    reducing = Belief(
        "waterflood reduces recovery",
        0.7,
        metadata={"relation_type": "reduces", "source_id": "waterflood", "target_id": "recovery_factor"},
    )
    report = ConsistencyChecker().check([increasing, reducing])
    assert report.consistent is False
    assert report.conflicts[0].severity == 0.7


def test_consistency_checker_ignores_unrelated_relation_types():
    measures = Belief("log measures porosity", 0.9, metadata={"relation_type": "measures", "source_id": "log", "target_id": "porosity"})
    increases = Belief("log increases porosity", 0.9, metadata={"relation_type": "increases", "source_id": "log", "target_id": "porosity"})
    assert ConsistencyChecker().check([measures, increases]).consistent is True


def test_consistency_checker_ignores_opposite_directions():
    forward = Belief("A increases B", 0.9, metadata={"relation_type": "increases", "source_id": "A", "target_id": "B"})
    backward = Belief("B reduces A", 0.9, metadata={"relation_type": "reduces", "source_id": "B", "target_id": "A"})
    assert ConsistencyChecker().check([forward, backward]).consistent is True


def test_consistency_checker_still_catches_statement_negation():
    report = ConsistencyChecker().check([Belief("rain occurred", .8), Belief("not rain occurred", .6)])
    assert report.consistent is False
    assert report.conflicts[0].severity == .6


def test_contradiction_detector_keeps_the_support_flag_rule_without_metadata():
    evidence = [Evidence("rain", "a", True), Evidence("no rain", "b", False)]
    assert len(ContradictionDetector().detect(evidence)) == 1


def test_contradiction_detector_finds_antonymous_relations_even_when_both_support():
    first = Evidence("waterflood increases recovery", "a", True, data={"relation_type": "increases", "source_id": "wf", "target_id": "rf"})
    second = Evidence("waterflood reduces recovery", "b", True, data={"relation_type": "reduces", "source_id": "wf", "target_id": "rf"})
    assert len(ContradictionDetector().detect([first, second])) == 1


def test_contradiction_detector_does_not_call_same_relation_a_contradiction():
    first = Evidence("log measures porosity", "a", True, data={"relation_type": "measures", "source_id": "log", "target_id": "por"})
    second = Evidence("log measures porosity", "b", False, data={"relation_type": "measures", "source_id": "log", "target_id": "por"})
    assert ContradictionDetector().detect([first, second]) == []


def test_controller_finds_opposing_typed_relations_in_the_graph():
    controller = CognitiveController()
    controller.learn_extracted_relations("Waterflood increases recovery factor.")
    controller.learn_extracted_relations("Waterflood reduces recovery factor.")
    conflicts_found = controller.find_relation_conflicts()
    assert len(conflicts_found) == 1
    assert {conflicts_found[0].first_relation, conflicts_found[0].second_relation} == {"increases", "reduces"}
    # The scan is read-only: the graph still holds both learned relations.
    assert controller.find_relation_conflicts() == conflicts_found
