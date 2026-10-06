"""Tests for the petroleum-engineering curriculum seed."""

from core.cognitive_controller import CognitiveController
from core.curriculum_knowledge import (
    course_categories,
    course_keys,
    course_labels,
    courses,
    curriculum_coverage,
    curriculum_summary,
    seed_curriculum,
)
from core.domain_knowledge import seed_ontology
from core.ontology import Ontology


def test_course_keys_are_unique_and_labelled():
    keys = course_keys()
    assert len(keys) == len(set(keys))
    assert set(course_labels()) == set(keys)
    assert set(course_categories()) == set(keys)


def test_every_course_declares_a_category():
    for course in courses():
        assert course.category.strip()
        assert course.concepts
        assert course.relations


def test_curriculum_spans_foundations_geoscience_and_disciplines():
    categories = set(course_categories().values())
    assert "أساسيات" in categories
    assert "علوم_الأرض" in categories
    assert "هندسة_أساسية" in categories
    assert "تخصصات_البترول" in categories


def test_seed_curriculum_covers_every_course():
    ontology = Ontology()
    seed_curriculum(ontology)

    coverage = curriculum_coverage(ontology)
    assert set(coverage) == set(course_keys())
    for key, counts in coverage.items():
        assert counts["concepts"] == counts["expected_concepts"], key
        assert counts["relations"] == counts["expected_relations"], key


def test_seed_curriculum_is_idempotent():
    ontology = Ontology()
    seed_curriculum(ontology)
    concepts = len(ontology.concepts)
    relations = len(ontology.relations)

    seed_curriculum(ontology)
    assert len(ontology.concepts) == concepts
    assert len(ontology.relations) == relations


def test_curriculum_shares_nodes_with_the_domain_seed():
    ontology = Ontology()
    seed_ontology(ontology)
    domain_concepts = len(ontology.concepts)

    seed_curriculum(ontology)
    # The reservoir concept is shared, not duplicated.
    assert len(ontology.find_concepts("المكمن")) == 1
    assert len(ontology.concepts) > domain_concepts


def test_coverage_reports_zero_on_empty_ontology():
    ontology = Ontology()
    coverage = curriculum_coverage(ontology)
    assert all(counts["concepts"] == 0 for counts in coverage.values())
    assert all(counts["relations"] == 0 for counts in coverage.values())


def test_curriculum_summary_names_courses():
    summary = curriculum_summary()
    assert "curriculum" in summary.lower()
    assert "هندسة_الحفر" in summary


def test_controller_seeds_the_curriculum():
    controller = CognitiveController()
    coverage = curriculum_coverage(controller.ontology)
    assert all(
        counts["concepts"] == counts["expected_concepts"] for counts in coverage.values()
    )
    # A curriculum concept is grounded without any explicit request.
    assert controller.ontology.find_concepts("الصخر_المصدر")
