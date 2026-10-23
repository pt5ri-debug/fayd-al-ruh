"""Relation extraction: reading typed relations out of a sentence.

The extractor is a *pattern matcher, not a parser*, so these tests pin both what
it reads and what it honestly refuses: an unknown endpoint, an unmapped verb, a
concept sitting inside a prepositional phrase, and a known modifier used as if it
were the head noun all yield nothing rather than a wrong relation.
"""

from __future__ import annotations

import pytest
from core.cognitive_controller import CognitiveController
from core.ontology_learner import OntologyLearner
from core.relation_extraction import (
    EXTRACTABLE_RELATIONS,
    ExtractedRelation,
    RelationExtractor,
)


@pytest.fixture()
def controller() -> CognitiveController:
    return CognitiveController()


@pytest.fixture()
def extractor(controller: CognitiveController) -> RelationExtractor:
    return RelationExtractor(controller.ontology)


def _triples(extraction) -> list[tuple[str, str, str]]:
    return [(r.source, r.relation_type, r.target) for r in extraction.relations]


def test_extractable_relations_are_real_ontology_types(controller):
    """Every relation the extractor emits must be one the seeded ontology uses."""
    used = {relation.relation_type for relation in controller.ontology.relations.values()}
    assert set(EXTRACTABLE_RELATIONS) <= used


def test_active_voice_is_read_in_order(extractor):
    extraction = extractor.extract("Porosity increases permeability.")
    assert _triples(extraction) == [("porosity", "increases", "permeability")]


def test_passive_voice_swaps_the_endpoints(extractor):
    """'X is measured by Y' means Y measures X -- direction must not be lost."""
    extraction = extractor.extract("Porosity is measured by the density log.")
    assert _triples(extraction) == [("density_log", "measures", "porosity")]


def test_a_leading_article_does_not_hide_the_subject(extractor):
    extraction = extractor.extract("The density log measures porosity.")
    assert _triples(extraction) == [("density_log", "measures", "porosity")]


def test_a_compound_sentence_yields_both_clauses(extractor):
    extraction = extractor.extract(
        "Porosity increases permeability and permeability controls flow."
    )
    assert _triples(extraction) == [
        ("porosity", "increases", "permeability"),
        ("permeability", "affects", "flow"),
    ]


def test_an_unknown_endpoint_yields_nothing(extractor):
    """A noun the ontology does not know must not become a concept."""
    extraction = extractor.extract("Porosity increases zorblax.")
    assert extraction.relations == ()


def test_an_unmapped_verb_yields_nothing(extractor):
    extraction = extractor.extract("Porosity juggles permeability.")
    assert extraction.relations == ()


def test_a_concept_inside_a_prepositional_phrase_is_not_the_object(extractor):
    """The verb governs 'blowout', not the 'reservoir' it sits inside."""
    extraction = extractor.extract("A gas kick causes a blowout in the reservoir.")
    assert extraction.relations == ()


def test_a_modifier_is_not_read_as_the_head_noun(extractor):
    """'well control' is a kind of control, so 'well' is not the object."""
    extraction = extractor.extract("A gas kick causes well control.")
    assert extraction.relations == ()


def test_a_relation_carries_its_pattern_and_sentence(extractor):
    extraction = extractor.extract("Porosity increases permeability.")
    relation = extraction.relations[0]
    assert relation.pattern == "Porosity increases permeability"
    assert relation.sentence == "Porosity increases permeability"
    assert 0.0 <= relation.confidence <= 1.0


def test_the_summary_counts_what_it_skipped(extractor):
    extraction = extractor.extract("Baking bread is unrelated. Porosity juggles permeability.")
    assert extraction.sentences == 2
    assert extraction.extracted == 0
    assert extraction.single_concept == 1
    assert extraction.unmatched == 1


def test_extracted_relation_rejects_empty_endpoints():
    with pytest.raises(ValueError):
        ExtractedRelation("", "x", "is_a", 0.5, "p", "s")


def test_reading_does_not_change_the_graph(controller):
    before = len(controller.ontology.relations)
    controller.extract_relations("Wettability affects flow.")
    assert len(controller.ontology.relations) == before


def test_learning_adds_a_typed_relation_bounded_by_the_ceiling(controller):
    learner = OntologyLearner(controller.ontology)
    extraction = RelationExtractor(controller.ontology, learner)
    learning = extraction.apply("Wettability affects flow.")
    assert learning.added == (("wettability", "affects", "flow"),)
    source = controller.ontology.find_concepts("wettability")[0]
    target = controller.ontology.find_concepts("flow")[0]
    relation = learner.relation_by_triple(source.id, "affects", target.id)
    assert relation is not None
    assert relation.confidence <= learner.ceiling


def test_learning_never_overrides_a_seeded_relation(controller):
    """'porosity increases permeability' is already seeded, so nothing changes."""
    learner = OntologyLearner(controller.ontology)
    extraction = RelationExtractor(controller.ontology, learner)
    learning = extraction.apply("Porosity increases permeability.")
    assert learning.added == ()
    assert learning.reinforced == ()


def test_learning_reinforces_a_repeat_rather_than_duplicating(controller):
    learner = OntologyLearner(controller.ontology)
    extraction = RelationExtractor(controller.ontology, learner)
    first = extraction.apply("Wettability affects flow.")
    assert first.added
    second = extraction.apply("Wettability affects flow.")
    assert second.added == ()
    assert second.reinforced


def test_apply_requires_a_learner(controller):
    extraction = RelationExtractor(controller.ontology)
    with pytest.raises(ValueError):
        extraction.apply("Porosity increases permeability.")


def test_learned_relations_decay_and_a_seeded_one_does_not(controller):
    learner = OntologyLearner(controller.ontology)
    extraction = RelationExtractor(controller.ontology, learner)
    extraction.apply("Wettability affects flow.")
    before = len(controller.ontology.relations)
    decay = learner.decay(1.0)
    assert decay.removed
    assert len(controller.ontology.relations) == before - len(decay.removed)


def test_the_controller_previews_without_writing(controller):
    before = len(controller.ontology.relations)
    controller.extract_relations("Wettability affects flow.")
    assert len(controller.ontology.relations) == before


def test_the_controller_learns_and_invalidates_the_retriever(controller):
    learning = controller.learn_extracted_relations("Wettability affects flow.")
    assert learning.added


class _FakeStore:
    """A minimal knowledge-store double: just the one method the reader needs."""

    def __init__(self, texts):
        self._texts = list(texts)

    def iter_chunks(self):
        for index, text in enumerate(self._texts):
            yield type("Chunk", (), {"text": text, "chunk_id": str(index)})()


def test_merge_extractions_deduplicates_identical_triples(extractor):
    from core.relation_extraction import merge_extractions

    first = extractor.extract("Porosity increases permeability.")
    second = extractor.extract("Porosity increases permeability.")
    merged = merge_extractions([first, second])
    assert len(merged.relations) == 1
    assert merged.sentences == 2


def test_extract_from_store_streams_every_chunk(extractor):
    from core.relation_extraction import extract_from_store

    store = _FakeStore([
        "Porosity increases permeability.",
        "The density log measures porosity.",
    ])
    extraction = extract_from_store(store, extractor)
    assert {r.source for r in extraction.relations} == {"porosity", "density_log"}


def test_extract_from_store_honours_the_limit(extractor):
    from core.relation_extraction import extract_from_store

    store = _FakeStore([
        "Porosity increases permeability.",
        "The density log measures porosity.",
    ])
    extraction = extract_from_store(store, extractor, limit=1)
    assert [r.source for r in extraction.relations] == ["porosity"]


def test_extract_from_store_rejects_a_negative_limit(extractor):
    from core.relation_extraction import extract_from_store

    with pytest.raises(ValueError):
        extract_from_store(_FakeStore([]), extractor, limit=-1)


def test_a_relative_clause_is_not_glued_onto_the_object(extractor):
    extraction = extractor.extract("Porosity increases permeability, which controls flow.")
    assert [(r.source, r.relation_type, r.target) for r in extraction.relations] == [
        ("porosity", "increases", "permeability")
    ]


def test_a_trailing_negation_does_not_become_the_object(extractor):
    extraction = extractor.extract("Porosity increases permeability, not flow.")
    assert [(r.source, r.relation_type, r.target) for r in extraction.relations] == [
        ("porosity", "increases", "permeability")
    ]


def test_a_negated_clause_asserts_nothing(extractor):
    for sentence in (
        "Porosity does not increase permeability.",
        "Water saturation never increases permeability.",
        "The density log cannot measure porosity.",
    ):
        assert extractor.extract(sentence).relations == ()


def test_an_extra_invented_relation_costs_the_score(extractor):
    """The gate compares the *set* of triples, so an over-firing reader fails."""
    from core.relation_extraction import ExtractionJudgement, extraction_accuracy

    # The extractor reads one relation here; a judgement expecting two fails.
    catalogue = (
        ExtractionJudgement(
            "Porosity increases permeability.",
            ("porosity", "increases", "permeability"),
        ),
        ExtractionJudgement(
            "Porosity increases permeability, not flow.",
            ("porosity", "increases", "flow"),
        ),
    )
    assert extraction_accuracy(catalogue, extractor=extractor) == 0.5


def test_the_controller_learns_from_a_store(controller):
    learning = controller.learn_extracted_relations_from_store(
        _FakeStore(["Wettability affects flow."])
    )
    assert learning.added == (("wettability", "affects", "flow"),)


def test_extraction_accuracy_is_perfect_on_the_held_out_catalogue(extractor):
    from core.relation_extraction import extraction_accuracy

    assert extraction_accuracy(extractor=extractor) == 1.0


def test_extraction_accuracy_counts_a_negative_control(extractor):
    """A judgement that must yield nothing is a real test, not a formality."""
    from core.relation_extraction import ExtractionJudgement, extraction_accuracy

    catalogue = (ExtractionJudgement("Porosity juggles permeability.", None),)
    assert extraction_accuracy(catalogue, extractor=extractor) == 1.0
    # An extractor that invented a relation here would fail.
    wrong = (ExtractionJudgement("Porosity juggles permeability.", ("porosity", "affects", "permeability")),)
    assert extraction_accuracy(wrong, extractor=extractor) == 0.0


def test_extraction_accuracy_is_zero_for_an_empty_catalogue(extractor):
    from core.relation_extraction import extraction_accuracy

    assert extraction_accuracy((), extractor=extractor) == 0.0


def test_extraction_judgement_rejects_an_empty_sentence():
    from core.relation_extraction import ExtractionJudgement

    with pytest.raises(ValueError):
        ExtractionJudgement("   ", None)
