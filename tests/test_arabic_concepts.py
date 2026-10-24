"""Arabic reaches the same concept node as its English label.

The project is Arabic-facing, but the ontology carried English-only aliases, so
almost none of the corpus's own Arabic glossary resolved. These tests pin the fix
in ``core.tokenize`` (the shared normalised space), ``core.concept`` (the
normalised terms), and ``core.ontology`` (the Arabic-aware whole-word matcher),
and the seeding that folds the corpus glossary into the concepts.
"""

from core.cognitive_controller import CognitiveController
from core.concept import Concept, is_arabic
from core.ontology import Ontology
from core.ontology_learner import OntologyLearner
from core.ontology_retrieval import OntologyGuidedRetriever
from core.petroleum_corpus import _ENTRIES
from core.tokenize import normalize_phrase


def _controller() -> CognitiveController:
    return CognitiveController()


def test_normalize_phrase_strips_the_article():
    assert normalize_phrase("المسامية") == normalize_phrase("مسامية")
    assert normalize_phrase("well log") == "well log"


def test_is_arabic_tells_the_scripts_apart():
    assert is_arabic("مسامية")
    assert not is_arabic("porosity")


def test_normalized_terms_put_a_term_and_a_text_in_one_space():
    concept = Concept("porosity", frozenset({"مسامية"}))
    assert normalize_phrase("المسامية") in concept.normalized_terms


def test_terms_for_splits_by_language():
    concept = Concept("porosity", frozenset({"مسامية"}))
    assert concept.terms_for("ar") == frozenset({"مسامية"})
    assert concept.terms_for("en") == frozenset({"porosity"})


def test_find_concepts_resolves_an_arabic_alias_with_the_article():
    # The corpus glossary seeds "مسامية" onto the porosity concept; the definite
    # article ("ال") must not stop the lookup.
    ontology = CognitiveController().ontology
    assert [concept.label for concept in ontology.find_concepts("المسامية")] == ["porosity"]


def test_the_corpus_glossary_reaches_its_concepts():
    ontology = CognitiveController().ontology
    glossary = {term: arabic for entry in _ENTRIES for term, arabic in entry.glossary}
    resolved = [arabic for arabic in glossary.values() if ontology.find_concepts(arabic)]
    # Most of the corpus's own glossary must now resolve; the few that do not name
    # a term the ontology has no node for ("oil in place", "gamma ray", ...).
    assert len(resolved) >= 25


def test_match_concepts_is_whole_word_and_arabic_aware():
    ontology = CognitiveController().ontology
    labels = [concept.label for _start, _end, concept in ontology.match_concepts("المسامية تزيد النفاذية.")]
    assert labels == ["porosity", "permeability"]


def test_match_concepts_does_not_match_inside_a_longer_word():
    ontology = CognitiveController().ontology
    assert ontology.match_concepts("microporosity is high") == []


def test_a_longer_term_wins_its_span():
    ontology = CognitiveController().ontology
    # "effective porosity" must win over the bare "porosity" it contains.
    labels = [concept.label for _start, _end, concept in ontology.match_concepts("effective porosity is high")]
    assert labels == ["effective_porosity"]


def test_an_inflected_token_still_matches():
    ontology = CognitiveController().ontology
    # Whole-word matching must not lose everyday inflection ("flowed" -> "flow").
    labels = {ontology.concepts[concept_id].label for concept_id in ontology.concepts_in("the well flowed")}
    assert "flow" in labels


def test_concepts_in_keeps_a_nested_concept():
    ontology = CognitiveController().ontology
    ids = ontology.concepts_in("log interpretation")
    labels = {ontology.concepts[concept_id].label for concept_id in ids}
    # Both the nested "well_log" ("log") and the longer node are named.
    assert "well_log" in labels


def test_the_learner_resolves_arabic():
    learner = OntologyLearner(CognitiveController().ontology)
    labels = [concept.label for concept in learner.concepts_in("المسامية تزيد النفاذية")]
    assert labels == ["permeability", "porosity"]


def test_the_retriever_resolves_arabic():
    controller = _controller()
    retriever = OntologyGuidedRetriever(controller.ontology)
    assert retriever.concepts_in("المسامية تزيد النفاذية") == controller.ontology.concepts_in(
        "المسامية تزيد النفاذية"
    )


def test_adding_an_alias_keeps_the_concept_id_and_relations():
    ontology = Ontology()
    from core.domain_knowledge import seed_ontology

    seed_ontology(ontology)
    concept = ontology.find_concepts("porosity")[0]
    before = {relation.id for relation in ontology.relations.values() if concept.id in (relation.source_id, relation.target_id)}
    updated = ontology.add_concept_alias(concept.id, "المسامية الفعالة")
    assert updated.id == concept.id
    after = {relation.id for relation in ontology.relations.values() if concept.id in (relation.source_id, relation.target_id)}
    assert before == after
    # Idempotent.
    assert ontology.add_concept_alias(concept.id, "المسامية الفعالة").aliases == updated.aliases


def test_adding_an_alias_to_an_unknown_concept_raises():
    import pytest

    with pytest.raises(KeyError):
        Ontology().add_concept_alias("CON-missing", "x")
