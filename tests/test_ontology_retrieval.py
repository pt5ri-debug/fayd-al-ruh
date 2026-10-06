import pytest

from core.concept import Concept
from core.ontology import Ontology
from core.ontology_reasoner import OntologyReasoner
from core.ontology_retrieval import OntologyGuidedRetriever
from core.semantic_relation import SemanticRelation


def _mini_ontology() -> Ontology:
    ontology = Ontology()
    concepts = {
        label: Concept(label, aliases)
        for label, aliases in (
            ("log", frozenset({"well log"})),
            ("reservoir", frozenset()),
            ("formation", frozenset()),
            ("porosity", frozenset({"permeability"})),
        )
    }
    for concept in concepts.values():
        ontology.add_concept(concept)
    ontology.add_relation(SemanticRelation(concepts["log"].id, concepts["porosity"].id, "measures"))
    ontology.add_relation(SemanticRelation(concepts["formation"].id, concepts["porosity"].id, "has_property"))
    ontology.add_relation(SemanticRelation(concepts["reservoir"].id, concepts["formation"].id, "contained_in"))
    ontology.add_relation(SemanticRelation(concepts["log"].id, concepts["reservoir"].id, "measures"))
    return ontology


def test_concepts_in_matches_labels_and_aliases_case_insensitively():
    retriever = OntologyGuidedRetriever(_mini_ontology())

    assert retriever.concepts_in("The WELL LOG reads high.") == {_id_of(retriever.ontology, "log")}
    assert retriever.concepts_in("A core shows permeability.") == {_id_of(retriever.ontology, "porosity")}
    assert retriever.concepts_in("nothing relevant here") == set()


def test_reachable_follows_explicit_edges_with_hop_decay():
    # Rule-free reasoner so the walk reflects only the explicit edges.
    retriever = OntologyGuidedRetriever(
        _mini_ontology(), OntologyReasoner(rules=()), max_hops=2, hop_decay=0.5
    )
    log_id = _id_of(retriever.ontology, "log")
    reservoir_id = _id_of(retriever.ontology, "reservoir")
    formation_id = _id_of(retriever.ontology, "formation")

    reachable = retriever.reachable({log_id})

    assert reachable[log_id] == (0, 1.0)
    assert reachable[reservoir_id][0] == 1
    assert reachable[formation_id][0] == 2
    assert reachable[formation_id][1] == pytest.approx(0.25)


def test_rank_reaches_a_related_item_without_shared_words():
    retriever = OntologyGuidedRetriever(_mini_ontology())
    items = {
        "log": "The well log reads a high value.",
        "formation": "The formation is deep.",  # related to log through reservoir
        "casing": "Steel casing was installed.",  # unrelated
    }

    ranked = retriever.rank(items, "well log")

    assert ranked[0].item_id == "log"
    assert ranked[0].hops == 0
    ids = {match.item_id for match in ranked}
    assert "formation" in ids
    assert "casing" not in ids


def test_rank_uses_base_scores_and_respects_limit():
    retriever = OntologyGuidedRetriever(_mini_ontology())
    items = {"log": "well log", "casing": "steel casing"}

    ranked = retriever.rank(items, "well log", base_scores={"casing": 0.95}, limit=2)
    scores = {match.item_id: match.score for match in ranked}

    assert scores["casing"] == 0.95
    assert scores["log"] == 1.0
    assert len(retriever.rank(items, "well log", base_scores={"casing": 0.95}, limit=1)) == 1


def test_rank_rejects_empty_query_and_bad_limit():
    retriever = OntologyGuidedRetriever(_mini_ontology())
    with pytest.raises(ValueError):
        retriever.rank({"log": "well log"}, "   ")
    with pytest.raises(ValueError):
        retriever.rank({"log": "well log"}, "log", limit=0)


def test_constructor_validates_parameters():
    with pytest.raises(ValueError):
        OntologyGuidedRetriever(_mini_ontology(), max_hops=0)
    with pytest.raises(ValueError):
        OntologyGuidedRetriever(_mini_ontology(), hop_decay=0.0)
    with pytest.raises(ValueError):
        OntologyGuidedRetriever(_mini_ontology(), hop_decay=1.5)


def _id_of(ontology: Ontology, label: str) -> str:
    return ontology.find_concepts(label)[0].id


def test_controller_recall_follows_derived_relations():
    from core.cognitive_controller import CognitiveController
    from core.observation import Observation

    controller = CognitiveController()
    reservoir = "The reservoir holds hydrocarbons in the formation."
    unrelated = "The rig crew completed a routine safety briefing."
    controller.memory.store("obs-reservoir", "observation", Observation(reservoir, "study", 0.9), importance=0.9)
    controller.memory.store("obs-other", "observation", Observation(unrelated, "hse", 0.5), importance=0.5)

    # "log" shares no word with the reservoir item; it is reached via السجل measures المكمن.
    recalls = controller.recall_with_ontology("log interpretation", limit=5)

    texts = [str(getattr(recall.item.content, "content", recall.item.content)) for recall in recalls]
    assert texts[0] == reservoir
    assert unrelated not in texts
    assert all(recall.semantic_similarity > 0 for recall in recalls)


def test_controller_ontology_recall_returns_empty_for_unrelated_query():
    from core.cognitive_controller import CognitiveController
    from core.observation import Observation

    controller = CognitiveController()
    controller.memory.store("obs-other", "observation", Observation("A routine safety briefing.", "hse", 0.5))

    assert controller.recall_with_ontology("quarterly budget meeting") == []
