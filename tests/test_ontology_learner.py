import pytest

from core.concept import Concept
from core.ontology import Ontology
from core.ontology_learner import OntologyLearner
from core.semantic_relation import SemanticRelation


def _mini_ontology() -> Ontology:
    ontology = Ontology()
    concepts = {
        label: Concept(label, aliases)
        for label, aliases in (
            ("pressure", frozenset({"الضغط"})),
            ("permeability", frozenset({"النفاذية"})),
            ("formation", frozenset({"الطبقة"})),
            ("reservoir", frozenset({"المكمن"})),
        )
    }
    for concept in concepts.values():
        ontology.add_concept(concept)
    # formation already has an explicit relation to permeability
    ontology.add_relation(SemanticRelation(concepts["formation"].id, concepts["permeability"].id, "has_property"))
    return ontology


def _pairs(added: tuple[tuple[str, str, str], ...]) -> set[frozenset[str]]:
    return {frozenset((source, target)) for source, _relation, target in added}


def test_observe_learns_weak_links_between_co_occurring_concepts():
    ontology = _mini_ontology()
    learner = OntologyLearner(ontology)
    learning = learner.observe("pressure rises with permeability in the formation")

    assert learning.changed
    assert set(learning.concepts) == {"formation", "permeability", "pressure"}
    assert frozenset({"pressure", "permeability"}) in _pairs(learning.added)
    assert frozenset({"pressure", "formation"}) in _pairs(learning.added)
    assert learner.confidence == 0.3


def test_observe_skips_pairs_that_already_have_an_explicit_relation():
    ontology = _mini_ontology()
    learner = OntologyLearner(ontology)
    before = len(ontology.relations)
    learner.observe("permeability and formation")

    # formation->permeability is seeded; the reverse must not be added either.
    assert len(ontology.relations) == before
    assert all(
        relation.relation_type != "associated_with"
        for relation in ontology.relations.values()
    )


def test_observe_is_idempotent():
    ontology = _mini_ontology()
    learner = OntologyLearner(ontology)
    first = learner.observe("pressure permeability formation")
    second = learner.observe("pressure permeability formation")

    assert first.added
    assert second.added == ()
    assert second.skipped > 0


def test_observe_rejects_empty_text():
    learner = OntologyLearner(_mini_ontology())
    with pytest.raises(ValueError):
        learner.observe("   ")


def test_confidence_and_relation_type_are_validated():
    with pytest.raises(ValueError):
        OntologyLearner(_mini_ontology(), confidence=0.0)
    with pytest.raises(ValueError):
        OntologyLearner(_mini_ontology(), confidence=1.5)
    with pytest.raises(ValueError):
        OntologyLearner(_mini_ontology(), relation_type="  ")


def test_learned_confidence_stays_below_seeded_relations():
    ontology = _mini_ontology()
    learner = OntologyLearner(ontology, confidence=0.3)
    learner.observe("pressure permeability formation")

    seeded = max(
        relation.confidence
        for relation in ontology.relations.values()
        if relation.relation_type != "associated_with"
    )
    learned = [
        relation.confidence
        for relation in ontology.relations.values()
        if relation.relation_type == "associated_with"
    ]
    assert learned
    assert max(learned) < seeded


def test_learn_from_observations_merges_reports():
    ontology = _mini_ontology()
    learner = OntologyLearner(ontology)
    learning = learner.learn_from_observations(
        ["pressure and permeability", "reservoir and formation"]
    )

    assert frozenset({"pressure", "permeability"}) in _pairs(learning.added)
    assert frozenset({"reservoir", "formation"}) in _pairs(learning.added)
    assert set(learning.concepts) == {"formation", "permeability", "pressure", "reservoir"}


def test_concepts_in_ignores_unknown_terms():
    ontology = _mini_ontology()
    learner = OntologyLearner(ontology)
    assert learner.concepts_in("nothing relevant here") == []


def test_controller_learning_extends_recall_through_the_new_relation():
    from core.cognitive_controller import CognitiveController
    from core.observation import Observation

    controller = CognitiveController()
    controller.memory.store(
        "obs-permeability",
        "observation",
        Observation("permeability controls flow.", "study", 0.9),
        importance=0.9,
    )

    # Before learning, "pressure" has no path to the permeability item.
    assert controller.recall_with_ontology("pressure") == []

    learning = controller.learn_ontology_from_observation("الضغط يرافقه النفاذية")
    assert learning.changed

    recalls = controller.recall_with_ontology("pressure")
    texts = [str(getattr(recall.item.content, "content", recall.item.content)) for recall in recalls]
    assert texts and texts[0] == "permeability controls flow."


def test_controller_learning_leaves_derived_relations_separate():
    from core.cognitive_controller import CognitiveController

    controller = CognitiveController()
    stored_before = len(controller.ontology.relations)
    derived_before = len(controller.derive_ontology_relations())

    controller.learn_ontology_from_observation("الضغط يرافقه النفاذية")

    # The learned relation is stored; the derived set is unchanged and still derived.
    assert len(controller.ontology.relations) > stored_before
    assert all(inference.derived for inference in controller.derive_ontology_relations())
    assert len(controller.derive_ontology_relations()) == derived_before
