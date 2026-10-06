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


def test_observe_reinforces_repeated_pairs_instead_of_duplicating():
    ontology = _mini_ontology()
    learner = OntologyLearner(ontology)
    first = learner.observe("pressure permeability formation")
    second = learner.observe("pressure permeability formation")

    assert first.added
    assert second.added == ()
    assert second.reinforced
    # formation<->permeability is seeded and still skipped.
    assert second.skipped == 1
    # Each repeated pair moved up one reinforcement step from the initial 0.3.
    assert all(confidence == pytest.approx(0.35) for _s, _t, confidence in second.reinforced)


def test_reinforcement_is_capped_below_the_seeded_relations():
    ontology = _mini_ontology()
    learner = OntologyLearner(ontology, confidence=0.3, ceiling=0.5, reinforcement=0.1)
    for _ in range(10):
        learner.observe("pressure permeability formation")

    learned = [
        relation.confidence
        for relation in ontology.relations.values()
        if relation.relation_type == "associated_with"
    ]
    assert learned
    assert max(learned) == pytest.approx(0.5)


def test_decay_weakens_then_removes_learned_relations():
    ontology = _mini_ontology()
    learner = OntologyLearner(ontology, confidence=0.3, reinforcement=0.1)
    learner.observe("pressure permeability formation")
    learned_before = {
        relation.id for relation in ontology.relations.values() if relation.relation_type == "associated_with"
    }
    assert learned_before

    first = learner.decay()
    assert first.decayed
    assert not first.removed
    assert all(confidence == pytest.approx(0.2) for _s, _t, confidence in first.decayed)

    learner.decay()
    second = learner.decay()
    assert second.removed
    assert all(
        relation.relation_type != "associated_with"
        for relation in ontology.relations.values()
    )


def test_decay_leaves_seeded_relations_untouched():
    ontology = _mini_ontology()
    learner = OntologyLearner(ontology)
    seeded_before = {
        relation.id: relation.confidence
        for relation in ontology.relations.values()
        if relation.relation_type != "associated_with"
    }
    learner.observe("pressure permeability")
    learner.decay()
    learner.decay()
    learner.decay()

    seeded_after = {
        relation.id: relation.confidence
        for relation in ontology.relations.values()
        if relation.relation_type != "associated_with"
    }
    assert seeded_after == seeded_before


def test_reinforce_by_label_requires_a_learned_relation():
    ontology = _mini_ontology()
    learner = OntologyLearner(ontology)
    learner.observe("pressure permeability")

    assert learner.reinforce("pressure", "permeability") == pytest.approx(0.35)
    with pytest.raises(KeyError):
        learner.reinforce("pressure", "reservoir")
    with pytest.raises(KeyError):
        learner.reinforce("unknown", "permeability")


def test_reinforcement_and_decay_amounts_are_validated():
    learner = OntologyLearner(_mini_ontology())
    learner.observe("pressure permeability")
    with pytest.raises(ValueError):
        learner.reinforce("pressure", "permeability", amount=0.0)
    with pytest.raises(ValueError):
        learner.decay(amount=0.0)


def test_ceiling_and_reinforcement_are_validated():
    with pytest.raises(ValueError):
        OntologyLearner(_mini_ontology(), confidence=0.3, ceiling=0.2)
    with pytest.raises(ValueError):
        OntologyLearner(_mini_ontology(), reinforcement=0.0)


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


def test_controller_decay_removes_the_recall_path_it_learned():
    from core.cognitive_controller import CognitiveController
    from core.observation import Observation

    controller = CognitiveController()
    controller.memory.store(
        "obs-permeability",
        "observation",
        Observation("permeability controls flow.", "study", 0.9),
        importance=0.9,
    )
    controller.learn_ontology_from_observation("الضغط يرافقه النفاذية")
    assert controller.recall_with_ontology("pressure")

    # Three decays of 0.1 exhaust the 0.3 learned link and drop it.
    controller.decay_ontology_learning(amount=0.1)
    controller.decay_ontology_learning(amount=0.1)
    decay = controller.decay_ontology_learning(amount=0.1)

    assert decay.removed
    assert controller.recall_with_ontology("pressure") == []


def test_controller_reinforce_raises_learned_confidence():
    from core.cognitive_controller import CognitiveController

    controller = CognitiveController()
    controller.learn_ontology_from_observation("الضغط يرافقه النفاذية")
    before = controller.ontology_learner.confidence

    after = controller.reinforce_ontology_relation("الضغط", "النفاذية")

    assert after == pytest.approx(before + controller.ontology_learner.reinforcement)


def test_strengthen_and_weaken_relation_by_id():
    ontology = _mini_ontology()
    learner = OntologyLearner(ontology)
    learning = learner.observe("pressure permeability")
    source, relation_type, target = learning.added[0]
    relation = learner.relation_by_triple(
        ontology.find_concepts(source)[0].id, relation_type, ontology.find_concepts(target)[0].id
    )
    assert relation is not None

    strengthened = learner.strengthen_relation(relation.id, amount=0.1)
    assert strengthened == pytest.approx(0.4)

    removed = learner.weaken_relation(relation.id, amount=0.4)
    assert removed is True
    assert learner.relation(relation.id) is None


def test_strengthen_relation_rejects_seeded_relations():
    ontology = _mini_ontology()
    learner = OntologyLearner(ontology)
    seeded = next(iter(ontology.relations.values()))

    with pytest.raises(KeyError):
        learner.strengthen_relation(seeded.id)
    with pytest.raises(KeyError):
        learner.weaken_relation("SEM-missing")


def test_apply_feedback_reinforces_or_weakens_a_learned_link():
    ontology = _mini_ontology()
    learner = OntologyLearner(ontology)
    learning = learner.observe("pressure permeability")
    source, relation_type, target = learning.added[0]
    relation = learner.relation_by_triple(
        ontology.find_concepts(source)[0].id, relation_type, ontology.find_concepts(target)[0].id
    )
    assert relation is not None

    good = learner.apply_feedback(relation.id, useful=True, amount=0.1)
    assert good.outcome == "reinforced"
    assert good.confidence == pytest.approx(0.4)
    assert not good.removed

    bad = learner.apply_feedback(relation.id, useful=False, amount=0.5)
    assert bad.outcome == "removed"
    assert bad.removed
    assert learner.relation(relation.id) is None
