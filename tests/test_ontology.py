from core.concept import Concept
from core.ontology import Ontology


def test_ontology_stores_and_finds_concepts():
    ontology = Ontology()
    concept = Concept("car", frozenset({"automobile"}))
    ontology.add_concept(concept)
    assert ontology.find_concepts("automobile") == [concept]


def _seeded_ontology() -> Ontology:
    ontology = Ontology()
    for name in ("a", "b", "c"):
        ontology.add_concept(Concept(name))
    return ontology


def test_add_relation_once_is_idempotent():
    ontology = _seeded_ontology()
    source, target = ontology.find_concepts("a")[0], ontology.find_concepts("b")[0]
    ontology.add_relation_once(source.id, target.id, "relates", 0.4)
    ontology.add_relation_once(source.id, target.id, "relates", 0.9)
    assert len(ontology.relations) == 1


def test_add_relation_once_survives_id_collisions(monkeypatch):
    from core import ontology as ontology_module
    from core.semantic_relation import SemanticRelation as RealRelation

    ids = iter(["SEM-dup", "SEM-dup", "SEM-fresh"])
    seen: list[str] = []

    def factory(source_id, target_id, relation_type, confidence):
        relation = RealRelation(source_id, target_id, relation_type, confidence)
        object.__setattr__(relation, "id", next(ids))
        seen.append(relation.id)
        return relation

    monkeypatch.setattr(ontology_module, "SemanticRelation", factory)
    ontology = _seeded_ontology()
    a, b, c = (ontology.find_concepts(n)[0] for n in ("a", "b", "c"))
    ontology.add_relation_once(a.id, b.id, "relates")
    ontology.add_relation_once(a.id, c.id, "relates")
    assert len(ontology.relations) == 2
    assert seen == ["SEM-dup", "SEM-dup", "SEM-fresh"]


def test_add_concept_once_survives_id_collisions(monkeypatch):
    from core import ontology as ontology_module
    from core.concept import Concept as RealConcept

    ids = iter(["CON-dup", "CON-dup", "CON-fresh"])

    def factory(label, aliases=frozenset()):
        concept = RealConcept(label, aliases)
        object.__setattr__(concept, "id", next(ids))
        return concept

    monkeypatch.setattr(ontology_module, "Concept", factory)
    ontology = Ontology()
    first = ontology.add_concept_once("alpha")
    second = ontology.add_concept_once("beta")
    assert {first.id, second.id} == {"CON-dup", "CON-fresh"}
    assert len(ontology.concepts) == 2
