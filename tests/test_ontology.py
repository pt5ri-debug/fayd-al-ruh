from core.concept import Concept
from core.ontology import Ontology


def test_ontology_stores_and_finds_concepts():
    ontology = Ontology(); concept = Concept("car", frozenset({"automobile"})); ontology.add_concept(concept)
    assert ontology.find_concepts("automobile") == [concept]
