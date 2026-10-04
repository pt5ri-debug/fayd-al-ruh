from core.concept import Concept
from core.language_reasoner import LanguageReasoner
from core.ontology import Ontology
from core.semantic_relation import SemanticRelation


def test_reasoner_infers_transitive_is_a():
    ontology = Ontology(); cat, mammal, animal = Concept("cat"), Concept("mammal"), Concept("animal")
    for concept in (cat, mammal, animal): ontology.add_concept(concept)
    ontology.add_relation(SemanticRelation(cat.id, mammal.id, "is_a")); ontology.add_relation(SemanticRelation(mammal.id, animal.id, "is_a"))
    assert LanguageReasoner().is_a(ontology, cat.id, animal.id)
