from __future__ import annotations

from core.concept import Concept
from core.semantic_relation import SemanticRelation


class Ontology:
    """A small concept graph, separate from instance/entity knowledge."""
    def __init__(self) -> None:
        self.concepts: dict[str, Concept] = {}
        self.relations: dict[str, SemanticRelation] = {}

    def add_concept(self, concept: Concept) -> None:
        if concept.id in self.concepts:
            raise ValueError(f"Concept '{concept.id}' already exists.")
        self.concepts[concept.id] = concept

    def add_relation(self, relation: SemanticRelation) -> None:
        if relation.id in self.relations:
            raise ValueError(f"Semantic relation '{relation.id}' already exists.")
        if relation.source_id not in self.concepts or relation.target_id not in self.concepts:
            raise ValueError("Semantic relations must reference known concepts.")
        self.relations[relation.id] = relation

    def find_concepts(self, term: str) -> list[Concept]:
        token = term.casefold().strip()
        return [concept for concept in self.concepts.values() if token in concept.terms]

    def outgoing(self, concept_id: str, relation_type: str | None = None) -> list[SemanticRelation]:
        return [relation for relation in self.relations.values() if relation.source_id == concept_id and (relation_type is None or relation.relation_type == relation_type)]
