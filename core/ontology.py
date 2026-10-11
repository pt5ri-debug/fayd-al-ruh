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

    def add_concept_once(self, label: str, aliases: frozenset[str] = frozenset()) -> Concept:
        """Add a concept, regenerating its id on the rare random-id collision.

        Concept ids carry a truncated random suffix, so two seeded concepts can in
        principle draw the same id and abort seeding. Regenerate until the id is
        free, mirroring :meth:`add_relation_once`, so seeding never fails on a
        clash. The label is not de-duplicated here -- callers check first.
        """
        concept = Concept(label, aliases)
        while concept.id in self.concepts:
            concept = Concept(label, aliases)
        self.add_concept(concept)
        return concept

    def add_relation(self, relation: SemanticRelation) -> None:
        if relation.id in self.relations:
            raise ValueError(f"Semantic relation '{relation.id}' already exists.")
        if relation.source_id not in self.concepts or relation.target_id not in self.concepts:
            raise ValueError("Semantic relations must reference known concepts.")
        self.relations[relation.id] = relation

    def replace_relation(self, relation: SemanticRelation) -> None:
        """Overwrite an existing relation in place, e.g. to adjust its confidence."""
        if relation.id not in self.relations:
            raise KeyError(f"Unknown semantic relation: {relation.id}")
        if relation.source_id not in self.concepts or relation.target_id not in self.concepts:
            raise ValueError("Semantic relations must reference known concepts.")
        self.relations[relation.id] = relation

    def add_relation_once(
        self,
        source_id: str,
        target_id: str,
        relation_type: str,
        confidence: float = 0.5,
    ) -> None:
        """Add a relation unless an equivalent one exists, retrying id collisions."""
        for relation in self.relations.values():
            if (
                relation.source_id == source_id
                and relation.target_id == target_id
                and relation.relation_type == relation_type
            ):
                return
        relation = SemanticRelation(source_id, target_id, relation_type, confidence)
        # Relation ids carry a truncated random suffix; regenerate on the rare
        # collision so seeding never fails on an id clash.
        while relation.id in self.relations:
            relation = SemanticRelation(source_id, target_id, relation_type, confidence)
        self.add_relation(relation)

    def remove_relation(self, relation_id: str) -> None:
        if relation_id not in self.relations:
            raise KeyError(f"Unknown semantic relation: {relation_id}")
        del self.relations[relation_id]

    def find_concepts(self, term: str) -> list[Concept]:
        token = term.casefold().strip()
        return [concept for concept in self.concepts.values() if token in concept.terms]

    def outgoing(self, concept_id: str, relation_type: str | None = None) -> list[SemanticRelation]:
        return [relation for relation in self.relations.values() if relation.source_id == concept_id and (relation_type is None or relation.relation_type == relation_type)]
