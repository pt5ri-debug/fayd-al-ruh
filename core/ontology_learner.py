"""Learn new ontology relations from experience, not only from the seed.

The seeded ontology (``core.domain_knowledge``) is a small, inspectable starting
frame. This module lets the architecture extend it from what it actually
observes: when two known concepts are grounded together in one text, a weak
``associated_with`` relation is recorded between them.

Learned relations stay honest:

- They are only drawn between concepts already in the ontology, so no meaning is
  invented from raw text.
- They carry a modest confidence below the seeded relations, so experience can
  suggest links without overriding curated domain knowledge.
- A pair that already has an explicit relation in either direction is left
  untouched, so learning never duplicates or overwrites what is known.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import permutations

from core.ontology import Ontology
from core.semantic_relation import SemanticRelation


@dataclass(frozen=True)
class OntologyLearning:
    """What one observation taught the ontology."""

    concepts: tuple[str, ...]
    added: tuple[tuple[str, str, str], ...]
    skipped: int

    @property
    def changed(self) -> bool:
        return bool(self.added)


class OntologyLearner:
    """Extend the ontology with weak relations from co-occurring concepts."""

    def __init__(self, ontology: Ontology, confidence: float = 0.3, relation_type: str = "associated_with") -> None:
        if not 0.0 < confidence <= 1.0:
            raise ValueError("Learned relation confidence must be in (0, 1].")
        if not relation_type.strip():
            raise ValueError("Learned relation type cannot be empty.")
        self.ontology = ontology
        self.confidence = confidence
        self.relation_type = relation_type

    def concepts_in(self, text: str) -> list:
        """Return the known concepts whose terms occur in the text, in label order."""
        folded = text.casefold()
        found = [
            concept
            for concept in self.ontology.concepts.values()
            if any(term in folded for term in concept.terms)
        ]
        return sorted(found, key=lambda concept: concept.label)

    def observe(self, text: str) -> OntologyLearning:
        """Record weak relations between the known concepts grounded in ``text``."""
        if not text.strip():
            raise ValueError("Observation text cannot be empty.")
        concepts = self.concepts_in(text)
        added: list[tuple[str, str, str]] = []
        skipped = 0
        for source, target in permutations(concepts, 2):
            if self._related(source.id, target.id):
                skipped += 1
                continue
            self.ontology.add_relation(
                SemanticRelation(source.id, target.id, self.relation_type, self.confidence)
            )
            added.append((source.label, self.relation_type, target.label))
        return OntologyLearning(
            concepts=tuple(concept.label for concept in concepts),
            added=tuple(added),
            skipped=skipped,
        )

    def learn_from_observations(self, texts: list[str]) -> OntologyLearning:
        """Observe several texts and merge what they taught into one report."""
        concepts: list[str] = []
        added: list[tuple[str, str, str]] = []
        skipped = 0
        for text in texts:
            learning = self.observe(text)
            concepts.extend(learning.concepts)
            added.extend(learning.added)
            skipped += learning.skipped
        seen: set[str] = set()
        ordered_concepts: list[str] = []
        for label in concepts:
            if label not in seen:
                seen.add(label)
                ordered_concepts.append(label)
        return OntologyLearning(concepts=tuple(ordered_concepts), added=tuple(added), skipped=skipped)

    def _related(self, source_id: str, target_id: str) -> bool:
        return any(
            (relation.source_id, relation.target_id) in {(source_id, target_id), (target_id, source_id)}
            for relation in self.ontology.relations.values()
        )


__all__ = ["OntologyLearner", "OntologyLearning"]
