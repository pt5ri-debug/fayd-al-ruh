"""Learn new ontology relations from experience, and let experience reshape them.

The seeded ontology (``core.domain_knowledge``) is a small, inspectable starting
frame. This module lets the architecture extend it from what it actually
observes: when two known concepts are grounded together in one text, a weak
``associated_with`` relation is recorded between them.

Experience does not just add links, it also weights them:

- Repetition strengthens a learned relation step by step, so links that keep
  showing up in experience become more trusted.
- Disuse decays a learned relation, and a link that fades to nothing is dropped,
  so spurious associations do not persist forever.

Learned relations stay honest:

- They are only drawn between concepts already in the ontology, so no meaning is
  invented from raw text.
- Their confidence is bounded by ``ceiling`` (below the seeded relations), so
  experience can suggest links without ever overriding curated domain knowledge.
- A pair that already has an explicit relation in either direction is left
  untouched, so learning never duplicates or overwrites what is known.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from itertools import combinations

from core.concept import Concept
from core.ontology import Ontology
from core.semantic_relation import SemanticRelation


@dataclass(frozen=True)
class OntologyLearning:
    """What one observation taught the ontology."""

    concepts: tuple[str, ...]
    added: tuple[tuple[str, str, str], ...]
    skipped: int
    reinforced: tuple[tuple[str, str, float], ...] = ()

    @property
    def changed(self) -> bool:
        return bool(self.added or self.reinforced)


@dataclass(frozen=True)
class OntologyDecay:
    """What disuse took away from the ontology."""

    decayed: tuple[tuple[str, str, float], ...]
    removed: tuple[tuple[str, str], ...]

    @property
    def changed(self) -> bool:
        return bool(self.decayed or self.removed)


class OntologyLearner:
    """Extend the ontology from co-occurring concepts and weight links by use."""

    def __init__(
        self,
        ontology: Ontology,
        confidence: float = 0.3,
        relation_type: str = "associated_with",
        *,
        ceiling: float = 0.6,
        reinforcement: float = 0.05,
    ) -> None:
        if not 0.0 < confidence <= 1.0:
            raise ValueError("Learned relation confidence must be in (0, 1].")
        if not relation_type.strip():
            raise ValueError("Learned relation type cannot be empty.")
        if not confidence <= ceiling <= 1.0:
            raise ValueError("Learned ceiling must be in [confidence, 1].")
        if not 0.0 < reinforcement <= 1.0:
            raise ValueError("Reinforcement step must be in (0, 1].")
        self.ontology = ontology
        self.confidence = confidence
        self.relation_type = relation_type
        self.ceiling = ceiling
        self.reinforcement = reinforcement

    def concepts_in(self, text: str) -> list[Concept]:
        """Return the known concepts whose terms occur in the text, in label order."""
        folded = text.casefold()
        found = [
            concept
            for concept in self.ontology.concepts.values()
            if any(term in folded for term in concept.terms)
        ]
        return sorted(found, key=lambda concept: concept.label)

    def observe(self, text: str) -> OntologyLearning:
        """Add new weak relations from ``text`` and strengthen any it repeats."""
        if not text.strip():
            raise ValueError("Observation text cannot be empty.")
        concepts = self.concepts_in(text)
        added: list[tuple[str, str, str]] = []
        reinforced: list[tuple[str, str, float]] = []
        skipped = 0
        for source, target in combinations(concepts, 2):
            if self._has_explicit(source.id, target.id):
                skipped += 1
                continue
            existing = self._learned_relation(source.id, target.id)
            if existing is not None:
                reinforced.append((source.label, target.label, self._strengthen(existing)))
                continue
            self.ontology.add_relation(
                SemanticRelation(source.id, target.id, self.relation_type, self.confidence)
            )
            added.append((source.label, self.relation_type, target.label))
        return OntologyLearning(
            concepts=tuple(concept.label for concept in concepts),
            added=tuple(added),
            skipped=skipped,
            reinforced=tuple(reinforced),
        )

    def learn_from_observations(self, texts: list[str]) -> OntologyLearning:
        """Observe several texts and merge what they taught into one report."""
        concepts: list[str] = []
        added: list[tuple[str, str, str]] = []
        reinforced: list[tuple[str, str, float]] = []
        skipped = 0
        for text in texts:
            learning = self.observe(text)
            concepts.extend(learning.concepts)
            added.extend(learning.added)
            reinforced.extend(learning.reinforced)
            skipped += learning.skipped
        seen: set[str] = set()
        ordered_concepts: list[str] = []
        for label in concepts:
            if label not in seen:
                seen.add(label)
                ordered_concepts.append(label)
        return OntologyLearning(
            concepts=tuple(ordered_concepts),
            added=tuple(added),
            skipped=skipped,
            reinforced=tuple(reinforced),
        )

    def reinforce(self, source_label: str, target_label: str, amount: float | None = None) -> float:
        """Strengthen the learned relation between two concepts, up to the ceiling.

        Raises ``KeyError`` if the concepts are unknown or no learned relation
        connects them (a seeded relation is never a target for reinforcement).
        """
        sources = self.ontology.find_concepts(source_label)
        targets = self.ontology.find_concepts(target_label)
        if not sources or not targets:
            raise KeyError(f"Unknown concept: {source_label} / {target_label}")
        relation = self._learned_relation(sources[0].id, targets[0].id)
        if relation is None:
            raise KeyError(f"No learned relation between {source_label} and {target_label}")
        return self._strengthen(relation, amount)

    def decay(self, amount: float | None = None) -> OntologyDecay:
        """Weaken every learned relation by ``amount``; drop any that reaches zero."""
        step = self.reinforcement if amount is None else amount
        if step <= 0.0:
            raise ValueError("Decay amount must be positive.")
        decayed: list[tuple[str, str, float]] = []
        removed: list[tuple[str, str]] = []
        for relation in list(self.ontology.relations.values()):
            if relation.relation_type != self.relation_type:
                continue
            source = self._label(relation.source_id)
            target = self._label(relation.target_id)
            new_confidence = round(relation.confidence - step, 6)
            if new_confidence <= 0.0:
                self.ontology.remove_relation(relation.id)
                removed.append((source, target))
            else:
                self.ontology.replace_relation(replace(relation, confidence=new_confidence))
                decayed.append((source, target, new_confidence))
        return OntologyDecay(decayed=tuple(decayed), removed=tuple(removed))

    def _strengthen(self, relation: SemanticRelation, amount: float | None = None) -> float:
        step = self.reinforcement if amount is None else amount
        if step <= 0.0:
            raise ValueError("Reinforcement amount must be positive.")
        new_confidence = round(min(self.ceiling, relation.confidence + step), 6)
        self.ontology.replace_relation(replace(relation, confidence=new_confidence))
        return new_confidence

    def _has_explicit(self, source_id: str, target_id: str) -> bool:
        pair = {(source_id, target_id), (target_id, source_id)}
        return any(
            (relation.source_id, relation.target_id) in pair and relation.relation_type != self.relation_type
            for relation in self.ontology.relations.values()
        )

    def _learned_relation(self, source_id: str, target_id: str) -> SemanticRelation | None:
        pair = {(source_id, target_id), (target_id, source_id)}
        for relation in self.ontology.relations.values():
            if relation.relation_type == self.relation_type and (relation.source_id, relation.target_id) in pair:
                return relation
        return None

    def _label(self, concept_id: str) -> str:
        concept = self.ontology.concepts.get(concept_id)
        return concept.label if concept is not None else concept_id


__all__ = ["OntologyLearner", "OntologyLearning", "OntologyDecay"]
