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


@dataclass(frozen=True)
class OntologyFeedback:
    """How a recall's outcome reshaped the learned link that produced it."""

    relation_id: str
    source: str
    target: str
    outcome: str
    confidence: float
    removed: bool = False

    @property
    def changed(self) -> bool:
        return True


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
        # Which relations this learner created. The type alone used to answer
        # "is this learned?" (everything of ``relation_type`` was), but a typed
        # relation read from a sentence (see :mod:`core.relation_extraction`)
        # carries a real relation type, so membership is the honest test.
        self._learned: set[str] = set()

    def concepts_in(self, text: str) -> list[Concept]:
        """Return the known concepts whose terms occur in the text, in label order.

        The match is whole-word and Arabic-normalised (through
        :meth:`Ontology.concepts_in`), so ``المسامية`` reaches ``porosity`` and
        ``porosity`` does not match inside ``microporosity`` -- a bare
        ``casefold`` substring test did neither.
        """
        ids = self.ontology.concepts_in(text)
        found = [self.ontology.concepts[concept_id] for concept_id in ids]
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
            relation = SemanticRelation(source.id, target.id, self.relation_type, self.confidence)
            self.ontology.add_relation(relation)
            self._learned.add(relation.id)
            added.append((source.label, self.relation_type, target.label))
        return OntologyLearning(
            concepts=tuple(concept.label for concept in concepts),
            added=tuple(added),
            skipped=skipped,
            reinforced=tuple(reinforced),
        )

    def observe_relation(
        self,
        source_label: str,
        target_label: str,
        relation_type: str,
        confidence: float,
    ) -> OntologyLearning:
        """Record one *typed* relation between two known concepts.

        This is the entry point for :mod:`core.relation_extraction`: the caller
        has already decided the endpoints and the relation type from the
        sentence's own grammar. The learner's job is only to keep the graph
        honest -- both endpoints must already be concepts, a pair that already
        has an explicit (non-learned) relation is left alone, and the confidence
        is clamped to the learner's ceiling, so a sentence can suggest a link but
        can never override curated knowledge. An identical typed relation that
        already exists is reinforced rather than duplicated.

        Raises ``KeyError`` when either endpoint is not a known concept, so a
        caller that reads a raw noun cannot silently invent a node.
        """
        if not relation_type.strip():
            raise ValueError("A typed relation needs a relation type.")
        sources = self.ontology.find_concepts(source_label)
        targets = self.ontology.find_concepts(target_label)
        if not sources or not targets:
            raise KeyError(f"Unknown concept: {source_label} / {target_label}")
        source = sources[0]
        target = targets[0]
        if source.id == target.id:
            return OntologyLearning(concepts=(source.label,), added=(), skipped=1)
        if self._has_explicit(source.id, target.id):
            return OntologyLearning(concepts=(source.label, target.label), added=(), skipped=1)
        existing = self.relation_by_triple(source.id, relation_type, target.id)
        if existing is not None:
            if existing.id not in self._learned:
                # A seeded relation with this exact triple: never duplicate it.
                return OntologyLearning(
                    concepts=(source.label, target.label), added=(), skipped=1
                )
            new_confidence = self._strengthen(existing)
            return OntologyLearning(
                concepts=(source.label, target.label),
                added=(),
                skipped=0,
                reinforced=((source.label, target.label, new_confidence),),
            )
        bounded = round(min(self.ceiling, max(0.0, confidence)), 6)
        relation = SemanticRelation(source.id, target.id, relation_type, bounded)
        self.ontology.add_relation(relation)
        self._learned.add(relation.id)
        return OntologyLearning(
            concepts=(source.label, target.label),
            added=((source.label, relation_type, target.label),),
            skipped=0,
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
            if relation.id not in self._learned:
                continue
            source = self._label(relation.source_id)
            target = self._label(relation.target_id)
            new_confidence = round(relation.confidence - step, 6)
            if new_confidence <= 0.0:
                self.ontology.remove_relation(relation.id)
                self._learned.discard(relation.id)
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

    def strengthen_relation(self, relation_id: str, amount: float | None = None) -> float:
        """Reinforce a learned relation by id after a successful use.

        Raises ``KeyError`` for an unknown relation or a seeded (non-learned) one.
        """
        relation = self.ontology.relations.get(relation_id)
        if relation is None:
            raise KeyError(f"Unknown semantic relation: {relation_id}")
        if relation.id not in self._learned:
            raise KeyError(f"Relation {relation_id} is not a learned relation.")
        return self._strengthen(relation, amount)

    def weaken_relation(self, relation_id: str, amount: float | None = None) -> bool:
        """Weaken a learned relation by id; drop it when it fades to zero.

        Returns ``True`` if the relation was removed, ``False`` if it was merely
        weakened. Raises ``KeyError`` for an unknown or seeded relation.
        """
        relation = self.ontology.relations.get(relation_id)
        if relation is None:
            raise KeyError(f"Unknown semantic relation: {relation_id}")
        if relation.id not in self._learned:
            raise KeyError(f"Relation {relation_id} is not a learned relation.")
        step = self.reinforcement if amount is None else amount
        if step <= 0.0:
            raise ValueError("Weakening amount must be positive.")
        new_confidence = round(relation.confidence - step, 6)
        if new_confidence <= 0.0:
            self.ontology.remove_relation(relation.id)
            self._learned.discard(relation.id)
            return True
        self.ontology.replace_relation(replace(relation, confidence=new_confidence))
        return False

    def relation(self, relation_id: str) -> SemanticRelation | None:
        """Return a relation by id, or ``None`` if it does not exist."""
        return self.ontology.relations.get(relation_id)

    def is_learned(self, relation_id: str) -> bool:
        """Whether this learner created the relation (so it may be reshaped).

        A seeded relation is never learned, so this is the honest test a caller
        should use before reinforcing or decaying a link.
        """
        return relation_id in self._learned

    def relation_by_triple(
        self, source_id: str, relation_type: str, target_id: str
    ) -> SemanticRelation | None:
        """Return the relation matching an exact (source, type, target) triple."""
        for relation in self.ontology.relations.values():
            if (
                relation.source_id == source_id
                and relation.relation_type == relation_type
                and relation.target_id == target_id
            ):
                return relation
        return None

    def apply_feedback(
        self, relation_id: str, *, useful: bool, amount: float | None = None
    ) -> OntologyFeedback:
        """Credit or blame a learned relation for the outcome of a recall.

        A useful recall reinforces the link that produced it; an unhelpful one
        weakens it, and the link is dropped if it fades to nothing. Only learned
        relations can be reshaped this way.
        """
        relation = self.ontology.relations.get(relation_id)
        if relation is None:
            raise KeyError(f"Unknown semantic relation: {relation_id}")
        source = self._label(relation.source_id)
        target = self._label(relation.target_id)
        if useful:
            confidence = self.strengthen_relation(relation_id, amount)
            return OntologyFeedback(relation_id, source, target, "reinforced", confidence)
        removed = self.weaken_relation(relation_id, amount)
        confidence = 0.0 if removed else self.ontology.relations[relation_id].confidence
        return OntologyFeedback(
            relation_id, source, target, "removed" if removed else "weakened", confidence, removed
        )

    def _has_explicit(self, source_id: str, target_id: str) -> bool:
        """Whether a *seeded* relation already connects the pair, in either direction.

        A relation this learner created is not explicit, so it never blocks the
        pair: an untyped association may still be learned beside a typed one, and
        a typed relation can be reinforced by a later sentence.
        """
        pair = {(source_id, target_id), (target_id, source_id)}
        return any(
            (relation.source_id, relation.target_id) in pair
            and relation.relation_type != self.relation_type
            and relation.id not in self._learned
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


__all__ = ["OntologyDecay", "OntologyFeedback", "OntologyLearner", "OntologyLearning"]
