"""Bounded, transparent inference over the concept ontology.

The seeded petroleum ontology (``core.domain_knowledge``) stores explicit
relations. This module derives relations that follow from those without being
stored, using a small set of declarative composition rules, so the ontology is
reasoned over rather than left as static data.

Derived relations are never asserted into the ontology: they are returned as
inferences with their composing steps and a propagated confidence, and stay
distinguishable from the explicit, evidence-carrying relations.
"""

from __future__ import annotations

from dataclasses import dataclass

from core.ontology import Ontology
from core.semantic_relation import SemanticRelation


@dataclass(frozen=True)
class RelationRule:
    """A transparent composition of two relations into a third.

    ``chain`` composes ``A -first-> B -second-> C`` into ``A -conclusion-> C``.
    ``shared_target`` composes ``A -first-> P <-second- B`` into
    ``A -conclusion-> B`` (the shared node is the target of both relations).
    """

    name: str
    kind: str
    first: str
    second: str
    conclusion: str
    confidence_factor: float = 1.0

    def __post_init__(self) -> None:
        if self.kind not in ("chain", "shared_target"):
            raise ValueError("Rule kind must be 'chain' or 'shared_target'.")
        if not 0.0 <= self.confidence_factor <= 1.0:
            raise ValueError("Rule confidence factor must be between 0 and 1.")


# Domain-sensible composition rules over the seeded petroleum relations.
RULES: tuple[RelationRule, ...] = (
    # A well that penetrates a reservoir also penetrates the formation holding it.
    RelationRule("penetrates_through_containment", "chain", "penetrates", "contained_in", "penetrates"),
    # Fluids held in a reservoir are also held in the reservoir's formation.
    RelationRule("exists_in_through_containment", "chain", "exists_in", "contained_in", "exists_in"),
    # A reservoir inherits the properties of the formation that contains it.
    RelationRule("containment_inherits_property", "chain", "contained_in", "has_property", "has_property"),
    # Containment is transitive.
    RelationRule("containment_transitive", "chain", "contained_in", "contained_in", "contained_in"),
    # A tool measuring a property also measures the host that has that property.
    RelationRule("measure_property_of_host", "shared_target", "measures", "has_property", "measures"),
)


@dataclass(frozen=True)
class OntologyInference:
    """A relation the ontology entails, with the steps that entail it."""

    source_id: str
    target_id: str
    relation_type: str
    confidence: float
    derived: bool
    steps: tuple[SemanticRelation, ...] = ()

    def __post_init__(self) -> None:
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("Inference confidence must be between 0 and 1.")


class OntologyReasoner:
    """Derives relations from the ontology using the declarative rules.

    The closure is bounded (``max_iterations``) so it always terminates, and
    derived relations are kept separate from the explicit ones.
    """

    def __init__(self, rules: tuple[RelationRule, ...] = RULES) -> None:
        self.rules = rules

    def infer(
        self,
        ontology: Ontology,
        source_id: str,
        target_id: str,
        relation_type: str,
    ) -> OntologyInference | None:
        """Return an explicit or derived relation between two concepts, or None."""
        for relation in ontology.relations.values():
            if (
                relation.source_id == source_id
                and relation.target_id == target_id
                and relation.relation_type == relation_type
            ):
                return OntologyInference(
                    source_id, target_id, relation_type, relation.confidence, derived=False, steps=(relation,)
                )
        key = (source_id, target_id, relation_type)
        derived = self._closure(ontology)
        return derived.get(key)

    def derive(self, ontology: Ontology, max_iterations: int = 4) -> tuple[OntologyInference, ...]:
        """Return every relation the ontology entails but does not store."""
        if max_iterations < 1:
            raise ValueError("Maximum iterations must be at least 1.")
        return tuple(self._closure(ontology, max_iterations).values())

    def _closure(self, ontology: Ontology, max_iterations: int = 4) -> dict[tuple[str, str, str], OntologyInference]:
        # Explicit relations are authoritative and never overwritten by derivations.
        explicit_keys = {
            (relation.source_id, relation.target_id, relation.relation_type)
            for relation in ontology.relations.values()
        }
        best: dict[tuple[str, str, str], OntologyInference] = {
            (relation.source_id, relation.target_id, relation.relation_type): OntologyInference(
                relation.source_id,
                relation.target_id,
                relation.relation_type,
                relation.confidence,
                derived=False,
                steps=(relation,),
            )
            for relation in ontology.relations.values()
        }
        derived: dict[tuple[str, str, str], OntologyInference] = {}

        for _ in range(max_iterations):
            current = list(best.values())
            changed = False
            for rule in self.rules:
                for inference in self._apply(rule, current):
                    key = (inference.source_id, inference.target_id, inference.relation_type)
                    if key in explicit_keys:
                        continue
                    existing = best.get(key)
                    if existing is not None and existing.confidence >= inference.confidence:
                        continue
                    best[key] = inference
                    derived[key] = inference
                    changed = True
            if not changed:
                break

        return derived

    def _apply(self, rule: RelationRule, relations: list[OntologyInference]) -> list[OntologyInference]:
        results: list[OntologyInference] = []
        for first in relations:
            if first.relation_type != rule.first:
                continue
            for second in relations:
                if second.relation_type != rule.second:
                    continue
                if rule.kind == "chain":
                    if first.target_id != second.source_id or first.source_id == second.target_id:
                        continue
                    source_id, target_id = first.source_id, second.target_id
                else:  # shared_target
                    if first.target_id != second.target_id or first.source_id == second.source_id:
                        continue
                    source_id, target_id = first.source_id, second.source_id
                results.append(
                    OntologyInference(
                        source_id,
                        target_id,
                        rule.conclusion,
                        round(first.confidence * second.confidence * rule.confidence_factor, 6),
                        derived=True,
                        steps=(*first.steps, *second.steps),
                    )
                )
        return results


__all__ = ["OntologyInference", "OntologyReasoner", "RelationRule", "RULES"]
