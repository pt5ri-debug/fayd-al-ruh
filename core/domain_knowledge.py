"""Seed knowledge that gives the cognitive architecture its petroleum-engineering frame.

This is a small, inspectable starting ontology -- not a full engineering corpus.
It makes the project's declared identity concrete: the concepts the system can
ground, and the relations it can reason over, are about reservoirs, wells,
petrophysics, and production. Everything remains evidence-aware: seeded
relations carry a modest confidence and can be revised like any other belief.
"""

from __future__ import annotations

from core.concept import Concept
from core.ontology import Ontology
from core.self_model import SelfModel
from core.semantic_relation import SemanticRelation

_CONCEPTS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("المكمن", ("مكمن", "reservoir")),
    ("الطبقة", ("طبقة", "formation")),
    ("المسامية", ("porosity",)),
    ("النفاذية", ("permeability",)),
    ("الضغط", ("pressure",)),
    ("الهيدروكربون", ("hydrocarbon", "نفط", "oil", "gas")),
    ("البئر", ("well", "بئر")),
    ("الحفر", ("drilling", "حفر")),
    ("الاسمنت", ("cement", "اسمنت")),
    ("السجل", ("well log", "log", "سجل بئر")),
    ("الانتاج", ("production", "انتاج")),
    ("الماء", ("water", "ماء")),
    ("الصخر", ("rock", "صخر")),
)

_FACTS: tuple[tuple[str, str, str, float], ...] = (
    ("الماء", "المكمن", "exists_in", 0.7),
    ("الهيدروكربون", "المكمن", "exists_in", 0.8),
    ("المكمن", "الطبقة", "contained_in", 0.8),
    ("الطبقة", "المسامية", "has_property", 0.7),
    ("الطبقة", "النفاذية", "has_property", 0.7),
    ("المسامية", "النفاذية", "increases", 0.6),
    ("الضغط", "الانتاج", "drives", 0.7),
    ("البئر", "المكمن", "penetrates", 0.8),
    ("الحفر", "البئر", "creates", 0.8),
    ("الاسمنت", "البئر", "seals", 0.7),
    ("السجل", "المسامية", "measures", 0.6),
    ("السجل", "النفاذية", "measures", 0.6),
    ("الصخر", "المسامية", "contains", 0.6),
)


def seed_ontology(ontology: Ontology) -> list[Concept]:
    """Add the petroleum concepts and relations, reusing any that already exist.

    Idempotent: calling it twice leaves the ontology unchanged, so it is safe to
    run on every controller start and on restored graphs.
    """
    concepts: dict[str, Concept] = {}
    for label, aliases in _CONCEPTS:
        existing = ontology.find_concepts(label)
        concepts[label] = existing[0] if existing else _add_concept(ontology, label, aliases)

    for source, target, relation_type, confidence in _FACTS:
        _add_relation(
            ontology,
            concepts[source],
            concepts[target],
            relation_type,
            confidence,
        )
    return list(concepts.values())


def domain_summary(self_model: SelfModel | None = None) -> str:
    """One-line identity frame injected into model requests."""
    model = self_model or SelfModel()
    return f"{model.identity_statement} Focus areas: {', '.join(model.expertise)}."


def _add_concept(ontology: Ontology, label: str, aliases: tuple[str, ...]) -> Concept:
    concept = Concept(label, frozenset(aliases))
    ontology.add_concept(concept)
    return concept


def _add_relation(
    ontology: Ontology,
    source: Concept,
    target: Concept,
    relation_type: str,
    confidence: float,
) -> None:
    if any(
        relation.source_id == source.id
        and relation.target_id == target.id
        and relation.relation_type == relation_type
        for relation in ontology.relations.values()
    ):
        return
    ontology.add_relation(
        SemanticRelation(source.id, target.id, relation_type, confidence)
    )


__all__ = ["domain_summary", "seed_ontology"]
