"""Seed knowledge that gives the cognitive architecture its petroleum-engineering frame.

This is a small, inspectable starting ontology -- not a full engineering corpus.
It makes the project's declared identity concrete: the concepts the system can
ground, and the relations it can reason over, are about reservoirs, wells,
petrophysics, and production. Everything remains evidence-aware: seeded
relations carry a modest confidence and can be revised like any other belief.

The seed is organized by petroleum-engineering discipline so the frame spans the
whole field -- reservoir, drilling, production, petrophysics, well logging,
completions, enhanced recovery, well testing, artificial lift, stimulation,
cementing, well control, economics, HSE, plus the newer frontiers of facilities
engineering, process safety, and unconventional resources. Each discipline
contributes its own concepts and relations; ``discipline_coverage`` reports how
much of that frame an ontology currently carries.
"""

from __future__ import annotations

from dataclasses import dataclass

from core.concept import Concept
from core.ontology import Ontology
from core.self_model import SelfModel

# The foundation layer: the concepts every discipline builds on. Kept separate so
# the original seed stays recognizable and stable.
_CONCEPTS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("reservoir", ("مكمن", "reservoir")),
    ("formation", ("طبقة", "formation")),
    ("porosity", ("porosity",)),
    ("permeability", ("permeability",)),
    ("pressure", ("pressure",)),
    ("hydrocarbon", ("hydrocarbon", "نفط", "oil", "gas")),
    ("well", ("well", "بئر")),
    ("drilling", ("drilling", "حفر")),
    ("cement", ("cement", "اسمنت")),
    ("well_log", ("well log", "log", "سجل بئر")),
    ("production", ("production", "انتاج")),
    ("water", ("water", "ماء")),
    ("rock", ("rock", "صخر")),
)

_FACTS: tuple[tuple[str, str, str, float], ...] = (
    ("water", "reservoir", "exists_in", 0.7),
    ("hydrocarbon", "reservoir", "exists_in", 0.8),
    ("reservoir", "formation", "contained_in", 0.8),
    ("formation", "porosity", "has_property", 0.7),
    ("formation", "permeability", "has_property", 0.7),
    ("porosity", "permeability", "increases", 0.6),
    ("pressure", "production", "drives", 0.7),
    ("well", "reservoir", "penetrates", 0.8),
    ("drilling", "well", "creates", 0.8),
    ("cement", "well", "seals", 0.7),
    ("well_log", "porosity", "measures", 0.6),
    ("well_log", "permeability", "measures", 0.6),
    ("rock", "porosity", "contains", 0.6),
)


@dataclass(frozen=True)
class DisciplineKnowledge:
    """The concepts and relations one petroleum-engineering discipline contributes."""

    key: str
    label: str
    concepts: tuple[tuple[str, tuple[str, ...]], ...]
    relations: tuple[tuple[str, str, str, float], ...]

    def __post_init__(self) -> None:
        if not self.key.strip() or not self.label.strip():
            raise ValueError("Discipline key and label cannot be empty.")
        if not self.concepts:
            raise ValueError(f"Discipline '{self.key}' needs at least one concept.")
        if not self.relations:
            raise ValueError(f"Discipline '{self.key}' needs at least one relation.")


_DISCIPLINES: tuple[DisciplineKnowledge, ...] = (
    DisciplineKnowledge(
        key="reservoir",
        label="Reservoir Engineering",
        concepts=(
            ("reserve", ("reserve", "احتياطي")),
            ("recovery_factor", ("recovery factor", "recovery")),
            ("waterflood", ("waterflood", "water flooding")),
            ("reservoir_pressure", ("reservoir pressure",)),
            ("reservoir_simulation", ("reservoir simulation", "simulation")),
        ),
        relations=(
            ("reservoir", "reserve", "contains", 0.7),
            ("reserve", "recovery_factor", "estimated_by", 0.6),
            ("waterflood", "reservoir_pressure", "maintains", 0.7),
            ("reservoir_pressure", "production", "drives", 0.7),
            ("reservoir_simulation", "reserve", "predicts", 0.6),
        ),
    ),
    DisciplineKnowledge(
        key="drilling",
        label="Drilling Engineering",
        concepts=(
            ("drilling_fluid", ("drilling fluid", "mud", "طين الحفر")),
            ("directional_well", ("directional well", "directional drilling")),
            ("drill_bit", ("drill bit", "bit")),
            ("gas_kick", ("gas kick", "kick", "اندفاع")),
            ("casing", ("casing", "casing string", "بطانة")),
        ),
        relations=(
            ("drilling_fluid", "drill_bit", "cools", 0.6),
            ("drilling_fluid", "gas_kick", "controls", 0.7),
            ("directional_well", "well", "creates", 0.6),
            ("drill_bit", "well", "drills", 0.7),
            ("casing", "well", "supports", 0.7),
        ),
    ),
    DisciplineKnowledge(
        key="production",
        label="Production Engineering",
        concepts=(
            ("production_rate", ("production rate", "flow rate")),
            ("production_decline", ("production decline", "decline")),
            ("Artificial Lift", ("artificial lift",)),
            ("well_stimulation", ("well stimulation", "stimulation", "acidizing")),
            ("pump", ("pump",)),
        ),
        relations=(
            ("Artificial Lift", "production_rate", "increases", 0.7),
            ("production_decline", "well_stimulation", "treated_by", 0.6),
            ("well_stimulation", "production_rate", "increases", 0.7),
            ("pump", "Artificial Lift", "enables", 0.6),
        ),
    ),
    DisciplineKnowledge(
        key="petrophysics",
        label="Petrophysics and Formation Evaluation",
        concepts=(
            ("saturation", ("saturation", "water saturation", "تشبع")),
            ("net_pay", ("net pay", "net_pay")),
            ("core_sample", ("core sample", "core", "لب")),
            ("conductivity", ("conductivity",)),
        ),
        relations=(
            ("well_log", "saturation", "measures", 0.6),
            ("saturation", "hydrocarbon", "indicates", 0.6),
            ("core_sample", "porosity", "measures", 0.6),
            ("core_sample", "permeability", "measures", 0.6),
            ("net_pay", "hydrocarbon", "contains", 0.7),
        ),
    ),
    DisciplineKnowledge(
        key="well_logging",
        label="Well Logging",
        concepts=(
            ("electrical_log", ("resistivity log", "resistivity", "سجل كهربائي")),
            ("neutron_log", ("neutron log", "neutron")),
            ("cement_bond_log", ("cement bond log", "cbl")),
        ),
        relations=(
            ("electrical_log", "saturation", "measures", 0.6),
            ("neutron_log", "porosity", "measures", 0.6),
            ("cement_bond_log", "cement", "evaluates", 0.6),
        ),
    ),
    DisciplineKnowledge(
        key="completions",
        label="Well Completions",
        concepts=(
            ("perforating", ("perforation", "perforate", "تثقيب")),
            ("completion", ("completion",)),
            ("gravel", ("gravel pack", "gravel")),
        ),
        relations=(
            ("perforating", "well", "connects", 0.7),
            ("completion", "well", "completes", 0.8),
            ("gravel", "production", "stabilizes", 0.6),
        ),
    ),
    DisciplineKnowledge(
        key="eor",
        label="Enhanced Oil Recovery",
        concepts=(
            ("Enhanced Oil Recovery", ("enhanced oil recovery", "eor")),
            ("chemical_injection", ("chemical flooding", "polymer flooding")),
            ("steam_flooding", ("steam flooding", "steam")),
        ),
        relations=(
            ("Enhanced Oil Recovery", "recovery_factor", "increases", 0.7),
            ("chemical_injection", "Enhanced Oil Recovery", "implements", 0.6),
            ("steam_flooding", "Enhanced Oil Recovery", "implements", 0.6),
        ),
    ),
    DisciplineKnowledge(
        key="well_testing",
        label="Well Testing",
        concepts=(
            ("pressure_transient_test", ("pressure transient test", "well test", "build-up test")),
            ("pressure_buildup", ("pressure buildup", "buildup")),
        ),
        relations=(
            ("pressure_transient_test", "reservoir_pressure", "measures", 0.7),
            ("pressure_buildup", "permeability", "estimates", 0.6),
        ),
    ),
    DisciplineKnowledge(
        key="artificial_lift",
        label="Artificial Lift",
        concepts=(
            ("gas_lift", ("gas lift",)),
            ("esp", ("esp", "electrical submersible pump")),
        ),
        relations=(
            ("gas_lift", "Artificial Lift", "implements", 0.6),
            ("esp", "Artificial Lift", "implements", 0.6),
        ),
    ),
    DisciplineKnowledge(
        key="stimulation",
        label="Well Stimulation",
        concepts=(
            ("acid_stimulation", ("acidizing", "acid stimulation")),
            ("hydraulic_fracturing", ("hydraulic fracturing", "fracturing", "fracking")),
        ),
        relations=(
            ("acid_stimulation", "permeability", "increases", 0.7),
            ("hydraulic_fracturing", "permeability", "increases", 0.7),
        ),
    ),
    DisciplineKnowledge(
        key="cementing",
        label="Well Cementing",
        concepts=(
            ("light_cement", ("light cement", "cement slurry")),
            ("well_casing_cementing", ("well casing cementing", "cementation")),
        ),
        relations=(
            ("light_cement", "cement", "is_a", 0.7),
            ("well_casing_cementing", "casing", "seals", 0.7),
        ),
    ),
    DisciplineKnowledge(
        key="well_control",
        label="Well Control",
        concepts=(
            ("safety_valve", ("blowout preventer", "bop", "preventer")),
            ("shut_in", ("shut-in", "shut in")),
            ("kill_mud", ("kill mud", "kill fluid")),
        ),
        relations=(
            ("safety_valve", "gas_kick", "prevents", 0.7),
            ("shut_in", "gas_kick", "controls", 0.6),
            ("kill_mud", "reservoir_pressure", "balances", 0.6),
        ),
    ),
    DisciplineKnowledge(
        key="economics",
        label="Petroleum Economics",
        concepts=(
            ("net_present_value", ("net present value", "npv")),
            ("cost", ("cost",)),
            ("revenue", ("revenue",)),
        ),
        relations=(
            ("production", "revenue", "generates", 0.8),
            ("revenue", "net_present_value", "evaluated_by", 0.6),
            ("cost", "net_present_value", "reduces", 0.6),
        ),
    ),
    DisciplineKnowledge(
        key="hse",
        label="Health, Safety and Environment",
        concepts=(
            ("well_integrity", ("well integrity",)),
            ("pressure_barrier", ("pressure barrier", "barrier")),
            ("inspection", ("inspection",)),
        ),
        relations=(
            ("well_integrity", "cement", "requires", 0.7),
            ("pressure_barrier", "gas_kick", "prevents", 0.7),
            ("inspection", "well_integrity", "verifies", 0.6),
        ),
    ),
    DisciplineKnowledge(
        key="facilities",
        label="Facilities Engineering",
        concepts=(
            ("separator", ("separator", "three-phase separator")),
            ("pipeline", ("pipeline", "flowline")),
            ("gas_processing", ("gas processing", "gas plant")),
            ("pump", ("pump", "pumping system")),
            ("storage_tank", ("storage tank", "tank farm")),
        ),
        relations=(
            ("separator", "production", "processes", 0.7),
            ("pipeline", "hydrocarbon", "transports", 0.7),
            ("gas_processing", "hydrocarbon", "treats", 0.6),
            ("pump", "pipeline", "pressurizes", 0.6),
            ("storage_tank", "hydrocarbon", "stores", 0.6),
        ),
    ),
    DisciplineKnowledge(
        key="process_safety",
        label="Process Safety",
        concepts=(
            ("process_safety_event", ("process safety event", "loss of containment")),
            ("layers_of_protection", ("layers of protection", "protection layers")),
            ("hazard_analysis", ("hazard analysis", "HAZOP")),
            ("safety_case", ("safety case", "safety management system")),
            ("barrier_management", ("barrier management", "barrier health")),
        ),
        relations=(
            ("hazard_analysis", "process_safety_event", "prevents", 0.7),
            ("layers_of_protection", "process_safety_event", "mitigates", 0.7),
            ("barrier_management", "layers_of_protection", "maintains", 0.6),
            ("safety_case", "hazard_analysis", "documents", 0.6),
            ("process_safety_event", "well_integrity", "threatens", 0.6),
        ),
    ),
    DisciplineKnowledge(
        key="unconventional",
        label="Unconventional Resources",
        concepts=(
            ("shale_play", ("shale play", "shale")),
            ("horizontal_lateral", ("horizontal lateral", "lateral")),
            ("multi_stage_fracturing", ("multi-stage fracturing", "multi-stage frac")),
            ("tight_gas", ("tight gas", "tight reservoir")),
            ("sweet_spot", ("sweet spot", "landing zone")),
        ),
        relations=(
            ("shale_play", "reservoir", "hosts", 0.6),
            ("horizontal_lateral", "shale_play", "drains", 0.7),
            ("multi_stage_fracturing", "horizontal_lateral", "stimulates", 0.7),
            ("sweet_spot", "horizontal_lateral", "guides", 0.6),
            ("tight_gas", "shale_play", "produced_from", 0.6),
        ),
    ),
)


def disciplines() -> tuple[DisciplineKnowledge, ...]:
    """The petroleum-engineering disciplines the seed spans."""
    return _DISCIPLINES


def discipline_keys() -> tuple[str, ...]:
    return tuple(discipline.key for discipline in _DISCIPLINES)


def discipline_labels() -> dict[str, str]:
    return {discipline.key: discipline.label for discipline in _DISCIPLINES}


def seed_ontology(ontology: Ontology) -> list[Concept]:
    """Add the petroleum concepts and relations, reusing any that already exist.

    Idempotent: calling it twice leaves the ontology unchanged, so it is safe to
    run on every controller start and on restored graphs.
    """
    concepts: dict[str, Concept] = {}

    def concept_for(label: str, aliases: tuple[str, ...]) -> Concept:
        if label not in concepts:
            existing = ontology.find_concepts(label)
            concepts[label] = existing[0] if existing else _add_concept(ontology, label, aliases)
        return concepts[label]

    for label, aliases in _CONCEPTS:
        concept_for(label, aliases)
    for discipline in _DISCIPLINES:
        for label, aliases in discipline.concepts:
            concept_for(label, aliases)

    for source, target, relation_type, confidence in _FACTS:
        _add_relation(ontology, concepts[source], concepts[target], relation_type, confidence)
    for discipline in _DISCIPLINES:
        for source, target, relation_type, confidence in discipline.relations:
            _add_relation(ontology, concepts[source], concepts[target], relation_type, confidence)

    return list(concepts.values())


def discipline_coverage(ontology: Ontology) -> dict[str, dict[str, int]]:
    """Per-discipline counts of seeded concepts and relations present.

    A relation counts when both endpoints resolve to concepts the ontology holds,
    so a partially-restored graph is reported honestly.
    """
    coverage: dict[str, dict[str, int]] = {}
    for discipline in _DISCIPLINES:
        concept_hits = sum(1 for label, _ in discipline.concepts if ontology.find_concepts(label))
        relation_hits = 0
        for source, target, relation_type, _ in discipline.relations:
            sources = ontology.find_concepts(source)
            targets = ontology.find_concepts(target)
            if not sources or not targets:
                continue
            source_ids = {concept.id for concept in sources}
            target_ids = {concept.id for concept in targets}
            if any(
                relation.source_id in source_ids
                and relation.target_id in target_ids
                and relation.relation_type == relation_type
                for relation in ontology.relations.values()
            ):
                relation_hits += 1
        coverage[discipline.key] = {
            "concepts": concept_hits,
            "relations": relation_hits,
            "expected_concepts": len(discipline.concepts),
            "expected_relations": len(discipline.relations),
        }
    return coverage


def domain_summary(self_model: SelfModel | None = None) -> str:
    """One-line identity frame injected into model requests."""
    model = self_model or SelfModel()
    labels = ", ".join(discipline.label for discipline in _DISCIPLINES)
    return (
        f"{model.identity_statement} Focus areas: {', '.join(model.expertise)}. "
        f"Disciplines: {labels}."
    )


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
    ontology.add_relation_once(source.id, target.id, relation_type, confidence)


__all__ = [
    "DisciplineKnowledge",
    "discipline_coverage",
    "discipline_keys",
    "discipline_labels",
    "disciplines",
    "domain_summary",
    "seed_ontology",
]
