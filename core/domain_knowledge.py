"""Seed knowledge that gives the cognitive architecture its petroleum-engineering frame.

This is a small, inspectable starting ontology -- not a full engineering corpus.
It makes the project's declared identity concrete: the concepts the system can
ground, and the relations it can reason over, are about reservoirs, wells,
petrophysics, and production. Everything remains evidence-aware: seeded
relations carry a modest confidence and can be revised like any other belief.

The seed is organized by petroleum-engineering discipline so the frame spans the
whole field -- reservoir, drilling, production, petrophysics, well logging,
completions, enhanced recovery, well testing, artificial lift, stimulation,
cementing, well control, economics, and HSE. Each discipline contributes its own
concepts and relations; ``discipline_coverage`` reports how much of that frame an
ontology currently carries.
"""

from __future__ import annotations

from dataclasses import dataclass

from core.concept import Concept
from core.ontology import Ontology
from core.self_model import SelfModel
from core.semantic_relation import SemanticRelation

# The foundation layer: the concepts every discipline builds on. Kept separate so
# the original seed stays recognizable and stable.
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
        label="هندسة المكامن",
        concepts=(
            ("الاحتياطي", ("reserve", "احتياطي")),
            ("عامل_الاستخراج", ("recovery factor", "recovery")),
            ("الدفع_المائي", ("waterflood", "water flooding")),
            ("ضغط_المكمن", ("reservoir pressure",)),
            ("محاكاة_المكمن", ("reservoir simulation", "simulation")),
        ),
        relations=(
            ("المكمن", "الاحتياطي", "contains", 0.7),
            ("الاحتياطي", "عامل_الاستخراج", "estimated_by", 0.6),
            ("الدفع_المائي", "ضغط_المكمن", "maintains", 0.7),
            ("ضغط_المكمن", "الانتاج", "drives", 0.7),
            ("محاكاة_المكمن", "الاحتياطي", "predicts", 0.6),
        ),
    ),
    DisciplineKnowledge(
        key="drilling",
        label="هندسة الحفر",
        concepts=(
            ("سائل_الحفر", ("drilling fluid", "mud", "طين الحفر")),
            ("البئر_الموجه", ("directional well", "directional drilling")),
            ("رأس_الحفر", ("drill bit", "bit")),
            ("رشح_الغاز", ("gas kick", "kick", "اندفاع")),
            ("بطانة_البئر", ("casing", "casing string", "بطانة")),
        ),
        relations=(
            ("سائل_الحفر", "رأس_الحفر", "cools", 0.6),
            ("سائل_الحفر", "رشح_الغاز", "controls", 0.7),
            ("البئر_الموجه", "البئر", "creates", 0.6),
            ("رأس_الحفر", "البئر", "drills", 0.7),
            ("بطانة_البئر", "البئر", "supports", 0.7),
        ),
    ),
    DisciplineKnowledge(
        key="production",
        label="هندسة الإنتاج",
        concepts=(
            ("معدل_الانتاج", ("production rate", "flow rate")),
            ("انخفاض_الانتاج", ("production decline", "decline")),
            ("الرفع_الصناعي", ("artificial lift",)),
            ("تحفيز_البئر", ("well stimulation", "stimulation", "acidizing")),
            ("مضخة", ("pump",)),
        ),
        relations=(
            ("الرفع_الصناعي", "معدل_الانتاج", "increases", 0.7),
            ("انخفاض_الانتاج", "تحفيز_البئر", "treated_by", 0.6),
            ("تحفيز_البئر", "معدل_الانتاج", "increases", 0.7),
            ("مضخة", "الرفع_الصناعي", "enables", 0.6),
        ),
    ),
    DisciplineKnowledge(
        key="petrophysics",
        label="البتروفيزياء وتقييم الطبقات",
        concepts=(
            ("التشبع", ("saturation", "water saturation", "تشبع")),
            ("صافي_الطبقة_المنتجة", ("net pay", "net_pay")),
            ("اللب_الصخري", ("core sample", "core", "لب")),
            ("التوصيلية", ("conductivity",)),
        ),
        relations=(
            ("السجل", "التشبع", "measures", 0.6),
            ("التشبع", "الهيدروكربون", "indicates", 0.6),
            ("اللب_الصخري", "المسامية", "measures", 0.6),
            ("اللب_الصخري", "النفاذية", "measures", 0.6),
            ("صافي_الطبقة_المنتجة", "الهيدروكربون", "contains", 0.7),
        ),
    ),
    DisciplineKnowledge(
        key="well_logging",
        label="تسجيلات الآبار",
        concepts=(
            ("السجل_الكهربائي", ("resistivity log", "resistivity", "سجل كهربائي")),
            ("السجل_النيوتروني", ("neutron log", "neutron")),
            ("سجل_الاسمنت", ("cement bond log", "cbl")),
        ),
        relations=(
            ("السجل_الكهربائي", "التشبع", "measures", 0.6),
            ("السجل_النيوتروني", "المسامية", "measures", 0.6),
            ("سجل_الاسمنت", "الاسمنت", "evaluates", 0.6),
        ),
    ),
    DisciplineKnowledge(
        key="completions",
        label="إكمال الآبار",
        concepts=(
            ("التثقيب", ("perforation", "perforate", "تثقيب")),
            ("الاكمال", ("completion",)),
            ("الرمل_المصقول", ("gravel pack", "gravel")),
        ),
        relations=(
            ("التثقيب", "البئر", "connects", 0.7),
            ("الاكمال", "البئر", "completes", 0.8),
            ("الرمل_المصقول", "الانتاج", "stabilizes", 0.6),
        ),
    ),
    DisciplineKnowledge(
        key="eor",
        label="الاستخلاص المعزز",
        concepts=(
            ("الاستخلاص_المعزز", ("enhanced oil recovery", "eor")),
            ("الحقن_الكيميائي", ("chemical flooding", "polymer flooding")),
            ("البخار", ("steam flooding", "steam")),
        ),
        relations=(
            ("الاستخلاص_المعزز", "عامل_الاستخراج", "increases", 0.7),
            ("الحقن_الكيميائي", "الاستخلاص_المعزز", "implements", 0.6),
            ("البخار", "الاستخلاص_المعزز", "implements", 0.6),
        ),
    ),
    DisciplineKnowledge(
        key="well_testing",
        label="اختبار الآبار",
        concepts=(
            ("اختبار_الضغط", ("pressure transient test", "well test", "build-up test")),
            ("البناء_الضغطي", ("pressure buildup", "buildup")),
        ),
        relations=(
            ("اختبار_الضغط", "ضغط_المكمن", "measures", 0.7),
            ("البناء_الضغطي", "النفاذية", "estimates", 0.6),
        ),
    ),
    DisciplineKnowledge(
        key="artificial_lift",
        label="الرفع الصناعي",
        concepts=(
            ("الرفع_بالغاز", ("gas lift",)),
            ("المضخة_الغاطسة", ("esp", "electrical submersible pump")),
        ),
        relations=(
            ("الرفع_بالغاز", "الرفع_الصناعي", "implements", 0.6),
            ("المضخة_الغاطسة", "الرفع_الصناعي", "implements", 0.6),
        ),
    ),
    DisciplineKnowledge(
        key="stimulation",
        label="تحفيز الآبار",
        concepts=(
            ("التحمض", ("acidizing", "acid stimulation")),
            ("التشقيق_الهيدروليكي", ("hydraulic fracturing", "fracturing", "fracking")),
        ),
        relations=(
            ("التحمض", "النفاذية", "increases", 0.7),
            ("التشقيق_الهيدروليكي", "النفاذية", "increases", 0.7),
        ),
    ),
    DisciplineKnowledge(
        key="cementing",
        label="إسمنت الآبار",
        concepts=(
            ("الاسمنت_الخفيف", ("light cement", "cement slurry")),
            ("تغليف_البئر", ("well casing cementing", "cementation")),
        ),
        relations=(
            ("الاسمنت_الخفيف", "الاسمنت", "is_a", 0.7),
            ("تغليف_البئر", "بطانة_البئر", "seals", 0.7),
        ),
    ),
    DisciplineKnowledge(
        key="well_control",
        label="التحكم في الآبار",
        concepts=(
            ("صمام_الامان", ("blowout preventer", "bop", "preventer")),
            ("الاغلاق", ("shut-in", "shut in")),
            ("طين_القتل", ("kill mud", "kill fluid")),
        ),
        relations=(
            ("صمام_الامان", "رشح_الغاز", "prevents", 0.7),
            ("الاغلاق", "رشح_الغاز", "controls", 0.6),
            ("طين_القتل", "ضغط_المكمن", "balances", 0.6),
        ),
    ),
    DisciplineKnowledge(
        key="economics",
        label="اقتصاديات البترول",
        concepts=(
            ("صافي_القيمة_الحالية", ("net present value", "npv")),
            ("التكلفة", ("cost",)),
            ("الايراد", ("revenue",)),
        ),
        relations=(
            ("الانتاج", "الايراد", "generates", 0.8),
            ("الايراد", "صافي_القيمة_الحالية", "evaluated_by", 0.6),
            ("التكلفة", "صافي_القيمة_الحالية", "reduces", 0.6),
        ),
    ),
    DisciplineKnowledge(
        key="hse",
        label="السلامة والصحة والبيئة",
        concepts=(
            ("سلامة_البئر", ("well integrity",)),
            ("حاجز_الضغط", ("pressure barrier", "barrier")),
            ("التفتيش", ("inspection",)),
        ),
        relations=(
            ("سلامة_البئر", "الاسمنت", "requires", 0.7),
            ("حاجز_الضغط", "رشح_الغاز", "prevents", 0.7),
            ("التفتيش", "سلامة_البئر", "verifies", 0.6),
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


__all__ = [
    "DisciplineKnowledge",
    "discipline_coverage",
    "discipline_keys",
    "discipline_labels",
    "disciplines",
    "domain_summary",
    "seed_ontology",
]
