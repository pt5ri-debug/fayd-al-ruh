from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class Domain(str, Enum):
    """The cognitive capability a scenario exercises."""

    MEMORY = "memory"
    REASONING = "reasoning"
    CAUSAL = "causal"
    PLANNING = "planning"
    UNCERTAINTY = "uncertainty"
    PETROLEUM = "petroleum"


@dataclass(frozen=True)
class ObservationSpec:
    content: str
    source: str
    confidence: float = 1.0
    supports: bool = True


@dataclass(frozen=True)
class BeliefSpec:
    statement: str
    confidence: float = 0.5


@dataclass(frozen=True)
class RetrievalExpectation:
    query: str
    expected_top: str


@dataclass(frozen=True)
class PredictionSpec:
    """A hypothesis-like forecast judged against an observed outcome.

    ``confidence`` is the forecast probability, ``outcome`` the observed
    result in ``[0, 1]`` (>= 0.5 counts as the event happening).
    """

    statement: str
    confidence: float
    outcome: float


@dataclass(frozen=True)
class CausalSpec:
    cause: str
    effect: str
    confidence: float = 0.8
    discovered_from: tuple[str, ...] = ()


@dataclass(frozen=True)
class DecisionActionSpec:
    name: str
    preconditions: frozenset[str] = frozenset()
    add_effects: frozenset[str] = frozenset()
    remove_effects: frozenset[str] = frozenset()
    cost: float = 1.0


@dataclass(frozen=True)
class DecisionSpec:
    initial_facts: frozenset[str]
    goal: str
    desired_facts: frozenset[str]
    actions: tuple[DecisionActionSpec, ...]
    expected_selected: str


@dataclass(frozen=True)
class PlanSpec:
    goal: str
    initial_facts: frozenset[str]
    desired_facts: frozenset[str]
    actions: tuple[DecisionActionSpec, ...]
    max_depth: int = 6


@dataclass(frozen=True)
class UncertaintySpec:
    """A belief revised from explicit evidence with a known true probability."""

    statement: str
    prior_confidence: float
    evidence: tuple[ObservationSpec, ...]
    true_probability: float


@dataclass(frozen=True)
class Scenario:
    """A single, fixed evaluation case."""

    id: str
    domain: Domain
    description: str
    observations: tuple[ObservationSpec, ...] = ()
    beliefs: tuple[BeliefSpec, ...] = ()
    expected_conflicts: int = 0
    retrieval: tuple[RetrievalExpectation, ...] = ()
    predictions: tuple[PredictionSpec, ...] = ()
    causal: tuple[CausalSpec, ...] = ()
    decisions: tuple[DecisionSpec, ...] = ()
    plans: tuple[PlanSpec, ...] = ()
    uncertainties: tuple[UncertaintySpec, ...] = ()
    # Expected seeded ontology relations as (source_label, relation_type, target_label).
    ontology_relations: tuple[tuple[str, str, str], ...] = ()
    # Relations that must be derivable (not stored) as (source_label, relation_type, target_label).
    ontology_inferences: tuple[tuple[str, str, str], ...] = ()


def _scenario_memory() -> Scenario:
    return Scenario(
        id="memory-recall",
        domain=Domain.MEMORY,
        description="Distinct episodes must be retrievable by topic, not by noise.",
        observations=(
            ObservationSpec("The garden receives rain and becomes wet.", "weather_sensor"),
            ObservationSpec("A robot arm moves with a motor and steel frame.", "lab_sensor"),
            ObservationSpec("Rain clouds gather above the garden again.", "weather_sensor"),
        ),
        beliefs=(BeliefSpec("Rain wets the garden.", 0.5),),
        retrieval=(
            RetrievalExpectation("rain garden wet", "The garden receives rain and becomes wet."),
            RetrievalExpectation("robot motor steel", "A robot arm moves with a motor and steel frame."),
        ),
    )


def _scenario_reasoning() -> Scenario:
    return Scenario(
        id="reasoning-prediction",
        domain=Domain.REASONING,
        description="Forecasts across the confidence range are judged against outcomes.",
        beliefs=(BeliefSpec("Recurring rain predicts a wet garden.", 0.5),),
        predictions=(
            PredictionSpec("It will rain again tomorrow.", 0.9, 1.0),
            PredictionSpec("The garden stays wet after rain.", 0.85, 1.0),
            PredictionSpec("No more rain for a week.", 0.15, 0.0),
            PredictionSpec("The soil is already saturated.", 0.2, 0.0),
            PredictionSpec("A dry spell is very unlikely.", 0.1, 0.0),
        ),
    )


def _scenario_causal() -> Scenario:
    return Scenario(
        id="causal-chain",
        domain=Domain.CAUSAL,
        description="A directed cause->effect chain is discovered and explained.",
        causal=(
            CausalSpec("rain", "wet_ground", 0.9, ("rain", "wet_ground", "rain", "wet_ground")),
            CausalSpec("wet_ground", "slippery_path", 0.8),
        ),
    )


def _scenario_planning() -> Scenario:
    return Scenario(
        id="planning-reach-goal",
        domain=Domain.PLANNING,
        description="A short action sequence must reach the desired facts.",
        plans=(
            PlanSpec(
                goal="Wet the garden.",
                initial_facts=frozenset({"dry_soil"}),
                desired_facts=frozenset({"wet_soil"}),
                actions=(
                    DecisionActionSpec("water", frozenset({"dry_soil"}), frozenset({"wet_soil"}), cost=1.0),
                    DecisionActionSpec("wait", frozenset({"dry_soil"}), frozenset(), cost=0.1),
                ),
            ),
        ),
    )


def _scenario_uncertainty() -> Scenario:
    return Scenario(
        id="uncertainty-calibration",
        domain=Domain.UNCERTAINTY,
        description="Belief revision tracks evidence and stays calibrated.",
        uncertainties=(
            UncertaintySpec(
                "Rain makes the ground wet.",
                0.5,
                (
                    ObservationSpec("Ground is wet after rain.", "sensor_a", 0.9, True),
                    ObservationSpec("Wet ground observed again.", "sensor_b", 0.8, True),
                ),
                0.85,
            ),
            UncertaintySpec(
                "The soil stays dry in rain.",
                0.5,
                (
                    ObservationSpec("Soil is dry after rain.", "sensor_c", 0.9, False),
                    ObservationSpec("No moisture found in the soil.", "sensor_d", 0.7, False),
                ),
                0.15,
            ),
        ),
    )


def _scenario_consistency() -> Scenario:
    return Scenario(
        id="consistency-contradiction",
        domain=Domain.REASONING,
        description="An explicit statement/negation pair must be flagged as a conflict.",
        beliefs=(
            BeliefSpec("The garden is wet.", 0.7),
            BeliefSpec("Not the garden is wet.", 0.6),
        ),
        expected_conflicts=1,
    )


def _scenario_decisions() -> Scenario:
    return Scenario(
        id="decision-explanation",
        domain=Domain.PLANNING,
        description="The highest utility-adjusted action is selected and explained.",
        decisions=(
            DecisionSpec(
                initial_facts=frozenset({"dry_soil"}),
                goal="Wet the garden.",
                desired_facts=frozenset({"wet_soil"}),
                actions=(
                    DecisionActionSpec("water", frozenset({"dry_soil"}), frozenset({"wet_soil"}), cost=1.0),
                    DecisionActionSpec("ignore", frozenset({"dry_soil"}), frozenset(), cost=0.0),
                ),
                expected_selected="water",
            ),
        ),
    )


def _scenario_petroleum_causal() -> Scenario:
    """Petroleum domain: pressure -> production -> revenue and permeability -> flow."""
    return Scenario(
        id="petroleum-causal",
        domain=Domain.PETROLEUM,
        description="A discovered pressure->production->revenue chain is explained in-domain.",
        causal=(
            CausalSpec("pressure", "production", 0.9, ("pressure", "production", "pressure", "production")),
            CausalSpec("production", "revenue", 0.8),
        ),
    )


def _scenario_petroleum_petrophysics() -> Scenario:
    """Petroleum domain: well-log readings ground the seeded petrophysics concepts."""
    return Scenario(
        id="petroleum-petrophysics",
        domain=Domain.PETROLEUM,
        description="Log-derived readings retrieve the seeded porosity/permeability concepts.",
        observations=(
            ObservationSpec("The well log shows high porosity in the reservoir.", "well_log", 0.9),
            ObservationSpec("A core sample reveals high permeability in the sandstone.", "core_lab", 0.9),
            ObservationSpec("Cement seals the well from the upper formation.", "cementing_report", 0.8),
        ),
        beliefs=(BeliefSpec("The formation is porous and permeable.", 0.5),),
        retrieval=(
            RetrievalExpectation(
                "well log porosity reservoir",
                "The well log shows high porosity in the reservoir.",
            ),
            RetrievalExpectation(
                "core sample permeability sandstone",
                "A core sample reveals high permeability in the sandstone.",
            ),
        ),
        ontology_relations=(
            ("السجل", "measures", "المسامية"),
            ("السجل", "measures", "النفاذية"),
            ("الضغط", "drives", "الانتاج"),
        ),
    )


def _scenario_petroleum_planning() -> Scenario:
    """Petroleum domain: complete a well and treat a production decline."""
    return Scenario(
        id="petroleum-planning",
        domain=Domain.PLANNING,
        description="A drilling/completion sequence and a production treatment both reach their goals.",
        plans=(
            PlanSpec(
                goal="Complete the well.",
                initial_facts=frozenset({"well_drilled"}),
                desired_facts=frozenset({"well_completed"}),
                actions=(
                    DecisionActionSpec("run_casing", frozenset({"well_drilled"}), frozenset({"casing_set"}), cost=1.0),
                    DecisionActionSpec("cement", frozenset({"casing_set"}), frozenset({"well_completed"}), cost=1.0),
                    DecisionActionSpec("wait", frozenset({"well_drilled"}), frozenset(), cost=0.1),
                ),
            ),
            PlanSpec(
                goal="Treat the production decline.",
                initial_facts=frozenset({"pressure_declining"}),
                desired_facts=frozenset({"production_restored"}),
                actions=(
                    DecisionActionSpec("water_injection", frozenset({"pressure_declining"}), frozenset({"pressure_supported"}), cost=1.5),
                    DecisionActionSpec("well_stimulation", frozenset({"pressure_supported"}), frozenset({"production_restored"}), cost=1.5),
                    DecisionActionSpec("do_nothing", frozenset({"pressure_declining"}), frozenset(), cost=0.0),
                ),
            ),
        ),
    )


def _scenario_petroleum_uncertainty() -> Scenario:
    """Petroleum domain: is production declining, revised from field evidence?"""
    return Scenario(
        id="petroleum-uncertainty",
        domain=Domain.UNCERTAINTY,
        description="Beliefs about production decline are revised from field evidence and stay calibrated.",
        uncertainties=(
            UncertaintySpec(
                "Production is declining.",
                0.5,
                (
                    ObservationSpec("Wellhead pressure dropped this month.", "pressure_gauge", 0.9, True),
                    ObservationSpec("Oil rate fell below the forecast.", "flow_meter", 0.8, True),
                ),
                0.85,
            ),
            UncertaintySpec(
                "Production is stable.",
                0.5,
                (
                    ObservationSpec("Oil rate held above the forecast.", "flow_meter", 0.9, False),
                    ObservationSpec("Wellhead pressure stayed constant.", "pressure_gauge", 0.8, False),
                ),
                0.15,
            ),
        ),
    )


def _scenario_petroleum_identity() -> Scenario:
    """Petroleum domain: the seeded ontology carries the declared domain frame."""
    return Scenario(
        id="petroleum-identity",
        domain=Domain.PETROLEUM,
        description="The seeded ontology exposes the petroleum relations the identity claims.",
        ontology_relations=(
            ("المكمن", "contained_in", "الطبقة"),
            ("الهيدروكربون", "exists_in", "المكمن"),
            ("البئر", "penetrates", "المكمن"),
            ("الاسمنت", "seals", "البئر"),
        ),
    )


def _scenario_petroleum_ontology_inference() -> Scenario:
    """Petroleum domain: relations follow from the seeded ontology without being stored."""
    return Scenario(
        id="petroleum-ontology-inference",
        domain=Domain.PETROLEUM,
        description="Transitive containment and property/tool composition are derived, not stored.",
        ontology_inferences=(
            # A well penetrating a reservoir also penetrates the formation holding it.
            ("البئر", "penetrates", "الطبقة"),
            # Fluids held in a reservoir are also held in the reservoir's formation.
            ("الهيدروكربون", "exists_in", "الطبقة"),
            # A reservoir inherits the properties of its containing formation.
            ("المكمن", "has_property", "المسامية"),
            # A tool measuring a property also measures the reservoir that has it.
            ("السجل", "measures", "المكمن"),
        ),
    )


SCENARIOS: tuple[Scenario, ...] = (
    _scenario_memory(),
    _scenario_reasoning(),
    _scenario_causal(),
    _scenario_planning(),
    _scenario_uncertainty(),
    _scenario_consistency(),
    _scenario_decisions(),
    _scenario_petroleum_causal(),
    _scenario_petroleum_petrophysics(),
    _scenario_petroleum_planning(),
    _scenario_petroleum_uncertainty(),
    _scenario_petroleum_identity(),
    _scenario_petroleum_ontology_inference(),
)


def scenario_ids() -> tuple[str, ...]:
    return tuple(scenario.id for scenario in SCENARIOS)
