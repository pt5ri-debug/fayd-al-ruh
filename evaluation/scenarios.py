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
    SAFETY = "safety"


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
    requires_ontology: bool = False


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
class OntologyLearningSpec:
    """An observed text and the concept links it should (not) teach.

    ``expected_links`` are concept-label pairs the observation must connect with
    a new weak relation. ``forbidden_links`` are pairs that must stay untouched,
    e.g. because an explicit relation already states them.
    """

    text: str
    expected_links: tuple[tuple[str, str], ...] = ()
    forbidden_links: tuple[tuple[str, str], ...] = ()


@dataclass(frozen=True)
class OntologyReinforcementSpec:
    """A repeated observation and how experience should reshape its learned links.

    The text is observed ``repeats`` times, so each pair it teaches is reinforced
    step by step. ``min_confidence`` is the confidence the strongest learned link
    must reach, and ``ceiling`` bounds every learned link so experience never
    rivals the seed. ``decay_rounds`` decays are then applied: if
    ``expect_removed`` is set, the learned links must fade away entirely while the
    seeded relations stay untouched.
    """

    text: str
    repeats: int = 1
    min_confidence: float = 0.0
    ceiling: float = 1.0
    decay_rounds: int = 0
    expect_removed: bool = True


@dataclass(frozen=True)
class OntologyFeedbackSpec:
    """A recall whose outcome should reshape the learned link behind it.

    A learned relation is established from ``text``, then a memory item is stored
    that only a related concept (``probe``) can reach. The link is credited or
    blamed through ``useful`` feedback, and ``expected_outcome`` records what must
    happen: ``reinforced`` raises the link's confidence, ``weakened``/``removed``
    lower or drop it. ``expect_recall_after`` says whether the item must still be
    reachable through the link once feedback has been applied.
    """

    text: str
    probe: str
    item_text: str
    useful: bool
    amount: float
    expected_outcome: str
    expect_recall_after: bool = True


@dataclass(frozen=True)
class GateCase:
    """One acceptance-gate judgement with a known-correct verdict.

    Unlike a retrieval expectation, whose answer the project's own seeds supply,
    a gate case states what the *policy* must decide for a source that was never
    part of the corpus. Half the cases are traps the policy must refuse (an
    untrusted host, thin credibility, a formula fragment, broken prose, an
    article about the wrong kind of "reservoir"), so a gate that accepts
    everything -- the failure mode a closed self-test cannot see -- scores below
    1.0 and moves this metric.
    """

    url: str
    title: str
    topic: str
    snippet: str
    credibility: float
    expect_accept: bool
    note: str = ""


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
    # Expectations reachable only by walking ontology relations (no shared word).
    ontology_retrieval: tuple[RetrievalExpectation, ...] = ()
    # Observations that should teach the ontology new weak relations.
    ontology_learning: tuple[OntologyLearningSpec, ...] = ()
    # Repeated observations that should reinforce/decay learned relations.
    ontology_reinforcement: tuple[OntologyReinforcementSpec, ...] = ()
    # Recalls whose outcomes should credit/blame the learned link behind them.
    ontology_feedback: tuple[OntologyFeedbackSpec, ...] = ()
    # Acceptance-gate judgements with a known-correct verdict (a held-out check).
    gate_cases: tuple[GateCase, ...] = ()


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
            ("well_log", "measures", "porosity"),
            ("well_log", "measures", "permeability"),
            ("pressure", "drives", "production"),
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
            ("reservoir", "contained_in", "formation"),
            ("hydrocarbon", "exists_in", "reservoir"),
            ("well", "penetrates", "reservoir"),
            ("cement", "seals", "well"),
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
            ("well", "penetrates", "formation"),
            # Fluids held in a reservoir are also held in the reservoir's formation.
            ("hydrocarbon", "exists_in", "formation"),
            # A reservoir inherits the properties of its containing formation.
            ("reservoir", "has_property", "porosity"),
            # A tool measuring a property also measures the reservoir that has it.
            ("well_log", "measures", "reservoir"),
        ),
    )


def _scenario_petroleum_multihop_retrieval() -> Scenario:
    """Petroleum domain: retrieval follows derived relations across hops.

    Each marked expectation targets an observation that shares no word with the
    query; it is only reachable by walking the seeded (and derived) relations,
    e.g. ``well_log measures reservoir`` reaching a reservoir observation from a log query.
    """
    return Scenario(
        id="petroleum-multihop-retrieval",
        domain=Domain.PETROLEUM,
        description="Log and drilling queries reach reservoir/well observations through ontology hops.",
        observations=(
            ObservationSpec("The reservoir holds hydrocarbons in the formation.", "reservoir_study", 0.9),
            ObservationSpec("The well was secured with steel casing.", "completion_report", 0.9),
            ObservationSpec("The rig crew completed a routine safety briefing.", "hse_log", 0.5),
        ),
        retrieval=(
            # A lexical hit the ordinary retriever already reaches (control).
            RetrievalExpectation(
                "safety briefing",
                "The rig crew completed a routine safety briefing.",
            ),
        ),
        ontology_retrieval=(
            # "log" -> well_log measures reservoir -> reservoir observation (no shared word).
            RetrievalExpectation(
                "log interpretation",
                "The reservoir holds hydrocarbons in the formation.",
                requires_ontology=True,
            ),
            # "drilling" -> drilling creates well -> well observation (no shared word).
            RetrievalExpectation(
                "drilling operations",
                "The well was secured with steel casing.",
                requires_ontology=True,
            ),
        ),
    )


def _scenario_petroleum_ontology_learning() -> Scenario:
    """Petroleum domain: the ontology grows from observations, not only its seed.

    A single field observation grounds three known concepts. The pairs with no
    explicit relation (pressure, permeability) and (pressure, formation) must be learned as weak
    ``associated_with`` links, while the already-seeded (formation, permeability) must be
    left untouched so experience never overwrites curated knowledge.
    """
    return Scenario(
        id="petroleum-ontology-learning",
        domain=Domain.PETROLEUM,
        description="Co-occurring concepts teach weak relations; seeded pairs stay untouched.",
        ontology_learning=(
            OntologyLearningSpec(
                text="rising pressure in the formation accompanies improved permeability",
                expected_links=(("permeability", "pressure"), ("formation", "pressure")),
                forbidden_links=(("formation", "permeability"),),
            ),
        ),
    )


def _scenario_petroleum_ontology_reinforcement() -> Scenario:
    """Petroleum domain: experience weights the links it learns.

    The same field observation is repeated, so the links it teaches should be
    reinforced step by step up to a bounded confidence below the seed. A second
    observation is then left to decay: its learned links must fade away entirely
    while the seeded relations stay untouched.
    """
    return Scenario(
        id="petroleum-ontology-reinforcement",
        domain=Domain.PETROLEUM,
        description="Repetition strengthens learned links; disuse decays them without touching the seed.",
        ontology_reinforcement=(
            OntologyReinforcementSpec(
                text="rising pressure in the formation accompanies improved permeability",
                repeats=5,
                min_confidence=0.5,
                ceiling=0.6,
            ),
            OntologyReinforcementSpec(
                text="a change in the reservoir affects the cement",
                repeats=1,
                min_confidence=0.3,
                ceiling=0.6,
                decay_rounds=6,
                expect_removed=True,
            ),
        ),
    )


def _scenario_petroleum_ontology_feedback() -> Scenario:
    """Petroleum domain: the outcome of a recall reshapes the link that produced it.

    A learned link connects pressure to permeability, so a query about permeability reaches
    an item that only mentions pressure. A useful outcome reinforces that link, while an
    unhelpful one weakens and finally drops it — so retrieval is shaped by results,
    and a seeded relation behind a recall is never credited.
    """
    return Scenario(
        id="petroleum-ontology-feedback",
        domain=Domain.PETROLEUM,
        description="Useful recalls strengthen the learned link behind them; unhelpful ones weaken it.",
        ontology_feedback=(
            OntologyFeedbackSpec(
                text="rising pressure accompanies improved permeability",
                probe="permeability",
                item_text="pressure controls flow.",
                useful=True,
                amount=0.1,
                expected_outcome="reinforced",
                expect_recall_after=True,
            ),
            OntologyFeedbackSpec(
                text="rising pressure accompanies improved permeability",
                probe="permeability",
                item_text="pressure controls flow.",
                useful=False,
                amount=0.5,
                expected_outcome="removed",
                expect_recall_after=False,
            ),
        ),
    )


def _scenario_gate_discrimination() -> Scenario:
    """Held-out gate discrimination: the acceptance policy must refuse the traps.

    Every other scenario checks that the project agrees with its *own* seeds, so
    a policy that accepted everything would still score 1.0. This one states the
    verdict for sources that were never part of the corpus -- five traps the gate
    must refuse and five statements it must accept -- so the metric moves when
    the gate weakens. It is the one place the project's judgement is measured
    against an external standard rather than against itself.
    """
    return Scenario(
        id="gate-discrimination",
        domain=Domain.SAFETY,
        description="The acceptance gate refuses the traps and keeps the real statements.",
        gate_cases=(
            # Must be refused: an untrusted host.
            GateCase(
                url="https://some-random-blog.example/petroleum",
                title="Reservoir pressure",
                topic="reservoir pressure",
                snippet="Reservoir pressure declines as hydrocarbons are produced.",
                credibility=0.95,
                expect_accept=False,
                note="untrusted host",
            ),
            # Must be refused: credibility below the floor.
            GateCase(
                url="https://en.wikipedia.org/wiki/Porosity",
                title="Porosity",
                topic="porosity",
                snippet="Porosity is the fraction of the bulk volume that is void.",
                credibility=0.35,
                expect_accept=False,
                note="credibility below floor",
            ),
            # Must be refused: symbolic markup, not a statement.
            GateCase(
                url="https://en.wikipedia.org/wiki/Darcy%27s_law",
                title="Darcy's law",
                topic="darcy law",
                snippet=r"\displaystyle q = \frac{kA}{\mu} \nabla p",
                credibility=0.9,
                expect_accept=False,
                note="formula fragment",
            ),
            # Must be refused: broken prose, not a quotable statement.
            GateCase(
                url="https://en.wikipedia.org/wiki/Drilling_fluid",
                title="Drilling fluid",
                topic="drilling fluid",
                snippet="drawbacks\u2014mud filtrate soaks into the earth's",
                credibility=0.9,
                expect_accept=False,
                note="incomplete statement",
            ),
            # Must be refused: a water reservoir is not a petroleum reservoir.
            GateCase(
                url="https://en.wikipedia.org/wiki/Quabbin_Reservoir",
                title="Quabbin Reservoir",
                topic="reservoir study",
                snippet="Quabbin Reservoir is the largest inland body of water in Massachusetts.",
                credibility=0.9,
                expect_accept=False,
                note="article about the wrong kind of reservoir",
            ),
            # Must be accepted: real, trusted, on-topic statements.
            GateCase(
                url="https://en.wikipedia.org/wiki/Porosity",
                title="Porosity",
                topic="porosity",
                snippet="Porosity is the fraction of the bulk volume that is void.",
                credibility=0.9,
                expect_accept=True,
                note="trusted and on topic",
            ),
            GateCase(
                url="https://www.spe.org/petrophysics/permeability",
                title="Permeability",
                topic="permeability",
                snippet="Permeability measures the ability of a rock to transmit fluids.",
                credibility=0.9,
                expect_accept=True,
                note="trusted and on topic",
            ),
            GateCase(
                url="https://en.wikipedia.org/wiki/Reservoir",
                title="Reservoir",
                topic="reservoir",
                snippet="A petroleum reservoir is a subsurface accumulation of hydrocarbons.",
                credibility=0.9,
                expect_accept=True,
                note="trusted and on topic",
            ),
            GateCase(
                url="https://en.wikipedia.org/wiki/Well_logging",
                title="Well logging",
                topic="well logging",
                snippet="Well logging records the properties of a borehole in a well.",
                credibility=0.9,
                expect_accept=True,
                note="trusted and on topic",
            ),
            GateCase(
                url="https://www.energy.gov/petroleum",
                title="Petroleum",
                topic="petroleum",
                snippet="Petroleum is a naturally occurring liquid found in geological formations.",
                credibility=0.9,
                expect_accept=True,
                note="trusted and on topic",
            ),
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
    _scenario_petroleum_multihop_retrieval(),
    _scenario_petroleum_ontology_learning(),
    _scenario_petroleum_ontology_reinforcement(),
    _scenario_petroleum_ontology_feedback(),
    _scenario_gate_discrimination(),
)


def scenario_ids() -> tuple[str, ...]:
    return tuple(scenario.id for scenario in SCENARIOS)
