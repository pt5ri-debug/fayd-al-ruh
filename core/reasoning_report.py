"""A user-visible reasoning trace: *why* this answer.

The project carries a whole reasoning layer -- `ReasoningTrace`,
`ExplanationEngine`, `ReasoningMonitor`, the ontology reasoner, the causal walk --
but none of it reached a surface a person reads. An answer arrived with its
citations, its graph facts, and (since Phase 38) its confidence, yet nothing tied
those pieces together into a single account of *how the answer was reached*. That
is Gap 6 of ``docs/REASONING_ASSESSMENT.md``.

This module assembles that account. :func:`reasoning_report` reads the *data* an
:class:`~core.assistant.AssistantAnswer` already carries -- the retrieved evidence,
the graph's stored relations, the procedure the planner ordered, the recommendation
the decision engine ranked, the belief the graph supports, the relations it entails,
the chains it walks, the causal paths it states, the contradictions it holds, and the
confidence breakdown -- and renders them as one ordered, labelled, auditable list of
steps.

It is **read-only** and **honest by construction**: every step names the source it
came from (the evidence, the graph, or a derivation), it never adds a fact the
answer did not already have, and it says so plainly when a step is empty rather
than padding the trace. The trace is *not* a hidden chain of thought -- it is the
record of what the program actually consulted and derived.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field

from core.answer_belief import AnswerBelief
from core.answer_confidence import AnswerConfidence
from core.answer_decision import AnswerDecision
from core.answer_plan import AnswerPlan
from core.answer_temporal import AnswerTemporal
from core.causal_paths import CausalExplanation
from core.rag import RetrievedPassage
from core.relation_connections import ConceptConnection
from core.relation_consistency import RelationConflict
from core.relation_facts import RelationFact
from core.relation_inference import InferredRelation

#: The label for each step, so every surface names a step the same way.
SOURCE_EVIDENCE = "Retrieved evidence"
SOURCE_STORED = "Stored in the knowledge graph"
SOURCE_PLAN = "The procedure the corpus states (ordered by the planner)"
SOURCE_DECISION = "The options the graph states (ranked by the decision engine)"
SOURCE_BELIEF = "The evidence-weighed belief the graph supports"
SOURCE_INFERRED = "Entailed by the graph (derived, not stored)"
SOURCE_CONNECTIONS = "Walked connections (multi-hop, not stored)"
SOURCE_CAUSAL = "Directed causal paths the graph states"
SOURCE_TEMPORAL = "The timeline the conversation holds (ordered by the temporal reasoner)"
SOURCE_CONFLICTS = "Contradictions the graph holds"
SOURCE_CONFIDENCE = "How the confidence was reached"

#: The order the steps are shown in: what was read first, then what was derived,
#: then the weighing. Fixed, so the same answer always renders the same trace.
STEP_ORDER: tuple[str, ...] = (
    SOURCE_EVIDENCE,
    SOURCE_STORED,
    SOURCE_PLAN,
    SOURCE_DECISION,
    SOURCE_BELIEF,
    SOURCE_INFERRED,
    SOURCE_CONNECTIONS,
    SOURCE_CAUSAL,
    SOURCE_TEMPORAL,
    SOURCE_CONFLICTS,
    SOURCE_CONFIDENCE,
)


@dataclass(frozen=True)
class ReasoningStep:
    """One labelled step of the reasoning trace."""

    label: str
    """Which kind of reasoning this step records (one of :data:`STEP_ORDER`)."""

    lines: tuple[str, ...]
    """The step's content, one readable line each, in the order it was read."""

    derived: bool = False
    """True when the step is a *derivation* (inferred/walked/causal), never a
    stored claim -- so a reader can tell a conclusion the program reached from one
    someone wrote."""

    def describe(self) -> str:
        """The step as a heading followed by its indented lines."""
        if not self.lines:
            return f"{self.label}: (nothing)"
        body = "\n".join(f"  - {line}" for line in self.lines)
        return f"{self.label}:\n{body}"


@dataclass(frozen=True)
class ReasoningReport:
    """The full, ordered account of how an answer was reached."""

    question: str
    steps: tuple[ReasoningStep, ...] = field(default_factory=tuple)

    @property
    def derived_steps(self) -> tuple[ReasoningStep, ...]:
        """Only the steps that record a derivation with content, never a stored claim."""
        return tuple(step for step in self.steps if step.derived and step.lines)

    @property
    def is_empty(self) -> bool:
        """True when no step carried anything -- the answer rested on nothing."""
        return not any(step.lines for step in self.steps)

    def describe(self) -> str:
        """The whole trace as one labelled, readable block."""
        if self.is_empty:
            return "How this answer was reached: no evidence and no graph reasoning was available."
        header = f"How this answer was reached ({self.question}):"
        return "\n".join([header, *(step.describe() for step in self.steps)])


def _evidence_lines(passages: Iterable[RetrievedPassage]) -> tuple[str, ...]:
    return tuple(
        f"{passage.citation} (score {passage.score:.2f}, {passage.signal})"
        for passage in passages
    )


def _confidence_lines(report: AnswerConfidence | None) -> tuple[str, ...]:
    if report is None:
        return ()
    lines = [
        f"evidence {report.evidence:.2f}",
        f"strongest graph signal {report.graph_support:.2f}",
        f"conflict penalty {report.conflict_penalty:.2f}",
        f"final {report.final:.2f}",
    ]
    # The terms above are the raw values; `reasons` is the arithmetic that combined
    # them (the graph signal enters at a reduced weight), so the two do not repeat.
    lines.extend(f"because {reason}" for reason in report.reasons)
    return tuple(lines)


def _plan_lines(plan: AnswerPlan | None) -> tuple[str, ...]:
    if plan is None:
        return ()
    return tuple(f"{index}. {step}" for index, step in enumerate(plan.steps, start=1))


def _decision_lines(decision: AnswerDecision | None) -> tuple[str, ...]:
    if decision is None:
        return ()
    return tuple(
        f"{option} ({relation}, confidence {confidence:.2f})"
        for option, relation, confidence in decision.options
    )


def _belief_lines(belief: AnswerBelief | None) -> tuple[str, ...]:
    if belief is None:
        return ()
    lines = [
        f"{belief.statement} — confidence {belief.confidence:.2f} ({belief.status})"
    ]
    lines.extend(f"supports: {line}" for line in belief.supporting)
    lines.extend(f"contradicts: {line}" for line in belief.contradicting)
    return tuple(lines)


def _temporal_lines(temporal: AnswerTemporal | None) -> tuple[str, ...]:
    if temporal is None:
        return ()
    lines = [
        f"{entry.role}: {entry.text} @ {entry.at.isoformat()}" for entry in temporal.entries
    ]
    lines.append(f"latest change: {temporal.change} ({temporal.relation})")
    if temporal.pattern is not None:
        lines.append(
            f"pattern \"{temporal.pattern.description}\" x{temporal.pattern.occurrences} "
            f"({temporal.pattern.direction})"
        )
    if temporal.prediction is not None:
        lines.append(f"next expected: {temporal.prediction.predicted_time.isoformat()}")
    return tuple(lines)


def reasoning_report(
    question: str,
    *,
    passages: Iterable[RetrievedPassage] = (),
    relation_facts: Iterable[RelationFact] = (),
    plan: AnswerPlan | None = None,
    decision: AnswerDecision | None = None,
    belief: AnswerBelief | None = None,
    inferred: Iterable[InferredRelation] = (),
    connections: Iterable[ConceptConnection] = (),
    causal_paths: Iterable[CausalExplanation] = (),
    temporal: AnswerTemporal | None = None,
    conflicts: Iterable[RelationConflict] = (),
    confidence: AnswerConfidence | None = None,
) -> ReasoningReport:
    """Assemble the ordered reasoning trace for one answer from its own data.

    Every argument is *optional*: a step with nothing to show is still present but
    empty, so the trace reads honestly ("nothing was derived here") rather than
    quietly omitting the fact that a kind of reasoning contributed nothing.
    """
    steps = (
        ReasoningStep(SOURCE_EVIDENCE, _evidence_lines(passages)),
        ReasoningStep(SOURCE_STORED, tuple(fact.describe() for fact in relation_facts)),
        ReasoningStep(
            SOURCE_PLAN, _plan_lines(plan), derived=True
        ),
        ReasoningStep(
            SOURCE_DECISION, _decision_lines(decision), derived=True
        ),
        ReasoningStep(
            SOURCE_BELIEF, _belief_lines(belief), derived=True
        ),
        ReasoningStep(
            SOURCE_INFERRED, tuple(item.describe() for item in inferred), derived=True
        ),
        ReasoningStep(
            SOURCE_CONNECTIONS,
            tuple(item.describe() for item in connections),
            derived=True,
        ),
        ReasoningStep(
            SOURCE_CAUSAL, tuple(item.describe() for item in causal_paths), derived=True
        ),
        ReasoningStep(SOURCE_TEMPORAL, _temporal_lines(temporal), derived=True),
        ReasoningStep(SOURCE_CONFLICTS, tuple(item.describe() for item in conflicts)),
        ReasoningStep(SOURCE_CONFIDENCE, _confidence_lines(confidence)),
    )
    # Keep the declared order even if a caller passed the arguments out of order.
    ordered = tuple(sorted(steps, key=lambda step: STEP_ORDER.index(step.label)))
    return ReasoningReport(question=question, steps=ordered)


@dataclass(frozen=True)
class TraceJudgement:
    """One held-out expectation about a reasoning trace.

    The scenario names which *kinds* of reasoning contributed (as :data:`STEP_ORDER`
    labels) and, for each, whether it must carry content (a positive control) or must
    be empty (a negative control). ``expected_stored``/``expected_derived`` then pin
    the two counts a reader can check by eye -- how many steps are stored claims and
    how many are derivations -- so a trace that mislabels a derivation as a stored
    claim, or drops an empty step to look fuller than it is, is caught.
    """

    name: str
    filled: tuple[str, ...]
    expected_stored: int
    expected_derived: int
    note: str = ""

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("A trace judgement needs a name.")
        for label in self.filled:
            if label not in STEP_ORDER:
                raise ValueError(f"Unknown step label in a trace judgement: {label}")
        if self.expected_stored < 0 or self.expected_derived < 0:
            raise ValueError("Expected step counts cannot be negative.")


#: The held-out trace catalogue -- hand-written with **negative controls**: the
#: "nothing contributed" case must render empty, an evidence-only case must keep
#: every other step present-but-empty (never dropped), and a mixed case must label
#: its derivations as derived rather than as stored claims.
TRACE_JUDGEMENTS: tuple[TraceJudgement, ...] = (
    TraceJudgement("nothing contributed", (), 0, 0, "an honest empty trace"),
    TraceJudgement(
        "evidence only", (SOURCE_EVIDENCE,), 1, 0, "no derivation, none invented"
    ),
    TraceJudgement(
        "one stored claim", (SOURCE_STORED,), 1, 0, "a stored relation is not a derivation"
    ),
    TraceJudgement(
        "one plan",
        (SOURCE_PLAN,),
        0,
        1,
        "a corpus procedure ordered by the planner is a derivation",
    ),
    TraceJudgement(
        "one recommendation",
        (SOURCE_DECISION,),
        0,
        1,
        "the graph's options ranked by the decision engine are a derivation",
    ),
    TraceJudgement(
        "one belief",
        (SOURCE_BELIEF,),
        0,
        1,
        "the graph's edges weighed by the belief engine are a derivation",
    ),
    TraceJudgement(
        "one walked connection",
        (SOURCE_CONNECTIONS,),
        0,
        1,
        "a walked chain is a derivation",
    ),
    TraceJudgement(
        "one causal path", (SOURCE_CAUSAL,), 0, 1, "a causal chain is a derivation"
    ),
    TraceJudgement(
        "one timeline",
        (SOURCE_TEMPORAL,),
        0,
        1,
        "a conversation ordered by the temporal reasoner is a derivation",
    ),
    TraceJudgement(
        "evidence and a causal path",
        (SOURCE_EVIDENCE, SOURCE_CAUSAL),
        1,
        1,
        "both a read and a derivation",
    ),
    TraceJudgement(
        "the full trace",
        (
            SOURCE_EVIDENCE,
            SOURCE_STORED,
            SOURCE_PLAN,
            SOURCE_DECISION,
            SOURCE_BELIEF,
            SOURCE_INFERRED,
            SOURCE_CONNECTIONS,
            SOURCE_CAUSAL,
            SOURCE_TEMPORAL,
            SOURCE_CONFLICTS,
            SOURCE_CONFIDENCE,
        ),
        4,
        7,
        "every step filled, four stored and seven derived",
    ),
)


def _judgement_report(judgement: TraceJudgement) -> ReasoningReport:
    """Build a trace whose *filled* steps are exactly the judgement's labels."""
    passages: tuple[RetrievedPassage, ...] = ()
    facts: tuple[RelationFact, ...] = ()
    plan: AnswerPlan | None = None
    decision: AnswerDecision | None = None
    belief: AnswerBelief | None = None
    inferred: tuple[InferredRelation, ...] = ()
    connections: tuple[ConceptConnection, ...] = ()
    causal: tuple[CausalExplanation, ...] = ()
    temporal: AnswerTemporal | None = None
    conflicts: tuple[RelationConflict, ...] = ()
    confidence: AnswerConfidence | None = None
    if SOURCE_EVIDENCE in judgement.filled:
        passages = (
            RetrievedPassage(
                doc_id="doc#1",
                title="doc",
                text="Permeability measures flow.",
                source="src",
                score=0.9,
                signal="lexical",
            ),
        )
    if SOURCE_STORED in judgement.filled:
        facts = (RelationFact("porosity", "increases", "permeability", 0.6),)
    if SOURCE_PLAN in judgement.filled:
        plan = AnswerPlan(topic="Reservoir rock properties", source="reservoir_rock_properties", steps=("Measure porosity.", "Compare with logs."))
    if SOURCE_DECISION in judgement.filled:
        decision = AnswerDecision(
            goal="permeability",
            direction="increase",
            selected="acid_stimulation",
            options=(("acid_stimulation", "increases", 0.7), ("porosity", "increases", 0.6)),
            sentences=("Selected: acid_stimulation.",),
        )
    if SOURCE_BELIEF in judgement.filled:
        belief = AnswerBelief(
            statement="does acid stimulation increase permeability?",
            confidence=0.71,
            status="provisional",
            supporting=("acid_stimulation —increases→ permeability (confidence 0.70)",),
            contradicting=(),
        )
    if SOURCE_INFERRED in judgement.filled:
        inferred = (InferredRelation("a", "is_a", "b", 0.5, ("a —is_a→ b",)),)
    if SOURCE_CONNECTIONS in judgement.filled:
        connections = (
            ConceptConnection(
                source="a",
                target="b",
                hops=2,
                confidence=0.5,
                steps=("a —r1→ c", "c —r2→ b"),
            ),
        )
    if SOURCE_CAUSAL in judgement.filled:
        causal = (
            CausalExplanation(
                cause="a",
                effect="b",
                hops=1,
                confidence=0.5,
                nodes=("a", "b"),
                mechanisms=("drives",),
                sentence="A drives B.",
            ),
        )
    if SOURCE_TEMPORAL in judgement.filled:
        from datetime import UTC, datetime, timedelta

        from core.answer_temporal import TimelineEntry

        base = datetime(2026, 1, 1, tzinfo=UTC)
        temporal = AnswerTemporal(
            entries=(
                TimelineEntry("Q", "what changed?", base),
                TimelineEntry("A", "an answer", base + timedelta(seconds=5)),
            ),
            relation="before",
            change="the discussion moved on",
            pattern=None,
            prediction=None,
            language="en",
        )
    if SOURCE_CONFLICTS in judgement.filled:
        conflicts = (RelationConflict("a", "increases", "b", "reduces", 0.7),)
    if SOURCE_CONFIDENCE in judgement.filled:
        confidence = AnswerConfidence(
            evidence=0.8,
            graph_support=0.5,
            conflict_penalty=0.0,
            final=0.83,
            reasons=(),
        )
    return reasoning_report(
        "judgement",
        passages=passages,
        relation_facts=facts,
        plan=plan,
        decision=decision,
        belief=belief,
        inferred=inferred,
        connections=connections,
        causal_paths=causal,
        temporal=temporal,
        conflicts=conflicts,
        confidence=confidence,
    )


def reasoning_trace_accuracy(
    judgements: tuple[TraceJudgement, ...] = TRACE_JUDGEMENTS,
) -> float:
    """The share of held-out trace judgements the report builder gets right.

    A judgement counts only when the trace matches it on **all** of: the set of
    steps that carry content, the stored/derived split, the fixed step order, and
    the honesty rule that a contributing step's content is non-empty. So a trace
    that invents a step, drops an empty one, mislabels a derivation as stored, or
    renders in the wrong order fails as surely as one that omits real evidence.
    Returns ``0.0`` for an empty catalogue.
    """
    catalogue = tuple(judgements)
    if not catalogue:
        return 0.0
    correct = 0
    for judgement in catalogue:
        report = _judgement_report(judgement)
        filled = tuple(step.label for step in report.steps if step.lines)
        stored = sum(1 for step in report.steps if step.lines and not step.derived)
        derived = len(report.derived_steps)
        in_order = tuple(step.label for step in report.steps) == STEP_ORDER
        # A negative control ("nothing contributed") must also read as empty.
        honesty = report.is_empty == (not judgement.filled)
        if (
            set(filled) == set(judgement.filled)
            and stored == judgement.expected_stored
            and derived == judgement.expected_derived
            and in_order
            and honesty
        ):
            correct += 1
    return correct / len(catalogue)


__all__ = [
    "SOURCE_BELIEF",
    "SOURCE_CAUSAL",
    "SOURCE_CONFIDENCE",
    "SOURCE_CONFLICTS",
    "SOURCE_CONNECTIONS",
    "SOURCE_DECISION",
    "SOURCE_EVIDENCE",
    "SOURCE_INFERRED",
    "SOURCE_PLAN",
    "SOURCE_STORED",
    "SOURCE_TEMPORAL",
    "STEP_ORDER",
    "TRACE_JUDGEMENTS",
    "ReasoningReport",
    "ReasoningStep",
    "TraceJudgement",
    "reasoning_report",
    "reasoning_trace_accuracy",
]
