"""Surface the *recommendation* the graph states for a question that asks what to do.

The project carries a whole decision layer -- ``DecisionEngine``, ``UtilityModel``,
``RiskModel``, ``DecisionExplainer`` -- but it was reachable only from the
``CognitiveController`` and the evaluation scenarios, never from an answer. So a
question like *"what should I do to increase permeability?"* got a definition back,
never the ranked recommendation the knowledge graph already states (``acid
stimulation --increases--> permeability``, ``hydraulic fracturing --increases-->
permeability``). That is Gap 7 of ``docs/REASONING_ASSESSMENT.md``, the decision
half.

This module reads the graph's own *typed* edges into the concept the question is
about, turns each into an ``Action``, and runs them through the project's real
``DecisionEngine`` -- so the recommendation that reaches the reader is the one the
decision engine actually produced, with its own utility, risk, and score. It is
**read-only** and **honest by construction**: it only builds a recommendation when
the question actually *asks what to do* (a definitional "what is permeability?"
gets none), it never invents an option the graph did not state, it only offers
edges whose relation type matches the direction the question asks for, and it
returns ``None`` rather than a guess when no option matches.

The recommendation is a *reported* artifact, not an action: nothing here executes
anything.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from core.action import Action
from core.decision_engine import DecisionEngine
from core.decision_explainer import DecisionExplainer
from core.goal import Goal
from core.state import State

if TYPE_CHECKING:
    from core.ontology import Ontology

#: The phrases that make a question *decisional* -- a request for what to do, in
#: either language. A recommendation is only produced for such a question, so a
#: definitional question never drags options into its answer.
DECISION_CUES: tuple[str, ...] = (
    "what should i",
    "what should we",
    "what do i",
    "what can i",
    "what would you",
    "which should i",
    "which is best",
    "which is better",
    "what is best",
    "what is the best",
    "what's the best",
    "do you recommend",
    "what do you recommend",
    "your recommendation",
    "best way to",
    "best approach",
    "recommend",
    "advise",
    "advice",
    "ماذا أفعل",
    "ماذا افعل",
    "ما الذي أفعل",
    "ما الأفضل",
    "ما هو الأفضل",
    "أيها أفضل",
    "بماذا تنصح",
    "ما توصيتك",
    "تنصح",
    "توصية",
)

#: The relation types each *direction* a question can ask for accepts, so an
#: option is only offered when the graph says it moves the goal the way the user
#: asked. ``increases`` is not offered for "how do I reduce ..." and vice versa.
DIRECTION_RELATIONS: dict[str, frozenset[str]] = {
    "increase": frozenset({"increases", "improves", "restores", "supports"}),
    "reduce": frozenset({"reduces", "limits"}),
    "prevent": frozenset({"prevents"}),
    "maintain": frozenset({"maintains", "stabilizes", "protects"}),
    "control": frozenset({"controls", "governs", "manages"}),
}

#: The cue words that select a direction, in the question's own words.
DIRECTION_CUES: dict[str, tuple[str, ...]] = {
    "increase": ("increase", "improve", "maximize", "maximise", "raise", "boost", "enhance"),
    "reduce": ("reduce", "lower", "decrease", "minimize", "minimise", "cut", "drop"),
    "prevent": ("prevent", "avoid", "stop", "eliminate"),
    "maintain": ("maintain", "keep", "sustain", "preserve"),
    "control": ("control", "manage", "govern"),
}

#: Every relation type a recommendation may be built from: the union of the
#: directions above. A question that asks what to do *without* naming a direction
#: is offered these, and the recommendation is labelled as the strongest stated
#: option rather than as the best way to reach a goal the user never named.
ACTIONABLE_RELATIONS: frozenset[str] = frozenset().union(*DIRECTION_RELATIONS.values())

#: How many options one recommendation states. A concept the graph links many
#: actions to would otherwise flood the answer.
DEFAULT_DECISION_LIMIT = 6


@dataclass(frozen=True)
class AnswerDecision:
    """The ranked recommendation the graph states for a question that asks what to do."""

    goal: str
    """The concept the question is about, so the recommendation is attributed to it."""

    direction: str
    """The direction the question asked for (a key of :data:`DIRECTION_RELATIONS`),
    or ``""`` when the question named none -- so a reader can tell a ranked
    recommendation from the strongest stated option."""

    selected: str
    """The option the decision engine chose: the source concept of the winning edge."""

    options: tuple[tuple[str, str, float], ...]
    """Every option, ranked: ``(option, relation, confidence)``, strongest first."""

    sentences: tuple[str, ...]
    """The decision engine's own explanation: the selection and each option's scores."""

    def describe(self) -> str:
        """The recommendation as a heading followed by its ranked options."""
        if self.direction:
            head = f"The graph's options to {self.direction} {self.goal} (ranked by the decision engine):"
        else:
            head = f"The options the graph states for {self.goal} (ranked by the decision engine):"
        body = "\n".join(
            f"  - {option} ({relation}, confidence {confidence:.2f})"
            for option, relation, confidence in self.options
        )
        return f"{head}\n{body}"


def is_decisional(question: str) -> bool:
    """Whether a question asks *what to do*, so a recommendation is warranted.

    Whole-phrase and case-insensitive: the cue must appear in the question, so a
    definitional question ("what is permeability?") is never read as decisional.
    """
    lowered = question.casefold()
    return any(cue in lowered for cue in DECISION_CUES)


def direction_of(question: str) -> str:
    """The direction a question asks for, or ``""`` when it names none.

    Scanned in a fixed order (increase, reduce, prevent, maintain, control), so the
    same question always yields the same direction; a question that names two
    directions resolves to the first, never to a guess.
    """
    lowered = question.casefold()
    for direction, cues in DIRECTION_CUES.items():
        if any(cue in lowered for cue in cues):
            return direction
    return ""


def _named_concepts(ontology: Ontology, question: str) -> list[str]:
    """The labels of the concepts the question names, first mention first."""
    labels: list[str] = []
    seen: set[str] = set()
    for _start, _end, concept in ontology.match_concepts(question):
        if concept.id in seen:
            continue
        seen.add(concept.id)
        labels.append(concept.label)
    return labels


def _inbound_edges(
    ontology: Ontology, concept_id: str, relations: frozenset[str]
) -> list[tuple[str, str, float]]:
    """The graph's edges into ``concept_id`` whose type is in ``relations``.

    Returns ``(source label, relation type, confidence)``, strongest first then by
    source label, so the same graph always yields the same order.
    """
    found: list[tuple[str, str, float]] = []
    for relation in ontology.relations.values():
        if relation.target_id != concept_id or relation.relation_type not in relations:
            continue
        source = ontology.concepts.get(relation.source_id)
        if source is None or source.label == concept_id:
            continue
        found.append((source.label, relation.relation_type, relation.confidence))
    found.sort(key=lambda item: (-item[2], item[0], item[1]))
    return found


@dataclass(frozen=True)
class _Option:
    """One candidate the graph states: a source concept, its strongest edge, and
    every goal that edge advances."""

    name: str
    relation: str
    confidence: float
    effects: frozenset[str]

    def action(self) -> Action:
        """The declarative action the decision engine ranks: it adds the goals this
        option advances, at a cost of ``1 - confidence`` (a better-attested option
        is cheaper)."""
        return Action(
            name=self.name,
            add_effects=self.effects,
            cost=round(max(0.0, 1.0 - self.confidence), 6),
        )


def _options(
    ontology: Ontology, goals: dict[str, str], relations: frozenset[str]
) -> tuple[_Option, ...]:
    """Every option the graph states for the question's goals.

    A source concept the graph links to more than one of the goals becomes **one**
    option that advances all of them -- that is the multi-objective case the
    utility model exists for. The relation and confidence of record are the
    strongest edge's, so the confidence shown next to the option is the one its
    cost was built from.
    """
    merged: dict[str, _Option] = {}
    for concept_id, label in goals.items():
        for source, relation, confidence in _inbound_edges(ontology, concept_id, relations):
            current = merged.get(source)
            if current is None:
                merged[source] = _Option(source, relation, confidence, frozenset({label}))
                continue
            strongest = (
                (relation, confidence)
                if confidence > current.confidence
                or (confidence == current.confidence and relation < current.relation)
                else (current.relation, current.confidence)
            )
            merged[source] = _Option(
                source, strongest[0], strongest[1], current.effects | {label}
            )
    return tuple(merged[name] for name in sorted(merged))


def _decide(
    ontology: Ontology, question: str, *, limit: int = DEFAULT_DECISION_LIMIT
) -> AnswerDecision | None:
    """Build the recommendation, or ``None`` when the graph states no option."""
    if ontology is None or not question.strip():
        return None
    labels = _named_concepts(ontology, question)
    if not labels:
        return None
    direction = direction_of(question)
    relations = DIRECTION_RELATIONS.get(direction, ACTIONABLE_RELATIONS)

    # The concept id of every label the question names, so the options are looked
    # up on the graph's own nodes.
    goals: dict[str, str] = {
        concept.id: concept.label
        for concept in ontology.concepts.values()
        if concept.label in labels
    }
    options = _options(ontology, goals, relations)
    if not options:
        return None

    goal = Goal(description=question.strip(), desired_facts=frozenset(labels))
    trace = DecisionEngine().decide(State(), goal, [option.action() for option in options])
    if trace.selected is None:
        return None

    by_name = {option.name: option for option in options}
    ranked = tuple(
        (
            candidate.action.name,
            by_name[candidate.action.name].relation,
            by_name[candidate.action.name].confidence,
        )
        for candidate in trace.candidates
        if candidate.action.name in by_name
    )[:limit]
    return AnswerDecision(
        goal=" and ".join(labels),
        direction=direction,
        selected=trace.selected.action.name,
        options=ranked,
        sentences=DecisionExplainer().explain(trace),
    )


def answer_decision(
    question: str,
    *,
    ontology: Ontology | None = None,
    limit: int = DEFAULT_DECISION_LIMIT,
) -> AnswerDecision | None:
    """The recommendation for a question, or ``None`` when none is warranted.

    A recommendation is produced only when the question is *decisional*
    (``is_decisional``) **and** the graph states at least one option whose relation
    matches the direction asked for. Otherwise ``None`` -- a definitional or
    unrelated question gets no recommendation rather than a guess.
    """
    if limit < 1:
        raise ValueError("Decision limit must be at least 1.")
    if not is_decisional(question):
        return None
    if ontology is None:
        from core.cognitive_controller import CognitiveController

        ontology = CognitiveController().ontology
    return _decide(ontology, question, limit=limit)


@dataclass(frozen=True)
class DecisionJudgement:
    """One held-out expectation about the decision bridge.

    ``kind`` selects what is checked: ``"recommend"`` that a decisional question
    selects ``expected``, ``"offers"`` that it offers the options in ``expected``,
    and ``"no-decision"`` that a question yields no recommendation at all (a
    negative control: a definitional question, or a decisional question the graph
    has no matching option for).
    """

    name: str
    kind: str
    question: str = ""
    expected: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("A decision judgement needs a name.")
        if self.kind not in ("recommend", "offers", "no-decision"):
            raise ValueError(f"Unknown decision judgement kind: {self.kind}")


#: The held-out decision catalogue -- hand-written with **negative controls**: a
#: definitional question and a decisional question the graph cannot answer must
#: both yield no recommendation, and a direction the graph holds no edge for must
#: not be answered with an edge that runs the other way.
DECISION_JUDGEMENTS: tuple[DecisionJudgement, ...] = (
    DecisionJudgement(
        "a definitional question gets no recommendation",
        "no-decision",
        question="what is permeability?",
    ),
    DecisionJudgement(
        "an unrelated question gets no recommendation",
        "no-decision",
        question="what should I do about the weather in Paris?",
    ),
    DecisionJudgement(
        "a decisional question the graph cannot answer gets none",
        "no-decision",
        question="what should I do to reduce permeability?",
    ),
    DecisionJudgement(
        "the graph's options to increase permeability are offered",
        "offers",
        question="what should I do to increase permeability?",
        expected=("acid_stimulation", "hydraulic_fracturing", "porosity"),
    ),
    DecisionJudgement(
        "a better-attested option is selected",
        "recommend",
        question="what should I do to increase permeability?",
        expected=("acid_stimulation",),
    ),
    DecisionJudgement(
        "the direction asked for decides which options are offered",
        "recommend",
        question="what should I do to maintain reservoir pressure?",
        expected=("drive_mechanism",),
    ),
)


def decision_recommendation_accuracy(
    judgements: tuple[DecisionJudgement, ...] = DECISION_JUDGEMENTS,
) -> float:
    """The share of held-out decision judgements the bridge gets right.

    A judgement counts only when the bridge does exactly what it must: select the
    expected option, offer exactly the expected options, or produce no
    recommendation for a question that warrants none. So a bridge that invents an
    option, ignores the direction asked for, or answers a definitional question
    fails the relevant judgement. Returns ``0.0`` for an empty catalogue.
    """
    catalogue = tuple(judgements)
    if not catalogue:
        return 0.0
    correct = 0
    for judgement in catalogue:
        decision = answer_decision(judgement.question)
        if judgement.kind == "no-decision":
            ok = decision is None
        elif decision is None:
            ok = False
        elif judgement.kind == "recommend":
            ok = decision.selected in judgement.expected
        else:  # offers
            offered = {option for option, _relation, _confidence in decision.options}
            ok = offered == set(judgement.expected)
        if ok:
            correct += 1
    return correct / len(catalogue)


__all__ = [
    "ACTIONABLE_RELATIONS",
    "DECISION_CUES",
    "DECISION_JUDGEMENTS",
    "DEFAULT_DECISION_LIMIT",
    "DIRECTION_CUES",
    "DIRECTION_RELATIONS",
    "AnswerDecision",
    "DecisionJudgement",
    "answer_decision",
    "decision_recommendation_accuracy",
    "direction_of",
    "is_decisional",
]
