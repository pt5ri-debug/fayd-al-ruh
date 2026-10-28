"""Surface the *belief* the graph supports for a question that asks whether a claim holds.

The project carries a whole belief layer -- ``Belief``, ``BeliefRevisionEngine``,
``ConfidenceModel``, ``ContradictionDetector`` -- but it was reachable only from
the ``CognitiveController`` and the evaluation scenarios, never from an answer. So
a question like *"is acid stimulation increasing permeability?"* got a definition
back, never the evidence-weighed verdict the knowledge graph already supports. That
is the belief half of Gap 7 of ``docs/REASONING_ASSESSMENT.md``.

This module reads the graph's own *typed* edges between the concepts the question
names, turns each into an ``Evidence`` record, and runs them through the project's
real ``BeliefRevisionEngine`` -- so the confidence and the status that reach the
reader are the ones the belief engine actually produced, with its own
evidence-weighted posterior and its own contradiction detection. It is
**read-only** and **honest by construction**: it only builds a belief when the
question actually *asks whether a claim holds* (a definitional "what is
permeability?" gets none), it never invents evidence the graph did not state, it
attributes the belief to the edges it rests on, and it returns ``None`` rather than
a guess when the question names fewer than two concepts or no edge connects them.

The belief is a *reported* artifact, not a commitment: nothing here changes the
knowledge graph.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from core.belief import Belief
from core.belief_revision import BeliefRevisionEngine
from core.evidence import Evidence

if TYPE_CHECKING:
    from core.ontology import Ontology

#: The openers that make a question *evaluative* -- a yes/no question about whether
#: a claim holds, in either language. The test is **anchored to the question's
#: opening**, so a definitional "what is permeability?" is never read as evaluative
#: and never drags a belief into a definitional answer. A question that merely
#: *contains* "is" (e.g. "what is the graph?") is not evaluative.
EVALUATIVE_OPENERS: tuple[str, ...] = (
    "is ",
    "are ",
    "was ",
    "were ",
    "does ",
    "do ",
    "did ",
    "can ",
    "could ",
    "will ",
    "would ",
    "should ",
    "هل ",
)

#: Phrases that make a question evaluative anywhere in it, because they are
#: unambiguous requests for a verdict.
EVALUATIVE_PHRASES: tuple[str, ...] = (
    "is it true",
    "is it the case",
    "is it correct",
    "is it accurate",
    "صحيح أن",
    "هل صحيح",
)

#: The source every evidence record is attributed to: the knowledge graph itself,
#: so a reader can tell a stored edge from an external source.
GRAPH_SOURCE = "knowledge_graph"


@dataclass(frozen=True)
class AnswerBelief:
    """The evidence-weighed verdict the graph supports for a question."""

    statement: str
    """The claim under test, taken from the question itself."""

    confidence: float
    """The belief engine's revised confidence in the claim, in ``[0, 1]``."""

    status: str
    """The lifecycle status the belief engine assigned
    (``accepted``/``provisional``/``suspended``/``rejected``)."""

    supporting: tuple[str, ...]
    """The graph edges that support the claim, as readable lines."""

    contradicting: tuple[str, ...]
    """The graph edges that oppose the claim, as readable lines. Empty when the
    graph is consistent on this claim."""

    def describe(self) -> str:
        """The verdict as a heading followed by the evidence it rests on."""
        head = (
            f"Evidence-weighed belief in \"{self.statement}\": "
            f"confidence {self.confidence:.2f} ({self.status})."
        )
        lines = [head]
        if self.supporting:
            lines.append("  Supporting the claim:")
            lines.extend(f"    - {line}" for line in self.supporting)
        if self.contradicting:
            lines.append("  Contradicting the claim:")
            lines.extend(f"    - {line}" for line in self.contradicting)
        if not self.supporting and not self.contradicting:
            lines.append("  No graph edge connects the concepts in the claim.")
        return "\n".join(lines)


def is_evaluative(question: str) -> bool:
    """Whether a question asks whether a claim holds, so a belief is warranted.

    The test is anchored to the question's opening (or an unambiguous verdict
    phrase anywhere), so a definitional question is never read as evaluative.
    """
    lowered = question.strip().casefold()
    if not lowered:
        return False
    if any(phrase in lowered for phrase in EVALUATIVE_PHRASES):
        return True
    return any(lowered.startswith(opener) for opener in EVALUATIVE_OPENERS)


def _named_concepts(ontology: Ontology, question: str) -> list[str]:
    """The ids of the concepts the question names, first mention first."""
    ids: list[str] = []
    seen: set[str] = set()
    for _start, _end, concept in ontology.match_concepts(question):
        if concept.id in seen:
            continue
        seen.add(concept.id)
        ids.append(concept.id)
    return ids


def _evidence(
    ontology: Ontology, concept_ids: list[str]
) -> tuple[list[Evidence], tuple[str, ...]]:
    """One ``Evidence`` record per graph edge between the question's concepts.

    Only edges whose **both** endpoints are concepts the question names are read,
    so the claim is about the concepts the reader asked about. Each record carries
    the edge's relation type in its metadata, which is what lets the belief
    engine's ``ContradictionDetector`` recognise two *antonymous* relations on the
    same ordered pair as a genuine disagreement. Returns the records and the
    readable lines, strongest first then by labels.
    """
    wanted = set(concept_ids)
    labels = {concept_id: concept.label for concept_id, concept in ontology.concepts.items()}
    rows: list[tuple[float, str, Evidence]] = []
    for relation in ontology.relations.values():
        if relation.source_id not in wanted or relation.target_id not in wanted:
            continue
        source = labels.get(relation.source_id)
        target = labels.get(relation.target_id)
        if source is None or target is None:
            continue
        line = f"{source} —{relation.relation_type}→ {target} (confidence {relation.confidence:.2f})"
        record = Evidence(
            description=line,
            source=GRAPH_SOURCE,
            supports=True,
            strength=relation.confidence,
            data={
                "relation_type": relation.relation_type,
                "source_id": relation.source_id,
                "target_id": relation.target_id,
            },
        )
        rows.append((relation.confidence, line, record))
    rows.sort(key=lambda row: (-row[0], row[1]))
    return [record for _confidence, _line, record in rows], tuple(line for _confidence, line, _record in rows)


def _decide(
    ontology: Ontology, question: str, *, prior: float = 0.5
) -> AnswerBelief | None:
    """Build the belief, or ``None`` when the graph cannot speak to the claim."""
    if ontology is None or not question.strip():
        return None
    concept_ids = _named_concepts(ontology, question)
    if len(concept_ids) < 2:
        return None
    evidence, lines = _evidence(ontology, concept_ids)
    if not evidence:
        return None

    belief = Belief(statement=question.strip(), confidence=prior)
    revision = BeliefRevisionEngine().revise(belief, evidence)
    supporting = tuple(
        line for line, record in zip(lines, evidence, strict=True) if record.supports
    )
    contradicting = tuple(
        line for line, record in zip(lines, evidence, strict=True) if not record.supports
    )
    return AnswerBelief(
        statement=belief.statement,
        confidence=revision.revised.confidence,
        status=belief.status.value,
        supporting=supporting,
        contradicting=contradicting,
    )


def answer_belief(
    question: str,
    *,
    ontology: Ontology | None = None,
    prior: float = 0.5,
) -> AnswerBelief | None:
    """The evidence-weighed belief for a question, or ``None`` when none is warranted.

    A belief is produced only when the question is *evaluative*
    (``is_evaluative``) **and** the graph holds at least one edge between the
    concepts it names. Otherwise ``None`` -- a definitional or unrelated question
    gets no belief rather than a guess.
    """
    if not 0.0 <= prior <= 1.0:
        raise ValueError("Prior confidence must be between 0 and 1.")
    if not is_evaluative(question):
        return None
    if ontology is None:
        from core.cognitive_controller import CognitiveController

        ontology = CognitiveController().ontology
    return _decide(ontology, question, prior=prior)


@dataclass(frozen=True)
class BeliefJudgement:
    """One held-out expectation about the belief bridge.

    ``kind`` selects what is checked: ``"status"`` that an evaluative question
    reaches the expected lifecycle status (the graph's seeded confidence is modest
    and revisable, so a well-supported claim is honestly *provisional* rather than
    *accepted*), ``"support"`` that the named edges are the ones the belief rests
    on, and ``"no-belief"`` that a question warrants no belief at all (a negative
    control: a definitional question, or an evaluative question the graph cannot
    speak to).
    """

    name: str
    kind: str
    question: str = ""
    expected: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("A belief judgement needs a name.")
        if self.kind not in ("status", "support", "no-belief"):
            raise ValueError(f"Unknown belief judgement kind: {self.kind}")


#: The held-out belief catalogue -- hand-written with **negative controls**: a
#: definitional question, a plain statement, and an evaluative question the graph
#: cannot speak to must all yield no belief.
BELIEF_JUDGEMENTS: tuple[BeliefJudgement, ...] = (
    BeliefJudgement(
        "a definitional question gets no belief",
        "no-belief",
        question="what is permeability?",
    ),
    BeliefJudgement(
        "a statement gets no belief",
        "no-belief",
        question="permeability measures flow.",
    ),
    BeliefJudgement(
        "an evaluative question naming one concept gets no belief",
        "no-belief",
        question="is permeability high?",
    ),
    BeliefJudgement(
        "an evaluative question the graph cannot speak to gets no belief",
        "no-belief",
        question="does porosity increase water cut?",
    ),
    BeliefJudgement(
        "a claim the graph supports is honestly provisional",
        "status",
        question="does acid stimulation increase permeability?",
        expected=("provisional",),
    ),
    BeliefJudgement(
        "the belief rests on the graph's own edges",
        "support",
        question="does acid stimulation increase permeability?",
        expected=("acid_stimulation —increases→ permeability",),
    ),
)


def belief_accuracy(
    judgements: tuple[BeliefJudgement, ...] = BELIEF_JUDGEMENTS,
) -> float:
    """The share of held-out belief judgements the bridge gets right.

    A judgement counts only when the bridge does exactly what it must: reach the
    expected status, rest on the expected edges, or produce no belief for a
    question that warrants none. So a bridge that invents evidence, or answers a
    definitional question, fails the relevant judgement. Returns ``0.0`` for an
    empty catalogue.
    """
    catalogue = tuple(judgements)
    if not catalogue:
        return 0.0
    correct = 0
    for judgement in catalogue:
        belief = answer_belief(judgement.question)
        if judgement.kind == "no-belief":
            ok = belief is None
        elif belief is None:
            ok = False
        elif judgement.kind == "status":
            ok = belief.status in judgement.expected
        else:  # support
            ok = all(
                any(line.startswith(expected) for line in belief.supporting)
                for expected in judgement.expected
            )
        if ok:
            correct += 1
    return correct / len(catalogue)


__all__ = [
    "BELIEF_JUDGEMENTS",
    "EVALUATIVE_OPENERS",
    "EVALUATIVE_PHRASES",
    "GRAPH_SOURCE",
    "AnswerBelief",
    "BeliefJudgement",
    "answer_belief",
    "belief_accuracy",
    "is_evaluative",
]
