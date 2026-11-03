"""One facade over the project's reasoning engines, so callers stop reaching into twenty modules.

The reasoning layer is rich but sprawling: deduction, induction, abduction,
causal walks, analogy, temporal ordering, planning, decision, belief, entailment,
multi-hop connection, contradiction analysis, and conclusion verification each
live in their own module with their own call shape. A caller that wants "reason
about this question" has to know which engine to reach for and how to call it --
and the reasoning bridge that already exists (``core/answer_*``) is spread across
a dozen modules.

:class:`ReasoningAPI` is the one entry point. It **delegates** -- every method
calls the existing engine and returns that engine's own object, unchanged -- so it
adds **no inference of its own** and cannot drift from the engines. It holds **no
state** beyond the graph it was built over, so two calls are independent. And it
works with **no language model**: every method is deterministic over the graph and
the question's own words.

The uniform return type is :class:`ReasoningResult`: the conclusion, the engine's
own output object, the reasoning steps, the confidence, the evidence, an optional
:class:`~core.conclusion_verification.VerificationResult`, and the honest
limitations. :meth:`ReasoningAPI.reason` is the top-level call -- it decomposes the
question (:mod:`core.problem_decomposition`) and routes each part to the engine its
kind warrants, joining the results.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import TYPE_CHECKING

from core.answer_analogy import AnswerAnalogy, answer_analogy
from core.answer_belief import AnswerBelief, answer_belief
from core.answer_decision import AnswerDecision, answer_decision
from core.answer_plan import AnswerPlan, answer_plan
from core.answer_temporal import AnswerTemporal, answer_temporal
from core.causal_paths import CausalExplanation, causal_explanations
from core.conclusion_verification import VerificationResult, verify_conclusion
from core.hypothesis_reasoning import (
    HypothesisComparison,
    compare_hypotheses,
)
from core.ontology import Ontology
from core.problem_decomposition import (
    KIND_COMPARISON,
    KIND_LOOKUP,
    KIND_PROCEDURE,
    KIND_RECOMMENDATION,
    KIND_TIMELINE,
    KIND_VERDICT,
    decompose,
)
from core.relation_connections import ConceptConnection, concept_connections
from core.relation_consistency import relation_conflicts
from core.relation_facts import RelationFact, relation_facts
from core.relation_inference import InferredRelation, inferred_relations
from core.relation_paths import ConnectionPath, find_connection

if TYPE_CHECKING:
    from core.conversation import Conversation

#: The kinds a :class:`ReasoningResult` can carry (one per engine, plus the
#: composite and the three classical methods).
KIND_COMPOSITE = "composite"
KIND_DEDUCTION = "deduction"
KIND_INDUCTION = "induction"
KIND_ABDUCTION = "abduction"
KIND_VERIFICATION = "verification"
KIND_HYPOTHESES = "hypotheses"

#: The analogy engine's ``score`` is the count of aligned relations; this is the
#: count treated as "full" when the facade normalises it to a bounded confidence.
_ANALOGY_SCORE_TARGET = 20.0


def _bounded(value: float) -> float:
    """Clamp a value into ``[0, 1]`` -- a facade never reports an out-of-range confidence."""
    if value != value:  # NaN
        return 0.0
    return max(0.0, min(1.0, float(value)))


@dataclass(frozen=True)
class ReasoningResult:
    """A uniform reasoning outcome: the conclusion, its output, its trace, and its limits.

    ``output`` is the engine's **own** object (an :class:`~core.answer_plan.AnswerPlan`,
    an :class:`~core.answer_decision.AnswerDecision`, ...), passed through unchanged,
    so a caller that wants the detail has it and a caller that wants a uniform shape
    has the rest of the fields. ``parts`` nests the per-part results of a composite
    :meth:`ReasoningAPI.reason` call.
    """

    kind: str
    question: str
    conclusion: str
    output: object | None = None
    trace: tuple[str, ...] = ()
    confidence: float = 0.0
    evidence: tuple[str, ...] = ()
    verification: VerificationResult | None = None
    limitations: tuple[str, ...] = ()
    parts: tuple[ReasoningResult, ...] = ()

    def __post_init__(self) -> None:
        if not self.kind.strip():
            raise ValueError("A reasoning result needs a kind.")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("Reasoning confidence must be between 0 and 1.")

    @property
    def derived(self) -> bool:
        """Whether the conclusion was *derived* (not merely retrieved)."""
        return self.kind in {
            KIND_PROCEDURE,
            KIND_RECOMMENDATION,
            KIND_COMPARISON,
            KIND_VERDICT,
            KIND_TIMELINE,
            KIND_DEDUCTION,
            KIND_INDUCTION,
            KIND_ABDUCTION,
        }

    def describe(self) -> str:
        """The conclusion with its kind, confidence, and limits (no hidden reasoning)."""
        lines = [f"[{self.kind}] {self.conclusion} (confidence {self.confidence:.2f})"]
        if self.trace:
            lines.extend(f"  - {step}" for step in self.trace)
        if self.limitations:
            lines.append("  limitations: " + "; ".join(self.limitations))
        return "\n".join(lines)


def _evidence_from(output: object) -> tuple[str, ...]:
    """The readable evidence lines an engine's output already carries, or ``()`.

    Each engine reports its own evidence in its own field; this reads the common
    ones (``evidence``, ``matched``, ``steps``) without inventing anything -- an
    engine that carries none yields ``()``.
    """
    for attribute in ("evidence", "matched", "steps", "edges"):
        value = getattr(output, attribute, None)
        if isinstance(value, (list, tuple)) and value:
            return tuple(_line_for(item) for item in value)
    return ()


def _line_for(item: object) -> str:
    """One readable line for an engine's evidence item (its own ``describe`` or ``str``)."""
    if isinstance(item, str):
        return item
    describe = getattr(item, "describe", None)
    return describe() if callable(describe) else str(item)


class ReasoningAPI:
    """The one facade over the reasoning engines. Delegates; holds no state.

    Every method returns a :class:`ReasoningResult` whose ``output`` is the engine's
    own object. Nothing here re-implements an engine's logic, so the facade cannot
    disagree with the engine it calls.
    """

    def __init__(self, *, ontology: Ontology | None = None) -> None:
        self._ontology = ontology

    @property
    def ontology(self) -> Ontology:
        """The graph this facade reasons over (the seeded one when none was given)."""
        if self._ontology is None:
            from core.cognitive_controller import CognitiveController

            self._ontology = CognitiveController().ontology
        return self._ontology

    # -- the classical methods (delegate to core.reasoning_engine) ----------
    def deduce(self, facts: Iterable[str], rules: list) -> ReasoningResult:
        """Forward-chaining deduction over explicit, transparent rules."""
        from core.inference import Inference
        from core.reasoning_engine import ReasoningEngine

        engine = ReasoningEngine()
        trace = engine.deduce(Inference(tuple(facts)), list(rules))
        conclusions = trace.conclusions
        return ReasoningResult(
            kind=KIND_DEDUCTION,
            question="deduction",
            conclusion="; ".join(conclusions) if conclusions else "no rule was satisfied",
            output=trace,
            trace=tuple(f"{result.method}: {result.conclusion}" for result in trace.results),
            confidence=min((r.confidence for r in trace.results), default=0.0),
            limitations=("a deduction is only as sound as its rules",),
        )

    def induce(self, observations: Iterable[str], *, minimum_occurrences: int = 2) -> ReasoningResult:
        """Induction from repeated observations -- a *tentative* generalisation."""
        from core.reasoning_engine import ReasoningEngine

        engine = ReasoningEngine()
        trace = engine.induce(list(observations))
        conclusions = trace.conclusions
        return ReasoningResult(
            kind=KIND_INDUCTION,
            question="induction",
            conclusion="; ".join(conclusions) if conclusions else "no recurring pattern",
            output=trace,
            trace=tuple(result.conclusion for result in trace.results),
            confidence=min((r.confidence for r in trace.results), default=0.0),
            limitations=(
                "an inductive pattern is a recurring observation, never a universal rule",
                f"at least {minimum_occurrences} occurrences are required",
            ),
        )

    def abduce(self, observation: str, rules: list) -> ReasoningResult:
        """Abduction: the possible explanations for an observation, ranked by confidence."""
        from core.reasoning_engine import ReasoningEngine

        engine = ReasoningEngine()
        trace = engine.abduce(observation, list(rules))
        conclusions = trace.conclusions
        return ReasoningResult(
            kind=KIND_ABDUCTION,
            question=observation,
            conclusion="; ".join(conclusions) if conclusions else "no explanation was available",
            output=trace,
            trace=tuple(result.conclusion for result in trace.results),
            confidence=max((r.confidence for r in trace.results), default=0.0),
            limitations=("an abduction is a plausible explanation, not a proof",),
        )

    # -- the graph engines (delegate to the answer bridges) ----------------
    def plan(self, question: str) -> ReasoningResult:
        """The ordered procedure the corpus states for a *how* question."""
        plan: AnswerPlan | None = answer_plan(question)
        return _from_plan(question, plan)

    def decide(self, question: str) -> ReasoningResult:
        """The ranked recommendation the graph states for a *what-to-do* question."""
        decision: AnswerDecision | None = answer_decision(question, ontology=self.ontology)
        return _from_decision(question, decision)

    def believe(self, question: str) -> ReasoningResult:
        """The evidence-weighed verdict the graph supports for a *does-it-hold* question."""
        belief: AnswerBelief | None = answer_belief(question, ontology=self.ontology)
        return _from_belief(question, belief)

    def compare(self, question: str) -> ReasoningResult:
        """The structural analogy the graph supports for an *is-it-like* question."""
        analogy: AnswerAnalogy | None = answer_analogy(question, ontology=self.ontology)
        return _from_analogy(question, analogy)

    def timeline(self, question: str, *, conversation: Conversation | None = None) -> ReasoningResult:
        """The ordered account of what happened, from the conversation's own turns."""
        temporal: AnswerTemporal | None = answer_temporal(question, conversation=conversation)
        return _from_temporal(question, temporal)

    def infer(self, question: str) -> ReasoningResult:
        """The relations the graph *entails* on the question's concepts."""
        derived = inferred_relations(self.ontology, question)
        return _from_derived(question, derived)

    def connect(self, question: str) -> ReasoningResult:
        """The walked multi-hop chains between the concepts the question names."""
        connections = concept_connections(self.ontology, question)
        return _from_connections(question, connections)

    def causal(self, question: str) -> ReasoningResult:
        """The directed causal paths the graph states for the question."""
        paths = causal_explanations(self.ontology, question)
        return _from_causal(question, paths)

    def facts(self, question: str) -> ReasoningResult:
        """The stored relations the question's own concepts participate in."""
        facts = relation_facts(self.ontology, question)
        return _from_facts(question, facts)

    def conflicts(self, question: str | None = None) -> ReasoningResult:
        """The contradictory relations the graph holds (scoped to a question if given)."""
        conflicts = relation_conflicts(self.ontology, question)
        return ReasoningResult(
            kind="conflicts",
            question=question or "",
            conclusion=(
                f"{len(conflicts)} contradiction(s) the graph holds"
                if conflicts
                else "the graph is consistent here"
            ),
            output=conflicts,
            trace=tuple(conflict.describe() for conflict in conflicts),
            confidence=1.0,
            limitations=(
                "the program does not know which side of a contradiction is right",
            ),
        )

    def hypotheses(self, question: str) -> ReasoningResult:
        """The candidate hypotheses the graph licenses for the question, ranked.

        Generation is delegated to :func:`core.hypothesis_reasoning.compare_hypotheses`,
        so the facade reports exactly the candidates the engine finds -- including a
        candidate the graph holds evidence against -- and yields an honest absence
        for a question the graph cannot speak to.
        """
        comparison: HypothesisComparison = compare_hypotheses(self.ontology, question)
        return _from_hypotheses(question, comparison)

    def path(self, source: str, target: str, *, max_hops: int = 3) -> ReasoningResult:
        """The shortest typed chain between two concepts, or an honest absence."""
        found: ConnectionPath | None = find_connection(
            self.ontology, source, target, max_hops=max_hops
        )
        if found is None:
            return ReasoningResult(
                kind="connection",
                question=f"{source} -> {target}",
                conclusion=f"no chain within {max_hops} hops connects {source} and {target}",
                output=None,
                confidence=0.0,
                limitations=("an absence of a path is not a claim that none exists",),
            )
        return ReasoningResult(
            kind="connection",
            question=f"{source} -> {target}",
            conclusion=found.describe(),
            output=found,
            trace=tuple(step.describe() for step in found.steps),
            confidence=found.confidence,
            limitations=("a walked chain is a derivation, not a stored fact",),
        )

    # -- verification -----------------------------------------------------
    def verify(
        self,
        source: str,
        relation_type: str,
        target: str,
        **kwargs: object,
    ) -> ReasoningResult:
        """Verify a ``source —relation→ target`` conclusion against the graph."""
        result = verify_conclusion(
            source, relation_type, target, ontology=self.ontology, **kwargs  # type: ignore[arg-type]
        )
        return ReasoningResult(
            kind=KIND_VERIFICATION,
            question=f"{source} {relation_type} {target}",
            conclusion=result.label,
            output=result,
            trace=result.reasons,
            confidence=result.confidence,
            evidence=tuple(p.describe() for p in result.premises),
            verification=result,
            limitations=("a confidence is not a probability",),
        )

    # -- the top-level call ------------------------------------------------
    def reason(self, question: str, *, conversation: Conversation | None = None) -> ReasoningResult:
        """Decompose ``question`` and route each part to the engine its kind warrants.

        This is the top-level entry point: it decomposes the question
        (:mod:`core.problem_decomposition`) and answers each part with the engine
        the part's kind names, joining the results. A single-part question yields a
        composite with one part; a blank question yields an honest empty composite.
        Nothing here decides *what* an engine returns -- it only chooses which engine
        a part belongs to, from the part's kind.
        """
        decomposition = decompose(question, ontology=self.ontology)
        if decomposition is None:
            return ReasoningResult(
                kind=KIND_COMPOSITE,
                question=question,
                conclusion="the question was empty; nothing was reasoned",
                confidence=0.0,
                limitations=("a blank question warrants no reasoning",),
            )
        parts = tuple(
            self._reason_part(part.text, part.kind, conversation=conversation)
            for part in decomposition.parts
        )
        derived = [part for part in parts if part.derived]
        confidence = min((part.confidence for part in derived), default=0.0)
        trace = (
            f"decomposition: {decomposition.objective}",
            *(f"[{part.kind}] {part.conclusion}" for part in parts),
        )
        limitations = (
            *(limitation for part in parts for limitation in part.limitations),
            *decomposition.unknown,
        )
        return ReasoningResult(
            kind=KIND_COMPOSITE,
            question=question,
            conclusion=_composite_conclusion(parts),
            output=decomposition,
            trace=trace,
            confidence=confidence,
            limitations=limitations,
            parts=parts,
        )

    def _reason_part(self, text: str, kind: str, *, conversation: Conversation | None) -> ReasoningResult:
        """Route one part to the engine its kind names."""
        if kind == KIND_PROCEDURE:
            return self.plan(text)
        if kind == KIND_RECOMMENDATION:
            return self.decide(text)
        if kind == KIND_COMPARISON:
            return self.compare(text)
        if kind == KIND_VERDICT:
            return self.believe(text)
        if kind == KIND_TIMELINE:
            return self.timeline(text, conversation=conversation)
        return self._lookup(text)

    def _lookup(self, text: str) -> ReasoningResult:
        """A definitional part: the graph's stored facts and any walked connection."""
        facts = relation_facts(self.ontology, text)
        connections = concept_connections(self.ontology, text)
        if not facts and not connections:
            return ReasoningResult(
                kind=KIND_LOOKUP,
                question=text,
                conclusion="no stored relation and no walked chain reaches this question",
                confidence=0.0,
                limitations=("the graph holds nothing on the concepts this question names",),
            )
        trace = tuple(fact.describe() for fact in facts) + tuple(
            connection.describe() for connection in connections
        )
        confidence = max(
            (fact.confidence for fact in facts),
            default=0.0,
        )
        return ReasoningResult(
            kind=KIND_LOOKUP,
            question=text,
            conclusion=trace[0] if trace else "the graph holds relations on these concepts",
            output=(facts, connections),
            trace=trace,
            confidence=confidence,
            evidence=tuple(fact.describe() for fact in facts),
            limitations=("stored relations are reported in their declared direction",),
        )


def _composite_conclusion(parts: tuple[ReasoningResult, ...]) -> str:
    """One readable conclusion from the parts, or an honest "nothing"."""
    if not parts:
        return "the question had no reasoning unit"
    lines = [f"{index + 1}. {part.conclusion}" for index, part in enumerate(parts)]
    return " | ".join(lines)


def _from_plan(question: str, plan: AnswerPlan | None) -> ReasoningResult:
    if plan is None:
        return ReasoningResult(
            kind=KIND_PROCEDURE,
            question=question,
            conclusion="no procedure matches this question",
            limitations=("a procedure is only reported when the corpus states one",),
        )
    return ReasoningResult(
        kind=KIND_PROCEDURE,
        question=question,
        conclusion=plan.topic,
        output=plan,
        trace=tuple(plan.steps),
        confidence=0.8,
        evidence=(f"procedure from {plan.source}",),
        limitations=("the steps are ordered, never executed",),
    )


def _from_decision(question: str, decision: AnswerDecision | None) -> ReasoningResult:
    if decision is None:
        return ReasoningResult(
            kind=KIND_RECOMMENDATION,
            question=question,
            conclusion="the graph states no option in the direction asked for",
            limitations=("an option is never offered when the graph moves the goal the other way",),
        )
    return ReasoningResult(
        kind=KIND_RECOMMENDATION,
        question=question,
        conclusion=decision.selected,
        output=decision,
        trace=decision.sentences,
        confidence=0.8,
        evidence=tuple(
            f"{option} ({relation}, confidence {confidence:.2f})"
            for option, relation, confidence in decision.options
        ),
        limitations=("a recommendation is reported, never executed",),
    )


def _from_belief(question: str, belief: AnswerBelief | None) -> ReasoningResult:
    if belief is None:
        return ReasoningResult(
            kind=KIND_VERDICT,
            question=question,
            conclusion="the graph cannot speak to this claim",
            limitations=("a verdict needs evidence the graph actually holds",),
        )
    return ReasoningResult(
        kind=KIND_VERDICT,
        question=question,
        conclusion=belief.statement,
        output=belief,
        trace=tuple(belief.supporting) + tuple(belief.contradicting),
        confidence=belief.confidence,
        evidence=belief.supporting,
        limitations=("a verdict is provisional until the graph is revised",),
    )


def _from_analogy(question: str, analogy: AnswerAnalogy | None) -> ReasoningResult:
    if analogy is None:
        return ReasoningResult(
            kind=KIND_COMPARISON,
            question=question,
            conclusion="the question names fewer than two concepts to compare",
            limitations=("a comparison needs two known concepts",),
        )
    verdict = "an analogy" if analogy.meaningful else "not an analogy"
    transferred_lines = tuple(
        f"{source} —{relation}→ {target} (confidence {confidence:.2f})"
        for source, relation, target, confidence in analogy.transferred
    )
    return ReasoningResult(
        kind=KIND_COMPARISON,
        question=question,
        conclusion=f"{analogy.base} vs {analogy.target}: {verdict}",
        output=analogy,
        trace=tuple(analogy.matched) + transferred_lines,
        # The analogy engine reports a *systematicity score* (a count of aligned
        # relations), not a probability; normalise it to a bounded confidence and
        # say so in the limitations, rather than presenting a count as certainty.
        confidence=_bounded(analogy.score / _ANALOGY_SCORE_TARGET),
        evidence=analogy.matched,
        limitations=(
            "a candidate inference is a suggestion, never asserted into the graph",
            "the score counts aligned relations, it is not a probability",
        ),
    )


def _from_temporal(question: str, temporal: AnswerTemporal | None) -> ReasoningResult:
    if temporal is None:
        return ReasoningResult(
            kind=KIND_TIMELINE,
            question=question,
            conclusion="no timeline is available",
            limitations=("a timeline needs at least two turns",),
        )
    return ReasoningResult(
        kind=KIND_TIMELINE,
        question=question,
        conclusion=temporal.change,
        output=temporal,
        trace=tuple(f"{entry.role}: {entry.text}" for entry in temporal.entries),
        confidence=1.0,
        limitations=("the timeline orders what the conversation holds, nothing more",),
    )


def _from_derived(question: str, derived: tuple[InferredRelation, ...]) -> ReasoningResult:
    return ReasoningResult(
        kind="inference",
        question=question,
        conclusion=(
            f"{len(derived)} relation(s) the graph entails"
            if derived
            else "the graph entails nothing on these concepts"
        ),
        output=derived,
        trace=tuple(item.explain() for item in derived),
        confidence=max((item.confidence for item in derived), default=0.0),
        limitations=("a derived relation is a conclusion, never a stored fact",),
    )


def _from_connections(question: str, connections: tuple[ConceptConnection, ...]) -> ReasoningResult:
    return ReasoningResult(
        kind="connection",
        question=question,
        conclusion=(
            f"{len(connections)} walked chain(s)"
            if connections
            else "no chain connects the concepts the question names"
        ),
        output=connections,
        trace=tuple(item.describe() for item in connections),
        confidence=max((item.confidence for item in connections), default=0.0),
        limitations=("a walked chain is not stored",),
    )


def _from_causal(question: str, paths: tuple[CausalExplanation, ...]) -> ReasoningResult:
    return ReasoningResult(
        kind="causal",
        question=question,
        conclusion=(
            f"{len(paths)} directed causal path(s)"
            if paths
            else "the graph states no causal path here"
        ),
        output=paths,
        trace=tuple(item.describe() for item in paths),
        confidence=max((item.confidence for item in paths), default=0.0),
        limitations=("an association is never read as a cause",),
    )


def _from_facts(question: str, facts: tuple[RelationFact, ...]) -> ReasoningResult:
    return ReasoningResult(
        kind=KIND_LOOKUP,
        question=question,
        conclusion=(
            f"{len(facts)} stored relation(s) on the question's concepts"
            if facts
            else "no stored relation on the question's concepts"
        ),
        output=facts,
        trace=tuple(fact.describe() for fact in facts),
        confidence=max((fact.confidence for fact in facts), default=0.0),
        evidence=tuple(fact.describe() for fact in facts),
    )


def _from_hypotheses(question: str, comparison: HypothesisComparison) -> ReasoningResult:
    """The ranked candidate hypotheses as a reasoning result (delegated, never invented)."""
    candidates = comparison.candidates
    if not candidates:
        return ReasoningResult(
            kind=KIND_HYPOTHESES,
            question=question,
            conclusion="no candidate hypothesis is licensed by the graph",
            output=None,
            confidence=0.0,
            limitations=("a hypothesis is generated only from what the graph already holds",),
        )
    best = comparison.best
    limitations: list[str] = [
        "a hypothesis is a candidate, never an established fact",
    ]
    if comparison.contested:
        limitations.append("the graph holds evidence against at least one candidate")
    if comparison.note:
        limitations.append(comparison.note)
    return ReasoningResult(
        kind=KIND_HYPOTHESES,
        question=question,
        conclusion=best.statement if best is not None else "",
        output=comparison,
        trace=tuple(candidate.describe() for candidate in candidates),
        confidence=best.confidence if best is not None else 0.0,
        evidence=tuple(line for candidate in candidates for line in candidate.evidence),
        limitations=tuple(limitations),
    )


@dataclass(frozen=True)
class FacadeJudgement:
    """One held-out expectation about what the facade must return."""

    name: str
    question: str
    method: str
    """The facade method to call."""

    expect_kind: str = ""
    expect_nonempty: bool = True
    """Whether the result must carry a non-empty conclusion (a negative control is False)."""

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("A facade judgement needs a name.")


#: The held-out facade catalogue: the facade must return exactly what the engine
#: returns, and a question the engine cannot answer must yield an honest absence --
#: with negative controls, so a facade that invents a result fails the gate.
FACADE_JUDGEMENTS: tuple[FacadeJudgement, ...] = (
    FacadeJudgement("plan-route", "how do I measure porosity?", "plan", expect_kind=KIND_PROCEDURE),
    FacadeJudgement("plan-absent", "what is porosity?", "plan", expect_nonempty=False),
    FacadeJudgement("decision-route", "what should I do to increase permeability?", "decide", expect_kind=KIND_RECOMMENDATION),
    FacadeJudgement("decision-absent", "what is permeability?", "decide", expect_nonempty=False),
    FacadeJudgement("belief-route", "does acid stimulation increase permeability?", "believe", expect_kind=KIND_VERDICT),
    FacadeJudgement("belief-absent", "what is permeability?", "believe", expect_nonempty=False),
    FacadeJudgement("compare-route", "is porosity like permeability?", "compare", expect_kind=KIND_COMPARISON),
    FacadeJudgement("compare-absent", "what is porosity?", "compare", expect_nonempty=False),
    FacadeJudgement("facts-route", "what is porosity?", "facts", expect_kind=KIND_LOOKUP),
    FacadeJudgement("hypotheses-route", "what is porosity?", "hypotheses", expect_kind=KIND_HYPOTHESES),
    FacadeJudgement("hypotheses-absent", "zzz_unseen qqq_unseen", "hypotheses", expect_nonempty=False),
)


def facade_accuracy(
    judgements: tuple[FacadeJudgement, ...] = FACADE_JUDGEMENTS,
    *,
    api: ReasoningAPI | None = None,
) -> float:
    """The share of held-out judgements the facade gets right.

    Each judgement runs a real facade method and checks the kind and whether a
    conclusion is present -- with negative controls (a definitional question must
    yield *no* plan/decision/belief/comparison), so a facade that invents a result
    the engine did not produce fails the gate. Returns ``0.0`` for an empty
    catalogue.
    """
    catalogue = tuple(judgements)
    if not catalogue:
        return 0.0
    facade = api or ReasoningAPI()
    correct = 0
    for judgement in catalogue:
        method = getattr(facade, judgement.method, None)
        if method is None:
            continue
        try:
            result = method(judgement.question)
        except Exception:
            continue
        if not isinstance(result, ReasoningResult):
            continue
        kind_ok = not judgement.expect_kind or result.kind == judgement.expect_kind
        # "Non-empty" for a plan/decision/etc. means the engine actually returned an
        # object, not just a conclusion line; the honest absence carries no output.
        has_output = result.output is not None
        nonempty_ok = has_output == judgement.expect_nonempty
        if kind_ok and nonempty_ok:
            correct += 1
    return correct / len(catalogue)


__all__ = [
    "FACADE_JUDGEMENTS",
    "KIND_ABDUCTION",
    "KIND_COMPOSITE",
    "KIND_DEDUCTION",
    "KIND_HYPOTHESES",
    "KIND_INDUCTION",
    "KIND_VERIFICATION",
    "FacadeJudgement",
    "ReasoningAPI",
    "ReasoningResult",
    "facade_accuracy",
]
