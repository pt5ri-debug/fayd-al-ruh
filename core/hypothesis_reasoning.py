"""Generate and compare candidate hypotheses, deterministically over the graph.

Phase 4 asks the reasoning engine to *generate and compare hypotheses*: when a
question does not have one obvious answer, propose the candidate explanations the
knowledge already supports, score each by its evidence, and report which is
best-supported and which conflict.

This module does that **without inventing anything**. A candidate is always a
conclusion the graph already licenses:

* a **stored relation** between two concepts the question names,
* a relation the graph **entails** (``core.relation_inference``),
* a **walked chain** between them (``core.relation_paths``),
* a **causal mechanism** the graph states (``core.causal_paths``), or
* a **decision option** the graph offers (``core.answer_decision``).

Nothing here calls a language model and nothing is asserted into the graph. Each
:class:`CandidateHypothesis` carries its statement, its confidence, the evidence
for it, and any evidence *against* it (an opposing relation on the same pair), so
:meth:`HypothesisComparison.describe` can rank the candidates honestly and name a
contradiction rather than hiding it. A question the graph cannot speak to yields
**no** candidates rather than a guess -- that absence is the honest answer.
"""

from __future__ import annotations

from dataclasses import dataclass

from core.ontology import Ontology
from core.relation_polarity import opposes

#: Where a candidate came from, so a reader can tell a stored fact from a derivation.
ORIGIN_STORED = "stored"
ORIGIN_ENTAILED = "entailed"
ORIGIN_WALKED = "walked"
ORIGIN_CAUSAL = "causal"
ORIGIN_DECISION = "decision"

#: The origin ranks: a stored relation outranks a derivation when scores tie.
_ORIGIN_RANK = {
    ORIGIN_STORED: 0,
    ORIGIN_DECISION: 1,
    ORIGIN_ENTAILED: 2,
    ORIGIN_CAUSAL: 3,
    ORIGIN_WALKED: 4,
}

DEFAULT_HYPOTHESIS_LIMIT = 8


@dataclass(frozen=True)
class CandidateHypothesis:
    """One candidate explanation: a statement the graph licenses, with its evidence."""

    statement: str
    confidence: float
    origin: str
    evidence: tuple[str, ...] = ()
    against: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.statement.strip():
            raise ValueError("A candidate hypothesis needs a statement.")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("A candidate's confidence must be between 0 and 1.")
        if self.origin not in _ORIGIN_RANK:
            raise ValueError(f"Unknown hypothesis origin: {self.origin!r}")

    @property
    def contested(self) -> bool:
        """Whether the graph holds evidence against this candidate."""
        return bool(self.against)

    def describe(self) -> str:
        lines = [f"[{self.origin}] {self.statement} (confidence {self.confidence:.2f})"]
        lines.extend(f"  for: {line}" for line in self.evidence)
        lines.extend(f"  against: {line}" for line in self.against)
        return "\n".join(lines)


@dataclass(frozen=True)
class HypothesisComparison:
    """The candidates for a question, ranked, with the best and any conflict named."""

    question: str
    candidates: tuple[CandidateHypothesis, ...] = ()
    note: str = ""

    @property
    def best(self) -> CandidateHypothesis | None:
        """The best-supported candidate, or ``None`` when there is none."""
        return self.candidates[0] if self.candidates else None

    @property
    def contested(self) -> tuple[CandidateHypothesis, ...]:
        """The candidates the graph holds evidence against."""
        return tuple(candidate for candidate in self.candidates if candidate.contested)

    def describe(self) -> str:
        if not self.candidates:
            return self.note or "no candidate hypothesis is licensed by the graph"
        lines = [f"Hypotheses for: {self.question}"]
        for index, candidate in enumerate(self.candidates, start=1):
            lines.append(f"  {index}. {candidate.statement} (confidence {candidate.confidence:.2f})")
        if self.best is not None:
            lines.append(f"Best supported: {self.best.statement}")
        if self.contested:
            lines.append(
                "Contested: "
                + "; ".join(candidate.statement for candidate in self.contested)
            )
        return "\n".join(lines)


def _against(ontology: Ontology, source_label: str, relation_type: str, target_label: str) -> tuple[str, ...]:
    """The relations on the same ordered pair that oppose ``relation_type``."""
    source_ids = {concept.id for concept in ontology.find_concepts(source_label)}
    target_ids = {concept.id for concept in ontology.find_concepts(target_label)}
    found: list[str] = []
    for relation in ontology.relations.values():
        if relation.source_id in source_ids and relation.target_id in target_ids:
            if opposes(relation_type, relation.relation_type):
                found.append(f"{relation.relation_type} (the graph holds the opposite)")
    return tuple(found)


def generate_hypotheses(
    ontology: Ontology,
    question: str,
    *,
    limit: int = DEFAULT_HYPOTHESIS_LIMIT,
) -> tuple[CandidateHypothesis, ...]:
    """The candidate explanations the graph licenses for ``question``, ranked.

    Candidates come from the question's **own** concepts: each stored relation
    between them, each entailed relation, each walked chain, each causal path, and
    (for a decisional question) each option the decision engine ranks. De-duplicated
    by statement and ordered strongest first. Returns ``()` when the question names
    no concepts or the graph holds nothing on them -- never a guess.
    """
    if limit <= 0:
        return ()
    from core.causal_paths import causal_explanations
    from core.ontology import Ontology as _Ontology  # noqa: F401  (typing clarity)
    from core.relation_connections import concept_connections
    from core.relation_facts import relation_facts
    from core.relation_inference import inferred_relations

    candidates: list[CandidateHypothesis] = []

    # 1. Stored relations on the question's concepts (a fact is a candidate too).
    for fact in relation_facts(ontology, question):
        statement = f"{fact.source} {fact.relation_type} {fact.target}"
        candidates.append(
            CandidateHypothesis(
                statement=statement,
                confidence=float(fact.confidence),
                origin=ORIGIN_STORED,
                evidence=(fact.describe(),),
                against=_against(ontology, fact.source, fact.relation_type, fact.target),
            )
        )

    # 2. Entailed relations (a derivation the graph licenses).
    for inferred in inferred_relations(ontology, question):
        statement = f"{inferred.source} {inferred.relation_type} {inferred.target}"
        candidates.append(
            CandidateHypothesis(
                statement=statement,
                confidence=float(inferred.confidence),
                origin=ORIGIN_ENTAILED,
                evidence=(inferred.explain(),),
                against=_against(ontology, inferred.source, inferred.relation_type, inferred.target),
            )
        )

    # 3. Walked chains between the question's concepts.
    for connection in concept_connections(ontology, question):
        statement = f"{connection.source} connects to {connection.target}"
        candidates.append(
            CandidateHypothesis(
                statement=statement,
                confidence=float(connection.confidence),
                origin=ORIGIN_WALKED,
                evidence=(connection.describe(),),
            )
        )

    # 4. Causal mechanisms.
    for path in causal_explanations(ontology, question):
        statement = f"{path.cause} causes {path.effect}"
        candidates.append(
            CandidateHypothesis(
                statement=statement,
                confidence=float(path.confidence),
                origin=ORIGIN_CAUSAL,
                evidence=(path.describe(),),
            )
        )

    # 5. Decision options (only for a decisional question).
    from core.answer_decision import answer_decision, is_decisional

    if is_decisional(question):
        decision = answer_decision(question, ontology=ontology)
        if decision is not None:
            for option in decision.options:
                label = option[0] if isinstance(option, tuple) else str(option)
                candidates.append(
                    CandidateHypothesis(
                        statement=f"do: {label}",
                        confidence=float(option[2]) if isinstance(option, tuple) and len(option) > 2 else 0.5,
                        origin=ORIGIN_DECISION,
                        evidence=(f"the graph links {label} to {decision.goal}",),
                    )
                )

    return _rank(candidates, limit=limit)


def _rank(candidates: list[CandidateHypothesis], *, limit: int) -> tuple[CandidateHypothesis, ...]:
    """De-duplicate by statement and order strongest first (deterministic tie-break)."""
    seen: dict[str, CandidateHypothesis] = {}
    for candidate in candidates:
        existing = seen.get(candidate.statement)
        if existing is None or candidate.confidence > existing.confidence:
            seen[candidate.statement] = candidate
    ordered = sorted(
        seen.values(),
        key=lambda item: (-item.confidence, _ORIGIN_RANK[item.origin], item.statement),
    )
    return tuple(ordered[:limit])


def compare_hypotheses(
    ontology: Ontology,
    question: str,
    *,
    limit: int = DEFAULT_HYPOTHESIS_LIMIT,
) -> HypothesisComparison:
    """Generate and compare the candidate hypotheses for ``question``.

    The comparison ranks the candidates strongest first and names the best-supported
    one and any the graph holds evidence against. It **does not** pick a winner the
    graph cannot justify: two equally-supported opposing candidates are both reported,
    and the ``note`` says so, because choosing between two evidenced claims is the
    reader's call.
    """
    candidates = generate_hypotheses(ontology, question, limit=limit)
    if not candidates:
        return HypothesisComparison(
            question=question,
            candidates=(),
            note="no candidate hypothesis is licensed by the graph",
        )
    note = ""
    top = candidates[0]
    if top.contested:
        note = "the best-supported candidate is itself contested by the graph"
    elif len(candidates) > 1 and candidates[1].confidence == top.confidence:
        note = "two candidates are equally supported; the graph does not prefer one"
    return HypothesisComparison(question=question, candidates=candidates, note=note)


@dataclass(frozen=True)
class HypothesisJudgement:
    """One held-out expectation about hypothesis generation for a question."""

    name: str
    question: str
    expect_candidates: bool
    """Whether the graph licenses at least one candidate (a negative control is False)."""

    expect_origin: str = ""
    """An origin the candidates must include, when the graph licenses one."""

    expect_contested: bool = False
    """Whether the best candidate must be reported as contested by the graph."""

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("A hypothesis judgement needs a name.")
        if self.expect_origin and self.expect_origin not in _ORIGIN_RANK:
            raise ValueError(f"Unknown origin in a hypothesis judgement: {self.expect_origin}")


def _gate_ontology() -> Ontology:
    """A small synthetic graph: a stored edge, its opposite, and a two-hop chain.

    ``alpha —increases→ beta`` (stored) with ``alpha —reduces→ beta`` (its opposite,
    so a candidate is contested), and ``beta —drives→ gamma`` (a walked chain from
    alpha to gamma). Nothing is invented beyond these.
    """
    ontology = Ontology()
    alpha = ontology.add_concept_once("alpha")
    beta = ontology.add_concept_once("beta")
    gamma = ontology.add_concept_once("gamma")
    ontology.add_relation_once(alpha.id, beta.id, "increases", 0.7)
    ontology.add_relation_once(alpha.id, beta.id, "reduces", 0.5)
    ontology.add_relation_once(beta.id, gamma.id, "drives", 0.6)
    return ontology


#: The held-out hypothesis catalogue -- hand-written with **negative controls**: a
#: question the graph cannot speak to must yield no candidate, and a claim the graph
#: holds the opposite of must be reported as contested rather than as supported.
HYPOTHESIS_JUDGEMENTS: tuple[HypothesisJudgement, ...] = (
    HypothesisJudgement(
        "stored-edge-is-a-candidate", "alpha increases beta", True, expect_origin=ORIGIN_STORED
    ),
    HypothesisJudgement(
        "contested-is-reported", "alpha increases beta", True, expect_contested=True
    ),
    HypothesisJudgement(
        "unknown-question-yields-nothing", "zzz_unseen qqq_unseen", False
    ),
)


def hypothesis_accuracy(
    judgements: tuple[HypothesisJudgement, ...] = HYPOTHESIS_JUDGEMENTS,
    *,
    ontology: Ontology | None = None,
) -> float:
    """The share of held-out judgements hypothesis generation gets right.

    A judgement counts only when the candidate presence, the origin present, and the
    contested flag all match -- so a generator that invents a candidate for an
    unknown question, misses a stored edge, or hides a contradiction fails as surely
    as one that finds nothing. Returns ``0.0`` for an empty catalogue.
    """
    catalogue = tuple(judgements)
    if not catalogue:
        return 0.0
    if ontology is None:
        ontology = _gate_ontology()
    correct = 0
    for judgement in catalogue:
        comparison = compare_hypotheses(ontology, judgement.question)
        has_candidates = bool(comparison.candidates)
        if has_candidates != judgement.expect_candidates:
            continue
        if judgement.expect_origin and not any(
            candidate.origin == judgement.expect_origin for candidate in comparison.candidates
        ):
            continue
        if judgement.expect_contested:
            best = comparison.best
            if best is None or not best.contested:
                continue
        correct += 1
    return correct / len(catalogue)


__all__ = [
    "DEFAULT_HYPOTHESIS_LIMIT",
    "HYPOTHESIS_JUDGEMENTS",
    "ORIGIN_CAUSAL",
    "ORIGIN_DECISION",
    "ORIGIN_ENTAILED",
    "ORIGIN_STORED",
    "ORIGIN_WALKED",
    "CandidateHypothesis",
    "HypothesisComparison",
    "HypothesisJudgement",
    "compare_hypotheses",
    "generate_hypotheses",
    "hypothesis_accuracy",
]
