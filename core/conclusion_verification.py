"""Verify an important factual conclusion before it is returned to a reader.

A reasoning engine can *derive* a conclusion -- deduce it from rules, walk a
multi-hop chain to it, weigh a belief for it -- but a derivation is not the same as
a verified claim. This module is the step that sits between "the engine produced a
conclusion" and "the program states it": it checks whether the conclusion's
premises are available, whether its inference path is intact, whether the graph
contradicts it, whether counterevidence exists, and whether it is temporally and
contextually valid, then assigns **one** verification status.

The statuses are the five the task requires (:data:`STATUSES`):

* ``supported`` -- the premises and a path to the conclusion are present, no
  contradiction, and no counterevidence;
* ``partially_supported`` -- some premises or part of the path is present, but not
  all of it;
* ``unverified`` -- the conclusion is stated but nothing in the graph corroborates
  it (a hypothesis, a model proposal);
* ``contradicted`` -- the graph holds a claim opposing it;
* ``insufficient_information`` -- the premises are not available at all, so no
  verdict can be reached.

**Honest by construction.** It never forces a verdict when the evidence is
insufficient (that case is ``insufficient_information``, not a guess), it never
silently discards a contradiction (a contradicted conclusion is reported as such,
never quietly downgraded), it treats a **confidence score as a confidence, never as
a probability** (``probability`` is ``None`` unless a caller supplies a calibrated
one), and it is **read-only** -- nothing here writes the graph. It is
**LLM-independent**: the checks are deterministic over the graph's own relations,
and an optional model may *propose* a conclusion but never verify one.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

from core.ontology import Ontology
from core.relation_consistency import relation_conflicts
from core.relation_polarity import opposes

#: The five verification statuses, in increasing order of how settled a conclusion
#: is. A caller compares by value, never by order, so the sequence is documentation
#: only -- but it is fixed so two runs report the same status for the same input.
STATUS_SUPPORTED = "supported"
STATUS_PARTIALLY_SUPPORTED = "partially_supported"
STATUS_UNVERIFIED = "unverified"
STATUS_CONTRADICTED = "contradicted"
STATUS_INSUFFICIENT = "insufficient_information"

STATUSES: tuple[str, ...] = (
    STATUS_SUPPORTED,
    STATUS_PARTIALLY_SUPPORTED,
    STATUS_UNVERIFIED,
    STATUS_CONTRADICTED,
    STATUS_INSUFFICIENT,
)

#: A readable label per status, so a surface never invents its own wording.
STATUS_LABELS: dict[str, str] = {
    STATUS_SUPPORTED: "Supported",
    STATUS_PARTIALLY_SUPPORTED: "Partially supported",
    STATUS_UNVERIFIED: "Unverified",
    STATUS_CONTRADICTED: "Contradicted",
    STATUS_INSUFFICIENT: "Insufficient information",
}

#: The default floor below which a premise is not counted as present at all.
DEFAULT_MIN_PREMISE_CONFIDENCE = 0.3


@dataclass(frozen=True)
class Premise:
    """One premise a conclusion rests on: a claim and whether the graph holds it."""

    statement: str
    available: bool
    confidence: float = 0.0
    reference: str = ""

    def __post_init__(self) -> None:
        if not self.statement.strip():
            raise ValueError("A premise needs a statement.")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("A premise's confidence must be between 0 and 1.")

    def describe(self) -> str:
        mark = "present" if self.available else "missing"
        return f"{self.statement} ({mark})"


@dataclass(frozen=True)
class VerificationResult:
    """One conclusion's verification, with every check that produced it."""

    conclusion: str
    status: str
    confidence: float
    """How strongly the evidence supports the conclusion -- a **confidence**, not a
    probability. ``probability`` stays ``None`` unless a calibrated one is supplied."""

    premises: tuple[Premise, ...] = ()
    path: tuple[str, ...] = ()
    """The inference steps the conclusion rests on, each a readable line."""

    contradicting: tuple[str, ...] = ()
    counterevidence: tuple[str, ...] = ()
    assumptions: tuple[str, ...] = ()
    reasons: tuple[str, ...] = ()
    probability: float | None = None
    """A **calibrated** probability, or ``None``. Never inferred from ``confidence``."""

    def __post_init__(self) -> None:
        if not self.conclusion.strip():
            raise ValueError("A verification needs a conclusion.")
        if self.status not in STATUSES:
            raise ValueError(f"Unknown verification status: {self.status!r}")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("Verification confidence must be between 0 and 1.")
        if self.probability is not None and not 0.0 <= self.probability <= 1.0:
            raise ValueError("A calibrated probability must be between 0 and 1.")

    @property
    def verified(self) -> bool:
        """True only for a ``supported`` conclusion -- the one status a caller may
        publish as established fact."""
        return self.status == STATUS_SUPPORTED

    @property
    def publishable(self) -> bool:
        """Whether the conclusion may be stated as knowledge rather than a hypothesis."""
        return self.status in (STATUS_SUPPORTED, STATUS_PARTIALLY_SUPPORTED)

    @property
    def label(self) -> str:
        return STATUS_LABELS[self.status]

    def describe(self) -> str:
        """A short, honest line: the status, the confidence, and the reasons."""
        head = f"{self.conclusion} — {self.label} (confidence {self.confidence:.2f}"
        if self.probability is not None:
            head += f", calibrated probability {self.probability:.2f}"
        else:
            head += ", not a probability"
        head += ")"
        if not self.reasons:
            return head
        return head + "\n" + "\n".join(f"  - {reason}" for reason in self.reasons)


def _named_relation(
    ontology: Ontology, source_label: str, relation_type: str, target_label: str
):
    """The stored relation matching a claim's triple, or ``None``."""
    sources = {concept.id for concept in ontology.find_concepts(source_label)}
    targets = {concept.id for concept in ontology.find_concepts(target_label)}
    if not sources or not targets:
        return None
    for relation in ontology.relations.values():
        if (
            relation.source_id in sources
            and relation.target_id in targets
            and relation.relation_type == relation_type
        ):
            return relation
    return None


def _opposing(
    ontology: Ontology, source_label: str, relation_type: str, target_label: str
) -> tuple[str, ...]:
    """The graph's claims that oppose this one on the same ordered pair.

    A claim is opposed when the graph holds the same ordered pair with an
    **antonymous** relation (``increases`` vs ``reduces``). Only the graph's own
    relation polarity decides this, never a word guess, so a merely different
    relation on the pair is not called a contradiction.
    """
    sources = {concept.id for concept in ontology.find_concepts(source_label)}
    targets = {concept.id for concept in ontology.find_concepts(target_label)}
    if not sources or not targets:
        return ()
    found: list[str] = []
    for relation in ontology.relations.values():
        if (
            relation.source_id in sources
            and relation.target_id in targets
            and opposes(relation_type, relation.relation_type)
        ):
            found.append(relation.relation_type)
    return tuple(sorted(set(found)))


def verify_conclusion(
    source: str,
    relation_type: str,
    target: str,
    *,
    ontology: Ontology | None = None,
    premises: Iterable[Premise] = (),
    path: Iterable[str] = (),
    assumptions: Iterable[str] = (),
    context_valid: bool = True,
    now: object | None = None,
    min_premise_confidence: float = DEFAULT_MIN_PREMISE_CONFIDENCE,
) -> VerificationResult:
    """Verify one conclusion -- a ``source —relation→ target`` claim -- against the graph.

    The checks run in a fixed order and the **first** that settles the status wins,
    so the same input always yields the same verdict:

    1. **Contradiction** -- the graph holds an antonymous claim on the same pair, so
       the conclusion is ``contradicted`` (the strongest, and never downgraded).
    2. **Premises** -- a premise the caller supplied but the graph does not hold, or
       a premise below the confidence floor, is missing; a premise the caller
       supplied and the graph corroborates is present. With **no** premise available
       and no stored relation, the conclusion is ``insufficient_information``.
    3. **Path** -- a stored relation, or a supplied inference path, is what makes the
       conclusion supported at all; a supplied path with no stored relation is a
       *partial* corroboration, not a full one.
    4. **Context** -- a conclusion the caller marks context-invalid is
       ``partially_supported`` at best, so a claim true only in another context is
       never published as settled.
    5. Otherwise the conclusion is ``supported``.

    The returned :class:`VerificationResult` carries the premises, the path, the
    contradicting claims, the counterevidence, the assumptions, and one reason per
    check that moved the verdict. It **never** presents ``confidence`` as a
    probability -- ``probability`` is ``None`` unless the caller passes a calibrated
    one in the premises' metadata.
    """
    if ontology is None:
        from core.cognitive_controller import CognitiveController

        ontology = CognitiveController().ontology

    premise_list = tuple(premises)
    path_list = tuple(path)
    assumptions_list = tuple(assumptions)
    reasons: list[str] = []

    # 1. Contradiction is settled first and never downgraded.
    opposing = _opposing(ontology, source, relation_type, target)
    stored = _named_relation(ontology, source, relation_type, target)
    if opposing:
        reasons.append(
            "the graph holds the opposing claim(s): " + ", ".join(opposing)
        )
        conflicts = relation_conflicts(ontology, f"{source} {target}")
        return VerificationResult(
            conclusion=f"{source} —{relation_type}→ {target}",
            status=STATUS_CONTRADICTED,
            confidence=0.0,
            premises=premise_list,
            path=path_list,
            contradicting=opposing,
            counterevidence=tuple(conflict.describe() for conflict in conflicts),
            assumptions=assumptions_list,
            reasons=tuple(reasons),
        )

    available = tuple(p for p in premise_list if p.available and p.confidence >= min_premise_confidence)
    missing = tuple(p for p in premise_list if p not in available)
    endpoints_known = bool(ontology.find_concepts(source)) and bool(ontology.find_concepts(target))

    # 2. An endpoint the graph does not hold at all means we cannot even speak to the
    # claim -- that is insufficient information, not an unverified hypothesis.
    if not endpoints_known and stored is None:
        reasons.append("the graph does not hold the concepts this conclusion names")
        return VerificationResult(
            conclusion=f"{source} —{relation_type}→ {target}",
            status=STATUS_INSUFFICIENT,
            confidence=0.0,
            premises=premise_list,
            path=path_list,
            assumptions=assumptions_list,
            reasons=tuple(reasons),
        )

    # 3. Both endpoints known but the graph holds no relation and no premise or path
    # corroborates it: the conclusion is stated but nothing supports it, so it is a
    # hypothesis -- *unverified*, never forced into a verdict.
    if stored is None and not available and not path_list:
        reasons.append(
            "nothing in the graph corroborates the conclusion; treat it as a hypothesis"
        )
        return VerificationResult(
            conclusion=f"{source} —{relation_type}→ {target}",
            status=STATUS_UNVERIFIED,
            confidence=0.0,
            premises=premise_list,
            path=path_list,
            assumptions=assumptions_list,
            counterevidence=_counterevidence(ontology, source, target, relation_type),
            reasons=tuple(reasons),
        )

    # The base confidence: a stored relation's own confidence, else the mean of the
    # available premises. A graph fact is not automatically certain -- its confidence
    # is the modest, revisable figure the seed declares.
    if stored is not None:
        base = float(stored.confidence)
        reasons.append(f"the graph stores this relation (confidence {base:.2f})")
    else:
        base = sum(p.confidence for p in available) / len(available) if available else 0.0
        reasons.append(f"the conclusion rests on {len(available)} available premise(s)")

    # 4. The path corroborates: a supplied path with no stored relation is partial.
    path_intact = bool(path_list)
    if path_list and stored is None:
        reasons.append("a supplied inference path reaches the conclusion, but the graph stores no direct edge")
    if missing:
        reasons.append("missing premise(s): " + ", ".join(p.statement for p in missing))

    # 5. Context: a claim valid only elsewhere is never fully supported.
    if not context_valid:
        reasons.append("the conclusion is valid only in another context, not this one")

    # Decide the status from what is present.
    if stored is not None and not missing and context_valid:
        status = STATUS_SUPPORTED
    else:
        status = STATUS_PARTIALLY_SUPPORTED
        if stored is None and not path_intact:
            reasons.append("only premises corroborate the conclusion; no path was established")

    # Counterevidence: any other relation on the same ordered pair is worth naming,
    # even when it is not a strict antonym, so a reader sees what else the graph says.
    counterevidence = _counterevidence(ontology, source, target, relation_type)

    return VerificationResult(
        conclusion=f"{source} —{relation_type}→ {target}",
        status=status,
        confidence=round(min(1.0, base), 6),
        premises=premise_list,
        path=path_list,
        counterevidence=counterevidence,
        assumptions=assumptions_list,
        reasons=tuple(reasons),
    )


def _counterevidence(
    ontology: Ontology, source: str, target: str, relation_type: str
) -> tuple[str, ...]:
    """Other relations the graph holds on the same ordered pair, as readable lines.

    Reported so a reader can see the fuller picture; an *opposing* relation is
    already handled as a contradiction and is not repeated here.
    """
    sources = {concept.id for concept in ontology.find_concepts(source)}
    targets = {concept.id for concept in ontology.find_concepts(target)}
    if not sources or not targets:
        return ()
    lines: list[str] = []
    for relation in ontology.relations.values():
        if relation.source_id not in sources or relation.target_id not in targets:
            continue
        if relation.relation_type == relation_type or opposes(relation_type, relation.relation_type):
            continue
        lines.append(f"{source} —{relation.relation_type}→ {target}")
    return tuple(sorted(set(lines)))


@dataclass(frozen=True)
class VerificationJudgement:
    """One held-out expectation about the status a conclusion must receive."""

    name: str
    source: str
    relation_type: str
    target: str
    expect: str
    """One of :data:`STATUSES`."""

    premise_available: bool | None = None
    """Whether the graph is asked to hold a premise (``None`` means "no premise")."""

    context_valid: bool = True

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("A verification judgement needs a name.")
        if self.expect not in STATUSES:
            raise ValueError(f"Unknown expected status: {self.expect!r}")


#: The held-out verification catalogue -- the status each conclusion must receive --
#: run against a **synthetic** graph built for the gate (so the expectation is exact
#: and never depends on the seed's contents), with negative controls: a claim the
#: graph does not hold must be unverified, a claim naming an unknown concept must be
#: insufficient, and a contradicted claim must never be reported as supported.
VERIFICATION_JUDGEMENTS: tuple[VerificationJudgement, ...] = (
    VerificationJudgement(
        "stored-relation-supported", "alpha", "increases", "beta",
        expect=STATUS_SUPPORTED,
    ),
    VerificationJudgement(
        "contradiction-wins", "delta", "reduces", "epsilon",
        expect=STATUS_CONTRADICTED,
    ),
    VerificationJudgement(
        "known-pair-no-relation-unverified", "alpha", "affects", "gamma",
        expect=STATUS_UNVERIFIED,
    ),
    VerificationJudgement(
        "unknown-concept-insufficient", "zzz_unseen", "measures", "qqq_unseen",
        expect=STATUS_INSUFFICIENT,
    ),
    VerificationJudgement(
        "premise-partial", "alpha", "drives", "beta",
        expect=STATUS_PARTIALLY_SUPPORTED,
        premise_available=True,
    ),
)


def _gate_ontology() -> Ontology:
    """A small synthetic graph exercising every verification status exactly.

    Built fresh for the gate so the expectation never depends on the seed's
    contents. Two *separate* concept pairs are used so one judgement cannot bleed
    into another: ``alpha —increases→ beta`` (supported, no opposing edge on that
    pair) and ``delta —increases→ epsilon`` plus ``delta —reduces→ epsilon`` (the
    contradiction). ``gamma`` is a known concept with no relation to ``alpha``
    (unverified). Nothing is invented beyond the gate's own relations.
    """
    ontology = Ontology()
    alpha = ontology.add_concept_once("alpha")
    beta = ontology.add_concept_once("beta")
    ontology.add_concept_once("gamma")
    delta = ontology.add_concept_once("delta")
    epsilon = ontology.add_concept_once("epsilon")
    ontology.add_relation_once(alpha.id, beta.id, "increases", 0.6)
    ontology.add_relation_once(delta.id, epsilon.id, "increases", 0.6)
    ontology.add_relation_once(delta.id, epsilon.id, "reduces", 0.5)
    return ontology


def verification_accuracy(
    judgements: tuple[VerificationJudgement, ...] = VERIFICATION_JUDGEMENTS,
    *,
    ontology: Ontology | None = None,
) -> float:
    """The share of held-out judgements the verifier gets right.

    Each judgement states the status a conclusion must receive. A verifier that
    forces a conclusion when the evidence is insufficient, or reports a contradicted
    claim as supported, fails the relevant judgement. Returns ``0.0`` for an empty
    catalogue. ``ontology`` is injectable for a test; with none the gate builds its
    own synthetic graph, so the metric is deterministic and seed-independent.
    """
    catalogue = tuple(judgements)
    if not catalogue:
        return 0.0
    if ontology is None:
        ontology = _gate_ontology()
    correct = 0
    for judgement in catalogue:
        premises: tuple[Premise, ...] = ()
        if judgement.premise_available is not None:
            premises = (
                Premise(
                    statement=f"{judgement.source} {judgement.relation_type} {judgement.target}",
                    available=judgement.premise_available,
                    confidence=0.8,
                ),
            )
        result = verify_conclusion(
            judgement.source,
            judgement.relation_type,
            judgement.target,
            ontology=ontology,
            premises=premises,
            context_valid=judgement.context_valid,
        )
        if result.status == judgement.expect:
            correct += 1
    return correct / len(catalogue)


__all__ = [
    "DEFAULT_MIN_PREMISE_CONFIDENCE",
    "STATUSES",
    "STATUS_CONTRADICTED",
    "STATUS_INSUFFICIENT",
    "STATUS_LABELS",
    "STATUS_PARTIALLY_SUPPORTED",
    "STATUS_SUPPORTED",
    "STATUS_UNVERIFIED",
    "VERIFICATION_JUDGEMENTS",
    "Premise",
    "VerificationJudgement",
    "VerificationResult",
    "verification_accuracy",
    "verify_conclusion",
]
