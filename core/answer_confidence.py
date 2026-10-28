"""Fold the graph's reasoning evidence into the *confidence* of an answer.

:class:`~core.assistant.AssistantAnswer` has always carried a confidence, but that
number was computed from the retrieved passages alone -- coverage and strength --
and ignored every other thing the answer path now knows: the relations the graph
*entails* (:mod:`core.relation_inference`), the chains it *walks*
(:mod:`core.relation_connections`), the causal paths it *states*
(:mod:`core.causal_paths`), and the claims it holds **contradictorily**
(:mod:`core.relation_consistency`). So an answer resting on a graph-derived
mechanism could not read as more confident than one resting on a single weak
passage, and an answer the graph *contradicts* could not read as less.

This module is the one place that number is computed. It is **deterministic** and
**bounded**, and it is honest by construction: it never lets a derived signal
carry more than a modest share of the confidence, it never invents confidence from
a graph when there is no retrieved evidence to be confident *about*, it never lets
a contradiction be silently outvoted by derived support, and it returns the whole
breakdown (`reasons`) so a reader can see exactly which term moved the number and
by how much. Nothing here asserts anything into the graph; it only weighs what the
answer path already found.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field

from core.rag import RetrievedPassage

#: How much of the *remaining* room to certainty a graph-derived signal may fill.
#: Deliberately modest: a derivation corroborates an answer, it does not replace
#: the retrieved evidence the answer is actually built from.
GRAPH_SUPPORT_WEIGHT = 0.25

#: How strongly a stored contradiction pulls the confidence down, per unit of
#: severity. A contradiction is a reason to trust the answer *less*, so this term
#: can remove more than the support term can add.
CONFLICT_PENALTY_WEIGHT = 0.5

#: How many retrieved passages count as "full coverage" in the evidence term.
COVERAGE_TARGET = 4


@dataclass(frozen=True)
class AnswerConfidence:
    """An answer's confidence with the terms it was built from, in the open."""

    evidence: float
    """The passage-only term: the strength and coverage of the retrieved evidence."""

    graph_support: float
    """The strongest graph-derived signal corroborating the answer (0 when none)."""

    conflict_penalty: float
    """How much the stored contradictions removed (0 when the graph is consistent)."""

    final: float
    """The confidence the answer carries, after support is added and conflict removed."""

    reasons: tuple[str, ...] = field(default_factory=tuple)
    """One readable line per term that actually moved the number, for the record."""

    def __post_init__(self) -> None:
        for name in ("evidence", "graph_support", "conflict_penalty", "final"):
            value = getattr(self, name)
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be between 0 and 1.")

    def describe(self) -> str:
        """The confidence with its terms, e.g. ``0.72 (evidence 0.65 + support 0.20 - conflict 0.10)``."""
        if self.graph_support == 0.0 and self.conflict_penalty == 0.0:
            return f"{self.final:.2f} (from the retrieved evidence alone)"
        return f"{self.final:.2f} (" + "; ".join(self.reasons) + ")"


def evidence_confidence(passages: Iterable[RetrievedPassage]) -> float:
    """The passage-only confidence: the strength and coverage of the evidence.

    The project's original formula, kept in one place so the answer path and its
    tests agree on the base term. ``0.0`` when nothing was retrieved.
    """
    items = list(passages)
    if not items:
        return 0.0
    top = items[0].score
    coverage = min(len(items), COVERAGE_TARGET) / COVERAGE_TARGET
    return round(min(1.0, 0.7 * top + 0.3 * coverage), 6)


def _strongest_support(
    inferences: Iterable[object],
    connections: Iterable[object],
    causal_paths: Iterable[object],
) -> float:
    """The single strongest confidence among the graph-derived signals.

    The **maximum**, not the sum: three weak derivations of the same fact are one
    piece of support, not three. A signal with no usable confidence is skipped.
    """
    best = 0.0
    for group in (inferences, connections, causal_paths):
        for item in group:
            value = getattr(item, "confidence", None)
            if isinstance(value, (int, float)) and 0.0 <= value <= 1.0:
                best = max(best, float(value))
    return best


def _conflict_severity(conflicts: Iterable[object]) -> float:
    """The total severity of the stored contradictions, capped at 1.0.

    Severities are summed (two independent contradictions are worse than one) but
    the total is capped, so a heavily contradicted graph cannot drive the number
    past the floor the penalty weight allows.
    """
    total = 0.0
    for conflict in conflicts:
        value = getattr(conflict, "severity", None)
        if isinstance(value, (int, float)) and value > 0:
            total += float(value)
    return min(1.0, total)


def answer_confidence(
    passages: Iterable[RetrievedPassage],
    *,
    inferences: Iterable[object] = (),
    connections: Iterable[object] = (),
    causal_paths: Iterable[object] = (),
    conflicts: Iterable[object] = (),
) -> AnswerConfidence:
    """Weigh the answer's confidence from its evidence and the graph's reasoning.

    The evidence term is the base. A graph-derived signal may then fill at most
    :data:`GRAPH_SUPPORT_WEIGHT` of the room between the base and certainty -- and
    **only when there is a base to fill**: with no retrieved evidence the
    confidence stays ``0.0``, so a graph fact can never manufacture confidence in
    an answer the sources do not support. Stored contradictions are applied last
    and multiply the result down, so a contradiction is never outvoted by derived
    support. The returned :class:`AnswerConfidence` carries the terms and the
    reasons that moved them.
    """
    evidence = evidence_confidence(passages)
    support = _strongest_support(inferences, connections, causal_paths)
    severity = _conflict_severity(conflicts)

    reasons: list[str] = []
    value = evidence

    if support > 0.0 and evidence > 0.0:
        added = (1.0 - evidence) * GRAPH_SUPPORT_WEIGHT * support
        value = min(1.0, value + added)
        # Only name the support when it is large enough to show: a term that
        # rounds to 0.00 is noise in the record, not a reason.
        if round(added, 2) > 0:
            reasons.append(f"evidence {evidence:.2f} + graph support {added:.2f}")

    penalty = 0.0
    if severity > 0.0:
        factor = 1.0 - CONFLICT_PENALTY_WEIGHT * severity
        before = value
        value = max(0.0, value * factor)
        penalty = round(before - value, 6)
        reasons.append(f"conflict -{penalty:.2f}")

    if not reasons:
        reasons.append(f"evidence {evidence:.2f}")

    return AnswerConfidence(
        evidence=round(evidence, 6),
        graph_support=round(support, 6),
        conflict_penalty=round(penalty, 6),
        final=round(value, 6),
        reasons=tuple(reasons),
    )


@dataclass(frozen=True)
class ConfidenceSignal:
    """A minimal graph-derived signal: just the confidence it corroborates with.

    The confidence model reads only a ``.confidence`` attribute off the signals it
    is handed, so this is the shape a caller (or the gate) can pass when it has a
    number but not the full ``InferredRelation``/``ConceptConnection``/explanation.
    """

    confidence: float

    def __post_init__(self) -> None:
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("A signal's confidence must be between 0 and 1.")


@dataclass(frozen=True)
class ConfidenceConflict:
    """A minimal contradiction: just the severity it removes confidence with."""

    severity: float

    def __post_init__(self) -> None:
        if not 0.0 <= self.severity <= 1.0:
            raise ValueError("A conflict's severity must be between 0 and 1.")


@dataclass(frozen=True)
class ConfidenceJudgement:
    """One held-out expectation about how a term must move the confidence."""

    name: str
    """What is being judged, e.g. ``support-raises``."""

    evidence: tuple[float, ...]
    """The passage scores the answer rests on."""

    support: float = 0.0
    """The strongest graph-derived signal, or 0 for none."""

    severity: float = 0.0
    """The total contradiction severity, or 0 for none."""

    expect: str = "bounded"
    """``support-raises``, ``conflict-lowers``, ``no-evidence-zero``, or ``bounded``."""

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("A confidence judgement needs a name.")
        if self.expect not in {"support-raises", "conflict-lowers", "no-evidence-zero", "bounded"}:
            raise ValueError(f"Unknown expectation: {self.expect!r}")
        for score in self.evidence:
            if not 0.0 <= score <= 1.0:
                raise ValueError("Passage scores must be between 0 and 1.")


#: The held-out confidence catalogue -- the direction each term must move the
#: number, written by hand, so a change that lets the graph *manufacture*
#: confidence, lets a contradiction be outvoted, or lets the number leave [0, 1]
#: is a measured regression rather than a silent one.
CONFIDENCE_JUDGEMENTS: tuple[ConfidenceJudgement, ...] = (
    ConfidenceJudgement("support-raises", (0.5,), support=0.8, expect="support-raises"),
    ConfidenceJudgement("support-raises-weak", (0.2,), support=1.0, expect="support-raises"),
    ConfidenceJudgement("conflict-lowers", (0.8,), severity=0.6, expect="conflict-lowers"),
    ConfidenceJudgement("conflict-lowers-strong", (0.9,), severity=1.0, expect="conflict-lowers"),
    ConfidenceJudgement("no-evidence-zero", (), support=1.0, expect="no-evidence-zero"),
    ConfidenceJudgement("bounded-high", (1.0,), support=1.0, expect="bounded"),
    ConfidenceJudgement("bounded-conflict", (0.0,), severity=1.0, expect="bounded"),
    ConfidenceJudgement("support-and-conflict", (0.6,), support=0.9, severity=0.5, expect="bounded"),
)


def answer_confidence_accuracy(
    judgements: tuple[ConfidenceJudgement, ...] = CONFIDENCE_JUDGEMENTS,
) -> float:
    """The share of held-out expectations the confidence model gets right.

    Each judgement states how a term *must* move the number: graph support raises
    it, a contradiction lowers it, no evidence pins it at zero, and the result is
    always inside ``[0, 1]``. A model that let the graph manufacture confidence,
    let a contradiction be outvoted, or produced an out-of-range value fails the
    relevant judgement. Returns ``0.0`` for an empty catalogue.
    """
    catalogue = tuple(judgements)
    if not catalogue:
        return 0.0
    correct = 0
    for judgement in catalogue:
        passages = [_passage_for(score) for score in judgement.evidence]
        report = answer_confidence(
            passages,
            causal_paths=[ConfidenceSignal(judgement.support)] if judgement.support else (),
            conflicts=[ConfidenceConflict(judgement.severity)] if judgement.severity else (),
        )
        base = evidence_confidence(passages)
        if not 0.0 <= report.final <= 1.0:
            continue
        if judgement.expect == "support-raises" and report.final > base:
            correct += 1
        elif judgement.expect == "conflict-lowers" and report.final < base:
            correct += 1
        elif judgement.expect == "no-evidence-zero" and report.final == 0.0:
            correct += 1
        elif judgement.expect == "bounded":
            correct += 1
    return correct / len(catalogue)


def _passage_for(score: float) -> RetrievedPassage:
    """A minimal retrieved passage at ``score``, for the confidence gate."""
    return RetrievedPassage(
        doc_id="gate", title="gate", text="gate", source="gate", score=score, signal="lexical"
    )


__all__ = [
    "CONFIDENCE_JUDGEMENTS",
    "CONFLICT_PENALTY_WEIGHT",
    "COVERAGE_TARGET",
    "GRAPH_SUPPORT_WEIGHT",
    "AnswerConfidence",
    "ConfidenceConflict",
    "ConfidenceJudgement",
    "ConfidenceSignal",
    "answer_confidence",
    "answer_confidence_accuracy",
    "evidence_confidence",
]
