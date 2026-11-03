"""One explainable confidence evaluation for a knowledge claim.

The project already had a confidence *model* (:mod:`core.confidence_model`, an
evidence-weighted posterior) and an *evidence weighting* rule
(:mod:`core.evidence_weighting`, strength x source reliability). What it did not
have was a single, **explainable** evaluation that a reader can audit: which
terms moved the number, and by how much.

This module is that evaluation. ``evaluate_confidence`` takes a claim's evidence
and returns a :class:`ConfidenceEvaluation` whose ``terms`` name every
contribution -- base prior, supporting weight, contradicting weight, source
diversity, recency -- so the figure is *derived*, not asserted. It builds on the
existing model rather than replacing it (the posterior is
``ConfidenceModel.update``'s), and it keeps three properties the task requires
**distinct**:

* **confidence** -- the evidence-weighted posterior (this module);
* **source reliability** -- a property of the *source*, read by
  ``EvidenceWeighting`` (never folded into the number as if it were truth);
* **verification status** -- the belief lifecycle (``BeliefStatus``), which this
  module reports but never sets.

Two honesty rules, stated in the code:

* a confidence score is **not proof of truth**, and this module says so in its
  own output (:attr:`ConfidenceEvaluation.note`);
* **independent sources** are counted by distinct ``source``, not by evidence
  count, so three snippets from one page cannot masquerade as three witnesses.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from datetime import UTC, datetime

from core.confidence_model import ConfidenceModel
from core.evidence import Evidence
from core.evidence_weighting import EvidenceWeighting

#: A half-life (in days) for recency weighting, or ``None`` to ignore time.
#: Recency is *relevant where it is relevant*: a measurement decays, a
#: definition does not. The caller chooses by passing ``recency_half_life_days``.
DEFAULT_RECENCY_HALF_LIFE_DAYS: float | None = None

#: The most independent sources a claim is credited for. Beyond this the
#: diversity bonus saturates, so a claim cannot be "proved" by sheer volume.
MAX_INDEPENDENT_SOURCES = 8

CONFIDENCE_NOTE = (
    "Confidence is an evidence-weighted posterior, not proof of truth; "
    "source reliability and verification status are separate properties."
)


@dataclass(frozen=True)
class ConfidenceTerm:
    """One named contribution to the evaluation, with its value and its weight."""

    name: str
    value: float
    detail: str = ""

    def describe(self) -> str:
        base = f"{self.name}: {self.value:.3f}"
        return f"{base} ({self.detail})" if self.detail else base


@dataclass(frozen=True)
class ConfidenceEvaluation:
    """The explainable result of evaluating a claim's evidence.

    ``confidence`` is the posterior; ``terms`` is the breakdown; ``sources`` is
    the set of distinct sources; ``verified`` is the claim's lifecycle status
    *if the caller passed one* (this module never sets it).
    """

    confidence: float
    terms: tuple[ConfidenceTerm, ...] = ()
    support_weight: float = 0.0
    contradict_weight: float = 0.0
    evidence_count: int = 0
    source_count: int = 0
    sources: tuple[str, ...] = ()
    verified: str = ""
    note: str = CONFIDENCE_NOTE
    metadata: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("Confidence must be between 0 and 1.")

    def term(self, name: str) -> ConfidenceTerm | None:
        for item in self.terms:
            if item.name == name:
                return item
        return None

    def describe(self) -> str:
        lines = [f"confidence: {self.confidence:.3f}"]
        lines.extend(f"  {term.describe()}" for term in self.terms)
        if self.sources:
            lines.append(f"  sources: {', '.join(self.sources)}")
        if self.verified:
            lines.append(f"  status: {self.verified}")
        lines.append(f"  note: {self.note}")
        return "\n".join(lines)

    def to_primitive(self) -> dict[str, object]:
        return {
            "confidence": self.confidence,
            "support_weight": self.support_weight,
            "contradict_weight": self.contradict_weight,
            "evidence_count": self.evidence_count,
            "source_count": self.source_count,
            "sources": list(self.sources),
            "verified": self.verified,
            "terms": [{"name": t.name, "value": t.value, "detail": t.detail} for t in self.terms],
        }


def _recency_factor(evidence: Evidence, half_life_days: float | None, *, now: datetime) -> float:
    """A [0, 1] factor: 1.0 today, 0.5 one half-life ago, decaying after."""
    if half_life_days is None or half_life_days <= 0:
        return 1.0
    created = getattr(evidence, "created_at", None)
    if not isinstance(created, datetime):
        return 1.0
    if created.tzinfo is None:
        created = created.replace(tzinfo=UTC)
    age_days = max(0.0, (now - created).total_seconds() / 86400.0)
    return float(0.5 ** (age_days / half_life_days))


def evaluate_confidence(
    evidence: Iterable[Evidence],
    *,
    prior: float = 0.5,
    prior_strength: float = 1.0,
    weighting: EvidenceWeighting | None = None,
    model: ConfidenceModel | None = None,
    recency_half_life_days: float | None = DEFAULT_RECENCY_HALF_LIFE_DAYS,
    now: datetime | None = None,
    verified: str = "",
) -> ConfidenceEvaluation:
    """Evaluate a claim's evidence into an explainable confidence.

    ``prior`` is the belief's own starting confidence (its previous posterior);
    ``verified`` is the claim's lifecycle status, passed *through* for reporting
    only -- this function never changes a belief. The posterior is the project's
    own :class:`~core.confidence_model.ConfidenceModel` result, so this is an
    explanation of the existing number, not a second, competing one.
    """
    items = list(evidence)
    if not 0.0 <= prior <= 1.0:
        raise ValueError("Prior confidence must be between 0 and 1.")

    weighting = weighting or EvidenceWeighting()
    model = model or ConfidenceModel(prior_strength=prior_strength, weighting=weighting)
    moment = now or datetime.now(UTC)

    support = 0.0
    contradict = 0.0
    sources: list[str] = []
    for item in items:
        weight = weighting.weight(item) * _recency_factor(item, recency_half_life_days, now=moment)
        if item.supports:
            support += weight
        else:
            contradict += weight
        if item.source not in sources:
            sources.append(item.source)

    posterior = model.update(prior, items).confidence

    independent = min(len(sources), MAX_INDEPENDENT_SOURCES)
    diversity = independent / MAX_INDEPENDENT_SOURCES if MAX_INDEPENDENT_SOURCES else 0.0

    terms = (
        ConfidenceTerm("prior", round(prior, 6), f"strength {prior_strength:g}"),
        ConfidenceTerm("support", round(support, 6), f"{sum(1 for i in items if i.supports)} item(s)"),
        ConfidenceTerm("contradict", round(contradict, 6), f"{sum(1 for i in items if not i.supports)} item(s)"),
        ConfidenceTerm("source_diversity", round(diversity, 6), f"{independent} independent source(s)"),
        ConfidenceTerm("posterior", round(posterior, 6), "evidence-weighted"),
    )

    return ConfidenceEvaluation(
        confidence=round(posterior, 6),
        terms=terms,
        support_weight=round(support, 6),
        contradict_weight=round(contradict, 6),
        evidence_count=len(items),
        source_count=len(sources),
        sources=tuple(sources),
        verified=verified,
        metadata={"recency_half_life_days": recency_half_life_days},
    )


__all__ = [
    "CONFIDENCE_NOTE",
    "DEFAULT_RECENCY_HALF_LIFE_DAYS",
    "MAX_INDEPENDENT_SOURCES",
    "ConfidenceEvaluation",
    "ConfidenceTerm",
    "evaluate_confidence",
]
