"""Source reliability, so unverified information is never stored as a fact.

The task is explicit: *do not store unreliable information as an established
fact*. This module is the one place that decides how much a statement may be
trusted, from **where it came from** and **whether it was checked**.

The model is deliberately simple and explainable:

* every :class:`~cognition.records.SourceKind` carries a base reliability
  (:data:`BASE_RELIABILITY`) -- curated seeds and the project's own corpus are
  trusted more than a generated thought or an unchecked teacher answer;
* :func:`score` adjusts the base for corroboration (independent agreement raises
  it), for the student's own confidence, and for an explicit verification;
* :func:`is_trusted` is the single gate a caller must pass before calling
  something a fact -- it requires **both** a high score and an explicit
  verification, so a high-confidence guess is still not a fact.

Dependency-free and Tk-free.
"""

from __future__ import annotations

from cognition.records import Reliability, Source, SourceKind

#: How much each origin is credited before any corroboration or check.
BASE_RELIABILITY: dict[SourceKind, float] = {
    SourceKind.CURRICULUM: 0.9,
    SourceKind.CORPUS: 0.9,
    SourceKind.ONTOLOGY: 0.85,
    SourceKind.KNOWLEDGE_BASE: 0.7,
    SourceKind.USER: 0.8,
    SourceKind.INFERENCE: 0.6,
    SourceKind.TEACHER: 0.55,
    SourceKind.INTERNET: 0.5,
    SourceKind.STUDENT: 0.4,
    SourceKind.SELF: 0.35,
    SourceKind.UNKNOWN: 0.3,
}

#: A score at or above this, **with** an explicit verification, is a fact.
TRUST_THRESHOLD = 0.75

#: How much each independent corroboration adds, and its cap.
CORROBORATION_STEP = 0.1
MAX_CORROBORATIONS = 3


def base_reliability(kind: SourceKind) -> float:
    """The credited reliability of an origin before any adjustment."""
    return BASE_RELIABILITY.get(kind, BASE_RELIABILITY[SourceKind.UNKNOWN])


def score(
    *,
    source: Source | None,
    confidence: float = 0.5,
    corroborations: int = 0,
    verified: bool = False,
) -> Reliability:
    """Score how much a statement from ``source`` can be trusted.

    The score starts from the origin's base, moves a little toward the student's
    own ``confidence``, adds a bounded amount for independent ``corroborations``,
    and is lifted to at least :data:`TRUST_THRESHOLD` when ``verified``. It never
    leaves ``[0, 1]``. ``verified`` is recorded on the result, so a caller can
    require *both* a high score and a real check.
    """
    if not 0.0 <= confidence <= 1.0:
        raise ValueError("Confidence must be between 0 and 1.")
    base = base_reliability(source.kind if source is not None else SourceKind.UNKNOWN)
    blended = 0.75 * base + 0.25 * confidence
    corroboration = min(max(corroborations, 0), MAX_CORROBORATIONS) * CORROBORATION_STEP
    value = blended + corroboration
    if verified:
        value = max(value, TRUST_THRESHOLD)
    return Reliability(
        score=round(min(max(value, 0.0), 1.0), 6),
        source=source,
        verified=verified,
    )


def is_trusted(reliability: Reliability) -> bool:
    """True only when a statement may be called a fact.

    Requires a score at or above :data:`TRUST_THRESHOLD` **and** an explicit
    verification, so a merely confident statement is never a fact.
    """
    return reliability.verified and reliability.score >= TRUST_THRESHOLD


__all__ = [
    "BASE_RELIABILITY",
    "CORROBORATION_STEP",
    "MAX_CORROBORATIONS",
    "TRUST_THRESHOLD",
    "base_reliability",
    "is_trusted",
    "score",
]
