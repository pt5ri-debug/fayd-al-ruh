from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Contradiction:
    """A pair of opposing evidence records attached to the same claim."""
    supporting_evidence_id: str
    contradicting_evidence_id: str
    severity: float
