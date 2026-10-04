from __future__ import annotations

from dataclasses import dataclass

from core.belief import BeliefStatus
from core.belief_conflict import BeliefConflict


@dataclass(frozen=True)
class ConflictResolution:
    resolved_belief_id: str | None
    suspended_belief_ids: tuple[str, ...]
    reason: str


class ConflictResolver:
    """Conservative resolver: evidence gap may suspend one side, otherwise both."""
    def __init__(self, minimum_confidence_gap: float = .15) -> None:
        if not 0.0 <= minimum_confidence_gap <= 1.0:
            raise ValueError("Minimum confidence gap must be between 0 and 1.")
        self.minimum_confidence_gap = minimum_confidence_gap

    def resolve(self, conflict: BeliefConflict) -> ConflictResolution:
        first, second = conflict.first, conflict.second
        gap = abs(first.confidence - second.confidence)
        if gap < self.minimum_confidence_gap:
            first.set_status(BeliefStatus.SUSPENDED)
            second.set_status(BeliefStatus.SUSPENDED)
            return ConflictResolution(None, (first.id, second.id), "Evidence is too close to select either belief.")
        winner, loser = (first, second) if first.confidence > second.confidence else (second, first)
        loser.set_status(BeliefStatus.SUSPENDED)
        return ConflictResolution(winner.id, (loser.id,), "Higher-confidence belief retained pending further evidence.")
