from __future__ import annotations

from dataclasses import dataclass, field

from core.cognitive_state import CognitiveState
from core.state_snapshot import StateSnapshot


@dataclass
class StateHistory:
    """
    Stores the chronological evolution of cognitive state.
    """

    snapshots: list[StateSnapshot] = field(
        default_factory=list
    )

    def record(
        self,
        state: CognitiveState,
    ) -> StateSnapshot:

        snapshot = StateSnapshot.capture(
            state
        )

        self.snapshots.append(snapshot)

        return snapshot

    @property
    def size(self) -> int:
        return len(self.snapshots)

    @property
    def latest(self) -> StateSnapshot | None:
        if not self.snapshots:
            return None

        return self.snapshots[-1]

    def clear(self) -> None:
        self.snapshots.clear()

    def confidence_change(self) -> float:
        if len(self.snapshots) < 2:
            return 0.0

        first = self.snapshots[0]
        last = self.snapshots[-1]

        return round(
            last.average_belief_confidence
            - first.average_belief_confidence,
            6,
        )
