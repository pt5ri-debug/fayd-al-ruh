from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class MemoryAssociation:
    source_id: str
    target_id: str
    strength: float


class MemoryAssociator:
    def __init__(self) -> None:
        self._associations: dict[tuple[str, str], MemoryAssociation] = {}

    def associate(self, source_id: str, target_id: str, strength: float = .5) -> MemoryAssociation:
        if not source_id.strip() or not target_id.strip() or source_id == target_id:
            raise ValueError("Association needs two distinct memory IDs.")
        if not 0.0 <= strength <= 1.0:
            raise ValueError("Association strength must be between 0 and 1.")
        association = MemoryAssociation(source_id, target_id, strength)
        self._associations[(source_id, target_id)] = association
        return association

    def related_to(self, memory_id: str) -> list[MemoryAssociation]:
        return [item for item in self._associations.values() if item.source_id == memory_id or item.target_id == memory_id]
