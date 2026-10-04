from __future__ import annotations

from core.episodic_memory import EpisodicMemory
from core.working_memory import WorkingMemory


class MemoryConsolidator:
    """Promotes important working items into durable episodic memory."""
    def consolidate(self, working: WorkingMemory, episodic: EpisodicMemory, minimum_activation: float = .6) -> list[str]:
        if not 0.0 <= minimum_activation <= 1.0:
            raise ValueError("Minimum activation must be between 0 and 1.")
        stored: list[str] = []
        for item in working.items():
            if item.activation >= minimum_activation and item.key not in episodic.episodes:
                episodic.store(item.key, item.content, item.activation)
                stored.append(item.key)
        return stored
