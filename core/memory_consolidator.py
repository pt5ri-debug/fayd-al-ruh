from __future__ import annotations

from typing import Any

from core.episodic_memory import EpisodicMemory
from core.working_memory import WorkingMemory


class MemoryConsolidator:
    """Promotes important working items into durable episodic memory.

    Phase 2 extends it: when a durable long-term memory is supplied, a promoted
    item is **also** written to the durable store as an episodic record, so a
    consolidation survives the process that made it. The in-process promotion is
    unchanged, so every existing caller behaves exactly as before.
    """

    def consolidate(
        self,
        working: WorkingMemory,
        episodic: EpisodicMemory,
        minimum_activation: float = .6,
        durable: Any | None = None,
    ) -> list[str]:
        if not 0.0 <= minimum_activation <= 1.0:
            raise ValueError("Minimum activation must be between 0 and 1.")
        stored: list[str] = []
        for item in working.items():
            if item.activation >= minimum_activation and item.key not in episodic.episodes:
                episodic.store(item.key, item.content, item.activation)
                stored.append(item.key)
                if durable is not None:
                    # A promoted item becomes a durable episodic memory with the
                    # working item's own activation as its importance. The durable
                    # capture is idempotent, so a repeated consolidation does not
                    # duplicate it.
                    self._persist(durable, item)
        return stored

    @staticmethod
    def _persist(durable: Any, item: Any) -> None:
        """Write one promoted item to the durable memory, honestly typed."""
        from core.memory_record import MemoryKind

        durable.capture(
            item.content,
            kind=MemoryKind.EPISODIC,
            importance=float(getattr(item, "activation", 0.5)),
            metadata={"promoted_from": "working", "working_key": getattr(item, "key", "")},
        )

