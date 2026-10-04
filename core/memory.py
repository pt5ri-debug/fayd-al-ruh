from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Iterable
import re

from core.semantic_memory import SemanticMemoryIndex
from core.episodic_memory import EpisodicMemory
from core.memory_associator import MemoryAssociator
from core.memory_consolidator import MemoryConsolidator
from core.procedural_memory import ProceduralMemory
from core.working_memory import WorkingMemory


@dataclass
class MemoryItem:
    """
    A single item stored in cognitive memory.
    """

    item_id: str

    item_type: str

    content: Any

    importance: float = 0.5

    created_at: datetime = field(
        default_factory=lambda: datetime.now(timezone.utc)
    )

    access_count: int = 0


@dataclass(frozen=True)
class MemoryRecall:
    """A memory selected for the current working context."""

    item: MemoryItem
    relevance: float
    activation: float
    semantic_similarity: float = 0.0


class CognitiveMemory:
    """
    Basic cognitive memory.

    Stores observations, evidence, beliefs,
    and other cognitive objects by ID.
    """

    def __init__(self) -> None:
        self._items: dict[str, MemoryItem] = {}
        self.semantic_index = SemanticMemoryIndex()
        self.working_memory = WorkingMemory()
        self.episodic_memory = EpisodicMemory()
        self.procedural_memory = ProceduralMemory()
        self.associator = MemoryAssociator()
        self.consolidator = MemoryConsolidator()

    def store(
        self,
        item_id: str,
        item_type: str,
        content: Any,
        importance: float = 0.5,
    ) -> MemoryItem:

        if not item_id.strip():
            raise ValueError(
                "Memory item ID cannot be empty."
            )

        if not item_type.strip():
            raise ValueError(
                "Memory item type cannot be empty."
            )

        if not 0.0 <= importance <= 1.0:
            raise ValueError(
                "Memory importance must be between 0 and 1."
            )

        if item_id in self._items:
            raise ValueError(
                f"Memory item '{item_id}' already exists."
            )

        item = MemoryItem(
            item_id=item_id,
            item_type=item_type,
            content=content,
            importance=importance,
        )

        self._items[item_id] = item
        self.semantic_index.add(item_id, self._text_for(content))
        self.working_memory.put(item_id, content, importance)
        if item_type == "observation":
            self.episodic_memory.store(item_id, content, importance)

        return item

    def get(
        self,
        item_id: str,
    ) -> MemoryItem | None:

        item = self._items.get(item_id)

        if item is not None:
            item.access_count += 1

        return item

    def remove(self, item_id: str) -> None:
        self._items.pop(item_id, None)
        self.semantic_index.remove(item_id)
        self.working_memory.remove(item_id)
        self.episodic_memory.episodes.pop(item_id, None)

    def find_by_type(
        self,
        item_type: str,
    ) -> list[MemoryItem]:

        return [
            item
            for item in self._items.values()
            if item.item_type == item_type
        ]

    def important_items(
        self,
        minimum_importance: float = 0.5,
    ) -> list[MemoryItem]:

        return [
            item
            for item in self._items.values()
            if item.importance >= minimum_importance
        ]

    def recall(
        self,
        query: str,
        *,
        item_types: Iterable[str] | None = None,
        limit: int = 5,
        minimum_relevance: float = 0.0,
    ) -> list[MemoryRecall]:
        """Retrieve a small, ranked working-memory context.

        Ranking combines lexical relevance, stored importance, and previous
        use.  It is deliberately deterministic and dependency-free, so it can
        later be replaced by embedding/vector retrieval without changing the
        controller API.
        """

        if not query.strip():
            raise ValueError("Recall query cannot be empty.")

        if limit < 1:
            raise ValueError("Recall limit must be at least 1.")

        if not 0.0 <= minimum_relevance <= 1.0:
            raise ValueError(
                "Minimum relevance must be between 0 and 1."
            )

        allowed_types = set(item_types) if item_types is not None else None
        query_tokens = self._tokens(query)
        semantic_scores = {
            match.item_id: max(0.0, match.score)
            for match in self.semantic_index.search(query, limit=max(limit * 4, self.size))
        }
        recalls: list[MemoryRecall] = []

        for item in self._items.values():
            if allowed_types is not None and item.item_type not in allowed_types:
                continue

            lexical_relevance = self._similarity(
                query_tokens,
                self._tokens(self._text_for(item.content)),
            )
            semantic_similarity = semantic_scores.get(item.item_id, 0.0)
            relevance = max(lexical_relevance, semantic_similarity)

            if relevance < minimum_relevance:
                continue

            familiarity = min(item.access_count, 5) / 5
            activation = round(
                0.7 * relevance + 0.2 * item.importance + 0.1 * familiarity,
                6,
            )
            recalls.append(
                MemoryRecall(
                    item=item,
                    relevance=round(relevance, 6),
                    activation=activation,
                    semantic_similarity=round(semantic_similarity, 6),
                )
            )

        recalls.sort(
            key=lambda recall: (
                -recall.activation,
                -recall.relevance,
                recall.item.item_id,
            )
        )
        selected = recalls[:limit]

        for recall in selected:
            recall.item.access_count += 1

        return selected

    @staticmethod
    def _text_for(content: Any) -> str:
        return str(getattr(content, "content", content))

    @staticmethod
    def _tokens(value: str) -> set[str]:
        return set(re.findall(r"\w+", value.casefold(), flags=re.UNICODE))

    @staticmethod
    def _similarity(left: set[str], right: set[str]) -> float:
        if not left or not right:
            return 0.0
        return len(left & right) / len(left | right)

    @property
    def size(self) -> int:
        return len(self._items)

    def clear(self) -> None:
        self._items.clear()
        self.semantic_index = SemanticMemoryIndex()
        self.working_memory = WorkingMemory()
        self.episodic_memory = EpisodicMemory()
        self.procedural_memory = ProceduralMemory()
        self.associator = MemoryAssociator()
        self.consolidator = MemoryConsolidator()
