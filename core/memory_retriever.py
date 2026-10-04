from __future__ import annotations

from core.episodic_memory import EpisodicMemory, Episode
from core.memory_associator import MemoryAssociator
from core.semantic_memory import SemanticMemoryIndex


class MemoryRetriever:
    """Retrieves episodes by semantic relevance, then follows strong associations."""
    def retrieve(self, query: str, episodic: EpisodicMemory, semantic_index: SemanticMemoryIndex, associator: MemoryAssociator, limit: int = 5) -> list[Episode]:
        matches = semantic_index.search(query, limit=max(limit * 3, len(episodic.episodes)))
        ranked_ids = [match.item_id for match in matches if match.item_id in episodic.episodes and match.score > 0]
        for memory_id in list(ranked_ids):
            ranked_ids.extend(item.target_id if item.source_id == memory_id else item.source_id for item in associator.related_to(memory_id) if item.strength >= .7)
        result: list[Episode] = []
        for episode_id in dict.fromkeys(ranked_ids):
            episode = episodic.recall(episode_id)
            if episode is not None:
                result.append(episode)
            if len(result) >= limit:
                break
        return result
