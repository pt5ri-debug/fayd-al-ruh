from __future__ import annotations

from typing import Any

from core.episodic_memory import Episode, EpisodicMemory
from core.memory_associator import MemoryAssociator
from core.semantic_memory import SemanticMemoryIndex


class MemoryRetriever:
    """Retrieves episodes by semantic relevance, then follows strong associations.

    Phase 2 extends it: when a durable long-term memory is supplied, its
    **episodic** records are a fourth source beside the in-process lexical and
    semantic ones, so an episode stored in an earlier session is retrievable now.
    With no durable store the behaviour is exactly as before.
    """

    def retrieve(
        self,
        query: str,
        episodic: EpisodicMemory,
        semantic_index: SemanticMemoryIndex,
        associator: MemoryAssociator,
        limit: int = 5,
        durable: Any | None = None,
    ) -> list[Episode]:
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
        if durable is not None and len(result) < limit:
            result.extend(self._durable_episodes(durable, query, limit - len(result)))
        return result

    @staticmethod
    def _durable_episodes(durable: Any, query: str, limit: int) -> list[Episode]:
        """Episodic records from the durable store, as ``Episode`` objects.

        Only episodic records are surfaced here -- the durable semantic and
        procedural categories have their own stores and their own retriever
        (:mod:`core.memory_retrieval`), so this fourth source stays an *episodic*
        one and does not duplicate them.
        """
        from core.memory_record import MemoryKind
        from core.memory_retrieval import MemoryRetriever as DurableRetriever

        report = DurableRetriever(durable).retrieve(
            query, kinds=(MemoryKind.EPISODIC,), limit=limit, touch=False
        )
        episodes: list[Episode] = []
        for hit in report.hits:
            record = hit.record
            episode = Episode(id=record.record_id, content=record.content, importance=record.importance)
            episodes.append(episode)
        return episodes

