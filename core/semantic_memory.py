from __future__ import annotations

import re
from dataclasses import dataclass
from hashlib import blake2b
from math import sqrt


def hashed_embedding(text: str, dimensions: int = 128) -> tuple[float, ...]:
    """Create a deterministic local semantic fingerprint.

    This avoids a mandatory cloud dependency.  A production embedding provider
    can replace this function while keeping ``SemanticMemoryIndex`` unchanged.
    """
    if dimensions < 8:
        raise ValueError("Embedding dimensions must be at least 8.")

    vector = [0.0] * dimensions
    tokens = re.findall(r"\w+", text.casefold(), flags=re.UNICODE)
    for token in tokens:
        digest = blake2b(token.encode("utf-8"), digest_size=8).digest()
        index = int.from_bytes(digest[:4], "big") % dimensions
        vector[index] += 1.0 if digest[4] % 2 else -1.0
    return tuple(vector)


@dataclass(frozen=True)
class SemanticMatch:
    item_id: str
    score: float


class SemanticMemoryIndex:
    """In-memory cosine-similarity index for episodic memory."""

    def __init__(self, dimensions: int = 128) -> None:
        self.dimensions = dimensions
        self._vectors: dict[str, tuple[float, ...]] = {}

    def add(self, item_id: str, text: str) -> None:
        self._vectors[item_id] = hashed_embedding(text, self.dimensions)

    def remove(self, item_id: str) -> None:
        self._vectors.pop(item_id, None)

    def search(self, query: str, limit: int = 5) -> list[SemanticMatch]:
        if not query.strip():
            raise ValueError("Semantic query cannot be empty.")
        if limit < 1:
            raise ValueError("Search limit must be at least 1.")

        query_vector = hashed_embedding(query, self.dimensions)
        matches = [
            SemanticMatch(item_id, round(self._cosine(query_vector, vector), 6))
            for item_id, vector in self._vectors.items()
        ]
        return sorted(matches, key=lambda match: (-match.score, match.item_id))[:limit]

    @staticmethod
    def _cosine(left: tuple[float, ...], right: tuple[float, ...]) -> float:
        left_norm = sqrt(sum(value * value for value in left))
        right_norm = sqrt(sum(value * value for value in right))
        if left_norm == 0 or right_norm == 0:
            return 0.0
        return sum(a * b for a, b in zip(left, right, strict=False)) / (left_norm * right_norm)


class SemanticMemory:
    """Durable, decontextualised facts backed by a semantic retrieval index."""
    def __init__(self) -> None:
        self.facts: dict[str, str] = {}
        self.index = SemanticMemoryIndex()

    def learn(self, fact_id: str, statement: str) -> None:
        if not fact_id.strip() or not statement.strip():
            raise ValueError("Semantic fact ID and statement cannot be empty.")
        self.facts[fact_id] = statement
        self.index.add(fact_id, statement)

    def retrieve(self, query: str, limit: int = 5) -> list[str]:
        return [match.item_id for match in self.index.search(query, limit) if match.score > 0]
