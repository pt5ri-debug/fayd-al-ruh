from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal

from core.memory import CognitiveMemory, MemoryItem


@dataclass
class Pattern:
    """
    Represents a recurring pattern discovered in memory.
    """

    key: str
    occurrences: int
    confidence: float
    memory_ids: list[str]


class PatternDetector:
    """
    Detects recurring patterns in cognitive memory.

    This component discovers patterns only.
    It does not create or modify beliefs.
    """

    def detect(
        self,
        memory: CognitiveMemory,
        item_type: str = "observation",
        minimum_occurrences: int = 2,
    ) -> list[Pattern]:

        if minimum_occurrences < 1:
            raise ValueError(
                "minimum_occurrences must be at least 1."
            )

        items = memory.find_by_type(item_type)

        grouped: dict[str, list[MemoryItem]] = {}

        for item in items:
            key = self._make_key(item)

            grouped.setdefault(
                key,
                [],
            ).append(item)

        patterns: list[Pattern] = []

        for key, group in grouped.items():

            if len(group) < minimum_occurrences:
                continue

            confidence = self._calculate_confidence(group)

            patterns.append(
                Pattern(
                    key=key,
                    occurrences=len(group),
                    confidence=confidence,
                    memory_ids=[
                        item.item_id
                        for item in group
                    ],
                )
            )

        return patterns

    @staticmethod
    def _make_key(item: MemoryItem) -> str:

        content = item.content

        if hasattr(content, "content"):
            return str(content.content).strip().lower()

        return str(content).strip().lower()

    @staticmethod
    def _calculate_confidence(
        items: list[MemoryItem],
    ) -> float:

        if not items:
            return 0.0

        values = [
            Decimal(str(item.importance))
            for item in items
        ]

        total = sum(values, Decimal("0"))

        confidence = (
            total / Decimal(len(values))
        )

        confidence = confidence.quantize(
            Decimal("0.000001")
        )

        confidence = max(
            Decimal("0"),
            min(
                Decimal("1"),
                confidence,
            ),
        )

        return float(confidence)