from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class CurriculumItem:
    topic: str
    difficulty: float
    prerequisites: frozenset[str] = field(default_factory=frozenset)

    def __post_init__(self) -> None:
        if not self.topic.strip() or not 0.0 <= self.difficulty <= 1.0:
            raise ValueError("Curriculum topic and difficulty are required.")


class Curriculum:
    """Selects an achievable next topic from explicitly mastered prerequisites."""
    def next_item(self, items: list[CurriculumItem], mastered: set[str]) -> CurriculumItem | None:
        available = [item for item in items if item.topic not in mastered and item.prerequisites <= mastered]
        return min(available, key=lambda item: (item.difficulty, item.topic)) if available else None
