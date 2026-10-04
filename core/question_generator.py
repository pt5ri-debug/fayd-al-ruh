from __future__ import annotations

from core.information_gap import InformationGap
from core.question import Question


class QuestionGenerator:
    """Turns prioritised information gaps into answerable evidence-seeking questions."""
    def generate(self, gaps: list[InformationGap], limit: int = 5) -> list[Question]:
        if limit < 1:
            raise ValueError("Question limit must be at least 1.")
        unresolved = (gap for gap in gaps if not gap.resolved)
        ranked = sorted(unresolved, key=lambda gap: (-gap.priority, gap.topic))[:limit]
        return [
            Question(
                text=f"What evidence would establish: {gap.topic}?",
                topic=gap.topic,
                priority=gap.priority,
            )
            for gap in ranked
        ]
