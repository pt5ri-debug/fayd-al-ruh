"""Motivation: decide which topics deserve attention, and say why.

The mind cannot pursue everything at once. This engine scores each candidate
topic on four named factors and returns a ranked, bounded list of
:class:`Drive`s:

* **gap** -- how much knowledge is missing (from
  :class:`~core.information_gap.InformationGap.priority`);
* **goal** -- how well the topic serves the project's current goals (a goal's
  words overlapping the topic);
* **novelty** -- how far the topic is from what has recently been thought;
* **feasibility** -- whether the graph can actually say anything about it (a
  topic naming known concepts is more feasible than one that names none).

The weights are explicit constants, so a reader can see exactly why a topic was
ranked where it was. Nothing here is random.
"""

from __future__ import annotations

from dataclasses import dataclass

from core.goal import Goal
from core.information_gap import InformationGap
from core.ontology import Ontology
from core.tokenize import normalize_token

from cognition.engines.ideas import NoveltyLike, NoveltyTracker

#: The weights of the four factors, summing to 1.0.
WEIGHTS: dict[str, float] = {
    "gap": 0.4,
    "goal": 0.3,
    "novelty": 0.2,
    "feasibility": 0.1,
}


@dataclass(frozen=True)
class Drive:
    """One ranked reason to pursue a topic, with its factor breakdown."""

    topic: str
    score: float
    factors: dict[str, float]

    def __post_init__(self) -> None:
        if not self.topic.strip():
            raise ValueError("Drive topic cannot be empty.")
        if not 0.0 <= self.score <= 1.0:
            raise ValueError("Drive score must be between 0 and 1.")

    def describe(self) -> str:
        parts = ", ".join(f"{k} {v:.2f}" for k, v in sorted(self.factors.items()))
        return f"{self.topic}: {self.score:.2f} ({parts})"


class MotivationEngine:
    """Rank topics by gap, goal fit, novelty, and feasibility."""

    def __init__(
        self,
        ontology: Ontology,
        *,
        novelty: NoveltyLike | None = None,
    ) -> None:
        self.ontology = ontology
        self.novelty: NoveltyLike = novelty or NoveltyTracker()
        self._labels = {c.label for c in ontology.concepts.values()}

    def rank(
        self,
        *,
        gaps: tuple[InformationGap, ...] = (),
        goals: tuple[Goal, ...] = (),
        limit: int = 10,
    ) -> tuple[Drive, ...]:
        """Return the most motivating topics, strongest first, bounded."""
        if limit < 1:
            raise ValueError("limit must be at least 1.")
        drives: list[Drive] = []
        for gap in gaps:
            factors = {
                "gap": round(gap.priority, 6),
                "goal": round(self._goal_fit(gap.topic, goals), 6),
                "novelty": round(self.novelty.novelty(gap.topic), 6),
                "feasibility": 1.0 if gap.topic in self._labels else 0.4,
            }
            score = round(sum(WEIGHTS[k] * factors[k] for k in WEIGHTS), 6)
            drives.append(Drive(topic=gap.topic, score=score, factors=factors))
        drives.sort(key=lambda d: (-d.score, d.topic))
        return tuple(drives[:limit])

    def _goal_fit(self, topic: str, goals: tuple[Goal, ...]) -> float:
        """How much a topic serves the goals: the best goal-word overlap."""
        if not goals:
            return 0.0
        topic_tokens = {normalize_token(w) for w in topic.split() if len(w) > 2}
        if not topic_tokens:
            return 0.0
        best = 0.0
        for goal in goals:
            goal_tokens = {normalize_token(w) for w in goal.description.split() if len(w) > 2}
            if not goal_tokens:
                continue
            overlap = len(topic_tokens & goal_tokens) / len(topic_tokens | goal_tokens)
            best = max(best, overlap * goal.priority)
        return round(min(best, 1.0), 6)


__all__ = ["WEIGHTS", "Drive", "MotivationEngine"]
