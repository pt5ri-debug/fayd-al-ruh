from __future__ import annotations

from dataclasses import dataclass

from core.architecture_learning import ArchitectureRecommendation


@dataclass(frozen=True)
class ImprovementPlan:
    recommendations: tuple[ArchitectureRecommendation, ...]
    requires_human_review: bool = True


class SelfImprovement:
    """Packages recommendations for review rather than self-modifying code or policy."""
    def propose(self, recommendations: list[ArchitectureRecommendation]) -> ImprovementPlan:
        return ImprovementPlan(tuple(recommendations))
