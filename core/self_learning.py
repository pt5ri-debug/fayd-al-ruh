from __future__ import annotations

from dataclasses import dataclass

from core.architecture_learning import ArchitectureLearning
from core.curriculum import Curriculum, CurriculumItem
from core.self_improvement import ImprovementPlan, SelfImprovement
from core.skill_learning import SkillLearning
from core.strategy_learning import StrategyLearning


@dataclass(frozen=True)
class SelfLearningReport:
    next_topic: CurriculumItem | None
    improvement_plan: ImprovementPlan


class SelfLearning:
    """Coordinates curriculum, skill, strategy, and reviewable improvement learning."""
    def __init__(self) -> None:
        self.curriculum = Curriculum()
        self.skills = SkillLearning()
        self.strategies = StrategyLearning()
        self.architecture = ArchitectureLearning()
        self.improvement = SelfImprovement()

    def assess(self, items: list[CurriculumItem], mastered: set[str], average_prediction_error: float, memory_retrieval: float) -> SelfLearningReport:
        next_topic = self.curriculum.next_item(items, mastered)
        proposals = self.architecture.recommend(average_prediction_error, memory_retrieval)
        return SelfLearningReport(next_topic, self.improvement.propose(proposals))
