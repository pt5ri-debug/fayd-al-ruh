from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ArchitectureRecommendation:
    area: str
    recommendation: str
    reason: str
    priority: float


class ArchitectureLearning:
    """Produces reviewable architecture recommendations; it never edits the system."""
    def recommend(self, average_prediction_error: float, memory_retrieval: float) -> list[ArchitectureRecommendation]:
        recommendations: list[ArchitectureRecommendation] = []
        if average_prediction_error > .3:
            recommendations.append(ArchitectureRecommendation("prediction", "Review prediction features and calibration.", "Prediction error remains elevated.", round(average_prediction_error, 6)))
        if memory_retrieval < .6:
            recommendations.append(ArchitectureRecommendation("memory", "Improve memory indexing or retrieval cues.", "Relevant memory activation is low.", round(1.0 - memory_retrieval, 6)))
        return sorted(recommendations, key=lambda item: -item.priority)
