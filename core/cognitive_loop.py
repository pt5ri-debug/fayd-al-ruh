from __future__ import annotations

from dataclasses import dataclass

from core.adaptive_learning import (
    AdaptiveLearningEngine,
    AdaptiveLearningResult,
)
from core.belief import Belief
from core.cognitive_cycle import CognitiveCycle
from core.cognitive_state import CognitiveState
from core.evidence_processor import EvidencePolarity
from core.hypothesis import Hypothesis
from core.hypothesis_engine import HypothesisEngine
from core.knowledge_graph import KnowledgeGraph
from core.memory import CognitiveMemory, MemoryRecall
from core.observation import Observation
from core.pattern_detector import Pattern, PatternDetector
from core.prediction import Prediction
from core.prediction_engine import PredictionEngine
from core.prediction_error import (
    PredictionError,
    PredictionErrorCalculator,
)


@dataclass
class CognitiveLoopResult:
    observation: Observation | None
    patterns: list[Pattern]
    hypotheses: list[Hypothesis]
    predictions: list[Prediction]
    prediction_errors: list[PredictionError]
    learning_results: list[AdaptiveLearningResult]
    recalled_memories: list[MemoryRecall]
    evidence_id: str | None = None


class CognitiveLoop:

    def __init__(
        self,
        graph: KnowledgeGraph,
        memory: CognitiveMemory,
        state: CognitiveState | None = None,
    ) -> None:

        self.graph = graph
        self.memory = memory

        self.state = (
            state
            if state is not None
            else CognitiveState()
        )

        self.cognitive_cycle = CognitiveCycle(
            graph=graph,
            memory=memory,
        )

        self.pattern_detector = PatternDetector()
        self.hypothesis_engine = HypothesisEngine()
        self.prediction_engine = PredictionEngine()
        self.error_calculator = PredictionErrorCalculator()
        self.learning_engine = AdaptiveLearningEngine()

    def observe(
        self,
        observation: Observation,
        belief: Belief,
        polarity: EvidencePolarity,
    ) -> CognitiveLoopResult:

        # Recall happens before storing the new observation so the current
        # input cannot trivially retrieve itself as "context".
        recalled_memories = self.memory.recall(
            observation.content,
            item_types=("observation",),
            minimum_relevance=0.01,
        )

        self.state.next_cycle()

        self.state.add_belief(
            belief
        )

        cycle_result = self.cognitive_cycle.process(
            observation=observation,
            belief=belief,
            polarity=polarity,
        )

        patterns = self.pattern_detector.detect(
            self.memory,
            minimum_occurrences=2,
        )

        for pattern in patterns:
            self.state.add_pattern(pattern)

        hypotheses: list[Hypothesis] = []

        for pattern in patterns:

            hypothesis = (
                self.hypothesis_engine.create(
                    pattern
                )
            )

            hypotheses.append(hypothesis)

            self.state.add_hypothesis(
                hypothesis
            )

        predictions: list[Prediction] = []

        for hypothesis in hypotheses:

            prediction = (
                self.prediction_engine.create(
                    hypothesis
                )
            )

            predictions.append(prediction)

            self.state.add_prediction(
                prediction
            )

        return CognitiveLoopResult(
            observation=observation,
            patterns=patterns,
            hypotheses=hypotheses,
            predictions=predictions,
            prediction_errors=[],
            learning_results=[],
            recalled_memories=recalled_memories,
            evidence_id=cycle_result.evidence_id,
        )

    def evaluate_prediction(
        self,
        prediction: Prediction,
        belief: Belief,
        actual_result: float,
    ) -> tuple[
        PredictionError,
        AdaptiveLearningResult,
    ]:

        prediction_error = (
            self.error_calculator.calculate(
                prediction,
                actual_result,
            )
        )

        self.prediction_engine.evaluate(
            prediction,
            actual_result >= 0.5,
        )

        self.state.add_prediction_error(
            prediction_error
        )

        learning_result = (
            self.learning_engine.learn(
                belief,
                prediction_error,
            )
        )

        self.state.add_belief(
            belief
        )

        return (
            prediction_error,
            learning_result,
        )
