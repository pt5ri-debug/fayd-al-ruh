from __future__ import annotations

from core.hypothesis import Hypothesis
from core.prediction import Prediction


class PredictionEngine:
    """
    Converts hypotheses into testable predictions.
    """

    def create(
        self,
        hypothesis: Hypothesis,
    ) -> Prediction:

        statement = (
            f"If the pattern continues, "
            f"'{hypothesis.source_pattern}' "
            f"may occur again."
        )

        return Prediction(
            statement=statement,
            confidence=hypothesis.confidence,
            hypothesis_id=hypothesis.id,
        )

    def evaluate(
        self,
        prediction: Prediction,
        fulfilled: bool,
    ) -> None:

        prediction.resolve(fulfilled)