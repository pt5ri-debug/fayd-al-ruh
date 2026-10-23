from __future__ import annotations

from core.belief import Belief
from core.prediction_error import PredictionError


class BeliefUpdater:
    """
    Updates a belief using the actual result represented
    by a PredictionError.

    The current v2.0 implementation uses the observed
    actual result as the new belief confidence.
    """

    def update(
        self,
        belief: Belief,
        prediction_error: PredictionError,
    ) -> Belief:
        """
        Update belief confidence from prediction error.

        The actual observed result becomes the new
        confidence value.
        """

        belief.set_confidence(
            prediction_error.actual_result
        )

        return belief
