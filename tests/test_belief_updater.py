from core.belief import Belief
from core.prediction_error import PredictionError
from core.belief_updater import BeliefUpdater


def test_prediction_error_reduces_belief_confidence():

    belief = Belief(
        statement="The ball is moving repeatedly.",
        confidence=0.85,
    )

    prediction_error = PredictionError(
        prediction_id="PRD-test",
        expected_confidence=0.85,
        actual_result=0.0,
        error=0.85,
    )

    updater = BeliefUpdater()

    updated_belief = updater.update(
        belief,
        prediction_error,
    )

    assert updated_belief is belief
    assert updated_belief.confidence == 0.0


def test_zero_prediction_error_preserves_confidence():

    belief = Belief(
        statement="The ball is moving repeatedly.",
        confidence=0.85,
    )

    prediction_error = PredictionError(
        prediction_id="PRD-test",
        expected_confidence=0.85,
        actual_result=0.85,
        error=0.0,
    )

    updater = BeliefUpdater()

    updater.update(
        belief,
        prediction_error,
    )

    assert belief.confidence == 0.85


def test_partial_prediction_error_reduces_confidence():

    belief = Belief(
        statement="The ball is moving repeatedly.",
        confidence=0.9,
    )

    prediction_error = PredictionError(
        prediction_id="PRD-test",
        expected_confidence=0.9,
        actual_result=0.6,
        error=0.3,
    )

    updater = BeliefUpdater()

    updater.update(
        belief,
        prediction_error,
    )

    assert belief.confidence == 0.6


def test_confidence_never_goes_below_zero():

    belief = Belief(
        statement="The ball is moving repeatedly.",
        confidence=0.2,
    )

    prediction_error = PredictionError(
        prediction_id="PRD-test",
        expected_confidence=0.2,
        actual_result=0.0,
        error=1.0,
    )

    updater = BeliefUpdater()

    updater.update(
        belief,
        prediction_error,
    )

    assert 0.0 <= belief.confidence <= 1.0


def test_confidence_never_exceeds_one():

    belief = Belief(
        statement="The ball is moving repeatedly.",
        confidence=0.9,
    )

    prediction_error = PredictionError(
        prediction_id="PRD-test",
        expected_confidence=0.9,
        actual_result=1.0,
        error=0.0,
    )

    updater = BeliefUpdater()

    updater.update(
        belief,
        prediction_error,
    )

    assert 0.0 <= belief.confidence <= 1.0