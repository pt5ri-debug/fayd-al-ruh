from core.adaptive_learning import (
    AdaptiveLearningEngine,
)
from core.belief import Belief
from core.prediction_error import (
    PredictionError,
)


def test_prediction_error_reduces_confidence():

    belief = Belief(
        statement="The ball moves.",
        confidence=0.9,
    )

    error = PredictionError(
        prediction_id="PRD-001",
        expected_confidence=0.9,
        actual_result=0.4,
        error=0.5,
    )

    engine = AdaptiveLearningEngine(
        learning_rate=0.1,
    )

    result = engine.learn(
        belief,
        error,
    )

    assert result.old_confidence == 0.9
    assert result.new_confidence == 0.85
    assert belief.confidence == 0.85


def test_successful_result_increases_confidence():

    belief = Belief(
        statement="The ball moves.",
        confidence=0.5,
    )

    error = PredictionError(
        prediction_id="PRD-001",
        expected_confidence=0.5,
        actual_result=0.9,
        error=0.4,
    )

    engine = AdaptiveLearningEngine(
        learning_rate=0.1,
    )

    result = engine.learn(
        belief,
        error,
    )

    assert result.new_confidence == 0.54


def test_adjustment_is_limited():

    belief = Belief(
        statement="The ball moves.",
        confidence=0.9,
    )

    error = PredictionError(
        prediction_id="PRD-001",
        expected_confidence=0.9,
        actual_result=0.0,
        error=1.0,
    )

    engine = AdaptiveLearningEngine(
        learning_rate=1.0,
        max_adjustment=0.2,
    )

    result = engine.learn(
        belief,
        error,
    )

    assert result.adjustment == 0.2
    assert result.new_confidence == 0.7


def test_confidence_never_exceeds_bounds():

    belief = Belief(
        statement="The ball moves.",
        confidence=0.95,
    )

    error = PredictionError(
        prediction_id="PRD-001",
        expected_confidence=0.95,
        actual_result=1.0,
        error=0.05,
    )

    engine = AdaptiveLearningEngine(
        learning_rate=1.0,
    )

    engine.learn(
        belief,
        error,
    )

    assert 0.0 <= belief.confidence <= 1.0
