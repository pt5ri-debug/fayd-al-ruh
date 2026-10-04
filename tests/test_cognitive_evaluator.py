from core.cognitive_evaluator import (
    CognitiveEvaluator,
)
from core.learning_metrics import LearningMetrics


def test_perfect_metrics_produce_high_score():

    metrics = LearningMetrics(
        prediction_count=10,
        resolved_prediction_count=10,
        correct_prediction_count=10,
        prediction_accuracy=1.0,
        average_prediction_error=0.0,
        confidence_change=0.0,
    )

    evaluator = CognitiveEvaluator()

    result = evaluator.evaluate(metrics)

    assert result.learning_score == 1.0
    assert result.prediction_score == 1.0
    assert result.stability_score == 1.0
    assert result.overall_score == 1.0


def test_poor_prediction_produces_lower_score():

    metrics = LearningMetrics(
        prediction_count=10,
        resolved_prediction_count=10,
        correct_prediction_count=2,
        prediction_accuracy=0.2,
        average_prediction_error=0.8,
        confidence_change=0.0,
    )

    evaluator = CognitiveEvaluator()

    result = evaluator.evaluate(metrics)

    assert result.prediction_score == 0.2
    assert result.learning_score == 0.2
    assert result.overall_score < 0.5


def test_large_confidence_change_reduces_stability():

    metrics = LearningMetrics(
        prediction_count=5,
        resolved_prediction_count=5,
        correct_prediction_count=5,
        prediction_accuracy=1.0,
        average_prediction_error=0.0,
        confidence_change=0.5,
    )

    evaluator = CognitiveEvaluator()

    result = evaluator.evaluate(metrics)

    assert result.stability_score == 0.5