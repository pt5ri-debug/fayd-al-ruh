from core.cognitive_evaluator import (
    CognitiveEvaluation,
)
from core.cognitive_report import (
    CognitiveReport,
)
from core.cognitive_state import (
    CognitiveState,
)
from core.learning_metrics import (
    LearningMetrics,
)


def test_generate_cognitive_report():

    state = CognitiveState()

    state.next_cycle()
    state.next_cycle()

    metrics = LearningMetrics(
        prediction_count=10,
        resolved_prediction_count=8,
        correct_prediction_count=6,
        prediction_accuracy=0.75,
        average_prediction_error=0.2,
        confidence_change=0.05,
    )

    evaluation = CognitiveEvaluation(
        learning_score=0.8,
        prediction_score=0.75,
        stability_score=0.95,
        overall_score=0.833333,
    )

    report = CognitiveReport.generate(
        state=state,
        metrics=metrics,
        evaluation=evaluation,
    )

    assert report.cycle_count == 2

    assert report.prediction_count == 0
    assert report.prediction_error_count == 0

    assert report.learning_score == 0.8
    assert report.prediction_score == 0.75
    assert report.stability_score == 0.95

    assert report.overall_score == 0.833333

    assert report.prediction_accuracy == 0.75

    assert (
        report.average_prediction_error
        == 0.2
    )


def test_report_is_immutable():

    state = CognitiveState()

    metrics = LearningMetrics(
        prediction_count=0,
        resolved_prediction_count=0,
        correct_prediction_count=0,
        prediction_accuracy=0.0,
        average_prediction_error=0.0,
        confidence_change=0.0,
    )

    evaluation = CognitiveEvaluation(
        learning_score=1.0,
        prediction_score=0.0,
        stability_score=1.0,
        overall_score=0.666667,
    )

    report = CognitiveReport.generate(
        state,
        metrics,
        evaluation,
    )

    try:
        report.overall_score = 0.1
        raise AssertionError("the report should be frozen")
    except AttributeError:
        pass
