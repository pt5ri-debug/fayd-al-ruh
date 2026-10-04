from core.hypothesis import Hypothesis
from core.prediction_engine import PredictionEngine


def test_hypothesis_creates_prediction():

    hypothesis = Hypothesis(
        statement="The ball moves repeatedly.",
        confidence=0.75,
        source_pattern="the ball moved.",
    )

    engine = PredictionEngine()

    prediction = engine.create(
        hypothesis
    )

    assert prediction.hypothesis_id == \
        hypothesis.id

    assert prediction.confidence == 0.75

    assert (
        "the ball moved."
        in prediction.statement
    )


def test_prediction_can_be_evaluated():

    hypothesis = Hypothesis(
        statement="The ball moves repeatedly.",
        confidence=0.75,
        source_pattern="the ball moved.",
    )

    engine = PredictionEngine()

    prediction = engine.create(
        hypothesis
    )

    engine.evaluate(
        prediction,
        fulfilled=True,
    )

    assert prediction.fulfilled is True