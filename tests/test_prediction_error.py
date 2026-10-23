import pytest
from core.prediction import Prediction
from core.prediction_error import (
    PredictionErrorCalculator,
)


def test_perfect_prediction_has_zero_error():

    prediction = Prediction(
        statement="The ball will move.",
        confidence=0.9,
        hypothesis_id="HYP-001",
    )

    calculator = PredictionErrorCalculator()

    result = calculator.calculate(
        prediction,
        actual_result=0.9,
    )

    assert result.error == 0.0
    assert result.prediction_id == prediction.id


def test_prediction_error_is_calculated():

    prediction = Prediction(
        statement="The ball will move.",
        confidence=0.9,
        hypothesis_id="HYP-001",
    )

    calculator = PredictionErrorCalculator()

    result = calculator.calculate(
        prediction,
        actual_result=0.4,
    )

    assert result.error == 0.5


def test_wrong_prediction_has_high_error():

    prediction = Prediction(
        statement="The ball will move.",
        confidence=1.0,
        hypothesis_id="HYP-001",
    )

    calculator = PredictionErrorCalculator()

    result = calculator.calculate(
        prediction,
        actual_result=0.0,
    )

    assert result.error == 1.0


def test_invalid_actual_result_is_rejected():

    prediction = Prediction(
        statement="The ball will move.",
        confidence=0.8,
        hypothesis_id="HYP-001",
    )

    calculator = PredictionErrorCalculator()

    with pytest.raises(ValueError):

        calculator.calculate(
            prediction,
            actual_result=1.5,
        )
