import pytest

from core.prediction import Prediction


def test_create_prediction():

    prediction = Prediction(
        statement="The ball will move again.",
        confidence=0.8,
        hypothesis_id="HYP-001",
    )

    assert prediction.statement == \
        "The ball will move again."

    assert prediction.confidence == 0.8

    assert prediction.hypothesis_id == \
        "HYP-001"

    assert prediction.fulfilled is None


def test_prediction_can_be_resolved():

    prediction = Prediction(
        statement="The ball will move again.",
        confidence=0.8,
        hypothesis_id="HYP-001",
    )

    prediction.resolve(True)

    assert prediction.fulfilled is True


def test_empty_prediction_is_rejected():

    with pytest.raises(ValueError):

        Prediction(
            statement="",
            confidence=0.8,
            hypothesis_id="HYP-001",
        )