from core.belief import Belief
from core.learner import Learner
from core.prediction_error import PredictionError


def test_learner_records_prediction_error_update():
    cycle = Learner().learn_from_prediction(Belief("rain", .5), PredictionError("P", .5, 1., .5), "observe")
    assert cycle.belief_after > cycle.belief_before and cycle.reinforcement is not None
