from core.belief import Belief
from core.learner import Learner
from core.prediction_error import PredictionError


def test_learning_contract_changes_belief_from_error():
    belief = Belief("rain", .5)
    Learner().learn_from_prediction(belief, PredictionError("P", .8, 1., .2))
    assert belief.confidence > .5
