from core.belief import Belief
from core.learning_signal import LearningSignal
from core.online_learner import OnlineLearner


def test_online_update_uses_error_direction():
    belief = Belief("rain", .5)
    assert OnlineLearner().update_belief(belief, LearningSignal("P", .4, .6, 1)).updated == .54
