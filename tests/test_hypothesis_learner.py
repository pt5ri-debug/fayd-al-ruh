from core.hypothesis import Hypothesis
from core.hypothesis_learner import HypothesisLearner
from core.learning_signal import LearningSignal


def test_hypothesis_remains_tentative_after_learning():
    result = HypothesisLearner().learn(Hypothesis("rain", .5, "pattern"), LearningSignal("P", .2, .8, 1))
    assert result.estimated_confidence > result.previous_confidence
