from core.belief import Belief
from core.belief_learner import BeliefLearner
from core.learning_signal import LearningSignal


def test_belief_learner_updates_confidence():
    assert BeliefLearner().learn(Belief("rain", .5), LearningSignal("P", .2, .8, -1)).updated < .5
