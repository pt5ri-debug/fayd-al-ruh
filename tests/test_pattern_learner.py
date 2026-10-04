from core.learning_signal import LearningSignal
from core.pattern_detector import Pattern
from core.pattern_learner import PatternLearner


def test_pattern_learner_estimates_from_signal():
    assert PatternLearner().learn(Pattern("rain", 2, .5, ["1", "2"]), LearningSignal("P", .2, .8, 1)).estimated_confidence > .5
