from core.learning_signal import LearningSignal
from core.reinforcement import ReinforcementLearner


def test_reinforcement_updates_bounded_action_value():
    assert ReinforcementLearner().reinforce("observe", LearningSignal("P", .1, .9, 1)).new_value > .5
