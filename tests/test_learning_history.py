from core.learning_history import LearningEvent, LearningHistory
from core.learning_signal import LearningSignal


def test_learning_history_tracks_error():
    history = LearningHistory(); history.record(LearningEvent(LearningSignal("P", .2, .8, 1), "B", .5, .52))
    assert history.average_error == .2
