from core.belief import Belief
from core.cognitive_state import CognitiveState
from core.state_history import StateHistory


def test_empty_history():

    history = StateHistory()

    assert history.size == 0
    assert history.latest is None
    assert history.confidence_change() == 0.0


def test_record_state():

    state = CognitiveState()
    history = StateHistory()

    snapshot = history.record(state)

    assert history.size == 1
    assert history.latest is snapshot


def test_history_keeps_multiple_snapshots():

    state = CognitiveState()
    history = StateHistory()

    history.record(state)

    state.next_cycle()

    history.record(state)

    state.next_cycle()

    history.record(state)

    assert history.size == 3
    assert history.latest.cycle_count == 2


def test_confidence_change():

    state = CognitiveState()
    history = StateHistory()

    belief = Belief(
        statement="The ball moves.",
        confidence=0.5,
    )

    state.add_belief(belief)

    history.record(state)

    belief.set_confidence(0.8)

    history.record(state)

    assert history.confidence_change() == 0.3


def test_clear_history():

    state = CognitiveState()
    history = StateHistory()

    history.record(state)

    assert history.size == 1

    history.clear()

    assert history.size == 0
    assert history.latest is None
