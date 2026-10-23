from core.belief import Belief
from core.cognitive_state import CognitiveState
from core.state_snapshot import StateSnapshot


def test_empty_state_snapshot():

    state = CognitiveState()

    snapshot = StateSnapshot.capture(
        state
    )

    assert snapshot.cycle_count == 0
    assert snapshot.belief_count == 0
    assert snapshot.pattern_count == 0
    assert snapshot.hypothesis_count == 0
    assert snapshot.prediction_count == 0
    assert snapshot.error_count == 0

    assert (
        snapshot.average_belief_confidence
        == 0.0
    )


def test_snapshot_captures_beliefs():

    state = CognitiveState()

    first = Belief(
        statement="The ball moves.",
        confidence=0.8,
    )

    second = Belief(
        statement="The light is on.",
        confidence=0.6,
    )

    state.add_belief(first)
    state.add_belief(second)

    snapshot = StateSnapshot.capture(
        state
    )

    assert snapshot.belief_count == 2

    assert (
        snapshot.average_belief_confidence
        == 0.7
    )


def test_snapshot_captures_cycle_count():

    state = CognitiveState()

    state.next_cycle()
    state.next_cycle()
    state.next_cycle()

    snapshot = StateSnapshot.capture(
        state
    )

    assert snapshot.cycle_count == 3


def test_snapshot_is_immutable():

    state = CognitiveState()

    snapshot = StateSnapshot.capture(
        state
    )

    try:
        snapshot.cycle_count = 10
        raise AssertionError("the snapshot should be frozen")
    except AttributeError:
        pass
