from core.belief import Belief
from core.cognitive_state import CognitiveState
from core.metacognition import Metacognition


def test_metacognition_contract_reports_unknown():
    state = CognitiveState()
    state.add_belief(Belief("unknown", .2))
    assert Metacognition().assess(state).knowledge.unknown_count == 1
