from core.belief import Belief
from core.cognitive_state import CognitiveState
from core.knowledge_state import KnowledgeState


def test_knowledge_state_counts_low_confidence_beliefs_as_unknown():
    state = CognitiveState()
    state.add_belief(Belief("uncertain", .3))
    assert KnowledgeState.capture(state).unknown_count == 1
