from core.belief import Belief
from core.cognitive_state import CognitiveState
from core.metacognition import Metacognition


def test_metacognition_reports_knowledge_gaps():
    state = CognitiveState()
    state.add_belief(Belief("uncertain", .4))
    report = Metacognition().assess(state)
    assert report.knowledge.unknown_count == 1 and len(report.low_confidence_beliefs) == 1
