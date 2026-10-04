from core.belief import Belief
from core.cognitive_cycle import CognitiveCycle
from core.evidence_processor import EvidencePolarity
from core.knowledge_graph import KnowledgeGraph
from core.memory import CognitiveMemory
from core.observation import Observation


def test_complete_cognitive_cycle():

    graph = KnowledgeGraph()
    memory = CognitiveMemory()

    cycle = CognitiveCycle(
        graph=graph,
        memory=memory,
    )

    belief = Belief(
        statement="The ball moves.",
        confidence=0.5,
    )

    observation = Observation(
        content="The ball moved.",
        source="motion_sensor",
        confidence=0.9,
    )

    result = cycle.process(
        observation=observation,
        belief=belief,
        polarity=EvidencePolarity.SUPPORTS,
    )

    assert result.observation_id == observation.id

    assert result.belief_id == belief.id

    assert graph.evidence_count == 1

    assert graph.belief_count == 1

    assert memory.size >= 3

    assert belief.confidence > 0.5


def test_contradicting_observation_updates_belief():

    graph = KnowledgeGraph()
    memory = CognitiveMemory()

    cycle = CognitiveCycle(
        graph=graph,
        memory=memory,
    )

    belief = Belief(
        statement="The ball moves.",
        confidence=0.9,
    )

    observation = Observation(
        content="The ball did not move.",
        source="motion_sensor",
        confidence=0.9,
    )

    cycle.process(
        observation=observation,
        belief=belief,
        polarity=EvidencePolarity.CONTRADICTS,
    )

    assert belief.confidence < 0.9


def test_observation_is_remembered():

    graph = KnowledgeGraph()
    memory = CognitiveMemory()

    cycle = CognitiveCycle(
        graph=graph,
        memory=memory,
    )

    belief = Belief(
        statement="The ball exists.",
    )

    observation = Observation(
        content="The ball is visible.",
        source="vision_sensor",
        confidence=0.8,
    )

    cycle.process(
        observation=observation,
        belief=belief,
        polarity=EvidencePolarity.SUPPORTS,
    )

    remembered = memory.get(
        observation.id
    )

    assert remembered is not None

    assert remembered.item_type == "observation"