from core.belief import Belief
from core.cognitive_controller import CognitiveController
from core.evidence_processor import EvidencePolarity
from core.memory import CognitiveMemory
from core.observation import Observation


def test_recall_prefers_relevant_memory_and_tracks_access():
    memory = CognitiveMemory()
    memory.store("A", "observation", "Rain falls on the garden", 0.3)
    memory.store("B", "observation", "A robot opens a door", 1.0)

    recalls = memory.recall("rain in garden", limit=2, minimum_relevance=0.01)

    assert [recall.item.item_id for recall in recalls] == ["A"]
    assert recalls[0].relevance > 0
    assert memory.get("A").access_count == 2


def test_controller_exposes_prior_relevant_context_only():
    controller = CognitiveController()
    belief = Belief(statement="The garden is wet.")

    controller.process(
        Observation(content="Rain falls on the garden", source="sensor"),
        belief,
        EvidencePolarity.SUPPORTS,
    )
    result = controller.process(
        Observation(content="The garden has rain", source="sensor"),
        belief,
        EvidencePolarity.SUPPORTS,
    )

    assert len(result.working_memory) == 1
    assert result.working_memory[0].item.item_type == "observation"


def test_repeated_belief_cycles_keep_distinct_state_memories():
    controller = CognitiveController()
    belief = Belief(statement="The garden is wet.")

    for content in ("Rain falls", "Rain continues"):
        controller.process(
            Observation(content=content, source="sensor"),
            belief,
            EvidencePolarity.SUPPORTS,
        )

    assert len(controller.memory.find_by_type("belief_state")) == 2
