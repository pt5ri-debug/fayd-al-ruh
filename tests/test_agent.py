from core.action import Action
from core.agent import Agent
from core.belief import Belief
from core.cognitive_controller import CognitiveController
from core.environment import Environment
from core.evidence_processor import EvidencePolarity
from core.goal import Goal


def test_agent_runs_perception_to_action_cycle():
    agent = Agent(CognitiveController(), Environment())
    cycle = agent.step(Belief("agent can see", .5), EvidencePolarity.SUPPORTS, Goal(frozenset({"seen"}), "see"), [Action("see", add_effects=frozenset({"seen"}))])
    assert cycle.execution is not None and "seen" in cycle.resulting_observation.data["facts"]
