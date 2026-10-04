from core.action import Action
from core.agent import Agent
from core.belief import Belief
from core.cognitive_controller import CognitiveController
from core.environment import Environment
from core.evidence_processor import EvidencePolarity
from core.goal import Goal
import pytest


pytestmark = pytest.mark.integration


def test_agent_controller_environment_contract():
    agent = Agent(CognitiveController(), Environment())
    cycle = agent.step(Belief("agent observes", .5), EvidencePolarity.SUPPORTS, Goal("see", frozenset({"seen"})), [Action("see", add_effects=frozenset({"seen"}))])
    assert cycle.execution is not None
