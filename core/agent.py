from __future__ import annotations

from core.agent_loop import AgentCycle, AgentLoop
from core.agent_state import AgentState
from core.belief import Belief
from core.cognition import Cognition
from core.cognitive_controller import CognitiveController
from core.environment import Environment
from core.evidence_processor import EvidencePolarity
from core.goal import Goal
from core.action import Action
from core.perception import Perception
from core.action_executor import ActionExecutor


class Agent:
    """Composes the safe perception-cognition-decision-action environment loop."""
    def __init__(self, controller: CognitiveController, environment: Environment) -> None:
        self.controller = controller
        self.environment = environment
        self.state = AgentState()
        self.loop = AgentLoop()
        self.perception = Perception()
        self.cognition = Cognition()
        self.executor = ActionExecutor()

    def step(self, belief: Belief, polarity: EvidencePolarity, goal: Goal, actions: list[Action]) -> AgentCycle:
        return self.loop.run_once(self.controller, self.environment, self.state, belief, polarity, goal, actions, self.perception, self.cognition, self.executor)
