from __future__ import annotations

from dataclasses import dataclass

from core.action import Action
from core.action_executor import ExecutionResult
from core.agent_state import AgentState
from core.belief import Belief
from core.cognition import Cognition
from core.cognitive_controller import CognitiveController, CognitiveCycleResult
from core.decision_trace import DecisionTrace
from core.environment import Environment
from core.evidence_processor import EvidencePolarity
from core.goal import Goal
from core.observation import Observation
from core.perception import Perception


@dataclass(frozen=True)
class AgentCycle:
    perception: Observation
    cognition: CognitiveCycleResult
    decision: DecisionTrace
    execution: ExecutionResult | None
    resulting_observation: Observation
    learning: CognitiveCycleResult | None


class AgentLoop:
    """One safe agent cycle; no unbounded autonomous looping is performed here."""
    def run_once(self, controller: CognitiveController, environment: Environment, state: AgentState, belief: Belief, polarity: EvidencePolarity, goal: Goal, actions: list[Action], perception: Perception | None = None, cognition: Cognition | None = None, executor=None) -> AgentCycle:
        perceiver = perception or Perception()
        thinker = cognition or Cognition()
        initial = perceiver.perceive(environment)
        state.record_observation(initial)
        cognitive_result = thinker.process(controller, initial, belief, polarity)
        decision = controller.decide(environment.state, goal, actions)
        execution = None
        if decision.selected is not None:
            from core.action_executor import ActionExecutor
            execution = (executor or ActionExecutor()).execute(environment, decision.selected.action)
            state.record_action(decision.selected.action.id)
        resulting = perceiver.perceive(environment)
        learning = None
        if cognitive_result.loop_result.predictions:
            learning = controller.evaluate_prediction(cognitive_result.loop_result.predictions[0].id, belief, resulting.data["result"])
        return AgentCycle(initial, cognitive_result, decision, execution, resulting, learning)
