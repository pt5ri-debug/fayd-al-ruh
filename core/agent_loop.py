from __future__ import annotations

from dataclasses import dataclass

from core.action import Action
from core.action_executor import ExecutionResult
from core.action_policy import ActionPolicy
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
    def run_once(self, controller: CognitiveController, environment: Environment, state: AgentState, belief: Belief, polarity: EvidencePolarity, goal: Goal, actions: list[Action], perception: Perception | None = None, cognition: Cognition | None = None, executor=None, policy: ActionPolicy | None = None) -> AgentCycle:
        perceiver = perception or Perception()
        thinker = cognition or Cognition()
        initial = perceiver.perceive(environment)
        state.record_observation(initial)
        cognitive_result = thinker.process(controller, initial, belief, polarity)
        decision = controller.decide(environment.state, goal, actions)
        # A learned policy, when supplied, may override the declarative choice.
        # It can only pick from the offered actions, and the chosen action still
        # runs through the same executor and precondition checks below.
        if policy is not None:
            decision = self._apply_policy(decision, policy, environment, actions)
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

    @staticmethod
    def _apply_policy(
        decision: DecisionTrace,
        policy: ActionPolicy,
        environment: Environment,
        actions: list[Action],
    ) -> DecisionTrace:
        """Re-rank decisions so the learned action becomes the selection.

        The learned policy can only choose among the actions the caller offered,
        so it cannot invent new effects; the surrounding safety guarantees hold.
        """
        by_name = {action.name: candidate for candidate in decision.candidates for action in (candidate.action,)}
        available = tuple(action.name for action in actions)
        chosen_name = policy.select(environment.state.facts, available)
        if chosen_name is None or chosen_name not in by_name:
            return decision
        ordered = sorted(
            decision.candidates,
            key=lambda candidate: (candidate.action.name != chosen_name, -candidate.score, candidate.action.name),
        )
        return DecisionTrace(
            tuple(ordered),
            by_name[chosen_name],
            "Selected by the learned policy.",
        )
