from __future__ import annotations

from dataclasses import dataclass

from core.goal import Goal
from core.plan import Plan
from core.state import State


@dataclass(frozen=True)
class PlanEvaluation:
    goal_achieved: bool
    prediction_error: float
    unexpected_facts: frozenset[str]
    missing_facts: frozenset[str]


class PlanEvaluator:
    """Compares predicted and observed states to close the planning-learning loop."""
    def evaluate(self, plan: Plan, goal: Goal, observed_state: State) -> PlanEvaluation:
        expected = plan.predicted_final_state.facts
        observed = observed_state.facts
        unexpected = observed - expected
        missing = expected - observed
        denominator = max(1, len(expected | observed))
        error = round((len(unexpected) + len(missing)) / denominator, 6)
        return PlanEvaluation(observed_state.satisfies(goal.desired_facts), error, frozenset(unexpected), frozenset(missing))
