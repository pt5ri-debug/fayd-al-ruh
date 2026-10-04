from __future__ import annotations

from core.action import Action
from core.goal import Goal
from core.risk import Risk
from core.state import State


class RiskModel:
    """Models planning risk from missing preconditions and harmful effects."""
    def evaluate(self, state: State, action: Action, goal: Goal) -> Risk:
        reasons: list[str] = []
        missing = action.preconditions - state.facts
        missing_component = len(missing) / max(1, len(action.preconditions))
        if missing:
            reasons.append("preconditions are not satisfied")
        harmful = action.remove_effects & goal.desired_facts
        harmful_component = len(harmful) / max(1, len(goal.desired_facts))
        if harmful:
            reasons.append("action removes a desired goal fact")
        cost_component = action.cost / (action.cost + 4.0)
        if action.cost > 1.0:
            reasons.append("action has elevated cost")
        value = max(missing_component, harmful_component, cost_component)
        return Risk(round(value, 6), tuple(reasons))
