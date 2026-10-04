from __future__ import annotations

from core.action import Action
from core.goal import Goal
from core.utility import Utility


class UtilityModel:
    """Estimates benefit from progress toward the goal and normalised action cost."""
    def evaluate(self, action: Action, goal: Goal) -> Utility:
        benefit = len(action.add_effects & goal.desired_facts) / len(goal.desired_facts)
        cost = action.cost / (action.cost + 1.0)
        value = max(0.0, min(1.0, benefit - .25 * cost))
        return Utility(round(value, 6), round(benefit, 6), round(cost, 6))
