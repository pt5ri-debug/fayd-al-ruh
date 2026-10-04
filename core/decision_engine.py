from __future__ import annotations

from core.action import Action
from core.decision import Decision
from core.decision_trace import DecisionTrace
from core.goal import Goal
from core.risk_model import RiskModel
from core.state import State
from core.utility_model import UtilityModel


class DecisionEngine:
    """Ranks actions by transparent utility minus risk; it never executes them."""
    def __init__(self, utility: UtilityModel | None = None, risk: RiskModel | None = None, risk_aversion: float = .5) -> None:
        if not 0.0 <= risk_aversion <= 1.0:
            raise ValueError("Risk aversion must be between 0 and 1.")
        self.utility_model = utility or UtilityModel()
        self.risk_model = risk or RiskModel()
        self.risk_aversion = risk_aversion

    def decide(self, state: State, goal: Goal, actions: list[Action]) -> DecisionTrace:
        candidates = tuple(sorted((self._evaluate(state, goal, action) for action in actions), key=lambda item: (-item.score, item.action.name)))
        feasible = [candidate for candidate in candidates if not (candidate.action.preconditions - state.facts)]
        if not feasible:
            return DecisionTrace(candidates, None, "No action has satisfied preconditions.")
        selected = feasible[0]
        return DecisionTrace(candidates, selected, "Selected highest utility adjusted for risk.")

    def _evaluate(self, state: State, goal: Goal, action: Action) -> Decision:
        utility = self.utility_model.evaluate(action, goal)
        risk = self.risk_model.evaluate(state, action, goal)
        score = round(utility.value - self.risk_aversion * risk.value, 6)
        return Decision(action, utility, risk, score)
