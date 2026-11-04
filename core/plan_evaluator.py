from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from core.goal import Goal
from core.plan import Plan
from core.state import State


@dataclass(frozen=True)
class PlanEvaluation:
    goal_achieved: bool
    prediction_error: float
    unexpected_facts: frozenset[str]
    missing_facts: frozenset[str]


@dataclass(frozen=True)
class OutcomeEvaluation:
    """The verdict on an executed run (Phase 8's planning-learning bridge).

    ``evaluate_outcome`` reads a run's own verification (duck-typed, so this
    module never imports the execution layer) and turns it into the one honest
    summary the learning loop needs: whether the goal was achieved, whether the
    run succeeded, the outcome's own name, and the reasons it gives.
    """

    goal_achieved: bool
    success: bool
    outcome: str
    reasons: tuple[str, ...] = ()

    def describe(self) -> str:
        verdict = "achieved" if self.goal_achieved else "not achieved"
        return f"{self.outcome}: goal {verdict}"


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

    # ---------------------------------------------------- Phase 8: the outcome

    def evaluate_outcome(
        self,
        goal: object,
        execution_result: object = None,
        *,
        verification: object = None,
    ) -> OutcomeEvaluation:
        """The verdict on an executed run, read from its own verification.

        ``execution_result`` is a :class:`core.execution.coordinator.ExecutionResult`
        (duck-typed: it needs ``verification``); a caller that already holds the
        verification may pass it directly as ``verification=``. Nothing is
        re-derived -- the verdict is the verifier's own, so the planning loop and
        the execution layer can never disagree about whether a goal was achieved.
        """
        if verification is None:
            verification = getattr(execution_result, "verification", None)
        if verification is None:
            return OutcomeEvaluation(False, False, "unverified", ("the run produced no verification.",))
        outcome = getattr(verification, "outcome", None)
        outcome_name = getattr(outcome, "value", str(outcome or "unverified"))
        return OutcomeEvaluation(
            goal_achieved=bool(getattr(verification, "goal_achieved", False)),
            success=bool(getattr(verification, "success", False)),
            outcome=outcome_name,
            reasons=tuple(str(r) for r in (getattr(verification, "reasons", ()) or ())),
        )

    def record_experience(
        self,
        evaluation: OutcomeEvaluation,
        goal: object,
        learning: Any,
        *,
        approach: str = "",
        scope: str = "",
    ) -> str:
        """Feed the evaluated outcome to the Phase-3 experience store.

        Returns the recorded event/memory id, or ``""`` when there is no learning
        service or the store refuses -- a learning failure never breaks a run, and
        this method never raises for that reason.
        """
        if learning is None:
            return ""
        try:
            from core.experience_learning import ExperienceRecord
            from core.learning_events import SOURCE_SYSTEM

            experience = ExperienceRecord(
                goal=str(getattr(goal, "description", goal)),
                outcome=evaluation.outcome,
                success=evaluation.success,
                approach=approach,
                errors=evaluation.reasons if not evaluation.success else (),
                evaluation=evaluation.describe(),
                confidence=1.0 if evaluation.success else 0.5,
                source=SOURCE_SYSTEM,
                scope=scope,
            )
            result = learning.learn_experience(experience)
            events = getattr(result, "events", ()) or ()
            if events:
                return str(getattr(events[0], "event_id", "") or "")
            return str(getattr(result, "memory_id", "") or "")
        except Exception:
            return ""
