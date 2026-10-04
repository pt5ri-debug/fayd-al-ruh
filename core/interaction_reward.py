"""Reward learned from real interaction outcomes, not from the sandbox.

The sandbox tier hands the agent a hand-written success predicate, so its reward
is a proxy: it says "the goal fact is present", not "the choice actually worked".
This module grounds reward in observed outcomes instead.

Two honest sources of real interaction feed the reward:

- the project's own evaluation corpus (``evaluation.scenarios``), where each
  forecast carries the outcome that was actually observed for it, and
- human review of a chosen action, via the existing approval gate.

From those outcomes it learns a small, explainable calibration model
(confidence -> observed success) and, separately, contextual action values. The
predicted success probability of a chosen action *is* the reward. Nothing here
runs external tools or performs side effects; it only records and learns from
outcomes that already happened.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from evaluation.scenarios import PredictionSpec, SCENARIOS


def _clamp(value: float) -> float:
    return max(0.0, min(1.0, value))


@dataclass(frozen=True)
class Outcome:
    """A recorded result, either a forecast that was judged or an action observed.

    ``predicted_confidence`` is the forecast probability when this outcome comes
    from a *prediction*; it is ``None`` for outcomes that come from observing an
    *action*, which are not forecasts and must not distort calibration. The
    ``observed`` field is the real result in ``[0, 1]`` (``>= 0.5`` counts as the
    event happening), and ``context`` is the situation it was recorded in.
    """

    context: str
    action: str
    predicted_confidence: float | None
    observed: float
    source: str = "interaction"

    def __post_init__(self) -> None:
        if not self.context.strip():
            raise ValueError("Outcome context cannot be empty.")
        if not self.action.strip():
            raise ValueError("Outcome action cannot be empty.")
        if self.predicted_confidence is not None and not 0.0 <= self.predicted_confidence <= 1.0:
            raise ValueError("predicted_confidence must be between 0 and 1, or None.")
        if not 0.0 <= self.observed <= 1.0:
            raise ValueError("observed must be between 0 and 1.")

    @property
    def succeeded(self) -> bool:
        return self.observed >= 0.5

    @property
    def calibration_error(self) -> float:
        if self.predicted_confidence is None:
            return 0.0
        return abs(self.predicted_confidence - self.observed)


@dataclass
class CalibrationBin:
    lower: float
    upper: float
    count: int = 0
    confidence_sum: float = 0.0
    outcome_sum: float = 0.0

    @property
    def average_confidence(self) -> float:
        return self.confidence_sum / self.count if self.count else 0.0

    @property
    def observed_rate(self) -> float:
        return self.outcome_sum / self.count if self.count else 0.0

    @property
    def gap(self) -> float:
        return abs(self.average_confidence - self.observed_rate)


class InteractionRewardModel:
    """Learns reward from observed outcomes instead of a hand-written predicate.

    The model keeps a binned calibration table (confidence -> observed success)
    and a contextual action table (context -> action -> observed success). Both
    are updated only from real outcomes, and both are inspectable, so any reward
    the model reports can be explained.
    """

    def __init__(self, *, num_bins: int = 10, default_reward: float = 0.5) -> None:
        if num_bins < 1:
            raise ValueError("num_bins must be at least 1.")
        if not 0.0 <= default_reward <= 1.0:
            raise ValueError("default_reward must be between 0 and 1.")
        self.num_bins = num_bins
        self.default_reward = default_reward
        self._bins: list[CalibrationBin] = [
            CalibrationBin(index / num_bins, (index + 1) / num_bins)
            for index in range(num_bins)
        ]
        self._actions: dict[str, dict[str, float]] = {}
        self._action_counts: dict[str, dict[str, int]] = {}
        self._squared_error_sum = 0.0
        self.outcomes_recorded = 0
        self.forecasts_recorded = 0

    # -- recording ---------------------------------------------------------
    def _bin_for(self, confidence: float) -> CalibrationBin:
        index = min(int(confidence * self.num_bins), self.num_bins - 1)
        return self._bins[index]

    def record(self, outcome: Outcome) -> None:
        """Fold one real outcome into the calibration and action tables.

        Only forecast outcomes (those carrying a ``predicted_confidence``) feed
        the calibration table; action observations still inform the contextual
        action table but never move the calibration metric.
        """
        if outcome.predicted_confidence is not None:
            bucket = self._bin_for(outcome.predicted_confidence)
            bucket.count += 1
            bucket.confidence_sum += outcome.predicted_confidence
            bucket.outcome_sum += outcome.observed
            self._squared_error_sum += outcome.calibration_error ** 2
            self.forecasts_recorded += 1

        by_action = self._actions.setdefault(outcome.context, {})
        counts = self._action_counts.setdefault(outcome.context, {})
        by_action[outcome.action] = by_action.get(outcome.action, 0.0) + outcome.observed
        counts[outcome.action] = counts.get(outcome.action, 0) + 1
        self.outcomes_recorded += 1

    def record_many(self, outcomes: Iterable[Outcome]) -> None:
        for outcome in outcomes:
            self.record(outcome)

    # -- reward ------------------------------------------------------------
    def calibration_reward(self, predicted_confidence: float) -> float:
        """Reward for a *prediction*: the observed success rate for its confidence.

        A forecast is rewarded by what actually happened to forecasts like it,
        which is what keeps the reward calibrated rather than optimistic.
        """
        if not 0.0 <= predicted_confidence <= 1.0:
            raise ValueError("predicted_confidence must be between 0 and 1.")
        bucket = self._bin_for(predicted_confidence)
        if bucket.count == 0:
            return self.default_reward
        return round(bucket.observed_rate, 6)

    def action_reward(self, context: str, action: str) -> float:
        """Reward for *acting*: the observed success rate of that action here.

        Falls back to the overall observed success rate for the context, then to
        the default, so an unseen action is never silently over-rewarded.
        """
        sums = self._actions.get(context)
        counts = self._action_counts.get(context)
        if not sums or not counts:
            return self.default_reward
        if action in sums:
            return round(sums[action] / counts[action], 6)
        total = sum(sums.values())
        total_count = sum(counts.values())
        return round(total / total_count, 6) if total_count else self.default_reward

    def rank_actions(self, context: str) -> list[tuple[str, float]]:
        """Actions for a context, ordered by learned reward, highest first."""
        sums = self._actions.get(context, {})
        counts = self._action_counts.get(context, {})
        ranked = [
            (action, round(value / counts[action], 6))
            for action, value in sums.items()
            if counts.get(action)
        ]
        return sorted(ranked, key=lambda item: (-item[1], item[0]))

    # -- diagnostics -------------------------------------------------------
    def expected_calibration_error(self) -> float:
        if not self.forecasts_recorded:
            return 0.0
        total = sum(bucket.count * bucket.gap for bucket in self._bins)
        return round(total / self.forecasts_recorded, 6)

    def brier_score(self) -> float:
        """Mean squared gap between forecast and observed outcome.

        Computed exactly from per-forecast squared errors, which are tracked
        alongside the sums so the score is not a bin-level approximation.
        """
        if not self.forecasts_recorded:
            return 0.0
        return round(self._squared_error_sum / self.forecasts_recorded, 6)

    def summary(self) -> dict[str, float]:
        return {
            "outcomes_recorded": float(self.outcomes_recorded),
            "forecasts_recorded": float(self.forecasts_recorded),
            "expected_calibration_error": self.expected_calibration_error(),
            "brier_score": self.brier_score(),
            "contexts": float(len(self._actions)),
        }

    # -- persistence -------------------------------------------------------
    def to_dict(self) -> dict[str, object]:
        return {
            "num_bins": self.num_bins,
            "default_reward": self.default_reward,
            "outcomes_recorded": self.outcomes_recorded,
            "forecasts_recorded": self.forecasts_recorded,
            "squared_error_sum": self._squared_error_sum,
            "bins": [
                {
                    "lower": bucket.lower,
                    "upper": bucket.upper,
                    "count": bucket.count,
                    "confidence_sum": bucket.confidence_sum,
                    "outcome_sum": bucket.outcome_sum,
                }
                for bucket in self._bins
            ],
            "actions": self._actions,
            "action_counts": self._action_counts,
        }

    def save(self, path: str | Path) -> Path:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(self.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
        return target

    @classmethod
    def load(cls, path: str | Path) -> "InteractionRewardModel":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        model = cls(num_bins=int(data["num_bins"]), default_reward=float(data["default_reward"]))
        model.outcomes_recorded = int(data["outcomes_recorded"])
        model.forecasts_recorded = int(data.get("forecasts_recorded", 0))
        model._squared_error_sum = float(data.get("squared_error_sum", 0.0))
        for bucket, raw in zip(model._bins, data["bins"], strict=True):
            bucket.count = int(raw["count"])
            bucket.confidence_sum = float(raw["confidence_sum"])
            bucket.outcome_sum = float(raw["outcome_sum"])
        model._actions = {
            str(context): {str(action): float(value) for action, value in actions.items()}
            for context, actions in dict(data["actions"]).items()
        }
        model._action_counts = {
            str(context): {str(action): int(count) for action, count in counts.items()}
            for context, counts in dict(data.get("action_counts", {})).items()
        }
        return model


def _context_from_statement(statement: str) -> str:
    """A stable, readable context key derived from a prediction statement."""
    return " ".join(statement.lower().split())


def outcomes_from_evaluation_corpus() -> list[Outcome]:
    """Real observed outcomes already recorded in the project's evaluation corpus.

    Each forecast in ``evaluation.scenarios`` carries the outcome that was
    actually observed for it, so it is a genuine (predicted, observed) pair
    rather than a synthetic label. The chosen action is the forecast's own
    subject, and the context is the situation it was made in.
    """
    outcomes: list[Outcome] = []
    for scenario in SCENARIOS:
        for spec in scenario.predictions:
            if isinstance(spec, PredictionSpec):
                outcomes.append(
                    Outcome(
                        context=scenario.id,
                        action=_context_from_statement(spec.statement),
                        predicted_confidence=spec.confidence,
                        observed=spec.outcome,
                        source=f"evaluation:{scenario.id}",
                    )
                )
    return outcomes


def outcomes_from_reviews(
    reviews: Iterable[tuple[str, str, float, float]],
) -> list[Outcome]:
    """Turn human reviews into outcomes.

    Each review is ``(context, action, predicted_confidence, observed)`` where
    ``observed`` is the reviewer's recorded result in ``[0, 1]``. This is the
    real-interaction channel: a person judged what actually happened.
    """
    return [
        Outcome(context, action, confidence, observed, source="human-review")
        for context, action, confidence, observed in reviews
    ]


def sample_outcome(rng: random.Random | None = None) -> Outcome:
    """Draw one recorded outcome at random (used by the interaction RL loop)."""
    corpus = outcomes_from_evaluation_corpus()
    chooser = rng or random
    return chooser.choice(corpus)


__all__ = [
    "CalibrationBin",
    "InteractionRewardModel",
    "Outcome",
    "outcomes_from_evaluation_corpus",
    "outcomes_from_reviews",
    "sample_outcome",
]
