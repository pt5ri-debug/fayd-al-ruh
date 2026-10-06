"""Continuous learning: fold live interaction and new knowledge back in safely.

This is the loop that makes the architecture improve after it ships. It ties
together the pieces that already exist rather than adding a parallel path:

* :class:`core.interaction_reward.InteractionRewardModel` learns reward from real
  outcomes (never from a hand-written predicate);
* :class:`core.knowledge_update.KnowledgeUpdater` folds a confirmed answer into
  the retrievable corpus, with provenance and decay;
* :class:`RegressionMonitor` compares a fresh metric snapshot against a stored
  baseline and reports any drop, so a change that helps one metric but quietly
  breaks another is caught before it is trusted.

Two rules keep the loop honest. Learning only happens from *explicit feedback*
(``confirmed`` / ``corrected``), never from the model's own output; and a new
statement enters the corpus only when a human or a test confirms it. The
regression monitor is the safety net: it can always refuse a change.
"""

from __future__ import annotations

from dataclasses import dataclass

from core.interaction_reward import InteractionRewardModel, Outcome
from core.knowledge_update import KnowledgeUpdate, KnowledgeUpdater, UpdateOutcome


@dataclass(frozen=True)
class Feedback:
    """Explicit feedback on one assistant answer."""

    question: str
    answer: str
    confirmed: bool
    source: str = "human"
    confidence: float = 0.7
    correction: str = ""

    def __post_init__(self) -> None:
        if not self.question.strip():
            raise ValueError("Feedback question cannot be empty.")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("Feedback confidence must be between 0 and 1.")
        if not self.confirmed and not self.correction.strip():
            raise ValueError("A rejected answer needs a correction.")


@dataclass(frozen=True)
class LearningStep:
    """What one feedback event taught."""

    reward: float
    update: UpdateOutcome | None = None

    @property
    def changed_corpus(self) -> bool:
        return self.update is not None and self.update.action in {"added", "reinforced", "superseded"}


@dataclass(frozen=True)
class RegressionCheck:
    """One metric compared against its baseline."""

    name: str
    baseline: float
    current: float
    tolerance: float

    @property
    def dropped(self) -> bool:
        return self.current < self.baseline - self.tolerance

    @property
    def delta(self) -> float:
        return self.current - self.baseline


@dataclass(frozen=True)
class RegressionReport:
    """The result of comparing a full snapshot against the baseline."""

    checks: tuple[RegressionCheck, ...]
    tolerance: float

    @property
    def regressions(self) -> tuple[RegressionCheck, ...]:
        return tuple(check for check in self.checks if check.dropped)

    @property
    def passed(self) -> bool:
        return not self.regressions

    @property
    def improved(self) -> tuple[RegressionCheck, ...]:
        return tuple(check for check in self.checks if check.delta > self.tolerance)


class RegressionMonitor:
    """Compares metric snapshots against a baseline, with a tolerance band."""

    def __init__(self, baseline: dict[str, float], *, tolerance: float = 0.0) -> None:
        if tolerance < 0.0:
            raise ValueError("tolerance cannot be negative.")
        self.baseline = dict(baseline)
        self.tolerance = tolerance

    def check(self, current: dict[str, float]) -> RegressionReport:
        names = sorted(set(self.baseline) | set(current))
        checks = tuple(
            RegressionCheck(
                name,
                self.baseline.get(name, 0.0),
                current.get(name, 0.0),
                self.tolerance,
            )
            for name in names
        )
        return RegressionReport(checks, self.tolerance)


class ContinuousLearner:
    """Learns from explicit feedback and keeps the corpus growing honestly."""

    def __init__(
        self,
        updater: KnowledgeUpdater,
        *,
        reward_model: InteractionRewardModel | None = None,
    ) -> None:
        self.updater = updater
        self.reward_model = reward_model or InteractionRewardModel()
        self.steps: list[LearningStep] = []

    def learn(self, feedback: Feedback) -> LearningStep:
        """Record reward from the outcome, and fold a confirmed statement in."""
        context = _topic(feedback.question)
        reward = 1.0 if feedback.confirmed else 0.0
        self.reward_model.record(
            Outcome(
                context=context,
                action="answer",
                predicted_confidence=feedback.confidence,
                observed=reward,
                source=feedback.source,
            )
        )

        update: UpdateOutcome | None = None
        if feedback.confirmed:
            update = self.updater.apply(
                KnowledgeUpdate(
                    statement=feedback.answer,
                    source=feedback.source,
                    strength=feedback.confidence,
                    topic=context,
                )
            )
        elif feedback.correction.strip():
            update = self.updater.apply(
                KnowledgeUpdate(
                    statement=feedback.correction,
                    source=feedback.source,
                    strength=max(feedback.confidence, 0.6),
                    topic=context,
                )
            )

        step = LearningStep(reward=reward, update=update)
        self.steps.append(step)
        return step

    def learn_many(self, events) -> list[LearningStep]:
        return [self.learn(event) for event in events]

    def reward_for(self, question: str) -> float:
        """The learned reward for answering a question in this topic."""
        return self.reward_model.action_reward(_topic(question), "answer")

    def summary(self) -> dict[str, float]:
        summary = self.reward_model.summary()
        summary["learning_steps"] = float(len(self.steps))
        summary["corpus_changes"] = float(sum(step.changed_corpus for step in self.steps))
        return summary


def _topic(question: str) -> str:
    from core.rag import _query_terms

    terms = sorted(_query_terms(question))
    return terms[0] if terms else "general"


__all__ = [
    "ContinuousLearner",
    "Feedback",
    "LearningStep",
    "RegressionCheck",
    "RegressionMonitor",
    "RegressionReport",
]
