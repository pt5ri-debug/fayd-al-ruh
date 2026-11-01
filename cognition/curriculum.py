"""Adaptive curriculum: gradual training that chooses its own next step.

:class:`~cognition.training.IncrementalTrainer` already trains in stages and
refuses a version that regresses. What it could not do is *adapt*: it ran the
same stage the same way every time, could not tell a plateau from a bad learning
rate, and could not decide for itself how far to go. This module is that
intelligence on top of the trainer.

Four parts:

* :class:`TrainingMemory` -- the durable record of every attempt: the stage, the
  settings, the verdict, and the before/after perplexity. It is what lets the
  curriculum remember a plateau across runs rather than rediscovering it.
* :class:`AdaptivePolicy` -- given the memory, decide the next step: raise the
  stage when the last one was accepted, **adjust the learning rate** when a stage
  plateaus (a rejection with no measurable improvement), and stop when the
  programme is complete or the effort budget is spent. It is deterministic: the
  same memory always yields the same next step, and every decision names its
  reason.
* :class:`MultiSeedEvaluator` -- measure a candidate on **several seeds** and
  report the mean and standard deviation, so an acceptance decision is not made
  on one lucky split. Falls back to the trainer's own single evaluation when the
  multi-seed evaluation is not available.
* :class:`AutonomousCurriculum` -- the loop that ties them together: it advances
  the programme, adapts on a plateau, and stops on its own terms, bounded by an
  explicit effort budget so "autonomous" never means "unbounded".

The hard safety rules are unchanged: the base model is never modified in place,
a candidate is adopted only through the registry's acceptance gate, and a failed
run rolls back. The curriculum only decides *what to try next*, never *what to
believe*.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime

from cognition.storage import NAMESPACE_CURRICULUM, doc_get, doc_set
from cognition.training import (
    STAGE_SETTINGS,
    IncrementalTrainer,
    StageSettings,
    TrainingOutcome,
    TrainingStage,
)

#: A stage is "plateaued" after this many consecutive rejected attempts with no
#: improvement, and the policy then changes the learning rate rather than the
#: stage.
PLATEAU_AFTER_ATTEMPTS = 2

#: The learning-rate multipliers tried in order after a plateau, so a stage that
#: is stuck is retried gently before it is retried harder.
LR_MULTIPLIERS: tuple[float, ...] = (0.5, 0.25, 2.0)

#: The default effort budget: the most stages one autonomous run may attempt.
DEFAULT_MAX_STAGES = 8

MAX_ATTEMPTS = 400


def _now() -> datetime:
    return datetime.now(UTC)


# ---------------------------------------------------------------------------
# Training memory
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TrainingAttempt:
    """One recorded attempt: what was tried and what came of it."""

    stage: str
    accepted: bool
    reason: str
    learning_rate: float
    steps: int
    validation_ppl: float | None = None
    created_at: datetime = field(default_factory=_now)

    def improved(self) -> bool:
        """Whether this attempt improved on the model (not merely was accepted)."""
        return self.accepted and self.reason.startswith("improves")

    def to_primitive(self) -> dict[str, object]:
        return {
            "stage": self.stage,
            "accepted": self.accepted,
            "reason": self.reason,
            "learning_rate": self.learning_rate,
            "steps": self.steps,
            "validation_ppl": self.validation_ppl,
            "created_at": self.created_at.isoformat(),
        }

    @classmethod
    def from_primitive(cls, data: dict[str, object]) -> TrainingAttempt:
        def _num(key: str, fallback: float) -> float:
            value = data.get(key, fallback)
            return float(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else fallback

        raw_ppl = data.get("validation_ppl")
        ppl = float(raw_ppl) if isinstance(raw_ppl, (int, float)) and not isinstance(raw_ppl, bool) else None
        created = data.get("created_at")
        try:
            created_at = datetime.fromisoformat(str(created)) if created else _now()
        except ValueError:
            created_at = _now()
        raw_steps = data.get("steps")
        steps = raw_steps if isinstance(raw_steps, int) and not isinstance(raw_steps, bool) else 0
        return cls(
            stage=str(data.get("stage", "")),
            accepted=bool(data.get("accepted", False)),
            reason=str(data.get("reason", "")),
            learning_rate=_num("learning_rate", 0.0),
            steps=steps,
            validation_ppl=ppl,
            created_at=created_at,
        )

    def describe(self) -> str:
        verdict = "accepted" if self.accepted else "rejected"
        ppl = f", ppl {self.validation_ppl:.3f}" if self.validation_ppl is not None else ""
        return f"{self.stage} {verdict} at lr {self.learning_rate:g}{ppl} -- {self.reason}"


class TrainingMemory:
    """The durable log of training attempts, read by the adaptive policy."""

    def __init__(self, *, attempts: tuple[TrainingAttempt, ...] = (), max_items: int = MAX_ATTEMPTS) -> None:
        self.max_items = max_items
        self._attempts: list[TrainingAttempt] = list(attempts)

    @classmethod
    def load(cls, *, max_items: int = MAX_ATTEMPTS) -> TrainingMemory:
        data = doc_get(NAMESPACE_CURRICULUM)
        attempts: list[TrainingAttempt] = []
        if isinstance(data, dict):
            raw = data.get("attempts")
            if isinstance(raw, list):
                attempts = [TrainingAttempt.from_primitive(d) for d in raw if isinstance(d, dict)]
        return cls(attempts=tuple(attempts), max_items=max_items)

    def save(self) -> None:
        doc_set(
            NAMESPACE_CURRICULUM,
            {"attempts": [a.to_primitive() for a in self._attempts]},
        )

    def record(self, attempt: TrainingAttempt) -> TrainingAttempt:
        self._attempts.append(attempt)
        del self._attempts[:-self.max_items]
        self.save()
        return attempt

    def attempts(self) -> tuple[TrainingAttempt, ...]:
        return tuple(self._attempts)

    def for_stage(self, stage: TrainingStage) -> tuple[TrainingAttempt, ...]:
        return tuple(a for a in self._attempts if a.stage == stage.value)

    def consecutive_failures(self, stage: TrainingStage) -> int:
        """How many times in a row the stage was attempted and rejected."""
        count = 0
        for attempt in reversed(self.for_stage(stage)):
            if attempt.accepted:
                break
            count += 1
        return count

    def describe(self) -> str:
        if not self._attempts:
            return "No training attempts have been recorded yet."
        lines = [f"{len(self._attempts)} training attempt(s):"]
        lines.extend(f"  {a.describe()}" for a in self._attempts[-10:])
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Adaptive policy
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TrainingDecision:
    """What the curriculum decided to do next, and why."""

    action: str
    stage: TrainingStage | None
    settings: StageSettings | None
    reason: str

    def describe(self) -> str:
        if self.stage is None:
            return f"{self.action}: {self.reason}"
        return f"{self.action} {self.stage.value}: {self.reason}"


class AdaptivePolicy:
    """Choose the next training step from the memory of what has been tried."""

    def __init__(self, *, max_stages: int = DEFAULT_MAX_STAGES) -> None:
        self.max_stages = max_stages

    def decide(
        self,
        *,
        trainer: IncrementalTrainer,
        memory: TrainingMemory,
    ) -> TrainingDecision:
        """The next step: advance, adjust on a plateau, or stop."""
        nxt = trainer.next_stage()
        if nxt is None:
            return TrainingDecision("stop", None, None, "every stage has been accepted")
        stage = nxt
        attempts = memory.for_stage(stage)
        if len(attempts) >= self.max_stages:
            return TrainingDecision(
                "stop", stage, None,
                f"the effort budget for the {stage.value} stage is spent",
            )
        failures = memory.consecutive_failures(stage)
        base = STAGE_SETTINGS[stage]
        if failures >= PLATEAU_AFTER_ATTEMPTS:
            # Plateau: change the learning rate rather than the stage. The
            # multiplier cycles deterministically, so a stuck stage is retried
            # gently, then more gently, then harder -- never at random.
            multiplier = LR_MULTIPLIERS[(failures - PLATEAU_AFTER_ATTEMPTS) % len(LR_MULTIPLIERS)]
            adjusted = StageSettings(
                steps=base.steps,
                repeats=base.repeats,
                structured_repeats=base.structured_repeats,
                learning_rate=round(base.learning_rate * multiplier, 8),
                warmup=base.warmup,
                val_every=base.val_every,
                include_characters=base.include_characters,
            )
            return TrainingDecision(
                "adjust", stage, adjusted,
                f"{stage.value} has plateaued {failures}x; retry at lr "
                f"{adjusted.learning_rate:g} ({multiplier}x)",
            )
        return TrainingDecision("advance", stage, None, f"train the next unfinished stage ({stage.value})")


# ---------------------------------------------------------------------------
# Multi-seed evaluation
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SeedReading:
    """A candidate's held-out perplexity across seeds, honestly summarised."""

    seeds: tuple[int, ...]
    mean_ppl: float
    std_ppl: float
    count: int
    reason: str = ""

    @property
    def measured(self) -> bool:
        return self.count > 0

    def describe(self) -> str:
        if not self.measured:
            return f"multi-seed evaluation not available ({self.reason or 'no reading'})"
        return f"{self.count} seed(s): ppl {self.mean_ppl:.3f} +/- {self.std_ppl:.3f}"


class MultiSeedEvaluator:
    """Measure a checkpoint across several seeds, so one split cannot mislead.

    It reuses the project's own :func:`ml.llm.evaluate.evaluate_checkpoint` (which
    already splits whole documents and holds dropout off), and simply runs it for
    a few seeds and reports the mean and standard deviation. When the machinery
    is unavailable it says so honestly rather than inventing a figure.
    """

    def __init__(self, *, seeds: tuple[int, ...] = (0, 1, 2), train_repeats: int = 8) -> None:
        self.seeds = seeds
        self.train_repeats = train_repeats

    def measure(self, checkpoint, units: list[str]) -> SeedReading:
        if not units or len(units) < 3:
            return SeedReading(seeds=(), mean_ppl=0.0, std_ppl=0.0, count=0, reason="too few units")
        try:
            from ml.llm.evaluate import evaluate_checkpoint

            values: list[float] = []
            for seed in self.seeds:
                report = evaluate_checkpoint(
                    checkpoint, units, seed=seed, train_repeats=self.train_repeats
                )
                values.append(float(report.validation_ppl))
        except Exception as error:
            return SeedReading(seeds=(), mean_ppl=0.0, std_ppl=0.0, count=0, reason=type(error).__name__)
        mean = sum(values) / len(values)
        variance = sum((v - mean) ** 2 for v in values) / len(values)
        return SeedReading(
            seeds=tuple(self.seeds),
            mean_ppl=round(mean, 6),
            std_ppl=round(variance ** 0.5, 6),
            count=len(values),
        )


# ---------------------------------------------------------------------------
# Autonomous curriculum
# ---------------------------------------------------------------------------


@dataclass
class CurriculumReport:
    """What one autonomous curriculum run did."""

    decisions: tuple[TrainingDecision, ...] = ()
    outcomes: tuple[TrainingOutcome, ...] = ()
    stopped_because: str = ""

    def describe(self) -> str:
        lines = [f"autonomous curriculum: {len(self.outcomes)} run(s)"]
        lines.extend(f"  {d.describe()}" for d in self.decisions)
        if self.stopped_because:
            lines.append(f"stopped: {self.stopped_because}")
        return "\n".join(lines)


class AutonomousCurriculum:
    """Run gradual training that adapts to its own results, and stops itself.

    Each step consults the :class:`AdaptivePolicy` (advance, adjust, or stop),
    runs the trainer, and records the attempt. It stops when the programme is
    complete, when the policy's effort budget is spent, or when a run is rejected
    with no measurable improvement after the policy has already adjusted -- so an
    autonomous run always terminates on its own terms and never loops.
    """

    def __init__(
        self,
        *,
        trainer: IncrementalTrainer | None = None,
        memory: TrainingMemory | None = None,
        policy: AdaptivePolicy | None = None,
        evaluator: MultiSeedEvaluator | None = None,
    ) -> None:
        self.trainer = trainer or IncrementalTrainer()
        self.memory = memory or TrainingMemory.load()
        self.policy = policy or AdaptivePolicy()
        self.evaluator = evaluator or MultiSeedEvaluator()

    def run(
        self,
        *,
        approved: bool = False,
        max_stages: int | None = None,
        progress=None,
    ) -> CurriculumReport:
        """Advance the programme, adapting on a plateau, until it stops."""
        note = progress or (lambda _message: None)
        if not approved:
            return CurriculumReport(
                decisions=(TrainingDecision("stop", None, None, "not approved"),),
                stopped_because="training is a governed action and was not approved",
            )
        budget = max_stages if max_stages is not None else self.policy.max_stages
        decisions: list[TrainingDecision] = []
        outcomes: list[TrainingOutcome] = []
        stopped = ""
        for _ in range(max(1, budget)):
            decision = self.policy.decide(trainer=self.trainer, memory=self.memory)
            decisions.append(decision)
            if decision.action == "stop":
                stopped = decision.reason
                break
            if decision.stage is None:
                stopped = "the policy produced no stage to train"
                break
            note(decision.describe())
            outcome = self.trainer.run(stage=decision.stage, settings=decision.settings, progress=note)
            outcomes.append(outcome)
            ppl = self._validation_ppl(outcome)
            self.memory.record(
                TrainingAttempt(
                    stage=decision.stage.value,
                    accepted=outcome.accepted,
                    reason=outcome.reason,
                    learning_rate=outcome.plan.settings.learning_rate,
                    steps=outcome.plan.settings.steps,
                    validation_ppl=ppl,
                )
            )
            if not outcome.accepted and decision.action == "adjust":
                # Already adapted and still rejected: stop rather than thrash.
                stopped = f"{decision.stage.value} still failed after adjusting the learning rate"
                break
        else:
            stopped = "the effort budget was spent"
        return CurriculumReport(decisions=tuple(decisions), outcomes=tuple(outcomes), stopped_because=stopped)

    def measure_candidate(self) -> SeedReading:
        """Measure the active candidate across seeds (for the UI and the CLI)."""
        units = self.trainer._units()  # the trainer already builds the held-out units
        from cognition.versions import candidate_path

        return self.evaluator.measure(candidate_path(), units)

    def _validation_ppl(self, outcome: TrainingOutcome) -> float | None:
        """The candidate's held-out validation perplexity, if it was measured."""
        if outcome.version is None:
            return None
        for evaluation in outcome.version.evaluations:
            if evaluation.metric == "validation_ppl":
                return evaluation.after
        return None

    def describe(self) -> str:
        return self.memory.describe()


__all__ = [
    "DEFAULT_MAX_STAGES",
    "LR_MULTIPLIERS",
    "PLATEAU_AFTER_ATTEMPTS",
    "AdaptivePolicy",
    "AutonomousCurriculum",
    "CurriculumReport",
    "MultiSeedEvaluator",
    "SeedReading",
    "TrainingAttempt",
    "TrainingDecision",
    "TrainingMemory",
]
