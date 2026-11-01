"""Incremental, staged training of the project's own model, with a rollback gate.

Training follows the task's eight steps and its five safety rules:

1. **prepare the data** -- the base corpus plus the qualified examples the
   teacher-student system accepted (:func:`build_dataset`);
2. **continue training from the current model**, never from scratch, and never on
   the base file itself (a copy is used);
3. **evaluate before and after** on a held-out split;
4. **accept** only when nothing regresses and something improves
   (:meth:`cognition.versions.ModelRegistry.should_accept`);
5. **save as a new version** and point the active model at it;
6. **roll back** to the last stable version when a run is rejected;
7. keep the **base model immutable**, and keep the previous versions;
8. record everything -- dataset, settings, results, date -- in the registry.

A *stage* is a bounded amount of work, so training is gradual and resumable:

* ``SMOKE`` -- a few steps, to prove the pipeline runs (seconds);
* ``QUICK`` -- a short fit for a fast iteration (a minute or two);
* ``STANDARD`` -- a real fit (minutes);
* ``FULL`` -- a long fit (as long as the machine allows).

Each stage is a different ``steps``/``learning_rate``/``warmup``, so a user can
grow the model over time instead of one enormous run.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import Enum
from pathlib import Path

from core.paths import default_checkpoint

from cognition.records import TrainingExample
from cognition.storage import ExampleStore
from cognition.versions import (
    Evaluation,
    ModelRegistry,
    ModelVersion,
    base_path,
    candidate_path,
    models_root,
    version_path,
)


class TrainingStage(str, Enum):
    """How much work one training run does."""

    SMOKE = "smoke"
    QUICK = "quick"
    STANDARD = "standard"
    FULL = "full"


@dataclass(frozen=True)
class StageSettings:
    """The trainer settings one stage uses."""

    steps: int
    repeats: int
    structured_repeats: int
    learning_rate: float
    warmup: int
    val_every: int
    include_characters: bool = True

    def describe(self) -> str:
        return (
            f"{self.steps} steps, lr {self.learning_rate}, warmup {self.warmup}, "
            f"repeats {self.repeats}/{self.structured_repeats}"
        )


#: One entry per stage, tuned so each stage is meaningfully larger than the last.
STAGE_SETTINGS: dict[TrainingStage, StageSettings] = {
    TrainingStage.SMOKE: StageSettings(
        steps=40, repeats=4, structured_repeats=6, learning_rate=3e-3,
        warmup=4, val_every=20,
    ),
    TrainingStage.QUICK: StageSettings(
        steps=300, repeats=8, structured_repeats=12, learning_rate=2e-3,
        warmup=15, val_every=100,
    ),
    TrainingStage.STANDARD: StageSettings(
        steps=1500, repeats=20, structured_repeats=30, learning_rate=1.5e-3,
        warmup=75, val_every=250,
    ),
    TrainingStage.FULL: StageSettings(
        steps=4000, repeats=40, structured_repeats=60, learning_rate=1e-3,
        warmup=200, val_every=500,
    ),
}


@dataclass
class TrainingPlan:
    """What a run will do, decided before any work starts."""

    stage: TrainingStage
    settings: StageSettings
    examples: int
    dataset_chars: int
    base: Path
    candidate: Path
    reason: str = ""

    def describe(self) -> str:
        return (
            f"{self.stage.value} stage: {self.settings.describe()}; "
            f"{self.examples} example(s), {self.dataset_chars} chars; "
            f"from {self.base.name} -> {self.candidate.name}"
        )


@dataclass
class TrainingOutcome:
    """What a run actually did, and whether it was accepted."""

    plan: TrainingPlan
    version: ModelVersion | None
    accepted: bool
    reason: str
    report: dict[str, object] = field(default_factory=dict)

    def describe(self) -> str:
        verdict = "accepted" if self.accepted else "rejected"
        return f"{self.stage().value} run {verdict}: {self.reason}"

    def stage(self) -> TrainingStage:
        return self.plan.stage


def stages() -> tuple[TrainingStage, ...]:
    """The stages, smallest first -- what the UI lists."""
    return tuple(STAGE_SETTINGS)


@dataclass
class StageProgress:
    """One stage's place in a gradual training programme."""

    stage: TrainingStage
    accepted: int
    attempts: int
    last_reason: str = ""

    @property
    def done(self) -> bool:
        """A stage is done once a run at it has been accepted."""
        return self.accepted > 0

    def describe(self) -> str:
        mark = "done" if self.done else ("failed" if self.attempts else "not started")
        return f"{self.stage.value}: {mark} ({self.accepted} accepted / {self.attempts} tried)"


@dataclass
class GradualReport:
    """What one gradual-training pass did, stage by stage."""

    progress: tuple[StageProgress, ...]
    outcomes: tuple[TrainingOutcome, ...]
    stopped_at: TrainingStage | None = None
    reason: str = ""

    def advanced(self) -> tuple[TrainingStage, ...]:
        return tuple(o.plan.stage for o in self.outcomes if o.accepted)

    def describe(self) -> str:
        lines = [p.describe() for p in self.progress]
        tail = f"stopped at {self.stopped_at.value}: {self.reason}" if self.stopped_at else "complete"
        return "\n".join([*lines, tail])


class IncrementalTrainer:
    """Run staged, resumable training with a candidate/version/rollback gate."""

    def __init__(
        self,
        *,
        registry: ModelRegistry | None = None,
        examples: ExampleStore | None = None,
        corpus_builder: Callable[..., str] | None = None,
        units_builder: Callable[..., list[str]] | None = None,
        trainer: Callable[..., object] | None = None,
    ) -> None:
        self.registry = registry or ModelRegistry()
        self.examples = examples or ExampleStore()
        self._corpus_builder = corpus_builder
        self._units_builder = units_builder
        self._trainer = trainer

    # -- gradual training ---------------------------------------------------
    def progression(self) -> tuple[StageProgress, ...]:
        """How far the gradual programme has come, derived from the version log.

        The registry already records the stage of every run, so the progression is
        read from it rather than stored a second time -- one source of truth for
        "which stage are we on". A stage counts as done once a run at it was
        **accepted**; a rejected run is remembered as an attempt, not a success.
        """
        by_stage: dict[TrainingStage, StageProgress] = {
            stage: StageProgress(stage=stage, accepted=0, attempts=0) for stage in stages()
        }
        for version in self.registry.versions():
            raw = version.settings.get("stage")
            try:
                stage = TrainingStage(str(raw))
            except ValueError:
                continue
            current = by_stage[stage]
            by_stage[stage] = StageProgress(
                stage=stage,
                accepted=current.accepted + (1 if version.accepted else 0),
                attempts=current.attempts + 1,
                last_reason=version.reason or current.last_reason,
            )
        return tuple(by_stage[stage] for stage in stages())

    def next_stage(self) -> TrainingStage | None:
        """The smallest stage not yet completed, or ``None`` when all are done."""
        for progress in self.progression():
            if not progress.done:
                return progress.stage
        return None

    def run_gradual(
        self,
        *,
        up_to: TrainingStage | None = None,
        progress: Callable[[str], None] | None = None,
    ) -> GradualReport:
        """Advance the programme one stage at a time, stopping on a failure.

        Gradual training is the task's "التدريب التدريجي": each call trains the
        next unfinished stage, and the run stops the moment a stage is **rejected**
        -- a failed stage means the model is not ready for more work, so the next
        one is never attempted on top of it. When a stage is accepted the loop
        moves on, and a later call continues from where it left off, because the
        progression is read from the registry rather than held in memory.
        """
        note = progress or (lambda _message: None)
        order = stages()
        limit = order.index(up_to) if up_to is not None else len(order) - 1
        outcomes: list[TrainingOutcome] = []
        stopped: TrainingStage | None = None
        reason = ""

        for index, stage in enumerate(order):
            if index > limit:
                break
            state = self.progression()[index]
            if state.done:
                note(f"{stage.value} already done; skipping")
                continue
            note(f"gradual training: {stage.value} stage")
            outcome = self.run(stage=stage, progress=note)
            outcomes.append(outcome)
            if not outcome.accepted:
                stopped = stage
                reason = outcome.reason
                break

        if stopped is None:
            reason = "all requested stages complete" if limit == len(order) - 1 else "up to the requested stage"
        return GradualReport(
            progress=self.progression(),
            outcomes=tuple(outcomes),
            stopped_at=stopped,
            reason=reason,
        )

    # -- dataset ------------------------------------------------------------
    def build_dataset(self, *, stage: TrainingStage = TrainingStage.QUICK) -> str:
        """The training text: the base corpus plus the qualified examples.

        Only **qualified** examples are added, and they are appended as plain
        ``Question: ... Answer: ...`` blocks, which is the shape the model is
        already trained on. The base corpus is produced by the project's own
        :func:`ml.llm.corpus.build_corpus`, so nothing is duplicated here.
        """
        settings = STAGE_SETTINGS[stage]
        builder = self._corpus_builder
        if builder is None:
            from ml.llm.corpus import build_corpus

            builder = build_corpus
        base = builder(
            repeats=settings.repeats,
            structured_repeats=settings.structured_repeats,
            include_characters=settings.include_characters,
        )
        qualified = self.examples.qualified()
        if not qualified:
            return base
        blocks = [self._example_block(e) for e in qualified]
        return base + "\n\n" + "\n\n".join(blocks)

    def _example_block(self, example: TrainingExample) -> str:
        return f"Question: {example.task.strip()}\nAnswer: {example.answer.strip()}"

    def plan(
        self,
        *,
        stage: TrainingStage = TrainingStage.QUICK,
        settings: StageSettings | None = None,
    ) -> TrainingPlan:
        """Prepare a run: copy the base model once, then describe the work.

        ``settings`` overrides the stage's own defaults, so the adaptive
        curriculum can retry a stage with a gentler learning rate or a longer
        warmup without inventing a new stage.
        """
        source = default_checkpoint()
        base = self.registry.ensure_base(source) if source.exists() else base_path()
        effective = settings or STAGE_SETTINGS[stage]
        dataset = self.build_dataset(stage=stage)
        return TrainingPlan(
            stage=stage,
            settings=effective,
            examples=self.examples.count(),
            dataset_chars=len(dataset),
            base=base,
            candidate=candidate_path(),
        )

    # -- run ----------------------------------------------------------------
    def run(
        self,
        *,
        stage: TrainingStage = TrainingStage.QUICK,
        settings: StageSettings | None = None,
        progress: Callable[[str], None] | None = None,
    ) -> TrainingOutcome:
        """Run one stage: train a candidate, evaluate it, accept or roll back."""
        plan = self.plan(stage=stage, settings=settings)
        note = progress or (lambda _message: None)

        if not plan.base.exists():
            return TrainingOutcome(
                plan=plan, version=None, accepted=False,
                reason=f"no base model at {plan.base}; train one with `python -m ml.llm`",
            )

        note(f"preparing data ({plan.dataset_chars} chars, {plan.examples} example(s))")
        before = self._evaluate(plan.base)
        note(f"baseline: {self._summarise(before)}")

        note(f"training a candidate from {plan.base.name}")
        trained = self._train(plan, note)
        if not trained:
            return TrainingOutcome(
                plan=plan, version=None, accepted=False, reason="training produced no model",
            )

        after = self._evaluate(plan.candidate)
        note(f"candidate: {self._summarise(after)}")

        evaluations = self._evaluations(before, after)
        accepted, reason = self.registry.should_accept(evaluations)
        number = self.registry.next_number()
        final_path = version_path(number) if accepted else plan.candidate

        version = ModelVersion(
            number=number,
            path=str(final_path),
            parent=self.registry.active_number(),
            dataset=f"base + {plan.examples} example(s)",
            examples=plan.examples,
            steps=plan.settings.steps,
            settings={
                "stage": plan.stage.value,
                "learning_rate": plan.settings.learning_rate,
                "repeats": plan.settings.repeats,
                "structured_repeats": plan.settings.structured_repeats,
            },
            evaluations=evaluations,
            accepted=accepted,
            reason=reason,
            created_at=datetime.now(UTC),
        )

        if accepted:
            self._promote(plan.candidate, final_path)
            self.registry.record(version)
            self.registry.set_active(number)
            note(f"accepted as v{number}: {reason}")
        else:
            self.registry.record(version)
            stable = self.registry.rollback()
            note(
                "rejected: "
                + reason
                + (f"; stayed on v{stable.number}" if stable else "; stayed on the base model")
            )

        return TrainingOutcome(
            plan=plan,
            version=version,
            accepted=accepted,
            reason=reason,
            report={"before": self._summarise(before), "after": self._summarise(after)},
        )

    def rollback(self) -> ModelVersion | None:
        """Return to the last accepted version (delegates to the registry)."""
        return self.registry.rollback()

    # -- internals ----------------------------------------------------------
    def _train(self, plan: TrainingPlan, note: Callable[[str], None]) -> bool:
        """Continue training from the base model's own weights.

        The base checkpoint is copied to the candidate (weights *and* tokenizer)
        and the trainer is told to **resume** from it, so the candidate keeps the
        base model's architecture and vocabulary. Building a fresh model instead
        would silently train a different network -- which is why the architecture
        is never chosen here, it is read from the base.
        """
        import io
        import shutil
        from contextlib import redirect_stdout

        plan.candidate.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(plan.base, plan.candidate)
        base_tokenizer = plan.base.with_suffix(".tokenizer.json")
        candidate_tokenizer = plan.candidate.with_suffix(".tokenizer.json")
        if base_tokenizer.exists():
            shutil.copy2(base_tokenizer, candidate_tokenizer)

        trainer = self._trainer
        if trainer is None:
            from ml.llm.__main__ import train_and_save

            trainer = train_and_save
        buffer = io.StringIO()
        try:
            with redirect_stdout(buffer):
                trainer(
                    steps=plan.settings.steps,
                    repeats=plan.settings.repeats,
                    structured_repeats=plan.settings.structured_repeats,
                    checkpoint=plan.candidate,
                    tokenizer_path=candidate_tokenizer,
                    learning_rate=plan.settings.learning_rate,
                    warmup=plan.settings.warmup,
                    val_every=plan.settings.val_every,
                    include_characters=plan.settings.include_characters,
                    resume=True,
                    verbose=False,
                )
        except Exception as error:
            note(f"training failed: {type(error).__name__}: {error}")
            return False
        finally:
            for line in buffer.getvalue().splitlines():
                if line.strip():
                    note(line.strip())
        return plan.candidate.exists()

    def _evaluate(self, checkpoint: Path) -> dict[str, float]:
        """Held-out perplexity for a checkpoint (empty when it cannot be measured)."""
        if not checkpoint.exists():
            return {}
        try:
            from ml.llm.evaluate import evaluate_checkpoint

            units = self._units()
            if len(units) < 3:
                return {}
            report = evaluate_checkpoint(checkpoint, units)
            return {
                "validation_ppl": round(float(report.validation_ppl), 6),
                "test_ppl": round(float(report.test_ppl), 6),
                "train_ppl": round(float(report.train_ppl), 6),
            }
        except Exception:
            return {}

    def _units(self) -> list[str]:
        if self._units_builder is not None:
            return self._units_builder()
        try:
            from ml.llm.corpus import corpus_units

            return corpus_units()
        except Exception:
            return []

    def _evaluations(
        self, before: dict[str, float], after: dict[str, float]
    ) -> tuple[Evaluation, ...]:
        """Turn the before/after perplexities into improvements (lower is better).

        Perplexity is *lower is better*, so the "improvement" reported to the
        acceptance gate is the fall in perplexity, expressed so that a positive
        delta always means better. A metric missing from either side is skipped,
        never guessed.
        """
        evaluations: list[Evaluation] = []
        for metric in ("validation_ppl", "test_ppl"):
            if metric in before and metric in after:
                evaluations.append(
                    Evaluation(
                        metric=metric,
                        before=before[metric],
                        after=after[metric],
                        lower_is_better=True,
                    )
                )
        return tuple(evaluations)

    def _summarise(self, figures: dict[str, float]) -> str:
        if not figures:
            return "not measured"
        return ", ".join(f"{k}={v:.3f}" for k, v in sorted(figures.items()))

    def _promote(self, candidate: Path, final: Path) -> None:
        import shutil

        final.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(candidate, final)
        tokenizer = candidate.with_suffix(".tokenizer.json")
        if tokenizer.exists():
            shutil.copy2(tokenizer, final.with_suffix(".tokenizer.json"))


__all__ = [
    "STAGE_SETTINGS",
    "GradualReport",
    "IncrementalTrainer",
    "StageProgress",
    "StageSettings",
    "TrainingOutcome",
    "TrainingPlan",
    "TrainingStage",
    "models_root",
    "stages",
]
