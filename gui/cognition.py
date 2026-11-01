"""The Autonomous Cognitive System service: the engine behind the window and CLI.

One Tk-free service drives the whole cognition surface, so the cognition window
and ``python -m cli cognition`` share **one** proactive mind, **one** learning
system, **one** version log, and **one** evaluator -- no logic is reimplemented
in either front end. It wraps the objects that already exist:

* :class:`cognition.proactive_mind.ProactiveMind` -- thinking cycles, ideas, drives;
* :class:`cognition.learning_system.CognitiveLearningSystem` -- facts, recall, experience;
* :class:`cognition.training.IncrementalTrainer` -- staged and gradual training,
  the version log, and the rollback gate;
* :class:`cognition.evaluation.CognitiveEvaluator` -- the ten indicators;
* :class:`cognition.teacher_student.TeacherStudentSystem` -- the student-first exchange;
* :class:`gui.character_evaluation.CharacterEvaluator` -- the eight characters.

Everything is exercised headless: no display, no model, and no network. The
proactive mind, the learning system, and the evaluator share one
:class:`~core.cognitive_controller.CognitiveController`, so the autonomous layer
reasons over the *same* ontology and causal graph the answer path does -- never a
second graph. Training is a **governed** action: :meth:`train` refuses unless it
is explicitly approved, exactly like every other external action in the project.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from cognition.evaluation import INDICATORS, LABELS, CognitiveReport
from cognition.learning_system import CognitiveLearningSystem
from cognition.proactive_mind import ProactiveMind
from cognition.records import Source, SourceKind
from cognition.runner import CognitiveRunner, ResourceMode
from cognition.training import STAGE_SETTINGS, IncrementalTrainer, TrainingStage
from cognition.versions import ModelRegistry, ModelVersion

from gui.character_evaluation import CharacterEvaluator, EvaluationReport


def _controller() -> Any:
    """Build a fresh controller (the single shared ontology for this service)."""
    from core.cognitive_controller import CognitiveController

    return CognitiveController()


def _fact_id(statement: str) -> str:
    """A stable, content-addressed id for a statement.

    ``hash()`` is salted per process, so an id built from it would change on every
    run and the same statement would be stored twice; a digest of the normalised
    text is the same across runs.
    """
    from hashlib import blake2b

    from core.tokenize import normalize_phrase

    digest = blake2b(normalize_phrase(statement).encode("utf-8"), digest_size=6)
    return f"gui-{digest.hexdigest()}"


@dataclass
class OperationReport:
    """A readable result for an action that either ran or was refused."""

    ok: bool
    title: str
    lines: tuple[str, ...] = ()
    detail: str = ""

    def describe(self) -> str:
        body = "\n".join(f"  {line}" for line in self.lines)
        text = f"{self.title}" + (f"\n{body}" if body else "")
        if self.detail:
            text += f"\n\n{self.detail}"
        return text


@dataclass
class StageRow:
    """One row of the gradual-training progression."""

    stage: str
    accepted: int
    attempts: int
    done: bool
    description: str = ""


@dataclass
class ProgressionReport:
    """How far gradual training has come, and what stage is next."""

    rows: tuple[StageRow, ...] = ()
    next_stage: str | None = None

    def describe(self) -> str:
        if not self.rows:
            return "No training stages have run yet."
        lines = [row.description or f"{row.stage}: {'done' if row.done else 'pending'}" for row in self.rows]
        nxt = self.next_stage or "none (all stages done)"
        lines.append(f"next stage: {nxt}")
        return "\n".join(lines)


@dataclass
class IdeaRow:
    """One idea awaiting review, as the window shows it."""

    text: str
    kind: str
    engine: str
    confidence: float
    novelty: float
    critique: tuple[str, ...] = ()

    def describe(self) -> str:
        lines = [
            f"[{self.kind}] {self.text}",
            f"    engine {self.engine}  confidence {self.confidence:.2f}  novelty {self.novelty:.2f}",
        ]
        lines.extend(f"    critique: {note}" for note in self.critique)
        return "\n".join(lines)


@dataclass
class RecallReport:
    """The result of recalling a query from every memory."""

    query: str
    found: dict[str, list[str]] = field(default_factory=dict)

    def describe(self) -> str:
        lines = [f"Recall for: {self.query}"]
        for kind, items in self.found.items():
            lines.append(f"  {kind}:")
            if items:
                lines.extend(f"    - {item}" for item in items)
            else:
                lines.append("    (nothing)")
        return "\n".join(lines)


class CognitiveService:
    """The one engine behind the cognition window and the CLI.

    ``data_root`` keeps the character stores and configuration under one folder
    (the window passes the project's ``data_dir``); a test passes a temporary one.
    The controller, the trainer, and the evaluator are built lazily, so opening
    the window costs nothing until an action is run.
    """

    def __init__(
        self,
        *,
        data_root: Path | None = None,
        controller: Any | None = None,
        trainer: IncrementalTrainer | None = None,
        evaluator: Any | None = None,
        character_evaluator: CharacterEvaluator | None = None,
    ) -> None:
        self._data_root = data_root
        self._controller = controller
        self._trainer = trainer
        self._evaluator = evaluator
        self._character_evaluator = character_evaluator

    # -- lazily built engines ----------------------------------------------

    @property
    def controller(self) -> Any:
        if self._controller is None:
            self._controller = _controller()
        return self._controller

    @property
    def mind(self) -> ProactiveMind:
        return self.controller.proactive_mind()

    @property
    def learning(self) -> CognitiveLearningSystem:
        return self.controller.cognitive_learning()

    @property
    def trainer(self) -> IncrementalTrainer:
        if self._trainer is None:
            self._trainer = IncrementalTrainer()
        return self._trainer

    @property
    def registry(self) -> ModelRegistry:
        return self.trainer.registry

    # -- proactive mind -----------------------------------------------------

    def status(self) -> dict[str, Any]:
        """The mind's own snapshot (a plain dict, as the CLI prints it)."""
        return self.mind.status()

    def metacognition(self) -> OperationReport:
        """The mind's model of its own thinking: which strategies pay off."""
        return OperationReport(
            ok=True,
            title="Metacognition — the mind's self-model",
            lines=tuple(self.mind.metacognition.describe().splitlines()),
        )

    def plan_next(self) -> OperationReport:
        """How the mind intends to search next, and why."""
        plan = self.mind.plan_next()
        return OperationReport(
            ok=True,
            title="Metacognitive plan for the next cycle",
            lines=tuple(plan.describe().splitlines()),
        )

    def think(self, *, max_ideas: int | None = None) -> OperationReport:
        """Run one thinking cycle and report it."""
        cycle = self.mind.cycle(max_ideas=max_ideas)
        lines = [cycle.describe()]
        lines.extend(f"drive: {note}" for note in cycle.notes)
        return OperationReport(ok=True, title="Thinking cycle", lines=tuple(lines))

    def ideas(self, *, limit: int = 20) -> tuple[IdeaRow, ...]:
        """The ideas awaiting review."""
        return tuple(
            IdeaRow(
                text=idea.text,
                kind=idea.kind.value,
                engine=idea.engine,
                confidence=idea.confidence,
                novelty=idea.novelty,
                critique=idea.critique,
            )
            for idea in self.mind.pending_ideas(limit=limit)
        )

    def drives(self) -> tuple[str, ...]:
        """What the mind would pursue next, as readable lines."""
        return tuple(drive.describe() for drive in self.mind.top_drives())

    # -- cognitive learning -------------------------------------------------

    def learn(
        self,
        statement: str,
        *,
        source: str = "user",
        confidence: float = 0.6,
        verified: bool = False,
    ) -> OperationReport:
        """Learn a statement, recording its source and whether it is trusted."""
        try:
            kind = SourceKind(source)
        except ValueError:
            kind = SourceKind.USER
        trusted = self.learning.learn_fact(
            _fact_id(statement),
            statement,
            source=Source(kind, "cognition-window"),
            confidence=confidence,
            verified=verified,
        )
        verdict = "trusted fact" if trusted else "stored, not a fact (unverified)"
        return OperationReport(ok=True, title=f"Learned: {statement}", lines=(verdict,))

    def recall(self, query: str, *, limit: int = 5) -> RecallReport:
        """Retrieve from every memory the learning system holds."""
        found = self.learning.recall(query, limit=limit)
        return RecallReport(query=query, found={k: list(v) for k, v in found.items()})

    def experience(self, *, expectation: str, outcome: str, success: bool = True) -> OperationReport:
        """Record an expectation and its outcome as an experience."""
        record = self.learning.learn_from_error(
            expectation=expectation,
            expected_confidence=0.8,
            outcome=outcome,
            success=success,
        )
        return OperationReport(
            ok=True,
            title=record.describe(),
            lines=(f"lesson: {record.lesson}",),
        )

    def learning_stats(self) -> str:
        """One line summarising what the learning system holds."""
        return self.learning.stats().describe()

    def consolidate(self) -> OperationReport:
        """Consolidate raw experience into durable, reviewed knowledge."""
        report = self.learning.consolidate()
        lines = [f"{len(report.patterns)} pattern(s) extracted"]
        lines.extend(f"pattern: {p.describe()}" for p in report.patterns)
        lines.extend(f"upgraded to a learned fact: {statement}" for statement in report.upgraded)
        lines.append(f"{report.scheduled} item(s) now under spaced review")
        return OperationReport(ok=True, title="Consolidation", lines=tuple(lines))

    def patterns(self) -> tuple[str, ...]:
        """The regularities the mind has extracted from its own experience."""
        return tuple(p.describe() for p in self.learning.consolidator().patterns())

    def review_due(self, *, remembered: bool = True) -> OperationReport:
        """Settle the spaced-review items that are due."""
        due = self.learning.consolidator().review_due(remembered=remembered)
        if not due:
            return OperationReport(ok=True, title="No knowledge is due for review yet.")
        return OperationReport(
            ok=True,
            title=f"Reviewed {len(due)} item(s)",
            lines=tuple(item.describe() for item in due),
        )

    def practise_skill(self, name: str, *, success: bool, steps: tuple[str, ...] = ()) -> OperationReport:
        """Practise a procedure and record its outcome (skill refinement)."""
        skill = self.learning.consolidator().practise_skill(name, success=success, steps=steps)
        return OperationReport(ok=True, title=f"Practised {name}", lines=(skill.describe(),))

    def skills(self) -> tuple[str, ...]:
        """The procedures the mind has practised, with their proficiency."""
        return tuple(s.describe() for s in self.learning.consolidator().skills.skills())

    # -- staged / gradual training -----------------------------------------

    def stages(self) -> tuple[TrainingStage, ...]:
        return tuple(TrainingStage)

    def stage_settings(self, stage: TrainingStage) -> str:
        return STAGE_SETTINGS[stage].describe()

    def train(
        self,
        *,
        stage: TrainingStage = TrainingStage.SMOKE,
        approved: bool = False,
        progress: Callable[[str], None] | None = None,
    ) -> OperationReport:
        """Run one staged training round. Refused unless ``approved`` is true."""
        if not approved:
            return OperationReport(
                ok=False,
                title="Training is a governed action.",
                detail="Confirm the run to proceed: training writes a candidate model "
                "version that is adopted only when it does not regress.",
            )
        outcome = self.trainer.run(stage=stage, progress=progress)
        return OperationReport(
            ok=outcome.accepted,
            title=outcome.describe(),
            lines=tuple(self.registry.report().splitlines()),
        )

    def run_gradual(
        self,
        *,
        up_to: TrainingStage | None = None,
        approved: bool = False,
        progress: Callable[[str], None] | None = None,
    ) -> OperationReport:
        """Advance stage by stage, stopping on the first rejected stage."""
        if not approved:
            return OperationReport(
                ok=False,
                title="Gradual training is a governed action.",
                detail="Confirm the run to proceed: it advances one stage at a time and "
                "stops at the first stage the evaluation does not accept.",
            )
        report = self.trainer.run_gradual(up_to=up_to, progress=progress)
        advanced = ", ".join(s.value for s in report.advanced()) or "none"
        return OperationReport(
            ok=report.stopped_at is None,
            title=report.describe(),
            lines=(f"advanced: {advanced}",),
            detail=report.reason,
        )

    def progression(self) -> ProgressionReport:
        """How far gradual training has come, and what stage is next."""
        rows = tuple(
            StageRow(
                stage=p.stage.value,
                accepted=p.accepted,
                attempts=p.attempts,
                done=p.done,
                description=p.describe(),
            )
            for p in self.trainer.progression()
        )
        nxt = self.trainer.next_stage()
        return ProgressionReport(rows=rows, next_stage=None if nxt is None else nxt.value)

    def run_autonomous(
        self,
        *,
        max_stages: int | None = None,
        approved: bool = False,
        progress: Callable[[str], None] | None = None,
    ) -> OperationReport:
        """Run the adaptive curriculum: advance, adapt on a plateau, stop itself."""
        if not approved:
            return OperationReport(
                ok=False,
                title="Autonomous training is a governed action.",
                detail="Confirm the run to proceed: it advances the stages on its own, "
                "adjusts the learning rate when a stage plateaus, and stops at an "
                "explicit effort budget -- adopting a version only through the gate.",
            )
        from cognition.curriculum import AutonomousCurriculum

        curriculum = AutonomousCurriculum(trainer=self.trainer)
        report = curriculum.run(max_stages=max_stages, approved=True, progress=progress)
        lines = [f"{len(report.outcomes)} run(s)"]
        lines.extend(d.describe() for d in report.decisions)
        lines.append(f"stopped: {report.stopped_because}")
        return OperationReport(
            ok=any(o.accepted for o in report.outcomes),
            title="Autonomous curriculum",
            lines=tuple(lines),
        )

    def curriculum_memory(self) -> tuple[str, ...]:
        """The log of training attempts the adaptive policy reads."""
        from cognition.curriculum import TrainingMemory

        return tuple(a.describe() for a in TrainingMemory.load().attempts())

    def measure_candidate(self) -> OperationReport:
        """Measure the active candidate across seeds, honestly."""
        from cognition.curriculum import MultiSeedEvaluator
        from cognition.versions import candidate_path

        reading = MultiSeedEvaluator().measure(candidate_path(), self.trainer._units())
        return OperationReport(ok=reading.measured, title=reading.describe())

    def versions(self) -> tuple[ModelVersion, ...]:
        return self.registry.versions()

    def versions_report(self) -> str:
        return self.registry.report()

    def rollback(self) -> OperationReport:
        """Return to the last accepted version, or report the base model in use."""
        stable = self.registry.rollback()
        if stable is None:
            return OperationReport(
                ok=True,
                title="No accepted version to roll back to; the base model is in use.",
            )
        return OperationReport(ok=True, title=f"Rolled back to v{stable.number}.")

    # -- evaluation ---------------------------------------------------------

    def evaluate(self, *, questions: tuple[str, ...] = ()) -> CognitiveReport:
        """Run the ten cognitive indicators against the live project."""
        if self._evaluator is None:
            from cognition.evaluation import CognitiveEvaluator

            self._evaluator = CognitiveEvaluator(self.controller.ontology)
        return self._evaluator.evaluate(questions=questions)

    def indicators(self) -> tuple[tuple[str, str], ...]:
        """The ten indicators as ``(key, readable label)`` pairs."""
        return tuple((key, LABELS[key]) for key in INDICATORS)

    # -- teacher / student --------------------------------------------------

    def teach(self, question: str) -> OperationReport:
        """Run one student-first exchange and report it."""
        from cognition.backends import build_backends
        from cognition.teacher_student import TeacherStudentSystem

        student, teacher = build_backends()
        if not student.available():
            return OperationReport(
                ok=False,
                title="The student model is not available.",
                detail="Train one first: python -m ml.llm  (or the Training tab here).",
            )
        system = TeacherStudentSystem(student, teacher, ontology=self.controller.ontology)
        outcome = system.teach(question)
        lines = list(outcome.notes)
        if outcome.example is not None:
            lines.append(f"example: {outcome.example.describe()}")
        return OperationReport(ok=True, title=outcome.describe(), lines=tuple(lines))

    # -- resource mode and background ticks ---------------------------------

    def mode(self) -> ResourceMode:
        return CognitiveRunner(self.mind).mode

    def set_mode(self, mode: ResourceMode) -> OperationReport:
        CognitiveRunner(self.mind).set_mode(mode)
        return OperationReport(ok=True, title=f"Resource mode set to {mode.value}.")

    def tick(self) -> OperationReport:
        """One background tick: run a cycle if the scheduler says it is due."""
        runner = CognitiveRunner(self.mind)
        cycle = runner.tick()
        if cycle is None:
            decision = runner.decision()
            reason = getattr(decision, "reason", "not due")
            return OperationReport(ok=True, title=f"No cycle this tick: {reason}")
        return OperationReport(ok=True, title=cycle.describe())

    # -- characters ---------------------------------------------------------

    def evaluate_characters(self) -> EvaluationReport:
        """Evaluate the eight characters from their real configuration and stores."""
        if self._character_evaluator is None:
            self._character_evaluator = CharacterEvaluator(data_root=self._data_root)
        return self._character_evaluator.evaluate()

    # -- a whole honest demonstration ---------------------------------------

    def prove(self, *, ideas: int | None = None) -> tuple[OperationReport, ...]:
        """The whole loop in order: think, measure, learn, recall.

        This is the honest demonstration the window's **Prove the loop** button
        runs: it shows the mind thinking, the ten indicators measured, and the
        learning system acquiring a fact and recalling it. Nothing is asserted
        that the run did not do.
        """
        reports: list[OperationReport] = []
        reports.append(self.think(max_ideas=ideas))
        report = self.evaluate()
        reports.append(
            OperationReport(
                ok=True,
                title=f"Ten indicators — overall {report.overall:.3f}",
                lines=tuple(f"{LABELS[k]}: {report.scores[k]:.3f}" for k in INDICATORS),
            )
        )
        reports.append(
            self.learn(
                "acid stimulation increases permeability",
                source="corpus",
                confidence=0.9,
                verified=True,
            )
        )
        reports.append(OperationReport(ok=True, title=self.recall("permeability", limit=3).describe()))
        return tuple(reports)


__all__ = [
    "CognitiveService",
    "IdeaRow",
    "OperationReport",
    "ProgressionReport",
    "RecallReport",
    "StageRow",
]
