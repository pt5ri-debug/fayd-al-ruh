"""Staged training, versioning/rollback, teacher-student, the evaluator, the runner.

These tests exercise the *development* half of the autonomous system. Training is
run through an injected trainer (a stub that writes a tiny checkpoint), so the
whole accept/rollback gate is proven with no real training and no base model
required -- and the teacher-student and evaluator paths run against the real
ontology. Nothing reaches the network or loads a real model.
"""

from __future__ import annotations

import itertools

import numpy as np
import pytest
from cognition.evaluation import INDICATORS, CognitiveEvaluator, CognitiveReport, load_report
from cognition.records import TrainingExample
from cognition.runner import CognitiveRunner, ResourceMode, load_mode, save_mode
from cognition.storage import ExampleStore
from cognition.teacher_student import CallableBackend, TeacherStudentSystem
from cognition.training import (
    STAGE_SETTINGS,
    IncrementalTrainer,
    TrainingStage,
    stages,
)
from cognition.versions import (
    Evaluation,
    ModelRegistry,
    ModelVersion,
    base_path,
    candidate_path,
    models_root,
    version_path,
)
from core.ontology import Ontology
from core.petroleum_corpus import seed_petroleum_corpus


@pytest.fixture(scope="module")
def ontology() -> Ontology:
    graph = Ontology()
    seed_petroleum_corpus(graph)
    return graph


# -- stages -----------------------------------------------------------------
def test_stages_are_ordered_smallest_first():
    ordered = stages()
    assert ordered[0] is TrainingStage.SMOKE
    steps = [STAGE_SETTINGS[s].steps for s in ordered]
    assert steps == sorted(steps)


def test_each_stage_is_larger_than_the_last():
    ordered = stages()
    for smaller, larger in itertools.pairwise(ordered):
        assert STAGE_SETTINGS[larger].steps > STAGE_SETTINGS[smaller].steps


# -- versioning -------------------------------------------------------------
def test_evaluation_delta_respects_lower_is_better():
    ppl = Evaluation(metric="test_ppl", before=10.0, after=5.0, lower_is_better=True)
    assert ppl.delta > 0  # a fall in perplexity is an improvement
    acc = Evaluation(metric="accuracy", before=0.5, after=0.7)
    assert acc.delta > 0


def test_registry_accepts_only_a_real_improvement(tmp_path, monkeypatch):
    monkeypatch.setenv("FAYD_DATA_DIR", str(tmp_path))
    registry = ModelRegistry()
    accepted, reason = registry.should_accept(
        (Evaluation(metric="test_ppl", before=10.0, after=8.0, lower_is_better=True),)
    )
    assert accepted and "improves" in reason


def test_registry_refuses_a_regression(tmp_path, monkeypatch):
    monkeypatch.setenv("FAYD_DATA_DIR", str(tmp_path))
    registry = ModelRegistry()
    accepted, reason = registry.should_accept(
        (Evaluation(metric="test_ppl", before=5.0, after=9.0, lower_is_better=True),)
    )
    assert not accepted and "regresses" in reason


def test_registry_refuses_a_run_with_no_evaluation(tmp_path, monkeypatch):
    monkeypatch.setenv("FAYD_DATA_DIR", str(tmp_path))
    accepted, reason = ModelRegistry().should_accept(())
    assert not accepted and reason == "no evaluation was run"


def test_registry_records_and_rolls_back(tmp_path, monkeypatch):
    monkeypatch.setenv("FAYD_DATA_DIR", str(tmp_path))
    registry = ModelRegistry()
    registry.record(
        ModelVersion(number=1, path=str(version_path(1)), parent=0, dataset="d", examples=0, steps=1, accepted=True)
    )
    registry.set_active(1)
    assert registry.active_number() == 1
    registry.record(
        ModelVersion(number=2, path=str(candidate_path()), parent=1, dataset="d", examples=0, steps=1)
    )
    stable = registry.rollback()
    assert stable is not None and stable.number == 1
    assert registry.active_number() == 1


def test_ensure_base_copies_once_and_never_overwrites(tmp_path, monkeypatch):
    monkeypatch.setenv("FAYD_DATA_DIR", str(tmp_path))
    source = tmp_path / "shipped.npz"
    source.write_bytes(b"shipped-weights")
    registry = ModelRegistry()
    first = registry.ensure_base(source)
    assert first.read_bytes() == b"shipped-weights"
    source.write_bytes(b"changed")
    again = registry.ensure_base(source)
    assert again.read_bytes() == b"shipped-weights"


def test_version_round_trips_through_primitive():
    version = ModelVersion(
        number=3,
        path="/x/v3.npz",
        parent=2,
        dataset="base",
        examples=1,
        steps=10,
        evaluations=(Evaluation("test_ppl", 9.0, 7.0, lower_is_better=True),),
        accepted=True,
    )
    restored = ModelVersion.from_primitive(version.to_primitive())
    assert restored.number == 3
    assert restored.evaluations[0].lower_is_better is True


# -- the trainer (with an injected trainer) ---------------------------------
class _StubTrainer:
    """Write a tiny but *real* npz so the gate has a candidate to inspect."""

    def __init__(self, improve: bool = True) -> None:
        self.improve = improve
        self.calls = 0

    def __call__(self, *, checkpoint, tokenizer_path, **kwargs):
        self.calls += 1
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        np.savez(checkpoint, weights=np.zeros(4, dtype=np.float32))


def test_trainer_builds_a_plan_from_the_base(tmp_path, monkeypatch):
    monkeypatch.setenv("FAYD_DATA_DIR", str(tmp_path))
    trainer = IncrementalTrainer(
        examples=ExampleStore(),
        corpus_builder=lambda **_: "small corpus",
        units_builder=lambda: [],
    )
    plan = trainer.plan(stage=TrainingStage.SMOKE)
    assert plan.stage is TrainingStage.SMOKE
    assert plan.dataset_chars > 0


def test_trainer_refuses_to_run_without_a_base(tmp_path, monkeypatch):
    monkeypatch.setenv("FAYD_DATA_DIR", str(tmp_path))
    monkeypatch.setattr("cognition.training.default_checkpoint", lambda: tmp_path / "absent.npz")
    trainer = IncrementalTrainer(
        examples=ExampleStore(),
        corpus_builder=lambda **_: "small corpus",
        units_builder=lambda: [],
        trainer=_StubTrainer(),
    )
    outcome = trainer.run(stage=TrainingStage.SMOKE)
    assert not outcome.accepted
    assert "no base model" in outcome.reason


def test_qualified_examples_enter_the_dataset(tmp_path, monkeypatch):
    monkeypatch.setenv("FAYD_DATA_DIR", str(tmp_path))
    examples = ExampleStore()
    examples.add(
        TrainingExample(task="what is porosity", answer="a measure of void space", qualified=True)
    )
    examples.add(TrainingExample(task="junk", answer="junk", qualified=False))
    trainer = IncrementalTrainer(
        examples=examples,
        corpus_builder=lambda **_: "base corpus",
        units_builder=lambda: [],
    )
    dataset = trainer.build_dataset(stage=TrainingStage.SMOKE)
    assert "what is porosity" in dataset
    assert "junk" not in dataset


# -- gradual training (التدريب التدريجي) ------------------------------------
def test_progression_starts_with_every_stage_undone(tmp_path, monkeypatch):
    monkeypatch.setenv("FAYD_DATA_DIR", str(tmp_path))
    trainer = IncrementalTrainer(examples=ExampleStore())
    progress = trainer.progression()
    assert [p.stage for p in progress] == list(stages())
    assert not any(p.done for p in progress)
    assert trainer.next_stage() is TrainingStage.SMOKE


def test_progression_reads_an_accepted_run_from_the_registry(tmp_path, monkeypatch):
    monkeypatch.setenv("FAYD_DATA_DIR", str(tmp_path))
    registry = ModelRegistry()
    registry.record(
        ModelVersion(
            number=1, path=str(version_path(1)), parent=0, dataset="d", examples=0,
            steps=40, settings={"stage": "smoke"}, accepted=True,
        )
    )
    trainer = IncrementalTrainer(registry=registry, examples=ExampleStore())
    progress = {p.stage: p for p in trainer.progression()}
    assert progress[TrainingStage.SMOKE].done
    assert trainer.next_stage() is TrainingStage.QUICK


def test_a_rejected_run_is_an_attempt_not_a_success(tmp_path, monkeypatch):
    monkeypatch.setenv("FAYD_DATA_DIR", str(tmp_path))
    registry = ModelRegistry()
    registry.record(
        ModelVersion(
            number=1, path=str(candidate_path()), parent=0, dataset="d", examples=0,
            steps=40, settings={"stage": "smoke"}, accepted=False, reason="no measurable improvement",
        )
    )
    trainer = IncrementalTrainer(registry=registry, examples=ExampleStore())
    progress = {p.stage: p for p in trainer.progression()}
    assert progress[TrainingStage.SMOKE].attempts == 1
    assert not progress[TrainingStage.SMOKE].done
    assert trainer.next_stage() is TrainingStage.SMOKE


def test_gradual_training_stops_on_a_rejected_stage(tmp_path, monkeypatch):
    monkeypatch.setenv("FAYD_DATA_DIR", str(tmp_path))
    trainer = IncrementalTrainer(
        examples=ExampleStore(),
        corpus_builder=lambda **_: "base corpus",
        units_builder=lambda: [],
        trainer=_StubTrainer(),
    )
    report = trainer.run_gradual(up_to=TrainingStage.STANDARD)
    # With no held-out units the run cannot be evaluated, so the first stage is
    # rejected and the programme stops rather than piling a bigger stage on top.
    assert report.stopped_at is TrainingStage.SMOKE
    assert report.advanced() == ()
    assert report.reason


def test_gradual_training_skips_a_finished_stage(tmp_path, monkeypatch):
    monkeypatch.setenv("FAYD_DATA_DIR", str(tmp_path))
    registry = ModelRegistry()
    registry.record(
        ModelVersion(
            number=1, path=str(version_path(1)), parent=0, dataset="d", examples=0,
            steps=40, settings={"stage": "smoke"}, accepted=True,
        )
    )
    trainer = IncrementalTrainer(registry=registry, examples=ExampleStore())
    # ``smoke`` is done, so nothing runs; the report advances nothing.
    report = trainer.run_gradual(up_to=TrainingStage.SMOKE)
    assert report.outcomes == ()
    assert report.stopped_at is None


def test_gradual_report_describes_every_stage(tmp_path, monkeypatch):
    monkeypatch.setenv("FAYD_DATA_DIR", str(tmp_path))
    report = IncrementalTrainer(examples=ExampleStore()).run_gradual(up_to=TrainingStage.SMOKE)
    text = report.describe()
    for stage in stages():
        assert stage.value in text


# -- teacher / student ------------------------------------------------------
def test_a_capable_student_is_not_sent_to_the_teacher(ontology):
    student = CallableBackend("student", lambda q: "porosity measures the void space in a rock")
    teacher = CallableBackend("teacher", lambda q: "never called")
    system = TeacherStudentSystem(student, teacher, ontology=ontology)
    outcome = system.teach("what is porosity")
    assert outcome.teacher_used is False
    assert "student was adequate" in " ".join(outcome.notes)


def test_a_weak_student_is_sent_to_the_teacher(ontology):
    student = CallableBackend("student", lambda q: "no")
    teacher = CallableBackend("teacher", lambda q: "porosity measures the void space in a rock")
    system = TeacherStudentSystem(student, teacher, ontology=ontology)
    outcome = system.teach("what is porosity")
    assert outcome.teacher_used is True
    assert outcome.teacher_answer


def test_a_teacher_answer_naming_no_concept_is_not_qualified(ontology):
    student = CallableBackend("student", lambda q: "no")
    teacher = CallableBackend("teacher", lambda q: "something entirely unrelated to the graph")
    system = TeacherStudentSystem(student, teacher, ontology=ontology)
    outcome = system.teach("what is porosity")
    assert outcome.example is not None
    assert outcome.example.qualified is False


def test_verify_requires_a_known_concept(ontology):
    system = TeacherStudentSystem(CallableBackend("s", lambda q: ""), None, ontology=ontology)
    verified, _confidence, reason = system.verify("q", "porosity and permeability")
    assert verified
    assert "porosity" in reason
    not_verified, _confidence, _reason = system.verify("q", "zzz qqq")
    assert not not_verified


# -- the evaluator ----------------------------------------------------------
def test_evaluator_measures_every_indicator(ontology, tmp_path, monkeypatch):
    monkeypatch.setenv("FAYD_DATA_DIR", str(tmp_path))
    report = CognitiveEvaluator(ontology).evaluate()
    assert set(report.scores) == set(INDICATORS)
    assert 0.0 <= report.overall <= 1.0
    assert report.describe()


def test_evaluator_is_deterministic(ontology, tmp_path, monkeypatch):
    monkeypatch.setenv("FAYD_DATA_DIR", str(tmp_path))
    first = CognitiveEvaluator(ontology).evaluate()
    second = CognitiveEvaluator(ontology).evaluate()
    for key in INDICATORS:
        if key == "response_speed":  # timing varies by machine load
            continue
        assert first.scores[key] == second.scores[key]


def test_report_round_trips_through_primitive(ontology, tmp_path, monkeypatch):
    monkeypatch.setenv("FAYD_DATA_DIR", str(tmp_path))
    report = CognitiveEvaluator(ontology).evaluate()
    restored = load_report(report.to_primitive())
    assert isinstance(restored, CognitiveReport)
    assert restored.scores == report.scores


def test_comparison_reports_a_change_per_indicator(ontology, tmp_path, monkeypatch):
    monkeypatch.setenv("FAYD_DATA_DIR", str(tmp_path))
    evaluator = CognitiveEvaluator(ontology)
    before = evaluator.evaluate()
    after = evaluator.evaluate()
    comparison = evaluator.compare(before, after)
    assert set(comparison.deltas()) == set(INDICATORS)


# -- the runner -------------------------------------------------------------
def test_runner_persists_the_resource_mode(ontology, tmp_path, monkeypatch):
    monkeypatch.setenv("FAYD_DATA_DIR", str(tmp_path))
    from cognition.proactive_mind import ProactiveMind

    runner = CognitiveRunner(ProactiveMind(ontology))
    runner.set_mode(ResourceMode.LIGHT)
    assert load_mode() is ResourceMode.LIGHT


def test_runner_tick_runs_a_cycle_when_due(ontology, tmp_path, monkeypatch):
    monkeypatch.setenv("FAYD_DATA_DIR", str(tmp_path))
    from cognition.proactive_mind import ProactiveMind
    from cognition.runner import MachineLoad

    runner = CognitiveRunner(
        ProactiveMind(ontology),
        mode=ResourceMode.LIGHT,
        load_reader=lambda: MachineLoad(cpu_load=0.1, free_memory_fraction=0.9),
    )
    cycle = runner.tick()
    assert cycle is not None
    assert runner.decision() is not None


def test_runner_defers_under_high_load(ontology, tmp_path, monkeypatch):
    monkeypatch.setenv("FAYD_DATA_DIR", str(tmp_path))
    from cognition.proactive_mind import ProactiveMind
    from cognition.runner import MachineLoad

    runner = CognitiveRunner(
        ProactiveMind(ontology),
        mode=ResourceMode.LIGHT,
        load_reader=lambda: MachineLoad(cpu_load=0.99, free_memory_fraction=0.9),
    )
    assert runner.tick() is None
    assert not runner.decision().allowed


def test_runner_stop_and_start(ontology, tmp_path, monkeypatch):
    monkeypatch.setenv("FAYD_DATA_DIR", str(tmp_path))
    from cognition.proactive_mind import ProactiveMind
    from cognition.runner import MachineLoad

    runner = CognitiveRunner(
        ProactiveMind(ontology),
        load_reader=lambda: MachineLoad(cpu_load=0.1, free_memory_fraction=0.9),
    )
    runner.stop()
    assert runner.tick() is None
    runner.start()
    assert runner.tick() is not None


def test_runner_status_is_honest(ontology, tmp_path, monkeypatch):
    monkeypatch.setenv("FAYD_DATA_DIR", str(tmp_path))
    from cognition.proactive_mind import ProactiveMind

    status = CognitiveRunner(ProactiveMind(ontology), mode=ResourceMode.BALANCED).status()
    assert status["mode"] == "balanced"
    assert "load" in status and "next_due" in status


def test_models_root_is_under_the_data_dir(tmp_path, monkeypatch):
    monkeypatch.setenv("FAYD_DATA_DIR", str(tmp_path))
    assert str(models_root()).startswith(str(tmp_path))
    assert base_path().name == "base.npz"


def test_save_mode_round_trips(tmp_path, monkeypatch):
    monkeypatch.setenv("FAYD_DATA_DIR", str(tmp_path))
    save_mode(ResourceMode.INTENSIVE)
    assert load_mode() is ResourceMode.INTENSIVE
