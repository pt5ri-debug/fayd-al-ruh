"""The Autonomous Cognitive System service: the Tk-free engine behind the window.

Every test runs headless with no display, no model, and no network. The service
wraps the proactive mind, the learning system, the staged trainer, the ten
indicators, and the character evaluation -- so this suite proves the *service*
layer over the real cognition modules rather than re-testing them.
"""

from __future__ import annotations

import pytest
from cognition.runner import ResourceMode
from cognition.training import TrainingStage
from gui.cognition import CognitiveService, OperationReport, ProgressionReport


@pytest.fixture(scope="module")
def controller() -> object:
    """One shared controller for the module, so the ontology is built once."""
    from core.cognitive_controller import CognitiveController

    return CognitiveController()


@pytest.fixture(scope="module")
def evaluator(controller: object) -> object:
    """One shared evaluator, so the retriever and document store build once."""
    from cognition.evaluation import CognitiveEvaluator

    return CognitiveEvaluator(controller.ontology)  # type: ignore[attr-defined]


def _service(controller: object, evaluator: object | None = None) -> CognitiveService:
    return CognitiveService(data_root=None, controller=controller, evaluator=evaluator)


def test_status_is_a_plain_snapshot(controller: object) -> None:
    status = _service(controller).status()
    assert "cycles" in status
    assert "ideas" in status
    assert isinstance(status["enabled"], bool)


def test_think_runs_a_cycle_and_reports_it(controller: object) -> None:
    report = _service(controller).think()
    assert isinstance(report, OperationReport)
    assert report.ok
    assert "cycle" in report.title.lower()


def test_ideas_are_readable_rows(controller: object) -> None:
    service = _service(controller)
    service.think()
    rows = service.ideas()
    for row in rows:
        assert row.text
        assert row.kind
        assert 0.0 <= row.confidence <= 1.0
        assert row.describe()


def test_drives_describe_is_a_tuple_of_strings(controller: object) -> None:
    service = _service(controller)
    service.think()
    for line in service.drives():
        assert isinstance(line, str)


def test_learn_stores_a_fact_and_marks_trust(controller: object) -> None:
    service = _service(controller)
    unverified = service.learn("water is wet", verified=False)
    assert unverified.ok
    assert any("not a fact" in line for line in unverified.lines)
    verified = service.learn("porosity measures pore space", verified=True)
    assert any("trusted fact" in line for line in verified.lines)


def test_recall_reports_every_memory(controller: object) -> None:
    service = _service(controller)
    service.learn("acid stimulation increases permeability", verified=True)
    report = service.recall("permeability")
    assert report.query == "permeability"
    assert report.found
    assert report.describe()


def test_learning_stats_is_one_line(controller: object) -> None:
    assert isinstance(_service(controller).learning_stats(), str)


def test_experience_records_a_lesson(controller: object) -> None:
    report = _service(controller).experience(
        expectation="the well flows", outcome="the well flowed", success=True
    )
    assert report.ok
    assert any("lesson" in line for line in report.lines)


def test_stages_are_the_four_known_stages(controller: object) -> None:
    stages = tuple(s.value for s in _service(controller).stages())
    assert stages == ("smoke", "quick", "standard", "full")


def test_stage_settings_describe(controller: object) -> None:
    assert _service(controller).stage_settings(TrainingStage.SMOKE)


def test_train_is_refused_without_approval(controller: object) -> None:
    report = _service(controller).train(stage=TrainingStage.SMOKE, approved=False)
    assert report.ok is False
    assert "governed" in report.title.lower()
    assert report.detail


def test_gradual_is_refused_without_approval(controller: object) -> None:
    report = _service(controller).run_gradual(up_to=TrainingStage.SMOKE, approved=False)
    assert report.ok is False
    assert "governed" in report.title.lower()


def test_progression_is_readable_before_any_training(controller: object) -> None:
    progression = _service(controller).progression()
    assert isinstance(progression, ProgressionReport)
    assert progression.next_stage == "smoke"
    assert progression.describe()


def test_versions_report_is_a_string(controller: object) -> None:
    assert isinstance(_service(controller).versions_report(), str)


def test_rollback_with_no_versions_reports_the_base_model(controller: object) -> None:
    report = _service(controller).rollback()
    assert report.ok
    assert "base model" in report.title.lower()


def test_evaluate_runs_the_ten_indicators(controller: object, evaluator: object) -> None:
    report = _service(controller, evaluator).evaluate(questions=("what is porosity?",))
    assert len(report.scores) == 10
    assert 0.0 <= report.overall <= 1.0


def test_indicators_are_pairs(controller: object) -> None:
    pairs = _service(controller).indicators()
    assert len(pairs) == 10
    for key, label in pairs:
        assert key and label


def test_teach_is_honest_when_the_student_is_missing(controller: object) -> None:
    # With no trained checkpoint on this machine the student is unavailable and
    # the service says so rather than pretending.
    report = _service(controller).teach("what is porosity?")
    assert isinstance(report, OperationReport)
    # Either it ran (a checkpoint exists) or it refused with the next step.
    if not report.ok:
        assert "student" in report.title.lower()
        assert report.detail


def test_mode_round_trips(controller: object) -> None:
    service = _service(controller)
    report = service.set_mode(ResourceMode.LIGHT)
    assert report.ok
    assert service.mode() is ResourceMode.LIGHT


def test_tick_reports_either_a_cycle_or_a_reason(controller: object) -> None:
    report = _service(controller).tick()
    assert isinstance(report, OperationReport)
    assert report.title


def test_evaluate_characters_returns_the_eight(controller: object) -> None:
    report = _service(controller).evaluate_characters()
    assert len(report.characters) == 8


def test_prove_runs_the_whole_loop(controller: object, evaluator: object) -> None:
    reports = _service(controller, evaluator).prove()
    assert len(reports) == 4
    assert all(isinstance(r, OperationReport) for r in reports)
    # The demonstration names the ten indicators and learns a fact.
    assert any("indicators" in r.title.lower() for r in reports)
    assert any("Learned" in r.title for r in reports)


def test_operation_report_describe_is_readable() -> None:
    report = OperationReport(ok=True, title="Title", lines=("one", "two"), detail="detail")
    text = report.describe()
    assert "Title" in text
    assert "one" in text
    assert "detail" in text


def test_service_shares_one_controller_for_mind_and_learning(controller: object) -> None:
    service = _service(controller)
    # The mind and the learning system are built over the same controller.
    assert service.mind.ontology is service.controller.ontology
    assert service.learning is service.controller.cognitive_learning()
