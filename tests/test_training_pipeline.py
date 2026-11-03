"""Phase 6: the trainer, the registry, promotion, and rollback.

These tests run a **real** (tiny) LoRA job over a small synthetic dataset and
exercise the whole life cycle: the run record, the base-preservation proof, the
run-id refusal, cancellation, the evaluation, the promotion gate, and the rollback
that keeps every run's history.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from ml.llm.tokenizer import ByteTokenizer
from ml.llm.transformer import TransformerConfig, TransformerLM
from ml.training.adapters import AdapterError, NumpyLoRAAdapter
from ml.training.config import TrainingConfig
from ml.training.coordinator import TrainingCoordinator
from ml.training.datasets import TrainingExample
from ml.training.evaluation import (
    METRIC_ACCURACY,
    METRIC_LOSS,
    EvaluationReport,
    build_report,
    compare,
    higher_is_better,
)
from ml.training.registry import (
    STATUS_ACCEPTED,
    STATUS_CANDIDATE,
    STATUS_REJECTED,
    STATUS_SUPERSEDED,
    AdapterRecord,
    PromotionManager,
    RollbackManager,
    TrainingRegistry,
    register_adapter_as_model,
)
from ml.training.trainer import (
    RUN_CANCELLED,
    RUN_COMPLETED,
    RUN_FAILED,
    AdapterTrainer,
    TrainingError,
    TrainingJob,
    TrainingRun,
)


def _examples(count: int = 24) -> tuple[TrainingExample, ...]:
    return tuple(
        TrainingExample(
            text=f"Q: what is porosity number {i}?\nA: porosity is the void fraction of the rock {i}.",
            owner="jarvis",
            source="test",
            license="project-authored",
        )
        for i in range(count)
    )


def _model() -> TransformerLM:
    return TransformerLM(TransformerConfig.preset("small"), seed=5)


def _config(**overrides: object) -> TrainingConfig:
    values: dict[str, object] = {
        "base_model_id": "fayd-inprocess-model",
        "dataset_id": "demo",
        "rank": 2,
        "alpha": 4,
        "batch_size": 2,
        "grad_accum": 1,
        "max_steps": 6,
        "learning_rate": 1e-2,
    }
    values.update(overrides)
    return TrainingConfig(**values)  # type: ignore[arg-type]


def _ineligible(coordinator: TrainingCoordinator):
    """A built dataset whose record is (deliberately) not eligible, for the tests."""
    import dataclasses

    built = coordinator.prepare_dataset("demo", _examples(), personality_scope=("jarvis",))
    record = dataclasses.replace(built.record, eligible=False, status="rejected")
    return dataclasses.replace(built, record=record)


@pytest.fixture()
def coordinator(tmp_path: Path) -> TrainingCoordinator:
    return TrainingCoordinator(root=tmp_path / "training")


# ------------------------------------------------------------- evaluation


def test_higher_is_better_only_for_loss():
    assert not higher_is_better(METRIC_LOSS)
    assert higher_is_better(METRIC_ACCURACY)


def test_compare_accepts_an_improvement():
    comparisons, accepted, verdict = compare(
        {METRIC_LOSS: 5.0}, {METRIC_LOSS: 4.0}
    )
    assert accepted and "improves" in verdict
    assert comparisons[0].delta == pytest.approx(1.0)


def test_compare_refuses_a_regression():
    _, accepted, verdict = compare({METRIC_LOSS: 4.0}, {METRIC_LOSS: 5.0})
    assert not accepted and "regresses" in verdict


def test_compare_refuses_a_flat_candidate():
    _, accepted, verdict = compare({METRIC_LOSS: 4.0}, {METRIC_LOSS: 4.0})
    assert not accepted and "improvement" in verdict


def test_compare_handles_no_shared_metric():
    _, accepted, verdict = compare({}, {METRIC_LOSS: 4.0})
    assert not accepted and "shared" in verdict


def test_report_round_trips():
    report = build_report(baseline={METRIC_LOSS: 5.0}, candidate={METRIC_LOSS: 4.0}, held_out_examples=3)
    assert EvaluationReport.from_primitive(report.to_primitive()).accepted == report.accepted
    assert "verdict" in report.describe()


# ------------------------------------------------------------- the trainer


def test_trainer_records_a_completed_run(coordinator, tmp_path):
    dataset = coordinator.prepare_dataset("demo", _examples(), personality_scope=("jarvis",))
    model = _model()
    result = coordinator.trainer.train(
        model, dataset, _config(), tokenizer=ByteTokenizer(), model_version="v1"
    )
    run = result.run
    assert run.status == RUN_COMPLETED
    assert run.model_version == "v1"
    assert run.software.get("python")
    assert run.hardware.get("cpu_count")
    assert run.train_metrics["final_loss"] > 0
    assert result.adapter_dir is not None and result.adapter_dir.is_dir()
    assert (tmp_path / "training" / "runs" / run.run_id / "run.json").is_file()
    assert result.ok


def test_trainer_refuses_a_duplicate_run_id(coordinator):
    dataset = coordinator.prepare_dataset("demo", _examples(), personality_scope=("jarvis",))
    model = _model()
    config = _config(run_id="fixed-run-id")
    coordinator.trainer.train(model, dataset, config, tokenizer=ByteTokenizer())
    with pytest.raises(TrainingError):
        coordinator.trainer.train(model, dataset, config, tokenizer=ByteTokenizer())


def test_trainer_refuses_an_ineligible_dataset(coordinator):
    dataset = _ineligible(coordinator)
    with pytest.raises(TrainingError):
        coordinator.trainer.train(_model(), dataset, _config(), tokenizer=ByteTokenizer())


def test_trainer_records_a_cancelled_run(coordinator):
    dataset = coordinator.prepare_dataset("demo", _examples(), personality_scope=("jarvis",))
    job = TrainingJob(run_id="x")
    job.cancel()
    result = coordinator.trainer.train(
        _model(), dataset, _config(max_steps=50), tokenizer=ByteTokenizer(), job=job
    )
    assert result.run.status == RUN_CANCELLED
    assert result.run.cancelled
    assert result.adapter_dir is None  # a cancelled run writes no adapter


def test_trainer_records_a_failed_run_and_leaves_no_adapter(coordinator):
    dataset = coordinator.prepare_dataset("demo", _examples(), personality_scope=("jarvis",))

    def _boom(**kwargs):
        raise AdapterError("the backend refused to build")

    trainer = AdapterTrainer(
        runs_dir=coordinator.root / "runs2",
        adapters_dir=coordinator.root / "adapters2",
        backend_factory=_boom,
    )
    result = trainer.train(_model(), dataset, _config(), tokenizer=ByteTokenizer())
    assert result.run.status == RUN_FAILED
    assert "refused" in result.run.error
    assert result.adapter_dir is None


def test_trainer_refuses_a_changed_base(coordinator, monkeypatch):
    dataset = coordinator.prepare_dataset("demo", _examples(), personality_scope=("jarvis",))

    class _SabotagingAdapter(NumpyLoRAAdapter):
        def train(self, examples, *, config, tokenizer, on_step=None, should_cancel=None):
            # Tamper with the base, then delegate: the trainer must notice.
            for matrix in self._model.parameters():
                matrix[...] = matrix + 0.5
            return super().train(
                examples, config=config, tokenizer=tokenizer, on_step=on_step, should_cancel=should_cancel
            )

    trainer = AdapterTrainer(
        runs_dir=coordinator.root / "runs3",
        adapters_dir=coordinator.root / "adapters3",
        backend_factory=lambda **kwargs: _SabotagingAdapter(
            rank=kwargs.get("rank", 2),
            alpha=kwargs.get("alpha", 4),
            dropout=kwargs.get("dropout", 0.0),
            seed=kwargs.get("seed", 0),
        ),
    )
    result = trainer.train(_model(), dataset, _config(), tokenizer=ByteTokenizer())
    assert result.run.status == RUN_FAILED
    assert "base model changed" in result.run.error


def test_trainer_lists_and_reads_runs(coordinator):
    dataset = coordinator.prepare_dataset("demo", _examples(), personality_scope=("jarvis",))
    result = coordinator.trainer.train(_model(), dataset, _config(), tokenizer=ByteTokenizer())
    runs = coordinator.trainer.runs()
    assert any(r.run_id == result.run.run_id for r in runs)
    assert coordinator.trainer.get(result.run.run_id) is not None
    assert coordinator.trainer.get("nonexistent") is None


def test_run_record_round_trips():
    run = TrainingRun(run_id="r1", model_id="m", train_metrics={"final_loss": 1.0})
    assert TrainingRun.from_primitive(run.to_primitive()) == run
    with pytest.raises(ValueError):
        TrainingRun.from_primitive({})


def test_cleanup_removes_a_failed_run_only(coordinator):
    dataset = coordinator.prepare_dataset("demo", _examples(), personality_scope=("jarvis",))
    good = coordinator.trainer.train(_model(), dataset, _config(), tokenizer=ByteTokenizer())
    with pytest.raises(TrainingError):
        coordinator.trainer.cleanup(good.run.run_id)


# ------------------------------------------------------------- the registry


def test_registry_records_and_reads_adapters(tmp_path):
    registry = TrainingRegistry(tmp_path / "reg")
    record = AdapterRecord(adapter_id="a1", run_id="a1", base_model_id="m", status=STATUS_CANDIDATE)
    registry.add(record)
    assert registry.get("a1") == record
    with pytest.raises(ValueError):
        registry.add(record)


def test_registry_updates_status_keeping_history(tmp_path):
    registry = TrainingRegistry(tmp_path / "reg")
    registry.add(AdapterRecord(adapter_id="a1", run_id="a1", base_model_id="m"))
    registry.update_status("a1", STATUS_ACCEPTED, evaluation={"accepted": True})
    assert registry.get("a1").status == STATUS_ACCEPTED
    with pytest.raises(ValueError):
        registry.update_status("a1", "nonsense")
    with pytest.raises(KeyError):
        registry.update_status("missing", STATUS_ACCEPTED)


def test_registry_active_pointer(tmp_path):
    registry = TrainingRegistry(tmp_path / "reg")
    registry.add(AdapterRecord(adapter_id="a1", run_id="a1", base_model_id="m"))
    assert registry.active() is None
    registry.set_active("a1")
    assert registry.active_id() == "a1"
    registry.clear_active()
    assert registry.active_id() == ""
    with pytest.raises(KeyError):
        registry.set_active("missing")


# ------------------------------------------------------------- promotion


def test_promotion_requires_an_accepted_evaluation_and_approval(tmp_path):
    registry = TrainingRegistry(tmp_path / "reg")
    registry.add(AdapterRecord(adapter_id="a1", run_id="a1", base_model_id="m"))
    manager = PromotionManager(registry)

    rejected = build_report(baseline={METRIC_LOSS: 4.0}, candidate={METRIC_LOSS: 5.0}, held_out_examples=2)
    decision = manager.evaluate_candidate("a1", rejected)
    assert not decision.promoted
    assert registry.get("a1").status == STATUS_REJECTED

    accepted = build_report(baseline={METRIC_LOSS: 5.0}, candidate={METRIC_LOSS: 4.0}, held_out_examples=2)
    decision = manager.evaluate_candidate("a1", accepted, approved=False)
    assert not decision.promoted and decision.requires_approval
    assert registry.get("a1").status == STATUS_ACCEPTED

    decision = manager.evaluate_candidate("a1", accepted, approved=True)
    assert decision.promoted
    assert registry.active_id() == "a1"


def test_promotion_supersedes_the_previous_adapter(tmp_path):
    registry = TrainingRegistry(tmp_path / "reg")
    for adapter_id in ("a1", "a2"):
        registry.add(AdapterRecord(adapter_id=adapter_id, run_id=adapter_id, base_model_id="m"))
    manager = PromotionManager(registry)
    accepted = build_report(baseline={METRIC_LOSS: 5.0}, candidate={METRIC_LOSS: 4.0}, held_out_examples=2)
    manager.evaluate_candidate("a1", accepted, approved=True)
    manager.evaluate_candidate("a2", accepted, approved=True)
    assert registry.get("a1").status == STATUS_SUPERSEDED
    assert registry.get("a2").status == STATUS_ACCEPTED


def test_promotion_requires_an_existing_adapter(tmp_path):
    manager = PromotionManager(TrainingRegistry(tmp_path / "reg"))
    with pytest.raises(KeyError):
        manager.evaluate_candidate("missing", build_report(baseline={}, candidate={}, held_out_examples=0))


# ------------------------------------------------------------- rollback


def test_rollback_returns_to_the_previous_accepted_adapter(tmp_path):
    registry = TrainingRegistry(tmp_path / "reg")
    for adapter_id in ("a1", "a2"):
        registry.add(AdapterRecord(adapter_id=adapter_id, run_id=adapter_id, base_model_id="m"))
        registry.update_status(adapter_id, STATUS_ACCEPTED)
    registry.set_active("a2")
    previous = RollbackManager(registry).rollback()
    assert previous is not None and previous.adapter_id == "a1"
    assert registry.active_id() == "a1"
    # The history is intact: nothing was deleted.
    assert len(registry.adapters()) == 2


def test_rollback_to_base_clears_the_pointer(tmp_path):
    registry = TrainingRegistry(tmp_path / "reg")
    registry.add(AdapterRecord(adapter_id="a1", run_id="a1", base_model_id="m"))
    registry.update_status("a1", STATUS_ACCEPTED)
    registry.set_active("a1")
    RollbackManager(registry).to_base()
    assert registry.active_id() == ""
    assert len(registry.adapters()) == 1


def test_rollback_with_no_accepted_adapter_clears(tmp_path):
    registry = TrainingRegistry(tmp_path / "reg")
    registry.add(AdapterRecord(adapter_id="a1", run_id="a1", base_model_id="m"))
    registry.set_active("a1")
    assert RollbackManager(registry).rollback() is None
    assert registry.active_id() == ""


# ------------------------------------------------------------- the whole path


def test_coordinator_runs_evaluates_and_promotes(coordinator):
    dataset = coordinator.prepare_dataset("demo", _examples(), personality_scope=("jarvis",))
    model = _model()
    before = [p.copy() for p in model.parameters()]

    feasibility = coordinator.feasibility(model, dataset, _config())
    assert feasibility.ok

    result = coordinator.start(model, dataset, _config(), tokenizer=ByteTokenizer())
    assert result.ok
    # The base is provably unchanged.
    after = [p.copy() for p in model.parameters()]
    assert all(np.array_equal(b, a) for b, a in zip(before, after, strict=True))

    report = coordinator.evaluate(
        model, dataset, tokenizer=ByteTokenizer(), adapter_dir=result.adapter_dir
    )
    assert report.comparisons  # at least one metric was measured

    refused = coordinator.promote(result.run.run_id, report, approved=False)
    assert not refused.promoted
    if report.accepted:
        promoted = coordinator.promote(result.run.run_id, report, approved=True)
        assert promoted.promoted
        assert coordinator.registry.active_id() == result.run.run_id
        previous = coordinator.rollback()
        assert previous is None  # only one accepted adapter, so rollback goes to base


def test_coordinator_feasibility_blocks_an_ineligible_dataset(coordinator):
    dataset = _ineligible(coordinator)
    report = coordinator.feasibility(_model(), dataset, _config())
    assert not report.ok
    assert any("eligible" in b for b in report.blockers)


def test_coordinator_start_refuses_when_infeasible(coordinator):
    dataset = _ineligible(coordinator)
    with pytest.raises(TrainingError):
        coordinator.start(_model(), dataset, _config(), tokenizer=ByteTokenizer())


def test_coordinator_status_is_readable(coordinator):
    text = coordinator.status()
    assert "Phase 6" in text
    assert "isolation clean" in text


# ------------------------------------------------------------- model manager integration


def test_register_adapter_as_model_appears_in_the_registry(tmp_path):
    record = AdapterRecord(
        adapter_id="a1", run_id="a1", base_model_id="m", dataset_id="d", dataset_version=1,
        method="lora_inprocess", path=str(tmp_path / "adapter"), status=STATUS_ACCEPTED,
    )
    model_record = register_adapter_as_model(record, root=tmp_path / "registry")
    assert model_record.runtime == "fayd-inprocess"
    assert "adapter" in model_record.tags
    assert model_record.extra["base_model"] == "m"


def test_model_manager_exposes_training_adapters(tmp_path):
    from core.model_manager import ModelManager

    root = tmp_path / "data"
    registry = TrainingRegistry(root)
    registry.add(
        AdapterRecord(
            adapter_id="a1", run_id="a1", base_model_id="m", status=STATUS_ACCEPTED,
            path=str(tmp_path / "adapter"),
        )
    )
    manager = ModelManager(root)
    assert [a.adapter_id for a in manager.training_adapters()] == ["a1"]
    # Activation refuses a non-accepted adapter and accepts an accepted one.
    manager.activate_training_adapter("a1")
    assert manager.active_training_adapter().adapter_id == "a1"
    manager.deactivate_training_adapter()
    assert manager.active_training_adapter() is None


def test_model_manager_refuses_activating_a_candidate(tmp_path):
    from core.model_manager import ModelManager

    root = tmp_path / "data"
    TrainingRegistry(root).add(
        AdapterRecord(adapter_id="c1", run_id="c1", base_model_id="m", status=STATUS_CANDIDATE)
    )
    manager = ModelManager(root)
    with pytest.raises(PermissionError):
        manager.activate_training_adapter("c1")
    with pytest.raises(KeyError):
        manager.activate_training_adapter("missing")
