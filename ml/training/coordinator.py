"""The training coordinator: the one governed entry point for Phase 6.

Every step the phase requires is a method here, and each one is **governed** --
it checks its preconditions and reports honestly rather than assuming:

* :meth:`TrainingCoordinator.feasibility` -- the pre-training feasibility report:
  the base model's compatibility, the dataset's eligibility, and the machine's
  resources, all measured rather than asserted.
* :meth:`TrainingCoordinator.prepare_dataset` -- build a governed, validated,
  versioned dataset from explicit examples.
* :meth:`TrainingCoordinator.start` -- validate everything, then run one adapter
  training job and record it.
* :meth:`TrainingCoordinator.evaluate` -- compare the base and the candidate on
  held-out data.
* :meth:`TrainingCoordinator.promote` / :meth:`rollback` -- the promotion and
  rollback gates.

The coordinator **never** promotes a candidate on its own and **never** writes a
base model. It also never trains anything by itself in a test: the model and the
adapter backend are injected, so the whole pipeline is exercised with tiny
fixtures.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ml.training.compatibility import CompatibilityVerdict, validate_compatibility
from ml.training.config import TrainingConfig
from ml.training.datasets import (
    BuiltDataset,
    DatasetBuilder,
    DatasetValidationError,
    TrainingExample,
    aggregate_examples,
)
from ml.training.evaluation import EvaluationReport, build_report, evaluate
from ml.training.isolation import IsolationVerdict, audit_dataset, main_model_dataset_is_clean
from ml.training.registry import (
    AdapterRecord,
    PromotionDecision,
    PromotionManager,
    RollbackManager,
    TrainingRegistry,
)
from ml.training.resources import (
    ResourceCheck,
    check_disk,
    check_memory,
    estimate_training_memory,
)
from ml.training.trainer import (
    AdapterTrainer,
    TrainingError,
    TrainingJob,
    TrainingResult,
)


@dataclass(frozen=True)
class FeasibilityReport:
    """The pre-training feasibility report the task requires before any run."""

    base_model_id: str
    compatibility: CompatibilityVerdict | None = None
    dataset: str = ""
    dataset_eligible: bool = False
    dataset_reason: str = ""
    isolation: IsolationVerdict | None = None
    disk: ResourceCheck | None = None
    memory: ResourceCheck | None = None
    estimated_memory_bytes: int = 0
    trainable: bool = False
    blockers: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()
    unknowns: tuple[str, ...] = ()

    @property
    def ok(self) -> bool:
        return self.trainable and not self.blockers

    def describe(self) -> str:
        lines = [
            f"Training feasibility for {self.base_model_id}",
            f"  trainable: {'yes' if self.trainable else 'no'}",
            f"  dataset: {self.dataset or '(none)'} -- "
            + ("eligible" if self.dataset_eligible else f"not eligible ({self.dataset_reason})"),
        ]
        if self.compatibility is not None:
            lines.append("  compatibility:")
            lines.extend(f"    {item.describe()}" for item in self.compatibility.methods)
        if self.isolation is not None:
            lines.append(f"  isolation: {self.isolation.describe()}")
        if self.disk is not None:
            lines.append(f"  {self.disk.describe()}")
        if self.memory is not None:
            lines.append(f"  {self.memory.describe()}")
        lines.extend(f"  blocker: {b}" for b in self.blockers)
        lines.extend(f"  warning: {w}" for w in self.warnings)
        lines.extend(f"  unknown: {u}" for u in self.unknowns)
        return "\n".join(lines)


class TrainingCoordinator:
    """Wire dataset governance, compatibility, training, evaluation, and promotion."""

    def __init__(
        self,
        *,
        root: Path | str | None = None,
        dataset_builder: DatasetBuilder | None = None,
        trainer: AdapterTrainer | None = None,
        registry: TrainingRegistry | None = None,
        model_factory: Callable[[Any], Any] | None = None,
        backend_factory: Callable[..., Any] | None = None,
    ) -> None:
        from core.paths import training_dir

        self.root = Path(root) if root is not None else training_dir()
        self.datasets = dataset_builder or DatasetBuilder(root=self.root / "datasets")
        trainer_kwargs: dict[str, Any] = {
            "runs_dir": self.root / "runs",
            "adapters_dir": self.root / "adapters",
        }
        if backend_factory is not None:
            trainer_kwargs["backend_factory"] = backend_factory
        if model_factory is not None:
            trainer_kwargs["model_factory"] = model_factory
        self.trainer = trainer or AdapterTrainer(**trainer_kwargs)
        self.registry = registry or TrainingRegistry(self.root / "registry")
        self.promotion = PromotionManager(self.registry)
        self.rollback_manager = RollbackManager(self.registry)

    # -- feasibility -------------------------------------------------------
    def feasibility(
        self,
        model: Any,
        dataset: BuiltDataset | None,
        config: TrainingConfig,
        *,
        method: str = "",
    ) -> FeasibilityReport:
        """Assess whether a run can start, without starting it."""
        compatibility = validate_compatibility(model, method=method or config.method)
        blockers: list[str] = []
        warnings: list[str] = list(compatibility.warnings)
        unknowns: list[str] = list(compatibility.unknowns)

        if not compatibility.trainable:
            blockers.append(
                f"no training method is supported for {compatibility.model_id} "
                f"({compatibility.model_format})"
            )
        elif method and not compatibility.support_for(method):
            blockers.append(f"the method {method!r} is unknown")

        dataset_name = ""
        dataset_eligible = False
        dataset_reason = "no dataset was supplied"
        isolation: IsolationVerdict | None = None
        if dataset is not None:
            dataset_name = dataset.record.key
            dataset_eligible = dataset.eligible
            dataset_reason = dataset.record.status
            for_main = config.owner in ("", "fayd_al_ruh", "model")
            isolation = audit_dataset(
                dataset.split,
                declared_scope=dataset.record.personality_scope,
                for_main_model=for_main,
            )
            if not dataset_eligible:
                blockers.append(f"the dataset {dataset_name} is not eligible for training")
            if not isolation.ok:
                blockers.append(f"the dataset fails isolation: {isolation.reason}")

        parameters = getattr(model, "parameter_count", None)
        if parameters is None and compatibility.parameter_count is not None:
            parameters = compatibility.parameter_count
        estimated = estimate_training_memory(
            parameter_count=parameters,
            rank=config.rank,
            batch_size=config.batch_size,
            grad_accum=config.grad_accum,
        )
        disk = check_disk(self.root, floor_bytes=config.disk_floor_bytes)
        memory = check_memory(estimated, limit_bytes=config.memory_limit_bytes)
        if not disk.ok:
            blockers.append(f"disk: {disk.detail}")
        if not memory.ok:
            blockers.append(f"memory: {memory.detail}")

        return FeasibilityReport(
            base_model_id=compatibility.model_id,
            compatibility=compatibility,
            dataset=dataset_name,
            dataset_eligible=dataset_eligible,
            dataset_reason=dataset_reason,
            isolation=isolation,
            disk=disk,
            memory=memory,
            estimated_memory_bytes=estimated,
            trainable=compatibility.trainable,
            blockers=tuple(blockers),
            warnings=tuple(dict.fromkeys(warnings)),
            unknowns=tuple(unknowns),
        )

    # -- datasets ----------------------------------------------------------
    def prepare_dataset(
        self,
        dataset_id: str,
        examples: Sequence[TrainingExample],
        **kwargs: Any,
    ) -> BuiltDataset:
        """Build a governed dataset version from explicit examples."""
        return self.datasets.build(dataset_id, examples, **kwargs)

    def prepare_aggregate_dataset(
        self, dataset_id: str, *, include_private: bool = False, **kwargs: Any
    ) -> BuiltDataset:
        """Build the project model's aggregated dataset from authorised records.

        The examples come from :func:`aggregate_examples`, which asks the Phase-5
        isolation policy -- so the private personality is excluded unless
        ``include_private`` is set, which the training path never sets. The dataset
        is then audited for the main model, so a private example would be refused
        even if one slipped in.
        """
        examples = aggregate_examples(root=self.root, include_private=include_private)
        if not examples:
            raise DatasetValidationError(
                "No authorised learning records were found to aggregate; characters "
                "must have learned something first (see `python -m cli characters`)."
            )
        scope = tuple(sorted({e.owner for e in examples if e.owner}))
        kwargs.setdefault("intended_model", "fayd_al_ruh")
        kwargs.setdefault("intended_task", "chat")
        built = self.datasets.build(dataset_id, examples, personality_scope=scope, **kwargs)
        verdict = audit_dataset(built.split, for_main_model=True)
        if not verdict.ok:
            raise DatasetValidationError(verdict.reason)
        return built

    # -- training ----------------------------------------------------------
    def start(
        self,
        model: Any,
        dataset: BuiltDataset,
        config: TrainingConfig,
        *,
        tokenizer: Any,
        on_step: Callable[[int, float], None] | None = None,
        job: TrainingJob | None = None,
        resume_from: Path | str | None = None,
        require_feasible: bool = True,
    ) -> TrainingResult:
        """Validate feasibility, then run one training job and record it."""
        report = self.feasibility(model, dataset, config)
        if require_feasible and not report.ok:
            raise TrainingError(
                "Training refused: " + "; ".join(report.blockers or ["not feasible"])
            )
        result = self.trainer.train(
            model,
            dataset,
            config,
            tokenizer=tokenizer,
            on_step=on_step,
            job=job,
            resume_from=resume_from,
        )
        if result.run.status == "completed":
            self.registry.add(
                AdapterRecord(
                    adapter_id=result.run.run_id,
                    run_id=result.run.run_id,
                    base_model_id=result.run.model_id,
                    base_model_version=result.run.model_version,
                    dataset_id=result.run.dataset_id,
                    dataset_version=result.run.dataset_version,
                    method=result.run.method,
                    owner=result.run.owner,
                    path=result.run.adapter_path,
                    created_at=result.run.ended_at,
                    compatibility={
                        "method": result.run.method,
                        "trainable_parameters": str(
                            result.run.train_metrics.get("trainable_parameters", 0)
                        ),
                    },
                )
            )
        return result

    # -- evaluation --------------------------------------------------------
    def evaluate(
        self,
        model: Any,
        dataset: BuiltDataset,
        *,
        tokenizer: Any,
        adapter_dir: Path | str | None = None,
        instruction_pairs: Sequence[tuple[str, str]] = (),
    ) -> EvaluationReport:
        """Compare the base model and the candidate on the dataset's held-out test split."""
        held_out = [example.text for example in dataset.split.test]
        if not held_out:
            held_out = [example.text for example in dataset.split.validation]
        if not held_out:
            return build_report(
                baseline={},
                candidate={},
                held_out_examples=0,
                notes=("the dataset has no held-out split to evaluate on",),
            )

        backend = None
        if adapter_dir is not None:
            from ml.training.adapters import NumpyLoRAAdapter

            backend = NumpyLoRAAdapter()
            if hasattr(backend, "bind"):
                backend.bind(model)
            backend.load(Path(adapter_dir))

        baseline = evaluate(model, None, held_out, tokenizer, instruction_pairs=instruction_pairs)
        candidate = evaluate(model, backend, held_out, tokenizer, instruction_pairs=instruction_pairs)
        return build_report(
            baseline=baseline,
            candidate=candidate,
            held_out_examples=len(held_out),
            notes=("metrics are held-out next-token loss/accuracy and a rough instruction score",),
        )

    # -- promotion / rollback ---------------------------------------------
    def promote(self, adapter_id: str, report: EvaluationReport, *, approved: bool = False) -> PromotionDecision:
        """Promote a candidate only when its evaluation accepts it and it is approved."""
        return self.promotion.evaluate_candidate(adapter_id, report, approved=approved)

    def rollback(self) -> AdapterRecord | None:
        """Return to the previous accepted adapter (or the base)."""
        return self.rollback_manager.rollback()

    def to_base(self) -> None:
        self.rollback_manager.to_base()

    # -- reporting ---------------------------------------------------------
    def status(self) -> str:
        lines = ["Controlled model-weight training (Phase 6)", self.registry.report()]
        lines.append(f"aggregate isolation clean: {main_model_dataset_is_clean()}")
        runs = self.trainer.runs()
        lines.append(f"training runs: {len(runs)}")
        for run in runs[-5:]:
            lines.append(f"  {run.describe()}")
        return "\n".join(lines)



__all__ = [
    "FeasibilityReport",
    "TrainingCoordinator",
]
