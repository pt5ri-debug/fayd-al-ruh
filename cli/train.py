"""The Training CLI: the headless twin of the Phase-6 training pipeline.

It drives the *same* :class:`ml.training.coordinator.TrainingCoordinator` the rest
of the project uses, so the CLI, a GUI, and the tests share one dataset registry,
one training run log, one adapter registry, and one promotion gate -- no logic is
reimplemented here.

Commands:

* ``status`` -- the training status: adapters, the active one, and recent runs;
* ``feasibility --model M`` -- the pre-training feasibility report (no run);
* ``compat [--model M] [--method X]`` -- model compatibility, method by method;
* ``datasets`` -- every governed dataset version and its eligibility;
* ``build-dataset ID --text ...`` -- build a validated, versioned dataset;
* ``validate ID`` -- validate a stored dataset and report what it found;
* ``runs [--json]`` -- the training-run history;
* ``run --model M --dataset ID [--preset P] [--steps N]`` -- a **dry run** by
  default: it reports the feasibility and stops; ``--approve`` runs the real job;
* ``evaluate RUN_ID`` -- compare the base and the candidate on held-out data;
* ``promote RUN_ID [--approve]`` -- promote a candidate (needs an accepted
  evaluation *and* explicit approval);
* ``rollback`` -- return to the previous accepted adapter or the base;
* ``adapters`` -- the adapter registry, with promotion status;
* ``presets`` -- the conservative hyper-parameter presets;
* ``gate`` -- the Phase-6 held-out gate report.

Exit codes: ``0`` success, ``2`` invalid arguments, ``3`` the model, dataset, or
run is unknown, ``4`` a gate refused the operation, ``5`` an unexpected failure.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from ml.training.compatibility import METHODS, validate_compatibility
from ml.training.config import (
    PRESETS,
    TrainingConfig,
    TrainingConfigError,
    preset_summaries,
    resolve_preset,
)
from ml.training.datasets import (
    DatasetBuilder,
    DatasetValidationError,
    TrainingExample,
)
from ml.training.gate import describe as gate_describe
from ml.training.registry import TrainingRegistry


def _coordinator(*, root: str = ""):
    from ml.training.coordinator import TrainingCoordinator

    return TrainingCoordinator(root=Path(root)) if root else TrainingCoordinator()


def _fail(message: str, code: int = 2) -> int:
    print(message)
    return code


# ------------------------------------------------------------------ commands

def _cmd_status(args: argparse.Namespace) -> int:
    coordinator = _coordinator(root=args.root)
    if args.json:
        registry = coordinator.registry
        print(
            json.dumps(
                {
                    "active": registry.active_id(),
                    "adapters": [a.to_primitive() for a in registry.adapters()],
                    "runs": [r.to_primitive() for r in coordinator.trainer.runs()],
                },
                indent=2,
                ensure_ascii=False,
            )
        )
        return 0
    print(coordinator.status())
    return 0


def _cmd_feasibility(args: argparse.Namespace) -> int:
    from ml.llm.transformer import TransformerConfig, TransformerLM

    coordinator = _coordinator(root=args.root)
    model = _load_model(args.model) if args.model else TransformerLM(TransformerConfig.preset("small"))
    dataset = None
    if args.dataset:
        dataset = coordinator.datasets.latest(args.dataset)
        if dataset is None:
            return _fail(f"No dataset {args.dataset!r} was found.", 3)
    config = _config_from_args(args, dataset)
    report = coordinator.feasibility(model, dataset, config)
    if args.json:
        print(json.dumps({
            "ok": report.ok,
            "trainable": report.trainable,
            "blockers": list(report.blockers),
            "warnings": list(report.warnings),
            "unknowns": list(report.unknowns),
            "estimated_memory_bytes": report.estimated_memory_bytes,
        }, indent=2))
        return 0 if report.ok else 4
    print(report.describe())
    return 0 if report.ok else 4


def _cmd_compat(args: argparse.Namespace) -> int:
    from ml.llm.transformer import TransformerConfig, TransformerLM

    model = _load_model(args.model) if args.model else TransformerLM(TransformerConfig.preset("small"))
    verdict = validate_compatibility(model, method=args.method)
    if args.json:
        print(json.dumps({
            "model_id": verdict.model_id,
            "format": verdict.model_format,
            "trainable": verdict.trainable,
            "preferred": verdict.preferred,
            "methods": [
                {"method": m.method, "supported": m.supported, "reason": m.reason, "next_step": m.next_step}
                for m in verdict.methods
            ],
            "warnings": list(verdict.warnings),
            "unknowns": list(verdict.unknowns),
        }, indent=2))
        return 0
    print(verdict.describe())
    return 0


def _dataset_builder(args: argparse.Namespace) -> DatasetBuilder:
    root = getattr(args, "root", "")
    if root:
        return DatasetBuilder(root=Path(root) / "datasets")
    return DatasetBuilder()


def _cmd_datasets(args: argparse.Namespace) -> int:
    builder = _dataset_builder(args)
    records = builder.list_datasets()
    if args.json:
        print(json.dumps([r.to_primitive() for r in records], indent=2, ensure_ascii=False))
        return 0
    if not records:
        print("No governed datasets yet. Build one with `build-dataset`.")
        return 0
    for record in records:
        print(record.describe())
    return 0


def _cmd_build_dataset(args: argparse.Namespace) -> int:
    builder = _dataset_builder(args)
    texts = list(args.text or ())
    if args.text_file:
        texts.append(Path(args.text_file).read_text(encoding="utf-8"))
    if not texts:
        return _fail("Provide at least one --text or a --text-file.", 2)
    examples = [
        TrainingExample(
            text=text,
            owner=args.owner,
            source=args.source,
            license=args.license,
        )
        for text in texts
    ]
    try:
        built = builder.build(
            args.dataset_id,
            examples,
            source=args.source,
            license=args.license,
            intended_model=args.intended_model,
            intended_task=args.intended_task,
            personality_scope=tuple(args.scope or (args.owner,) if args.owner else args.scope or ()),
            allow_unverified=args.allow_unverified,
        )
    except DatasetValidationError as error:
        return _fail(str(error), 4)
    print(built.describe())
    if not built.report.ok:
        print(built.report.describe())
        return 4
    return 0


def _cmd_validate(args: argparse.Namespace) -> int:
    builder = _dataset_builder(args)
    version = args.version or 0
    try:
        built = builder.load(args.dataset_id, version) if version else builder.latest(args.dataset_id)
    except DatasetValidationError as error:
        return _fail(str(error), 3)
    if built is None:
        return _fail(f"No dataset {args.dataset_id!r} was found.", 3)
    print(built.describe())
    print(built.report.describe())
    return 0 if built.report.ok else 4


def _cmd_runs(args: argparse.Namespace) -> int:
    coordinator = _coordinator(root=args.root)
    runs = coordinator.trainer.runs()
    if args.json:
        print(json.dumps([r.to_primitive() for r in runs], indent=2, ensure_ascii=False))
        return 0
    if not runs:
        print("No training runs yet.")
        return 0
    for run in runs:
        print(run.describe())
    return 0


def _cmd_run(args: argparse.Namespace) -> int:
    from ml.llm.tokenizer import ByteTokenizer
    from ml.llm.transformer import TransformerConfig, TransformerLM

    coordinator = _coordinator(root=args.root)
    dataset = coordinator.datasets.latest(args.dataset)
    if dataset is None:
        return _fail(f"No dataset {args.dataset!r} was found.", 3)
    model = _load_model(args.model) if args.model else TransformerLM(TransformerConfig.preset("small"))
    config = _config_from_args(args, dataset)
    feasibility = coordinator.feasibility(model, dataset, config)
    print(feasibility.describe())
    if not feasibility.ok:
        return _fail("Training refused: " + "; ".join(feasibility.blockers), 4)
    if not args.approve:
        print("\nDry run: feasibility passed. Re-run with --approve to train.")
        return 0
    resume_from = Path(args.resume_from) if args.resume_from else None
    if resume_from is not None and not resume_from.exists():
        return _fail(f"No adapter to resume from at {resume_from}.", 3)
    if resume_from is not None:
        print(f"Continuing from the adapter at {resume_from}.")
    result = coordinator.start(
        model, dataset, config, tokenizer=ByteTokenizer(), resume_from=resume_from
    )
    print()
    print(result.describe())
    return 0 if result.ok else 4


def _cmd_evaluate(args: argparse.Namespace) -> int:
    from ml.llm.tokenizer import ByteTokenizer
    from ml.llm.transformer import TransformerConfig, TransformerLM

    coordinator = _coordinator(root=args.root)
    run = coordinator.trainer.get(args.run_id)
    if run is None:
        return _fail(f"No training run {args.run_id!r} was found.", 3)
    dataset = coordinator.datasets.latest(run.dataset_id)
    if dataset is None:
        return _fail(f"The run's dataset {run.dataset_id!r} was not found.", 3)
    model = TransformerLM(TransformerConfig.preset("small"))
    report = coordinator.evaluate(
        model, dataset, tokenizer=ByteTokenizer(), adapter_dir=run.adapter_path
    )
    print(report.describe())
    if args.json:
        print(json.dumps(report.to_primitive(), indent=2))
    return 0


def _cmd_promote(args: argparse.Namespace) -> int:
    coordinator = _coordinator(root=args.root)
    record = coordinator.registry.get(args.run_id)
    if record is None:
        return _fail(f"No adapter {args.run_id!r} is registered.", 3)
    from ml.training.evaluation import EvaluationReport

    if not record.evaluation:
        return _fail(
            "This candidate has no evaluation; run `evaluate <run_id>` first.", 4
        )
    report = EvaluationReport.from_primitive(record.evaluation)
    decision = coordinator.promote(args.run_id, report, approved=args.approve)
    print(decision.describe())
    return 0 if decision.promoted else 4


def _cmd_rollback(args: argparse.Namespace) -> int:
    coordinator = _coordinator(root=args.root)
    if args.to_base:
        coordinator.to_base()
        print("Rolled back to the base models (no adapter active).")
        return 0
    previous = coordinator.rollback()
    if previous is None:
        print("No previous accepted adapter; the base models are active.")
    else:
        print(f"Rolled back to {previous.adapter_id}.")
    return 0


def _cmd_adapters(args: argparse.Namespace) -> int:
    root = getattr(args, "root", "")
    registry = TrainingRegistry(Path(root) / "registry") if root else TrainingRegistry()
    if args.json:
        print(json.dumps([a.to_primitive() for a in registry.adapters()], indent=2, ensure_ascii=False))
        return 0
    print(registry.report())
    return 0


def _cmd_presets(args: argparse.Namespace) -> int:
    for name, summary in preset_summaries():
        print(f"{name}: {summary}")
    return 0


def _cmd_resume_gate(args: argparse.Namespace) -> int:
    from ml.resume_gate import describe as resume_describe
    from ml.resume_gate import judgements

    if args.json:
        print(json.dumps(
            [{"name": j.name, "passed": j.passed, "detail": j.detail} for j in judgements()],
            indent=2,
        ))
        return 0
    print(resume_describe())
    return 0


def _cmd_gate(args: argparse.Namespace) -> int:
    if args.json:
        from ml.training.gate import judgements

        print(json.dumps(
            [{"name": j.name, "passed": j.passed, "detail": j.detail} for j in judgements()],
            indent=2,
        ))
        return 0
    print(gate_describe())
    return 0


# ------------------------------------------------------------------ helpers

def _load_model(model_id: str):
    """Load a model by id from the Model Manager, or a live in-project checkpoint."""
    if not model_id or model_id in ("fayd", "fayd-llm", "fayd-inprocess-model"):
        from core.paths import default_checkpoint
        from ml.llm.transformer import TransformerLM

        checkpoint = default_checkpoint()
        if checkpoint.exists():
            return TransformerLM.load(checkpoint)
        from ml.llm.transformer import TransformerConfig

        return TransformerLM(TransformerConfig.preset("small"))
    from core.model_manager import ModelManager

    manager = ModelManager()
    record = manager.get(model_id)
    return record


def _config_from_args(args: argparse.Namespace, dataset) -> TrainingConfig:
    dataset_id = dataset.record.dataset_id if dataset is not None else (args.dataset or "none")
    version = dataset.record.version if dataset is not None else 0
    values: dict = {
        "base_model_id": args.model or "fayd-inprocess-model",
        "dataset_id": dataset_id,
        "dataset_version": version,
    }
    if getattr(args, "preset", ""):
        values.update(resolve_preset(args.preset))
    if getattr(args, "steps", None) is not None:
        values["max_steps"] = args.steps
    if getattr(args, "rank", None) is not None:
        values["rank"] = args.rank
        values.setdefault("alpha", args.rank * 2)
    if getattr(args, "batch_size", None) is not None:
        values["batch_size"] = args.batch_size
    config = TrainingConfig(**values)
    config.validate()
    return config


# ------------------------------------------------------------------- parser

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m cli train",
        description="Controlled model-weight training (Phase 6): governed datasets, "
        "LoRA adapters, evaluation, promotion, and rollback.",
    )
    parser.add_argument("--root", default="", help="the training data root (default: data/training)")
    sub = parser.add_subparsers(dest="command")

    p_status = sub.add_parser("status", help="the training status")
    p_status.add_argument("--json", action="store_true")
    p_status.set_defaults(func=_cmd_status)

    p_feas = sub.add_parser("feasibility", help="the pre-training feasibility report")
    p_feas.add_argument("--model", default="")
    p_feas.add_argument("--dataset", default="")
    p_feas.add_argument("--method", default="lora_inprocess", choices=list(METHODS))
    p_feas.add_argument("--preset", default="", choices=list(PRESETS))
    p_feas.add_argument("--steps", type=int, default=None)
    p_feas.add_argument("--rank", type=int, default=None)
    p_feas.add_argument("--batch-size", type=int, default=None)
    p_feas.add_argument("--json", action="store_true")
    p_feas.set_defaults(func=_cmd_feasibility)

    p_compat = sub.add_parser("compat", help="model compatibility, method by method")
    p_compat.add_argument("--model", default="")
    p_compat.add_argument("--method", default="", choices=["", *METHODS])
    p_compat.add_argument("--json", action="store_true")
    p_compat.set_defaults(func=_cmd_compat)

    p_ds = sub.add_parser("datasets", help="every governed dataset version")
    p_ds.add_argument("--json", action="store_true")
    p_ds.set_defaults(func=_cmd_datasets)

    p_build = sub.add_parser("build-dataset", help="build a validated, versioned dataset")
    p_build.add_argument("dataset_id")
    p_build.add_argument("--text", action="append", default=[])
    p_build.add_argument("--text-file", default="")
    p_build.add_argument("--owner", default="")
    p_build.add_argument("--scope", action="append", default=[])
    p_build.add_argument("--source", default="")
    p_build.add_argument("--license", default="")
    p_build.add_argument("--intended-model", default="")
    p_build.add_argument("--intended-task", default="")
    p_build.add_argument("--allow-unverified", action="store_true")
    p_build.set_defaults(func=_cmd_build_dataset)

    p_val = sub.add_parser("validate", help="validate a stored dataset")
    p_val.add_argument("dataset_id")
    p_val.add_argument("--version", type=int, default=0)
    p_val.set_defaults(func=_cmd_validate)

    p_runs = sub.add_parser("runs", help="the training-run history")
    p_runs.add_argument("--json", action="store_true")
    p_runs.set_defaults(func=_cmd_runs)

    p_run = sub.add_parser("run", help="run a training job (dry run unless --approve)")
    p_run.add_argument("--model", default="")
    p_run.add_argument("--dataset", required=True)
    p_run.add_argument("--method", default="lora_inprocess", choices=list(METHODS))
    p_run.add_argument("--preset", default="cpu-safe", choices=list(PRESETS))
    p_run.add_argument("--steps", type=int, default=None)
    p_run.add_argument("--rank", type=int, default=None)
    p_run.add_argument("--batch-size", type=int, default=None)
    p_run.add_argument(
        "--resume-from",
        default="",
        help="continue from the adapter saved by an earlier run (a path to its "
        "adapter directory) instead of starting a fresh adapter",
    )
    p_run.add_argument("--approve", action="store_true", help="actually run the job")
    p_run.set_defaults(func=_cmd_run)

    p_eval = sub.add_parser("evaluate", help="compare the base and the candidate")
    p_eval.add_argument("run_id")
    p_eval.add_argument("--json", action="store_true")
    p_eval.set_defaults(func=_cmd_evaluate)

    p_prom = sub.add_parser("promote", help="promote a candidate (needs --approve)")
    p_prom.add_argument("run_id")
    p_prom.add_argument("--approve", action="store_true")
    p_prom.set_defaults(func=_cmd_promote)

    p_rb = sub.add_parser("rollback", help="return to the previous accepted adapter")
    p_rb.add_argument("--to-base", action="store_true")
    p_rb.set_defaults(func=_cmd_rollback)

    p_ad = sub.add_parser("adapters", help="the adapter registry")
    p_ad.add_argument("--json", action="store_true")
    p_ad.set_defaults(func=_cmd_adapters)

    p_pre = sub.add_parser("presets", help="the conservative hyper-parameter presets")
    p_pre.set_defaults(func=_cmd_presets)

    p_gate = sub.add_parser("gate", help="the Phase-6 held-out gate report")
    p_gate.add_argument("--json", action="store_true")
    p_gate.set_defaults(func=_cmd_gate)

    p_resume = sub.add_parser(
        "resume-gate", help="the held-out resumable-training gate report"
    )
    p_resume.add_argument("--json", action="store_true")
    p_resume.set_defaults(func=_cmd_resume_gate)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)
    if getattr(args, "func", None) is None:
        return _cmd_status(argparse.Namespace(root="", json=False))
    try:
        return int(args.func(args))
    except TrainingConfigError as error:
        return _fail(str(error), 2)
    except DatasetValidationError as error:
        return _fail(str(error), 4)
    except (ValueError, OSError, KeyError) as error:
        return _fail(f"{type(error).__name__}: {error}", 5)


__all__ = ["build_parser", "main"]
