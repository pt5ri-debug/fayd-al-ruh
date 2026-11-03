# Phase 6 — Controlled Model Weight Training

**Status:** implemented, gated, and wired.
**Metric:** `training_accuracy` (held-out, 10 judgements, 1.0000).
**Baseline:** `evaluation/baseline.json` `overall` 0.9898.

## 1. Goal

Add a **safe, reproducible, local** model-training and fine-tuning pipeline:
controlled model-weight adaptation through LoRA/PEFT-style techniques, **only when
the selected model, runtime, and hardware actually support them**, trained on
**authorised** data, and never touching the Cognitive Core or the base weights.

## 2. What was already there (reused, not rebuilt)

Phase 6 **extends** the existing architecture rather than replacing it:

| Existing module | What Phase 6 reuses |
| --- | --- |
| `ml/llm/transformer.py` | the real `TransformerLM`/`TransformerConfig` (forward, `loss`, `backward`, `parameters`) — the training base |
| `ml/llm/tokenizer.py` | `ByteTokenizer` for encoding the corpus |
| `core/model_registry.py` | `ModelRegistry`, `ModelRecord`, `FORMAT_*`, `SOURCE_TRAINED` |
| `core/model_manager.py` | the one place models are listed/selected — a **view** over the training registry |
| `core/optional_deps.py` | the `[training]` extra (`FEATURE_TRAINING`, `requirements-training.txt`) |
| `core/paths.py` | `training_runs_dir`, `training_adapters_dir` (git-ignored `data/`) |
| `core/learning_isolation.py`, `core/agents/learning_scope.py` | the **Natasha exclusion** — asked, never re-implemented |
| `evaluation/` | the metric/benchmark/baseline gate |

## 3. New modules (`ml/training/`)

| Module | Responsibility |
| --- | --- |
| `config.py` | `TrainingConfig` (validated, bounded, preset-driven) |
| `compatibility.py` | which methods a given model/runtime/hardware **really** supports |
| `resources.py` | honest memory/disk estimates and a live `ResourceMonitor` |
| `adapters.py` | the LoRA backends: `NumpyLoRAAdapter` (in-process) and `PeftAdapterBackend` (transformers, optional extra) |
| `datasets.py` | `TrainingExample`, `DatasetBuilder`, `DatasetValidator`, splits, provenance, versioning, `aggregate_examples` |
| `isolation.py` | the training-side isolation audit (independent scope + Natasha exclusion) |
| `trainer.py` | the run life cycle: feasibility, the job, the run record, base-preservation proof, cancellation, failure |
| `evaluation.py` | held-out measurement, `compare()`, the acceptance rule (`min_improvement`, `max_regression`) |
| `registry.py` | `TrainingRegistry`, `PromotionManager`, `RollbackManager`, `register_adapter_as_model` |
| `coordinator.py` | the one governed entry point tying every step together |
| `gate.py` | the held-out gate (10 judgements) + `training_accuracy` |

Plus `cli/train.py` (the headless twin) and the `python -m cli train` command.

## 4. The three learning mechanisms (kept separate)

Phase 6 implements only the **third**, and keeps it apart from the other two:

1. **Cognitive knowledge learning** — `core/learning_pipeline.py`, `cognition/`
   (graph/relations/beliefs). *Unchanged.*
2. **Experience-based learning** — `cognition/study.py`, `consolidation.py`,
   `CharacterLearning` records. *Unchanged.*
3. **Controlled model-weight training** — this phase. The **only** path that may
   touch weights, and only ever an **adapter**, never the base.

## 5. Safety properties (each proven by the gate)

- **The base is never modified.** `NumpyLoRAAdapter` freezes the base and keeps
  its own `A`/`B` matrices; the context manager restores the exact pre-add values,
  so the base is **bit-for-bit** unchanged (pinned by `test_adapter_training_lowers_loss_and_leaves_the_base`
  and the trainer's own `base_unchanged` check).
- **A GGUF is never a training base.** `validate_compatibility` refuses it by
  name — a quantised inference file cannot be fine-tuned in place.
- **QLoRA needs a CUDA GPU.** Refused honestly on this machine (AMD/Intel GPUs
  have no modern CUDA/ROCm support).
- **No run overwrites another.** A duplicate `run_id` is refused; every run is a
  new record.
- **A failed or cancelled run leaves no adapter.**
- **Promotion is never automatic.** A candidate is accepted only on a measured
  improvement and, for activation, an explicit approval.
- **Rollback restores the previous adapter and deletes nothing.**
- **Natasha is excluded.** `aggregate_examples` asks the Phase-5 isolation policy;
  `audit_dataset` re-derives the answer from the examples, and a hand-built
  aggregate containing her is refused.

## 6. Offline operation

Everything runs **locally**. The in-process method needs no network and no extra;
the transformers path is an optional extra whose weights are a one-time,
human-approved download. No cloud service or external API is introduced.

## 7. Measured result (this machine)

- `python -m pytest tests/test_training_*.py` — 112 tests pass.
- The gate: `training_accuracy` = **1.0000** (10/10 judgements).
- An end-to-end real LoRA run lowered held-out loss **5.57 → 4.97** with the base
  provably untouched; `overall` = **0.9898**.

## 8. CLI

```
python -m cli train status|compat|presets|feasibility|build-dataset|datasets|
                   validate|run [--approve]|runs|evaluate|adapters|promote
                   [--approve]|rollback|gate
```

Exit codes: `0` success, `2` invalid arguments, `3` the run/dataset is unknown,
`4` a gate refused the operation, `5` an unexpected failure.

## 9. Acceptance criteria

| Criterion | Evidence |
| --- | --- |
| A base model file is never modified | bit-for-bit base check in the adapter, trainer, and gate |
| A rejected run leaves the active version untouched | `PromotionManager` refuses; history intact |
| Dataset validation, provenance, versioning, splitting | `DatasetBuilder`/`DatasetValidator` + gate judgements 1–4 |
| The Natasha exclusion holds | `aggregate_examples`/`audit_dataset` + gate judgement 5 |
| Only supported methods are offered | `validate_compatibility` + gate judgement 6 |
| Controlled promotion and rollback | `PromotionManager`/`RollbackManager` + gate judgement 10 |
| The metric is wired and green | `training_accuracy` = 1.0000 in `evaluation` |

## 10. What is deliberately **not** done here

- **No in-place base-weight training.** GGUF/quantised bases are refused.
- **No automatic promotion.** The gate always asks.
- **No QLoRA on this hardware.** Refused by name; CPU LoRA is the supported path.
- **No change to the Cognitive Core, the personalities' datasets, or the model
  weights.** Phase 6 adds a pipeline beside them.
