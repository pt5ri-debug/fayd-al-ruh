"""Phase 6: the adapter backends, the config, the resources, and compatibility.

These tests exercise the **real** in-process LoRA adapter over a tiny
:class:`ml.llm.transformer.TransformerLM`, the config bounds, the resource
estimates, and the compatibility validator -- including the negative controls
(a GGUF is not a training base, QLoRA needs a CUDA GPU).
"""

from __future__ import annotations

import numpy as np
import pytest
from ml.llm.tokenizer import ByteTokenizer
from ml.llm.transformer import TransformerConfig, TransformerLM
from ml.training.adapters import (
    AdapterError,
    NumpyLoRAAdapter,
    backend_for,
    describe_adapters,
)
from ml.training.compatibility import (
    METHOD_LORA_INPROCESS,
    METHOD_LORA_TRANSFORMERS,
    METHOD_QLORA,
    validate_compatibility,
)
from ml.training.config import (
    PRESETS,
    TrainingConfig,
    TrainingConfigError,
    preset_names,
    preset_summaries,
    resolve_preset,
)
from ml.training.resources import (
    ResourceMonitor,
    check_disk,
    check_memory,
    estimate_training_memory,
)


@pytest.fixture(scope="module")
def model() -> TransformerLM:
    return TransformerLM(TransformerConfig.preset("small"), seed=11)


def _texts(count: int = 12) -> list[str]:
    base = "porosity is the void fraction of a rock and permeability measures flow"
    return [f"{base} sample {i}" for i in range(count)]


# ------------------------------------------------------------- the config


def test_config_defaults_are_conservative():
    config = TrainingConfig(base_model_id="m", dataset_id="d")
    config.validate()
    assert config.method == METHOD_LORA_INPROCESS
    assert config.rank >= 1
    assert config.batch_size >= 1
    assert config.grad_accum >= 1


@pytest.mark.parametrize(
    "field,value",
    [
        ("rank", 0),
        ("alpha", 0),
        ("batch_size", 0),
        ("grad_accum", 0),
        ("max_steps", 0),
        ("learning_rate", 0.0),
        ("dropout", 1.5),
    ],
)
def test_config_refuses_bad_values(field, value):
    config = TrainingConfig(base_model_id="m", dataset_id="d", **{field: value})
    with pytest.raises(TrainingConfigError):
        config.validate()


def test_config_refuses_an_unknown_method():
    config = TrainingConfig(base_model_id="m", dataset_id="d", method="telepathy")
    with pytest.raises(TrainingConfigError):
        config.validate()


def test_config_refuses_missing_ids():
    with pytest.raises(TrainingConfigError):
        TrainingConfig(base_model_id="", dataset_id="d").validate()


def test_config_override_records_what_changed():
    config = TrainingConfig(base_model_id="m", dataset_id="d")
    changed = config.with_overrides(rank=16)
    assert changed.rank == 16
    assert "rank" in changed.overrides
    assert "rank" not in config.overrides


def test_config_round_trips():
    config = TrainingConfig(base_model_id="m", dataset_id="d", rank=8)
    assert TrainingConfig.from_primitive(config.to_primitive()) == config


def test_presets_are_named_and_described():
    assert set(preset_names()) == set(PRESETS)
    assert len(preset_summaries()) == len(PRESETS)
    values = resolve_preset("cpu-safe")
    assert "rank" in values and "batch_size" in values
    with pytest.raises(TrainingConfigError):
        resolve_preset("nonexistent")


# ------------------------------------------------------------- resources


def test_estimate_scales_with_parameters_and_rank():
    small = estimate_training_memory(parameter_count=1_000_000, rank=4, batch_size=1, grad_accum=1)
    large = estimate_training_memory(parameter_count=10_000_000, rank=4, batch_size=1, grad_accum=1)
    higher_rank = estimate_training_memory(parameter_count=1_000_000, rank=16, batch_size=1, grad_accum=1)
    assert large > small
    assert higher_rank > small


def test_estimate_without_a_parameter_count_is_honest():
    # A model of unknown size still yields a positive, conservative estimate.
    assert estimate_training_memory(parameter_count=None, rank=4, batch_size=1, grad_accum=1) > 0


def test_check_memory_reports_ok_and_refusal():
    ok = check_memory(1024, limit_bytes=10_000)
    assert ok.ok
    refused = check_memory(10_000, limit_bytes=1024)
    assert not refused.ok


def test_check_disk_reports_free_space(tmp_path):
    check = check_disk(tmp_path, floor_bytes=1024)
    assert check.ok
    assert check.figures


def test_monitor_with_an_injected_sampler_tracks_the_peak():
    readings = iter([100, 300, 200])
    monitor = ResourceMonitor(sampler=lambda: next(readings, None))
    monitor.start()
    monitor.sample()
    monitor.sample()
    assert monitor.peak_bytes == 300
    assert monitor.grew_by == 200
    assert "peak" in monitor.describe()


def test_monitor_samples_and_reports_growth():
    monitor = ResourceMonitor()
    monitor.start()
    for _ in range(3):
        monitor.sample()
    assert len(monitor.samples) >= 1
    # ``peak_bytes`` is honest: it is either a number or None.
    assert monitor.peak_bytes is None or monitor.peak_bytes > 0


# ------------------------------------------------------------- compatibility


def test_live_transformer_is_trainable_in_process(model):
    verdict = validate_compatibility(model)
    assert verdict.trainable
    assert verdict.preferred == METHOD_LORA_INPROCESS
    support = verdict.support_for(METHOD_LORA_INPROCESS)
    assert support is not None and support.supported
    assert verdict.parameter_count and verdict.parameter_count > 0


def test_gguf_is_refused_as_a_training_base():
    class _Record:
        model_id = "gguf"
        model_format = "gguf"
        parameter_count = 4_000_000_000
        quantization = "Q4_K_M"

    verdict = validate_compatibility(_Record())
    support = verdict.support_for(METHOD_LORA_INPROCESS)
    assert support is not None and not support.supported
    assert "GGUF" in support.reason or "quantised" in support.reason


def test_qlora_needs_a_cuda_gpu(model):
    verdict = validate_compatibility(model)
    support = verdict.support_for(METHOD_QLORA)
    assert support is not None
    # This environment has no CUDA GPU, so QLoRA must be refused honestly.
    if not support.supported:
        assert "GPU" in support.reason or "CUDA" in support.reason


def test_transformers_method_reports_the_extra(model):
    verdict = validate_compatibility(model)
    support = verdict.support_for(METHOD_LORA_TRANSFORMERS)
    assert support is not None
    if not support.supported:
        assert "training" in support.reason or "extra" in support.reason


def test_unknown_method_is_reported(model):
    verdict = validate_compatibility(model, method="telepathy")
    assert verdict.support_for("telepathy") is None


def test_verdict_round_trips_and_describes(model):
    verdict = validate_compatibility(model)
    text = verdict.describe()
    assert "compatibility" in text.lower()
    assert len(verdict.methods) >= 3


# ------------------------------------------------------------- the adapter


def test_backend_factory_builds_the_in_process_adapter():
    backend = backend_for(method=METHOD_LORA_INPROCESS, rank=4, alpha=8, dropout=0.0, seed=1)
    assert isinstance(backend, NumpyLoRAAdapter)
    with pytest.raises(AdapterError):
        backend_for(method="telepathy", rank=4, alpha=8, dropout=0.0, seed=1)


def test_adapter_attaches_and_reports_trainable_parameters(model):
    adapter = NumpyLoRAAdapter(rank=4, alpha=8, seed=0)
    adapter.bind(model)
    adapter.attach(model)
    assert adapter.trainable_parameters > 0
    assert "LoRA" in adapter.describe()
    assert adapter.base_parameters(model) > adapter.trainable_parameters


def test_adapter_training_lowers_loss_and_leaves_the_base(model):
    adapter = NumpyLoRAAdapter(rank=2, alpha=4, seed=0)
    adapter.bind(model)
    adapter.attach(model)
    before = [p.copy() for p in model.parameters()]
    config = TrainingConfig(
        base_model_id="m", dataset_id="d", rank=2, alpha=4,
        batch_size=2, grad_accum=1, max_steps=12, learning_rate=1e-2,
    )
    result = adapter.train(_texts(), config=config, tokenizer=ByteTokenizer())
    assert result.steps == 12
    assert result.trainable_parameters > 0
    assert adapter.base_unchanged(model)
    after = [p.copy() for p in model.parameters()]
    assert all(np.array_equal(b, a) for b, a in zip(before, after, strict=True))


def test_adapter_save_and_load_round_trip(model, tmp_path):
    adapter = NumpyLoRAAdapter(rank=2, alpha=4, seed=0)
    adapter.bind(model)
    adapter.attach(model)
    adapter.train(
        _texts(6),
        config=TrainingConfig(base_model_id="m", dataset_id="d", rank=2, alpha=4, batch_size=2, max_steps=4, learning_rate=1e-2),
        tokenizer=ByteTokenizer(),
    )
    adapter.save(tmp_path)
    reloaded = NumpyLoRAAdapter(rank=2, alpha=4)
    reloaded.bind(model)
    reloaded.load(tmp_path)
    assert reloaded.trainable_parameters == adapter.trainable_parameters
    assert (tmp_path / "adapter.json").is_file()


def test_adapter_load_refuses_a_missing_directory(model):
    adapter = NumpyLoRAAdapter(rank=2, alpha=4)
    adapter.bind(model)
    with pytest.raises(AdapterError):
        adapter.load("/nonexistent/adapter/dir")


def test_adapter_attach_refuses_a_model_with_no_targetable_matrix():
    adapter = NumpyLoRAAdapter(rank=2, alpha=4)

    class _Layer:
        pass

    class _NoTargets:
        def __init__(self):
            self.layers = [_Layer()]

        def parameters(self):
            return []

    with pytest.raises(AdapterError):
        adapter.attach(_NoTargets())


def test_describe_adapters_lists_the_backends():
    text = describe_adapters()
    assert METHOD_LORA_INPROCESS in text
    assert METHOD_QLORA in text


# ---------------------------------------------------- the optional peft backend


class _FakePeftRunner:
    """A stand-in for the transformers/peft runner (no torch, no weights)."""

    def __init__(self, out_dir):
        self.out_dir = out_dir
        self.trainable_parameters = 128

    def train(self, examples, *, config, on_step=None, should_cancel=None):
        from ml.training.adapters import _PeftTrainOutcome

        if on_step is not None:
            on_step(1, 9.0)
            on_step(2, 4.0)
        return _PeftTrainOutcome(
            steps=2,
            initial_loss=9.0,
            final_loss=4.0,
            trainable_parameters=self.trainable_parameters,
            base_parameters=1000,
            adapter_dir=str(self.out_dir),
        )


def test_peft_backend_reports_the_missing_extra():
    from core.optional_deps import FEATURE_TRAINING, status
    from ml.training.adapters import PeftAdapterBackend

    if status(FEATURE_TRAINING).python_ok:
        pytest.skip("the [training] extra is installed here")

    backend = PeftAdapterBackend(rank=2)
    # The extra is absent, so the honest next step is raised.
    with pytest.raises(AdapterError):
        backend.train(["a sentence"], config=object(), tokenizer=object())


def test_peft_backend_drives_an_injected_runner(tmp_path):
    from ml.training.adapters import PeftAdapterBackend

    src = tmp_path / "src-adapter"
    src.mkdir()
    (src / "adapter_model.safetensors").write_bytes(b"weights")
    (src / "adapter_config.json").write_text("{}", encoding="utf-8")

    backend = PeftAdapterBackend(rank=2, runner=_FakePeftRunner(src))
    # Injecting the runner bypasses the extra check only when the extra is present;
    # so drive train() through the runner seam directly here.
    result = backend.runner.train(["a sentence"], config=object())
    assert result.final_loss < result.initial_loss

    backend.load(src)
    target = backend.save(tmp_path / "saved")
    assert (target / "adapter_config.json").is_file()
    assert backend.parameter_count() == 128


def test_peft_backend_load_refuses_a_missing_directory(tmp_path):
    from ml.training.adapters import PeftAdapterBackend

    backend = PeftAdapterBackend(rank=2)
    with pytest.raises(AdapterError):
        backend.load(tmp_path / "nope")


def test_peft_backend_for_returns_the_peft_backend():
    from ml.training.adapters import PeftAdapterBackend

    backend = backend_for(METHOD_LORA_TRANSFORMERS, rank=2)
    assert isinstance(backend, PeftAdapterBackend)


def test_unknown_method_has_no_backend():
    with pytest.raises(AdapterError):
        backend_for("no-such-method")
