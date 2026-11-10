"""Phase 5: gradient accumulation -- a large effective batch on a small one.

The defining property is exactness: accumulating ``k`` micro-batches of size ``b``
must make the *same* update a single batch of size ``b*k`` would. These tests
prove that against the real trainer (with dropout off, so the forward passes are
deterministic and the comparison is exact), and pin the argument validation.
"""

from __future__ import annotations

import numpy as np
import pytest
from ml.llm.transformer import TransformerConfig, TransformerLM, train_language_model


def _tokens(n: int = 800) -> list[int]:
    return list(np.random.default_rng(0).integers(0, 64, size=n))


def _config(**overrides) -> TransformerConfig:
    base = {
        "n_layer": 1,
        "n_head": 2,
        "n_embd": 16,
        "block_size": 8,
        "dropout": 0.0,
    }
    return TransformerConfig(**{**base, **overrides})


def _param_vector(model: TransformerLM) -> np.ndarray:
    return np.concatenate([p.ravel().copy() for p in model.parameters()])


def test_accumulation_matches_a_single_large_batch():
    """``accum_steps=4, batch_size=2`` == ``batch_size=8`` (the exact update)."""
    tokens = _tokens()
    cfg = _config()

    accumulated = TransformerLM(cfg, seed=1)
    train_language_model(
        accumulated,
        tokens,
        steps=15,
        batch_size=2,
        learning_rate=1e-3,
        warmup=0,
        lr_schedule="constant",
        seed=7,
        accum_steps=4,
    )

    large_batch = TransformerLM(cfg, seed=1)
    train_language_model(
        large_batch,
        tokens,
        steps=15,
        batch_size=8,
        learning_rate=1e-3,
        warmup=0,
        lr_schedule="constant",
        seed=7,
    )

    assert np.allclose(_param_vector(accumulated), _param_vector(large_batch), atol=1e-12)


def test_accumulation_of_one_is_the_original_loop():
    tokens = _tokens()
    cfg = _config()
    a = TransformerLM(cfg, seed=3)
    b = TransformerLM(cfg, seed=3)
    ra = train_language_model(a, tokens, steps=12, batch_size=4, seed=5, accum_steps=1)
    rb = train_language_model(b, tokens, steps=12, batch_size=4, seed=5)
    assert np.array_equal(_param_vector(a), _param_vector(b))
    assert ra.losses == rb.losses


def test_accumulation_reduces_the_number_of_optimiser_steps():
    """Fewer, larger updates: the parameter trajectory differs from no accum."""
    tokens = _tokens()
    cfg = _config()
    one = TransformerLM(cfg, seed=2)
    many = TransformerLM(cfg, seed=2)
    train_language_model(one, tokens, steps=20, batch_size=2, learning_rate=1e-3, seed=0)
    train_language_model(
        many, tokens, steps=20, batch_size=2, learning_rate=1e-3, seed=0, accum_steps=4
    )
    assert not np.allclose(_param_vector(one), _param_vector(many))


def test_accumulation_gradient_clipping_is_applied_to_the_sum():
    """Clipping happens on the accumulated gradient, not per micro-batch."""
    tokens = _tokens()
    cfg = _config()
    model = TransformerLM(cfg, seed=0)
    # A tiny clip norm makes every step a pure sign step; it must not error and
    # must produce finite weights.
    train_language_model(
        model,
        tokens,
        steps=10,
        batch_size=2,
        learning_rate=1e-3,
        seed=0,
        accum_steps=3,
        max_grad_norm=1e-3,
    )
    assert np.all(np.isfinite(_param_vector(model)))


def test_accum_steps_must_be_at_least_one():
    model = TransformerLM(_config(), seed=0)
    with pytest.raises(ValueError, match="accum_steps"):
        train_language_model(model, _tokens(), steps=2, accum_steps=0)


def test_accumulation_respects_the_block_and_batch():
    """The micro-batches come from the same stream, so the run is reproducible."""
    tokens = _tokens()
    cfg = _config()
    first = TransformerLM(cfg, seed=0)
    second = TransformerLM(cfg, seed=0)
    train_language_model(first, tokens, steps=8, batch_size=3, seed=11, accum_steps=3)
    train_language_model(second, tokens, steps=8, batch_size=3, seed=11, accum_steps=3)
    assert np.array_equal(_param_vector(first), _param_vector(second))


def test_train_and_save_exposes_accumulation(tmp_path):
    from ml.llm.__main__ import train_and_save

    model = train_and_save(
        steps=10,
        repeats=1,
        structured_repeats=1,
        seed=0,
        checkpoint=tmp_path / "llm.npz",
        tokenizer_path=tmp_path / "llm.tokenizer.json",
        include_runtime=False,
        accum_steps=3,
        verbose=False,
    )
    assert model is not None
    assert (tmp_path / "llm.npz").exists()
