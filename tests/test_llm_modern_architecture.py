"""The modern architecture primitives must be correct, not merely fast.

The hand-written backprop is the risky part of adding RMSNorm, RoPE, and SwiGLU:
a wrong gradient still trains (it just trains the wrong model), so this file
checks the analytic gradients against **numerical** ones by finite differences,
for every combination of the variants. It also pins the invariants the rest of
the project relies on: the defaults reproduce the original architecture exactly,
the parameter/gradient lists stay aligned, a checkpoint round-trips, and the
incremental cache is still exact under RoPE.
"""

from __future__ import annotations

import numpy as np
import pytest
from ml.llm.transformer import (
    ACTIVATION_KINDS,
    NORM_KINDS,
    POSITION_KINDS,
    KVCache,
    TransformerConfig,
    TransformerLM,
    _apply_rope,
    _rms_norm,
    _rms_norm_backward,
    _rope_tables,
    _silu,
    _silu_grad,
)


def _config(**overrides) -> TransformerConfig:
    base = {"n_layer": 2, "n_head": 2, "n_embd": 16, "block_size": 16, "dropout": 0.0}
    return TransformerConfig(**{**base, **overrides})


def _numeric_grads(model: TransformerLM, idx: np.ndarray, targets: np.ndarray, eps=1e-6):
    """Central-difference gradients for every parameter, as a flat list."""
    params = model.parameters()
    numeric = []
    for param in params:
        grad = np.zeros_like(param)
        flat = param.reshape(-1)
        gflat = grad.reshape(-1)
        for i in range(flat.size):
            original = flat[i]
            flat[i] = original + eps
            model.eval()
            plus = model.loss(model.forward(idx), targets)
            flat[i] = original - eps
            model.eval()
            minus = model.loss(model.forward(idx), targets)
            flat[i] = original
            gflat[i] = (plus - minus) / (2 * eps)
        numeric.append(grad)
    return numeric


@pytest.mark.parametrize("norm", NORM_KINDS)
@pytest.mark.parametrize("position", POSITION_KINDS)
@pytest.mark.parametrize("activation", ACTIVATION_KINDS)
def test_analytic_gradients_match_numerical(norm, position, activation):
    """Every variant's backprop must agree with finite differences."""
    model = TransformerLM(
        _config(norm=norm, position=position, activation=activation), seed=1
    )
    idx = np.array([[1, 2, 3, 4, 5]], dtype=np.int64)
    targets = np.array([[2, 3, 4, 5, 6]], dtype=np.int64)

    model.eval()
    model.loss(model.forward(idx), targets)
    analytic = model.backward(targets)
    numeric = _numeric_grads(model, idx, targets)

    assert len(analytic) == len(numeric) == len(model.parameters())
    for index, (a, n) in enumerate(zip(analytic, numeric, strict=True)):
        assert np.allclose(a, n, rtol=1e-4, atol=1e-5), (
            f"gradient {index} differs for {norm}/{position}/{activation}"
        )


def test_defaults_reproduce_the_original_architecture():
    """The default config is exactly the pre-upgrade model."""
    model = TransformerLM(_config(), seed=3)
    config = model.config
    assert (config.norm, config.position, config.activation) == ("layer", "learned", "gelu")
    # The original per-layer parameter list, in the original order.
    names = model._layer_param_names()
    assert names == (
        "ln1_w", "ln1_b", "attn_w", "attn_b",
        "proj_w", "proj_b", "ln2_w", "ln2_b",
        "fc_w", "fc_b", "fc2_w", "fc2_b",
    )


def test_parameter_and_gradient_lists_stay_aligned_for_every_variant():
    """``parameters()`` and ``_flatten_grads`` must describe the same list."""
    for norm in NORM_KINDS:
        for position in POSITION_KINDS:
            for activation in ACTIVATION_KINDS:
                model = TransformerLM(
                    _config(norm=norm, position=position, activation=activation), seed=2
                )
                idx = np.array([[1, 2, 3, 4]], dtype=np.int64)
                targets = np.array([[2, 3, 4, 5]], dtype=np.int64)
                model.eval()
                model.loss(model.forward(idx), targets)
                grads = model.backward(targets)
                params = model.parameters()
                assert len(grads) == len(params)
                for p, g in zip(params, grads, strict=True):
                    assert p.shape == g.shape


def test_rms_norm_has_no_bias_and_no_mean_subtraction():
    x = np.array([[1.0, 2.0, 3.0, 4.0]])
    weight = np.ones(4)
    normed, _ = _rms_norm(x, weight)
    # RMS of [1,2,3,4] is sqrt(30/4); the output is x / rms, with no centring.
    rms = np.sqrt(np.mean(x * x) + 1e-5)
    assert np.allclose(normed, x / rms)
    assert not np.allclose(normed.mean(), 0.0)


def test_rms_norm_backward_matches_finite_differences():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(2, 3, 8))
    weight = rng.normal(size=8)
    grad_out = rng.normal(size=(2, 3, 8))

    normed, cache = _rms_norm(x, weight)
    _ = normed
    grad_x, grad_w = _rms_norm_backward(grad_out, cache, weight, 8)

    eps = 1e-6
    numeric_x = np.zeros_like(x)
    for index in np.ndindex(x.shape):
        plus = x.copy()
        plus[index] += eps
        minus = x.copy()
        minus[index] -= eps
        numeric_x[index] = (
            np.sum(_rms_norm(plus, weight)[0] * grad_out)
            - np.sum(_rms_norm(minus, weight)[0] * grad_out)
        ) / (2 * eps)
    assert np.allclose(grad_x, numeric_x, rtol=1e-5, atol=1e-7)

    numeric_w = np.zeros_like(weight)
    for i in range(weight.size):
        plus = weight.copy()
        plus[i] += eps
        minus = weight.copy()
        minus[i] -= eps
        numeric_w[i] = (
            np.sum(_rms_norm(x, plus)[0] * grad_out)
            - np.sum(_rms_norm(x, minus)[0] * grad_out)
        ) / (2 * eps)
    assert np.allclose(grad_w, numeric_w, rtol=1e-5, atol=1e-7)


def test_silu_and_its_gradient():
    x = np.linspace(-4, 4, 21)
    eps = 1e-6
    numeric = (_silu(x + eps) - _silu(x - eps)) / (2 * eps)
    assert np.allclose(_silu_grad(x), numeric, rtol=1e-5, atol=1e-6)


def test_rope_is_orthogonal_and_position_dependent():
    head_dim = 8
    x = np.random.default_rng(0).normal(size=(1, 2, 3, head_dim))
    cos, sin = _rope_tables(head_dim, np.arange(3))
    rotated = _apply_rope(x, cos[None, None], sin[None, None])
    # Rotation preserves the norm of each vector.
    assert np.allclose(np.linalg.norm(rotated, axis=-1), np.linalg.norm(x, axis=-1))
    # The inverse rotation recovers the original.
    recovered = _apply_rope(rotated, cos[None, None], sin[None, None], inverse=True)
    assert np.allclose(recovered, x, atol=1e-12)
    # Different positions rotate differently.
    assert not np.allclose(rotated[0, 0, 0], rotated[0, 0, 1])


def test_rope_config_has_no_learned_position_embedding():
    model = TransformerLM(_config(position="rope"), seed=0)
    # The position table is not part of the parameter list under RoPE.
    ids = {id(p) for p in model.parameters()}
    assert id(model.position_embedding) not in ids
    learned_ids = {id(p) for p in TransformerLM(_config(position="learned"), seed=0).parameters()}
    assert len(ids) == len(learned_ids) - 1


def test_swiglu_uses_the_gated_hidden_width():
    gelu = _config(activation="gelu")
    swiglu = _config(activation="swiglu")
    assert gelu.mlp_hidden == 4 * gelu.n_embd
    assert swiglu.mlp_hidden == 48  # ceil(8*16/3 / 8) * 8
    # The two blocks carry a comparable parameter count.
    gelu_params = sum(p.size for p in TransformerLM(gelu, seed=0).parameters())
    swiglu_params = sum(p.size for p in TransformerLM(swiglu, seed=0).parameters())
    assert abs(gelu_params - swiglu_params) / gelu_params < 0.2


def test_modern_preset_is_the_three_primitives():
    config = TransformerConfig.preset("modern")
    assert (config.norm, config.position, config.activation) == ("rms", "rope", "swiglu")


def test_checkpoint_round_trips_for_every_variant(tmp_path):
    for norm in NORM_KINDS:
        for position in POSITION_KINDS:
            for activation in ACTIVATION_KINDS:
                model = TransformerLM(
                    _config(norm=norm, position=position, activation=activation), seed=5
                )
                model.eval()
                path = tmp_path / f"{norm}-{position}-{activation}.npz"
                model.save(path)
                loaded = TransformerLM.load(path)
                assert loaded.config.norm == norm
                assert loaded.config.position == position
                assert loaded.config.activation == activation
                idx = np.array([[1, 2, 3, 4]], dtype=np.int64)
                assert np.allclose(loaded.forward(idx), model.forward(idx), atol=1e-12)


def test_old_checkpoint_without_architecture_loads_as_the_original(tmp_path):
    """A checkpoint written before this change must still load (as LayerNorm/GELU)."""
    model = TransformerLM(_config(), seed=7)
    path = tmp_path / "old.npz"
    model.save(path)
    # Strip the architecture array to mimic a pre-upgrade checkpoint.
    data = dict(np.load(path, allow_pickle=False))
    data.pop("architecture")
    np.savez_compressed(path, **data)
    loaded = TransformerLM.load(path)
    assert (loaded.config.norm, loaded.config.position, loaded.config.activation) == (
        "layer",
        "learned",
        "gelu",
    )


def test_incremental_cache_is_exact_under_rope():
    model = TransformerLM(_config(position="rope", activation="swiglu", norm="rms"), seed=9).eval()
    tokens = [1, 2, 3, 4, 5, 6]
    full = model.forward(np.array(tokens, dtype=np.int64))[0]
    cache = KVCache()
    for index, token in enumerate(tokens):
        logits = model.forward_incremental(np.array([token], dtype=np.int64), cache)[0, -1]
        assert np.allclose(logits, full[index], atol=1e-10)


def test_training_step_reduces_loss_for_every_variant():
    """A real optimisation check: one variant must actually learn, not just run."""
    for activation in ACTIVATION_KINDS:
        model = TransformerLM(_config(activation=activation), seed=4)
        idx = np.array([[1, 2, 3, 4, 5, 6]], dtype=np.int64)
        targets = np.array([[2, 3, 4, 5, 6, 7]], dtype=np.int64)
        model.eval()
        first = model.loss(model.forward(idx), targets)
        grads = model.backward(targets)
        from ml.models import Adam

        optimizer = Adam(model.parameters(), lr=0.02)
        optimizer.step(grads)
        last = model.loss(model.forward(idx), targets)
        assert last < first
