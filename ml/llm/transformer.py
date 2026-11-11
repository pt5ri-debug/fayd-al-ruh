"""A decoder-only transformer language model written from scratch in numpy.

This is the project's own language model: token and position embeddings, causal
multi-head self-attention, a GELU feed-forward block, and a softmax output head,
with hand-written backpropagation and an Adam optimiser. It depends only on
numpy and runs entirely inside the process, so no external model server is
needed.

It is a real transformer, but a small one: a few layers and a modest embedding
width, trained in-project on a small corpus. It is not, and does not claim to
be, a frontier model.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from core.progress import ProgressCallback

from ml.llm.tokenizer import VOCAB_SIZE
from ml.models import Adam

_EPS = 1e-5
_GELU_C = float(np.sqrt(2.0 / np.pi))

#: Normalisation variants. ``layer`` is the original LayerNorm; ``rms`` is RMSNorm
#: (no mean subtraction, no bias), the standard modern choice.
NORM_KINDS = ("layer", "rms")
#: Positional-encoding variants. ``learned`` is the original learned table;
#: ``rope`` rotates queries/keys by position (rotary embeddings).
POSITION_KINDS = ("learned", "rope")
#: Feed-forward variants. ``gelu`` is the original GELU MLP; ``swiglu`` is a gated
#: SwiGLU block.
ACTIVATION_KINDS = ("gelu", "swiglu")
#: RoPE base frequency (the ``10000`` of the original paper).
_ROPE_BASE = 10000.0


@dataclass(frozen=True)
class TransformerConfig:
    vocab_size: int = VOCAB_SIZE
    n_layer: int = 3
    n_head: int = 4
    n_embd: int = 96
    block_size: int = 64
    dropout: float = 0.0
    tie_weights: bool = False
    # Architecture variants. The defaults reproduce the original model exactly,
    # so an existing checkpoint and every existing test are unaffected; a caller
    # (or the ``modern`` preset) opts into the modern primitives deliberately.
    norm: str = "layer"
    position: str = "learned"
    activation: str = "gelu"

    def __post_init__(self) -> None:
        if self.n_embd % self.n_head != 0:
            raise ValueError("n_embd must be divisible by n_head.")
        if min(self.vocab_size, self.n_layer, self.n_head, self.n_embd, self.block_size) < 1:
            raise ValueError("All transformer dimensions must be positive.")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in [0, 1).")
        if self.norm not in NORM_KINDS:
            raise ValueError(f"norm must be one of {NORM_KINDS}, got {self.norm!r}.")
        if self.position not in POSITION_KINDS:
            raise ValueError(f"position must be one of {POSITION_KINDS}, got {self.position!r}.")
        if self.activation not in ACTIVATION_KINDS:
            raise ValueError(
                f"activation must be one of {ACTIVATION_KINDS}, got {self.activation!r}."
            )

    @property
    def mlp_hidden(self) -> int:
        """Hidden width of the feed-forward block for this activation.

        GELU keeps the classic ``4 * n_embd``. SwiGLU uses the usual ``8/3`` ratio
        (rounded up to a multiple of 8), so its gate+up projections keep the
        parameter count close to the GELU block's rather than doubling it.
        """
        if self.activation == "swiglu":
            hidden = int(np.ceil(8 * self.n_embd / 3 / 8) * 8)
            return max(8, hidden)
        return 4 * self.n_embd

    @classmethod
    def preset(cls, name: str, *, vocab_size: int | None = None) -> TransformerConfig:
        """A named capacity tier, so growth is a deliberate, recorded choice."""
        if name not in PRESETS:
            raise ValueError(f"Unknown preset {name!r}; expected one of {tuple(PRESETS)}.")
        spec = dict(PRESETS[name])
        if vocab_size is not None:
            spec["vocab_size"] = vocab_size
        return cls(**spec)


# Capacity tiers. ``small`` is the original model and stays the default so
# existing checkpoints and tests are unaffected; ``release-1`` is the first
# deliberately larger tier; ``capable`` is the stronger, slower option.
PRESETS: dict[str, dict] = {
    "small": {"n_layer": 3, "n_head": 4, "n_embd": 96, "block_size": 64},
    "release-1": {
        "n_layer": 4,
        "n_head": 8,
        "n_embd": 128,
        "block_size": 128,
        "dropout": 0.1,
        "tie_weights": True,
    },
    "capable": {
        "n_layer": 6,
        "n_head": 8,
        "n_embd": 192,
        "block_size": 128,
        "dropout": 0.1,
        "tie_weights": True,
    },
    # The modern tier: RMSNorm + RoPE + SwiGLU. Same width/layer count as
    # ``release-1`` so a comparison isolates the primitives, not the capacity.
    "modern": {
        "n_layer": 4,
        "n_head": 8,
        "n_embd": 128,
        "block_size": 128,
        "dropout": 0.1,
        "tie_weights": True,
        "norm": "rms",
        "position": "rope",
        "activation": "swiglu",
    },
    # The learned tiers: the modern primitives at a genuinely larger capacity, so
    # a longer run has somewhere to put what it learns. ``learned-large`` is the
    # tier a release trains and ships (a wider 4-layer model with a longer
    # context); ``learned-xl`` is the stronger option a user with time may build.
    # Both keep the same architecture family as ``modern`` -- only the capacity
    # changes -- so a comparison isolates size, not a different model.
    "learned-large": {
        "n_layer": 4,
        "n_head": 8,
        "n_embd": 256,
        "block_size": 256,
        "dropout": 0.1,
        "tie_weights": True,
        "norm": "rms",
        "position": "rope",
        "activation": "swiglu",
    },
    "learned-xl": {
        "n_layer": 8,
        "n_head": 8,
        "n_embd": 320,
        "block_size": 256,
        "dropout": 0.1,
        "tie_weights": True,
        "norm": "rms",
        "position": "rope",
        "activation": "swiglu",
    },
}


def _gelu(x: np.ndarray) -> np.ndarray:
    # ``x*x*x`` rather than ``x**3``: numpy routes an integer power through the
    # slow general ``power`` ufunc, and this activation dominates the step.
    c = _GELU_C
    return 0.5 * x * (1.0 + np.tanh(c * (x + 0.044715 * x * x * x)))


def _gelu_grad(x: np.ndarray) -> np.ndarray:
    c = _GELU_C
    x2 = x * x
    inner = c * (x + 0.044715 * x2 * x)
    tanh = np.tanh(inner)
    sech2 = 1.0 - tanh * tanh
    return 0.5 * (1.0 + tanh) + 0.5 * x * sech2 * c * (1.0 + 3 * 0.044715 * x2)


def _softmax_rows(x: np.ndarray) -> np.ndarray:
    shifted = x - x.max(axis=-1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=-1, keepdims=True)


def _apply_mask(grad: np.ndarray, mask: np.ndarray | None) -> np.ndarray:
    """Backpropagate through inverted dropout: zero what was dropped."""
    return grad if mask is None else grad * mask


def _layer_norm(x: np.ndarray, weight: np.ndarray, bias: np.ndarray):
    mean = x.mean(axis=-1, keepdims=True)
    var = x.var(axis=-1, keepdims=True)
    std = np.sqrt(var + _EPS)
    normed = (x - mean) / std
    return normed * weight + bias, (normed, std)


def _layer_norm_backward(
    grad_out: np.ndarray, cache, weight: np.ndarray, n_embd: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    normed, std = cache
    grad_weight = (grad_out * normed).reshape(-1, n_embd).sum(axis=0)
    grad_bias = grad_out.reshape(-1, n_embd).sum(axis=0)
    grad_normed = grad_out * weight
    grad_x = (1.0 / std) * (
        grad_normed
        - grad_normed.mean(axis=-1, keepdims=True)
        - normed * (grad_normed * normed).mean(axis=-1, keepdims=True)
    )
    return grad_x, grad_weight, grad_bias


def _rms_norm(x: np.ndarray, weight: np.ndarray):
    """RMSNorm: scale by the root-mean-square, no mean subtraction, no bias."""
    mean_square = np.mean(x * x, axis=-1, keepdims=True)
    inv_rms = 1.0 / np.sqrt(mean_square + _EPS)
    normed = x * inv_rms
    return normed * weight, (normed, inv_rms)


def _rms_norm_backward(
    grad_out: np.ndarray, cache, weight: np.ndarray, n_embd: int
) -> tuple[np.ndarray, np.ndarray]:
    """Gradient for RMSNorm. Returns ``(grad_x, grad_weight)`` -- there is no bias.

    With ``x_hat = x / rms`` and ``y = w * x_hat``, ``dL/dx = (1/rms) * (g - x_hat
    * mean(g * x_hat))`` where ``g = grad_out * w``: the mean over the last axis
    makes every element's gradient depend on all the others.
    """
    normed, inv_rms = cache
    grad_weight = (grad_out * normed).reshape(-1, n_embd).sum(axis=0)
    grad_normed = grad_out * weight
    correction = np.mean(grad_normed * normed, axis=-1, keepdims=True)
    grad_x = inv_rms * (grad_normed - normed * correction)
    return grad_x, grad_weight


def _silu(x: np.ndarray) -> np.ndarray:
    return x / (1.0 + np.exp(-x))


def _silu_grad(x: np.ndarray) -> np.ndarray:
    sig = 1.0 / (1.0 + np.exp(-x))
    return sig * (1.0 + x * (1.0 - sig))


def _rope_tables(head_dim: int, positions: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Cosine/sine tables for rotary position embedding.

    ``positions`` is a 1-D array of token positions. Returns ``(cos, sin)`` each
    shaped ``(len(positions), head_dim // 2)`` -- one angle per rotation pair, so
    they broadcast against ``x[..., 0::2]`` in :func:`_apply_rope`.
    """
    half = head_dim // 2
    inv_freq = 1.0 / (_ROPE_BASE ** (np.arange(half) / half))
    angles = positions[:, None] * inv_freq[None, :]
    return np.cos(angles), np.sin(angles)


def _apply_rope(x: np.ndarray, cos: np.ndarray, sin: np.ndarray, *, inverse: bool = False) -> np.ndarray:
    """Rotate ``x`` (shape ``(batch, n_head, seq, head_dim)``) by position.

    Rotation pairs ``(2i, 2i+1)`` of the head dimension. The inverse rotation
    (``sin`` negated) is the transpose of the forward rotation, which is what the
    backward pass needs.
    """
    even = x[..., 0::2]
    odd = x[..., 1::2]
    if inverse:
        sin = -sin
    rotated_even = even * cos - odd * sin
    rotated_odd = even * sin + odd * cos
    out = np.empty_like(x)
    out[..., 0::2] = rotated_even
    out[..., 1::2] = rotated_odd
    return out


@dataclass
class KVCache:
    """Per-layer keys and values for incremental (cached) generation.

    Generation used to recompute the whole window for every new token, which is
    quadratic in the sequence length: a 100-token answer ran the forward pass 100
    times over growing windows. A cache holds each layer's keys and values from
    the tokens already seen, so one new token costs one token's work. It is
    inference-only (dropout is never applied through it) and it is exact: the
    logits it produces are the ones the full forward pass produces, which
    ``tests/test_llm_kv_cache.py`` proves to machine precision.
    """

    keys: list[np.ndarray] = field(default_factory=list)
    values: list[np.ndarray] = field(default_factory=list)
    length: int = 0

    def reset(self) -> None:
        self.keys = []
        self.values = []
        self.length = 0


@dataclass
class _LayerParams:
    attn_w: np.ndarray
    attn_b: np.ndarray
    proj_w: np.ndarray
    proj_b: np.ndarray
    ln1_w: np.ndarray
    ln1_b: np.ndarray
    ln2_w: np.ndarray
    ln2_b: np.ndarray
    fc_w: np.ndarray
    fc_b: np.ndarray
    fc2_w: np.ndarray
    fc2_b: np.ndarray
    # SwiGLU projections. Allocated for every layer (a small amount of unused
    # memory for a GELU model) so the layer object has one shape; only the names
    # the config selects are returned by ``parameters()``.
    gate_w: np.ndarray | None = None
    up_w: np.ndarray | None = None
    down_w: np.ndarray | None = None


class TransformerLM:
    """A small decoder-only transformer with full numpy backprop."""

    def __init__(self, config: TransformerConfig | None = None, *, seed: int = 0) -> None:
        self.config = config or TransformerConfig()
        self._rng = np.random.default_rng(seed)
        n_embd = self.config.n_embd
        self.token_embedding = self._normal((self.config.vocab_size, n_embd))
        self.position_embedding = self._normal((self.config.block_size, n_embd))
        self.layers = [self._init_layer() for _ in range(self.config.n_layer)]
        self.ln_f_w = np.ones(n_embd)
        self.ln_f_b = np.zeros(n_embd)
        # With weight tying the output head is the transpose of the token
        # embedding, so the vocabulary parameters are shared instead of doubled.
        self._head = None if self.config.tie_weights else self._normal((n_embd, self.config.vocab_size))
        self._cache: dict = {}
        self._training = False

    @property
    def head(self) -> np.ndarray:
        if self.config.tie_weights:
            return self.token_embedding.T
        assert self._head is not None
        return self._head

    def _normal(self, shape: tuple[int, ...]) -> np.ndarray:
        return self._rng.normal(0.0, 0.02, size=shape)

    def _layer_param_names(self) -> tuple[str, ...]:
        """The per-layer parameters in order, so every list stays in one place.

        ``parameters()``, ``save``/``load`` and ``_flatten_grads`` all read this,
        so they cannot drift apart when the architecture variants change which
        parameters exist. The order is the **original** one for the default
        (LayerNorm/GELU) model, so an existing checkpoint's ``p{i}`` arrays load
        into the same parameters they were saved from; a variant drops only the
        parameters it does not have (RMSNorm's biases) or swaps the feed-forward
        pair for SwiGLU's gate/up/down, in place.
        """
        names = ["ln1_w"]
        if self.config.norm != "rms":
            names.append("ln1_b")
        names += ["attn_w", "attn_b", "proj_w", "proj_b", "ln2_w"]
        if self.config.norm != "rms":
            names.append("ln2_b")
        if self.config.activation == "swiglu":
            names += ["gate_w", "up_w", "down_w"]
        else:
            names += ["fc_w", "fc_b", "fc2_w", "fc2_b"]
        return tuple(names)

    def _init_layer(self) -> _LayerParams:
        n_embd = self.config.n_embd
        hidden = self.config.mlp_hidden
        params = _LayerParams(
            attn_w=self._normal((n_embd, 3 * n_embd)),
            attn_b=np.zeros(3 * n_embd),
            proj_w=self._normal((n_embd, n_embd)),
            proj_b=np.zeros(n_embd),
            ln1_w=np.ones(n_embd),
            ln1_b=np.zeros(n_embd),
            ln2_w=np.ones(n_embd),
            ln2_b=np.zeros(n_embd),
            fc_w=self._normal((n_embd, hidden)),
            fc_b=np.zeros(hidden),
            fc2_w=self._normal((hidden, n_embd)),
            fc2_b=np.zeros(n_embd),
            gate_w=self._normal((n_embd, hidden)),
            up_w=self._normal((n_embd, hidden)),
            down_w=self._normal((hidden, n_embd)),
        )
        return params

    def parameters(self) -> list[np.ndarray]:
        params: list[np.ndarray] = [self.token_embedding]
        if self.config.position == "learned":
            params.append(self.position_embedding)
        names = self._layer_param_names()
        for layer in self.layers:
            params.extend(getattr(layer, name) for name in names)
        params.append(self.ln_f_w)
        if self.config.norm != "rms":
            params.append(self.ln_f_b)
        if not self.config.tie_weights:
            params.append(self.head)
        return params

    def train(self, mode: bool = True) -> TransformerLM:
        """Toggle training mode, which enables dropout when the config sets it."""
        self._training = mode
        return self

    def eval(self) -> TransformerLM:
        return self.train(False)

    # --- forward --------------------------------------------------------
    def _norm_forward(self, x, layer, which):
        if self.config.norm == "rms":
            return _rms_norm(x, getattr(layer, f"{which}_w"))
        return _layer_norm(x, getattr(layer, f"{which}_w"), getattr(layer, f"{which}_b"))

    def _norm_backward(self, grad_out, cache, layer, which, n_embd):
        """Return ``(grad_x, grad_weight, grad_bias_or_None)``."""
        if self.config.norm == "rms":
            grad_x, grad_w = _rms_norm_backward(grad_out, cache, getattr(layer, f"{which}_w"), n_embd)
            return grad_x, grad_w, None
        return _layer_norm_backward(grad_out, cache, getattr(layer, f"{which}_w"), n_embd)

    def _mlp_forward(self, normed, layer, p):
        """Feed-forward block; returns the block output and its cache.

        Dropout is applied to the block's hidden activation (the same place the
        original GELU model applied it), so the default architecture is unchanged
        and a dropout model keeps its exact behaviour.
        """
        if self.config.activation == "swiglu":
            gate = normed @ layer.gate_w
            up = normed @ layer.up_w
            hidden, mask = self._dropout(_silu(gate) * up, p)
            return hidden @ layer.down_w, {"gate": gate, "up": up, "hidden": hidden, "mask": mask}
        pre = normed @ layer.fc_w + layer.fc_b
        hidden, mask = self._dropout(_gelu(pre), p)
        return hidden @ layer.fc2_w + layer.fc2_b, {"pre_gelu": pre, "hidden": hidden, "mask": mask}

    def _attention(self, x: np.ndarray, layer: _LayerParams, *, positions=None):
        batch, seq, n_embd = x.shape
        n_head = self.config.n_head
        head_dim = n_embd // n_head
        qkv = x @ layer.attn_w + layer.attn_b
        q, k, v = np.split(qkv, 3, axis=-1)

        def split(a):
            return a.reshape(batch, seq, n_head, head_dim).transpose(0, 2, 1, 3)

        q, k, v = split(q), split(k), split(v)
        if positions is not None:
            cos, sin = _rope_tables(head_dim, positions)
            # Broadcast (seq, head_dim) over batch and heads.
            q = _apply_rope(q, cos[None, None], sin[None, None])
            k = _apply_rope(k, cos[None, None], sin[None, None])
        scores = (q @ k.transpose(0, 1, 3, 2)) / np.sqrt(head_dim)
        mask = np.triu(np.ones((seq, seq), dtype=bool), k=1)
        scores = np.where(mask, -np.inf, scores)
        weights = _softmax_rows(scores)
        attn = weights @ v
        merged = attn.transpose(0, 2, 1, 3).reshape(batch, seq, n_embd)
        out = merged @ layer.proj_w + layer.proj_b
        return out, (q, k, v, weights, merged)

    def forward(self, idx: np.ndarray) -> np.ndarray:
        """Return next-token logits for a batch of token id sequences."""
        idx = np.asarray(idx, dtype=np.int64)
        if idx.ndim == 1:
            idx = idx[None, :]
        _batch, seq = idx.shape
        if seq > self.config.block_size:
            raise ValueError(f"Sequence length {seq} exceeds block_size {self.config.block_size}.")
        cache: dict = {"idx": idx, "seq": seq, "layers": [], "drop_masks": []}
        x = self.token_embedding[idx]
        if self.config.position == "learned":
            x = x + self.position_embedding[:seq][None, :, :]
        p = self.config.dropout if self._training else 0.0
        x, mask = self._dropout(x, p)
        cache["drop_masks"].append(mask)
        positions = np.arange(seq) if self.config.position == "rope" else None
        for layer in self.layers:
            residual = x
            normed1, ln1_cache = self._norm_forward(x, layer, "ln1")
            attn_out, attn_cache = self._attention(normed1, layer, positions=positions)
            attn_out, attn_mask = self._dropout(attn_out, p)
            cache["drop_masks"].append(attn_mask)
            x = residual + attn_out
            residual = x
            normed2, ln2_cache = self._norm_forward(x, layer, "ln2")
            mlp_out, mlp_cache = self._mlp_forward(normed2, layer, p)
            cache["drop_masks"].append(mlp_cache["mask"])
            x = residual + mlp_out
            cache["layers"].append(
                {
                    "ln1": ln1_cache,
                    "normed1": normed1,
                    "attn": attn_cache,
                    "ln2": ln2_cache,
                    "normed2": normed2,
                    "mlp": mlp_cache,
                }
            )
        final_normed, ln_f_cache = self._norm_forward(x, self, "ln_f")
        logits = final_normed @ self.head
        cache["ln_f"] = ln_f_cache
        cache["normed_final"] = final_normed
        self._cache = cache
        return logits

    def forward_incremental(self, idx: np.ndarray, cache: KVCache) -> np.ndarray:
        """Return next-token logits for new tokens, reusing cached keys/values.

        ``idx`` holds only the *new* tokens (the generator feeds one at a time).
        The cache is extended in place and the logits for the new positions are
        returned -- exactly the logits :meth:`forward` would return for the whole
        sequence, but computed in time linear in the sequence length rather than
        quadratic. Inference only: dropout is never applied through it, so it must
        not be called while ``train(True)``.
        """
        idx = np.asarray(idx, dtype=np.int64)
        if idx.ndim == 1:
            idx = idx[None, :]
        _batch, new = idx.shape
        start = cache.length
        total = start + new
        if total > self.config.block_size:
            raise ValueError(
                f"Sequence length {total} exceeds block_size {self.config.block_size}."
            )
        positions = np.arange(start, total)
        x = self.token_embedding[idx]
        if self.config.position == "learned":
            x = x + self.position_embedding[positions][None, :, :]
        if not cache.keys:
            cache.keys = [None] * len(self.layers)  # type: ignore[list-item]
            cache.values = [None] * len(self.layers)  # type: ignore[list-item]
        rope_positions = positions if self.config.position == "rope" else None
        for index, layer in enumerate(self.layers):
            residual = x
            normed1, _ = self._norm_forward(x, layer, "ln1")
            attn_out, keys, values = self._attention_step(
                normed1, layer, cache.keys[index], cache.values[index], rope_positions
            )
            cache.keys[index] = keys
            cache.values[index] = values
            x = residual + attn_out
            residual = x
            normed2, _ = self._norm_forward(x, layer, "ln2")
            mlp_out, _ = self._mlp_forward(normed2, layer, 0.0)
            x = residual + mlp_out
        cache.length = total
        final_normed, _ = self._norm_forward(x, self, "ln_f")
        return final_normed @ self.head

    def _attention_step(self, x, layer, cached_keys, cached_values, positions=None):
        """One attention step over ``x`` appended to the cache's keys/values.

        Returns the attention output for the new positions plus the extended
        key/value tensors, shaped ``(batch, n_head, seq, head_dim)``.
        """
        batch, seq, n_embd = x.shape
        n_head = self.config.n_head
        head_dim = n_embd // n_head
        qkv = x @ layer.attn_w + layer.attn_b
        q, k, v = np.split(qkv, 3, axis=-1)

        def split(a):
            return a.reshape(batch, seq, n_head, head_dim).transpose(0, 2, 1, 3)

        q, k, v = split(q), split(k), split(v)
        if positions is not None:
            cos, sin = _rope_tables(head_dim, positions)
            q = _apply_rope(q, cos[None, None], sin[None, None])
            k = _apply_rope(k, cos[None, None], sin[None, None])
        if cached_keys is not None:
            k = np.concatenate([cached_keys, k], axis=2)
            v = np.concatenate([cached_values, v], axis=2)
        scores = (q @ k.transpose(0, 1, 3, 2)) / np.sqrt(head_dim)
        # A causal mask against the *whole* cached length: the new positions may
        # attend to every earlier position but never to a later one.
        total = k.shape[2]
        query_positions = np.arange(total - seq, total)[:, None]
        key_positions = np.arange(total)[None, :]
        mask = key_positions > query_positions
        scores = np.where(mask, -np.inf, scores)
        weights = _softmax_rows(scores)
        attn = weights @ v
        merged = attn.transpose(0, 2, 1, 3).reshape(batch, seq, n_embd)
        out = merged @ layer.proj_w + layer.proj_b
        return out, k, v

    def _dropout(self, x: np.ndarray, p: float) -> tuple[np.ndarray, np.ndarray | None]:
        """Inverted dropout: scale by ``1/(1-p)`` so evaluation needs no change."""
        if p <= 0.0:
            return x, None
        keep = (self._rng.random(x.shape) >= p).astype(x.dtype) / (1.0 - p)
        return x * keep, keep

    def loss(self, logits: np.ndarray, targets: np.ndarray) -> float:
        """Mean cross-entropy over all positions."""
        targets = np.asarray(targets, dtype=np.int64)
        probs = _softmax_rows(logits)
        flat = probs.reshape(-1, probs.shape[-1])
        rows = np.arange(flat.shape[0])
        picked = flat[rows, targets.reshape(-1)]
        self._cache["probs"] = probs
        return float(-np.log(np.clip(picked, 1e-12, 1.0)).mean())

    def backward(self, targets: np.ndarray) -> list[np.ndarray]:
        """Backpropagate through the cached forward pass; returns parameter grads."""
        cache = self._cache
        probs = cache["probs"]
        seq = cache["seq"]
        batch = probs.shape[0]
        targets = np.asarray(targets, dtype=np.int64)
        grad_logits = probs.copy()
        grad_logits.reshape(-1, grad_logits.shape[-1])[
            np.arange(batch * seq), targets.reshape(-1)
        ] -= 1.0
        grad_logits /= batch * seq

        grads: dict[str, np.ndarray] = {}
        normed_final = cache["normed_final"]
        grads["head"] = normed_final.reshape(-1, normed_final.shape[-1]).T @ grad_logits.reshape(
            -1, grad_logits.shape[-1]
        )
        grad_x, grads["ln_f_w"], grad_ln_f_b = self._norm_backward(
            grad_logits @ self.head.T, cache["ln_f"], self, "ln_f", self.config.n_embd
        )
        if grad_ln_f_b is not None:
            grads["ln_f_b"] = grad_ln_f_b
        masks = cache.get("drop_masks", [])

        for index in reversed(range(len(self.layers))):
            layer = self.layers[index]
            layer_cache = cache["layers"][index]
            if self.config.activation == "swiglu":
                mlp = layer_cache["mlp"]
                gate_w, up_w, down_w = layer.gate_w, layer.up_w, layer.down_w
                assert gate_w is not None and up_w is not None and down_w is not None
                grads[f"down_w_{index}"] = mlp["hidden"].reshape(-1, mlp["hidden"].shape[-1]).T @ grad_x.reshape(-1, grad_x.shape[-1])
                grad_hidden = _apply_mask(grad_x @ down_w.T, mlp["mask"])
                # d(silu(gate) * up)/d(gate) and /d(up).
                grad_gate = grad_hidden * mlp["up"] * _silu_grad(mlp["gate"])
                grad_up = grad_hidden * _silu(mlp["gate"])
                grads[f"gate_w_{index}"] = layer_cache["normed2"].reshape(-1, self.config.n_embd).T @ grad_gate.reshape(-1, grad_gate.shape[-1])
                grads[f"up_w_{index}"] = layer_cache["normed2"].reshape(-1, self.config.n_embd).T @ grad_up.reshape(-1, grad_up.shape[-1])
                grad_normed2 = grad_gate @ gate_w.T + grad_up @ up_w.T
            else:
                mlp = layer_cache["mlp"]
                grads[f"fc2_w_{index}"] = mlp["hidden"].reshape(-1, mlp["hidden"].shape[-1]).T @ grad_x.reshape(-1, grad_x.shape[-1])
                grads[f"fc2_b_{index}"] = grad_x.reshape(-1, grad_x.shape[-1]).sum(axis=0)
                grad_hidden = _apply_mask(grad_x @ layer.fc2_w.T, mlp["mask"])
                # Reuse the forward pass's pre-activation instead of recomputing
                # the ``normed2 @ fc_w + fc_b`` matmul in the backward pass.
                grad_pre_gelu = grad_hidden * _gelu_grad(mlp["pre_gelu"])
                grads[f"fc_w_{index}"] = layer_cache["normed2"].reshape(-1, layer_cache["normed2"].shape[-1]).T @ grad_pre_gelu.reshape(-1, grad_pre_gelu.shape[-1])
                grads[f"fc_b_{index}"] = grad_pre_gelu.reshape(-1, grad_pre_gelu.shape[-1]).sum(axis=0)
                grad_normed2 = grad_pre_gelu @ layer.fc_w.T
            grad_x2, grads[f"ln2_w_{index}"], g_ln2_b = self._norm_backward(
                grad_normed2, layer_cache["ln2"], layer, "ln2", self.config.n_embd
            )
            if g_ln2_b is not None:
                grads[f"ln2_b_{index}"] = g_ln2_b
            grad_x = grad_x + grad_x2

            grad_attn_out = _apply_mask(grad_x, masks[1 + 2 * index])
            grads[f"proj_w_{index}"] = layer_cache["attn"][4].reshape(-1, layer_cache["attn"][4].shape[-1]).T @ grad_attn_out.reshape(-1, grad_attn_out.shape[-1])
            grads[f"proj_b_{index}"] = grad_attn_out.reshape(-1, grad_attn_out.shape[-1]).sum(axis=0)
            grad_merged = grad_attn_out @ layer.proj_w.T
            grad_normed1 = self._attention_backward(grad_merged, layer_cache, layer, grads, index)

            # The first norm sees only the attention path, not the residual skip.
            grad_x_ln1, g_ln1_w, g_ln1_b = self._norm_backward(
                grad_normed1, layer_cache["ln1"], layer, "ln1", self.config.n_embd
            )
            grads[f"ln1_w_{index}"] = g_ln1_w
            if g_ln1_b is not None:
                grads[f"ln1_b_{index}"] = g_ln1_b
            grad_x = grad_x + grad_x_ln1

        grad_x = _apply_mask(grad_x, masks[0] if masks else None)
        grads["token_embedding"] = self._embedding_grad(cache["idx"], grad_x)
        if self.config.position == "learned":
            grads["position_embedding"] = self._position_grad(seq, grad_x)
        # The tied head *is* the token embedding, so its gradient is added to the
        # embedding's before the parameter list is flattened (which omits the head).
        if self.config.tie_weights:
            grads["token_embedding"] = grads["token_embedding"] + grads["head"].T
        return self._flatten_grads(grads)

    def _attention_backward(self, grad_merged, layer_cache, layer, grads, index):
        q, k, v, weights, _merged = layer_cache["attn"]
        batch, n_head, seq, head_dim = q.shape
        n_embd = self.config.n_embd
        grad_attn = grad_merged.reshape(batch, seq, n_head, head_dim).transpose(0, 2, 1, 3)
        grad_weights = grad_attn @ v.transpose(0, 1, 3, 2)
        grad_v = weights.transpose(0, 1, 3, 2) @ grad_attn
        grad_scores = weights * (
            grad_weights - (grad_weights * weights).sum(axis=-1, keepdims=True)
        )
        scale = 1.0 / np.sqrt(head_dim)
        grad_q = (grad_scores @ k) * scale
        grad_k = (grad_scores.transpose(0, 1, 3, 2) @ q) * scale
        if self.config.position == "rope":
            # The forward rotation is orthogonal, so its transpose is the inverse
            # rotation with the sine negated.
            cos, sin = _rope_tables(head_dim, np.arange(seq))
            grad_q = _apply_rope(grad_q, cos[None, None], sin[None, None], inverse=True)
            grad_k = _apply_rope(grad_k, cos[None, None], sin[None, None], inverse=True)

        def merge(a):
            return a.transpose(0, 2, 1, 3).reshape(batch, seq, n_embd)

        grad_qkv = np.concatenate([merge(grad_q), merge(grad_k), merge(grad_v)], axis=-1)
        normed1 = layer_cache["normed1"]
        grads[f"attn_w_{index}"] = normed1.reshape(-1, n_embd).T @ grad_qkv.reshape(-1, grad_qkv.shape[-1])
        grads[f"attn_b_{index}"] = grad_qkv.reshape(-1, grad_qkv.shape[-1]).sum(axis=0)
        return grad_qkv @ layer.attn_w.T

    def _embedding_grad(self, idx: np.ndarray, grad_x: np.ndarray) -> np.ndarray:
        grad = np.zeros_like(self.token_embedding)
        np.add.at(grad, idx.reshape(-1), grad_x.reshape(-1, grad_x.shape[-1]))
        return grad

    def _position_grad(self, seq: int, grad_x: np.ndarray) -> np.ndarray:
        grad = np.zeros_like(self.position_embedding)
        grad[:seq] = grad_x.sum(axis=0)
        return grad

    def _flatten_grads(self, grads: dict[str, np.ndarray]) -> list[np.ndarray]:
        ordered: list[np.ndarray] = [grads["token_embedding"]]
        if self.config.position == "learned":
            ordered.append(grads["position_embedding"])
        for index, _layer in enumerate(self.layers):
            ordered.extend(grads[f"{name}_{index}"] for name in self._layer_param_names())
        ordered.append(grads["ln_f_w"])
        if self.config.norm != "rms":
            ordered.append(grads["ln_f_b"])
        if not self.config.tie_weights:
            ordered.append(grads["head"])
        return ordered

    # --- persistence ----------------------------------------------------
    def save(self, path) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        arrays: dict[str, np.ndarray] = {f"p{i}": p for i, p in enumerate(self.parameters())}
        arrays["config"] = np.array(
            [
                self.config.vocab_size,
                self.config.n_layer,
                self.config.n_head,
                self.config.n_embd,
                self.config.block_size,
                round(self.config.dropout, 8),
                1 if self.config.tie_weights else 0,
            ],
            dtype=np.float64,
        )
        # The architecture variants are stored as strings in their own array so a
        # checkpoint records exactly which primitives produced it. An older
        # checkpoint has no such array and loads as the original architecture.
        arrays["architecture"] = np.array(
            [self.config.norm, self.config.position, self.config.activation]
        )
        # Persist the dropout RNG so a resumed run draws the same masks an
        # uninterrupted run would: without this a dropout model resumes to a
        # *different* trajectory, and "resume" would quietly mean "retrain".
        arrays["rng"] = np.array([json.dumps(self._rng.bit_generator.state)])
        np.savez_compressed(target, **arrays)  # type: ignore[arg-type]

    @classmethod
    def load(cls, path) -> TransformerLM:
        data = np.load(Path(path), allow_pickle=False)
        values = data["config"]
        # ``config`` gained dropout/tie_weights in the release-1 work; older
        # checkpoints have the first five entries only.
        dropout = float(values[5]) if values.shape[0] > 5 else 0.0
        tie = bool(int(values[6])) if values.shape[0] > 6 else False
        # The architecture variants (added with the modern primitives) are stored
        # separately; an older checkpoint has none and loads as the original.
        norm, position, activation = "layer", "learned", "gelu"
        if "architecture" in data.files:
            architecture = [str(value) for value in data["architecture"]]
            if len(architecture) == 3:
                norm, position, activation = architecture
        config = TransformerConfig(
            vocab_size=int(values[0]),
            n_layer=int(values[1]),
            n_head=int(values[2]),
            n_embd=int(values[3]),
            block_size=int(values[4]),
            dropout=dropout,
            tie_weights=tie,
            norm=norm,
            position=position,
            activation=activation,
        )
        model = cls(config)
        for i, param in enumerate(model.parameters()):
            param[...] = data[f"p{i}"]
        if "rng" in data.files:
            model._rng.bit_generator.state = json.loads(str(data["rng"][0]))
        model.eval()
        return model


@dataclass
class TrainReport:
    steps: int
    losses: list[float] = field(default_factory=list)
    val_losses: list[float] = field(default_factory=list)
    learning_rates: list[float] = field(default_factory=list)
    #: The step (1-based, cumulative) at which validation was best, and the loss
    #: there. ``None`` when validation was never run.
    best_val_step: int | None = None
    best_val: float | None = None

    @property
    def initial_loss(self) -> float:
        return self.losses[0] if self.losses else 0.0

    @property
    def final_loss(self) -> float:
        return self.losses[-1] if self.losses else 0.0

    @property
    def best_val_loss(self) -> float:
        if self.best_val is not None:
            return self.best_val
        return min(self.val_losses) if self.val_losses else float("inf")


def _lr_at(step: int, *, steps: int, base_lr: float, warmup: int, schedule: str = "cosine") -> float:
    """Warmup then a decay, the standard schedule for a small transformer.

    ``cosine`` (the default, the original behaviour) decays along a cosine;
    ``linear`` decays in a straight line; ``constant`` holds the base rate after
    warmup. The warmup ramp is identical for all three.
    """
    if warmup > 0 and step <= warmup:
        return base_lr * step / warmup
    progress = (step - warmup) / max(1, steps - warmup)
    progress = min(1.0, max(0.0, progress))
    if schedule == "constant":
        return base_lr
    if schedule == "linear":
        return base_lr * (1.0 - progress)
    return 0.5 * base_lr * (1.0 + np.cos(np.pi * progress))


def _clip_grads(grads: list[np.ndarray], max_norm: float) -> float:
    """Scale gradients so their global norm stays under ``max_norm``."""
    if max_norm <= 0:
        return 0.0
    total = float(np.sqrt(sum(float(np.sum(g * g)) for g in grads)))
    if total > max_norm:
        scale = max_norm / (total + 1e-12)
        for grad in grads:
            grad *= scale
    return total


def _sample_batch(token_stream: list[int], block: int, starts: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    x = np.stack([token_stream[s : s + block] for s in starts])
    y = np.stack([token_stream[s + 1 : s + block + 1] for s in starts])
    return x, y


def _validation_loss(
    model: TransformerLM, stream: list[int], *, batch_size: int, seed: int, batches: int = 8
) -> float:
    """Mean loss over a fixed set of windows with dropout disabled."""
    block = model.config.block_size
    if len(stream) < block + 2:
        return float("nan")
    rng = np.random.default_rng(seed)
    max_start = len(stream) - block - 1
    was_training = model._training
    model.eval()
    losses: list[float] = []
    for _ in range(batches):
        starts = rng.integers(0, max_start, size=batch_size)
        x, y = _sample_batch(stream, block, starts)
        losses.append(model.loss(model.forward(x), y))
    if was_training:
        model.train(True)
    return float(np.mean(losses))


def train_language_model(
    model: TransformerLM,
    token_stream: list[int],
    *,
    steps: int = 300,
    batch_size: int = 8,
    learning_rate: float = 3e-3,
    seed: int = 0,
    log_every: int = 0,
    warmup: int = 0,
    max_grad_norm: float = 0.0,
    val_stream: list[int] | None = None,
    val_every: int = 0,
    val_batches: int = 8,
    optimizer: Adam | None = None,
    start_step: int = 0,
    weight_decay: float = 0.0,
    lr_schedule: str = "cosine",
    restore_best: bool = False,
    accum_steps: int = 1,
    progress: ProgressCallback | None = None,
) -> TrainReport:
    """Train on a flat token stream by next-token prediction.

    Adds the standard engineering a serious run needs: a warmup+cosine learning
    rate, optional gradient clipping, dropout while training, and periodic
    held-out validation with dropout disabled. With ``warmup=0`` and
    ``max_grad_norm=0`` and no ``val_stream`` the behaviour is the original
    constant-rate loop, so existing tests and callers are unaffected.

    ``optimizer`` and ``start_step`` let a caller resume a run: pass the Adam
    optimiser loaded from a saved state and the number of steps already taken,
    and training continues where it stopped instead of restarting the moment
    estimates. The learning-rate schedule then spans ``start_step + steps``, so
    asking for more steps extends the horizon rather than freezing it.

    ``weight_decay`` (AdamW, decoupled) regularises the weights; ``lr_schedule``
    picks the decay shape (``cosine``/``linear``/``constant``). With
    ``restore_best`` the model is left holding the weights from the step whose
    held-out loss was lowest -- a run that overfits late no longer ships its
    worst checkpoint -- and the chosen step and loss are recorded on the report.

    ``accum_steps`` (>= 1) accumulates the gradients of that many micro-batches
    before one optimiser step, so a large *effective* batch fits in the memory of
    a single micro-batch. The accumulated gradient is the exact mean of the
    micro-batches' gradients, so ``accum_steps=k`` with ``batch_size=b`` trains on
    ``b*k`` examples per step -- the same update a single ``b*k`` batch would
    make (up to float rounding). ``accum_steps=1`` is the original behaviour.

    ``progress`` is called with the number of steps completed in this run and
    the requested budget, so a caller can show a live bar while training.
    """
    block = model.config.block_size
    if len(token_stream) < block + 2:
        raise ValueError("Token stream is too short for the configured block size.")
    if start_step < 0:
        raise ValueError("start_step must not be negative.")
    if weight_decay < 0.0:
        raise ValueError("weight_decay must not be negative.")
    if accum_steps < 1:
        raise ValueError("accum_steps must be at least 1.")
    total_steps = start_step + steps
    max_start = len(token_stream) - block - 1
    # Seed once and advance to the resume point, so the batch sequence of a
    # resumed run is exactly the uninterrupted run's sequence. Re-seeding at
    # ``seed + start_step`` drew a *different* batch order after a resume, which
    # made "resume" quietly mean "continue differently".
    rng = np.random.default_rng(seed)
    for _ in range(start_step):
        rng.integers(0, max_start, size=batch_size)
    if optimizer is None:
        optimizer = Adam(
            model.parameters(), lr=learning_rate, weight_decay=weight_decay, decoupled=True
        )
    if len(optimizer.params) != len(model.parameters()):
        raise ValueError("Optimiser state does not match the model's parameters.")
    report = TrainReport(steps=steps)
    model.train(True)
    best_snapshot: list[np.ndarray] | None = None

    def _record_validation(step: int, loss: float) -> None:
        nonlocal best_snapshot
        report.val_losses.append(loss)
        if report.best_val is None or loss < report.best_val:
            report.best_val = loss
            report.best_val_step = step
            if restore_best:
                best_snapshot = [param.copy() for param in model.parameters()]

    for step in range(start_step + 1, total_steps + 1):
        lr = _lr_at(
            step, steps=total_steps, base_lr=learning_rate, warmup=warmup, schedule=lr_schedule
        )
        optimizer.lr = lr
        # Accumulate ``accum_steps`` micro-batches into one optimiser step. The
        # running sum is scaled by 1/accum_steps, so the update matches a single
        # batch of ``batch_size * accum_steps`` (the micro-batches are drawn from
        # the same generator, so with a large enough budget the data is identical
        # and only float rounding differs). ``micro_loss`` reports the last
        # micro-batch's loss, which for the default accum_steps=1 is the step's.
        micro_loss = 0.0
        accumulated: list[np.ndarray] | None = None
        for _ in range(accum_steps):
            starts = rng.integers(0, max_start, size=batch_size)
            x, y = _sample_batch(token_stream, block, starts)
            logits = model.forward(x)
            micro_loss = model.loss(logits, y)
            grads = model.backward(y)
            if accumulated is None:
                accumulated = grads
            else:
                for acc, g in zip(accumulated, grads, strict=True):
                    acc += g
        assert accumulated is not None
        if accum_steps > 1:
            scale = 1.0 / accum_steps
            for acc in accumulated:
                acc *= scale
        _clip_grads(accumulated, max_grad_norm)
        optimizer.step(accumulated)
        report.losses.append(micro_loss)
        report.learning_rates.append(lr)
        if val_stream is not None and val_every and (step - start_step) % val_every == 0:
            _record_validation(
                step,
                _validation_loss(model, val_stream, batch_size=batch_size, seed=seed, batches=val_batches),
            )
        if log_every and (step - start_step) % log_every == 0:
            extra = f"  lr {lr:.2e}"
            if report.val_losses:
                extra += f"  val {report.val_losses[-1]:.4f}"
            print(f"step {step:5d}  loss {micro_loss:.4f}{extra}")
        if progress is not None:
            detail = f"loss {micro_loss:.4f}  lr {lr:.2e}"
            if report.val_losses:
                detail += f"  val {report.val_losses[-1]:.4f}"
            progress(step - start_step, steps, detail=detail)
    model.eval()
    if val_stream is not None:
        _record_validation(
            total_steps,
            _validation_loss(model, val_stream, batch_size=batch_size, seed=seed, batches=val_batches),
        )
    if restore_best and best_snapshot is not None:
        for param, saved in zip(model.parameters(), best_snapshot, strict=True):
            param[...] = saved
    return report


__all__ = ["PRESETS", "TrainReport", "TransformerConfig", "TransformerLM", "train_language_model"]
