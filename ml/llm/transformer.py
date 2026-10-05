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

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from ml.llm.tokenizer import VOCAB_SIZE
from ml.models import Adam

_EPS = 1e-5


@dataclass(frozen=True)
class TransformerConfig:
    vocab_size: int = VOCAB_SIZE
    n_layer: int = 3
    n_head: int = 4
    n_embd: int = 96
    block_size: int = 64
    dropout: float = 0.0

    def __post_init__(self) -> None:
        if self.n_embd % self.n_head != 0:
            raise ValueError("n_embd must be divisible by n_head.")
        if min(self.vocab_size, self.n_layer, self.n_head, self.n_embd, self.block_size) < 1:
            raise ValueError("All transformer dimensions must be positive.")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in [0, 1).")


def _gelu(x: np.ndarray) -> np.ndarray:
    c = np.sqrt(2.0 / np.pi)
    return 0.5 * x * (1.0 + np.tanh(c * (x + 0.044715 * x**3)))


def _gelu_grad(x: np.ndarray) -> np.ndarray:
    c = np.sqrt(2.0 / np.pi)
    inner = c * (x + 0.044715 * x**3)
    tanh = np.tanh(inner)
    sech2 = 1.0 - tanh**2
    return 0.5 * (1.0 + tanh) + 0.5 * x * sech2 * c * (1.0 + 3 * 0.044715 * x**2)


def _softmax_rows(x: np.ndarray) -> np.ndarray:
    shifted = x - x.max(axis=-1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=-1, keepdims=True)


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


@dataclass
class _LayerParams:
    ln1_w: np.ndarray
    ln1_b: np.ndarray
    attn_w: np.ndarray
    attn_b: np.ndarray
    proj_w: np.ndarray
    proj_b: np.ndarray
    ln2_w: np.ndarray
    ln2_b: np.ndarray
    fc_w: np.ndarray
    fc_b: np.ndarray
    fc2_w: np.ndarray
    fc2_b: np.ndarray


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
        self.head = self._normal((n_embd, self.config.vocab_size))
        self._cache: dict = {}

    def _normal(self, shape: tuple[int, ...]) -> np.ndarray:
        return self._rng.normal(0.0, 0.02, size=shape)

    def _init_layer(self) -> _LayerParams:
        n_embd = self.config.n_embd
        return _LayerParams(
            ln1_w=np.ones(n_embd),
            ln1_b=np.zeros(n_embd),
            attn_w=self._normal((n_embd, 3 * n_embd)),
            attn_b=np.zeros(3 * n_embd),
            proj_w=self._normal((n_embd, n_embd)),
            proj_b=np.zeros(n_embd),
            ln2_w=np.ones(n_embd),
            ln2_b=np.zeros(n_embd),
            fc_w=self._normal((n_embd, 4 * n_embd)),
            fc_b=np.zeros(4 * n_embd),
            fc2_w=self._normal((4 * n_embd, n_embd)),
            fc2_b=np.zeros(n_embd),
        )

    def parameters(self) -> list[np.ndarray]:
        params: list[np.ndarray] = [self.token_embedding, self.position_embedding]
        for layer in self.layers:
            params.extend(
                [
                    layer.ln1_w, layer.ln1_b, layer.attn_w, layer.attn_b,
                    layer.proj_w, layer.proj_b, layer.ln2_w, layer.ln2_b,
                    layer.fc_w, layer.fc_b, layer.fc2_w, layer.fc2_b,
                ]
            )
        params.extend([self.ln_f_w, self.ln_f_b, self.head])
        return params

    # --- forward --------------------------------------------------------
    def _attention(self, x: np.ndarray, layer: _LayerParams):
        batch, seq, n_embd = x.shape
        n_head = self.config.n_head
        head_dim = n_embd // n_head
        qkv = x @ layer.attn_w + layer.attn_b
        q, k, v = np.split(qkv, 3, axis=-1)
        split = lambda a: a.reshape(batch, seq, n_head, head_dim).transpose(0, 2, 1, 3)
        q, k, v = split(q), split(k), split(v)
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
        batch, seq = idx.shape
        if seq > self.config.block_size:
            raise ValueError(f"Sequence length {seq} exceeds block_size {self.config.block_size}.")
        cache: dict = {"idx": idx, "seq": seq, "layers": []}
        x = self.token_embedding[idx] + self.position_embedding[:seq][None, :, :]
        for layer in self.layers:
            residual = x
            normed1, ln1_cache = _layer_norm(x, layer.ln1_w, layer.ln1_b)
            attn_out, attn_cache = self._attention(normed1, layer)
            x = residual + attn_out
            residual = x
            normed2, ln2_cache = _layer_norm(x, layer.ln2_w, layer.ln2_b)
            hidden = _gelu(normed2 @ layer.fc_w + layer.fc_b)
            fc2_out = hidden @ layer.fc2_w + layer.fc2_b
            x = residual + fc2_out
            cache["layers"].append(
                {
                    "ln1": ln1_cache,
                    "normed1": normed1,
                    "attn": attn_cache,
                    "ln2": ln2_cache,
                    "normed2": normed2,
                    "hidden": hidden,
                }
            )
        final_normed, ln_f_cache = _layer_norm(x, self.ln_f_w, self.ln_f_b)
        logits = final_normed @ self.head
        cache["ln_f"] = ln_f_cache
        cache["normed_final"] = final_normed
        self._cache = cache
        return logits

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
        grad_x, grads["ln_f_w"], grads["ln_f_b"] = _layer_norm_backward(
            grad_logits @ self.head.T, cache["ln_f"], self.ln_f_w, self.config.n_embd
        )

        for index in reversed(range(len(self.layers))):
            layer = self.layers[index]
            layer_cache = cache["layers"][index]
            grads[f"fc2_w_{index}"] = layer_cache["hidden"].reshape(-1, layer_cache["hidden"].shape[-1]).T @ grad_x.reshape(-1, grad_x.shape[-1])
            grads[f"fc2_b_{index}"] = grad_x.reshape(-1, grad_x.shape[-1]).sum(axis=0)
            grad_hidden = grad_x @ layer.fc2_w.T
            grad_pre_gelu = grad_hidden * _gelu_grad(layer_cache["normed2"] @ layer.fc_w + layer.fc_b)
            grads[f"fc_w_{index}"] = layer_cache["normed2"].reshape(-1, layer_cache["normed2"].shape[-1]).T @ grad_pre_gelu.reshape(-1, grad_pre_gelu.shape[-1])
            grads[f"fc_b_{index}"] = grad_pre_gelu.reshape(-1, grad_pre_gelu.shape[-1]).sum(axis=0)
            grad_normed2 = grad_pre_gelu @ layer.fc_w.T
            grad_x2, grads[f"ln2_w_{index}"], grads[f"ln2_b_{index}"] = _layer_norm_backward(
                grad_normed2, layer_cache["ln2"], layer.ln2_w, self.config.n_embd
            )
            grad_x = grad_x + grad_x2

            grad_attn_out = grad_x
            grads[f"proj_w_{index}"] = layer_cache["attn"][4].reshape(-1, layer_cache["attn"][4].shape[-1]).T @ grad_attn_out.reshape(-1, grad_attn_out.shape[-1])
            grads[f"proj_b_{index}"] = grad_attn_out.reshape(-1, grad_attn_out.shape[-1]).sum(axis=0)
            grad_merged = grad_attn_out @ layer.proj_w.T
            grad_normed1 = self._attention_backward(grad_merged, layer_cache, layer, grads, index)

            # Layer norm 1 sees only the attention path, not the residual skip.
            grad_x_ln1, g_ln1_w, g_ln1_b = _layer_norm_backward(
                grad_normed1, layer_cache["ln1"], layer.ln1_w, self.config.n_embd
            )
            grads[f"ln1_w_{index}"] = g_ln1_w
            grads[f"ln1_b_{index}"] = g_ln1_b
            grad_x = grad_x + grad_x_ln1

        grads["token_embedding"] = self._embedding_grad(cache["idx"], grad_x)
        grads["position_embedding"] = self._position_grad(seq, grad_x)
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
        merge = lambda a: a.transpose(0, 2, 1, 3).reshape(batch, seq, n_embd)
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
        ordered: list[np.ndarray] = [grads["token_embedding"], grads["position_embedding"]]
        for index, _layer in enumerate(self.layers):
            ordered.extend(
                [
                    grads[f"ln1_w_{index}"], grads[f"ln1_b_{index}"],
                    grads[f"attn_w_{index}"], grads[f"attn_b_{index}"],
                    grads[f"proj_w_{index}"], grads[f"proj_b_{index}"],
                    grads[f"ln2_w_{index}"], grads[f"ln2_b_{index}"],
                    grads[f"fc_w_{index}"], grads[f"fc_b_{index}"],
                    grads[f"fc2_w_{index}"], grads[f"fc2_b_{index}"],
                ]
            )
        ordered.extend([grads["ln_f_w"], grads["ln_f_b"], grads["head"]])
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
            ],
            dtype=np.int64,
        )
        np.savez_compressed(target, **arrays)  # type: ignore[arg-type]

    @classmethod
    def load(cls, path) -> "TransformerLM":
        data = np.load(Path(path), allow_pickle=False)
        values = data["config"]
        config = TransformerConfig(
            vocab_size=int(values[0]),
            n_layer=int(values[1]),
            n_head=int(values[2]),
            n_embd=int(values[3]),
            block_size=int(values[4]),
        )
        model = cls(config)
        for i, param in enumerate(model.parameters()):
            param[...] = data[f"p{i}"]
        return model


@dataclass
class TrainReport:
    steps: int
    losses: list[float] = field(default_factory=list)

    @property
    def initial_loss(self) -> float:
        return self.losses[0] if self.losses else 0.0

    @property
    def final_loss(self) -> float:
        return self.losses[-1] if self.losses else 0.0


def train_language_model(
    model: TransformerLM,
    token_stream: list[int],
    *,
    steps: int = 300,
    batch_size: int = 8,
    learning_rate: float = 3e-3,
    seed: int = 0,
    log_every: int = 0,
) -> TrainReport:
    """Train on a flat token stream by next-token prediction."""
    block = model.config.block_size
    if len(token_stream) < block + 2:
        raise ValueError("Token stream is too short for the configured block size.")
    rng = np.random.default_rng(seed)
    optimizer = Adam(model.parameters(), lr=learning_rate)
    report = TrainReport(steps=steps)
    max_start = len(token_stream) - block - 1
    for step in range(1, steps + 1):
        starts = rng.integers(0, max_start, size=batch_size)
        x = np.stack([token_stream[s : s + block] for s in starts])
        y = np.stack([token_stream[s + 1 : s + block + 1] for s in starts])
        logits = model.forward(x)
        loss = model.loss(logits, y)
        optimizer.step(model.backward(y))
        report.losses.append(loss)
        if log_every and step % log_every == 0:
            print(f"step {step:5d}  loss {loss:.4f}")
    return report


__all__ = ["TrainReport", "TransformerConfig", "TransformerLM", "train_language_model"]
