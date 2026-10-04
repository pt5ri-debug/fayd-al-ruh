"""Trainable models: a softmax classifier and a small MLP, with Adam.

Both models expose the same tiny interface (``forward``, ``loss``,
``backward``, ``parameters``) so the trainer can treat them interchangeably.
Parameters live in plain numpy arrays and can be saved/loaded as ``.npz``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np


def softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max(axis=-1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=-1, keepdims=True)


def one_hot(labels: np.ndarray, num_classes: int) -> np.ndarray:
    out = np.zeros((labels.shape[0], num_classes), dtype=np.float64)
    out[np.arange(labels.shape[0]), labels] = 1.0
    return out


def cross_entropy(probs: np.ndarray, labels: np.ndarray) -> float:
    clipped = np.clip(probs[np.arange(labels.shape[0]), labels], 1e-12, 1.0)
    return float(-np.mean(np.log(clipped)))


@dataclass
class AdamState:
    """Per-parameter moment estimates for the Adam optimiser."""

    m: list[np.ndarray] = field(default_factory=list)
    v: list[np.ndarray] = field(default_factory=list)
    t: int = 0


class Adam:
    """Adam optimiser over a list of numpy parameter arrays."""

    def __init__(self, params: list[np.ndarray], lr: float = 0.01, beta1: float = 0.9,
                 beta2: float = 0.999, eps: float = 1e-8, weight_decay: float = 0.0) -> None:
        if lr <= 0:
            raise ValueError("Learning rate must be positive.")
        self.params = params
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.weight_decay = weight_decay
        self.state = AdamState(
            m=[np.zeros_like(p) for p in params],
            v=[np.zeros_like(p) for p in params],
        )

    def step(self, grads: list[np.ndarray]) -> None:
        if len(grads) != len(self.params):
            raise ValueError("Gradient count does not match parameter count.")
        self.state.t += 1
        for i, (param, grad) in enumerate(zip(self.params, grads)):
            g = grad + self.weight_decay * param
            self.state.m[i] = self.beta1 * self.state.m[i] + (1 - self.beta1) * g
            self.state.v[i] = self.beta2 * self.state.v[i] + (1 - self.beta2) * (g * g)
            m_hat = self.state.m[i] / (1 - self.beta1**self.state.t)
            v_hat = self.state.v[i] / (1 - self.beta2**self.state.t)
            param -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)


class SoftmaxClassifier:
    """Multinomial logistic regression trained by gradient descent."""

    def __init__(self, num_features: int, num_classes: int, seed: int = 0) -> None:
        if num_features < 1 or num_classes < 2:
            raise ValueError("Classifier needs at least one feature and two classes.")
        rng = np.random.default_rng(seed)
        self.num_features = num_features
        self.num_classes = num_classes
        self.W = rng.normal(0.0, 0.01, size=(num_features, num_classes))
        self.b = np.zeros(num_classes, dtype=np.float64)

    def parameters(self) -> list[np.ndarray]:
        return [self.W, self.b]

    def forward(self, features: np.ndarray) -> np.ndarray:
        return softmax(features @ self.W + self.b)

    def loss(self, features: np.ndarray, labels: np.ndarray) -> float:
        return cross_entropy(self.forward(features), labels)

    def backward(self, features: np.ndarray, labels: np.ndarray) -> list[np.ndarray]:
        probs = self.forward(features)
        probs[np.arange(labels.shape[0]), labels] -= 1.0
        probs /= labels.shape[0]
        return [features.T @ probs, probs.sum(axis=0)]

    def predict(self, features: np.ndarray) -> np.ndarray:
        return np.argmax(self.forward(features), axis=-1)

    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        return self.forward(features)


class MLP:
    """A one-hidden-layer perceptron with ReLU and softmax output."""

    def __init__(self, num_features: int, hidden: int, num_classes: int, seed: int = 0) -> None:
        if min(num_features, hidden, num_classes) < 1:
            raise ValueError("MLP dimensions must be positive.")
        rng = np.random.default_rng(seed)
        self.num_features = num_features
        self.hidden = hidden
        self.num_classes = num_classes
        self.W1 = rng.normal(0.0, np.sqrt(2.0 / num_features), size=(num_features, hidden))
        self.b1 = np.zeros(hidden, dtype=np.float64)
        self.W2 = rng.normal(0.0, np.sqrt(2.0 / hidden), size=(hidden, num_classes))
        self.b2 = np.zeros(num_classes, dtype=np.float64)

    def parameters(self) -> list[np.ndarray]:
        return [self.W1, self.b1, self.W2, self.b2]

    def _hidden(self, features: np.ndarray) -> np.ndarray:
        return np.maximum(0.0, features @ self.W1 + self.b1)

    def forward(self, features: np.ndarray) -> np.ndarray:
        return softmax(self._hidden(features) @ self.W2 + self.b2)

    def loss(self, features: np.ndarray, labels: np.ndarray) -> float:
        return cross_entropy(self.forward(features), labels)

    def backward(self, features: np.ndarray, labels: np.ndarray) -> list[np.ndarray]:
        hidden = self._hidden(features)
        probs = softmax(hidden @ self.W2 + self.b2)
        probs[np.arange(labels.shape[0]), labels] -= 1.0
        probs /= labels.shape[0]
        gW2 = hidden.T @ probs
        gb2 = probs.sum(axis=0)
        dhidden = (probs @ self.W2.T) * (hidden > 0)
        gW1 = features.T @ dhidden
        gb1 = dhidden.sum(axis=0)
        return [gW1, gb1, gW2, gb2]

    def predict(self, features: np.ndarray) -> np.ndarray:
        return np.argmax(self.forward(features), axis=-1)

    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        return self.forward(features)


class LinearRegressor:
    """Linear predictor trained with mean-squared error (used as a world model)."""

    def __init__(self, num_features: int, num_outputs: int, seed: int = 0) -> None:
        if num_features < 1 or num_outputs < 1:
            raise ValueError("Regressor dimensions must be positive.")
        rng = np.random.default_rng(seed)
        self.num_features = num_features
        self.num_outputs = num_outputs
        self.W = rng.normal(0.0, 0.01, size=(num_features, num_outputs))
        self.b = np.zeros(num_outputs, dtype=np.float64)

    def parameters(self) -> list[np.ndarray]:
        return [self.W, self.b]

    def forward(self, features: np.ndarray) -> np.ndarray:
        return features @ self.W + self.b

    def loss(self, features: np.ndarray, targets: np.ndarray) -> float:
        diff = self.forward(features) - targets
        return float(np.mean(diff * diff))

    def backward(self, features: np.ndarray, targets: np.ndarray) -> list[np.ndarray]:
        diff = self.forward(features) - targets
        scale = 2.0 / features.shape[0]
        return [scale * (features.T @ diff), scale * diff.sum(axis=0)]


Model = SoftmaxClassifier | MLP | LinearRegressor


def save_model(model: Model, path: str | Path) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    arrays: dict[str, Any] = {"kind": np.array(type(model).__name__)}
    for name in ("W", "b", "W1", "b1", "W2", "b2"):
        if hasattr(model, name):
            arrays[name] = getattr(model, name)
    np.savez(target, **arrays)
    return target


def load_model(path: str | Path) -> Model:
    data = np.load(Path(path), allow_pickle=False)
    kind = str(data["kind"])
    if kind == "SoftmaxClassifier":
        classifier = SoftmaxClassifier(int(data["W"].shape[0]), int(data["W"].shape[1]))
        classifier.W = data["W"]
        classifier.b = data["b"]
        return classifier
    if kind == "MLP":
        mlp = MLP(
            int(data["W1"].shape[0]), int(data["W1"].shape[1]), int(data["W2"].shape[1])
        )
        mlp.W1, mlp.b1 = data["W1"], data["b1"]
        mlp.W2, mlp.b2 = data["W2"], data["b2"]
        return mlp
    if kind == "LinearRegressor":
        regressor = LinearRegressor(int(data["W"].shape[0]), int(data["W"].shape[1]))
        regressor.W, regressor.b = data["W"], data["b"]
        return regressor
    raise ValueError(f"Unknown model kind: {kind}")
