"""Training loop with metrics, early stopping, and persistence."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from core.progress import ProgressCallback
from ml.dataset import Dataset
from ml.models import Adam, MLP, SoftmaxClassifier, save_model

Classifier = SoftmaxClassifier | MLP


def accuracy(probs: np.ndarray, labels: np.ndarray) -> float:
    if labels.shape[0] == 0:
        return 0.0
    return float(np.mean(np.argmax(probs, axis=-1) == labels))


def brier_score(probs: np.ndarray, labels: np.ndarray, num_classes: int) -> float:
    """Mean squared error between predicted probabilities and one-hot labels."""
    if labels.shape[0] == 0:
        return 0.0
    onehot = np.zeros_like(probs)
    onehot[np.arange(labels.shape[0]), labels] = 1.0
    return float(np.mean(np.sum((probs - onehot) ** 2, axis=-1)))


@dataclass
class EpochReport:
    epoch: int
    train_loss: float
    val_loss: float
    train_accuracy: float
    val_accuracy: float
    val_brier: float


@dataclass
class TrainingReport:
    epochs: tuple[EpochReport, ...]
    best_epoch: int
    best_val_loss: float
    stopped_early: bool
    history: list[float] = field(default_factory=list)

    @property
    def final(self) -> EpochReport:
        return self.epochs[-1]


@dataclass
class Trainer:
    """Mini-batch gradient-descent trainer with early stopping."""

    learning_rate: float = 0.05
    batch_size: int = 16
    max_epochs: int = 200
    patience: int = 20
    min_delta: float = 1e-4
    weight_decay: float = 1e-4
    seed: int = 0

    def __post_init__(self) -> None:
        if min(self.learning_rate, self.batch_size, self.max_epochs, self.patience) <= 0:
            raise ValueError("Trainer hyperparameters must be positive.")

    def fit(
        self,
        model: Classifier,
        train: Dataset,
        val: Dataset | None = None,
        *,
        progress: ProgressCallback | None = None,
    ) -> TrainingReport:
        if len(train) == 0:
            raise ValueError("Training dataset is empty.")
        val = val if val is not None and len(val) else train
        optimiser = Adam(model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        rng = np.random.default_rng(self.seed)
        epochs: list[EpochReport] = []
        best_val = float("inf")
        best_epoch = 0
        best_params = [p.copy() for p in model.parameters()]
        stale = 0
        stopped_early = False

        for epoch in range(1, self.max_epochs + 1):
            order = rng.permutation(len(train))
            for start in range(0, len(train), self.batch_size):
                idx = order[start : start + self.batch_size]
                grads = model.backward(train.features[idx], train.labels[idx])
                optimiser.step(grads)

            train_probs = model.forward(train.features)
            val_probs = model.forward(val.features)
            report = EpochReport(
                epoch=epoch,
                train_loss=cross_entropy_safe(train_probs, train.labels),
                val_loss=cross_entropy_safe(val_probs, val.labels),
                train_accuracy=accuracy(train_probs, train.labels),
                val_accuracy=accuracy(val_probs, val.labels),
                val_brier=brier_score(val_probs, val.labels, model.num_classes),
            )
            epochs.append(report)
            if progress is not None:
                progress(
                    epoch,
                    self.max_epochs,
                    detail=f"train {report.train_accuracy:.3f}  val {report.val_accuracy:.3f}  "
                    f"loss {report.val_loss:.4f}",
                )

            if report.val_loss < best_val - self.min_delta:
                best_val = report.val_loss
                best_epoch = epoch
                best_params = [p.copy() for p in model.parameters()]
                stale = 0
            else:
                stale += 1
                if stale >= self.patience:
                    stopped_early = True
                    break

        for param, best in zip(model.parameters(), best_params):
            param[...] = best
        return TrainingReport(
            epochs=tuple(epochs),
            best_epoch=best_epoch,
            best_val_loss=best_val,
            stopped_early=stopped_early,
            history=[epoch.val_loss for epoch in epochs],
        )

    def fit_and_save(
        self, model: Classifier, train: Dataset, val: Dataset | None, path: str | Path
    ) -> tuple[TrainingReport, Path]:
        report = self.fit(model, train, val)
        return report, save_model(model, path)


def cross_entropy_safe(probs: np.ndarray, labels: np.ndarray) -> float:
    clipped = np.clip(probs[np.arange(labels.shape[0]), labels], 1e-12, 1.0)
    return float(-np.mean(np.log(clipped)))
