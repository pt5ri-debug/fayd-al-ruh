import numpy as np

from ml.dataset import Dataset
from ml.models import (
    Adam,
    LinearRegressor,
    MLP,
    SoftmaxClassifier,
    load_model,
    save_model,
    softmax,
)
from ml.trainer import Trainer


def _linearly_separable(seed: int = 0, n: int = 120) -> Dataset:
    rng = np.random.default_rng(seed)
    positives = rng.normal(1.5, 0.5, size=(n // 2, 2))
    negatives = rng.normal(-1.5, 0.5, size=(n // 2, 2))
    features = np.vstack([positives, negatives])
    labels = np.array([1] * (n // 2) + [0] * (n // 2))
    return Dataset(features, labels)


def test_softmax_is_a_distribution():
    probs = softmax(np.array([[1.0, 2.0, 3.0]]))
    assert np.isclose(probs.sum(), 1.0)
    assert np.all(probs >= 0)


def test_adam_rejects_mismatched_gradients():
    param = np.zeros(2)
    optimiser = Adam([param])
    try:
        optimiser.step([np.zeros(3)])
    except ValueError:
        pass
    else:  # pragma: no cover
        raise AssertionError("expected ValueError")


def test_classifier_learns_separable_data():
    dataset = _linearly_separable()
    model = SoftmaxClassifier(2, 2, seed=0)
    report = Trainer(learning_rate=0.5, batch_size=16, max_epochs=100, patience=30).fit(
        model, dataset, dataset
    )

    assert report.final.train_accuracy >= 0.95


def test_mlp_learns_separable_data():
    dataset = _linearly_separable()
    model = MLP(2, 8, 2, seed=0)
    report = Trainer(learning_rate=0.1, batch_size=16, max_epochs=200, patience=40).fit(
        model, dataset, dataset
    )

    assert report.final.train_accuracy >= 0.9
    assert report.final.val_brier < 0.2


def test_linear_regressor_fits_linear_target():
    rng = np.random.default_rng(0)
    features = rng.normal(size=(80, 3))
    targets = features @ np.array([[1.0], [-2.0], [0.5]]) + 0.3
    model = LinearRegressor(3, 1, seed=0)

    initial = model.loss(features, targets)
    optimiser = Adam(model.parameters(), lr=0.2)
    for _ in range(300):
        optimiser.step(model.backward(features, targets))

    assert model.loss(features, targets) < initial * 0.05


def test_model_save_and_load_round_trip(tmp_path):
    model = MLP(4, 6, 3, seed=1)
    path = save_model(model, tmp_path / "model.npz")
    restored = load_model(path)

    rng = np.random.default_rng(0)
    sample = rng.normal(size=(5, 4))
    assert np.allclose(model.forward(sample), restored.forward(sample))


def test_load_model_rejects_unknown_kind(tmp_path):
    path = tmp_path / "bad.npz"
    np.savez(path, kind=np.array("Nope"))
    try:
        load_model(path)
    except ValueError:
        pass
    else:  # pragma: no cover
        raise AssertionError("expected ValueError")
