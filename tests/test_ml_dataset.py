import numpy as np
from ml.dataset import (
    FeatureHasher,
    ReplayBuffer,
    Transition,
    build_classification_dataset,
    train_val_split,
)


def test_feature_hasher_is_deterministic_and_normalized():
    hasher = FeatureHasher(32)
    a = hasher.encode(["wet_soil", "dry_soil"])
    b = hasher.encode(["dry_soil", "wet_soil"])

    assert np.allclose(a, b)  # order independent
    assert a.shape == (32,)
    assert np.linalg.norm(a) <= 1.0 + 1e-9


def test_feature_hasher_handles_empty_tokens():
    hasher = FeatureHasher(8)
    assert np.allclose(hasher.encode([]), np.zeros(8))
    assert hasher.encode_many([]).shape == (0, 8)


def test_transition_round_trip():
    transition = Transition(("a", "b"), 2, 0.5, ("c",), True)
    restored = Transition.from_dict(transition.to_dict())
    assert restored == transition


def test_replay_buffer_respects_capacity():
    buffer = ReplayBuffer(capacity=3)
    for index in range(5):
        buffer.add(Transition((str(index),), index % 2, 0.0, (), False))

    assert len(buffer) == 3
    assert buffer.items[0].state_tokens == ("2",)


def test_replay_buffer_load_dicts_truncates():
    buffer = ReplayBuffer(capacity=2)
    buffer.load_dicts(
        [Transition((str(i),), 0, 0.0, (), False).to_dict() for i in range(5)]
    )
    assert len(buffer) == 2


def test_build_classification_dataset_labels_by_reward():
    transitions = [
        Transition(("a",), 0, 1.0, (), True),
        Transition(("b",), 1, -0.02, (), False),
    ]
    dataset = build_classification_dataset(transitions, FeatureHasher(16))

    assert len(dataset) == 2
    assert list(dataset.labels) == [1, 0]


def test_train_val_split_partitions_dataset():
    transitions = [Transition((str(i),), 0, float(i % 2), (), False) for i in range(20)]
    dataset = build_classification_dataset(transitions, FeatureHasher(16))

    train, val = train_val_split(dataset, val_fraction=0.25, seed=0)

    assert len(train) + len(val) == len(dataset)
    assert len(val) == 5


def test_replay_buffer_sample_draws_from_memory():
    import random

    buffer = ReplayBuffer(capacity=10)
    for index in range(5):
        buffer.add(Transition(("a",), index, 0.0, ("b",), False))

    batch = buffer.sample(8, rng=random.Random(0))

    assert len(batch) == 8
    assert all(isinstance(item, Transition) for item in batch)


def test_replay_buffer_sample_handles_empty_and_invalid():
    import pytest

    buffer = ReplayBuffer(capacity=10)
    assert buffer.sample(4) == []

    buffer.add(Transition(("a",), 0, 0.0, ("b",), False))
    with pytest.raises(ValueError):
        buffer.sample(0)
