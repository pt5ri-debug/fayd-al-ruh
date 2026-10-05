"""Tests for the in-project language model.

The heavy checkpoint test is skipped when the committed model is absent; the rest
train tiny models on the fly so the suite stays fast.
"""

import numpy as np
import pytest

from ai.local_model import DEFAULT_CHECKPOINT, LocalLanguageModel
from core.model_interface import ModelRequest
from ml.llm.corpus import build_corpus
from ml.llm.generator import LocalGenerator
from ml.llm.tokenizer import BOS_ID, ByteTokenizer
from ml.llm.transformer import TransformerConfig, TransformerLM, train_language_model

_SMALL = TransformerConfig(n_layer=1, n_head=2, n_embd=16, block_size=16)


def _token_stream() -> list[int]:
    return ByteTokenizer().encode(build_corpus(repeats=2), bos=False)


def test_tokenizer_round_trips_arabic_and_english():
    tokenizer = ByteTokenizer()
    for text in ("الماء يجعل التربة رطبة.", "Water makes the soil wet.", "x"):
        ids = tokenizer.encode(text)
        assert ids[0] == BOS_ID
        assert tokenizer.decode(ids) == text


def test_tokenizer_ignores_bos_when_decoding():
    tokenizer = ByteTokenizer()
    ids = tokenizer.encode("hi", bos=False)
    assert BOS_ID not in ids
    assert tokenizer.decode([BOS_ID, *ids]) == "hi"


def test_gradients_match_numerical_estimates():
    model = TransformerLM(_SMALL, seed=0)
    x = np.array([[1, 2, 3, 4, 5]])
    y = np.array([[2, 3, 4, 5, 6]])

    model.loss(model.forward(x), y)
    grads = model.backward(y)
    params = model.parameters()
    rng = np.random.default_rng(0)

    for param, grad in zip(params, grads):
        direction = rng.normal(size=param.shape)
        direction /= np.linalg.norm(direction)
        eps = 1e-5
        original = param.copy()
        param[...] = original + eps * direction
        high = model.loss(model.forward(x), y)
        param[...] = original - eps * direction
        low = model.loss(model.forward(x), y)
        param[...] = original
        numeric = (high - low) / (2 * eps)
        analytic = float(np.sum(grad * direction))
        assert abs(numeric - analytic) <= 1e-3 * max(1.0, abs(numeric) + abs(analytic))


def test_training_reduces_loss():
    model = TransformerLM(_SMALL, seed=0)
    report = train_language_model(
        model, _token_stream(), steps=60, batch_size=4, learning_rate=5e-3, seed=0
    )
    assert report.final_loss < report.initial_loss


def test_save_and_load_preserve_logits(tmp_path):
    model = TransformerLM(_SMALL, seed=0)
    train_language_model(model, _token_stream(), steps=10, batch_size=4, seed=0)
    path = tmp_path / "model.npz"
    model.save(path)

    restored = TransformerLM.load(path)
    x = np.array([[1, 2, 3, 4, 5]])
    assert np.allclose(model.forward(x), restored.forward(x))


def test_generator_is_deterministic_for_a_seed():
    model = TransformerLM(_SMALL, seed=0)
    train_language_model(model, _token_stream(), steps=20, batch_size=4, seed=0)
    generator = LocalGenerator(model)

    first = generator.generate("الماء", max_new_tokens=10, seed=7)
    second = generator.generate("الماء", max_new_tokens=10, seed=7)

    assert first == second
    assert first.startswith("الماء")


def test_local_language_model_serves_the_cognitive_interface(tmp_path):
    model = TransformerLM(_SMALL, seed=0)
    train_language_model(model, _token_stream(), steps=20, batch_size=4, seed=0)
    checkpoint = tmp_path / "llm.npz"
    model.save(checkpoint)

    cognitive_model = LocalLanguageModel(checkpoint, max_new_tokens=8, temperature=0.0)
    response = cognitive_model.generate(
        ModelRequest(inputs=(), system_instruction="", context=("الماء",))
    )

    assert response.model_name.startswith("fayd-local-llm")
    assert response.text


def test_missing_checkpoint_is_reported_clearly(tmp_path):
    with pytest.raises(FileNotFoundError):
        LocalLanguageModel(tmp_path / "absent.npz")


@pytest.mark.skipif(not DEFAULT_CHECKPOINT.exists(), reason="no committed checkpoint")
def test_committed_checkpoint_loads_and_generates():
    model = LocalLanguageModel(temperature=0.0, max_new_tokens=16)
    response = model.generate(ModelRequest(inputs=(), system_instruction="", context=("الماء",)))
    assert response.text
