"""Tests for the release-1 language-model work.

These cover the training engineering (weight tying, dropout, the learning-rate
schedule, gradient clipping), the honest evaluation harness (document splits and
perplexity), and the corpus additions (numbered procedures, held-out exclusion).
"""

from __future__ import annotations

import numpy as np
import pytest
from ml.llm.corpus import build_corpus, corpus_units, procedural_sentences
from ml.llm.evaluate import (
    aggregate,
    document_split,
    encode_documents,
    evaluate_model,
    perplexity,
)
from ml.llm.tokenizer import ByteTokenizer
from ml.llm.transformer import (
    PRESETS,
    TransformerConfig,
    TransformerLM,
    _clip_grads,
    _lr_at,
    train_language_model,
)

_SMALL = TransformerConfig(n_layer=1, n_head=2, n_embd=16, block_size=16)


def _stream(length: int = 4000) -> list[int]:
    return list(np.random.default_rng(0).integers(0, 256, size=length))


def _grad_check(config: TransformerConfig, seed: int = 0) -> float:
    """Worst relative error between analytic and numeric gradients."""
    model = TransformerLM(config, seed=seed)
    model.eval()
    x = np.array([[1, 2, 3, 4, 5]])
    y = np.array([[2, 3, 4, 5, 6]])
    model.loss(model.forward(x), y)
    grads = model.backward(y)
    rng = np.random.default_rng(0)
    worst = 0.0
    for param, grad in zip(model.parameters(), grads, strict=False):
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
        worst = max(worst, abs(numeric - analytic) / max(1.0, abs(numeric) + abs(analytic)))
    return worst


def test_presets_are_defined_and_release_one_is_larger():
    assert "small" in PRESETS and "release-1" in PRESETS and "capable" in PRESETS
    small = TransformerLM(TransformerConfig.preset("small")).parameters()
    release = TransformerLM(TransformerConfig.preset("release-1")).parameters()
    # release-1 sets tie_weights, so it lists fewer arrays but more parameters.
    assert sum(p.size for p in release) > sum(p.size for p in small)
    assert TransformerConfig.preset("release-1").block_size == 128


def test_unknown_preset_is_rejected():
    with pytest.raises(ValueError):
        TransformerConfig.preset("enormous")


def test_tied_weights_gradients_are_correct():
    tied = TransformerConfig(n_layer=2, n_head=2, n_embd=16, block_size=8, tie_weights=True)
    assert _grad_check(tied) < 1e-5


def test_untied_weights_gradients_are_correct():
    plain = TransformerConfig(n_layer=2, n_head=2, n_embd=16, block_size=8)
    assert _grad_check(plain) < 1e-5


def test_tied_head_shares_the_token_embedding():
    model = TransformerLM(TransformerConfig(n_layer=1, n_head=2, n_embd=16, block_size=8, tie_weights=True))
    assert np.allclose(model.head, model.token_embedding.T)
    assert np.shares_memory(model.head, model.token_embedding)
    assert not any(param.shape == (16, 257) for param in model.parameters())


def test_dropout_changes_training_forward_but_not_eval():
    config = TransformerConfig(n_layer=1, n_head=2, n_embd=16, block_size=8, dropout=0.3)
    model = TransformerLM(config, seed=1)
    x = np.array([[1, 2, 3, 4]])
    model.train(True)
    assert not np.allclose(model.forward(x), model.forward(x))
    model.eval()
    assert np.allclose(model.forward(x), model.forward(x))


def test_save_and_load_round_trips_new_config_fields(tmp_path):
    config = TransformerConfig(n_layer=1, n_head=2, n_embd=16, block_size=8, dropout=0.2, tie_weights=True)
    model = TransformerLM(config, seed=0)
    path = tmp_path / "m.npz"
    model.save(path)
    restored = TransformerLM.load(path)
    assert restored.config.dropout == pytest.approx(0.2)
    assert restored.config.tie_weights is True
    x = np.array([[1, 2, 3, 4]])
    assert np.allclose(model.forward(x), restored.forward(x))


def test_lr_schedule_warms_up_then_decays():
    rates = [_lr_at(step, steps=100, base_lr=1.0, warmup=10) for step in range(1, 101)]
    assert rates[0] < rates[9]  # warmup rises to base
    assert rates[9] == pytest.approx(1.0)
    assert rates[-1] < 0.05  # cosine decays to ~0
    assert max(rates) <= 1.0 + 1e-12


def test_gradient_clipping_scales_down_large_norms():
    grads = [np.full((4,), 10.0)]
    norm = _clip_grads(grads, 1.0)
    assert norm == pytest.approx(20.0)
    assert np.linalg.norm(grads[0]) == pytest.approx(1.0)


def test_training_with_schedule_and_clipping_reduces_loss():
    model = TransformerLM(_SMALL, seed=0)
    report = train_language_model(
        model,
        _stream(),
        steps=40,
        batch_size=4,
        learning_rate=5e-3,
        seed=0,
        warmup=5,
        max_grad_norm=1.0,
        val_stream=_stream(2000),
        val_every=20,
    )
    assert report.final_loss < report.initial_loss
    assert report.learning_rates[0] < report.learning_rates[4]
    assert report.best_val_loss < float("inf")


def test_document_split_is_disjoint_and_complete():
    docs = [f"sentence number {i}" for i in range(50)]
    split = document_split(docs, val_fraction=0.2, test_fraction=0.2, seed=0)
    train, val, test = set(split.train), set(split.validation), set(split.test)
    assert not (train & val) and not (train & test) and not (val & test)
    assert train | val | test == set(docs)
    assert len(val) > 0 and len(test) > 0


def test_document_split_rejects_impossible_fractions():
    with pytest.raises(ValueError):
        document_split([f"d{i}" for i in range(10)], val_fraction=0.6, test_fraction=0.6, seed=0)


def test_perplexity_is_finite_and_lower_for_better_fit():
    model = TransformerLM(_SMALL, seed=0)
    train = _stream()
    before = perplexity(model, train, batches=4)
    train_language_model(model, train, steps=60, batch_size=4, seed=0)
    after = perplexity(model, train, batches=4)
    assert np.isfinite(before) and np.isfinite(after)
    assert after < before


def test_evaluate_model_reports_all_splits():
    model = TransformerLM(_SMALL, seed=0)
    docs = [f"the reservoir holds oil and gas {i}." for i in range(30)]
    split = document_split(docs, seed=0)
    report = evaluate_model(model, ByteTokenizer(), split, train_repeats=2, batches=4, seed=0)
    for value in (report.train_ppl, report.validation_ppl, report.test_ppl):
        assert np.isfinite(value) and value > 0
    assert report.test_tokens > 0


def test_aggregate_summarises_seeds():
    from ml.llm.evaluate import PerplexityReport

    reports = [PerplexityReport(1.0, 2.0, 3.0, 10, seed=0), PerplexityReport(1.2, 2.4, 3.6, 10, seed=1)]
    agg = aggregate(reports)
    assert agg.seed_count == 2
    assert agg.test_ppl_mean == pytest.approx(3.3)
    assert agg.test_ppl_std > 0


def test_procedural_sentences_number_the_steps():
    sentences = procedural_sentences()
    assert sentences, "the corpus has procedures"
    assert any(sentence.startswith("Step 1 of ") for sentence in sentences)
    assert any("has" in sentence and "steps." in sentence for sentence in sentences)


def test_build_corpus_excludes_held_out_units():
    units = corpus_units()
    held = units[0]
    text = build_corpus(repeats=1, structured_repeats=1, exclude={held})
    assert held not in text
    assert units[-1] in text or units[1] in text


def test_encode_documents_opens_each_with_bos():
    from ml.llm.tokenizer import BOS_ID

    stream = encode_documents(ByteTokenizer(), ["a", "b"])
    assert stream[0] == BOS_ID
    assert BOS_ID in stream[2:]
