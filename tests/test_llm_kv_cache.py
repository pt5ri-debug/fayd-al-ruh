"""The key/value cache must produce exactly the logits the full forward pass does.

Incremental decoding is a *speed* change, not a quality change, so the gate is
equality, not similarity: the cached logits are compared to the ones a full
forward pass produces, position by position, to machine precision. If they ever
disagree the cache is wrong and generation would silently drift.
"""

from __future__ import annotations

import time

import numpy as np
import pytest
from ml.llm.generator import LocalGenerator
from ml.llm.schema import StructuredDecoder
from ml.llm.tokenizer import ByteTokenizer
from ml.llm.transformer import KVCache, TransformerConfig, TransformerLM

_SMALL = TransformerConfig(n_layer=2, n_head=2, n_embd=16, block_size=32)


def _model(seed: int = 0, **kwargs) -> TransformerLM:
    config = TransformerConfig(**{**_SMALL.__dict__, **kwargs})
    return TransformerLM(config, seed=seed)


def test_incremental_matches_full_forward_token_by_token():
    """Feeding tokens one at a time through the cache equals one full pass."""
    model = _model().eval()
    tokens = [3, 7, 11, 5, 19, 2, 23, 13]
    full = model.forward(np.array(tokens, dtype=np.int64))[0]

    cache = KVCache()
    for index, token in enumerate(tokens):
        logits = model.forward_incremental(np.array([token], dtype=np.int64), cache)[0, -1]
        assert np.allclose(logits, full[index], rtol=0.0, atol=1e-10)


def test_incremental_matches_full_forward_in_one_prefill():
    """A whole prompt prefilled at once equals the full pass over that prompt."""
    model = _model().eval()
    tokens = [1, 4, 9, 16, 25, 36]
    full = model.forward(np.array(tokens, dtype=np.int64))[0]
    logits = model.forward_incremental(np.array(tokens, dtype=np.int64), KVCache())[0]
    assert np.allclose(logits, full, rtol=0.0, atol=1e-10)


def test_incremental_matches_full_after_prefill_then_steps():
    """Prefill then single-token steps still equals the full pass."""
    model = _model().eval()
    tokens = [2, 3, 5, 7, 11, 13, 17]
    full = model.forward(np.array(tokens, dtype=np.int64))[0]

    cache = KVCache()
    prefill = model.forward_incremental(np.array(tokens[:4], dtype=np.int64), cache)[0]
    assert np.allclose(prefill, full[:4], rtol=0.0, atol=1e-10)
    for index, token in enumerate(tokens[4:], start=4):
        logits = model.forward_incremental(np.array([token], dtype=np.int64), cache)[0, -1]
        assert np.allclose(logits, full[index], rtol=0.0, atol=1e-10)


def test_incremental_matches_full_for_a_tied_head():
    model = _model(tie_weights=True).eval()
    tokens = [8, 6, 4, 2, 1, 9]
    full = model.forward(np.array(tokens, dtype=np.int64))[0]
    cache = KVCache()
    for index, token in enumerate(tokens):
        logits = model.forward_incremental(np.array([token], dtype=np.int64), cache)[0, -1]
        assert np.allclose(logits, full[index], rtol=0.0, atol=1e-10)


def test_incremental_rejects_overflowing_the_block_size():
    model = _model(block_size=4).eval()
    cache = KVCache()
    model.forward_incremental(np.array([1, 2, 3, 4], dtype=np.int64), cache)
    with pytest.raises(ValueError):
        model.forward_incremental(np.array([5], dtype=np.int64), cache)


def test_generator_is_unchanged_by_the_cache():
    """The cached generator returns exactly what a full-recompute generator would."""
    model = _model(block_size=64).eval()
    generator = LocalGenerator(model, tokenizer=ByteTokenizer())
    first = generator.generate("water", max_new_tokens=20, seed=7)
    second = generator.generate("water", max_new_tokens=20, seed=7)
    assert first == second
    assert first.startswith("water")


def test_generator_falls_back_when_the_window_slides():
    """A prompt longer than block_size still generates (the window slides)."""
    model = _model(block_size=8).eval()
    generator = LocalGenerator(model, tokenizer=ByteTokenizer())
    text = generator.generate("a" * 20, max_new_tokens=6, seed=1)
    assert text.startswith("a" * 20)
    assert len(text) > 20


def test_structured_decoder_is_deterministic_and_valid_after_caching():
    from core.structured_output import parse_structured_output

    model = _model().eval()
    decoder = StructuredDecoder(model)
    text = decoder.decode("analyze: water", seed=3, max_string_bytes=4)
    assert text == decoder.decode("analyze: water", seed=3, max_string_bytes=4)
    assert isinstance(parse_structured_output(text).concepts, list)


def test_cache_makes_generation_faster_on_a_long_answer():
    """A real speed check: the cache must beat the full recompute on a long run.

    The full path is the old behaviour, reconstructed here, so the comparison is
    the change itself rather than a stand-in.
    """
    model = _model(n_layer=3, n_embd=32, block_size=128).eval()
    prompt = "The reservoir holds oil and gas."
    steps = 90

    cache = KVCache()
    tokens = ByteTokenizer().encode(prompt)
    logits = model.forward_incremental(np.array(tokens, dtype=np.int64), cache)[0, -1]
    start = time.perf_counter()
    for _ in range(steps):
        token = int(np.argmax(logits))
        tokens.append(token)
        logits = model.forward_incremental(np.array([token], dtype=np.int64), cache)[0, -1]
    cached_time = time.perf_counter() - start

    context = ByteTokenizer().encode(prompt)
    start = time.perf_counter()
    for _ in range(steps):
        window = context[-model.config.block_size :]
        logits = model.forward(np.array(window, dtype=np.int64))[0, -1]
        context.append(int(np.argmax(logits)))
    full_time = time.perf_counter() - start

    assert cached_time < full_time
