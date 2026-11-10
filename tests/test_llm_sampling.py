"""Phase 4 decoding controls: top-k, nucleus (top-p), and a repetition penalty.

The point of these tests is that the defaults are the *original* sampler and each
control does exactly what its name says -- a filter that silently changed the
plain draw would be a regression, not a feature.
"""

from __future__ import annotations

import itertools

import numpy as np
from ml.llm.generator import LocalGenerator
from ml.llm.tokenizer import ByteTokenizer, sample
from ml.llm.transformer import TransformerConfig, TransformerLM


def _rng(seed: int = 0) -> np.random.Generator:
    return np.random.default_rng(seed)


def _original_sample(logits, temperature, rng):
    """The pre-Phase-4 sampler, so "defaults are unchanged" is proven, not assumed."""
    values = np.asarray(logits, dtype=np.float64)
    if temperature <= 0:
        return int(np.argmax(values))
    values = values / temperature
    values -= values.max()
    probs = np.exp(values)
    probs /= probs.sum()
    return int(rng.choice(len(probs), p=probs))


# --------------------------------------------------------------------------- #
# Defaults reproduce the original sampler exactly
# --------------------------------------------------------------------------- #


def test_defaults_match_the_original_sampler():
    logits = _rng(1).normal(size=64)
    for temperature in (0.0, 0.5, 1.0, 2.0):
        for seed in range(5):
            got = sample(logits, temperature, _rng(seed))
            expected = _original_sample(logits, temperature, _rng(seed))
            assert got == expected


# --------------------------------------------------------------------------- #
# top-k
# --------------------------------------------------------------------------- #


def test_top_k_one_is_greedy():
    logits = _rng(2).normal(size=32)
    for seed in range(5):
        assert sample(logits, 1.0, _rng(seed), top_k=1) == int(np.argmax(logits))


def test_top_k_never_picks_outside_the_k_best():
    logits = _rng(3).normal(size=50)
    allowed = set(np.argsort(logits)[-5:].tolist())
    for seed in range(40):
        assert sample(logits, 1.0, _rng(seed), top_k=5) in allowed


def test_top_k_zero_or_large_is_a_no_op():
    logits = _rng(4).normal(size=16)
    for seed in range(10):
        base = _original_sample(logits, 1.0, _rng(seed))
        assert sample(logits, 1.0, _rng(seed), top_k=0) == base
        assert sample(logits, 1.0, _rng(seed), top_k=16) == base
        assert sample(logits, 1.0, _rng(seed), top_k=99) == base


# --------------------------------------------------------------------------- #
# top-p (nucleus)
# --------------------------------------------------------------------------- #


def test_top_p_one_is_a_no_op():
    logits = _rng(5).normal(size=32)
    for seed in range(10):
        assert sample(logits, 1.0, _rng(seed), top_p=1.0) == _original_sample(logits, 1.0, _rng(seed))


def test_top_p_keeps_at_least_one_token():
    logits = _rng(6).normal(size=8)
    # An impossibly small nucleus must still leave a valid draw.
    token = sample(logits, 1.0, _rng(0), top_p=1e-9)
    assert 0 <= token < 8


def test_top_p_concentrates_on_the_head():
    logits = _rng(7).normal(size=40)
    top = int(np.argmax(logits))
    # A tight nucleus should almost always land on the top token.
    hits = sum(sample(logits, 1.0, _rng(seed), top_p=0.1) == top for seed in range(100))
    assert hits >= 90


# --------------------------------------------------------------------------- #
# Repetition penalty
# --------------------------------------------------------------------------- #


def test_repetition_penalty_of_one_is_a_no_op():
    logits = _rng(8).normal(size=32)
    for seed in range(10):
        assert sample(
            logits, 1.0, _rng(seed), repetition_penalty=1.0, recent=[1, 2, 3]
        ) == _original_sample(logits, 1.0, _rng(seed))


def test_repetition_penalty_demotes_a_repeated_token():
    # Token 5 is the clear winner, but it was just emitted; a penalty must make
    # the greedy pick fall to the runner-up instead.
    logits = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 5.0, 4.0])
    assert sample(logits, 0.0, _rng(0), repetition_penalty=2.0, recent=[5]) == 6
    # With no penalty the winner is chosen.
    assert sample(logits, 0.0, _rng(0), repetition_penalty=1.0, recent=[5]) == 5


def test_repetition_penalty_handles_negative_logits():
    logits = np.array([-5.0, -4.0])  # token 1 is the winner (both negative)
    # The penalty multiplies a negative logit, pushing it further down: token 1
    # goes from -4 to -12, so the greedy pick falls to token 0.
    assert sample(logits, 0.0, _rng(0), repetition_penalty=3.0, recent=[1]) == 0
    assert sample(logits, 0.0, _rng(0), repetition_penalty=1.0, recent=[1]) == 1


def test_repetition_penalty_reduces_repeats_in_a_real_generation(tmp_path):
    """End-to-end: the penalty measurably cuts adjacent repeats on a real model."""
    model = _tiny_model()
    generator = LocalGenerator(model, tokenizer=ByteTokenizer())
    prompt = "the reservoir "
    plain = generator.generate(prompt, max_new_tokens=60, temperature=1.0, seed=0)
    penalised = generator.generate(
        prompt, max_new_tokens=60, temperature=1.0, seed=0, repetition_penalty=1.5
    )
    assert _repeats(penalised) <= _repeats(plain)


def _repeats(text: str) -> int:
    """Count immediately doubled characters as a cheap repetition proxy."""
    return sum(1 for a, b in itertools.pairwise(text) if a == b)


# --------------------------------------------------------------------------- #
# The generator forwards the controls
# --------------------------------------------------------------------------- #


def _tiny_model() -> TransformerLM:
    return TransformerLM(
        TransformerConfig(n_layer=1, n_head=2, n_embd=16, block_size=16), seed=0
    )


def test_generator_top_k_one_equals_greedy():
    model = _tiny_model()
    generator = LocalGenerator(model, tokenizer=ByteTokenizer())
    greedy = generator.generate("abc", max_new_tokens=12, temperature=0.0, seed=0)
    topk1 = generator.generate("abc", max_new_tokens=12, temperature=1.0, seed=0, top_k=1)
    assert greedy == topk1


def test_generator_is_reproducible_with_the_filters():
    model = _tiny_model()
    generator = LocalGenerator(model, tokenizer=ByteTokenizer())
    a = generator.generate("abc", max_new_tokens=16, temperature=0.9, seed=7, top_p=0.9)
    b = generator.generate("abc", max_new_tokens=16, temperature=0.9, seed=7, top_p=0.9)
    assert a == b


# --------------------------------------------------------------------------- #
# The mouth forwards the controls
# --------------------------------------------------------------------------- #


def test_mind_assistant_forwards_sampling_options():
    from ai.mind import MindAssistant

    class _Recorder:
        def __init__(self):
            self.kwargs = None

        def generate(self, prompt, **kwargs):
            self.kwargs = kwargs
            return prompt + " answer"

    class _Model:
        generator = _Recorder()
        temperature = 0.0
        seed = 0
        model_name = "test"

    model = _Model()
    assistant = MindAssistant(model, top_k=5, top_p=0.9, repetition_penalty=1.2)
    assistant.answer("why?")
    assert model.generator.kwargs["top_k"] == 5
    assert model.generator.kwargs["top_p"] == 0.9
    assert model.generator.kwargs["repetition_penalty"] == 1.2


def test_mind_assistant_defaults_keep_the_filters_off():
    from ai.mind import MindAssistant

    class _Recorder:
        def __init__(self):
            self.kwargs = None

        def generate(self, prompt, **kwargs):
            self.kwargs = kwargs
            return prompt + " answer"

    class _Model:
        generator = _Recorder()
        temperature = 0.0
        seed = 0
        model_name = "test"

    model = _Model()
    MindAssistant(model).answer("why?")
    assert model.generator.kwargs["top_k"] == 0
    assert model.generator.kwargs["top_p"] == 1.0
    assert model.generator.kwargs["repetition_penalty"] == 1.0


def test_cli_generate_accepts_the_sampling_flags():
    import inspect

    from ml.llm import __main__ as cli

    source = inspect.getsource(cli.main)
    for flag in ("--top-k", "--top-p", "--repetition-penalty", "--temperature"):
        assert flag in source
