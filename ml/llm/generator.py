"""Text generation with the in-project language model.

The generator turns a prompt into a continuation by sampling one byte at a time
from the model. Sampling is seeded, so a generation is reproducible, and greedy
decoding is available for deterministic answers.
"""

from __future__ import annotations

import numpy as np

from ml.llm.tokenizer import BOS_ID, ByteTokenizer, sample
from ml.llm.transformer import KVCache, TransformerLM

#: How many of the most recent tokens the repetition penalty looks back over.
DEFAULT_REPETITION_WINDOW = 64


class LocalGenerator:
    """Samples text from a :class:`TransformerLM`.

    Decoding is incremental: the prompt is prefilled once and each new token
    extends a key/value cache, so a token costs one token's work instead of a full
    forward pass over the whole window (the previous behaviour, which was
    quadratic in the answer length). When the context grows past ``block_size``
    the window must start dropping the earliest tokens, which the cache cannot
    represent, so generation falls back to the full sliding-window forward pass --
    the output is identical either way.
    """

    def __init__(self, model: TransformerLM, tokenizer: ByteTokenizer | None = None) -> None:
        self.model = model
        self.tokenizer = tokenizer or ByteTokenizer()

    def generate(
        self,
        prompt: str,
        *,
        max_new_tokens: int = 64,
        temperature: float = 0.8,
        seed: int | None = None,
        top_k: int = 0,
        top_p: float = 1.0,
        repetition_penalty: float = 1.0,
    ) -> str:
        if max_new_tokens < 1:
            raise ValueError("max_new_tokens must be at least 1.")
        rng = np.random.default_rng(seed) if seed is not None else np.random.default_rng()
        context = self.tokenizer.encode(prompt)
        block = self.model.config.block_size
        produced: list[int] = []
        cache = KVCache()
        cached = len(context) <= block
        logits = None
        if cached and context:
            logits = self.model.forward_incremental(
                np.array(context, dtype=np.int64), cache
            )[0, -1]
        for _ in range(max_new_tokens):
            if logits is None:
                window = context[-block:]
                logits = self.model.forward(np.array(window, dtype=np.int64))[0, -1]
            token = sample(
                logits,
                temperature,
                rng,
                top_k=top_k,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
                recent=context[-DEFAULT_REPETITION_WINDOW:] if repetition_penalty != 1.0 else None,
            )
            if token == BOS_ID:
                break
            context.append(token)
            produced.append(token)
            if cached and len(context) <= block:
                logits = self.model.forward_incremental(
                    np.array([token], dtype=np.int64), cache
                )[0, -1]
            else:
                cached = False
                logits = None
        return prompt + self.tokenizer.decode(produced)


__all__ = ["DEFAULT_REPETITION_WINDOW", "LocalGenerator"]
