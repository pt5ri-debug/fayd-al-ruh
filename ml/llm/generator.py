"""Text generation with the in-project language model.

The generator turns a prompt into a continuation by sampling one byte at a time
from the model. Sampling is seeded, so a generation is reproducible, and greedy
decoding is available for deterministic answers.
"""

from __future__ import annotations

import numpy as np

from ml.llm.tokenizer import BOS_ID, ByteTokenizer, sample
from ml.llm.transformer import TransformerLM


class LocalGenerator:
    """Samples text from a :class:`TransformerLM`."""

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
    ) -> str:
        if max_new_tokens < 1:
            raise ValueError("max_new_tokens must be at least 1.")
        rng = np.random.default_rng(seed) if seed is not None else np.random.default_rng()
        context = self.tokenizer.encode(prompt)
        produced: list[int] = []
        for _ in range(max_new_tokens):
            window = context[-self.model.config.block_size :]
            logits = self.model.forward(np.array(window, dtype=np.int64))[0, -1]
            token = sample(logits, temperature, rng)
            if token == BOS_ID:
                break
            context.append(token)
            produced.append(token)
        return prompt + self.tokenizer.decode(produced)


__all__ = ["LocalGenerator"]
