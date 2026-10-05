"""A byte-level tokenizer for the in-project language model.

Bytes are the vocabulary: no tokenizer training, no external file, and every
input round-trips exactly. A leading ``<bos>`` token marks the start of a
document and is the token the generator seeds with. This keeps the whole
pipeline self-contained, which is the point of having the model in the project.
"""

from __future__ import annotations

import numpy as np

BOS_ID = 256
VOCAB_SIZE = 257


class ByteTokenizer:
    """Encodes text to byte ids and back, with a start-of-document token."""

    vocab_size = VOCAB_SIZE

    def encode(self, text: str, *, bos: bool = True) -> list[int]:
        ids = list(text.encode("utf-8"))
        return [BOS_ID, *ids] if bos else ids

    def decode(self, ids) -> str:
        payload = [int(i) for i in ids if int(i) != BOS_ID]
        return bytes(b for b in payload if 0 <= b < 256).decode("utf-8", errors="replace")


def sample(logits, temperature: float, rng) -> int:
    """Sample a token id from logits (greedy when ``temperature <= 0``)."""
    values = np.asarray(logits, dtype=np.float64)
    if temperature <= 0:
        return int(np.argmax(values))
    values = values / temperature
    values -= values.max()
    probs = np.exp(values)
    probs /= probs.sum()
    return int(rng.choice(len(probs), p=probs))


__all__ = ["BOS_ID", "VOCAB_SIZE", "ByteTokenizer", "sample"]
