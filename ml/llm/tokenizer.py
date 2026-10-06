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


TOKENIZER_KINDS = ("byte", "bpe")


def train_tokenizer(kind: str, text: str, *, vocab_size: int = 1024):
    """Build the requested tokenizer kind over ``text``.

    ``byte`` needs no training and ignores ``vocab_size``; ``bpe`` learns merges
    up to ``vocab_size``. Both expose ``encode``/``decode``/``vocab_size``.
    """
    if kind == "byte":
        return ByteTokenizer()
    if kind == "bpe":
        from ml.llm.bpe import BPETokenizer

        return BPETokenizer.train(text, vocab_size=vocab_size)
    raise ValueError(f"Unknown tokenizer kind {kind!r}; expected one of {TOKENIZER_KINDS}.")


def save_tokenizer(tokenizer, path) -> None:
    """Persist a tokenizer so generation can rebuild it from the checkpoint dir.

    Byte tokenizers write a tiny marker (there is nothing to learn); BPE
    tokenizers delegate to their own JSON format.
    """
    from pathlib import Path

    target = Path(path)
    if hasattr(tokenizer, "save"):
        tokenizer.save(target)
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text('{"kind": "byte", "merges": []}', encoding="utf-8")


def load_tokenizer(path):
    """Load the tokenizer saved by :func:`save_tokenizer`, else a byte tokenizer."""
    import json
    from pathlib import Path

    source = Path(path)
    if not source.exists():
        return ByteTokenizer()
    data = json.loads(source.read_text(encoding="utf-8"))
    if data.get("kind", "bpe") == "byte":
        return ByteTokenizer()
    from ml.llm.bpe import BPETokenizer

    merges = [(int(left), int(right)) for left, right in data.get("merges", [])]
    return BPETokenizer(merges)


__all__ = [
    "BOS_ID",
    "VOCAB_SIZE",
    "ByteTokenizer",
    "TOKENIZER_KINDS",
    "load_tokenizer",
    "sample",
    "save_tokenizer",
    "train_tokenizer",
]
