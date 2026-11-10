"""A byte-level tokenizer for the in-project language model.

Bytes are the vocabulary: no tokenizer training, no external file, and every
input round-trips exactly. A leading ``<bos>`` token marks the start of a
document and is the token the generator seeds with. This keeps the whole
pipeline self-contained, which is the point of having the model in the project.
"""

from __future__ import annotations

from collections.abc import Sequence

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


def sample(
    logits,
    temperature: float,
    rng,
    *,
    top_k: int = 0,
    top_p: float = 1.0,
    repetition_penalty: float = 1.0,
    recent: Sequence[int] | None = None,
) -> int:
    """Sample a token id from logits.

    With every knob at its default this is the original behaviour: greedy when
    ``temperature <= 0`` and otherwise the plain softmax draw. The optional
    filters are the standard decoding controls a small model needs:

    - ``repetition_penalty`` (1.0 = off) divides the logit of every token in
      ``recent`` that is positive and multiplies it when negative, so a token the
      model just emitted is less likely to be repeated (the CTRL penalty).
    - ``top_k`` (> 0) keeps only the k highest-scoring tokens.
    - ``top_p`` (< 1.0) keeps the smallest set of tokens whose cumulative
      probability reaches ``top_p`` (nucleus sampling), always keeping at least
      one.

    The filters are applied in that order, so the numbers a caller passes mean
    what they say.
    """
    values = np.asarray(logits, dtype=np.float64)
    if repetition_penalty and repetition_penalty != 1.0 and recent:
        values = values.copy()
        seen = {int(token) for token in recent}
        for token in seen:
            if 0 <= token < values.shape[0]:
                if values[token] > 0:
                    values[token] /= repetition_penalty
                else:
                    values[token] *= repetition_penalty
    if temperature <= 0:
        return int(np.argmax(values))
    values = values / temperature
    if top_k and top_k > 0 and top_k < values.shape[0]:
        threshold = np.partition(values, -top_k)[-top_k]
        values = np.where(values < threshold, -np.inf, values)
    values = values - values.max()
    probs = np.exp(values)
    probs /= probs.sum()
    if top_p and 0.0 < top_p < 1.0:
        order = np.argsort(probs)[::-1]
        cumulative = np.cumsum(probs[order])
        cutoff = int(np.searchsorted(cumulative, top_p, side="left")) + 1
        cutoff = max(1, min(cutoff, probs.shape[0]))
        keep = order[:cutoff]
        mask = np.zeros_like(probs)
        mask[keep] = probs[keep]
        probs = mask / mask.sum()
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
    "TOKENIZER_KINDS",
    "VOCAB_SIZE",
    "ByteTokenizer",
    "load_tokenizer",
    "sample",
    "save_tokenizer",
    "train_tokenizer",
]
