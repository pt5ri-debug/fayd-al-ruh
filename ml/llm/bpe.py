"""A byte-level BPE tokenizer trained inside the project.

The byte tokenizer treats every UTF-8 byte as a token, so an Arabic word costs
two or three tokens per letter. A subword tokenizer learns frequent byte pairs
from the corpus and merges them, so common Arabic words, English terms, and
punctuation become single tokens. The vocabulary is still built only from the
corpus and stays self-contained: no external tokenizer file, no download.

The scheme is standard byte-level BPE:

* ids ``0..255`` are the raw bytes, ``256`` is ``<bos>``;
* each learned merge appends a new id, so ``vocab_size = 257 + len(merges)``;
* encoding applies the merges in the order they were learned;
* decoding expands ids back to bytes and UTF-8 decodes them.

A leading ``<bos>`` marks a document start, exactly like the byte tokenizer, so
the two are drop-in compatible.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

BOS_ID = 256
_BASE_VOCAB = 257


class BPETokenizer:
    """Byte-level BPE: learns frequent byte merges and applies them in order."""

    def __init__(self, merges: list[tuple[int, int]] | None = None) -> None:
        self.merges: list[tuple[int, int]] = list(merges or [])
        self._ranks: dict[tuple[int, int], int] = {
            pair: rank for rank, pair in enumerate(self.merges)
        }
        self._vocab: dict[int, bytes] = {i: bytes([i]) for i in range(256)}
        for index, (left, right) in enumerate(self.merges):
            self._vocab[_BASE_VOCAB + index] = self._vocab[left] + self._vocab[right]

    @property
    def vocab_size(self) -> int:
        return _BASE_VOCAB + len(self.merges)

    # --- training -------------------------------------------------------
    @classmethod
    def train(
        cls,
        text: str,
        *,
        vocab_size: int = 1024,
        min_frequency: int = 2,
    ) -> "BPETokenizer":
        """Learn byte merges until ``vocab_size`` is reached or no pair repeats."""
        if vocab_size < _BASE_VOCAB:
            raise ValueError(f"vocab_size must be at least {_BASE_VOCAB}.")
        if min_frequency < 2:
            raise ValueError("min_frequency must be at least 2.")

        sequences: list[list[int]] = [list(chunk.encode("utf-8")) for chunk in _chunks(text)]
        merges: list[tuple[int, int]] = []
        while _BASE_VOCAB + len(merges) < vocab_size:
            pair = _most_frequent_pair(sequences, min_frequency)
            if pair is None:
                break
            merges.append(pair)
            sequences = [_merge(sequence, pair, _BASE_VOCAB + len(merges) - 1) for sequence in sequences]
        return cls(merges)

    # --- encoding -------------------------------------------------------
    def encode(self, text: str, *, bos: bool = True) -> list[int]:
        ids = list(text.encode("utf-8"))
        for rank, pair in enumerate(self.merges):
            ids = _merge(ids, pair, _BASE_VOCAB + rank)
        return [BOS_ID, *ids] if bos else ids

    def decode(self, ids) -> str:
        payload = bytearray()
        for token in ids:
            value = int(token)
            if value == BOS_ID:
                continue
            chunk = self._vocab.get(value)
            if chunk is None:
                if 0 <= value < 256:
                    payload.append(value)
                continue
            payload.extend(chunk)
        return bytes(payload).decode("utf-8", errors="replace")

    # --- persistence ----------------------------------------------------
    def save(self, path) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(
            json.dumps({"merges": [list(pair) for pair in self.merges]}, ensure_ascii=False),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path) -> "BPETokenizer":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        merges = [(int(left), int(right)) for left, right in data["merges"]]
        return cls(merges)


def _chunks(text: str) -> list[str]:
    """Split into words plus their surrounding space, preserving all characters."""
    chunks: list[str] = []
    current = ""
    for character in text:
        if character.isspace():
            if current:
                chunks.append(current)
                current = ""
            if chunks:
                chunks[-1] += character
            else:
                current = character
        else:
            current += character
    if current:
        chunks.append(current)
    return chunks or [text]


def _most_frequent_pair(
    sequences: list[list[int]], min_frequency: int
) -> tuple[int, int] | None:
    counts: Counter[tuple[int, int]] = Counter()
    for sequence in sequences:
        counts.update(zip(sequence, sequence[1:]))
    if not counts:
        return None
    pair, frequency = max(counts.items(), key=lambda item: (item[1], -item[0][0], -item[0][1]))
    return pair if frequency >= min_frequency else None


def _merge(sequence: list[int], pair: tuple[int, int], new_id: int) -> list[int]:
    merged: list[int] = []
    index = 0
    length = len(sequence)
    while index < length:
        if (
            index < length - 1
            and sequence[index] == pair[0]
            and sequence[index + 1] == pair[1]
        ):
            merged.append(new_id)
            index += 2
        else:
            merged.append(sequence[index])
            index += 1
    return merged


__all__ = ["BOS_ID", "BPETokenizer"]
