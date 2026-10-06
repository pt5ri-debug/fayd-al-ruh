"""Tests for the byte-level BPE tokenizer and the tokenizer factory."""

from __future__ import annotations

import pytest

from ml.llm.bpe import BOS_ID, BPETokenizer
from ml.llm.tokenizer import (
    ByteTokenizer,
    load_tokenizer,
    save_tokenizer,
    train_tokenizer,
)

_TEXT = "water in the reservoir, water in the well. Water flows through porous rock."


def test_bpe_round_trips_arabic_and_english():
    tokenizer = BPETokenizer.train(_TEXT, vocab_size=400)
    ids = tokenizer.encode(_TEXT)
    assert tokenizer.decode(ids) == _TEXT


def test_bpe_marks_document_start_with_bos():
    tokenizer = BPETokenizer.train(_TEXT, vocab_size=300)
    ids = tokenizer.encode("hi")
    assert ids[0] == BOS_ID
    assert tokenizer.decode([BOS_ID, *tokenizer.encode("hi", bos=False)]) == "hi"


def test_bpe_compresses_common_arabic_words():
    tokenizer = BPETokenizer.train(_TEXT * 4, vocab_size=400)
    byte_length = len("reservoir".encode("utf-8"))
    bpe_length = len(tokenizer.encode("reservoir", bos=False))
    assert bpe_length < byte_length


def test_bpe_vocabulary_grows_with_merges():
    tokenizer = BPETokenizer.train(_TEXT * 4, vocab_size=500)
    assert tokenizer.vocab_size == 257 + len(tokenizer.merges)
    assert 0 < len(tokenizer.merges) < 500


def test_bpe_respects_a_small_vocabulary_budget():
    tokenizer = BPETokenizer.train(_TEXT * 4, vocab_size=280)
    assert tokenizer.vocab_size <= 280


def test_bpe_rejects_invalid_configuration():
    with pytest.raises(ValueError):
        BPETokenizer.train(_TEXT, vocab_size=10)
    with pytest.raises(ValueError):
        BPETokenizer.train(_TEXT, min_frequency=1)


def test_bpe_save_and_load_round_trip(tmp_path):
    tokenizer = BPETokenizer.train(_TEXT * 4, vocab_size=400)
    path = tmp_path / "tokenizer.json"
    tokenizer.save(path)

    restored = BPETokenizer.load(path)
    assert restored.merges == tokenizer.merges
    assert restored.encode(_TEXT) == tokenizer.encode(_TEXT)


def test_train_tokenizer_factory_builds_both_kinds():
    byte = train_tokenizer("byte", _TEXT)
    assert isinstance(byte, ByteTokenizer)

    bpe = train_tokenizer("bpe", _TEXT * 4, vocab_size=350)
    assert isinstance(bpe, BPETokenizer)
    assert bpe.decode(bpe.encode(_TEXT)) == _TEXT


def test_train_tokenizer_rejects_unknown_kind():
    with pytest.raises(ValueError):
        train_tokenizer("wordpiece", _TEXT)


def test_save_tokenizer_handles_byte_and_bpe(tmp_path):
    byte = ByteTokenizer()
    byte_path = tmp_path / "byte.json"
    save_tokenizer(byte, byte_path)
    assert isinstance(load_tokenizer(byte_path), ByteTokenizer)

    bpe = BPETokenizer.train(_TEXT * 4, vocab_size=350)
    bpe_path = tmp_path / "bpe.json"
    save_tokenizer(bpe, bpe_path)
    restored = load_tokenizer(bpe_path)
    assert isinstance(restored, BPETokenizer)
    assert restored.merges == bpe.merges


def test_load_tokenizer_falls_back_to_byte_when_missing(tmp_path):
    assert isinstance(load_tokenizer(tmp_path / "absent.json"), ByteTokenizer)
