"""An in-project language model.

A decoder-only transformer written from scratch in numpy, trained inside the
repository on a small bundled corpus, and exposed through the project's
``CognitiveModel`` interface. It needs no external model server.
"""

from __future__ import annotations

from ml.llm.generator import LocalGenerator
from ml.llm.tokenizer import BOS_ID, VOCAB_SIZE, ByteTokenizer
from ml.llm.transformer import TransformerConfig, TransformerLM, train_language_model

__all__ = [
    "BOS_ID",
    "VOCAB_SIZE",
    "ByteTokenizer",
    "LocalGenerator",
    "TransformerConfig",
    "TransformerLM",
    "train_language_model",
]
