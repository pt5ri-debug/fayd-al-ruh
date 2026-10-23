"""Honest evaluation for the in-project language model.

Training loss is easy to lower and easy to mislead with: a model trained and
scored on the same windows can memorise them. This module gives the project a
measurement it can trust and record:

* **document-level splits** -- whole sentences are assigned to train, validation,
  or test, so no test sentence is ever seen during training;
* **held-out perplexity** -- the exponential of the mean negative log-likelihood
  over fixed windows, with dropout disabled;
* **mean and standard deviation across seeds**, so a number can be reported with
  its variability rather than as a single lucky run.

Perplexity is the standard language-model yardstick: lower is better, and a
difference only counts when it exceeds the run-to-run spread.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ml.llm.tokenizer import load_tokenizer
from ml.llm.transformer import TransformerLM


@dataclass(frozen=True)
class Split:
    """Whole-document train/validation/test partition."""

    train: tuple[str, ...]
    validation: tuple[str, ...]
    test: tuple[str, ...]


@dataclass(frozen=True)
class PerplexityReport:
    """Held-out perplexity for one trained model on one split."""

    train_ppl: float
    validation_ppl: float
    test_ppl: float
    test_tokens: int
    seed: int
    test_bytes: int = 0

    @property
    def test_bits_per_byte(self) -> float:
        """Cross-tokenizer yardstick: negative log-likelihood per raw byte.

        Perplexity depends on the tokenizer, so it cannot be compared across a
        byte and a BPE model. Bits per byte divides by the number of bytes the
        test text actually is, so the two are directly comparable.
        """
        if self.test_bytes <= 0 or not np.isfinite(self.test_ppl):
            return float("nan")
        return float(np.log(self.test_ppl) / np.log(2.0) * self.test_tokens / self.test_bytes)


@dataclass(frozen=True)
class AggregatePerplexity:
    """Perplexity summarised across seeds as mean and standard deviation."""

    train_ppl_mean: float
    train_ppl_std: float
    validation_ppl_mean: float
    validation_ppl_std: float
    test_ppl_mean: float
    test_ppl_std: float
    seed_count: int

    def as_dict(self) -> dict[str, float]:
        return {
            "train_ppl_mean": round(self.train_ppl_mean, 6),
            "train_ppl_std": round(self.train_ppl_std, 6),
            "validation_ppl_mean": round(self.validation_ppl_mean, 6),
            "validation_ppl_std": round(self.validation_ppl_std, 6),
            "test_ppl_mean": round(self.test_ppl_mean, 6),
            "test_ppl_std": round(self.test_ppl_std, 6),
            "seed_count": self.seed_count,
        }


def document_split(
    documents: Sequence[str],
    *,
    val_fraction: float = 0.1,
    test_fraction: float = 0.1,
    seed: int = 0,
) -> Split:
    """Partition whole documents into train/validation/test.

    Documents are shuffled with a fixed seed and cut by fraction. Whole
    documents are kept intact so a test sentence never appears in training.
    """
    if not 0.0 <= val_fraction < 1.0 or not 0.0 <= test_fraction < 1.0:
        raise ValueError("Split fractions must be in [0, 1).")
    if val_fraction + test_fraction >= 1.0:
        raise ValueError("Validation and test fractions must sum to less than 1.")
    items = [document for document in documents if document.strip()]
    if len(items) < 3:
        raise ValueError("Need at least three documents to split into train/val/test.")
    order = np.random.default_rng(seed).permutation(len(items))
    shuffled = [items[i] for i in order]
    n_test = max(1, round(len(shuffled) * test_fraction))
    n_val = max(1, round(len(shuffled) * val_fraction))
    test = tuple(shuffled[:n_test])
    validation = tuple(shuffled[n_test : n_test + n_val])
    train = tuple(shuffled[n_test + n_val :])
    if not train:
        raise ValueError("Split left no training documents; lower the fractions.")
    return Split(train=train, validation=validation, test=test)


def perplexity(
    model: TransformerLM,
    stream: Sequence[int],
    *,
    batch_size: int = 16,
    batches: int = 32,
    seed: int = 0,
) -> float:
    """Exponential of mean negative log-likelihood over ``batches`` windows."""
    block = model.config.block_size
    if len(stream) < block + 2:
        return float("nan")
    rng = np.random.default_rng(seed)
    max_start = len(stream) - block - 1
    was_training = model._training
    model.eval()
    losses: list[float] = []
    for _ in range(batches):
        starts = rng.integers(0, max_start, size=batch_size)
        x = np.stack([stream[s : s + block] for s in starts])
        y = np.stack([stream[s + 1 : s + block + 1] for s in starts])
        losses.append(model.loss(model.forward(x), y))
    if was_training:
        model.train(True)
    return float(np.exp(np.mean(losses)))


def encode_documents(tokenizer, documents: Sequence[str]) -> list[int]:
    """Concatenate documents into one token stream, each opened with ``<bos>``."""
    stream: list[int] = []
    for document in documents:
        stream.extend(tokenizer.encode(document, bos=True))
    return stream


def evaluate_model(
    model: TransformerLM,
    tokenizer,
    split: Split,
    *,
    train_repeats: int = 8,
    batch_size: int = 16,
    batches: int = 32,
    seed: int = 0,
) -> PerplexityReport:
    """Perplexity on each part of a split for an already-trained model."""
    train_stream = encode_documents(tokenizer, list(split.train) * train_repeats)
    val_stream = encode_documents(tokenizer, split.validation)
    test_stream = encode_documents(tokenizer, split.test)
    test_bytes = sum(len(document.encode("utf-8")) for document in split.test)
    return PerplexityReport(
        train_ppl=perplexity(model, train_stream, batch_size=batch_size, batches=batches, seed=seed),
        validation_ppl=perplexity(model, val_stream, batch_size=batch_size, batches=batches, seed=seed),
        test_ppl=perplexity(model, test_stream, batch_size=batch_size, batches=batches, seed=seed),
        test_tokens=len(test_stream),
        seed=seed,
        test_bytes=test_bytes,
    )


def aggregate(reports: Sequence[PerplexityReport]) -> AggregatePerplexity:
    """Mean and standard deviation of perplexity across seeds."""
    if not reports:
        raise ValueError("No perplexity reports to aggregate.")
    stack = np.array([[r.train_ppl, r.validation_ppl, r.test_ppl] for r in reports], dtype=np.float64)
    means = stack.mean(axis=0)
    stds = stack.std(axis=0)
    return AggregatePerplexity(
        train_ppl_mean=float(means[0]),
        train_ppl_std=float(stds[0]),
        validation_ppl_mean=float(means[1]),
        validation_ppl_std=float(stds[1]),
        test_ppl_mean=float(means[2]),
        test_ppl_std=float(stds[2]),
        seed_count=len(reports),
    )


def evaluate_checkpoint(
    checkpoint: str | Path,
    documents: Sequence[str],
    *,
    tokenizer=None,
    val_fraction: float = 0.1,
    test_fraction: float = 0.1,
    train_repeats: int = 8,
    seed: int = 0,
) -> PerplexityReport:
    """Measure a saved checkpoint against a document split."""
    model = TransformerLM.load(checkpoint)
    if tokenizer is None:
        tokenizer = load_tokenizer(Path(checkpoint).with_suffix(".tokenizer.json"))
    split = document_split(documents, val_fraction=val_fraction, test_fraction=test_fraction, seed=seed)
    return evaluate_model(model, tokenizer, split, train_repeats=train_repeats, seed=seed)


def _main(argv: list[str] | None = None) -> int:
    import argparse

    from ml.llm.corpus import corpus_units

    parser = argparse.ArgumentParser(description="Held-out perplexity for the in-project language model.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)
    report = evaluate_checkpoint(
        args.checkpoint,
        corpus_units(),
        tokenizer=load_tokenizer(args.tokenizer) if args.tokenizer else None,
        seed=args.seed,
    )
    print(
        f"perplexity  train {report.train_ppl:.3f}  "
        f"validation {report.validation_ppl:.3f}  test {report.test_ppl:.3f}  "
        f"(test tokens {report.test_tokens}, bits/byte {report.test_bits_per_byte:.3f})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())


__all__ = [
    "AggregatePerplexity",
    "PerplexityReport",
    "Split",
    "aggregate",
    "document_split",
    "encode_documents",
    "evaluate_checkpoint",
    "evaluate_model",
    "perplexity",
]
