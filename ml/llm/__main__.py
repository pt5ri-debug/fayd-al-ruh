"""Train the in-project language model.

    python -m ml.llm                 # train and save the default checkpoint
    python -m ml.llm --steps 800     # train longer
    python -m ml.llm --generate "الماء"   # train, then sample a continuation

The checkpoint is written into ``ml/llm/checkpoints`` and committed, so the
project has a working language model without any external service.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from core.paths import checkpoints_dir
from ml.llm.corpus import build_corpus
from ml.llm.generator import LocalGenerator
from ml.llm.tokenizer import (
    TOKENIZER_KINDS,
    load_tokenizer,
    save_tokenizer,
    train_tokenizer,
)
from ml.llm.transformer import TransformerConfig, TransformerLM, train_language_model

CHECKPOINT = checkpoints_dir() / "fayd-llm.npz"
TOKENIZER = checkpoints_dir() / "fayd-llm.tokenizer.json"


def build_model(*, seed: int = 0, config: TransformerConfig | None = None) -> TransformerLM:
    return TransformerLM(config or TransformerConfig(), seed=seed)


def train_and_save(
    *,
    steps: int = 4000,
    repeats: int = 40,
    structured_repeats: int = 60,
    seed: int = 0,
    checkpoint: Path = CHECKPOINT,
    tokenizer_path: Path | None = None,
    tokenizer_kind: str = "byte",
    vocab_size: int = 1024,
    config: TransformerConfig | None = None,
    log_every: int = 100,
) -> TransformerLM:
    text = build_corpus(repeats=repeats, structured_repeats=structured_repeats)
    tokenizer = train_tokenizer(tokenizer_kind, text, vocab_size=vocab_size)
    tokenizer_path = tokenizer_path or Path(checkpoint).with_suffix(".tokenizer.json")
    save_tokenizer(tokenizer, tokenizer_path)

    stream = tokenizer.encode(text, bos=False)
    model_config = config or TransformerConfig(vocab_size=tokenizer.vocab_size)
    model = build_model(seed=seed, config=model_config)
    report = train_language_model(
        model,
        stream,
        steps=steps,
        batch_size=16,
        learning_rate=3e-3,
        seed=seed,
        log_every=log_every,
    )
    model.save(checkpoint)
    print(
        f"trained {steps} steps  loss {report.initial_loss:.3f} -> "
        f"{report.final_loss:.3f}  tokens {len(stream)}  "
        f"tokenizer {tokenizer_kind}/{tokenizer.vocab_size}  saved {checkpoint}"
    )
    return model


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Train the in-project language model.")
    parser.add_argument("--steps", type=int, default=4000)
    parser.add_argument("--repeats", type=int, default=40)
    parser.add_argument("--structured-repeats", type=int, default=60)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--checkpoint", type=Path, default=CHECKPOINT)
    parser.add_argument("--tokenizer", type=Path, default=None)
    parser.add_argument("--tokenizer-kind", choices=TOKENIZER_KINDS, default="byte")
    parser.add_argument("--vocab-size", type=int, default=1024)
    parser.add_argument("--generate", type=str, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    args = parser.parse_args(argv)

    model = train_and_save(
        steps=args.steps,
        repeats=args.repeats,
        structured_repeats=args.structured_repeats,
        seed=args.seed,
        checkpoint=args.checkpoint,
        tokenizer_path=args.tokenizer,
        tokenizer_kind=args.tokenizer_kind,
        vocab_size=args.vocab_size,
    )
    if args.generate is not None:
        tokenizer = load_tokenizer(
            args.tokenizer or Path(args.checkpoint).with_suffix(".tokenizer.json")
        )
        generator = LocalGenerator(model, tokenizer=tokenizer)
        print(generator.generate(args.generate, max_new_tokens=args.max_new_tokens, seed=args.seed))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
