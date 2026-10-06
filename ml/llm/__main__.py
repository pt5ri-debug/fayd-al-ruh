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
from ml.llm.tokenizer import ByteTokenizer
from ml.llm.transformer import TransformerConfig, TransformerLM, train_language_model

CHECKPOINT = checkpoints_dir() / "fayd-llm.npz"


def build_model(*, seed: int = 0, config: TransformerConfig | None = None) -> TransformerLM:
    return TransformerLM(config or TransformerConfig(), seed=seed)


def train_and_save(
    *,
    steps: int = 4000,
    repeats: int = 40,
    structured_repeats: int = 60,
    seed: int = 0,
    checkpoint: Path = CHECKPOINT,
    config: TransformerConfig | None = None,
    log_every: int = 100,
) -> TransformerLM:
    tokenizer = ByteTokenizer()
    stream = tokenizer.encode(
        build_corpus(repeats=repeats, structured_repeats=structured_repeats), bos=False
    )
    model = build_model(seed=seed, config=config)
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
        f"{report.final_loss:.3f}  saved {checkpoint}"
    )
    return model


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Train the in-project language model.")
    parser.add_argument("--steps", type=int, default=4000)
    parser.add_argument("--repeats", type=int, default=40)
    parser.add_argument("--structured-repeats", type=int, default=60)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--checkpoint", type=Path, default=CHECKPOINT)
    parser.add_argument("--generate", type=str, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    args = parser.parse_args(argv)

    model = train_and_save(
        steps=args.steps,
        repeats=args.repeats,
        structured_repeats=args.structured_repeats,
        seed=args.seed,
        checkpoint=args.checkpoint,
    )
    if args.generate is not None:
        generator = LocalGenerator(model)
        print(generator.generate(args.generate, max_new_tokens=args.max_new_tokens, seed=args.seed))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
