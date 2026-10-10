"""Train the in-project language model.

    python -m ml.llm                 # train and save the default checkpoint
    python -m ml.llm --steps 800     # train longer
    python -m ml.llm --preset release-1 --tokenizer-kind bpe   # the release tier
    python -m ml.llm --generate "water"  # train, then sample a continuation

The checkpoint is written into ``ml/llm/checkpoints`` and committed, so the
project has a working language model without any external service.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from core.paths import checkpoints_dir
from core.progress import ProgressReporter
from ml.llm.corpus import build_corpus, corpus_units
from ml.llm.evaluate import document_split, encode_documents
from ml.llm.generator import LocalGenerator
from ml.llm.schedule import (
    ScheduleState,
    ShardPlan,
    load_schedule,
    save_schedule,
    schedule_path,
    select_shard,
)
from ml.llm.state import (
    load_training_state,
    save_training_state,
    state_path as _state_path,
)
from ml.llm.tokenizer import (
    TOKENIZER_KINDS,
    load_tokenizer,
    save_tokenizer,
    train_tokenizer,
)
from ml.llm.transformer import (
    PRESETS,
    TransformerConfig,
    TransformerLM,
    train_language_model,
)
from ml.models import Adam

CHECKPOINT = checkpoints_dir() / "fayd-llm.npz"
TOKENIZER = checkpoints_dir() / "fayd-llm.tokenizer.json"
# BPE merges are learned from a bounded prefix: enough to capture common words
# without spending minutes scanning the repeated training text.
BPE_SAMPLE_CHARS = 300_000


def build_model(
    *,
    seed: int = 0,
    config: TransformerConfig | None = None,
    preset: str = "small",
    vocab_size: int = 257,
) -> TransformerLM:
    return TransformerLM(config or TransformerConfig.preset(preset, vocab_size=vocab_size), seed=seed)


def _shard_stream_is_too_short(
    train_units: list[str], repeats: int, structured_repeats: int, *, min_bytes: int = 2560
) -> bool:
    """Whether a shard holds too little text to fill even a small training window.

    The byte block size reaches 256, so a shard must carry comfortably more than
    a few windows; below this a shard is a misconfiguration, not a short shard.
    """
    length = sum(len(unit) for unit in train_units) * max(1, repeats)
    length += structured_repeats * 200  # the always-included JSON block
    return length < min_bytes


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
    preset: str = "small",
    log_every: int = 100,
    kb_max_chunks: int = 0,
    kb_repeats: int = 1,
    include_runtime: bool = True,
    learning_rate: float = 3e-3,
    warmup: int | None = None,
    max_grad_norm: float = 1.0,
    val_every: int = 200,
    resume: bool = False,
    state_path: Path | None = None,
    verbose: bool = True,
    progress_every: int = 0,
    shard_count: int = 0,
    schedule_path_arg: Path | None = None,
) -> TransformerLM:
    units = corpus_units()
    split = document_split(units, val_fraction=0.1, test_fraction=0.1, seed=seed)
    held_out = set(split.validation) | set(split.test)
    plan = ShardPlan(shard_count) if shard_count > 0 else None
    schedule_file = schedule_path_arg or schedule_path(checkpoint)
    schedule: ScheduleState | None = None
    shard: list[str] | None = None
    shard_index: int | None = None
    if plan is not None:
        # The schedule advances only when resuming, so a fresh run restarts the
        # sequence just as it restarts the weights: `--resume` is the single
        # switch for both.
        previous = load_schedule(schedule_file) if resume else None
        if previous is not None and not previous.matches(plan, seed):
            raise ValueError(
                f"Schedule {schedule_file} is for {previous.count} shards at seed "
                f"{previous.seed}, but this run asks for {plan.count} at seed {seed}."
            )
        schedule = previous or ScheduleState(count=plan.count, seed=seed)
        shard, shard_index = select_shard(units, plan, schedule, seed=seed)
        train_units = [unit for unit in shard if unit not in held_out]
        if _shard_stream_is_too_short(train_units, repeats, structured_repeats):
            raise ValueError(
                f"Shard {shard_index} of {plan.count} has too few documents to fill a "
                "window; lower --llm-shards."
            )
        if verbose:
            print(
                f"Sharded schedule: training on shard {shard_index + 1}/{plan.count} "
                f"({len(train_units)} documents, {len(schedule.consumed)} consumed).",
                flush=True,
            )
    if verbose:
        print(f"Building training corpus ({len(units)} documents) ...", flush=True)
    text = build_corpus(
        repeats=repeats,
        structured_repeats=structured_repeats,
        kb_max_chunks=kb_max_chunks,
        kb_repeats=kb_repeats,
        include_runtime=include_runtime,
        exclude=held_out,
        include=set(shard) if shard is not None else None,
    )
    # The tokenizer sees distinct training documents only; held-out sentences are
    # kept out of the text it learns merges from as well.
    tokenizer_source = text[:BPE_SAMPLE_CHARS] if tokenizer_kind == "bpe" else text
    tokenizer_path = tokenizer_path or Path(checkpoint).with_suffix(".tokenizer.json")

    # Resuming must not silently change the token index space: start from the
    # saved weights and the tokenizer they were trained with, then keep training.
    model: TransformerLM | None = None
    optimizer: Adam | None = None
    start_step = 0
    if resume:
        if Path(checkpoint).exists():
            model = TransformerLM.load(checkpoint)
            tokenizer = load_tokenizer(tokenizer_path)
            loaded = load_training_state(state_path or _state_path(checkpoint), model)
            if loaded is not None:
                optimizer, start_step = loaded
            print(
                f"resuming from {checkpoint} at step {start_step} "
                f"(tokenizer {tokenizer_kind}/{tokenizer.vocab_size})"
            )
        else:
            print(f"no checkpoint at {checkpoint}; starting a fresh run")

    if model is None:
        if verbose:
            print(
                f"Training {tokenizer_kind} tokenizer (vocab {vocab_size}) ...",
                flush=True,
            )
        tokenizer = train_tokenizer(tokenizer_kind, tokenizer_source, vocab_size=vocab_size)
        save_tokenizer(tokenizer, tokenizer_path)
        stream = tokenizer.encode(text, bos=False)
    else:
        stream = tokenizer.encode(text, bos=False)
    val_stream = encode_documents(tokenizer, split.validation)
    model_config = config or TransformerConfig.preset(preset, vocab_size=tokenizer.vocab_size)
    if model is None:
        model = build_model(seed=seed, config=model_config)
    # Own the optimiser so its moments are real when we save the sidecar: a fresh
    # run must persist the moments it actually learned with, not a zeroed stand-in.
    optimizer = optimizer or Adam(model.parameters())
    warmup = warmup if warmup is not None else max(1, steps // 20)
    progress = (
        ProgressReporter("language model", steps, every=progress_every, enabled=verbose)
        if verbose
        else None
    )
    report = train_language_model(
        model,
        stream,
        steps=steps,
        batch_size=16,
        learning_rate=learning_rate,
        seed=seed,
        log_every=log_every,
        warmup=warmup,
        max_grad_norm=max_grad_norm,
        val_stream=val_stream,
        val_every=val_every,
        optimizer=optimizer,
        start_step=start_step,
        progress=progress,
    )
    model.save(checkpoint)
    total_steps = start_step + steps
    save_training_state(optimizer, total_steps, state_path or _state_path(checkpoint))
    if schedule is not None and shard_index is not None:
        # Record the shard only after the run succeeded, so a crash does not
        # silently consume it.
        schedule.consumed.append(shard_index)
        schedule.units = len(shard) if shard is not None else schedule.units
        save_schedule(schedule, schedule_file)
    params = sum(p.size for p in model.parameters())
    print(
        f"trained {steps} steps  loss {report.initial_loss:.3f} -> "
        f"{report.final_loss:.3f}  val {report.best_val_loss:.3f}  "
        f"params {params}  tokens {len(stream)}  "
        f"tokenizer {tokenizer_kind}/{tokenizer.vocab_size}  "
        f"preset {preset}  saved {checkpoint}"
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
    parser.add_argument("--preset", choices=tuple(PRESETS), default="small")
    parser.add_argument("--learning-rate", type=float, default=3e-3)
    parser.add_argument("--warmup", type=int, default=None)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--val-every", type=int, default=200)
    parser.add_argument(
        "--kb-max-chunks",
        type=int,
        default=0,
        help="Train on a relevance-filtered sample of the bulk knowledge base of this size "
        "(0 disables it; a release uses a positive value).",
    )
    parser.add_argument(
        "--no-runtime-docs",
        action="store_true",
        help="Skip documents written at runtime (internet learning, Jarvis feedback).",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Continue training from the checkpoint at --checkpoint (and its "
        "optimiser-state sidecar) instead of starting from scratch. Falls back to "
        "a fresh run when the checkpoint does not exist.",
    )
    parser.add_argument(
        "--llm-shards",
        type=int,
        default=0,
        help="Sharded, resumable schedule: split the document universe into this "
        "many shards and train one shard per run, recording the consumed shard so a "
        "later --resume reaches new material instead of replaying the same corpus "
        "(0 keeps the whole-corpus behaviour).",
    )
    parser.add_argument("--generate", type=str, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Print only the final summary, not the live progress lines.",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=0,
        help="Print a progress line every N steps (0 picks about twenty lines).",
    )
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
        preset=args.preset,
        learning_rate=args.learning_rate,
        warmup=args.warmup,
        max_grad_norm=args.max_grad_norm,
        val_every=args.val_every,
        kb_max_chunks=args.kb_max_chunks,
        include_runtime=not args.no_runtime_docs,
        resume=args.resume,
        shard_count=args.llm_shards,
        progress_every=args.progress_every,
        verbose=not args.quiet,
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
