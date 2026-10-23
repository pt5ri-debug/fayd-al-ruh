"""CLI for the project's learned distributional semantic space.

    python -m cli embeddings status                 # what is installed/learned
    python -m cli embeddings train                  # (re)build it from the corpus
    python -m cli embeddings similar permeability   # nearest terms by meaning
    python -m cli embeddings compare permeability "flow through rock"
    python -m cli embeddings evaluate               # held-out, honest numbers

The space is *learned from the project's own text* (see :mod:`core.embeddings`),
not a downloaded model: related words end up close because they are used in
similar company. ``status`` and ``evaluate`` state plainly what it can and cannot
do, so the numbers are never a claim the space cannot support.

Exit codes: ``0`` success, ``2`` invalid arguments, ``4`` no space is available
and one could not be built, ``5`` an unexpected failure.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence

from core.paths import semantic_space_path

EXIT_OK = 0
EXIT_USAGE = 2
EXIT_UNAVAILABLE = 4
EXIT_FAILURE = 5


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="embeddings", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    status = sub.add_parser("status", help="Report whether a space is trained and what it holds.")
    status.add_argument("--json", action="store_true", help="Print machine-readable JSON.")

    train = sub.add_parser("train", help="Train (or retrain) the space from the project's corpus.")
    train.add_argument("--dimensions", type=int, default=None, help="Vector dimensions.")
    train.add_argument("--window", type=int, default=None, help="Co-occurrence window in tokens.")
    train.add_argument("--min-count", type=int, default=None, help="Drop terms rarer than this.")
    train.add_argument("--json", action="store_true", help="Print machine-readable JSON.")

    similar = sub.add_parser("similar", help="The terms nearest a word or phrase, by meaning.")
    similar.add_argument("text", nargs="+", help="The word or phrase to place.")
    similar.add_argument("--limit", type=int, default=10, help="How many to show (default 10).")
    similar.add_argument("--json", action="store_true", help="Print machine-readable JSON.")

    compare = sub.add_parser(
        "compare",
        help="Cosine similarity of two words or phrases (quote a phrase).",
    )
    compare.add_argument("left", help="The first word or phrase (quote it if it has spaces).")
    compare.add_argument("right", help="The second word or phrase (quote it if it has spaces).")
    compare.add_argument("--json", action="store_true", help="Print machine-readable JSON.")

    evaluate = sub.add_parser(
        "evaluate", help="Measure the space on relations held out of its training text."
    )
    evaluate.add_argument("--hold-out", type=int, default=None, help="Entries to hold out.")
    evaluate.add_argument("--json", action="store_true", help="Print machine-readable JSON.")

    return parser


def _load_space(*, build: bool):
    from core.embeddings import semantic_space

    return semantic_space(build=build)


def run_status(args: argparse.Namespace) -> int:
    path = semantic_space_path()
    space = _load_space(build=False)
    payload = {
        "trained": space is not None,
        "path": str(path),
        "vocabulary_size": space.vocabulary_size if space else 0,
        "dimensions": space.dimensions if space else 0,
        "documents": space.documents if space else 0,
        "tokens": space.tokens if space else 0,
        "window": space.window if space else 0,
        "min_count": space.min_count if space else 0,
    }
    if args.json:
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return EXIT_OK
    if space is None:
        print("No semantic space has been trained yet.")
        print("Build it with: python -m cli embeddings train")
        print("Until then, retrieval falls back to the hash-based semantic signal.")
        return EXIT_OK
    print(space.describe())
    print(f"cached at {path}")
    print(
        "The space is learned from the project's own text: it places a word near the "
        "words it is used with, so a synonym or a paraphrase can be found with no "
        "shared word. It is not a downloaded model and knows only the vocabulary it "
        "was trained on."
    )
    return EXIT_OK


def run_train(args: argparse.Namespace) -> int:
    from core.embeddings import (
        DEFAULT_DIMENSIONS,
        DEFAULT_MIN_COUNT,
        DEFAULT_WINDOW,
        reset_semantic_cache,
        train_and_save_semantic_space,
    )

    # ``is None`` rather than ``or``: a caller who passes ``--dimensions 0`` wants
    # an error, not the silent substitution of the default.
    dimensions = DEFAULT_DIMENSIONS if args.dimensions is None else args.dimensions
    window = DEFAULT_WINDOW if args.window is None else args.window
    min_count = DEFAULT_MIN_COUNT if args.min_count is None else args.min_count
    reset_semantic_cache()
    space = train_and_save_semantic_space(
        dimensions=dimensions,
        window=window,
        min_count=min_count,
        progress=None if args.json else lambda message: print(message, file=sys.stderr),
    )
    if args.json:
        print(
            json.dumps(
                {
                    "path": str(semantic_space_path()),
                    "vocabulary_size": space.vocabulary_size,
                    "dimensions": space.dimensions,
                    "documents": space.documents,
                    "tokens": space.tokens,
                    "window": space.window,
                    "min_count": space.min_count,
                },
                ensure_ascii=False,
            )
        )
        return EXIT_OK
    print(space.describe())
    return EXIT_OK


def run_similar(args: argparse.Namespace) -> int:
    space = _load_space(build=True)
    if space is None:
        print("No semantic space is available and none could be built.")
        return EXIT_UNAVAILABLE
    text = " ".join(args.text)
    neighbours = space.similar_terms(text, limit=args.limit)
    if args.json:
        print(json.dumps([{"term": term, "score": score} for term, score in neighbours], ensure_ascii=False, indent=2))
        return EXIT_OK
    if not neighbours:
        print(f"No known term in {text!r}: the space has never seen these words.")
        return EXIT_OK
    print(f"Terms nearest {text!r} (by meaning, not spelling):")
    for term, score in neighbours:
        print(f"  {score:.3f}  {term}")
    return EXIT_OK


def run_compare(args: argparse.Namespace) -> int:
    space = _load_space(build=True)
    if space is None:
        print("No semantic space is available and none could be built.")
        return EXIT_UNAVAILABLE
    score = space.similarity(args.left, args.right)
    if args.json:
        print(json.dumps({"left": args.left, "right": args.right, "similarity": score}, ensure_ascii=False, indent=2))
        return EXIT_OK
    print(f"similarity({args.left!r}, {args.right!r}) = {score:.4f}")
    if score == 0.0:
        print("Zero: the space knows none of these words, or they share no company at all.")
    return EXIT_OK


def run_evaluate(args: argparse.Namespace) -> int:
    from core.embeddings import DEFAULT_HELD_OUT, evaluate_held_out

    space, report = evaluate_held_out(
        hold_out=args.hold_out if args.hold_out is not None else DEFAULT_HELD_OUT
    )
    if args.json:
        print(
            json.dumps(
                {
                    "vocabulary_size": space.vocabulary_size,
                    "dimensions": space.dimensions,
                    "pairs_measured": report.pairs_measured,
                    "related_cosine": report.related_cosine,
                    "random_cosine": report.random_cosine,
                    "separation": report.separation,
                    "nearest_accuracy": report.nearest_accuracy,
                    "neighbours": report.neighbours,
                    "learned": report.learned,
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return EXIT_OK
    print("Held-out evaluation (relations removed from the training text):")
    print("  " + space.describe())
    print(f"  {report.summary()}")
    print(f"  learned: {report.learned}")
    return EXIT_OK


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(list(argv) if argv is not None else sys.argv[1:])
    handlers = {
        "status": run_status,
        "train": run_train,
        "similar": run_similar,
        "compare": run_compare,
        "evaluate": run_evaluate,
    }
    handler = handlers.get(args.command)
    if handler is None:
        return EXIT_USAGE
    try:
        return handler(args)
    except ValueError as error:
        print(f"Invalid arguments: {error}")
        return EXIT_USAGE


if __name__ == "__main__":  # pragma: no cover - module entry point
    raise SystemExit(main())
