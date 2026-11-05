"""CLI for the project's universal linguistic polish and Arabic vocalization.

    python -m cli polish "النص هنا , بدون تنسيق ؟"      # polish + report
    python -m cli polish --tashkeel "النص"                # polish and fully vocalize
    python -m cli polish --tashkeel-mode colloquial "..."  # spoken vowels, no إعراب
    python -m cli polish --quiet "..."                     # the text alone
    python -m cli tashkeel "المسامية في الصخر"             # vocalize only
    python -m cli tashkeel --mode colloquial "..."         # the spoken form
    python -m cli tashkeel --status                        # the saved setting

The polish is the same one every answering character runs through
(:mod:`core.text_polish`): deterministic, presentation-only, idempotent, and
explainable -- it prints the text and the list of changes it made. Vocalization is
off unless ``--tashkeel`` (or the saved preference) turns it on.

Exit codes: ``0`` success, ``2`` invalid arguments, ``5`` an unexpected failure.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence

from core.text_polish import (
    TASHKEEL_FORMAL,
    TASHKEEL_MODES,
    polish_answer,
    resolve_tashkeel,
    tashkeel_text,
)

EXIT_OK = 0
EXIT_USAGE = 2
EXIT_FAILURE = 5


def _read_text(value: str | None, *, stdin: bool) -> str:
    """The text to act on: the argument, or stdin when ``--stdin`` was given."""
    if stdin:
        return sys.stdin.read()
    return value or ""


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="polish", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    polish = sub.add_parser("polish", help="Polish text (typography, optionally vocalize).")
    polish.add_argument("text", nargs="?", help="The text; omit with --stdin.")
    polish.add_argument("--stdin", action="store_true", help="Read the text from standard input.")
    polish.add_argument("--language", choices=("ar", "en"), help="Force the language.")
    polish.add_argument("--tashkeel", action="store_true", help="Fully vocalize Arabic.")
    polish.add_argument("--no-tashkeel", action="store_true", help="Never vocalize (overrides the saved setting).")
    polish.add_argument("--tashkeel-mode", choices=TASHKEEL_MODES, help="formal (default) or colloquial.")
    polish.add_argument("--quiet", action="store_true", help="Print only the polished text.")

    tashkeel = sub.add_parser("tashkeel", help="Fully vocalize Arabic text directly.")
    tashkeel.add_argument("text", nargs="?", help="The text; omit with --stdin.")
    tashkeel.add_argument("--stdin", action="store_true", help="Read the text from standard input.")
    tashkeel.add_argument("--mode", choices=TASHKEEL_MODES, default=TASHKEEL_FORMAL, help="formal (default) or colloquial.")
    tashkeel.add_argument("--status", action="store_true", help="Print the saved setting and exit.")

    return parser


def run_polish(args: argparse.Namespace) -> int:
    text = _read_text(args.text, stdin=args.stdin)
    if not text.strip():
        print("Nothing to polish. Pass text or use --stdin.", file=sys.stderr)
        return EXIT_USAGE
    enabled = None
    if args.tashkeel:
        enabled = True
    if args.no_tashkeel:
        enabled = False
    report = polish_answer(
        text, language=args.language, tashkeel=enabled, tashkeel_mode=args.tashkeel_mode
    )
    print(report.text)
    if not args.quiet:
        print(f"\n[{report.describe()}]", file=sys.stderr)
    return EXIT_OK


def run_tashkeel(args: argparse.Namespace) -> int:
    if args.status:
        enabled, mode = resolve_tashkeel()
        state = "on" if enabled else "off"
        print(f"tashkeel: {state} (mode: {mode})")
        return EXIT_OK
    text = _read_text(args.text, stdin=args.stdin)
    if not text.strip():
        print("Nothing to vocalize. Pass text or use --stdin.", file=sys.stderr)
        return EXIT_USAGE
    print(tashkeel_text(text, mode=args.mode))
    return EXIT_OK


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(list(argv) if argv is not None else sys.argv[1:])
    handlers = {"polish": run_polish, "tashkeel": run_tashkeel}
    handler = handlers.get(args.command)
    if handler is None:
        return EXIT_USAGE
    try:
        return handler(args)
    except ValueError as error:
        print(str(error), file=sys.stderr)
        return EXIT_USAGE
    except Exception as error:
        print(f"Unexpected failure: {error}", file=sys.stderr)
        return EXIT_FAILURE
