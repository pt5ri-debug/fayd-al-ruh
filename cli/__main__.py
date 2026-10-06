"""Entry point for the portable build and ``python -m cli``.

Usage:
    python -m cli                     # interactive numbered menu
    python -m cli paths               # show asset/data locations
    python -m cli demo                # local cognitive demo
    python -m cli ml --episodes 60    # ML pipeline
    python -m cli llm --steps 200     # language-model trainer
    python -m cli eval                # evaluation suite

The commands mirror the modules developers already use, so nothing about the
in-repo workflow changes.
"""

from __future__ import annotations

import sys

from cli.menu import (
    describe_paths,
    menu_loop,
    missing_dependency_message,
    run_demo,
    run_evaluation,
    run_llm,
    run_ml,
    run_selfcheck,
)

_COMMANDS = {
    "demo": run_demo,
    "ml": run_ml,
    "llm": run_llm,
    "eval": run_evaluation,
    "evaluation": run_evaluation,
    "selfcheck": run_selfcheck,
}


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if not args:
        return menu_loop()
    command, *rest = args
    if command in {"paths", "info"}:
        print(describe_paths())
        return 0
    if command in {"-h", "--help", "help"}:
        print(__doc__)
        return 0
    action = _COMMANDS.get(command)
    if action is None:
        print(f"Unknown command: {command!r}\n")
        print(__doc__)
        return 1
    try:
        return action(rest)
    except ModuleNotFoundError as error:
        print(missing_dependency_message(error))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
