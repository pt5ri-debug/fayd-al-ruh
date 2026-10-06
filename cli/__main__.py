"""Entry point for the portable build and ``python -m cli``.

Usage:
    python -m cli                     # interactive numbered menu
    python -m cli ask "your question"   # grounded assistant (cited answers)
    python -m cli voice "your question" # one voice turn
    python -m cli agent               # tool-agent demo in the safety sandbox
    python -m cli learn               # continuous learning + regression gate
    python -m cli paths               # show asset/data locations
    python -m cli demo                # local cognitive demo
    python -m cli ml --episodes 60    # ML pipeline
    python -m cli llm --steps 200     # language-model trainer
    python -m cli petroleum           # petroleum-engineering task training
    python -m cli curriculum          # show the seeded degree curriculum
    python -m cli release             # train everything for a release
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
    run_agent,
    run_ask,
    run_curriculum,
    run_demo,
    run_evaluation,
    run_learn,
    run_llm,
    run_ml,
    run_petroleum,
    run_release,
    run_selfcheck,
    run_voice,
)

_COMMANDS = {
    "ask": run_ask,
    "voice": run_voice,
    "agent": run_agent,
    "learn": run_learn,
    "demo": run_demo,
    "ml": run_ml,
    "llm": run_llm,
    "petroleum": run_petroleum,
    "curriculum": run_curriculum,
    "release": run_release,
    "eval": run_evaluation,
    "evaluation": run_evaluation,
    "selfcheck": run_selfcheck,
}


def _use_utf8_console() -> None:
    """Make stdout/stderr tolerate Arabic on legacy terminals (e.g. Windows cp1252).

    The project prints Arabic labels, but a frozen Windows build inherits a
    code-page encoding that cannot represent them, which would crash the
    portable bundle. Reconfiguring to UTF-8 with a safe fallback keeps output
    readable instead of raising ``UnicodeEncodeError``.
    """
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is None:
            continue
        try:
            reconfigure(encoding="utf-8", errors="replace")
        except (ValueError, OSError):
            pass


def main(argv: list[str] | None = None) -> int:
    _use_utf8_console()
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
