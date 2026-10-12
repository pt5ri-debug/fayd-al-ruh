"""Install the optional features (PDF, OCR) from the command line.

    python -m cli deps               # show what is installed and what is missing
    python -m cli deps --install ocr # install the OCR packages
    python -m cli deps --install pdf --install ocr

The installer is the same one the window offers (:mod:`core.optional_deps`): it
runs the bundled ``requirements-<feature>.txt`` with the interpreter that is
running this program, which is the command that actually works from a source
checkout. It never claims to install the Tesseract *system* package -- that is
reported separately, because pip cannot install it.
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence

from core import optional_deps


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m cli deps",
        description="Install the optional PDF and OCR features.",
    )
    parser.add_argument(
        "--install",
        action="append",
        choices=sorted(optional_deps.FEATURES),
        default=[],
        metavar="FEATURE",
        help="install a feature's Python packages (pdf or ocr); repeatable.",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="only report what is installed (the default when --install is absent).",
    )
    return parser


def _report() -> int:
    any_missing = False
    for key in sorted(optional_deps.FEATURES):
        feature = optional_deps.feature(key)
        state = optional_deps.status(key)
        marker = "ok" if state.available else "missing"
        print(f"{feature.title:14s} [{marker}] {feature.summary}")
        if not state.available:
            any_missing = True
            print("    " + optional_deps.install_hint(key).replace("\n", "\n    "))
    return 1 if any_missing else 0


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(list(argv) if argv is not None else None)
    if not args.install:
        return _report()
    failed = False
    for key in args.install:
        if optional_deps.is_available(key):
            print(f"{optional_deps.feature(key).title} is already installed.")
            continue
        result = optional_deps.install(key, progress=lambda message: print(message, flush=True))
        print(result.message)
        if not result.ok:
            failed = True
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
