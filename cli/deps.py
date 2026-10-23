"""Install the optional features (PDF, OCR, learned image) from the command line.

    python -m cli deps               # show what is installed and what is missing
    python -m cli deps --install ocr # install the OCR packages
    python -m cli deps --install pdf --install ocr

The installer is the same one the window offers (:mod:`core.optional_deps`): it
runs the bundled ``requirements-<feature>.txt`` with the interpreter that is
running this program, which is the command that actually works from a source
checkout. It never claims to install the Tesseract *system* package -- that is
reported separately, because pip cannot install it. The learned image backend's
model weights are a similar separate step (a one-time download).
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence

from core import optional_deps


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m cli deps",
        description="Install the optional PDF, OCR, and learned-image features.",
    )
    parser.add_argument(
        "--install",
        action="append",
        choices=[*sorted(optional_deps.FEATURES), "speech"],
        default=[],
        metavar="FEATURE",
        help="install a feature's Python packages (pdf, ocr, image); "
        "use 'speech' for the espeak-ng system engine; repeatable.",
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
            if key == optional_deps.FEATURE_OCR and state.python_ok and not state.system_ok:
                # The packages are present; the *engine* is what is missing, and
                # pip cannot install it. Show the real system-package step.
                from core.pdf_ocr import ocr_install_plan

                print("    " + ocr_install_plan().hint)
    any_missing = _report_speech(any_missing)
    return 1 if any_missing else 0


def _report_speech(any_missing: bool) -> bool:
    """Report the speech engine too: it is a *system* package, not a pip extra."""
    from core.speech import speech_available, speech_install_plan

    ok = speech_available()
    marker = "ok" if ok else "missing"
    print(f"{'Speech engine':14s} [{marker}] Read chat text aloud (text to speech).")
    if not ok:
        print("    " + speech_install_plan().hint)
        print("    Or run `python -m cli speak --install-engine`.")
        return True
    return any_missing


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(list(argv) if argv is not None else None)
    if not args.install:
        return _report()
    failed = False
    for key in args.install:
        if key == "speech":
            failed = _install_speech(failed)
            continue
        if optional_deps.is_available(key):
            print(f"{optional_deps.feature(key).title} is already installed.")
            continue
        result = optional_deps.install(key, progress=lambda message: print(message, flush=True))
        print(result.message)
        if not result.ok:
            failed = True
        # OCR has a *system* half too: pip installs the packages, but the
        # Tesseract engine is an OS package. Offer the real next step here, the
        # same way `--install speech` does for espeak-ng.
        if key == optional_deps.FEATURE_OCR and result.ok:
            failed = _install_ocr_engine(failed)
    return 1 if failed else 0


def _install_ocr_engine(failed: bool) -> bool:
    """Install the Tesseract engine (a system package) via the platform package manager."""
    from core.pdf_ocr import (
        install_ocr_engine,
        missing_ocr_python_packages,
        ocr_available,
        ocr_install_plan,
    )

    # The engine is only worth installing once the Python packages are actually
    # importable: without them the app cannot even ask Tesseract for its version,
    # so a package manager would be run for nothing. (In CI the OCR extra is not
    # installed, so this keeps `--install ocr` honest and exits 0 there.)
    if missing_ocr_python_packages():
        return failed
    if ocr_available():
        print("The Tesseract OCR engine is already installed.")
        return failed
    plan = ocr_install_plan()
    if plan.can_install:
        print(f"Installing the Tesseract OCR engine: {' '.join(plan.command)}")
    result = install_ocr_engine()
    print(result.message)
    return failed or not result.ok


def _install_speech(failed: bool) -> bool:
    """Install the speech engine (a system package) via the platform package manager."""
    from core.speech import install_speech_engine, speech_available, speech_install_plan

    if speech_available():
        print("The speech engine is already installed.")
        return failed
    plan = speech_install_plan()
    if plan.can_install:
        print(f"Installing a speech engine: {' '.join(plan.command)}")
    result = install_speech_engine()
    print(result.message)
    return failed or not result.ok


if __name__ == "__main__":
    raise SystemExit(main())
