"""Build the portable bundle and stage the launchers next to it.

    python scripts/build.py            # build dist/fayd-al-ruh/
    python scripts/build.py --clean    # remove build/ and dist/ first

PyInstaller is a build-time tool only; it is intentionally not a runtime
dependency, so the development install stays light.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SPEC = PROJECT_ROOT / "fayd-al-ruh.spec"
LAUNCHERS = ("run.sh", "run.bat")
DIST = PROJECT_ROOT / "dist"
BUNDLE = DIST / "fayd-al-ruh"


def clean() -> None:
    """Remove previous build outputs."""
    for target in (PROJECT_ROOT / "build", DIST):
        if target.exists():
            shutil.rmtree(target)
            print(f"removed {target}")


def build(*, clean_first: bool = False) -> Path:
    """Run PyInstaller and stage the launchers. Returns the bundle path."""
    try:
        import PyInstaller  # noqa: F401  # type: ignore[import-untyped]
    except ImportError as error:  # pragma: no cover - environment dependent
        raise SystemExit(
            "PyInstaller is required to build the portable bundle.\n"
            "Install it with: pip install pyinstaller"
        ) from error

    if clean_first:
        clean()

    subprocess.run(
        [sys.executable, "-m", "PyInstaller", "--noconfirm", str(SPEC)],
        cwd=PROJECT_ROOT,
        check=True,
    )

    for name in LAUNCHERS:
        source = PROJECT_ROOT / name
        if not source.exists():
            continue
        target = BUNDLE / name
        if name.endswith(".bat"):
            # Windows batch files need CRLF regardless of the build machine.
            text = source.read_text(encoding="utf-8").replace("\r\n", "\n")
            target.write_bytes(text.replace("\n", "\r\n").encode("utf-8"))
        else:
            shutil.copy2(source, target)

    return BUNDLE


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clean", action="store_true", help="remove build/ and dist/ first")
    args = parser.parse_args(argv)

    bundle = build(clean_first=args.clean)
    print(f"\nPortable bundle ready: {bundle}")
    print(f"Run it with: {bundle / 'run.sh'}  (or run.bat on Windows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
