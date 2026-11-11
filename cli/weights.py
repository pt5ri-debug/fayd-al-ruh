"""The committed weights pack: what the project ships, and how to continue it.

    python -m cli weights status                     # what the pack offers, verified
    python -m cli weights list                        # the entries, one per line
    python -m cli weights verify [KEY]                # check a file against the pack
    python -m cli weights install [KEY ...]           # install into the trainer's folder
    python -m cli weights install --all --force       # every entry, replacing with a backup
    python -m cli weights plan KEY                    # the command that continues it
    python -m cli weights generate                    # (re)build the manifest from disk

The headless twin of the weights-pack engine. It drives the *same*
:mod:`core.weights_pack` engine, which in turn installs through
:mod:`core.checkpoint_import`, so the CLI and the Model Manager window (its
*Weights pack* section, and the ``models weights-pack`` twin) can never disagree
about what the pack holds, what a file is, or where it goes.

Nothing here trains: it verifies a file against the pack's digest, kind, and
architecture, installs it into the exact folder the project's trainer already
resumes from, and names the command that continues it. Installing writes only to
the git-ignored ``data/`` folder; the pack itself is read-only.

Exit codes: ``0`` success, ``2`` invalid arguments, ``3`` the pack or an entry
could not be read, ``4`` a verification refused the install, ``5`` an unexpected
failure.
"""

from __future__ import annotations

import argparse
import json
import sys

from core.weights_pack import (
    Verification,
    WeightsPackError,
    generate_manifest,
    install_pack,
    load_pack,
    pack_status,
    verify_entry,
)


def _print_json(value: object) -> None:
    print(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True))


def _status(root: str = "", as_json: bool = False) -> int:
    status = pack_status(root or None)
    if as_json:
        _print_json(status.to_primitive())
        return 0 if status.pack.available else 3
    print(status.describe())
    return 0 if status.pack.available else 3


def _list(root: str = "", as_json: bool = False) -> int:
    pack = load_pack(root or None)
    if not pack.available:
        if as_json:
            _print_json(pack.to_primitive())
        else:
            print(pack.detail, file=sys.stderr)
        return 3
    if as_json:
        _print_json([entry.to_primitive() for entry in pack.entries])
        return 0
    for entry in pack.entries:
        command = f"  → {entry.continue_command}" if entry.continue_command else ""
        print(f"{entry.key}\t{entry.kind}\t{entry.path}{command}")
    return 0


def _verify(root: str = "", key: str = "", as_json: bool = False) -> int:
    pack = load_pack(root or None)
    if not pack.available:
        print(pack.detail, file=sys.stderr)
        return 3
    if key:
        entry = pack.get(key)
        if entry is None:
            print(
                f"No pack entry named {key!r}. Available: {', '.join(pack.keys())}.",
                file=sys.stderr,
            )
            return 3
        verifications: tuple[Verification, ...] = (verify_entry(entry),)
    else:
        from core.weights_pack import verify_pack

        verifications = verify_pack(pack)
    if as_json:
        _print_json([item.to_primitive() for item in verifications])
    else:
        for item in verifications:
            print(item.describe())
    return 0 if all(item.ok for item in verifications) else 4


def _install(
    root: str = "",
    keys: tuple[str, ...] = (),
    *,
    install_all: bool = False,
    force: bool = False,
    no_backup: bool = False,
    as_json: bool = False,
) -> int:
    pack = load_pack(root or None)
    if not pack.available:
        print(pack.detail, file=sys.stderr)
        return 3
    # ``--all`` (or no keys at all) means the whole pack; explicit keys install
    # only those. Either way ``install_pack`` verifies each entry first.
    selected = () if install_all else keys
    report = install_pack(pack, keys=selected, force=force, backup=not no_backup)
    if as_json:
        _print_json(report.to_primitive())
    else:
        print(report.describe())
    if report.detail:
        return 4
    return 0 if report.refused == 0 else 4


def _plan(root: str = "", key: str = "", as_json: bool = False) -> int:
    pack = load_pack(root or None)
    if not pack.available:
        print(pack.detail, file=sys.stderr)
        return 3
    entry = pack.get(key)
    if entry is None:
        print(
            f"No pack entry named {key!r}. Available: {', '.join(pack.keys())}.",
            file=sys.stderr,
        )
        return 3
    if as_json:
        _print_json(entry.to_primitive())
        return 0
    print(f"{entry.title} ({entry.kind})")
    print(f"  file: {entry.path}")
    print(f"  resumable: {'yes' if entry.resumable else 'no'}")
    if entry.continue_command:
        print(f"  continue with: {entry.continue_command}")
    if entry.notes:
        print(f"  {entry.notes}")
    return 0


def _generate(root: str = "", as_json: bool = False) -> int:
    pack = generate_manifest(root=root or None)
    if as_json:
        _print_json(pack.to_primitive())
    else:
        print(pack.describe())
    return 0 if pack.entries else 3


def _continue(
    root: str = "",
    key: str = "",
    *,
    run: bool = False,
    force: bool = False,
    as_json: bool = False,
) -> int:
    """Install one entry, then name (or run) the command that continues it.

    The one-step path a clean clone uses: verify the entry against the pack's
    own digest, install it into the exact folder the trainer resumes from, and
    hand back the command -- so continuing the project's own trained model is
    ``python -m cli weights continue fayd-llm`` rather than reading the docs.
    """
    import shlex
    import subprocess

    pack = load_pack(root or None)
    if not pack.available:
        print(pack.detail, file=sys.stderr)
        return 3
    entry = pack.get(key)
    if entry is None:
        print(
            f"No pack entry named {key!r}. Available: {', '.join(pack.keys())}.",
            file=sys.stderr,
        )
        return 3
    if not entry.continue_command:
        print(
            f"The pack entry {key!r} names no trainer to continue it; install it "
            "with `python -m cli weights install`.",
            file=sys.stderr,
        )
        return 3
    report = install_pack(pack, keys=(key,), force=force, backup=True)
    if not report.ok:
        if as_json:
            _print_json(report.to_primitive())
        else:
            print(report.describe(), file=sys.stderr)
        return 4
    payload = {
        "key": key,
        "installed": report.installed,
        "command": entry.continue_command,
        "ran": False,
    }
    if run:
        print(f"Continuing: {entry.continue_command}", flush=True)
        argv = shlex.split(entry.continue_command)
        completed = subprocess.run(argv)
        payload["ran"] = True
        payload["returncode"] = completed.returncode
        return completed.returncode
    if as_json:
        _print_json(payload)
    else:
        print(report.describe())
        print(f"\nContinue with:\n  {entry.continue_command}")
    return 0


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="weights",
        description=(
            "The committed pack of ready-made .npz weights the project ships, with "
            "the trainer that continues each. Verify a file against the pack's own "
            "digest and install it into the folder the trainer resumes from."
        ),
    )
    sub = parser.add_subparsers(dest="command", required=True)

    status = sub.add_parser("status", help="what the pack offers, verified")
    status.add_argument("--root", help="the pack folder (default: weights/)")
    status.add_argument("--json", dest="as_json", action="store_true", help="print JSON")

    listing = sub.add_parser("list", help="the pack entries, one per line")
    listing.add_argument("--root", help="the pack folder (default: weights/)")
    listing.add_argument("--json", dest="as_json", action="store_true", help="print JSON")

    verify = sub.add_parser("verify", help="check a file against the pack")
    verify.add_argument("key", nargs="?", default="", help="one entry key (default: all)")
    verify.add_argument("--root", help="the pack folder (default: weights/)")
    verify.add_argument("--json", dest="as_json", action="store_true", help="print JSON")

    install = sub.add_parser("install", help="install entries into the trainer's folder")
    install.add_argument("keys", nargs="*", help="entry keys (default: the whole pack)")
    install.add_argument("--all", dest="install_all", action="store_true", help="every entry")
    install.add_argument("--root", help="the pack folder (default: weights/)")
    install.add_argument("--force", action="store_true", help="replace an existing file")
    install.add_argument("--no-backup", action="store_true", help="do not keep a backup")
    install.add_argument("--json", dest="as_json", action="store_true", help="print JSON")

    plan = sub.add_parser("plan", help="the command that continues an entry's weights")
    plan.add_argument("key", help="the entry key")
    plan.add_argument("--root", help="the pack folder (default: weights/)")
    plan.add_argument("--json", dest="as_json", action="store_true", help="print JSON")

    generate = sub.add_parser("generate", help="(re)build the manifest from the weights on disk")
    generate.add_argument("--root", help="the pack folder (default: weights/)")
    generate.add_argument("--json", dest="as_json", action="store_true", help="print JSON")

    cont = sub.add_parser(
        "continue",
        help="install an entry, then name (or run) the command that continues it",
    )
    cont.add_argument("key", help="the entry key (e.g. fayd-llm)")
    cont.add_argument("--root", help="the pack folder (default: weights/)")
    cont.add_argument("--force", action="store_true", help="replace an existing file")
    cont.add_argument(
        "--run",
        action="store_true",
        help="also run the trainer command (it continues the weights it just installed)",
    )
    cont.add_argument("--json", dest="as_json", action="store_true", help="print JSON")

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    root = getattr(args, "root", "") or ""
    as_json = getattr(args, "as_json", False)
    try:
        if args.command == "status":
            return _status(root, as_json)
        if args.command == "list":
            return _list(root, as_json)
        if args.command == "verify":
            return _verify(root, args.key, as_json)
        if args.command == "install":
            return _install(
                root,
                tuple(args.keys),
                install_all=getattr(args, "install_all", False),
                force=getattr(args, "force", False),
                no_backup=getattr(args, "no_backup", False),
                as_json=as_json,
            )
        if args.command == "plan":
            return _plan(root, args.key, as_json)
        if args.command == "generate":
            return _generate(root, as_json)
        if args.command == "continue":
            return _continue(
                root,
                args.key,
                run=getattr(args, "run", False),
                force=getattr(args, "force", False),
                as_json=as_json,
            )
    except WeightsPackError as error:
        print(f"error: {error}", file=sys.stderr)
        return 4
    except (OSError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 5
    parser.error(f"unknown command {args.command!r}")
    return 2


__all__ = ["main"]


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
