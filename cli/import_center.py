"""Command line for the Knowledge Import Center.

    python -m cli import-center                      # prompt for a file/folder
    python -m cli import-center files a.txt b.md     # upload-style import
    python -m cli import-center folder ./papers      # recursive, bounded batches
    python -m cli import-center folder ./papers --job papers  # resumable
    python -m cli import-center resume papers        # continue a job
    python -m cli import-center jobs                 # list jobs
    python -m cli import-center stats                # store + ledger
    python -m cli import-center formats              # supported formats

This is the *headless* front end to the same engine the window's **＋** control
uses (:class:`gui.import_center.KnowledgeImporter`), which in turn drives the
existing :mod:`ingest` pipeline. It reuses the CLI's own ``ingest`` machinery
rather than reimplementing it, so the two entry points cannot drift.

Nothing here is new ingestion logic: verify -> extract -> dedup -> chunk+store ->
record -> retry, with per-file results, retry, and resume, exactly as the window
offers. With no subcommand (or ``interactive``) it prompts for a path, so the
numbered menu's "Import files and folders" entry actually imports something
instead of only listing the formats.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path

from gui.import_center import SUPPORTED_SUFFIXES, UNSUPPORTED_SUFFIXES, KnowledgeImporter


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="import-center", description=__doc__)
    parser.add_argument("--store-dir", default=None, help="Knowledge base folder (default: <data>/knowledge_base).")
    parser.add_argument("--no-normalize", action="store_true", help="Store text verbatim.")
    parser.add_argument("--fold-arabic", action="store_true", help="Also unify Arabic letter forms (opt-in).")
    parser.add_argument("--retries", type=int, default=2, help="Per-file retries for a transient read failure.")
    sub = parser.add_subparsers(dest="command")

    files = sub.add_parser("files", help="Import one or many chosen files.")
    files.add_argument("paths", nargs="+")
    files.add_argument("--json", action="store_true", dest="as_json")

    folder = sub.add_parser("folder", help="Import every supported file under a folder.")
    folder.add_argument("folder", nargs="?", default=None)
    folder.add_argument("--no-recursive", action="store_true", help="Do not scan subfolders.")
    folder.add_argument("--job", default=None, help="Resumable job id (saved under the store).")
    folder.add_argument("--json", action="store_true", dest="as_json")

    resume = sub.add_parser("resume", help="Resume an interrupted folder job.")
    resume.add_argument("job_id")
    resume.add_argument("--json", action="store_true", dest="as_json")

    interactive = sub.add_parser(
        "interactive", help="Prompt for a file or folder and import it (the menu entry)."
    )
    interactive.add_argument("path", nargs="?", default=None, help="File or folder (prompts when omitted).")
    interactive.add_argument("--json", action="store_true", dest="as_json")

    sub.add_parser("jobs", help="List recorded import jobs.")
    sub.add_parser("stats", help="Show the store and import bookkeeping.")
    sub.add_parser("formats", help="Show the supported and unsupported formats.")
    return parser


def _importer(args: argparse.Namespace) -> KnowledgeImporter:
    return KnowledgeImporter(
        store_dir=args.store_dir,
        normalize=not args.no_normalize,
        fold_arabic=args.fold_arabic,
        retries=args.retries,
    )


def _print_report(report, as_json: bool) -> int:
    if as_json:
        print(json.dumps(report.as_dict(), ensure_ascii=False, indent=2))
    else:
        print(report.summary())
        for outcome in report.outcomes:
            marker = {"ok": "ok", "unchanged": "unchanged", "failed": "FAILED"}.get(outcome.status, "?")
            detail = outcome.message if outcome.status == "failed" else f"{outcome.chunks} chunk(s)"
            print(f"  [{marker}] {outcome.filename}: {detail}")
    return 0 if report.ok else 1


def run_files(args: argparse.Namespace) -> int:
    report = _importer(args).import_files(args.paths, progress=lambda m: print(m, flush=True))
    return _print_report(report, args.as_json)


def run_folder(args: argparse.Namespace) -> int:
    target = args.folder or _prompt_path("Folder to import")
    if not target:
        print("No folder given.")
        return 1
    report = _importer(args).import_folder(
        target,
        recursive=not args.no_recursive,
        job_id=args.job,
        progress=lambda m: print(m, flush=True),
    )
    return _print_report(report, args.as_json)


def run_resume(args: argparse.Namespace) -> int:
    report = _importer(args).resume_job(args.job_id, progress=lambda m: print(m, flush=True))
    return _print_report(report, args.as_json)


def run_jobs(args: argparse.Namespace) -> int:
    jobs = _importer(args).list_jobs()
    if not jobs:
        print("No import jobs recorded.")
        return 0
    for state in jobs:
        print(f"{state.job_id} [{state.status}] pos {state.position} chunks {state.chunks} — {state.source}")
    return 0


def run_stats(args: argparse.Namespace) -> int:
    print(json.dumps(_importer(args).stats(), ensure_ascii=False, indent=2, default=str))
    return 0


def run_formats(args: argparse.Namespace) -> int:
    print("Supported:     " + ", ".join(SUPPORTED_SUFFIXES))
    print("Unsupported:   " + ", ".join(UNSUPPORTED_SUFFIXES) + " (save .doc/.rtf as .docx)")
    print("Optional PDF:  needs fayd-al-ruh[pdf] (pypdf); scanned PDFs need [ocr] + Tesseract.")
    return 0


def _prompt_path(prompt: str) -> str:
    """Ask for a local path on the terminal, tolerating a missing stdin."""
    try:
        return input(f"{prompt}: ").strip().strip('"')
    except (EOFError, KeyboardInterrupt):
        print()
        return ""


def run_interactive(args: argparse.Namespace) -> int:
    """Prompt for a file or folder and import it (what the menu entry runs).

    A file is imported as one source; a folder is imported in bounded batches,
    exactly like the window's two buttons. A path may also be given directly, and
    with no terminal to prompt on it prints the supported formats and how to pass
    a path, so the entry is never a silent no-op.
    """
    print("Knowledge Import Center")
    print("Supported: " + ", ".join(SUPPORTED_SUFFIXES))
    print("Not supported: " + ", ".join(UNSUPPORTED_SUFFIXES) + " (save as .docx).")
    target = getattr(args, "path", None) or ""
    if not target:
        if not sys.stdin or not sys.stdin.isatty():
            print("No terminal to prompt on. Pass a path directly, e.g.:")
            print("  python -m cli import-center files report.pdf")
            print("  python -m cli import-center folder ./papers")
            return 0
        target = _prompt_path("File or folder to import (Enter to cancel)")
    if not target:
        print("Cancelled.")
        return 0
    path = Path(target).expanduser()
    if not path.exists():
        print(f"No such file or folder: {path}")
        return 1
    importer = _importer(args)
    if path.is_dir():
        report = importer.import_folder(
            path,
            job_id=f"import:{path.name}",
            progress=lambda m: print(m, flush=True),
        )
    else:
        report = importer.import_files([path], progress=lambda m: print(m, flush=True))
    return _print_report(report, getattr(args, "as_json", False))


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(list(argv) if argv is not None else sys.argv[1:])
    handlers = {
        "files": run_files,
        "folder": run_folder,
        "resume": run_resume,
        "jobs": run_jobs,
        "stats": run_stats,
        "formats": run_formats,
        "interactive": run_interactive,
    }
    command = args.command or "interactive"
    return handlers[command](args)


if __name__ == "__main__":
    raise SystemExit(main())
