"""Fill one character's knowledge base from the command line.

    python -m cli kb characters                     # the owners you can add to
    python -m cli kb status natasha                 # the selected corpus, or the next step
    python -m cli kb add-files natasha a.txt b.pdf  # add documents
    python -m cli kb add-folder natasha ./papers     # add a whole folder
    python -m cli kb paste natasha --text-file note.txt   # add pasted text
    python -m cli kb search natasha "reservoir simulation" --approve   # governed search
    python -m cli kb fetch natasha https://example.org/a --approve     # fetch a page
    python -m cli kb fetch natasha https://forum.example/board --follow --approve
    python -m cli kb board natasha https://forum.example/board --pages 10 --approve
    python -m cli kb board natasha https://forum.example/board --list --approve   # just list
    python -m cli kb index natasha                   # build the full-text index

The headless twin of the **Knowledge Base Manager** window: it drives the *same*
:class:`gui.kb_manager.KnowledgeBaseManagerService`, so the CLI and the window
share one set of characters, one per-character corpus, and one governed web path.
Every source flows into the **selected character's** own store
(``data/characters/<owner>/knowledge_base``).

The web path is denied until ``--approve`` is given (the same permission +
human-approval posture the rest of the project uses), so a search or a fetch
without it is refused honestly rather than reaching the network.

Exit codes: ``0`` success, ``2`` invalid arguments, ``3`` the character or the
corpus is unknown/empty, ``4`` the web path was not approved, ``5`` an unexpected
failure.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence

from gui.kb_manager import KnowledgeBaseManagerService


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="kb", description=__doc__)
    # ``--json`` and ``--data-root`` are accepted **before or after** the
    # subcommand, the way the rest of the project's CLIs allow it.
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--data-root", default=argparse.SUPPRESS, help="Data root (default: <data>).")
    common.add_argument("--json", action="store_true", default=argparse.SUPPRESS, help="Machine-readable output.")
    parser.add_argument("--data-root", default=None, help="Data root (default: <data>).")
    parser.add_argument("--json", action="store_true", help="Machine-readable output.")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("characters", parents=[common], help="List the owners a corpus can be added to.")

    status = sub.add_parser("status", parents=[common], help="Show one character's corpus, or the next step.")
    status.add_argument("character", nargs="?", default=None, help="Owner key (default: Jarvis).")

    add_files = sub.add_parser("add-files", parents=[common], help="Add one or more files.")
    add_files.add_argument("character")
    add_files.add_argument("paths", nargs="+")

    add_folder = sub.add_parser("add-folder", parents=[common], help="Add every supported file under a folder.")
    add_folder.add_argument("character")
    add_folder.add_argument("folder")
    add_folder.add_argument("--no-recursive", action="store_true")

    paste = sub.add_parser("paste", parents=[common], help="Add text supplied directly.")
    paste.add_argument("character")
    paste.add_argument("--text", default=None, help="The text to add.")
    paste.add_argument("--text-file", default=None, help="Read the text from a file.")
    paste.add_argument("--title", default="", help="A title for the pasted text.")

    search = sub.add_parser("search", parents=[common], help="Search the web (needs --approve).")
    search.add_argument("character")
    search.add_argument("query")
    search.add_argument("--limit", type=int, default=8)
    search.add_argument("--approve", action="store_true", help="Approve the web path.")

    fetch = sub.add_parser("fetch", parents=[common], help="Fetch a page's whole text (needs --approve).")
    fetch.add_argument("character")
    fetch.add_argument("url")
    fetch.add_argument("--follow", action="store_true", help="Also fetch the page's links.")
    fetch.add_argument("--no-same-host", action="store_true", help="Follow off-host links too.")
    fetch.add_argument("--limit", type=int, default=25, help="Max linked pages to follow.")
    fetch.add_argument("--approve", action="store_true", help="Approve the web path.")

    board = sub.add_parser(
        "board",
        parents=[common],
        help="Import every story a board's pages link to (needs --approve).",
    )
    board.add_argument("character")
    board.add_argument("url", help="The board / listing URL (e.g. a forum's thread list).")
    board.add_argument("--pages", type=int, default=10, help="How many listing pages to walk.")
    board.add_argument("--limit", type=int, default=1000, help="Max story links to import.")
    board.add_argument(
        "--match",
        default="/threads/",
        help="Keep only links containing this text (default '/threads/' for a forum board).",
    )
    board.add_argument("--no-same-host", action="store_true", help="Follow off-host links too.")
    board.add_argument(
        "--descend",
        action="store_true",
        help="Also enter the board's sub-forums (a category page holds most stories there).",
    )
    board.add_argument(
        "--boards", type=int, default=10, help="Max boards to read when --descend is given."
    )
    board.add_argument(
        "--list", action="store_true", help="Only list the discovered stories; do not import."
    )
    board.add_argument("--approve", action="store_true", help="Approve the web path.")

    index = sub.add_parser("index", parents=[common], help="Build the full-text index for a corpus.")
    index.add_argument("character", nargs="?", default=None)

    return parser


def _service(args: argparse.Namespace) -> KnowledgeBaseManagerService:
    return KnowledgeBaseManagerService(data_root=args.data_root)


def _select(service: KnowledgeBaseManagerService, key: str | None) -> int:
    if key is None:
        return 0
    try:
        service.set_character(key)
    except KeyError:
        print(f"error: unknown character: {key!r}", file=sys.stderr)
        return 3
    return 0


def _cmd_characters(args: argparse.Namespace, service: KnowledgeBaseManagerService) -> int:
    characters = service.characters()
    if args.json:
        import json

        print(
            json.dumps(
                [
                    {"owner": c.owner, "mouth": c.mouth, "name": c.name, "private": c.private}
                    for c in characters
                ],
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0
    print("Characters you can add to (each has its own knowledge base):")
    for character in characters:
        flag = "  (private)" if character.private else ""
        print(f"  {character.owner:<12} {character.name}{flag}")
    return 0


def _cmd_status(args: argparse.Namespace, service: KnowledgeBaseManagerService) -> int:
    code = _select(service, args.character)
    if code:
        return code
    stats = service.stats()
    if args.json:
        import json

        print(json.dumps(stats, ensure_ascii=False, indent=2))
    else:
        print(service.describe())
        if stats.get("present"):
            print(f"  folder: {stats.get('dir')}")
    return 0 if stats.get("present") else 3


def _cmd_add_files(args: argparse.Namespace, service: KnowledgeBaseManagerService) -> int:
    if _select(service, args.character):
        return 3
    report = service.import_files(args.paths)
    return _report(args, report, "files")


def _cmd_add_folder(args: argparse.Namespace, service: KnowledgeBaseManagerService) -> int:
    if _select(service, args.character):
        return 3
    report = service.import_folder(args.folder, recursive=not args.no_recursive)
    return _report(args, report, "folder")


def _cmd_paste(args: argparse.Namespace, service: KnowledgeBaseManagerService) -> int:
    if _select(service, args.character):
        return 3
    if args.text_file:
        from pathlib import Path

        text = Path(args.text_file).read_text(encoding="utf-8", errors="replace")
    else:
        text = args.text or ""
    if not text.strip():
        print("error: nothing to add (use --text or --text-file).", file=sys.stderr)
        return 2
    report = service.import_text(text, title=args.title)
    return _report(args, report, "text")


def _cmd_search(args: argparse.Namespace, service: KnowledgeBaseManagerService) -> int:
    if _select(service, args.character):
        return 3
    service.set_web_approval(bool(args.approve))
    report = service.search_web(args.query, limit=args.limit)
    if not report.allowed:
        print(f"error: {report.reason}", file=sys.stderr)
        return 4
    if args.json:
        import json

        print(
            json.dumps(
                [
                    {"title": r.title, "url": r.url, "snippet": r.snippet}
                    for r in report.results
                ],
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0
    if not report.results:
        print("No results.")
        return 0
    for result in report.results:
        print(f"- {result.title}\n  {result.url}")
    print("\nFetch one with:  python -m cli kb fetch <character> <url> --approve")
    return 0


def _cmd_fetch(args: argparse.Namespace, service: KnowledgeBaseManagerService) -> int:
    if _select(service, args.character):
        return 3
    service.set_web_approval(bool(args.approve))
    report = service.fetch_and_import(
        args.url,
        follow_links=bool(args.follow),
        same_host=not args.no_same_host,
        limit=args.limit,
    )
    if not report.allowed:
        print(f"error: {report.reason}", file=sys.stderr)
        return 4
    return _report(args, report, "page(s)")


def _cmd_board(args: argparse.Namespace, service: KnowledgeBaseManagerService) -> int:
    if _select(service, args.character):
        return 3
    service.set_web_approval(bool(args.approve))
    report = service.walk_board(
        args.url,
        same_host=not args.no_same_host,
        max_pages=args.pages,
        limit=args.limit,
        match=args.match,
        descend=args.descend,
        max_boards=args.boards,
    )
    if not report.allowed:
        print(f"error: {report.reason}", file=sys.stderr)
        return 4
    if args.list:
        if args.json:
            import json

            print(json.dumps([r.url for r in report.results], ensure_ascii=False, indent=2))
        else:
            print(f"{len(report.results)} story link(s) found:")
            for result in report.results:
                print(f"  {result.url}")
        return 0
    if not report.results:
        print("No stories were found on this board.")
        return 0
    print(f"Found {len(report.results)} story link(s); fetching each page …", file=sys.stderr)
    imported = service.import_urls([r.url for r in report.results])
    return _report(args, imported, "stories")


def _cmd_index(args: argparse.Namespace, service: KnowledgeBaseManagerService) -> int:
    if _select(service, args.character):
        return 3
    result = service.build_index()
    if args.json:
        import json

        print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        print(f"Index built for {result.get('indexed', 0)} chunk(s).")
    return 0


def _report(args: argparse.Namespace, report: object, what: str) -> int:
    if args.json:
        import json

        payload = report.as_dict() if hasattr(report, "as_dict") else {
            "pages": [p.__dict__ for p in getattr(report, "pages", [])],
            "chunks": getattr(report, "chunks", 0),
        }
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    else:
        summary = report.summary() if hasattr(report, "summary") else str(report)
        print(f"Added {what}: {summary}")
        for failure in getattr(report, "failures", [])[:10]:
            print(f"  ! {failure.source}: {failure.message}", file=sys.stderr)
        for page in [p for p in getattr(report, "pages", []) if not p.ok][:10]:
            print(f"  ! {page.url}: {page.message}", file=sys.stderr)
    ok = getattr(report, "ok", True) if hasattr(report, "ok") else getattr(report, "failed", 0) == 0
    return 0 if ok else 5


_COMMANDS = {
    "characters": _cmd_characters,
    "status": _cmd_status,
    "add-files": _cmd_add_files,
    "add-folder": _cmd_add_folder,
    "paste": _cmd_paste,
    "search": _cmd_search,
    "fetch": _cmd_fetch,
    "board": _cmd_board,
    "index": _cmd_index,
}


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(list(argv) if argv is not None else None)
    try:
        service = _service(args)
        return _COMMANDS[args.command](args, service)
    except SystemExit:
        raise
    except Exception as error:  # an unexpected failure is reported, not a traceback
        print(f"error: {error}", file=sys.stderr)
        return 5


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = ["main"]
