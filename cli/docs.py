"""CLI for Doc's Studio: read any document, then ask about it.

    python -m cli docs status                    # is the local vision model ready?
    python -m cli docs models                    # list the models it can fetch
    python -m cli docs download --approve        # one-time, ~3.4 GB (weights + projector)
    python -m cli docs open report.pdf           # load it and show its contents
    python -m cli docs text report.pdf           # print the extracted text
    python -m cli docs pages report.pdf          # which pages need the vision model
    python -m cli docs ask report.pdf "what is the porosity on page 3?"
    python -m cli docs ask scan.pdf "summarise this" --pages 1-3

This is the headless twin of the window (``＋ → Doc's Studio``): it drives the
*same* :class:`gui.docs_studio.DocsStudioService`, so the CLI and the window share
one loader, one page renderer, and one model, and no workflow logic is
reimplemented here.

Exit codes: ``0`` success, ``2`` invalid arguments, ``3`` the vision model is not
available, ``4`` the document could not be read or the question could not be
answered, ``5`` an unexpected failure.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence

from core.docs_model import DocsModelUnavailable
from core.docs_studio import DocsStudioError, parse_page_range
from gui.docs_studio import DocsStudioService

EXIT_OK = 0
EXIT_USAGE = 2
EXIT_MODEL = 3
EXIT_DOCUMENT = 4
EXIT_FAILURE = 5


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="docs", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("status", help="Report whether the local vision model can read.")

    models = sub.add_parser("models", help="List the models Doc's Studio can fetch.")
    models.add_argument("--json", action="store_true", help="Print machine-readable JSON.")

    download = sub.add_parser("download", help="Download a model's weights and projector (one-time).")
    download.add_argument("model", nargs="?", default="", help="Catalog key (default: qwen3-vl-4b).")
    download.add_argument("--approve", action="store_true", help="Record the human approval.")
    download.add_argument("--json", action="store_true", help="Print machine-readable JSON.")

    for name, help_text in (
        ("open", "Load a document and show a summary of its contents."),
        ("text", "Print a document's extracted text."),
        ("pages", "List the pages that need the vision model (no text layer)."),
    ):
        command = sub.add_parser(name, help=help_text)
        command.add_argument("document", help="Path to the document.")
        command.add_argument("--no-render", action="store_true", help="Do not render page images.")
        command.add_argument("--json", action="store_true", help="Print machine-readable JSON.")

    ask = sub.add_parser("ask", help="Ask the vision model a question about a document.")
    ask.add_argument("document", help="Path to the document.")
    ask.add_argument("question", help="The question to ask about the document.")
    ask.add_argument("--pages", default="", help="Restrict to pages, e.g. 1-3 or 2,5,7.")
    ask.add_argument("--no-render", action="store_true", help="Do not render page images.")
    ask.add_argument("--json", action="store_true", help="Print machine-readable JSON.")

    return parser


def _service() -> DocsStudioService:
    return DocsStudioService()


def run_status(_args: argparse.Namespace) -> int:
    status = _service().model_status()
    print(status.summary())
    if status.hint and not status.available:
        print(f"Next step: {status.hint}")
    return EXIT_OK


def run_models(args: argparse.Namespace) -> int:
    service = _service()
    rows = service.model_summaries()
    if args.json:
        import json

        print(json.dumps([{"key": key, "title": title, "size_gb": size, "summary": summary}
                          for key, title, size, summary in rows], ensure_ascii=False, indent=2))
        return EXIT_OK
    print("Doc's Studio models (download with `python -m cli docs download <key> --approve`):")
    for key, title, size, summary in rows:
        print(f"  {key:<14} {title:<36} ~{size:.1f} GB  {summary}")
    return EXIT_OK


def run_download(args: argparse.Namespace) -> int:
    service = _service()
    try:
        model_path, mmproj_path = service.download_model(
            args.model, approved=args.approve, progress=lambda message: print(message, file=sys.stderr)
        )
    except DocsModelUnavailable as error:
        print(f"Cannot download: {error}")
        return EXIT_MODEL
    if args.json:
        import json

        print(json.dumps({"model": str(model_path), "mmproj": str(mmproj_path)}, indent=2))
    else:
        print(f"Model:    {model_path}")
        print(f"Projector: {mmproj_path}")
    return EXIT_OK


def _load(args: argparse.Namespace) -> tuple[DocsStudioService, str] | int:
    service = _service()
    try:
        opened = service.open_document(args.document, render=not args.no_render)
    except DocsStudioError as error:
        print(f"Could not read the document: {error}")
        return EXIT_DOCUMENT
    return service, opened.key


def run_open(args: argparse.Namespace) -> int:
    loaded = _load(args)
    if isinstance(loaded, int):
        return loaded
    service, key = loaded
    opened = service.get(key)
    assert opened is not None
    document = opened.document
    if args.json:
        import json

        print(json.dumps(
            {
                "filename": document.filename,
                "kind": document.kind,
                "pages": document.page_count,
                "words": document.word_count,
                "scanned": document.scanned,
                "truncated": document.truncated,
                "vision_pages": list(document.vision_pages),
                "notes": document.notes,
            },
            ensure_ascii=False,
            indent=2,
        ))
        return EXIT_OK
    print(document.summary())
    if document.notes:
        print("Notes: " + "; ".join(document.notes))
    if document.vision_pages:
        print("Pages read as images: " + ", ".join(str(n) for n in document.vision_pages))
    preview = document.text[:2000]
    if preview.strip():
        print("\n" + preview)
        if len(document.text) > len(preview):
            print("\n… (the rest is available to the model when you ask)")
    else:
        print("\nThis document has no text layer; the vision model reads its page image(s).")
    return EXIT_OK


def run_text(args: argparse.Namespace) -> int:
    loaded = _load(args)
    if isinstance(loaded, int):
        return loaded
    service, key = loaded
    text = service.document_text(key)
    if args.json:
        import json

        print(json.dumps({"text": text}, ensure_ascii=False))
        return EXIT_OK
    print(text)
    return EXIT_OK


def run_pages(args: argparse.Namespace) -> int:
    loaded = _load(args)
    if isinstance(loaded, int):
        return loaded
    service, key = loaded
    opened = service.get(key)
    assert opened is not None
    document = opened.document
    rows = [
        {
            "number": page.number,
            "has_text": page.has_text,
            "from_image": page.from_image,
            "chars": len(page.text),
        }
        for page in document.pages
    ]
    if args.json:
        import json

        print(json.dumps(rows, ensure_ascii=False, indent=2))
        return EXIT_OK
    for row in rows:
        source = "image" if row["from_image"] else ("text" if row["has_text"] else "empty")
        print(f"  page {row['number']:<4} {source:<6} {row['chars']} chars")
    return EXIT_OK


def run_ask(args: argparse.Namespace) -> int:
    loaded = _load(args)
    if isinstance(loaded, int):
        return loaded
    service, key = loaded
    pages: tuple[int, ...] | None = None
    if args.pages:
        try:
            pages = parse_page_range(args.pages)
        except ValueError as error:
            print(f"Invalid --pages: {error}")
            return EXIT_USAGE
    answer = service.ask(key, args.question, pages=pages)
    if args.json:
        import json

        print(json.dumps(
            {
                "ok": answer.ok,
                "question": answer.question,
                "text": answer.text,
                "pages_used": list(answer.pages_used),
                "message": answer.message,
            },
            ensure_ascii=False,
            indent=2,
        ))
        return EXIT_OK if answer.ok else EXIT_DOCUMENT
    if not answer.ok:
        print(f"Cannot answer: {answer.message}")
        return EXIT_MODEL if "model" in answer.message.lower() else EXIT_DOCUMENT
    print(answer.summary())
    return EXIT_OK


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(list(argv) if argv is not None else sys.argv[1:])
    handlers = {
        "status": run_status,
        "models": run_models,
        "download": run_download,
        "open": run_open,
        "text": run_text,
        "pages": run_pages,
        "ask": run_ask,
    }
    handler = handlers.get(args.command)
    if handler is None:
        return EXIT_USAGE
    try:
        return handler(args)
    except DocsModelUnavailable as error:
        print(f"The local vision model is not available: {error}")
        return EXIT_MODEL
    except DocsStudioError as error:
        print(f"Could not complete the request: {error}")
        return EXIT_DOCUMENT
    except Exception as error:  # noqa: BLE001 - the last line of defence
        print(f"Unexpected failure: {error}")
        return EXIT_FAILURE


if __name__ == "__main__":
    raise SystemExit(main())
