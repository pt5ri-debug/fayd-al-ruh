"""CLI for the bulk knowledge base: build it, inspect it, search it.

    python -m cli knowledge build --target-mb 100
    python -m cli knowledge stats
    python -m cli knowledge search "hydraulic fracturing"

The store lives under the writable data directory (``data/knowledge_base`` by
default, overridable with ``FAYD_DATA_DIR``) and is git-ignored. Building needs
network access to Wikipedia; searching a built store is offline.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence

from core.knowledge_base import (
    DEFAULT_CATEGORIES,
    DEFAULT_DEPTH,
    DEFAULT_SUBCAT_LIMIT,
    DEFAULT_TARGET_MB,
    KnowledgeBaseStore,
    build_knowledge_base,
    default_knowledge_base_dir,
    iter_local_documents,
)
from core.knowledge_index import KnowledgeTextIndex, build_index
from core.rag import BulkCorpusRetriever


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="knowledge", description=__doc__)
    parser.add_argument(
        "--store-dir",
        default=None,
        help="Knowledge-base folder (default: <data>/knowledge_base).",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    build = sub.add_parser("build", help="Fetch and store chunks until the target size.")
    build.add_argument("--target-mb", type=float, default=DEFAULT_TARGET_MB)
    build.add_argument("--max-articles", type=int, default=None)
    build.add_argument(
        "--category",
        action="append",
        dest="categories",
        default=None,
        help="Category to include (repeatable; default: the broad seed).",
    )
    build.add_argument("--batch", type=int, default=50)
    build.add_argument("--chunk-chars", type=int, default=1200)
    build.add_argument("--overlap", type=int, default=150)
    build.add_argument(
        "--depth",
        type=int,
        default=DEFAULT_DEPTH,
        help="Subcategory depth to walk (0 = seed categories only).",
    )
    build.add_argument("--subcat-limit", type=int, default=DEFAULT_SUBCAT_LIMIT)

    sub.add_parser("stats", help="Show the store size and chunk count.")
    sub.add_parser(
        "build-index",
        help="Build the fast full-text index (run after `build`).",
    )

    search = sub.add_parser("search", help="Search a built store (offline).")
    search.add_argument("query")
    search.add_argument("--limit", type=int, default=5)
    return parser


def _resolve_dir(value: str | None):
    if value:
        from pathlib import Path

        return Path(value)
    return default_knowledge_base_dir()


def run_build(args: argparse.Namespace) -> int:
    store_dir = _resolve_dir(args.store_dir)
    categories = tuple(args.categories) if args.categories else DEFAULT_CATEGORIES

    def progress(message: str) -> None:
        print(message, flush=True)

    print(f"Building knowledge base in {store_dir}")
    print(f"Target: {args.target_mb:.0f} MB across {len(categories)} categories")
    manifest = build_knowledge_base(
        store_dir,
        target_mb=args.target_mb,
        categories=categories,
        batch=args.batch,
        chunk_chars=args.chunk_chars,
        overlap=args.overlap,
        depth=args.depth,
        subcat_limit=args.subcat_limit,
        max_articles=args.max_articles,
        progress=progress,
    )
    print(
        f"\nDone: {manifest['total_mb']:.2f} MB, "
        f"{manifest['total_chunks']} chunks, {manifest['total_articles']} articles "
        f"({'complete' if manifest['complete'] else 'partial'})"
    )
    return 0 if manifest["complete"] else 1


def run_stats(args: argparse.Namespace) -> int:
    store_dir = _resolve_dir(args.store_dir)
    store = KnowledgeBaseStore(store_dir)
    stats = store.stats()
    print(f"Knowledge base : {stats['path']}")
    print(f"Size           : {stats['size_mb']:.2f} MB ({stats['size_bytes']} bytes)")
    print(f"Shards         : {stats['shards']}")
    print(f"Chunks         : {stats['chunks']}")
    print(f"Articles       : {stats['articles']}")
    print(f"Target         : {stats['target_mb']:.0f} MB")
    print(f"Complete       : {stats['complete']}")
    index_stats = KnowledgeTextIndex(store_dir).stats()
    if index_stats["present"]:
        print(f"Index          : {index_stats['size_mb']:.2f} MB ({index_stats['chunks']} entries)")
    else:
        print("Index          : absent (run `python -m cli knowledge build-index`)")
    return 0


def run_build_index(args: argparse.Namespace) -> int:
    store_dir = _resolve_dir(args.store_dir)
    if not KnowledgeBaseStore(store_dir).shard_paths():
        print(f"No knowledge base at {store_dir}. Build it with `python -m cli knowledge build`.")
        return 1
    print(f"Building text index in {store_dir}")
    summary = build_index(store_dir, progress=lambda message: print(message, flush=True))
    print(f"Done: indexed {summary['indexed']} chunks ({summary['size_mb']:.2f} MB)")
    return 0


def run_search(args: argparse.Namespace) -> int:
    store_dir = _resolve_dir(args.store_dir)
    if not KnowledgeBaseStore(store_dir).shard_paths():
        print(f"No knowledge base at {store_dir}. Build it with `python -m cli knowledge build`.")
        return 1
    index = KnowledgeTextIndex(store_dir)
    retriever = BulkCorpusRetriever(
        lambda: iter_local_documents(store_dir),
        index=index if index.exists() else None,
    )
    passages = retriever.retrieve(args.query, limit=args.limit)
    if not passages:
        print("No match.")
        return 0
    for passage in passages:
        print(f"[{passage.score:.3f}] {passage.title} ({passage.doc_id})")
        print(f"    {passage.text[:300]}")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(list(argv) if argv is not None else sys.argv[1:])
    if args.command == "build":
        return run_build(args)
    if args.command == "stats":
        return run_stats(args)
    if args.command == "build-index":
        return run_build_index(args)
    if args.command == "search":
        return run_search(args)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
