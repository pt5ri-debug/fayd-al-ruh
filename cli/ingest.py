"""CLI for the ingestion subsystem: add sources, inspect connectors, search.

    python -m cli ingest add ./papers --store-dir data/knowledge_base
    python -m cli ingest add data.csv
    python -m cli ingest add sqlite:///data/wells.db
    python -m cli ingest add https://www.spe.org/... --allow-web --approve
    python -m cli ingest list-connectors
    python -m cli ingest stats
    python -m cli ingest search "hydraulic fracturing"

Local sources run with no network and no extra dependency. A web source is the
one exception: it needs ``--allow-web`` to even register the connector, and
``--approve`` to pass the permission and human-approval gates, so importing from
the internet is a deliberate, audited act rather than a default.

Ingestion writes only to the knowledge store (``data/knowledge_base`` by
default). It never touches training artifacts or the evaluation baseline, so a
large import cannot change a release by accident.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence

from core.knowledge_base import (
    DEFAULT_CHUNK_CHARS,
    DEFAULT_OVERLAP,
    KnowledgeBaseStore,
    default_knowledge_base_dir,
    iter_local_documents,
)
from core.knowledge_index import KnowledgeTextIndex, build_index
from core.permissions import Capability, PermissionPolicy
from core.rag import BulkCorpusRetriever
from ingest.dedup import DedupLedger
from ingest.pipeline import IngestionPipeline, ledger_path
from ingest.records import SourceSpec
from ingest.registry import build_default_registry

_WEB_TOOL = "ingest_web"


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="ingest", description=__doc__)
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument(
        "--store-dir",
        default=argparse.SUPPRESS,
        help="Knowledge-base folder (default: <data>/knowledge_base).",
    )
    parser.add_argument(
        "--store-dir",
        default=None,
        help="Knowledge-base folder (default: <data>/knowledge_base).",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    add = sub.add_parser("add", parents=[common], help="Ingest one source (file, folder, SQL, or URL).")
    add.add_argument("source", help="File, directory, sqlite:// URI, or http(s) URL.")
    add.add_argument("--connector", default=None, help="Force a connector by name.")
    add.add_argument("--option", action="append", default=[], metavar="KEY=VALUE", help="Connector option (repeatable).")
    add.add_argument("--chunk-chars", type=int, default=DEFAULT_CHUNK_CHARS)
    add.add_argument("--overlap", type=int, default=DEFAULT_OVERLAP)
    add.add_argument("--no-incremental", action="store_true", help="Re-read every item, ignoring fingerprints.")
    add.add_argument("--allow-web", action="store_true", help="Register the internet connector.")
    add.add_argument("--approve", action="store_true", help="Grant the external capability for a web source.")
    add.add_argument("--quiet", action="store_true", help="Suppress progress.")
    add.add_argument("--json", action="store_true", dest="as_json", help="Print the report as JSON.")
    add.add_argument("--job", default=None, help="Resumable job id (saved under the store).")

    connectors = sub.add_parser("list-connectors", parents=[common], help="List the connectors and their formats.")
    connectors.add_argument("--allow-web", action="store_true")
    connectors.add_argument("--json", action="store_true", dest="as_json")

    stats = sub.add_parser("stats", parents=[common], help="Show the store and ingestion ledger.")
    stats.add_argument("--json", action="store_true", dest="as_json")

    sub.add_parser("build-index", parents=[common], help="Build the fast full-text index over the store.")

    search = sub.add_parser("search", parents=[common], help="Search the ingested store (offline).")
    search.add_argument("query")
    search.add_argument("--limit", type=int, default=5)

    delete = sub.add_parser("delete", parents=[common], help="Remove an ingested source by key.")
    delete.add_argument("source_key", help="The source key recorded in the ledger.")
    delete.add_argument("--json", action="store_true", dest="as_json")

    sub.add_parser("jobs", parents=[common], help="List resumable ingestion jobs.")
    return parser


def _resolve_dir(value: str | None):
    from pathlib import Path

    return Path(value) if value else default_knowledge_base_dir()


def _parse_options(pairs: Sequence[str]) -> dict[str, str]:
    options: dict[str, str] = {}
    for pair in pairs:
        if "=" not in pair:
            raise SystemExit(f"Option {pair!r} must be KEY=VALUE.")
        key, value = pair.split("=", 1)
        options[key.strip()] = value
    return options


def run_add(args: argparse.Namespace) -> int:
    store_dir = _resolve_dir(args.store_dir)
    options = _parse_options(args.option)
    spec = SourceSpec(args.source, options=options)
    registry = build_default_registry(allow_web=args.allow_web)
    try:
        connector = registry.get(args.connector) if args.connector else registry.resolve(spec)
    except Exception as error:  # noqa: BLE001 - a resolver error is a user-facing message
        print(f"error: {error}")
        return 1
    if connector is None:
        print(f"error: no connector named {args.connector!r}.")
        return 1

    if connector.info.remote and not _authorise_web(args):
        return 1

    store = KnowledgeBaseStore(store_dir)
    ledger = DedupLedger(ledger_path(store.root))
    progress = (lambda message: print(message, flush=True)) if not args.quiet else None
    job = None
    jobs = None
    if args.job:
        from ingest.jobs import JobState, JobStore
        from ingest.pipeline import ingest_jobs_dir

        jobs = JobStore(ingest_jobs_dir(store.root))
        job = jobs.load(args.job) or JobState(job_id=args.job, source=args.source)
        job.connector = connector.info.name
    pipeline = IngestionPipeline(store, ledger, progress=progress, job=job, jobs=jobs)
    report = pipeline.ingest(
        spec, connector=connector, incremental=not args.no_incremental
    )
    _print_report(report, pipeline, args.as_json)
    return 0 if report.failed == 0 else 1


def _authorise_web(args: argparse.Namespace) -> bool:
    """Run a web import only through the permission and approval gates."""
    policy = PermissionPolicy(
        name="ingest-cli",
        allowed=frozenset(
            {Capability.SYMBOLIC_SIMULATION, Capability.SANDBOX_TASK, Capability.EXTERNAL_TOOL}
        ),
        allowed_external_tools=frozenset({_WEB_TOOL}),
    )
    decision = policy.decide(Capability.EXTERNAL_TOOL, _WEB_TOOL)
    if not decision.allowed or not args.approve:
        print(
            "error: importing from the internet needs --approve "
            "(and the host on allowed_hosts or the source authorized)."
        )
        return False
    return True


def _print_report(report, pipeline, as_json: bool) -> None:
    if as_json:
        payload = report.as_dict()
        payload["errors"] = pipeline.errors.as_dicts()
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return
    data = report.as_dict()
    print(
        f"records {data['records']}, chunks {data['chunks']}, "
        f"{data['bytes_written'] / 1024:.1f} KB written, "
        f"skipped {data['skipped_unchanged']} unchanged"
    )
    if pipeline.errors:
        print(f"{len(pipeline.errors)} source(s) failed:")
        for error in pipeline.errors.as_dicts():
            print(f"  {error['source']}: {error['kind']}: {error['message']}")


def run_list_connectors(args: argparse.Namespace) -> int:
    registry = build_default_registry(allow_web=args.allow_web)
    infos = registry.infos()
    if args.as_json:
        payload = [
            {
                "name": info.name,
                "description": info.description,
                "suffixes": sorted(info.suffixes),
                "schemes": sorted(info.schemes),
                "remote": info.remote,
                "optional_dependency": info.optional_dependency,
            }
            for info in infos
        ]
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return 0
    for info in infos:
        formats = ", ".join(sorted(info.suffixes) or sorted(info.schemes)) or "any path"
        flag = " (remote, denied by default)" if info.remote else ""
        print(f"{info.name:<8} {formats}{flag}")
        print(f"         {info.description}")
    return 0


def run_stats(args: argparse.Namespace) -> int:
    store_dir = _resolve_dir(args.store_dir)
    store = KnowledgeBaseStore(store_dir)
    ledger_stats = DedupLedger(ledger_path(store.root)).stats()
    store_stats = store.stats()
    index_stats = KnowledgeTextIndex(store_dir).stats()
    if args.as_json:
        print(
            json.dumps(
                {"store": store_stats, "ledger": ledger_stats, "index": index_stats},
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0
    print(f"Knowledge base : {store_stats['path']}")
    print(f"Chunks         : {store_stats['chunks']}")
    print(f"Size           : {store_stats['size_mb']:.2f} MB")
    print(f"Ingested items : {ledger_stats['sources']} ({ledger_stats['chunks']} chunks tracked)")
    if index_stats["present"]:
        print(f"Index          : {index_stats['size_mb']:.2f} MB ({index_stats['chunks']} entries)")
    else:
        print("Index          : absent (run `python -m cli ingest build-index`)")
    return 0


def run_build_index(args: argparse.Namespace) -> int:
    store_dir = _resolve_dir(args.store_dir)
    if not KnowledgeBaseStore(store_dir).shard_paths():
        print("No store to index. Ingest a source first with `python -m cli ingest add`.")
        return 1
    print(f"Building text index in {store_dir}")
    summary = build_index(store_dir, progress=lambda message: print(message, flush=True))
    print(f"Done: indexed {summary['indexed']} chunks ({summary['size_mb']:.2f} MB)")
    return 0


def run_search(args: argparse.Namespace) -> int:
    store_dir = _resolve_dir(args.store_dir)
    if not KnowledgeBaseStore(store_dir).shard_paths():
        print("No store to search. Ingest a source first with `python -m cli ingest add`.")
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


def run_delete(args: argparse.Namespace) -> int:
    from ingest.delete import delete_source

    store_dir = _resolve_dir(args.store_dir)
    report = delete_source(args.source_key, store_dir=store_dir)
    if args.as_json:
        print(json.dumps(report.as_dict(), ensure_ascii=False, indent=2))
        return 0
    if report.removed == 0:
        print(f"No chunks recorded for source key {args.source_key!r}.")
        return 1
    print(f"Removed {report.removed} chunk(s); {report.remaining} remain.")
    print("Rebuild the index with `python -m cli ingest build-index`.")
    return 0


def run_jobs(args: argparse.Namespace) -> int:
    from ingest.jobs import JobStore
    from ingest.pipeline import ingest_jobs_dir

    store_dir = _resolve_dir(args.store_dir)
    states = JobStore(ingest_jobs_dir(store_dir)).list()
    if not states:
        print("No ingestion jobs recorded.")
        return 0
    for state in states:
        print(
            f"{state.job_id} [{state.status}] pos {state.position} "
            f"chunks {state.chunks} — {state.source}"
        )
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(list(argv) if argv is not None else sys.argv[1:])
    if args.command == "add":
        return run_add(args)
    if args.command == "list-connectors":
        return run_list_connectors(args)
    if args.command == "stats":
        return run_stats(args)
    if args.command == "build-index":
        return run_build_index(args)
    if args.command == "search":
        return run_search(args)
    if args.command == "delete":
        return run_delete(args)
    if args.command == "jobs":
        return run_jobs(args)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
