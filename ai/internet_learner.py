"""Learn petroleum knowledge continuously from trusted internet sites.

    python -m cli internet-learner                 # dry run over a few topics
    python -m cli internet-learner --approve       # allow search, still dry run
    python -m cli internet-learner --approve --write
    python -m cli internet-learner --approve --write --max-topics 5
    python -m cli internet-learner --topics "Porosity" "Permeability"
    python -m cli internet-learner --list-topics

Search is an external tool, so it is **denied by default**: nothing reaches the
network until ``--approve`` grants the capability and records a human approval,
and even then the run is a **dry run** unless ``--write`` is given, so you can
always inspect what would be added before it is saved.

Only results from trusted domains (SPE, OnePetro, API, ISO, ASTM, Wikipedia, and
government or academic hosts) are considered, and a statement is folded into the
project's own data store only when it shares terms with what the project already
knows. Topics are read from the project's petroleum corpus, so the material is
petroleum-relevant by construction rather than a fixed question list.
"""

from __future__ import annotations

import argparse
import json as _json
import os

from core.human_approval import HumanApprovalGate
from core.permissions import Capability, PermissionPolicy
from core.web_source import (
    FallbackSearchProvider,
    HttpSearchProvider,
    StaticSearchProvider,
    WebSearchProvider,
    WikipediaSearchProvider,
)

_ENDPOINT_ENV = "FAYD_SEARCH_ENDPOINT"
_API_KEY_ENV = "FAYD_SEARCH_API_KEY"


def _permissions(approved: bool) -> PermissionPolicy:
    if not approved:
        return PermissionPolicy()
    return PermissionPolicy(
        allowed=frozenset(
            {
                Capability.SYMBOLIC_SIMULATION,
                Capability.SANDBOX_TASK,
                Capability.EXTERNAL_TOOL,
            }
        ),
        allowed_external_tools=frozenset({"web_search"}),
    )


def _approvals(approved: bool) -> HumanApprovalGate:
    gate = HumanApprovalGate()
    if approved:
        request = gate.request("web_search", requested_by="internet-learner")
        request.approve("cli-reviewer")
    return gate


def _provider(endpoint: str | None) -> WebSearchProvider:
    if endpoint:
        return HttpSearchProvider(endpoint, api_key=os.environ.get(_API_KEY_ENV))
    return FallbackSearchProvider(WikipediaSearchProvider(), StaticSearchProvider())


def _documents():
    """The project's persistent data store, or an in-memory one if read-only."""
    from core.documents import build_document_store
    from core.paths import data_dir
    from storage.json_store import JsonStore

    try:
        documents = build_document_store(store=JsonStore(data_dir() / "documents"))
        documents.load_all()
        return documents
    except (OSError, ValueError):
        return build_document_store()


def _print_report(report, *, as_json: bool) -> None:
    if as_json:
        payload = {
            "dry_run": report.dry_run,
            "saved": report.saved,
            "search_allowed": report.search_allowed,
            "topics_considered": report.topics_considered,
            "added_count": report.added_count,
            "outcomes": [
                {
                    "topic": outcome.topic,
                    "allowed": outcome.allowed,
                    "reason": outcome.reason,
                    "results": [result.url for result in outcome.results],
                    "added": list(outcome.added),
                    "skipped": [{"url": url, "reason": reason} for url, reason in outcome.skipped],
                }
                for outcome in report.outcomes
            ],
        }
        print(_json.dumps(payload, ensure_ascii=False, indent=2))
        return
    print(report.summary())


def main(argv: list[str] | None = None) -> int:
    from core.internet_learner import build_internet_learner, corpus_topics

    parser = argparse.ArgumentParser(
        description="Learn petroleum knowledge from trusted internet sites, into the data store."
    )
    parser.add_argument("--approve", action="store_true", help="grant the external search capability")
    parser.add_argument("--write", action="store_true", help="save accepted statements (default: dry run)")
    parser.add_argument("--list-topics", action="store_true", help="print the corpus-derived topics and exit")
    parser.add_argument("--topics", nargs="*", default=None, help="explicit topics (default: from the corpus)")
    parser.add_argument(
        "--max-topics",
        type=int,
        default=5,
        help="how many topics to run in this batch (default: 5; 0 means all)",
    )
    parser.add_argument("--per-topic", type=int, default=3, help="max statements to keep per topic")
    parser.add_argument(
        "--min-credibility",
        type=float,
        default=0.6,
        help="minimum source credibility to accept (0-1)",
    )
    parser.add_argument(
        "--allow-untrusted",
        action="store_true",
        help="do not restrict to trusted domains (not recommended)",
    )
    parser.add_argument(
        "--endpoint",
        type=str,
        default=os.environ.get(_ENDPOINT_ENV),
        help=f"search endpoint URL (or set {_ENDPOINT_ENV}); default: Wikipedia, then offline",
    )
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)

    topics = tuple(args.topics) if args.topics else corpus_topics()
    if args.list_topics:
        for topic in topics:
            print(topic)
        return 0

    limit = None if args.max_topics == 0 else max(0, args.max_topics)
    documents = _documents()
    learner = build_internet_learner(
        provider=_provider(args.endpoint),
        store=documents,
        permissions=_permissions(args.approve),
        approvals=_approvals(args.approve),
        persist=documents.save,
        trusted_only=not args.allow_untrusted,
        min_credibility=args.min_credibility,
        per_topic=args.per_topic,
    )
    report = learner.learn_topics(topics, limit=limit, dry_run=not args.write)
    _print_report(report, as_json=args.json)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
