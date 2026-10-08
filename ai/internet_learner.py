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
from core.internet_learner import LearnerReport
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


class InteractiveLearner:
    """Walk the science branches -> courses -> concepts, search, review, save.

    At every level a number drills down, ``s`` searches everything at the
    current level, ``b`` goes back one level, and ``q`` quits. A search always
    runs as a dry run first and prints the results; only ``save`` writes them to
    the project store, and it does so without searching again.
    """

    def __init__(self, learner, *, input_func=input, print_func=print, promote_path=None) -> None:
        self.learner = learner
        self.input = input_func
        self.print = print_func
        self.promote_path = promote_path

    def run(self) -> int:
        from core.knowledge_navigator import science_branches

        branches = science_branches()
        while True:
            labels = [
                f"{branch.label} ({len(branch.courses)} courses, {branch.topic_count} topics)"
                for branch in branches
            ]
            choice = self._pick("Science branches", labels, {"a": "search all topics", "q": "quit"})
            if choice == "q":
                return 0
            if choice == "a":
                self._review(_all_segments(branches), "all branches")
                continue
            branch = branches[choice]
            result = self._branch_loop(branch)
            if result == "quit":
                return 0

    def _branch_loop(self, branch):
        while True:
            labels = [
                f"{course.label} ({len(course.topics)} topics)" for course in branch.courses
            ]
            choice = self._pick(
                branch.label, labels, {"s": "search this branch", "b": "back", "q": "quit"}
            )
            if choice == "q":
                return "quit"
            if choice == "b":
                return "back"
            if choice == "s":
                self._review(_branch_segments(branch), branch.label)
                continue
            course = branch.courses[choice]
            result = self._course_loop(branch, course)
            if result == "quit":
                return "quit"

    def _course_loop(self, branch, course):
        while True:
            labels = list(course.topics)
            choice = self._pick(
                f"{branch.label} > {course.label}",
                labels,
                {"s": "search this course", "b": "back", "q": "quit"},
            )
            if choice == "q":
                return "quit"
            if choice == "b":
                return "back"
            if choice == "s":
                self._review([(course.topics, course.label)], course.label)
                continue
            self._review([((course.topics[choice],), course.label)], course.topics[choice])

    def _review(self, segments, label) -> None:
        segments = [(topics, context) for topics, context in segments if any(t.strip() for t in topics)]
        total = sum(len(topics) for topics, _ in segments)
        if not total:
            self.print("  (nothing to search here)")
            return
        self.print(f"\nSearching {total} topic(s): {label}")
        outcomes = []
        for topics, context in segments:
            report = self.learner.learn_topics(topics, limit=None, dry_run=True, context=context)
            outcomes.extend(report.outcomes)
        report = LearnerReport(tuple(outcomes), dry_run=True, saved=False)
        self.print(report.summary())
        for outcome in report.outcomes:
            accepted = set(outcome.added)
            if not outcome.results:
                continue
            self.print(f"\n  {outcome.topic}:")
            for result in outcome.results:
                mark = "accepted" if result.url in accepted else "skipped"
                self.print(f"    [{mark}] {result.title} <{result.url}>")
                if result.snippet:
                    self.print(f"        {result.snippet}")
        if report.added_count == 0:
            self.print("\n  Nothing to save.\n")
            return
        answer = self.input(f"  {report.added_count} statement(s) to save. [s]ave / [i]gnore: ")
        if answer.strip().casefold() in {"s", "save", "y", "yes"}:
            stored = self.learner.commit(report)
            self.print(f"  Saved {stored} statement(s) to the project store.")
            if self.promote_path is not None:
                promoted = self.learner.promote(report, self.promote_path)
                self.print(f"  Promoted {promoted} new statement(s) into {self.promote_path}.")
            self.print("")
        else:
            self.print("  Ignored; nothing was saved.\n")

    def _pick(self, title, labels, extra):
        self.print(f"\n{title}:")
        for index, label in enumerate(labels, 1):
            self.print(f"  {index}) {label}")
        for key, description in extra.items():
            self.print(f"  {key}) {description}")
        while True:
            raw = self.input("Choose: ").strip().casefold()
            if raw in extra:
                return raw
            if raw.isdigit() and 1 <= int(raw) <= len(labels):
                return int(raw) - 1
            self.print("  Invalid choice, try again.")


def _all_segments(branches):
    return [
        (course.topics, course.label) for branch in branches for course in branch.courses
    ]


def _branch_segments(branch):
    return [(course.topics, course.label) for course in branch.courses]


def main(argv: list[str] | None = None) -> int:
    from core.internet_learner import build_internet_learner, corpus_topics

    parser = argparse.ArgumentParser(
        description="Learn petroleum knowledge from trusted internet sites, into the data store."
    )
    parser.add_argument("--approve", action="store_true", help="grant the external search capability")
    parser.add_argument("--write", action="store_true", help="save accepted statements (default: dry run)")
    parser.add_argument("--list-topics", action="store_true", help="print the corpus-derived topics and exit")
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="pick science branches/courses/concepts with numbers, search, then save or ignore",
    )
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
    parser.add_argument(
        "--promote",
        action="store_true",
        help="with --interactive, save approved statements into knowledge/learned_petroleum.json so they ship with the project",
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
    if args.interactive:
        from core.learned_knowledge import learned_path

        promote_path = learned_path() if args.promote else None
        return InteractiveLearner(learner, promote_path=promote_path).run()
    report = learner.learn_topics(topics, limit=limit, dry_run=not args.write)
    _print_report(report, as_json=args.json)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
