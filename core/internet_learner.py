"""Continuous internet learning: grow the corpus from trusted petroleum sites.

The existing web path (:mod:`core.web_verifier`) answers *one* question and may
learn from it. This module generalises that into a batch, topic-driven learner:
it takes a list of petroleum topics, searches the internet for each, keeps only
results from trusted domains, and folds accepted statements into the project's
document store through the same governed updater the rest of the system uses.

Four rules keep it safe, all inherited rather than reinvented:

* **Denied by default** -- every search goes through ``WebSearchService``, so it
  runs only when a permission policy grants the external capability *and* a live
  human approval exists. A plain call reaches nothing.
* **Trusted domains only** -- a result whose host is not on the explicit trusted
  list (and not a government, academic, or standards host) is discarded, so a
  random blog can never enter the store.
* **Corpus-gated acceptance** -- ``KnowledgeUpdater`` rejects any statement that
  shares no terms with what the project already knows, so off-topic text cannot
  flood the store.
* **Dry-run by default** -- the learner only reports what it *would* add unless
  the caller asks to persist.

Topics come from the project's own petroleum corpus (:func:`corpus_topics`), so
the material learned is relevant to petroleum engineering by construction, not
by a hand-written question list.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable
from urllib.parse import urlparse

from core.documents import Document, DocumentStore
from core.knowledge_update import KnowledgeUpdate, KnowledgeUpdater, UpdateOutcome
from core.web_source import WebResult, WebSearchService

# Hosts we treat as authoritative for petroleum engineering and general facts.
_TRUSTED_HOSTS = (
    "spe.org",
    "onepetro.org",
    "api.org",
    "iso.org",
    "astm.org",
    "wikipedia.org",
)
# Top-level suffixes that mark a source as government or academic.
_TRUSTED_SUFFIXES = (".gov", ".edu", ".mil", ".ac.uk", ".gov.uk")


def is_trusted(url: str) -> bool:
    """Whether a URL's host is on the trusted list or a government/academic host."""
    host = urlparse(url).netloc.casefold()
    if not host:
        return False
    if host.endswith(_TRUSTED_SUFFIXES):
        return True
    return any(host == trusted or host.endswith(f".{trusted}") for trusted in _TRUSTED_HOSTS)


def corpus_topics() -> tuple[str, ...]:
    """The petroleum topics to learn about, taken from the corpus itself.

    Each corpus entry's title is a ready-made, domain-relevant search topic, so
    the list grows automatically as the corpus grows.
    """
    from core.petroleum_corpus import _ENTRIES

    seen: list[str] = []
    for entry in _ENTRIES:
        if entry.title not in seen:
            seen.append(entry.title)
    return tuple(seen)


@dataclass(frozen=True)
class TopicOutcome:
    """What happened for one topic: the search, and what was (or would be) added."""

    topic: str
    query: str
    allowed: bool
    reason: str
    results: tuple[WebResult, ...]
    added: tuple[str, ...]
    skipped: tuple[tuple[str, str], ...]
    dry_run: bool

    @property
    def learned(self) -> bool:
        return bool(self.added)


@dataclass(frozen=True)
class LearnerReport:
    """The result of one learning batch."""

    outcomes: tuple[TopicOutcome, ...]
    dry_run: bool
    saved: bool

    @property
    def topics_considered(self) -> int:
        return len(self.outcomes)

    @property
    def added_count(self) -> int:
        return sum(len(outcome.added) for outcome in self.outcomes)

    @property
    def search_allowed(self) -> bool:
        return any(outcome.allowed for outcome in self.outcomes)

    def summary(self) -> str:
        mode = "dry run (nothing saved)" if self.dry_run else "applied"
        lines = [
            f"Continuous internet learner: {self.topics_considered} topic(s), "
            f"{self.added_count} statement(s) {'to add' if self.dry_run else 'added'} [{mode}]."
        ]
        if not self.search_allowed:
            lines.append("  (no search ran: grant the external capability with --approve)")
        for outcome in self.outcomes:
            if not outcome.allowed:
                lines.append(f"  - {outcome.topic}: search not allowed ({outcome.reason})")
                continue
            state = "would add" if outcome.dry_run else "added"
            lines.append(
                f"  - {outcome.topic}: {len(outcome.results)} result(s), "
                f"{len(outcome.added)} {state}, {len(outcome.skipped)} skipped"
            )
            for url, reason in outcome.skipped:
                lines.append(f"      skipped {url}: {reason}")
        return "\n".join(lines)


class InternetLearner:
    """Search trusted sites for each topic and fold accepted statements in."""

    def __init__(
        self,
        service: WebSearchService,
        updater: KnowledgeUpdater,
        *,
        trusted_only: bool = True,
        min_credibility: float = 0.6,
        per_topic: int = 3,
        persist: Callable[[Document], None] | None = None,
    ) -> None:
        if not 0.0 <= min_credibility <= 1.0:
            raise ValueError("min_credibility must be between 0 and 1.")
        if per_topic < 1:
            raise ValueError("per_topic must be at least 1.")
        self.service = service
        self.updater = updater
        self.trusted_only = trusted_only
        self.min_credibility = min_credibility
        self.per_topic = per_topic
        self.persist = persist

    def learn_topics(
        self,
        topics: Iterable[str],
        *,
        limit: int | None = None,
        dry_run: bool = True,
    ) -> LearnerReport:
        """Run one batch over ``topics``; ``limit`` caps how many topics run.

        A dry run still exercises the real acceptance logic, but against a
        scratch copy of the store, so it can say what *would* be added without
        mutating the project's actual knowledge.
        """
        chosen = [topic for topic in topics if topic.strip()]
        if limit is not None:
            chosen = chosen[:limit]
        updater = self.updater
        if dry_run:
            updater = self._scratch_updater()
        outcomes: list[TopicOutcome] = []
        added_any = False
        for topic in chosen:
            outcome = self._learn_topic(topic, updater=updater, dry_run=dry_run)
            outcomes.append(outcome)
            if outcome.added:
                added_any = True
        saved = added_any and not dry_run
        return LearnerReport(tuple(outcomes), dry_run=dry_run, saved=saved)

    def _scratch_updater(self) -> KnowledgeUpdater:
        """A throwaway updater over a copy of the store, for dry runs."""
        scratch = DocumentStore()
        for document in self.updater.store.documents():
            scratch.add(document)
        return KnowledgeUpdater(scratch)

    def _learn_topic(self, topic: str, *, updater: KnowledgeUpdater, dry_run: bool) -> TopicOutcome:
        search = self.service.search(topic, limit=self.per_topic * 3)
        if not search.allowed:
            return TopicOutcome(topic, topic, False, search.reason, (), (), (), dry_run)
        added: list[str] = []
        skipped: list[tuple[str, str]] = []
        for result in search.results:
            if len(added) >= self.per_topic:
                break
            reason = self._rejection(result)
            if reason is not None:
                skipped.append((result.url, reason))
                continue
            outcome = self._apply(topic, result, updater=updater, dry_run=dry_run)
            if outcome.action in {"added", "reinforced"}:
                added.append(result.url)
            else:
                skipped.append((result.url, outcome.reason or outcome.action))
        return TopicOutcome(topic, topic, True, search.reason, search.results, tuple(added), tuple(skipped), dry_run)

    def _rejection(self, result: WebResult) -> str | None:
        if self.trusted_only and not is_trusted(result.url):
            return "domain not trusted"
        if result.credibility < self.min_credibility:
            return f"credibility {result.credibility:.2f} below {self.min_credibility:.2f}"
        if not result.snippet.strip():
            return "empty snippet"
        return None

    def _apply(self, topic: str, result: WebResult, *, updater: KnowledgeUpdater, dry_run: bool) -> UpdateOutcome:
        statement = " ".join(result.snippet.split()) or f"{result.title} ({result.url})"
        outcome = updater.apply(
            KnowledgeUpdate(
                statement=statement,
                source=result.url,
                strength=result.credibility,
                topic=topic,
            )
        )
        if not dry_run and outcome.action in {"added", "reinforced"} and self.persist is not None:
            document = self.updater.store.get(outcome.doc_id)
            if document is not None:
                self.persist(document)
        return outcome


def build_internet_learner(
    *,
    provider,
    store,
    permissions=None,
    approvals=None,
    audit=None,
    persist: Callable[[Document], None] | None = None,
    trusted_only: bool = True,
    min_credibility: float = 0.6,
    per_topic: int = 3,
) -> InternetLearner:
    """Assemble the learner over the project's own corpus and governance.

    ``store`` is the shared document store (the project's data repository); the
    same ``KnowledgeUpdater`` gate the web verifier uses is applied here, so a
    statement enters the store only when it connects to existing knowledge.
    """
    service = WebSearchService(provider, permissions=permissions, approvals=approvals, audit=audit)
    updater = KnowledgeUpdater(store)
    if persist is None and hasattr(store, "save"):
        persist = store.save
    return InternetLearner(
        service,
        updater,
        trusted_only=trusted_only,
        min_credibility=min_credibility,
        per_topic=per_topic,
        persist=persist,
    )


__all__ = [
    "InternetLearner",
    "LearnerReport",
    "TopicOutcome",
    "build_internet_learner",
    "corpus_topics",
    "is_trusted",
]
