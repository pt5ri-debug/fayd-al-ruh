"""Tests for the continuous internet learner.

The learner searches trusted petroleum sites and folds accepted statements into
the project's data store. These tests pin the safety properties: search is denied
by default, only trusted domains count, corpus-gated statements are the only ones
accepted, and a dry run never writes.
"""

from __future__ import annotations

from core.documents import DocumentStore
from core.internet_learner import (
    build_internet_learner,
    corpus_topics,
    is_trusted,
)
from core.web_source import StaticSearchProvider, WebResult
from core.web_verifier import approved_gate, granted_permissions

_TRUSTED_URL = "https://www.spe.org/petrophysics/permeability"
_TRUSTED_SNIPPET = (
    "Permeability measures how easily fluid flows through the connected pores "
    "of the reservoir rock, alongside porosity which measures void space."
)
_UNTRUSTED_URL = "https://randompetroleumblog.com/permeability"
_OFF_TOPIC_SNIPPET = "This page is about gardening and growing tomatoes in spring."


def _store() -> DocumentStore:
    store = DocumentStore()
    store.add_text(
        "note:porosity",
        "Porosity",
        "Porosity measures the void space in reservoir rock; permeability measures flow.",
        source="notes",
    )
    return store


def _result(url: str, snippet: str, credibility: float = 0.9) -> WebResult:
    return WebResult(url=url, title="Petroleum note", snippet=snippet, credibility=credibility)


def _learner(store, results, *, approved=True, trusted_only=True, persist=None):
    return build_internet_learner(
        provider=StaticSearchProvider(tuple(results)),
        store=store,
        permissions=granted_permissions() if approved else None,
        approvals=approved_gate() if approved else None,
        trusted_only=trusted_only,
        min_credibility=0.6,
        per_topic=3,
        persist=persist,
    )


# --- domain trust ---------------------------------------------------------


def test_is_trusted_accepts_standards_and_academic_hosts():
    assert is_trusted("https://www.spe.org/x")
    assert is_trusted("https://sub.onepetro.org/x")
    assert is_trusted("https://en.wikipedia.org/wiki/Permeability")
    assert is_trusted("https://petroleum.stanford.edu/x")
    assert is_trusted("https://www.energy.gov/x")


def test_is_trusted_rejects_generic_hosts():
    assert not is_trusted("https://randompetroleumblog.com/x")
    assert not is_trusted("")
    assert not is_trusted("https://example.net/x")


# --- the default is off ---------------------------------------------------


def test_search_is_denied_without_approval():
    store = _store()
    learner = _learner(store, [_result(_TRUSTED_URL, _TRUSTED_SNIPPET)], approved=False)
    report = learner.learn_topics(["Permeability"], dry_run=True)
    assert report.search_allowed is False
    assert report.added_count == 0
    assert len(store) == 1  # nothing changed


# --- trusted domains only -------------------------------------------------


def test_untrusted_domain_is_skipped():
    store = _store()
    learner = _learner(store, [_result(_UNTRUSTED_URL, _TRUSTED_SNIPPET)])
    report = learner.learn_topics(["Permeability"], dry_run=False)
    outcome = report.outcomes[0]
    assert outcome.added == ()
    assert any(url == _UNTRUSTED_URL for url, _reason in outcome.skipped)


def test_trusted_statement_is_added_on_write():
    store = _store()
    persisted: list[str] = []
    learner = _learner(store, [_result(_TRUSTED_URL, _TRUSTED_SNIPPET)], persist=lambda doc: persisted.append(doc.doc_id))
    report = learner.learn_topics(["Permeability"], dry_run=False)
    assert report.added_count == 1
    assert len(store) == 2
    assert persisted and persisted[0].startswith("learned:")


# --- dry run never writes -------------------------------------------------


def test_dry_run_reports_but_saves_nothing():
    store = _store()
    persisted: list[str] = []
    learner = _learner(store, [_result(_TRUSTED_URL, _TRUSTED_SNIPPET)], persist=lambda doc: persisted.append(doc.doc_id))
    report = learner.learn_topics(["Permeability"], dry_run=True)
    assert report.added_count == 1  # it would add one
    assert report.saved is False
    assert len(store) == 1  # but the store is untouched
    assert persisted == []


def test_off_topic_statement_is_rejected_even_when_trusted():
    store = _store()
    learner = _learner(store, [_result(_TRUSTED_URL, _OFF_TOPIC_SNIPPET)])
    report = learner.learn_topics(["Permeability"], dry_run=False)
    assert report.added_count == 0
    assert len(store) == 1


# --- topics come from the corpus -----------------------------------------


def test_corpus_topics_are_petroleum_relevant_and_non_empty():
    topics = corpus_topics()
    assert topics
    assert any("permeability" in topic.casefold() or "reservoir" in topic.casefold() for topic in topics)


def test_limit_caps_the_batch():
    store = _store()
    learner = _learner(store, [_result(_TRUSTED_URL, _TRUSTED_SNIPPET)])
    report = learner.learn_topics(corpus_topics(), limit=2, dry_run=True)
    assert report.topics_considered == 2
