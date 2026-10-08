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


def _result(url: str, snippet: str, credibility: float = 0.9, title: str = "Permeability") -> WebResult:
    return WebResult(url=url, title=title, snippet=snippet, credibility=credibility)


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


def test_result_not_about_the_topic_is_rejected():
    store = _store()
    # Trusted, petroleum-flavoured prose, but the article is a water reservoir.
    snippet = "Permeability of the reservoir rock controls groundwater flow in the aquifer."
    learner = _learner(
        store, [_result(_TRUSTED_URL, snippet, title="Pandan Reservoir MRT station")]
    )
    report = learner.learn_topics(["Permeability"], dry_run=False)
    assert report.added_count == 0
    assert any("not about" in reason for _url, reason in report.outcomes[0].skipped)


def test_formula_fragment_is_rejected():
    store = _store()
    snippet = r"{\displaystyle \phi} is the effective porosity and k is the absolute permeability."
    learner = _learner(store, [_result(_TRUSTED_URL, snippet)])
    report = learner.learn_topics(["Permeability"], dry_run=False)
    assert report.added_count == 0
    assert any(reason == "formula fragment" for _url, reason in report.outcomes[0].skipped)


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


# --- science-branch navigator --------------------------------------------


def test_science_branches_are_ordered_and_populated():
    from core.knowledge_navigator import science_branches

    branches = science_branches()
    assert branches
    assert branches[0].label == "Petroleum Disciplines"
    labels = [branch.label for branch in branches]
    assert labels.index("Petroleum Disciplines") < labels.index("Foundations")
    assert all(branch.courses for branch in branches)
    assert all(course.topics for branch in branches for course in branch.courses)


def test_concept_topics_are_readable_phrases():
    from core.knowledge_navigator import science_branches

    topics = [
        topic for branch in science_branches() for course in branch.courses for topic in course.topics
    ]
    assert topics
    assert all("_" not in topic for topic in topics)
    assert any(topic == "effective porosity" for topic in topics)


# --- commit persists a preview without searching again --------------------


class _CountingProvider(StaticSearchProvider):
    def __init__(self, results):
        super().__init__(tuple(results))
        self.calls = 0

    def search(self, query, *, limit=5):
        self.calls += 1
        return super().search(query, limit=limit)


class _EchoProvider(StaticSearchProvider):
    """Returns a result whose title is the query, so it is always on-topic."""

    def __init__(self, snippet, url=_TRUSTED_URL, credibility=0.9):
        super().__init__()
        self._snippet = snippet
        self._url = url
        self._credibility = credibility

    def search(self, query, *, limit=5):
        return (
            WebResult(
                url=self._url,
                title=query,
                snippet=self._snippet,
                credibility=self._credibility,
            ),
        )


def test_commit_saves_previewed_statements_without_searching_again():
    store = _store()
    provider = _CountingProvider([_result(_TRUSTED_URL, _TRUSTED_SNIPPET)])
    persisted: list[str] = []
    learner = build_internet_learner(
        provider=provider,
        store=store,
        permissions=granted_permissions(),
        approvals=approved_gate(),
        persist=lambda doc: persisted.append(doc.doc_id),
    )
    report = learner.learn_topics(["Permeability"], dry_run=True)
    assert report.added_count == 1
    calls_after_preview = provider.calls
    assert len(store) == 1

    stored = learner.commit(report)
    assert stored == 1
    assert len(store) == 2
    assert provider.calls == calls_after_preview  # no second search
    assert persisted and persisted[0].startswith("learned:")


def test_commit_is_a_noop_for_a_non_dry_run_report():
    store = _store()
    learner = _learner(store, [_result(_TRUSTED_URL, _TRUSTED_SNIPPET)])
    report = learner.learn_topics(["Permeability"], dry_run=False)
    assert learner.commit(report) == 0


# --- promote into a committed project file --------------------------------


def test_promote_writes_accepted_statements_to_the_project_file(tmp_path):
    import json

    store = _store()
    learner = _learner(store, [_result(_TRUSTED_URL, _TRUSTED_SNIPPET)])
    report = learner.learn_topics(["Permeability"], dry_run=True)
    path = tmp_path / "knowledge" / "learned_petroleum.json"
    added = learner.promote(report, path)
    assert added == 1
    records = json.loads(path.read_text(encoding="utf-8"))
    assert len(records) == 1
    assert records[0]["doc_id"].startswith("learned:")
    assert records[0]["metadata"]["topic"] == "Permeability"

    # Promoting again is idempotent: the same statement is not duplicated.
    assert learner.promote(report, path) == 0
    assert len(json.loads(path.read_text(encoding="utf-8"))) == 1


def test_learned_documents_loader_reads_promoted_file(tmp_path, monkeypatch):
    import json

    from core import learned_knowledge

    path = tmp_path / "learned_petroleum.json"
    path.write_text(
        json.dumps(
            [
                {
                    "doc_id": "learned:test1234",
                    "title": "Learned statement: permeability",
                    "text": "Permeability measures how easily fluid flows through the rock.",
                    "source": "learned/https://www.spe.org/x",
                    "metadata": {"topic": "permeability"},
                }
            ]
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(learned_knowledge, "learned_path", lambda: path)
    documents = learned_knowledge.learned_documents()
    assert len(documents) == 1
    assert documents[0].doc_id == "learned:test1234"


def test_learned_documents_loader_is_empty_when_missing(tmp_path, monkeypatch):
    from core import learned_knowledge

    monkeypatch.setattr(learned_knowledge, "learned_path", lambda: tmp_path / "nope.json")
    assert learned_knowledge.learned_documents() == []


# --- interactive walker ----------------------------------------------------


def test_interactive_drill_down_search_and_save():
    from ai.internet_learner import InteractiveLearner

    store = _store()
    provider = _EchoProvider(_TRUSTED_SNIPPET)
    persisted: list[str] = []
    learner = build_internet_learner(
        provider=provider,
        store=store,
        permissions=granted_permissions(),
        approvals=approved_gate(),
        persist=lambda doc: persisted.append(doc.doc_id),
    )
    # branches -> Petroleum Disciplines (1) -> Petrophysics course (1) ->
    # first concept (1) -> search happens -> save (s)
    scripted = iter(["1", "1", "1", "s", "q"])
    out: list[str] = []
    code = InteractiveLearner(
        learner, input_func=lambda _prompt: next(scripted), print_func=out.append
    ).run()
    assert code == 0
    assert any("Science branches" in line for line in out)
    assert any("[accepted]" in line for line in out)
    assert len(store) == 2  # one learned statement was saved


def test_interactive_ignore_saves_nothing():
    from ai.internet_learner import InteractiveLearner

    store = _store()
    provider = _EchoProvider(_TRUSTED_SNIPPET)
    learner = build_internet_learner(
        provider=provider,
        store=store,
        permissions=granted_permissions(),
        approvals=approved_gate(),
    )
    scripted = iter(["1", "1", "1", "i", "q"])
    code = InteractiveLearner(
        learner, input_func=lambda _p: next(scripted), print_func=lambda _l: None
    ).run()
    assert code == 0
    assert len(store) == 1  # ignored


def test_interactive_promote_writes_into_the_project_file(tmp_path):
    from ai.internet_learner import InteractiveLearner

    store = _store()
    provider = _EchoProvider(_TRUSTED_SNIPPET)
    learner = build_internet_learner(
        provider=provider,
        store=store,
        permissions=granted_permissions(),
        approvals=approved_gate(),
    )
    promote_path = tmp_path / "learned_petroleum.json"
    scripted = iter(["1", "1", "1", "s", "q"])
    out: list[str] = []
    code = InteractiveLearner(
        learner,
        input_func=lambda _p: next(scripted),
        print_func=out.append,
        promote_path=promote_path,
    ).run()
    assert code == 0
    assert promote_path.exists()
    assert any("Promoted" in line for line in out)
