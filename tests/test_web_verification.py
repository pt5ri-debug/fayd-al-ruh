"""Tests for governed web search, answer verification, and trusted learning."""

from __future__ import annotations

import pytest

from core.human_approval import HumanApprovalGate
from core.permissions import Capability, PermissionPolicy
from core.web_source import (
    HttpSearchProvider,
    StaticSearchProvider,
    WebResult,
    WebSearchService,
    estimate_credibility,
)
from core.web_verifier import WebVerifier, build_web_verifier

_RELATED = WebResult(
    url="https://www.spe.org/petrophysics/permeability",
    title="Porosity and permeability relationship",
    snippet=(
        "Porosity measures the void space in the rock, while permeability measures "
        "how easily fluid flows through the connected pores of the reservoir formation."
    ),
    credibility=0.9,
)
_AGREEING = WebResult(
    url="https://en.wikipedia.org/wiki/Permeability_(earth_sciences)",
    title="Permeability in petroleum reservoirs",
    snippet=(
        "In petroleum reservoirs, permeability measures how easily fluid flows through "
        "porous rock, while porosity measures the fraction of void space in the formation."
    ),
    credibility=0.7,
)
_OFF_TOPIC = WebResult(
    url="https://randomblog.example.com/cooking",
    title="A cooking blog",
    snippet="Simmer the sauce for ten minutes and season with salt.",
    credibility=0.4,
)
_CONTRADICTING = WebResult(
    url="https://www.spe.org/disputed",
    title="Disputed claim about porosity",
    snippet="Porosity does not measure the void space in rock.",
    credibility=0.9,
)


def _granted() -> tuple[PermissionPolicy, HumanApprovalGate]:
    policy = PermissionPolicy(
        allowed=frozenset(
            {Capability.SYMBOLIC_SIMULATION, Capability.SANDBOX_TASK, Capability.EXTERNAL_TOOL}
        ),
        allowed_external_tools=frozenset({"web_search"}),
    )
    approvals = HumanApprovalGate()
    approvals.request("web_search", requested_by="user").approve("reviewer")
    return policy, approvals


# --- credibility heuristic ------------------------------------------------


def test_estimate_credibility_ranks_authoritative_sources_higher():
    assert estimate_credibility("https://www.spe.org/x") == 0.9
    assert estimate_credibility("https://www.energy.gov/x") == 0.9
    assert estimate_credibility("https://en.wikipedia.org/wiki/x") == 0.7
    assert estimate_credibility("https://someblog.com/x") == 0.4


# --- governance -----------------------------------------------------------


def test_search_is_denied_by_default():
    service = WebSearchService(StaticSearchProvider((_RELATED,)))
    outcome = service.search("porosity permeability")

    assert not outcome.allowed
    assert outcome.results == ()
    assert service.audit.denials()


def test_search_requires_approval_even_when_capability_granted():
    policy = PermissionPolicy(
        allowed=frozenset({Capability.EXTERNAL_TOOL}),
        allowed_external_tools=frozenset({"web_search"}),
    )
    service = WebSearchService(StaticSearchProvider((_RELATED,)), permissions=policy)
    outcome = service.search("porosity")

    assert not outcome.allowed
    assert "approval" in outcome.reason.casefold()


def test_granted_and_approved_search_runs_and_audits():
    policy, approvals = _granted()
    service = WebSearchService(StaticSearchProvider((_RELATED,)), permissions=policy, approvals=approvals)
    outcome = service.search("porosity permeability")

    assert outcome.allowed
    assert outcome.result_count == 1
    assert any(record.outcome.value == "allowed" for record in service.audit.records)


def test_provider_failure_is_contained():
    class _Broken:
        name = "broken"

        def search(self, query: str, *, limit: int = 5, offset: int = 0, fresh: bool = False):
            raise RuntimeError("network down")

    policy, approvals = _granted()
    service = WebSearchService(_Broken(), permissions=policy, approvals=approvals)
    outcome = service.search("porosity")

    assert not outcome.allowed
    assert "failed" in outcome.reason.casefold()


def test_http_provider_parses_json_results(monkeypatch):
    payload = {"results": [{"url": "https://www.spe.org/x", "title": "SPE page", "snippet": "oil and gas"}]}

    class _Response:
        def raise_for_status(self) -> None:
            return None

        def json(self):
            return payload

    import requests

    monkeypatch.setattr(requests, "get", lambda *a, **k: _Response())
    results = HttpSearchProvider("https://api.example.com/search").search("oil")

    assert len(results) == 1
    assert results[0].credibility == 0.9


# --- verification ---------------------------------------------------------


def _verifier(results, **kwargs) -> WebVerifier:
    policy, approvals = _granted()
    return build_web_verifier(
        provider=StaticSearchProvider(tuple(results)),
        permissions=policy,
        approvals=approvals,
        **kwargs,
    )


def test_trusted_answer_is_cited_compared_and_learned():
    verifier = _verifier((_RELATED, _AGREEING))
    before = len(verifier.updater.store)
    assessment = verifier.verify("What is the relationship between porosity and permeability?")

    assert assessment.search_allowed
    assert assessment.citations  # every answer cites its sources
    assert assessment.internal_passages  # compared against internal knowledge
    assert assessment.trust > 0.6
    assert assessment.learned
    assert assessment.learned_doc_id is not None
    assert len(verifier.updater.store) == before + 1


def test_low_trust_answer_is_not_learned():
    verifier = _verifier((_OFF_TOPIC,))
    before = len(verifier.updater.store)
    assessment = verifier.verify("What is the relationship between porosity and permeability?")

    assert not assessment.learned
    assert assessment.learned_doc_id is None
    assert len(verifier.updater.store) == before


def test_contradicting_source_is_never_learned():
    verifier = _verifier((_CONTRADICTING,))
    assessment = verifier.verify("What is the relationship between porosity and permeability?")

    assert assessment.contradicts_internal
    assert not assessment.learned


def test_denied_search_returns_ungrounded_assessment():
    policy = PermissionPolicy()  # default denies external tools
    from core.rag import RagRetriever
    from core.documents import build_document_store

    retriever = RagRetriever(build_document_store())
    verifier = WebVerifier(retriever, WebSearchService(StaticSearchProvider((_RELATED,)), permissions=policy))
    assessment = verifier.verify("porosity")

    assert not assessment.search_allowed
    assert assessment.trust == 0.0
    assert not assessment.learned
    assert "denied" in assessment.reason.casefold()


def test_learning_can_be_disabled():
    verifier = _verifier((_RELATED, _AGREEING))
    assessment = verifier.verify(
        "What is the relationship between porosity and permeability?", learn=False
    )

    assert assessment.trust > 0.6
    assert not assessment.learned


def test_verifier_rejects_empty_question():
    verifier = _verifier((_RELATED,))
    with pytest.raises(ValueError):
        verifier.verify("   ")


def test_assessment_requires_scores_in_range():
    from core.web_verifier import WebAssessment

    with pytest.raises(ValueError):
        WebAssessment(
            question="q",
            answer="a",
            citations=(),
            sources=(),
            internal_passages=(),
            credibility=1.5,
            agreement=0.0,
            support=0.0,
            trust=0.0,
            learned=False,
        )


# --- internet providers ---------------------------------------------------


def test_wikipedia_provider_parses_search_results(monkeypatch):
    payload = {
        "query": {
            "search": [
                {"title": "Gas kick", "snippet": "<span>gas</span> kick", "pageid": 123},
                {"title": "Blowout", "snippet": "loss of <b>well</b> control"},
            ]
        }
    }

    class _Response:
        def raise_for_status(self) -> None:
            return None

        def json(self):
            return payload

    import requests

    captured: dict[str, object] = {}

    def _fake_get(url, **kwargs):
        captured["url"] = url
        captured.update(kwargs)
        return _Response()

    monkeypatch.setattr(requests, "get", _fake_get)
    from core.web_source import WikipediaSearchProvider

    results = WikipediaSearchProvider().search("gas kick", limit=2)

    assert [result.title for result in results] == ["Gas kick", "Blowout"]
    assert results[0].snippet == "gas kick"  # HTML tags stripped
    assert results[0].url.endswith("curid=123")
    assert "User-Agent" in captured["headers"]  # Wikipedia requires a UA


def test_fallback_provider_uses_backup_on_failure_and_on_empty():
    from core.web_source import FallbackSearchProvider, StaticSearchProvider

    class _Broken:
        name = "broken"

        def search(self, query: str, *, limit: int = 5, offset: int = 0, fresh: bool = False):
            raise RuntimeError("offline")

    class _Empty:
        name = "empty"

        def search(self, query: str, *, limit: int = 5, offset: int = 0, fresh: bool = False):
            return ()

    backup = StaticSearchProvider((_RELATED,))
    assert FallbackSearchProvider(_Broken(), backup).search("porosity permeability") == (_RELATED,)
    assert FallbackSearchProvider(_Empty(), backup).search("porosity permeability") == (_RELATED,)


def test_web_cli_prompts_for_the_question(monkeypatch):
    import builtins

    import ai.web_verify as web_verify

    monkeypatch.setattr(builtins, "input", lambda prompt="": "What is a gas kick?")
    captured: dict[str, str] = {}

    class _FakeVerifier:
        def verify(self, question, **kwargs):
            captured["question"] = question
            return web_verify.WebAssessment(
                question=question,
                answer="",
                citations=(),
                sources=(),
                internal_passages=(),
                credibility=0.0,
                agreement=0.0,
                support=0.0,
                trust=0.0,
                learned=False,
            )

    monkeypatch.setattr(web_verify, "build_web_verifier", lambda **kwargs: _FakeVerifier())
    code = web_verify.main(["--approve"])

    assert code == 0
    assert captured["question"] == "What is a gas kick?"


def test_web_cli_uses_fallback_internet_provider_by_default(monkeypatch):
    import ai.web_verify as web_verify

    captured: dict[str, object] = {}

    class _FakeVerifier:
        def verify(self, question, **kwargs):
            return web_verify.WebAssessment(
                question=question,
                answer="",
                citations=(),
                sources=(),
                internal_passages=(),
                credibility=0.0,
                agreement=0.0,
                support=0.0,
                trust=0.0,
                learned=False,
            )

    def _fake_build(*, provider, **kwargs):
        captured["provider"] = provider
        return _FakeVerifier()

    monkeypatch.setattr(web_verify, "build_web_verifier", _fake_build)
    web_verify.main(["question"])

    from core.web_source import FallbackSearchProvider

    assert isinstance(captured["provider"], FallbackSearchProvider)
    assert captured["provider"].primary.name == "wikipedia"
