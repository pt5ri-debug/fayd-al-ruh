"""Tests for internet-grounded conversation: search, compare, and learn.

These exercise the wiring added on top of :mod:`core.web_verifier`: the
assistant consults the web only when its own evidence is weak, compares the
result with internal knowledge, and folds trusted answers into the same store
it retrieves from -- while an off-topic web answer is shown but never learned.
"""

from __future__ import annotations

from core.assistant import GroundedAssistant, build_default_assistant
from core.documents import DocumentStore
from core.rag import RagRetriever
from core.web_source import StaticSearchProvider, WebResult
from core.web_verifier import (
    approved_gate,
    build_web_verifier,
    granted_permissions,
)

# Uses corpus terms (porosity/permeability/reservoir formation) so a trusted
# statement is accepted by the updater, which only learns statements that share
# vocabulary with the existing corpus.
_TRUSTED_URL = "https://www.spe.org/petrophysics/archie"
_TRUSTED_SNIPPET = (
    "The Archie equation estimates water saturation from resistivity in the "
    "reservoir rock using porosity and the formation water resistivity."
)


def _store() -> DocumentStore:
    store = DocumentStore()
    store.add_text(
        "note:porosity",
        "Porosity",
        "Porosity measures the void space in reservoir rock and is measured in the formation.",
        source="notes",
    )
    return store


def _provider(results) -> StaticSearchProvider:
    return StaticSearchProvider(tuple(results))


def _augmented(store, results, *, trust_threshold=0.5, weak_evidence=0.5):
    # Build the assistant first so the verifier shares its exact retriever
    # (production wiring), then turn the web on and lower the "weak" bar so a
    # deliberately thin store still triggers a search.
    assistant = build_default_assistant(documents=store)
    web = build_web_verifier(
        provider=_provider(results),
        permissions=granted_permissions(),
        approvals=approved_gate(),
        retriever=assistant.retriever,
        trust_threshold=trust_threshold,
    )
    assistant.web = web
    assistant.web_enabled = True
    assistant.weak_evidence = weak_evidence
    assistant._on_learn = store.save
    return assistant, assistant.retriever


# --- assistant consults the web only when internal evidence is weak -------


def test_strong_internal_evidence_does_not_search():
    store = _store()
    provider_calls: list[str] = []

    class _RecordingProvider(StaticSearchProvider):
        def search(self, query, *, limit=5):  # type: ignore[override]
            provider_calls.append(query)
            return super().search(query, limit=limit)

    retriever = RagRetriever(store)
    web = build_web_verifier(
        provider=_RecordingProvider((_TRUSTED_URL,)),
        permissions=granted_permissions(),
        approvals=approved_gate(),
        retriever=retriever,
    )
    assistant = GroundedAssistant(retriever, web=web, web_enabled=True)

    # A question the tiny corpus answers well: its own document shares the
    # question's terms, so the top passage is strong and no search is needed.
    answer = assistant.ask("What does porosity measure in the void space?")
    assert not answer.used_web
    assert provider_calls == []


def test_weak_evidence_searches_and_cites_web_sources():
    store = _store()
    assistant, _ = _augmented(
        store,
        [
            WebResult(_TRUSTED_URL, "Archie water saturation", _TRUSTED_SNIPPET, credibility=0.9),
        ],
    )
    answer = assistant.ask("How does the Archie equation find water saturation in reservoir rock?")

    assert answer.used_web
    assert any("spe.org" in citation or "Archie" in citation for citation in answer.citations)
    assert "Water saturation" in answer.text or "http" in answer.text


def test_off_topic_web_answer_is_shown_but_not_learned():
    store = _store()
    before = len(store)
    assistant, _ = _augmented(
        store,
        [WebResult("https://randomblog.example.com/cooking", "Cooking", "Simmer the sauce.", credibility=0.4)],
    )
    answer = assistant.ask("How does the Archie equation find water saturation in reservoir rock?")

    assert answer.used_web
    assert not answer.learned
    assert len(store) == before


def test_trusted_web_answer_is_learned_and_retrievable_immediately():
    store = _store()
    assistant, retriever = _augmented(
        store,
        [
            WebResult(_TRUSTED_URL, "Archie water saturation", _TRUSTED_SNIPPET, credibility=0.9),
        ],
    )
    question = "How does the Archie equation find water saturation in reservoir rock?"
    answer = assistant.ask(question)

    assert answer.learned
    assert answer.learned_doc_id is not None
    assert store.get(answer.learned_doc_id) is not None
    # The freshly learned document is part of this very answer's evidence.
    assert any(passage.doc_id == answer.learned_doc_id for passage in answer.passages)
    # And the next retrieval has it indexed, so the same question no longer
    # depends on a fresh search.
    top = retriever.retrieve(question, limit=1)[0]
    assert top.doc_id == answer.learned_doc_id


def test_web_can_be_turned_off():
    store = _store()
    assistant, _ = _augmented(
        store,
        [WebResult(_TRUSTED_URL, "Archie water saturation", _TRUSTED_SNIPPET, credibility=0.9)],
    )
    assistant.web_enabled = False
    answer = assistant.ask("How does the Archie equation find water saturation in reservoir rock?")

    assert not answer.used_web
    assert not answer.learned


# --- the builder shares one corpus with the caller ------------------------


def test_builder_shares_the_corpus_with_the_supplied_retriever():
    store = _store()
    retriever = RagRetriever(store)
    web = build_web_verifier(retriever=retriever)
    assert web.retriever is retriever
    assert web.updater is not None
    assert web.updater.store is store


def test_builder_accepts_a_documents_store_alone():
    store = _store()
    web = build_web_verifier(documents=store)
    assert web.retriever.documents is store
    assert web.updater.store is store


# --- persistence ----------------------------------------------------------


def test_learned_document_is_persisted(tmp_path):
    from core.documents import build_document_store
    from storage.json_store import JsonStore

    store = build_document_store(store=JsonStore(tmp_path / "documents"))
    assistant, _ = _augmented(
        store,
        [WebResult(_TRUSTED_URL, "Archie water saturation", _TRUSTED_SNIPPET, credibility=0.9)],
        weak_evidence=0.9,
    )
    answer = assistant.ask("How does the Archie equation find water saturation in reservoir rock?")
    assert answer.learned

    # A brand-new store over the same directory sees the learned statement.
    reopened = build_document_store(store=JsonStore(tmp_path / "documents"))
    reopened.load_all()
    assert reopened.get(answer.learned_doc_id) is not None
