"""Tests for conversation memory, documents, RAG retrieval, and the assistant."""

from __future__ import annotations

import pytest

from core.assistant import GroundedAssistant
from core.conversation import Conversation, ConversationMemory, Message, Role
from core.documents import Document, DocumentStore, build_document_store, build_knowledge_documents
from core.model_interface import ModelResponse
from core.ontology import Ontology
from core.domain_knowledge import seed_ontology
from core.curriculum_knowledge import seed_curriculum
from core.rag import RagRetriever
from storage.json_store import JsonStore


# --- conversation ---------------------------------------------------------


def test_conversation_records_turns_with_roles():
    conversation = Conversation(title="اختبار")
    conversation.add_user("سؤال")
    conversation.add_assistant("جواب", ["[src:1] عنوان"])

    assert [message.role for message in conversation.messages] == [Role.USER, Role.ASSISTANT]
    assert conversation.last_user_message().text == "سؤال"
    assert conversation.messages[1].citations == ("[src:1] عنوان",)


def test_conversation_window_drops_oldest_turns():
    conversation = Conversation(title="نافذة", max_turns=3)
    for index in range(5):
        conversation.add_user(f"سؤال {index}")

    assert len(conversation.messages) == 3
    assert conversation.messages[0].text == "سؤال 2"


def test_conversation_rejects_empty_message_and_bad_window():
    with pytest.raises(ValueError):
        Conversation(title="س", max_turns=0)
    conversation = Conversation(title="س")
    with pytest.raises(ValueError):
        conversation.add_user("   ")


def test_conversation_round_trips_through_primitives():
    conversation = Conversation(title="جلسة")
    conversation.add_user("ما المكمن؟")
    conversation.add_assistant("المكمن خزان.", ["[doc:1]"])

    restored = Conversation.from_primitive(conversation.to_primitive())
    assert restored.id == conversation.id
    assert restored.render() == conversation.render()
    assert restored.messages[1].citations == ("[doc:1]",)


def test_conversation_memory_persists_to_a_store(tmp_path):
    store = JsonStore(tmp_path)
    memory = ConversationMemory(store=store)
    conversation = memory.start(title="محفوظة")
    conversation.add_user("مرحبا")
    memory.save(conversation)

    reopened = ConversationMemory(store=store)
    loaded = reopened.load(conversation.id)
    assert loaded is not None
    assert loaded.messages[0].text == "مرحبا"
    assert conversation.id in reopened.list_saved()


def test_message_from_primitive_keeps_role():
    message = Message.from_primitive(
        {
            "id": "MSG-1",
            "role": "assistant",
            "text": "نص",
            "citations": ["[a]"],
            "created_at": "2026-01-01T00:00:00+00:00",
        }
    )
    assert message.role is Role.ASSISTANT


# --- documents ------------------------------------------------------------


def test_document_validation_and_round_trip():
    with pytest.raises(ValueError):
        Document("", "t", "body")
    with pytest.raises(ValueError):
        Document("d", "", "body")
    with pytest.raises(ValueError):
        Document("d", "t", "  ")

    document = Document("d1", "عنوان", "نص", "manual", {"k": "v"})
    restored = Document.from_primitive(document.to_primitive())
    assert restored == document


def test_document_store_adds_and_rejects_duplicates(tmp_path):
    store = DocumentStore(store=JsonStore(tmp_path))
    store.add_text("d1", "عنوان", "نص")
    assert len(store) == 1
    with pytest.raises(ValueError):
        store.add_text("d1", "عنوان", "نص")

    store.save(store.get("d1"))
    reopened = DocumentStore(store=JsonStore(tmp_path))
    assert reopened.load_all() == 1
    assert reopened.get("d1").text == "نص"


def test_knowledge_documents_cover_every_discipline_and_course():
    documents = build_knowledge_documents()
    assert len(documents) == len(build_document_store().documents())
    titles = " ".join(document.title for document in documents)
    assert "تخصص:" in titles
    assert "مقرر:" in titles
    assert any(document.doc_id.startswith("discipline:") for document in documents)
    assert any(document.doc_id.startswith("course:") for document in documents)


# --- RAG ------------------------------------------------------------------


def _retriever() -> RagRetriever:
    ontology = Ontology()
    seed_ontology(ontology)
    seed_curriculum(ontology)
    return RagRetriever(build_document_store(), ontology=ontology)


def test_retrieval_ranks_the_relevant_discipline_first():
    retriever = _retriever()
    passages = retriever.retrieve("ما علاقة المسامية بالنفاذية؟", limit=3)
    assert passages
    titles = " ".join(passage.title for passage in passages)
    assert "البتروفيزياء" in titles or "تسجيلات_الآبار" in titles
    assert passages[0].score >= passages[-1].score


def test_retrieval_citations_identify_the_source():
    passage = _retriever().retrieve("النفاذية", limit=1)[0]
    assert passage.citation.startswith("[")
    assert passage.doc_id in passage.citation
    assert passage.signal in {"lexical", "semantic", "ontology"}


def test_retrieval_rejects_empty_query_and_bad_limit():
    retriever = _retriever()
    with pytest.raises(ValueError):
        retriever.retrieve("   ")
    with pytest.raises(ValueError):
        retriever.retrieve("النفاذية", limit=0)


def test_index_document_makes_new_text_retrievable():
    retriever = _retriever()
    retriever.index_document(
        Document("note:x", "ملاحظة خاصة", "الزيت الثقيل يحتاج تسخين", "notes")
    )
    passages = retriever.retrieve("الزيت الثقيل تسخين", limit=2)
    assert passages[0].doc_id == "note:x"


# --- assistant ------------------------------------------------------------


class _StubModel:
    """A model that returns a grounded answer, to test the generation path."""

    def generate(self, request):
        return ModelResponse(text="النفاذية خاصية للطبقة.", confidence=0.9, model_name="stub")


def test_assistant_answers_extractively_without_a_model():
    answer = GroundedAssistant(_retriever()).ask("ما علاقة المسامية بالنفاذية؟")
    assert answer.used_model is False
    assert answer.citations
    assert "المصادر" in answer.text or answer.text
    assert 0.0 <= answer.confidence <= 1.0


def test_assistant_uses_an_injected_model():
    answer = GroundedAssistant(_retriever(), model=_StubModel()).ask("النفاذية")
    assert answer.used_model is True
    assert answer.text == "النفاذية خاصية للطبقة."


def test_assistant_rejects_ungrounded_model_output():
    class _Ungrounded:
        def generate(self, request):
            return ModelResponse(text="كلام لا علاقة له بالمراجع.", confidence=0.9, model_name="bad")

    answer = GroundedAssistant(_retriever(), model=_Ungrounded()).ask("النفاذية")
    assert answer.used_model is False
    assert "النفاذية" in answer.text


def test_assistant_keeps_conversation_history():
    assistant = GroundedAssistant(_retriever())
    conversation = assistant.memory.start(title="جلسة")
    assistant.ask("ما المكمن؟", conversation=conversation)
    assistant.ask("وما النفاذية؟", conversation=conversation)
    assert len(conversation.messages) == 4
    assert conversation.last_user_message().text == "وما النفاذية؟"


def test_assistant_rejects_empty_question():
    with pytest.raises(ValueError):
        GroundedAssistant(_retriever()).ask("   ")


def test_assistant_falls_back_when_the_model_raises():
    class _Broken:
        def generate(self, request):
            raise ValueError("boom")

    answer = GroundedAssistant(_retriever(), model=_Broken()).ask("النفاذية")
    assert answer.used_model is False
    assert answer.text
