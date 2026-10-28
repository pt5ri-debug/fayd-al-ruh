"""Tests for Phase 50: the Arabic output path (prose + citable documents).

The project is Arabic-first, but before Phase 50 every generated text block was
100% English: the language model's training corpus and the citable documents the
grounded assistant retrieves from. These tests pin the fix -- the Arabic renderer,
its wiring into the corpus and the document store, and the composer bug it exposed
(an Arabic sentence was dropped as a fragment because it does not start with an
ASCII capital).
"""

from __future__ import annotations

import pytest
from core.arabic_glossary import has_arabic
from core.arabic_prose import (
    CORE_ARABIC_SENTENCES,
    DEFAULT_RELATION_ARABIC,
    RELATION_ARABIC,
    TOPIC_ARABIC,
    arabic_corpus_sentences,
    arabic_curriculum_sentences,
    arabic_document_lines,
    arabic_document_sections,
    arabic_label,
    arabic_procedural_sentences,
    arabic_relation_sentence,
    arabic_sentence_blocks,
    relation_arabic,
)


def _is_arabic(text: str) -> bool:
    return any(has_arabic(char) for char in text)


# --- the renderer ----------------------------------------------------------


def test_core_arabic_sentences_are_all_arabic():
    assert CORE_ARABIC_SENTENCES
    for sentence in CORE_ARABIC_SENTENCES:
        assert _is_arabic(sentence), sentence


def test_relation_sentence_uses_an_arabic_clause():
    sentence = arabic_relation_sentence("المسامية", "measures", "النفاذية")
    assert _is_arabic(sentence)
    assert sentence.endswith(".")
    # The relation's own clause is used, not the default.
    assert relation_arabic("measures") in sentence
    assert relation_arabic("measures") != DEFAULT_RELATION_ARABIC


def test_unknown_relation_falls_back_instead_of_dropping_the_edge():
    clause = relation_arabic("a_relation_the_map_never_named")
    assert clause == DEFAULT_RELATION_ARABIC
    assert _is_arabic(clause)


def test_every_relation_in_the_map_is_arabic():
    for relation, clause in RELATION_ARABIC.items():
        assert _is_arabic(clause), f"{relation} -> {clause}"


def test_topic_table_is_arabic():
    assert TOPIC_ARABIC
    for label, arabic in TOPIC_ARABIC.items():
        assert _is_arabic(arabic), f"{label} -> {arabic}"


def test_arabic_label_prefers_the_ontology_alias():
    from core.cognitive_controller import CognitiveController

    ontology = CognitiveController().ontology
    assert arabic_label("porosity", ontology) == "مسامية"
    assert arabic_label("permeability", ontology) == "نفاذية"


def test_arabic_label_translates_a_discipline_phrase():
    from core.cognitive_controller import CognitiveController

    ontology = CognitiveController().ontology
    assert arabic_label("Reservoir Engineering", ontology) == "هندسة المكامن"
    assert arabic_label("Drilling Engineering", ontology) == "هندسة الحفر"


def test_arabic_label_leaves_an_unknown_latin_identifier_alone():
    # A Latin identifier the domain keeps must not be guessed at.
    assert arabic_label("SPE", None) == "SPE"
    assert arabic_label("API", None) == "API"


def test_curriculum_sentences_are_arabic_and_cover_the_graph():
    sentences = arabic_curriculum_sentences()
    assert sentences
    for sentence in sentences:
        assert _is_arabic(sentence), sentence
    # A relation sentence for a known edge is present.
    assert any("نفاذية" in sentence for sentence in sentences)


def test_corpus_sentences_are_arabic():
    sentences = arabic_corpus_sentences()
    assert sentences
    for sentence in sentences:
        assert _is_arabic(sentence), sentence
    # The corpus's own glossary pairs reach the Arabic text.
    assert any("مسامية" in sentence for sentence in sentences)


def test_procedural_sentences_are_arabic():
    sentences = arabic_procedural_sentences()
    assert sentences
    for sentence in sentences:
        assert _is_arabic(sentence), sentence


def test_sentence_blocks_are_all_arabic():
    blocks = arabic_sentence_blocks()
    assert len(blocks) > 500
    for block in blocks:
        assert _is_arabic(block), block


def test_document_lines_are_arabic():
    lines = arabic_document_lines()
    assert lines
    for line in lines:
        assert _is_arabic(line), line


def test_document_sections_are_arabic_and_identifiable():
    sections = arabic_document_sections()
    assert sections
    for doc_id, title, lines in sections:
        assert doc_id.startswith("arabic:")
        assert _is_arabic(title), title
        assert lines
        assert all(_is_arabic(line) for line in lines)


# --- the composer bug the wiring exposed -----------------------------------


def test_an_arabic_sentence_is_not_dropped_as_a_fragment():
    from core.composition import complete_sentences

    text = "المسامية تقيس الفراغات في الصخر. النفاذية تقيس سهولة تدفق المائع."
    assert complete_sentences(text) == text


def test_an_arabic_sentence_splits_on_the_same_boundary():
    from core.composition import sentences

    text = "المسامية تقيس الفراغات. النفاذية تقيس التدفق."
    assert sentences(text) == ["المسامية تقيس الفراغات.", "النفاذية تقيس التدفق."]


def test_a_decimal_is_still_not_split():
    from core.composition import sentences

    # The Arabic-aware splitter must not break a decimal either.
    assert sentences("The value is 0.052 psi/ft.") == ["The value is 0.052 psi/ft."]


# --- the wiring into the corpus --------------------------------------------


def test_the_corpus_carries_arabic_text():
    from ml.llm.corpus import arabic_sentences, build_corpus

    assert arabic_sentences()
    corpus = build_corpus(repeats=1, structured_repeats=1)
    arabic = sum(1 for char in corpus if has_arabic(char))
    assert arabic > 0
    # Phase 50 moved this from 0.00% to a real share of the training text.
    assert arabic / len(corpus) > 0.05


def test_corpus_units_include_the_arabic_block():
    from ml.llm.corpus import arabic_sentences, corpus_units

    units = set(corpus_units())
    assert set(arabic_sentences()) <= units


# --- the wiring into the citable documents ---------------------------------


def test_the_document_store_carries_arabic_documents():
    from core.documents import build_knowledge_documents

    documents = build_knowledge_documents()
    arabic_documents = [doc for doc in documents if doc.source == "arabic_prose"]
    assert arabic_documents
    for document in arabic_documents:
        assert _is_arabic(document.text)
        assert document.metadata.get("language") == "ar"


def test_an_arabic_question_is_answered_with_arabic_body():
    from core.assistant import build_default_assistant

    assistant = build_default_assistant()
    result = assistant.ask("ما هي المسامية؟")
    text = result.text
    arabic = sum(1 for char in text if has_arabic(char))
    latin = sum(1 for char in text if char.isascii() and char.isalpha())
    # Phase 50 moved an Arabic answer's body from ~4% Arabic to mostly Arabic.
    assert arabic > latin


def test_an_english_question_is_still_answered_in_english():
    from core.assistant import build_default_assistant

    assistant = build_default_assistant()
    result = assistant.ask("what is porosity?")
    text = result.text
    arabic = sum(1 for char in text if has_arabic(char))
    latin = sum(1 for char in text if char.isascii() and char.isalpha())
    # The English path must not regress: English stays English.
    assert latin > arabic * 5


@pytest.mark.parametrize(
    "question",
    [
        "ما هي المسامية؟",
        "كيف أقيس النفاذية؟",
        "ما هي خطوات اختبار الضغط؟",
    ],
)
def test_arabic_answers_stay_grounded(question):
    from core.assistant import build_default_assistant

    assistant = build_default_assistant()
    result = assistant.ask(question)
    assert result.citations
    assert result.text.strip()
