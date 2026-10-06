"""Tests for the seeded petroleum-engineering corpus and its integration.

The corpus is the project's knowledge content: definitions, relations, formulas,
procedures, standards, and glossary terms across every discipline. These tests
check that it is well-formed, covers the whole field, is idempotent when seeded,
and is actually wired into the ontology, the citable documents, and the
language-model training text.
"""

from __future__ import annotations

from core.documents import build_document_store, build_knowledge_documents
from core.domain_knowledge import discipline_keys
from core.ontology import Ontology
from core.petroleum_corpus import (
    CorpusEntry,
    corpus_coverage,
    corpus_coverage_score,
    corpus_disciplines,
    corpus_documents,
    corpus_sentences,
    entries,
    entries_for,
    entry_keys,
    seed_petroleum_corpus,
)


def test_corpus_keys_are_unique_and_non_empty():
    keys = entry_keys()
    assert len(keys) == len(set(keys))
    assert all(key.strip() for key in keys)


def test_corpus_covers_every_petroleum_discipline():
    assert set(discipline_keys()).issubset(set(corpus_disciplines()))


def test_every_entry_is_well_formed():
    for entry in entries():
        assert isinstance(entry, CorpusEntry)
        assert entry.definition.strip()
        assert entry.concepts or entry.facts, entry.key
        for label, _aliases in entry.concepts:
            assert label.strip()
        for source, target, relation, confidence in entry.facts:
            assert source.strip() and target.strip() and relation.strip()
            assert 0.0 < confidence <= 1.0


def test_corpus_has_formulas_procedures_and_standards():
    assert any(entry.formulas for entry in entries())
    assert any(entry.procedures for entry in entries())
    assert any(entry.standards for entry in entries())


def test_entries_for_filters_by_discipline():
    drilling = entries_for("drilling")
    assert drilling
    assert all(entry.discipline == "drilling" for entry in drilling)


def test_seed_petroleum_corpus_plants_concepts_and_relations():
    ontology = Ontology()
    seed_petroleum_corpus(ontology)

    assert ontology.find_concepts("proppant")
    assert ontology.find_concepts("net_present_value")
    assert ontology.find_concepts("skin_factor")
    assert ontology.relations
    assert corpus_coverage_score(ontology) == 1.0


def test_seed_petroleum_corpus_is_idempotent():
    ontology = Ontology()
    first = seed_petroleum_corpus(ontology)
    second = seed_petroleum_corpus(ontology)

    assert len(second) == len(first)
    assert len(ontology.concepts) == len(first)
    assert corpus_coverage_score(ontology) == 1.0


def test_corpus_shares_nodes_with_the_domain_frame():
    ontology = Ontology()
    from core.domain_knowledge import seed_ontology

    seed_ontology(ontology)
    before = len(ontology.concepts)
    seed_petroleum_corpus(ontology)
    # "reservoir" and "permeability" already exist, so they are reused, not duplicated.
    assert len(ontology.find_concepts("reservoir")) == 1
    assert len(ontology.concepts) > before


def test_corpus_coverage_reports_missing_knowledge_honestly():
    ontology = Ontology()
    coverage = corpus_coverage(ontology)

    assert set(coverage) == set(corpus_disciplines())
    assert all(counts["concepts"] == 0 for counts in coverage.values())
    assert all(counts["relations"] == 0 for counts in coverage.values())
    assert corpus_coverage_score(ontology) == 0.0


def test_corpus_documents_are_citable():
    documents = corpus_documents()
    assert len(documents) == len(entries())
    for document in documents:
        assert document.doc_id.startswith("corpus:")
        assert document.source == "petroleum_corpus"
        assert document.text.strip()


def test_build_knowledge_documents_includes_the_corpus():
    doc_ids = {document.doc_id for document in build_knowledge_documents()}
    assert any(doc_id.startswith("corpus:") for doc_id in doc_ids)
    store = build_document_store()
    assert store.get("corpus:hydraulic_fracturing") is not None


def test_corpus_sentences_include_definitions_and_formulas():
    text = " ".join(corpus_sentences())
    assert "hydraulic fracturing" in text.lower()
    assert "NPV = sum" in text
    assert "is a concept in" in text
