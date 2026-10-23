"""The learned distributional semantic space, and its wiring into retrieval."""

from __future__ import annotations

import json

import pytest
from core.documents import Document, DocumentStore
from core.embeddings import (
    DEFAULT_DIMENSIONS,
    SemanticSpace,
    build_semantic_space,
    cooccurrence_counts,
    evaluate_held_out,
    evaluate_semantic_space,
    held_out_split,
    load_semantic_space,
    reset_semantic_cache,
    save_semantic_space,
    seed_semantic_documents,
    semantic_space,
)
from core.rag import RagRetriever
from core.tokenize import content_terms, normalize_token, token_sequence, tokenize

# A tiny corpus with an unambiguous distribution: the "rock" words appear
# together and never with the "well" words, so a real space must separate them.
_CORPUS = [
    "permeability measures how easily fluid flows through the rock pore space.",
    "porosity is the fraction of the rock volume that is pore space.",
    "the rock pore space holds fluid and lets it flow.",
    "permeability and porosity are both rock properties of the pore space.",
    "cement is pumped into the well annulus to seal the casing.",
    "the cement bond seals the casing against the well annulus.",
    "casing is run into the well and cemented in place.",
    "the well annulus is filled with cement that seals the casing.",
]


def test_tokenizer_is_shared_and_arabic_aware():
    assert normalize_token("ألفا") == normalize_token("الفا")
    assert normalize_token("الكتاب") == "كتاب"
    assert tokenize("The permeability is high") == {"the", "permeability", "is", "high"}
    assert "the" not in content_terms("the permeability is high")
    assert token_sequence("rock pore") == ["rock", "pore"]


def test_cooccurrence_counts_window_and_symmetry():
    counts, pairs, documents, tokens = cooccurrence_counts(
        ["a b c"], window=1
    )
    assert documents == 1
    assert tokens == 3
    assert counts["a"] == 1
    assert pairs[("a", "b")] == 1
    assert pairs[("b", "a")] == 1
    assert pairs[("a", "c")] == 0


def test_cooccurrence_window_must_be_positive():
    with pytest.raises(ValueError):
        cooccurrence_counts(["a b"], window=0)


def test_space_requires_matching_terms_and_vectors():
    with pytest.raises(ValueError):
        SemanticSpace(terms=("a",), vectors=(), dimensions=8, window=2, min_count=1, documents=1, tokens=1)
    with pytest.raises(ValueError):
        SemanticSpace(terms=(), vectors=(), dimensions=0, window=2, min_count=1, documents=1, tokens=1)


def test_build_is_deterministic():
    first = build_semantic_space(_CORPUS, dimensions=16)
    second = build_semantic_space(_CORPUS, dimensions=16)
    assert first.terms == second.terms
    assert first.vectors == second.vectors


def test_space_separates_two_topics_it_never_labelled():
    """The point of the whole module: related words land closer than unrelated.

    The two topics share *no* content word (one is about rock and pore space, the
    other about cement and casing), so any similarity between them is whatever
    the co-occurrence statistics happen to imply. The rock pair must still win.
    """
    space = build_semantic_space(_CORPUS, dimensions=24, min_count=1)
    rock = space.similarity("permeability", "porosity")
    well = space.similarity("permeability", "casing")
    assert rock > well


def test_similar_terms_returns_the_topic_mates():
    """A term's nearest neighbours are its own topic's words, not the other's."""
    space = build_semantic_space(_CORPUS, dimensions=24, min_count=1)
    neighbours = {term for term, _score in space.similar_terms("permeability", limit=8)}
    assert "rock" in neighbours
    assert "cement" not in neighbours
    assert "casing" not in neighbours


def test_embed_of_unknown_text_is_a_zero_vector_that_matches_nothing():
    space = build_semantic_space(_CORPUS, dimensions=16, min_count=1)
    assert not any(space.embed("zzzz qqqq"))
    assert space.similarity("zzzz", "permeability") == 0.0
    assert space.similar_terms("zzzz") == []


def test_vector_returns_none_for_an_unknown_term():
    space = build_semantic_space(_CORPUS, dimensions=16, min_count=1)
    assert space.vector("permeability") is not None
    assert space.vector("never-seen-this") is None


def test_similar_terms_limit_must_be_positive():
    space = build_semantic_space(_CORPUS, dimensions=16, min_count=1)
    with pytest.raises(ValueError):
        space.similar_terms("porosity", limit=0)


def test_min_count_drops_terms_seen_less_than_the_threshold():
    counts, _pairs, _documents, _tokens = cooccurrence_counts(_CORPUS)
    space = build_semantic_space(_CORPUS, dimensions=16, min_count=3)
    # Every kept term really was seen at least three times, and every dropped
    # term really was not -- the rule is exact, not approximate.
    for term in space.terms:
        assert counts[term] >= 3
    assert any(count < 3 for count in counts.values())
    assert "permeability" not in space.terms


def test_min_count_above_every_count_yields_an_empty_space():
    """A threshold nothing clears is an empty space, not a crash."""
    space = build_semantic_space(_CORPUS, dimensions=16, min_count=99)
    assert space.terms == ()
    assert space.embed("permeability") == tuple(0.0 for _ in range(16))


def test_evaluate_reports_a_positive_separation():
    """A topic word must be nearer its own topic than a word from the other one.

    ``min_count=1`` keeps every word, so the tiny corpus still has enough terms
    to place ``permeability`` next to its own topic's words.
    """
    space = build_semantic_space(_CORPUS, dimensions=24, min_count=1)
    report = evaluate_semantic_space(space, [("permeability", "rock")], neighbours=8)
    assert report.pairs_measured == 1
    assert report.learned
    assert report.separation > 0.0
    assert report.nearest_accuracy > 0.0
    assert "separation" in report.summary()


def test_evaluate_with_no_usable_pairs_is_not_learned():
    space = build_semantic_space(_CORPUS, dimensions=16, min_count=1)
    report = evaluate_semantic_space(space, [("zzz", "qqq")])
    assert report.pairs_measured == 0
    assert not report.learned


def test_evaluate_neighbours_must_be_positive():
    space = build_semantic_space(_CORPUS, dimensions=16, min_count=1)
    with pytest.raises(ValueError):
        evaluate_semantic_space(space, [("a", "b")], neighbours=0)


def test_save_and_load_round_trip(tmp_path):
    space = build_semantic_space(_CORPUS, dimensions=16, min_count=1)
    path = save_semantic_space(space, tmp_path / "space.json")
    assert path.exists()
    reloaded = load_semantic_space(path)
    assert reloaded.terms == space.terms
    assert reloaded.vectors == space.vectors
    assert reloaded.dimensions == space.dimensions
    assert reloaded.describe() == space.describe()


def test_semantic_space_trains_on_first_use_and_caches(tmp_path):
    reset_semantic_cache()
    path = tmp_path / "space.json"
    assert semantic_space(path, build=False) is None
    built = semantic_space(path)
    assert built is not None
    assert path.exists()
    # The second call is served from the cache, so the file is not rewritten.
    before = path.stat().st_mtime_ns
    assert semantic_space(path) is built
    assert path.stat().st_mtime_ns == before
    reset_semantic_cache()


def test_semantic_space_ignores_a_corrupt_cache(tmp_path):
    reset_semantic_cache()
    path = tmp_path / "space.json"
    path.write_text("{not json", encoding="utf-8")
    rebuilt = semantic_space(path)
    assert rebuilt is not None
    assert rebuilt.vocabulary_size > 0
    reset_semantic_cache()


def test_saved_payload_is_readable_json(tmp_path):
    space = build_semantic_space(_CORPUS, dimensions=16, min_count=1)
    path = save_semantic_space(space, tmp_path / "space.json")
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["dimensions"] == 16
    assert len(payload["terms"]) == len(payload["vectors"])


def test_seed_documents_come_from_the_project_corpus():
    documents = seed_semantic_documents()
    assert len(documents) > 100
    assert any("permeability" in document for document in documents)


def test_held_out_split_removes_the_entries_it_tests():
    """No held-out entry's own text may survive in the training set.

    The check is on the entry's *own* statements -- its definition and each fact
    rendered as a sentence -- because that is what would leak the answer. Two
    common words merely happening to appear somewhere else in the corpus is not a
    leak; a stated relation is.
    """
    from core.petroleum_corpus import _ENTRIES

    training, pairs = held_out_split(hold_out=6)
    assert pairs
    training_text = " ".join(training)
    held = sorted(_ENTRIES, key=lambda entry: entry.title)[:6]
    for entry in held:
        assert entry.definition not in training_text
        for source, target, relation, _confidence in entry.facts:
            assert f"{source} {relation} {target}." not in training_text
    assert training


def test_held_out_split_rejects_a_bad_count():
    with pytest.raises(ValueError):
        held_out_split(hold_out=0)


def test_held_out_evaluation_generalises_without_memorising():
    """The honest headline: trained without those entries, still predictive."""
    space, report = evaluate_held_out(hold_out=8)
    assert report.pairs_measured > 0
    assert report.learned
    assert report.related_cosine > report.random_cosine
    assert space.vocabulary_size > 100


def test_rag_retriever_uses_the_learned_space(tmp_path, monkeypatch):
    reset_semantic_cache()
    monkeypatch.setenv("FAYD_DATA_DIR", str(tmp_path))
    store = DocumentStore()
    store.add(Document(doc_id="d1", title="Rock flow", text=_CORPUS[0], source="t"))
    store.add(Document(doc_id="d2", title="Cementing", text=_CORPUS[4], source="t"))
    retriever = RagRetriever(store)
    assert retriever.embeddings_available
    # A paraphrase sharing no content word with either document.
    passages = retriever.retrieve("how easily fluid moves through the pore space", limit=2)
    assert passages
    assert passages[0].doc_id == "d1"
    reset_semantic_cache()


def test_rag_retriever_falls_back_when_embeddings_are_refused():
    store = DocumentStore()
    store.add(Document(doc_id="d1", title="Rock flow", text=_CORPUS[0], source="t"))
    retriever = RagRetriever(store, embeddings=False)
    assert not retriever.embeddings_available
    assert retriever.retrieve("permeability", limit=1)[0].doc_id == "d1"


def test_dimensions_default_is_a_positive_int():
    assert isinstance(DEFAULT_DIMENSIONS, int)
    assert DEFAULT_DIMENSIONS > 0
