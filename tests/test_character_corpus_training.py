"""Tests for the optional per-character knowledge-corpus training path.

A character answers from its own corpus by retrieval; this is the *opt-in* path
that also folds an opted-in character's corpus into the project model's training
text. Everything here is offline: the corpus is a tiny store built with the
offline fake fetcher, so no network is used. The isolation rule -- a private
character's corpus never enters the training text -- is checked directly.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from core import character_config as cc
from core.knowledge_base import build_knowledge_base
from core.learning_isolation import CHARACTER_OWNERS
from ml.llm import corpus as corpus_mod


class _FakeFetcher:
    """Offline fetcher: one category, one article."""

    source = "wikipedia"
    license = "CC BY-SA 4.0 (Wikipedia)"
    lang = "en"

    def subcategories(self, category, limit=500):
        return []

    def category_titles(self, category):
        return ["Love"]

    def extracts(self, titles):
        return [(t, ("Love is a strong affection and a positive emotional state. " * 5)) for t in titles]


def _build(owner: str, root: Path) -> None:
    store_dir = root / "characters" / owner / "knowledge_base"
    build_knowledge_base(
        store_dir,
        target_mb=1,
        categories=["Love"],
        fetcher=_FakeFetcher(),
        batch=5,
        sleep=0,
    )


@pytest.fixture
def data_root(tmp_path, monkeypatch):
    root = tmp_path / "data"
    root.mkdir()
    monkeypatch.setenv("FAYD_DATA_DIR", str(root))
    return root


# ------------------------------------------------------------------ opt-in


def test_no_character_opts_in_by_default():
    assert corpus_mod.training_corpus_keys() == ()
    assert corpus_mod.character_knowledge_sentences() == []
    assert corpus_mod.character_knowledge_units() == []


def test_a_character_that_opts_in_contributes_its_corpus(data_root: Path, monkeypatch):
    from core.character_registry import CharacterRegistry

    registry = CharacterRegistry()
    registry.save(cc.replace_knowledge(registry.load("jarvis"), train_on_corpus=True))
    _build("jarvis", data_root)

    assert corpus_mod.training_corpus_keys() == ("jarvis",)
    sentences = corpus_mod.character_knowledge_sentences()
    assert sentences
    assert any("affection" in s.lower() for s in sentences)
    units = corpus_mod.character_knowledge_units()
    assert len(units) == 1 and "affection" in units[0].lower()


def test_an_opted_in_character_with_no_corpus_contributes_nothing(data_root: Path):
    from core.character_registry import CharacterRegistry

    registry = CharacterRegistry()
    registry.save(cc.replace_knowledge(registry.load("ultron"), train_on_corpus=True))
    assert corpus_mod.training_corpus_keys() == ("ultron",)
    assert corpus_mod.character_knowledge_sentences() == []


def test_the_chunk_bound_is_honoured(data_root: Path):
    from core.character_registry import CharacterRegistry

    registry = CharacterRegistry()
    registry.save(cc.replace_knowledge(registry.load("vision"), train_on_corpus=True))
    _build("vision", data_root)
    many = corpus_mod.character_knowledge_sentences(max_chunks=1000)
    assert many
    # A bound of 1 keeps at most the first chunk's worth.
    assert len(corpus_mod.character_knowledge_sentences(max_chunks=1)) <= len(many)


# ------------------------------------------------------------- isolation


def test_natasha_can_never_opt_in(data_root: Path):
    """A hand-edited file that turns training on for the private character is refused."""
    from core.character_registry import CharacterRegistry

    registry = CharacterRegistry()
    registry.save(cc.replace_knowledge(registry.load("natasha"), train_on_corpus=True))
    _build("natasha", data_root)
    assert "natasha" not in corpus_mod.training_corpus_keys()
    assert corpus_mod.character_knowledge_sentences() == []


def test_an_explicit_private_key_is_still_refused(data_root: Path):
    """Naming the private key explicitly does not read her corpus either."""
    _build("natasha", data_root)
    # The aggregate never reaches her.
    assert corpus_mod.character_knowledge_sentences() == []
    # Nor does an explicit key: the guard refuses her before any path is read.
    assert corpus_mod.character_knowledge_sentences(["natasha"]) == []
    assert corpus_mod.character_knowledge_units(["natasha"]) == []


def test_the_default_keys_are_all_shareable():
    from core.learning_isolation import PRIVATE_OWNERS

    assert not (set(corpus_mod.training_corpus_keys()) & set(PRIVATE_OWNERS))
    assert set(corpus_mod.training_corpus_keys()) <= set(CHARACTER_OWNERS)


# --------------------------------------------------------------- wiring


def test_build_corpus_excludes_the_corpora_by_default(data_root: Path):
    from core.character_registry import CharacterRegistry

    registry = CharacterRegistry()
    registry.save(cc.replace_knowledge(registry.load("banner"), train_on_corpus=True))
    _build("banner", data_root)
    without = corpus_mod.build_corpus(repeats=1, structured_repeats=1, kb_max_chunks=0)
    with_it = corpus_mod.build_corpus(
        repeats=1, structured_repeats=1, kb_max_chunks=0, include_character_knowledge=True
    )
    assert len(with_it) > len(without)
    assert "strong affection" in with_it


def test_corpus_units_include_the_corpora_only_when_asked(data_root: Path):
    from core.character_registry import CharacterRegistry

    registry = CharacterRegistry()
    registry.save(cc.replace_knowledge(registry.load("cloud"), train_on_corpus=True))
    _build("cloud", data_root)
    base = corpus_mod.corpus_units()
    extended = corpus_mod.corpus_units(include_character_knowledge=True)
    assert len(extended) == len(base) + 1


def test_the_training_switch_reaches_the_registry(data_root: Path):
    """An APPLY (a saved Knowledge section) is what the corpus path reads."""
    from core.character_registry import CharacterRegistry

    registry = CharacterRegistry()
    assert corpus_mod.training_corpus_keys() == ()
    registry.save(cc.replace_knowledge(registry.load("hakeem"), train_on_corpus=True))
    assert corpus_mod.training_corpus_keys() == ("hakeem",)
    registry.save(cc.replace_knowledge(registry.load("hakeem"), train_on_corpus=False))
    assert corpus_mod.training_corpus_keys() == ()


# ------------------------------------------------------------------- gate


def test_the_isolation_gate_holds_with_the_training_control():
    from core.character_isolation_gate import character_isolation_accuracy

    assert character_isolation_accuracy() == 1.0
