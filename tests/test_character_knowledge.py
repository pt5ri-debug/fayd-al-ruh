"""Tests for the per-character knowledge corpus loader and its mouth wiring.

Everything here is offline: the corpus is either a fake retriever (the test seam)
or a tiny real store built with the offline fake fetcher, so no network is used.
The isolation rule -- a character can only reach its own corpus -- is checked
directly.
"""

from __future__ import annotations

import pytest
from core.character_knowledge import (
    KNOWLEDGE_HEADER,
    KNOWLEDGE_LIMIT,
    CharacterKnowledge,
    character_knowledge_dir,
)
from core.knowledge_base import build_knowledge_base
from core.learning_isolation import LearningAccessError


class _Passage:
    def __init__(self, doc_id: str, title: str, text: str, source: str = "wikipedia", score: float = 0.9):
        self.doc_id = doc_id
        self.title = title
        self.text = text
        self.source = source
        self.score = score


class _FakeRetriever:
    def __init__(self, passages):
        self.passages = passages
        self.queries: list[str] = []

    def retrieve(self, query: str, *, limit: int = 4):
        self.queries.append(query)
        return self.passages[:limit]


# ------------------------------------------------------------------- directory


def test_knowledge_dir_is_owner_scoped_and_under_characters(tmp_path):
    d = character_knowledge_dir("natasha", root=tmp_path)
    assert d == tmp_path / "characters" / "natasha" / "knowledge_base"
    # Either vocabulary resolves to the same folder.
    assert character_knowledge_dir("jarvis", root=tmp_path) == character_knowledge_dir(
        "sources", root=tmp_path
    )


def test_knowledge_dir_rejects_an_unknown_character(tmp_path):
    with pytest.raises(LearningAccessError):
        character_knowledge_dir("nobody", root=tmp_path)


def test_a_character_only_ever_reaches_its_own_folder(tmp_path):
    natasha = CharacterKnowledge("natasha", root=tmp_path)
    hakeem = CharacterKnowledge("hakeem", root=tmp_path)
    assert natasha.dir != hakeem.dir
    assert natasha.dir == tmp_path / "characters" / "natasha" / "knowledge_base"
    assert hakeem.dir == tmp_path / "characters" / "hakeem" / "knowledge_base"


def test_knowledge_root_matches_the_learning_root_convention(tmp_path):
    """The corpus lives beside the learning record, under one shared data root.

    A mouth is built with a single ``root`` (the GUI's ``data_root``); the
    learning store reads ``<root>/characters/learning`` and the knowledge loader
    must read ``<root>/characters/<owner>/knowledge_base`` -- the same
    ``characters`` folder. If the two conventions drift, a built corpus is looked
    for in the wrong place and silently never reaches the model (the bug this
    pins).
    """
    from core.character_learning import CharacterLearning

    learning = CharacterLearning("natasha", root=tmp_path, in_memory=False)
    assert learning.root == tmp_path / "characters" / "learning"
    corpus = character_knowledge_dir("natasha", root=tmp_path)
    assert corpus == tmp_path / "characters" / "natasha" / "knowledge_base"
    assert corpus.parent.parent == learning.root.parent == tmp_path / "characters"


# --------------------------------------------------------------------- reading


def test_absent_corpus_is_honest(tmp_path):
    knowledge = CharacterKnowledge("natasha", root=tmp_path)
    assert knowledge.available is False
    assert knowledge.retrieve("love") == ()
    assert knowledge.reference_block("love") == ""
    assert "no knowledge corpus yet" in knowledge.describe()
    assert knowledge.stats()["present"] is False


def test_blank_question_retrieves_nothing(tmp_path):
    knowledge = CharacterKnowledge(
        "natasha", root=tmp_path, retriever=_FakeRetriever([_Passage("1", "Love", "text")])
    )
    assert knowledge.retrieve("   ") == ()
    assert knowledge.reference_block("") == ""


def test_reference_block_renders_a_bounded_labelled_block(tmp_path):
    retriever = _FakeRetriever(
        [
            _Passage("1", "Romantic love", "A strong affection and desire."),
            _Passage("2", "Eros", "Passionate love in Greek thought."),
        ]
    )
    knowledge = CharacterKnowledge("natasha", root=tmp_path, retriever=retriever)
    block = knowledge.reference_block("what is romantic love?")
    assert block.startswith(KNOWLEDGE_HEADER)
    assert "- Romantic love: A strong affection and desire." in block
    assert "- Eros: Passionate love in Greek thought." in block
    assert retriever.queries == ["what is romantic love?"]


def test_reference_block_elides_a_long_passage(tmp_path):
    retriever = _FakeRetriever([_Passage("1", "Long", "word " * 500)])
    knowledge = CharacterKnowledge("natasha", root=tmp_path, retriever=retriever)
    block = knowledge.reference_block("anything")
    assert "..." in block
    # The header plus a bounded passage, never the whole 2500-char text.
    assert len(block) < 1200


def test_retrieve_respects_the_limit(tmp_path):
    passages = [_Passage(str(i), f"T{i}", "text") for i in range(10)]
    knowledge = CharacterKnowledge("natasha", root=tmp_path, retriever=_FakeRetriever(passages))
    assert len(knowledge.retrieve("x", limit=2)) == 2
    assert len(knowledge.retrieve("x")) == KNOWLEDGE_LIMIT


def test_available_is_true_with_an_injected_retriever(tmp_path):
    knowledge = CharacterKnowledge("natasha", root=tmp_path, retriever=_FakeRetriever([]))
    assert knowledge.available is True


# ---------------------------------------------------------------- real store


class _FakeFetcher:
    """Offline fetcher: one category, one long article."""

    source = "wikipedia"
    license = "CC BY-SA 4.0 (Wikipedia)"
    lang = "en"

    def subcategories(self, category, limit=500):
        return []

    def category_titles(self, category):
        return ["Love"]

    def extracts(self, titles):
        return [(t, ("Love is a strong affection and a positive emotional state. " * 20)) for t in titles]


def test_reads_a_real_built_store(tmp_path):
    store_dir = tmp_path / "characters" / "natasha" / "knowledge_base"
    build_knowledge_base(
        store_dir,
        target_mb=1,
        categories=["Love"],
        fetcher=_FakeFetcher(),
        batch=5,
        sleep=0,
    )
    knowledge = CharacterKnowledge("natasha", root=tmp_path)
    assert knowledge.available is True
    hits = knowledge.retrieve("affection", limit=2)
    assert hits and "affection" in hits[0].text.lower()
    stats = knowledge.stats()
    assert stats["present"] is True and stats["chunks"] >= 1
    assert "own corpus" in knowledge.describe()


def test_reading_a_store_touches_no_network(tmp_path, monkeypatch):
    store_dir = tmp_path / "characters" / "natasha" / "knowledge_base"
    build_knowledge_base(
        store_dir, target_mb=1, categories=["Love"], fetcher=_FakeFetcher(), batch=5, sleep=0
    )

    import socket

    def _no_network(*args, **kwargs):
        raise AssertionError("the retrieval path must not open a socket")

    monkeypatch.setattr(socket.socket, "connect", _no_network)
    knowledge = CharacterKnowledge("natasha", root=tmp_path)
    assert knowledge.retrieve("love")  # served from the local FTS index / shards


# ------------------------------------------------------------------- mouth wiring


class _FakeEngine:
    system_prompt = "engine base prompt"


class _FakeAssistant:
    def __init__(self):
        self.engine = _FakeEngine()
        self.asked: list[str] = []

    def ask(self, question, **kwargs):
        class _A:
            text = "the model's answer"

        self.asked.append(question)
        return _A()


def test_mouth_injects_its_own_corpus_into_the_prompt(tmp_path):
    from ai.characters import CharacterMouth

    store_dir = tmp_path / "characters" / "natasha" / "knowledge_base"
    build_knowledge_base(
        store_dir, target_mb=1, categories=["Love"], fetcher=_FakeFetcher(), batch=5, sleep=0
    )
    assistant = _FakeAssistant()
    mouth = CharacterMouth("natasha", assistant, root=tmp_path)
    assert mouth.knowledge is not None
    mouth.ask("what is love?")
    prompt = assistant.engine.system_prompt
    assert KNOWLEDGE_HEADER in prompt
    # The character's own standing instruction is still in front of it.
    assert "Natasha" in prompt or "natasha" in prompt.lower()


def test_mouth_without_a_corpus_is_unchanged(tmp_path):
    from ai.characters import CharacterMouth

    assistant = _FakeAssistant()
    mouth = CharacterMouth("natasha", assistant, root=tmp_path)
    mouth.ask("what is love?")
    prompt = assistant.engine.system_prompt
    assert KNOWLEDGE_HEADER not in prompt


def test_mouth_refreshes_knowledge_for_each_question(tmp_path):
    from ai.characters import CharacterMouth

    assistant = _FakeAssistant()
    mouth = CharacterMouth("natasha", assistant, root=tmp_path)
    # A fake reader lets us assert the block is question-specific.
    mouth._knowledge = CharacterKnowledge(
        "natasha",
        root=tmp_path,
        retriever=_FakeRetriever([_Passage("1", "Eros", "passionate love")]),
    )
    mouth._refresh_knowledge("first question")
    assert "Eros" in assistant.engine.system_prompt
    mouth._knowledge = CharacterKnowledge(
        "natasha",
        root=tmp_path,
        retriever=_FakeRetriever([_Passage("2", "Agape", "selfless love")]),
    )
    mouth._refresh_knowledge("second question")
    assert "Agape" in assistant.engine.system_prompt
    assert "Eros" not in assistant.engine.system_prompt


# ------------------------------------------------------------------------ CLI


def test_cli_reports_an_absent_corpus(tmp_path, capsys, monkeypatch):
    from cli import characters as cli_characters

    monkeypatch.setenv("FAYD_DATA_DIR", str(tmp_path))
    assert cli_characters.main(["knowledge", "natasha"]) == 0
    out = capsys.readouterr().out
    assert "no knowledge corpus yet" in out


def test_cli_rejects_an_unknown_character(capsys):
    from cli import characters as cli_characters

    assert cli_characters.main(["knowledge", "nobody"]) == 3
    assert "not a known character" in capsys.readouterr().out


def test_cli_queries_a_built_corpus(tmp_path, capsys, monkeypatch):
    from cli import characters as cli_characters

    store_dir = tmp_path / "characters" / "natasha" / "knowledge_base"
    build_knowledge_base(
        store_dir, target_mb=1, categories=["Love"], fetcher=_FakeFetcher(), batch=5, sleep=0
    )
    monkeypatch.setenv("FAYD_DATA_DIR", str(tmp_path))
    assert cli_characters.main(["knowledge", "natasha", "affection"]) == 0
    out = capsys.readouterr().out
    assert "affection" in out.lower()


# ------------------------------------------------------------------------ gate


def test_gate_is_perfect_on_the_shipped_wiring():
    from core.character_knowledge_gate import character_knowledge_accuracy

    assert character_knowledge_accuracy() == 1.0


def test_gate_is_zero_for_an_empty_catalogue():
    from core.character_knowledge_gate import character_knowledge_accuracy

    assert character_knowledge_accuracy(()) == 0.0


def test_gate_discriminates_a_leaked_corpus(tmp_path, monkeypatch):
    # A reader that returned another character's passages would fail the gate's
    # cross-character negative control; simulate it and prove the metric drops.
    import core.character_knowledge_gate as gate

    real = gate.CharacterKnowledge

    class _Leak:
        def __init__(self, *args, **kwargs):
            self._real = real(*args, **kwargs)

        def __getattr__(self, name):
            return getattr(self._real, name)

        @property
        def available(self):
            return True

        def retrieve(self, question, *, limit=4):
            passage = type("P", (), {"text": "leaked from another character", "score": 0.9})()
            return (passage,)

    monkeypatch.setattr(gate, "CharacterKnowledge", _Leak)
    assert gate.character_knowledge_accuracy() < 1.0


def test_gate_discriminates_a_dropped_label(tmp_path, monkeypatch):
    import core.character_knowledge_gate as gate

    monkeypatch.setattr(gate, "KNOWLEDGE_HEADER", "")
    assert gate.character_knowledge_accuracy() < 1.0
