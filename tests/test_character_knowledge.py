"""Tests for the per-character knowledge corpus loader and its mouth wiring.

Everything here is offline: the corpus is either a fake retriever (the test seam)
or a tiny real store built with the offline fake fetcher, so no network is used.
The isolation rule -- a character can only reach its own corpus -- is checked
directly.
"""

from __future__ import annotations

import pytest
from core import character_config as cc
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
    # The header plus a bounded passage, never the whole 2500-char text. The bound
    # is the reader's own resolved passage limit, so raising the limit is not a
    # test failure -- only failing to elide is.
    assert len(block) < len(KNOWLEDGE_HEADER) + knowledge.passage_chars + 100
    assert len(block) < len("word " * 500)


def test_retrieve_respects_the_limit(tmp_path):
    passages = [_Passage(str(i), f"T{i}", "text") for i in range(10)]
    knowledge = CharacterKnowledge("natasha", root=tmp_path, retriever=_FakeRetriever(passages))
    assert len(knowledge.retrieve("x", limit=2)) == 2
    assert len(knowledge.retrieve("x")) == KNOWLEDGE_LIMIT


# ----------------------------------------------------- injection dials


def test_the_shipped_limits_are_raised_and_bounded():
    from core.character_knowledge import (
        KNOWLEDGE_MAX_CHARS,
        KNOWLEDGE_PASSAGE_CHARS,
        LIMIT_RANGE,
        MAX_CHARS_RANGE,
        PASSAGE_CHARS_RANGE,
    )

    # The task asked for more knowledge in the prompt, so the shipped defaults are
    # above the old 4 passages / 4000 chars, and still inside their bounds.
    assert KNOWLEDGE_LIMIT >= 6
    assert KNOWLEDGE_MAX_CHARS >= 8000
    assert LIMIT_RANGE[0] <= KNOWLEDGE_LIMIT <= LIMIT_RANGE[1]
    assert MAX_CHARS_RANGE[0] <= KNOWLEDGE_MAX_CHARS <= MAX_CHARS_RANGE[1]
    assert PASSAGE_CHARS_RANGE[0] <= KNOWLEDGE_PASSAGE_CHARS <= PASSAGE_CHARS_RANGE[1]


def test_resolve_limits_prefers_the_explicit_argument(tmp_path):
    from core.character_knowledge import resolve_knowledge_limits

    assert resolve_knowledge_limits(limit=9, max_chars=1234, passage_chars=321) == (9, 1234, 321)


def test_resolve_limits_reads_the_env_then_the_default(monkeypatch):
    from core.character_knowledge import (
        ENV_KNOWLEDGE_LIMIT,
        ENV_KNOWLEDGE_MAX_CHARS,
        ENV_KNOWLEDGE_PASSAGE_CHARS,
        KNOWLEDGE_LIMIT,
        resolve_knowledge_limits,
    )

    monkeypatch.setenv(ENV_KNOWLEDGE_LIMIT, "12")
    monkeypatch.setenv(ENV_KNOWLEDGE_MAX_CHARS, "20000")
    monkeypatch.setenv(ENV_KNOWLEDGE_PASSAGE_CHARS, "500")
    limit, max_chars, passage = resolve_knowledge_limits()
    assert (limit, max_chars, passage) == (12, 20000, 500)
    # A bad value is ignored rather than breaking retrieval.
    monkeypatch.setenv(ENV_KNOWLEDGE_LIMIT, "not-a-number")
    assert resolve_knowledge_limits()[0] == KNOWLEDGE_LIMIT


def test_resolve_limits_clamps_an_out_of_range_value():
    from core.character_knowledge import LIMIT_RANGE, MAX_CHARS_RANGE, resolve_knowledge_limits

    limit, max_chars, _passage = resolve_knowledge_limits(limit=10_000, max_chars=1)
    assert limit == LIMIT_RANGE[1]
    assert max_chars == MAX_CHARS_RANGE[0]


def test_a_reader_honours_its_own_limits(tmp_path):
    """A CharacterKnowledge built with limits uses them, not the module defaults."""
    passages = [_Passage(str(i), f"T{i}", "text") for i in range(10)]
    knowledge = CharacterKnowledge(
        "natasha", root=tmp_path, retriever=_FakeRetriever(passages), limit=3
    )
    assert len(knowledge.retrieve("x")) == 3
    assert len(knowledge.retrieve("x", limit=7)) == 7


def test_the_block_respects_the_configured_max_chars(tmp_path):
    retriever = _FakeRetriever([_Passage("1", "A", "x" * 5000)])
    knowledge = CharacterKnowledge(
        "natasha", root=tmp_path, retriever=retriever, max_chars=2000, passage_chars=2000
    )
    block = knowledge.reference_block("anything")
    assert block
    assert len(block) <= 2000


def test_the_block_elides_at_the_configured_passage_size(tmp_path):
    retriever = _FakeRetriever([_Passage("1", "A", "word " * 2000)])
    knowledge = CharacterKnowledge(
        "natasha", root=tmp_path, retriever=retriever, max_chars=8000, passage_chars=300
    )
    block = knowledge.reference_block("anything")
    assert "..." in block
    # The single passage is cut at ~300 chars, not carried whole.
    assert len(block) < len(KNOWLEDGE_HEADER) + 400


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


def test_mouth_builds_its_reader_from_the_characters_knowledge_section(tmp_path, monkeypatch):
    """The reader's limits come from the character's own configuration."""
    from ai.characters import CharacterMouth
    from core import character_registry as registry_mod

    monkeypatch.setenv("FAYD_DATA_DIR", str(tmp_path / "data"))
    registry = registry_mod.CharacterRegistry()
    config = registry.load("natasha")
    config = registry.save(cc.replace_knowledge(config, limit=11, max_chars=9000, passage_chars=700))
    assistant = _FakeAssistant()
    mouth = CharacterMouth("natasha", assistant, root=tmp_path, registry=registry)
    assert mouth.knowledge is not None
    assert mouth.knowledge.limit == 11
    assert mouth.knowledge.max_chars == 9000
    assert mouth.knowledge.passage_chars == 700


def test_reload_config_rebuilds_the_reader_with_the_new_limits(tmp_path, monkeypatch):
    """An APPLY (a saved Knowledge section) reaches the next answer."""
    from ai.characters import CharacterMouth
    from core import character_registry as registry_mod

    monkeypatch.setenv("FAYD_DATA_DIR", str(tmp_path / "data"))
    registry = registry_mod.CharacterRegistry()
    assistant = _FakeAssistant()
    mouth = CharacterMouth("natasha", assistant, root=tmp_path, registry=registry)
    assert mouth.knowledge.limit == cc.default_config_for("natasha").knowledge.limit
    registry.save(
        cc.replace_knowledge(registry.load("natasha"), limit=9, max_chars=7000, passage_chars=600)
    )
    mouth.reload_config()
    assert mouth.knowledge.limit == 9
    assert mouth.knowledge.max_chars == 7000
    assert mouth.knowledge.passage_chars == 600


def test_a_private_characters_config_cannot_enable_corpus_training():
    from core.character_config import default_config_for, replace_knowledge

    natasha = replace_knowledge(default_config_for("natasha"), train_on_corpus=True)
    assert natasha.effective_training is False
    jarvis = replace_knowledge(default_config_for("jarvis"), train_on_corpus=True)
    assert jarvis.effective_training is True


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
