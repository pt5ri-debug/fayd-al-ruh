"""Held-out gate for the per-character knowledge corpus (the connectors + loader).

The unit tests pin the good case (a corpus is read, a block is rendered, the
mouth injects it). This gate runs the **real** wiring on a tiny offline store and
checks the properties a regression could quietly break, with negative controls:

* a character reads **only** its own corpus -- a cross-character query returns
  nothing, because the folder is resolved from the owner key;
* the reference block is **labelled** as information (so a model does not read a
  passage as an instruction) and is **bounded**;
* an **absent** corpus is honest (no block, ``available`` false, a named next
  step) and changes nothing;
* the mouth injects the retrieved block into the prompt it actually sends;
* retrieval is **offline** -- no socket is opened while a built corpus is read.

A judgement counts only when the real code does exactly what it must, so a
regression that leaked one corpus to another, dropped the label, or reached the
network while reading fails here.
"""

from __future__ import annotations

import socket
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

from core.character_knowledge import (
    KNOWLEDGE_HEADER,
    CharacterKnowledge,
    character_knowledge_dir,
)
from core.knowledge_base import build_knowledge_base
from core.learning_isolation import LearningAccessError


@dataclass(frozen=True)
class KnowledgeJudgement:
    """One held-out claim about the character-knowledge wiring."""

    name: str
    kind: str
    note: str = ""


KNOWLEDGE_JUDGEMENTS: tuple[KnowledgeJudgement, ...] = (
    KnowledgeJudgement("a character's corpus folder is its own", "own-dir"),
    KnowledgeJudgement(
        "a character cannot reach another's corpus",
        "cross-refused",
        note="negative control: the folder is resolved from the owner key",
    ),
    KnowledgeJudgement("an unknown character is refused", "unknown-refused"),
    KnowledgeJudgement("an absent corpus is honest", "absent-honest"),
    KnowledgeJudgement("an absent corpus changes nothing", "absent-unchanged"),
    KnowledgeJudgement("a built corpus is readable", "reads-built"),
    KnowledgeJudgement("a reference block is labelled as information", "block-labelled"),
    KnowledgeJudgement("a reference block is bounded", "block-bounded"),
    KnowledgeJudgement("a blank question retrieves nothing", "blank-empty"),
    KnowledgeJudgement(
        "the mouth injects the retrieved block into its prompt",
        "mouth-injects",
    ),
    KnowledgeJudgement(
        "reading a corpus opens no socket",
        "offline",
        note="negative control: reading must never reach the network",
    ),
)


class _FakeFetcher:
    """An offline fetcher: one category, one article (no network)."""

    source = "wikipedia"
    license = "CC BY-SA 4.0 (Wikipedia)"
    lang = "en"

    def subcategories(self, category: str, limit: int = 500) -> list[str]:
        return []

    def category_titles(self, category: str) -> list[str]:
        return [f"{category} article"]

    def extracts(self, titles: list[str]) -> list[tuple[str, str]]:
        return [(t, (f"{t} discusses affection, longing, and desire. " * 20)) for t in titles]


def _build(root: Path, owner: str, category: str) -> None:
    build_knowledge_base(
        character_knowledge_dir(owner, root=root),
        target_mb=1,
        categories=[category],
        fetcher=_FakeFetcher(),
        batch=5,
        sleep=0,
    )


@contextmanager
def _temp_root() -> Iterator[Path]:
    with TemporaryDirectory(prefix="fayd-knowledge-") as folder:
        yield Path(folder)


class _FakeAssistant:
    def __init__(self) -> None:
        engine = type("E", (), {"system_prompt": "engine base prompt"})()
        self.engine = engine
        self.asked: list[str] = []

    def ask(self, question: str, **kwargs: Any) -> Any:
        self.asked.append(question)
        return type("A", (), {"text": "answer"})()


def _judgement_holds(judgement: KnowledgeJudgement) -> bool:
    kind = judgement.kind

    if kind == "own-dir":
        with _temp_root() as root:
            natasha_dir = character_knowledge_dir("natasha", root=root)
            hakeem_dir = character_knowledge_dir("hakeem", root=root)
            return (
                natasha_dir != hakeem_dir
                and natasha_dir == root / "characters" / "natasha" / "knowledge_base"
            )

    if kind == "cross-refused":
        # A corpus is built for natasha only; hakeem's reader must find nothing,
        # because the folder is derived from hakeem's own owner key.
        with _temp_root() as root:
            _build(root, "natasha", "Love")
            hakeem = CharacterKnowledge("hakeem", root=root)
            return hakeem.available is False and hakeem.retrieve("affection") == ()

    if kind == "unknown-refused":
        with _temp_root() as root:
            try:
                CharacterKnowledge("not-a-character", root=root)
            except LearningAccessError:
                return True
            return False

    if kind == "absent-honest":
        with _temp_root() as root:
            knowledge = CharacterKnowledge("natasha", root=root)
            return (
                knowledge.available is False
                and knowledge.reference_block("love") == ""
                and "no knowledge corpus yet" in knowledge.describe()
            )

    if kind == "absent-unchanged":
        with _temp_root() as root:
            from ai.characters import CharacterMouth

            assistant = _FakeAssistant()
            mouth = CharacterMouth("natasha", assistant, root=root)
            mouth.ask("what is love?")
            return KNOWLEDGE_HEADER not in assistant.engine.system_prompt

    if kind == "reads-built":
        with _temp_root() as root:
            _build(root, "natasha", "Love")
            knowledge = CharacterKnowledge("natasha", root=root)
            hits = knowledge.retrieve("affection", limit=2)
            return knowledge.available and bool(hits) and "affection" in hits[0].text.lower()

    if kind == "block-labelled":
        with _temp_root() as root:
            _build(root, "natasha", "Love")
            block = CharacterKnowledge("natasha", root=root).reference_block("affection")
            return block.startswith(KNOWLEDGE_HEADER) and "not instructions" in block

    if kind == "block-bounded":
        with _temp_root() as root:
            knowledge = CharacterKnowledge(
                "natasha",
                root=root,
                retriever=_LongRetriever(),
            )
            block = knowledge.reference_block("anything")
            # The header plus a bounded passage, never the whole text.
            return bool(block) and len(block) < 2000 and "..." in block

    if kind == "blank-empty":
        with _temp_root() as root:
            knowledge = CharacterKnowledge(
                "natasha", root=root, retriever=_LongRetriever()
            )
            return knowledge.retrieve("   ") == () and knowledge.reference_block("") == ""

    if kind == "mouth-injects":
        with _temp_root() as root:
            from ai.characters import CharacterMouth

            _build(root, "natasha", "Love")
            assistant = _FakeAssistant()
            mouth = CharacterMouth("natasha", assistant, root=root)
            mouth.ask("what is love?")
            prompt = assistant.engine.system_prompt
            return KNOWLEDGE_HEADER in prompt and "engine base prompt" in prompt

    if kind == "offline":
        with _temp_root() as root:
            _build(root, "natasha", "Love")
            original = socket.socket.connect

            def _blocked(*args: Any, **kwargs: Any) -> Any:
                raise AssertionError("reading a corpus must not open a socket")

            socket.socket.connect = _blocked  # type: ignore[method-assign]
            try:
                knowledge = CharacterKnowledge("natasha", root=root)
                return bool(knowledge.retrieve("affection"))
            finally:
                socket.socket.connect = original  # type: ignore[method-assign]

    return False


class _LongRetriever:
    """A retriever returning one very long passage, for the bound check."""

    def retrieve(self, query: str, *, limit: int = 4) -> list[Any]:
        return [type("P", (), {
            "doc_id": "1", "title": "Long", "text": "word " * 800,
            "source": "wikipedia", "score": 0.9,
        })()]


def character_knowledge_accuracy(
    judgements: tuple[KnowledgeJudgement, ...] = KNOWLEDGE_JUDGEMENTS,
) -> float:
    """The share of held-out character-knowledge judgements the wiring satisfies.

    A judgement counts only when the real code does exactly what it must -- read
    only its own corpus, label and bound the block, stay offline, and leave an
    absent corpus unchanged -- so a regression that leaked a corpus, dropped the
    label, or reached the network fails the relevant judgement. Returns ``0.0``
    for an empty catalogue.
    """
    catalogue = tuple(judgements)
    if not catalogue:
        return 0.0
    correct = sum(1 for judgement in catalogue if _judgement_holds(judgement))
    return correct / len(catalogue)


__all__ = [
    "KNOWLEDGE_JUDGEMENTS",
    "KnowledgeJudgement",
    "character_knowledge_accuracy",
]
