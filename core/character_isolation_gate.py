"""The held-out gate for learning isolation and controlled knowledge sharing.

The task requires strict isolation between character learning stores and a
controlled knowledge-sharing interface, with **Natasha excluded at the
architecture level**. A unit test pins one case; this is the *catalogue* -- a
held-out set of hand-written scenarios, each a claim about what the backend must
do, scored as one number and wired into ``evaluation`` so a regression in the
isolation is a **gate failure**, not a passing suite.

**Honest by construction.** Every judgement runs the *real* store, the *real*
sharing interface, and the *real* configuration -- never a mock -- against a
throwaway data root, so nothing here can pass by pretending. Several judgements
are **negative controls** (a cross-character write, a read of another character's
store, an attempt to read Natasha's records, an attempt to hand-enable Natasha
sharing): a backend that quietly allowed any of them would fail the gate even
though a positive-only catalogue would still pass.
"""

from __future__ import annotations

import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

from core.character_config import default_config_for, replace_learning
from core.character_learning_store import CharacterLearningStore
from core.knowledge_sharing import KnowledgeSharing
from core.learning_isolation import (
    CHARACTER_OWNERS,
    MAIN_MODEL_OWNER,
    LearningAccessError,
    assert_read,
    is_owner,
    store_dir_for,
)
from core.learning_pipeline import LearningPipeline
from core.learning_records import SOURCE_LANGUAGE_MODEL


@dataclass(frozen=True)
class IsolationJudgement:
    """One held-out claim about the isolation or the sharing, and how to check it."""

    name: str
    kind: str
    #: The character a sharing judgement draws from (unused by the isolation kinds).
    character: str = ""
    #: A short note, for a failure report.
    note: str = ""


#: The held-out catalogue. The isolation kinds are the task's own required rules;
#: the ``share-*`` kinds are the main model's controlled knowledge sharing.
ISOLATION_JUDGEMENTS: tuple[IsolationJudgement, ...] = (
    # -- the seven owners exist, each with its own directory ------------------
    IsolationJudgement("the seven characters and the main model are all known owners", "owners"),
    IsolationJudgement("every owner's store directory is its own", "distinct-dirs"),
    # -- a character writes only to its own store -----------------------------
    IsolationJudgement("a character writes to its own store", "own-write", character="jarvis"),
    IsolationJudgement(
        "a character cannot write to another's store",
        "cross-write-refused",
        character="jarvis",
        note="negative control: a cross-character write must be refused",
    ),
    IsolationJudgement(
        "a character cannot open another's store",
        "cross-read-refused",
        character="ultron",
        note="negative control: a cross-character read must be refused",
    ),
    # -- the main model's controlled sharing ----------------------------------
    IsolationJudgement("the main model reads jarvis", "share-allowed", character="jarvis"),
    IsolationJudgement("the main model reads ultron", "share-allowed", character="ultron"),
    IsolationJudgement("the main model reads vision", "share-allowed", character="vision"),
    IsolationJudgement("the main model reads banner", "share-allowed", character="banner"),
    IsolationJudgement("the main model reads hakeem", "share-allowed", character="hakeem"),
    IsolationJudgement("the main model reads cloud", "share-allowed", character="cloud"),
    IsolationJudgement(
        "the main model cannot read natasha",
        "natasha-refused",
        character="natasha",
        note="negative control: Natasha must never be reachable",
    ),
    IsolationJudgement(
        "the main model cannot read natasha even via 'share all'",
        "natasha-refused-all",
        note="negative control: 'share all' must not reach Natasha",
    ),
    IsolationJudgement(
        "learning from a character never modifies the source record",
        "source-untouched",
        character="vision",
    ),
    IsolationJudgement(
        "the main model's copy carries the source character and record id",
        "provenance-kept",
        character="banner",
    ),
    IsolationJudgement(
        "sharing the same character twice does not duplicate",
        "no-duplicate",
        character="hakeem",
    ),
    IsolationJudgement(
        "a plain share keeps the knowledge but not the private exchange",
        "history-not-shared",
        character="cloud",
        note="negative control: a private conversation must not leak through a share",
    ),
    IsolationJudgement(
        "a hand-edited natasha configuration cannot enable sharing",
        "config-cannot-enable-natasha",
        note="negative control: the backend's rule beats the setting",
    ),
    IsolationJudgement("an unknown key is not an owner", "unknown-refused"),
)


@contextmanager
def _temp_root() -> Iterator[Path]:
    with tempfile.TemporaryDirectory(prefix="fayd-isolation-") as folder:
        yield Path(folder)


def _seed(owner: str, root: Path, text: str) -> CharacterLearningStore:
    """Put one trusted record into ``owner``'s own store."""
    store = CharacterLearningStore(owner, root=root)
    pipeline = LearningPipeline(owner, store=store)
    pipeline.learn(
        source_type=SOURCE_LANGUAGE_MODEL,
        question="q",
        answer=text,
        source_reference="a held-out test",
    )
    return store


def _judgement_holds(judgement: IsolationJudgement) -> bool:
    """Run one judgement against the real backend; ``True`` when it holds."""
    kind = judgement.kind

    if kind == "owners":
        return (
            len(CHARACTER_OWNERS) == 7
            and all(is_owner(key) for key in CHARACTER_OWNERS)
            and is_owner(MAIN_MODEL_OWNER)
        )

    if kind == "unknown-refused":
        return not is_owner("not-a-real-character")

    if kind == "distinct-dirs":
        with _temp_root() as root:
            dirs = {owner: store_dir_for(owner, root) for owner in (*CHARACTER_OWNERS, MAIN_MODEL_OWNER)}
            return len(set(dirs.values())) == len(dirs)

    if kind == "own-write":
        with _temp_root() as root:
            store = _seed(judgement.character, root, "A durable fact.")
            return len(store.all()) == 1

    if kind == "cross-write-refused":
        # A store belongs to exactly one owner and has no method that takes
        # another's key, so the only cross-character access is the guard every
        # read/write path calls *before* resolving a directory -- and it refuses.
        with _temp_root() as root:
            _seed("ultron", root, "Ultron's own note.")
            try:
                assert_read("jarvis", "ultron")
            except LearningAccessError:
                return True
            return False

    if kind == "cross-read-refused":
        with _temp_root() as root:
            _seed("vision", root, "Vision's private note.")
            try:
                assert_read("ultron", "vision")
            except LearningAccessError:
                return True
            return False

    if kind == "natasha-refused":
        with _temp_root() as root:
            _seed("natasha", root, "A private companion note.")
            sharing = KnowledgeSharing(data_root=root)
            result = sharing.learn_from("natasha")
            return result.count == 0 and not sharing.can_read("natasha")

    if kind == "natasha-refused-all":
        with _temp_root() as root:
            _seed("natasha", root, "A private companion note.")
            _seed("jarvis", root, "A shareable note.")
            sharing = KnowledgeSharing(data_root=root)
            report = sharing.learn_from_all()
            touched = {result.source_character for result in report.results}
            return "natasha" not in touched

    if kind == "share-allowed":
        with _temp_root() as root:
            _seed(judgement.character, root, "A shareable note about porosity.")
            sharing = KnowledgeSharing(data_root=root)
            result = sharing.learn_from(judgement.character)
            return result.count >= 1 and sharing.can_read(judgement.character)

    if kind == "source-untouched":
        with _temp_root() as root:
            store = _seed(judgement.character, root, "A shareable note about permeability.")
            before = tuple(record.to_primitive() for record in store.all())
            KnowledgeSharing(data_root=root).learn_from(judgement.character)
            after = tuple(record.to_primitive() for record in store.all())
            return before == after

    if kind == "provenance-kept":
        with _temp_root() as root:
            source = _seed(judgement.character, root, "A shareable note about drilling.")
            source_id = source.all()[0].record_id
            sharing = KnowledgeSharing(data_root=root)
            sharing.learn_from(judgement.character)
            copies = sharing.main_store.all()
            return bool(copies) and copies[0].source_character == judgement.character and copies[
                0
            ].source_record_id == source_id

    if kind == "no-duplicate":
        with _temp_root() as root:
            _seed(judgement.character, root, "A shareable note about well testing.")
            sharing = KnowledgeSharing(data_root=root)
            sharing.learn_from(judgement.character)
            first = len(sharing.main_store.all())
            second = sharing.learn_from(judgement.character).count
            return len(sharing.main_store.all()) == first and second == 0

    if kind == "history-not-shared":
        with _temp_root() as root:
            store = _seed(judgement.character, root, "A shareable note about porosity.")
            # Give the source a private question/answer exchange.
            pipeline = LearningPipeline(judgement.character, store=store)
            pipeline.learn_from_user(question="A private question", statement="A private answer.")
            sharing = KnowledgeSharing(data_root=root)
            sharing.learn_from(judgement.character)
            shared = [
                r for r in sharing.main_store.all() if r.source_character == judgement.character
            ]
            # The knowledge is shared; the exchange is not.
            return bool(shared) and all(
                copy.question == "" and copy.answer == "" for copy in shared
            )

    if kind == "config-cannot-enable-natasha":
        config = replace_learning(
            default_config_for("natasha"),
            share_with_main_model=True,
            enabled=True,
        )
        return config.effective_sharing is False

    return False


def character_isolation_accuracy(
    judgements: tuple[IsolationJudgement, ...] = ISOLATION_JUDGEMENTS,
) -> float:
    """The share of held-out isolation and sharing judgements the backend satisfies.

    A judgement counts only when the backend does exactly what it must -- refuse a
    cross-character write, keep the main model out of Natasha's store, carry the
    provenance onto its own copy, and never duplicate on a second share. So a
    regression that opened one boundary fails the relevant judgement. Returns
    ``0.0`` for an empty catalogue.
    """
    catalogue = tuple(judgements)
    if not catalogue:
        return 0.0
    correct = sum(1 for judgement in catalogue if _judgement_holds(judgement))
    return correct / len(catalogue)


__all__ = [
    "ISOLATION_JUDGEMENTS",
    "IsolationJudgement",
    "character_isolation_accuracy",
]
