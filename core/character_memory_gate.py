"""The held-out gate for memory revival: seeding a character from real knowledge.

The task this measures: a fresh character's memory is *real but empty*, so it
reads as "ready but unused". The fix (:mod:`core.character_seeding`) teaches each
character the slice of the project's **own authored** knowledge that fits it.
This module is the held-out catalogue of hand-written claims about that seeding,
scored as one number and wired into ``evaluation``, so a regression that seeds an
invented fact, overfills a store, breaks idempotence, or writes into another
character's memory is a **gate failure**.

Every judgement runs the **real** seeding code against a real, isolated data root,
so it observes what the seeding actually did rather than asserting a table of
praise. Several judgements are **negative controls**: an unknown character must
resolve rather than raise, Hakeem must never be seeded with an invented religious
text, a seed must never exceed the store's ceiling, and a second seed must add
nothing.

The module is dependency-free, Tk-free, and loads no model, so it runs on a fresh
clone with no display and no weights. Conversation seeding uses the project's own
grounded assistant (its extractive fallback), which needs no model.
"""

from __future__ import annotations

import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

from core.character_learning import (
    CHARACTER_KEYS,
    MAX_FACTS,
    PROJECT_MODEL_KEY,
    CharacterLearning,
    persona_for,
)
from core.character_seeding import (
    CHARACTER_SEED_DOMAIN,
    DOMAIN_GENERAL,
    DOMAIN_PETROLEUM,
    seed_all,
    seed_character,
    seed_conversations,
)

#: The owner keys, in the project's order, that a whole-seed judgement walks.
_ALL_OWNERS: tuple[str, ...] = (*CHARACTER_KEYS, PROJECT_MODEL_KEY)


@dataclass(frozen=True)
class SeedJudgement:
    """One held-out claim about memory seeding, and how to check it."""

    name: str
    kind: str
    character: str = ""
    note: str = ""


#: The held-out catalogue. The positive kinds are the task's own claim -- a
#: character's memory really fills from the project's authored knowledge; the
#: rest are the negative controls.
SEED_JUDGEMENTS: tuple[SeedJudgement, ...] = (
    # -- every character is seeded, from the project's own knowledge ----------
    SeedJudgement("every character has a seed domain", "domain-map"),
    SeedJudgement("the grounded mouth draws on the petroleum corpus", "domain-petroleum", character="jarvis"),
    SeedJudgement("the petroleum mouth draws on the petroleum corpus", "domain-petroleum", character="vision"),
    SeedJudgement("the project model draws on the petroleum corpus", "domain-petroleum", character="fayd_al_ruh"),
    SeedJudgement("a general mouth is oriented to the project", "domain-general", character="ultron"),
    SeedJudgement("a whole seed pass fills every character", "seed-all"),
    SeedJudgement("a petroleum fact is one the corpus states", "fact-from-corpus", character="vision"),
    SeedJudgement("a character's own persona is seeded first", "persona-first", character="vision"),
    # -- exchanges are the project's own cited answers ------------------------
    SeedJudgement("conversation seeding records cited exchanges", "conversations-cited", character="jarvis"),
    SeedJudgement("a general mouth gets fewer exchanges than a petroleum one", "conversations-scoped"),
    # -- negative controls ----------------------------------------------------
    SeedJudgement(
        "an unknown character resolves without raising",
        "unknown-resolves",
        note="negative control: a stale key is never a crash",
    ),
    SeedJudgement(
        "Hakeem is not seeded with an invented religious corpus",
        "no-invented-religion",
        character="hakeem",
        note="negative control: the project authors no religious text, and none is invented",
    ),
    SeedJudgement(
        "a seed never exceeds the store ceiling",
        "ceiling",
        character="ultron",
        note="negative control: seeding is bounded",
    ),
    SeedJudgement(
        "a second seed adds nothing",
        "idempotent",
        character="ultron",
        note="negative control: seeding is idempotent",
    ),
    SeedJudgement(
        "the project model's persona is not mistaken for Jarvis's",
        "model-persona",
        note="negative control: owner name fayd_al_ruh must resolve to the model",
    ),
)


@contextmanager
def _isolated() -> Iterator[Path]:
    """A fresh, isolated data root, so a judgement never touches real data."""
    with tempfile.TemporaryDirectory() as tmp:
        yield Path(tmp)


def _facts(character: str, root: Path) -> tuple[str, ...]:
    from core.learning_isolation import owner_for

    return CharacterLearning(owner_for(character), root=root).facts()


def _judgement_holds(judgement: SeedJudgement) -> bool:
    kind = judgement.kind
    character = judgement.character
    with _isolated() as root:
        if kind == "domain-map":
            return all(k in CHARACTER_SEED_DOMAIN for k in _ALL_OWNERS)
        if kind == "domain-petroleum":
            from core.learning_isolation import owner_for

            return CHARACTER_SEED_DOMAIN.get(owner_for(character)) == DOMAIN_PETROLEUM
        if kind == "domain-general":
            from core.learning_isolation import owner_for

            return CHARACTER_SEED_DOMAIN.get(owner_for(character)) == DOMAIN_GENERAL
        if kind == "seed-all":
            report = seed_all(limit=12, root=root)
            return all(o.added > 0 for o in report.outcomes)
        if kind == "fact-from-corpus":
            seed_character(character, limit=40, root=root)
            from core.petroleum_corpus import entries

            definitions = [entry.definition for entry in entries()]
            return any(
                any(d[:40] in fact for d in definitions) for fact in _facts(character, root)
            )
        if kind == "persona-first":
            seed_character(character, limit=20, root=root)
            facts = _facts(character, root)
            return bool(facts) and persona_for(character).name in facts[0]
        if kind == "conversations-cited":
            seed_conversations(character, root=root)
            exchanges = CharacterLearning(character, root=root).exchanges()
            return bool(exchanges) and any(e.citations for e in exchanges)
        if kind == "conversations-scoped":
            seed_conversations("vision", root=root)
            seed_conversations("ultron", root=root)
            petroleum = len(CharacterLearning("vision", root=root).exchanges())
            general = len(CharacterLearning("ultron", root=root).exchanges())
            return petroleum > general
        if kind == "unknown-resolves":
            outcome = seed_character("nobody-at-all", limit=5, root=root)
            return bool(outcome.key)
        if kind == "no-invented-religion":
            seed_character(character, limit=40, root=root)
            joined = " ".join(_facts(character, root)).lower()
            # No scripture, no invented verse: only the project's own orientation.
            return "quran" not in joined and "bible" not in joined and "hadith" not in joined
        if kind == "ceiling":
            seed_character(character, limit=10_000, root=root)
            return len(_facts(character, root)) <= MAX_FACTS
        if kind == "idempotent":
            seed_character(character, limit=20, root=root)
            second = seed_character(character, limit=20, root=root)
            return second.added == 0
        if kind == "model-persona":
            outcome = seed_character("fayd_al_ruh", limit=5, root=root)
            return outcome.name == persona_for("model").name
    return False


def memory_seeding_accuracy(
    judgements: tuple[SeedJudgement, ...] = SEED_JUDGEMENTS,
) -> float:
    """The share of held-out seeding judgements the code satisfies.

    A judgement counts only when the real seeding does exactly what it must --
    each character is seeded from the project's own authored knowledge, an
    exchange is a cited answer, the seed is bounded and idempotent, and no
    invented fact (a religious text, an unknown key's crash) is produced. So a
    regression in the seeding fails the relevant judgement. Returns ``0.0`` for an
    empty catalogue.
    """
    catalogue = tuple(judgements)
    if not catalogue:
        return 0.0
    correct = sum(1 for judgement in catalogue if _judgement_holds(judgement))
    return correct / len(catalogue)


__all__ = [
    "SEED_JUDGEMENTS",
    "SeedJudgement",
    "memory_seeding_accuracy",
]
