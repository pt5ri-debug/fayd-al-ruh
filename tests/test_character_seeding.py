"""Seeding a character's memory from the project's own authored knowledge.

The feature this pins: a fresh character's memory is real but *empty*, so every
character reads as "ready but unused". :mod:`core.character_seeding` teaches each
character the slice of the project's own knowledge that fits it -- the petroleum
corpus's own definitions and facts, the project's orientation, the disciplines and
courses, and the character's own persona -- and, optionally, the project's own
cited answers as exchanges.

Everything here is honest by construction and headless: no fact is invented, each
character is seeded in character, seeding respects the store's ceilings, and a
second seed adds nothing (idempotent). The project's own grounded assistant
answers with no model (its extractive fallback), so conversation seeding needs
nothing downloaded and reaches the network never.
"""

from __future__ import annotations

import pytest
from core import character_seeding as seeding
from core.character_learning import (
    CHARACTER_KEYS,
    MAX_FACTS,
    PROJECT_MODEL_KEY,
    CharacterLearning,
    persona_for,
)


@pytest.fixture
def data_root(tmp_path, monkeypatch):
    root = tmp_path / "data"
    root.mkdir()
    monkeypatch.setenv("FAYD_DATA_DIR", str(root))
    return root


# --- the domain map -------------------------------------------------------


def test_every_character_has_a_seed_domain() -> None:
    for key in (*CHARACTER_KEYS, PROJECT_MODEL_KEY):
        assert key in seeding.CHARACTER_SEED_DOMAIN


def test_the_grounded_and_petroleum_mouths_get_the_petroleum_corpus() -> None:
    assert seeding.CHARACTER_SEED_DOMAIN["jarvis"] == seeding.DOMAIN_PETROLEUM
    assert seeding.CHARACTER_SEED_DOMAIN["vision"] == seeding.DOMAIN_PETROLEUM
    assert seeding.CHARACTER_SEED_DOMAIN["fayd_al_ruh"] == seeding.DOMAIN_PETROLEUM


def test_hakeem_is_not_seeded_with_a_religious_corpus() -> None:
    # The project authors no religious text; Hakeem is seeded only from the
    # project's own orientation and his own persona, never an invented scripture.
    assert seeding.CHARACTER_SEED_DOMAIN["hakeem"] == seeding.DOMAIN_GENERAL


# --- fact seeding ---------------------------------------------------------


def test_seeding_a_character_fills_its_memory_from_the_project(data_root) -> None:
    outcome = seeding.seed_character("vision", limit=40)
    assert outcome.added > 0
    learning = CharacterLearning("vision", root=data_root)
    facts = learning.facts()
    assert len(facts) == outcome.added
    # The character's own persona is seeded first (it is true of the character).
    assert any("Vision" in fact for fact in facts)


def test_the_seeded_petroleum_facts_come_from_the_corpus(data_root) -> None:
    seeding.seed_character("vision", limit=40)
    facts = CharacterLearning("vision", root=data_root).facts()
    # At least one fact is a corpus definition the project itself authored.
    from core.petroleum_corpus import entries

    definitions = [entry.definition for entry in entries()]
    assert any(any(d[:40] in fact for d in definitions) for fact in facts)


def test_a_general_character_is_oriented_to_the_project(data_root) -> None:
    seeding.seed_character("ultron", limit=40)
    facts = CharacterLearning("ultron", root=data_root).facts()
    assert any("Fayd Al-Ruh" in fact for fact in facts)


def test_seeding_is_idempotent(data_root) -> None:
    first = seeding.seed_character("ultron", limit=20)
    assert first.added > 0
    second = seeding.seed_character("ultron", limit=20)
    assert second.added == 0
    assert second.already_known == first.added


def test_seeding_respects_the_store_ceiling(data_root) -> None:
    seeding.seed_character("ultron", limit=10_000)
    facts = CharacterLearning("ultron", root=data_root).facts()
    assert len(facts) <= MAX_FACTS


def test_an_unknown_character_seeds_without_raising(data_root) -> None:
    # An unknown key resolves through the isolation policy to a real owner rather
    # than raising, so a stale saved key never breaks a seed pass.
    outcome = seeding.seed_character("nobody", limit=10)
    assert outcome.key  # resolved to a real owner
    assert outcome.domain in {seeding.DOMAIN_GENERAL, seeding.DOMAIN_PETROLEUM}


# --- conversation seeding -------------------------------------------------


def test_seeding_conversations_records_cited_exchanges(data_root) -> None:
    outcome = seeding.seed_conversations("jarvis")
    assert outcome.added > 0
    exchanges = CharacterLearning("jarvis", root=data_root).exchanges()
    assert exchanges
    # A grounded exchange carries a citation, so it is honest about its source.
    assert any(exchange.citations for exchange in exchanges)


def test_the_petroleum_mouth_gets_more_exchanges_than_a_general_one(data_root) -> None:
    seeding.seed_conversations("vision")
    seeding.seed_conversations("ultron")
    petroleum = len(CharacterLearning("vision", root=data_root).exchanges())
    general = len(CharacterLearning("ultron", root=data_root).exchanges())
    assert petroleum > general


def test_seeding_conversations_is_idempotent(data_root) -> None:
    seeding.seed_conversations("jarvis")
    second = seeding.seed_conversations("jarvis")
    assert second.added == 0


def test_seed_character_with_conversations_adds_both(data_root) -> None:
    outcome = seeding.seed_character("vision", limit=20, conversations=True)
    learning = CharacterLearning("vision", root=data_root)
    assert learning.facts()
    assert learning.exchanges()
    assert outcome.added >= len(learning.exchanges())


# --- the report -----------------------------------------------------------


def test_seed_all_covers_every_character(data_root) -> None:
    report = seeding.seed_all(limit=12)
    assert len(report.outcomes) == len(CHARACTER_KEYS) + 1
    assert report.total_added > 0
    assert "Seeded" in report.describe()


def test_seed_all_never_seeds_natasha_into_an_unreachable_state(data_root) -> None:
    # Natasha's records are private; seeding writes only into her *own* store and
    # never touches the main model's.
    seeding.seed_all(limit=12)
    natasha = CharacterLearning("natasha", root=data_root)
    assert natasha.facts()


def test_the_persona_of_the_project_model_is_not_mistaken_for_a_character(data_root) -> None:
    # A regression guard: the owner name "fayd_al_ruh" must resolve to the
    # project model's persona, never to Jarvis's (which the owner name "jarvis"
    # would name).
    outcome = seeding.seed_character("fayd_al_ruh", limit=10)
    assert outcome.name == persona_for("model").name


# --- the held-out gate ----------------------------------------------------


def test_the_memory_seeding_gate_is_green(data_root) -> None:
    from core.character_memory_gate import memory_seeding_accuracy

    assert memory_seeding_accuracy() == 1.0


def test_the_gate_fails_an_empty_catalogue() -> None:
    from core.character_memory_gate import memory_seeding_accuracy

    assert memory_seeding_accuracy(()) == 0.0


def test_the_gate_reports_a_missing_domain() -> None:
    from core.character_memory_gate import SEED_JUDGEMENTS, SeedJudgement, memory_seeding_accuracy

    # A judgement about a character with no domain must fail, so the gate
    # discriminates rather than always returning 1.0.
    broken = (SeedJudgement("an unknown character has a domain", "domain-petroleum", character="nobody"),)
    assert len(SEED_JUDGEMENTS) > 1
    assert memory_seeding_accuracy(broken) == 0.0
