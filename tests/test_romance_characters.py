"""Tests for the Character Studio: the personality engine and the manager."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from core.romance import (
    CharacterDraft,
    CharacterError,
    CharacterManager,
    CharacterPersonalityEngine,
    CompanionStorageService,
    PersonalityError,
    human_passing_identity,
)
from core.romance.personality import SAFETY_FLOOR, clamp_dial, dial_line
from core.romance.seed import EXAMPLES, SEED_MARKER, seed_examples


@pytest.fixture()
def manager(tmp_path: Path) -> CharacterManager:
    return CharacterManager(CompanionStorageService(root=tmp_path))


def _draft(**overrides) -> CharacterDraft:
    base: dict = {"owner": "natasha", "name": "Amina", "fictional_age": 30}
    base.update(overrides)
    return CharacterDraft(**base)


# ------------------------------------------------------------ personality


def test_dial_line_reads_the_three_poles():
    low = dial_line("warmth", 0.1)
    mid = dial_line("warmth", 0.5)
    high = dial_line("warmth", 0.95)
    assert low != mid != high
    assert "0.9" not in high  # never the bare number


def test_clamp_refuses_a_non_number():
    assert clamp_dial(2) == 1.0
    assert clamp_dial(-1) == 0.0
    with pytest.raises(PersonalityError):
        clamp_dial("warm")


def test_render_says_it_shapes_the_voice_not_the_claims():
    from core.romance.model import CharacterPersonality

    engine = CharacterPersonalityEngine()
    personality = CharacterPersonality(record_id="PER-1", character_id="CHR-1", warmth=0.9, response_length="brief")
    block = engine.render(personality)
    assert "tone guidance" in block
    assert "never what you" in block


def test_validate_refuses_unknown_field_and_length():
    engine = CharacterPersonalityEngine()
    with pytest.raises(PersonalityError):
        engine.validate({"nonsense": 1})
    with pytest.raises(PersonalityError):
        engine.validate({"response_length": "epic"})
    clean = engine.validate({"warmth": 5, "response_length": "brief"})
    assert clean["warmth"] == 1.0 and clean["response_length"] == "brief"


def test_human_passing_identity_never_says_ai():
    text = human_passing_identity("Layla", biography="a designer")
    assert "person" in text
    assert "never describe yourself as a program" in text
    assert "a designer" in text


def test_safety_floor_names_adults_and_consent():
    assert "18 or older" in SAFETY_FLOOR
    assert "consent" in SAFETY_FLOOR


# --------------------------------------------------------------- manager


def test_create_reads_back_the_bundle(manager: CharacterManager):
    bundle = manager.create(
        _draft(
            personality={"warmth": 0.9, "response_length": "brief"},
            profile_items=({"kind": "interest", "label": "poetry"},),
            boundaries=({"kind": "topic", "value": "no politics"},),
        )
    )
    assert bundle.name == "Amina"
    assert bundle.personality is not None and bundle.personality.warmth == 0.9
    assert bundle.interests() == ("poetry",)
    assert len(bundle.boundaries) == 1


def test_a_character_needs_a_name(manager: CharacterManager):
    with pytest.raises(CharacterError):
        manager.create(_draft(name="   "))


def test_adult_eligible_requires_a_known_adult_age(manager: CharacterManager):
    with pytest.raises(CharacterError):
        manager.create(_draft(fictional_age=None, adult_eligible=True))
    with pytest.raises(CharacterError):
        manager.create(_draft(fictional_age=17, adult_eligible=True))
    bundle = manager.create(_draft(fictional_age=18, adult_eligible=True))
    assert bundle.character.adult_eligible is True


def test_update_is_validated_and_recorded(manager: CharacterManager):
    bundle = manager.create(_draft())
    updated = manager.update(bundle.id, name="Amina B.", biography="a poet")
    assert updated.character.name == "Amina B."
    assert bundle.character.name == "Amina"  # the original is untouched
    with pytest.raises(CharacterError):
        manager.update(bundle.id, name="")
    with pytest.raises(CharacterError):
        manager.update(bundle.id, nonsense=1)


def test_set_personality_creates_then_updates(manager: CharacterManager):
    bundle = manager.create(_draft())
    manager.set_personality(bundle.id, {"playfulness": 0.9})
    again = manager.set_personality(bundle.id, {"playfulness": 0.2})
    assert again.personality is not None and again.personality.playfulness == 0.2


def test_profile_items_and_boundaries_round_trip(manager: CharacterManager):
    bundle = manager.create(_draft())
    bundle = manager.add_profile_item(bundle.id, kind="hobby", label="painting")
    item_id = bundle.profile_items[0].record_id
    with pytest.raises(CharacterError):
        manager.add_profile_item(bundle.id, kind="nonsense", label="x")
    with pytest.raises(CharacterError):
        manager.add_profile_item(bundle.id, kind="hobby", label="  ")
    bundle = manager.add_boundary(bundle.id, value="go slowly")
    assert bundle.interests() == ("painting",)
    bundle = manager.remove_profile_item(bundle.id, item_id)
    assert bundle.interests() == ()


def test_set_model_records_on_the_character_and_the_profile(manager: CharacterManager):
    bundle = manager.create(_draft())
    bundle = manager.set_model(bundle.id, model_key="qwen3-4b", temperature=0.6)
    assert bundle.character.model_key == "qwen3-4b"
    assert bundle.model_profile is not None and bundle.model_profile.temperature == 0.6
    bundle = manager.set_model(bundle.id, model_key="qwen3-8b")
    assert bundle.model_profile is not None and bundle.model_profile.model_key == "qwen3-8b"


def test_set_voice_creates_a_profile(manager: CharacterManager):
    bundle = manager.create(_draft())
    bundle = manager.set_voice(bundle.id, engine="piper", voice_key="ar_JO-kareem", language="ar")
    assert bundle.voice is not None and bundle.voice.engine == "piper"


def test_duplicate_copies_the_children(manager: CharacterManager):
    bundle = manager.create(
        _draft(
            personality={"warmth": 0.8},
            profile_items=({"kind": "interest", "label": "poetry"},),
            boundaries=({"kind": "topic", "value": "no politics"},),
        )
    )
    manager.set_voice(bundle.id, engine="piper", voice_key="ar_JO-kareem", language="ar")
    copy = manager.duplicate(bundle.id)
    assert copy.id != bundle.id
    assert copy.name == "Amina (copy)"
    assert copy.interests() == ("poetry",)
    assert copy.personality is not None and copy.personality.warmth == 0.8
    assert copy.voice is not None and copy.voice.engine == "piper"


def test_archive_hides_and_restore_shows(manager: CharacterManager):
    bundle = manager.create(_draft())
    manager.archive(bundle.id)
    assert manager.list_characters() == []
    assert len(manager.list_characters(include_archived=True)) == 1
    manager.restore_from_archive(bundle.id)
    assert len(manager.list_characters()) == 1


def test_favorite_filter(manager: CharacterManager):
    bundle = manager.create(_draft())
    manager.set_favorite(bundle.id, True)
    assert len(manager.list_characters(favorite=True)) == 1
    assert manager.list_characters(favorite=False) == []


def test_delete_requires_a_retention_choice(manager: CharacterManager):
    bundle = manager.create(_draft())
    with pytest.raises(CharacterError):
        manager.delete(bundle.id, choice="whatever")
    report = manager.delete(bundle.id, choice="purge")
    assert report.deactivated is True


def test_counts_are_gathered_without_a_query_per_character(manager: CharacterManager):
    first = manager.create(_draft(name="One"))
    second = manager.create(_draft(name="Two"))
    storage = manager.storage
    storage.domain.create("companion_conversations", character_id=first.id, title="c1")
    storage.domain.create("companion_memories", character_id=first.id, content="m1")
    storage.domain.create("companion_memories", character_id=second.id, content="m2")
    rows = {s.name: s for s in manager.list_characters()}
    assert rows["One"].conversation_count == 1 and rows["One"].memory_count == 1
    assert rows["Two"].memory_count == 1 and rows["Two"].conversation_count == 0


def test_export_then_import_round_trips(manager: CharacterManager, tmp_path: Path):
    bundle = manager.create(
        _draft(
            personality={"playfulness": 0.7},
            profile_items=({"kind": "interest", "label": "poetry"},),
        )
    )
    target = manager.export_character(bundle.id, tmp_path / "amina.json")
    imported = manager.import_character(target, name="Amina Imported")
    assert imported.name == "Amina Imported"
    assert imported.interests() == ("poetry",)
    assert imported.personality is not None and imported.personality.playfulness == 0.7


def test_import_refuses_a_foreign_file(manager: CharacterManager, tmp_path: Path):
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps({"format": "something-else"}), encoding="utf-8")
    with pytest.raises(CharacterError):
        manager.import_character(bad)


def test_import_refuses_a_file_that_loses_its_age(manager: CharacterManager, tmp_path: Path):
    source = tmp_path / "adult.json"
    source.write_text(
        json.dumps(
            {
                "format": "fayd-romance-character",
                "version": 1,
                "character": {"name": "Adult", "fictional_age": None, "adult_eligible": True},
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(CharacterError):
        manager.import_character(source)


# ------------------------------------------------------------------ seed


def test_seed_examples_are_adults_and_labelled(manager: CharacterManager):
    created = seed_examples(manager)
    assert len(created) == len(EXAMPLES)
    for bundle in created:
        assert bundle.character.metadata.get("seed") == SEED_MARKER
        assert isinstance(bundle.character.fictional_age, int)
        assert bundle.character.fictional_age >= 18


def test_seed_examples_are_idempotent(manager: CharacterManager):
    seed_examples(manager)
    again = seed_examples(manager)
    assert again == []
    assert len(manager.list_characters()) == len(EXAMPLES)


def test_seed_examples_can_be_owned_by_a_key(manager: CharacterManager):
    created = seed_examples(manager, owner="jarvis")
    assert all(bundle.character.owner == "jarvis" for bundle in created)


# ------------------------------------------------------------------ gate


def test_the_studio_gate_covers_the_character_work_and_scores_full():
    from core.romance.gate import STUDIO_JUDGEMENTS, companion_studio_accuracy

    assert companion_studio_accuracy() == 1.0
    kinds = {judgement.kind for judgement in STUDIO_JUDGEMENTS}
    for required in (
        "character-round-trip",
        "character-duplicate",
        "character-export-import",
        "negative-import",
        "negative-adult-age",
        "human-passing",
        "safety-floor",
    ):
        assert required in kinds
