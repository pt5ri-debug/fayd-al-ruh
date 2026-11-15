"""Tests for a companion's own files and its romantic reach.

Covers the two additions of this slice:

* :mod:`core.romance.character_files` -- each companion's **own** folder: its
  attributes (``profile.json``), its knowledge bank (a real knowledge-base store
  the companion can retrieve from), and its learning (facts and exchanges).
* the **reach dials** and **kinship** -- how a character leads a romantic scene,
  and the relationship degree it holds toward the user -- plus the library being
  adult-eligible at the source and the parents archetype.

Everything runs with no model and no display.
"""

from __future__ import annotations

import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import pytest
from core.romance.character_files import (
    CharacterFiles,
    CharacterFilesError,
    CharacterProfileDocument,
)
from core.romance.library import (
    CATEGORY_PARENTS,
    LIBRARY_CHARACTERS,
    library_by_category,
)
from core.romance.personality import REACH_DIALS, SAFETY_FLOOR
from gui.romance import RomanceStudioService


class _Seam:
    def __init__(self, *replies: str) -> None:
        self.replies = list(replies) or ["Hello."]
        self.calls: list[str] = []

    def __call__(self, prompt: str, key, handle) -> tuple[str, bool, str]:
        self.calls.append(prompt)
        return self.replies[min(len(self.calls) - 1, len(self.replies) - 1)], True, ""


@contextmanager
def _service(seam=None) -> Iterator[RomanceStudioService]:
    with tempfile.TemporaryDirectory() as folder:
        yield RomanceStudioService(root=Path(folder), generator=seam or _Seam())


# -- the companion's own files --------------------------------------------


def test_a_companion_profile_round_trips(tmp_path: Path):
    files = CharacterFiles("CON-abc", root=tmp_path)
    profile = CharacterProfileDocument(
        character_id="CON-abc",
        name="Mona",
        fictional_age=52,
        adult_eligible=True,
        kinship="family",
        interests=("cooking", "gardening"),
    )
    files.write_profile(profile)
    back = files.read_profile()
    assert back is not None
    assert back.name == "Mona"
    assert back.kinship == "family"
    assert back.interests == ("cooking", "gardening")
    assert back.adult_eligible is True


def test_a_missing_profile_reads_as_none(tmp_path: Path):
    assert CharacterFiles("CON-none", root=tmp_path).read_profile() is None


def test_a_newer_profile_version_is_refused(tmp_path: Path):
    files = CharacterFiles("CON-x", root=tmp_path)
    files.ensure()
    files.profile_path.write_text('{"version": 99, "character_id": "CON-x"}', encoding="utf-8")
    with pytest.raises(CharacterFilesError, match="version"):
        files.read_profile()


def test_a_character_id_never_escapes_its_folder(tmp_path: Path):
    files = CharacterFiles("../../etc/passwd", root=tmp_path)
    # The folder stays inside the root, whatever the id looks like.
    assert tmp_path in files.directory.parents
    assert ".." not in files.directory.name


def test_the_knowledge_bank_is_real_and_retrievable(tmp_path: Path):
    files = CharacterFiles("CON-kb", root=tmp_path)
    assert files.knowledge_available() is False
    written = files.add_knowledge(
        "Recipes", "She cooks molokhia every Friday. The secret is garlic and coriander."
    )
    assert written >= 1
    assert files.knowledge_available() is True
    hits = files.retrieve("what does she cook")
    assert hits, "the companion's own knowledge must be retrievable"
    assert "molokhia" in hits[0].text
    block = files.reference_block("what does she cook")
    assert block.startswith("Reference material from your own knowledge")
    assert "molokhia" in block


def test_a_blank_document_adds_no_knowledge(tmp_path: Path):
    files = CharacterFiles("CON-kb", root=tmp_path)
    assert files.add_knowledge("Empty", "   ") == 0
    assert files.knowledge_available() is False


def test_facts_are_durable_and_de_duplicated(tmp_path: Path):
    files = CharacterFiles("CON-facts", root=tmp_path)
    assert files.add_fact("Loves jasmine tea") is True
    assert files.add_fact("loves jasmine tea") is False  # a repeat is refused
    assert files.facts() == ("Loves jasmine tea",)
    assert files.forget_fact("Loves jasmine tea") is True
    assert files.facts() == ()


def test_exchanges_are_recorded_and_recalled(tmp_path: Path):
    files = CharacterFiles("CON-ex", root=tmp_path)
    files.remember("How are you?", "I am well, habibi.")
    assert len(files.exchanges()) == 1
    recall = files.recall()
    assert any("How are you?" in line for line in recall)


def test_a_corrupt_learning_file_reads_as_empty(tmp_path: Path):
    files = CharacterFiles("CON-bad", root=tmp_path)
    files.ensure()
    files.learning_path.write_text("{not json", encoding="utf-8")
    assert files.facts() == ()
    assert files.exchanges() == ()


def test_the_summary_reports_what_the_folder_holds(tmp_path: Path):
    files = CharacterFiles("CON-sum", root=tmp_path)
    files.write_profile(CharacterProfileDocument(character_id="CON-sum", name="Lina"))
    files.add_knowledge("Note", "She grew up by the sea and still swims every morning.")
    files.add_fact("Has a sister")
    files.remember("Hi", "Hello")
    summary = files.summary()
    assert summary.profile_exists is True
    assert summary.knowledge_chunks >= 1
    assert summary.facts == 1
    assert summary.exchanges == 1
    assert "1 fact" in summary.describe()


def test_one_companion_cannot_read_anothers_files(tmp_path: Path):
    a = CharacterFiles("CON-a", root=tmp_path)
    b = CharacterFiles("CON-b", root=tmp_path)
    a.add_fact("a secret")
    assert b.facts() == ()
    assert a.directory != b.directory


# -- the service wires the files ------------------------------------------


def test_creating_a_character_writes_its_profile_file():
    with _service() as service:
        bundle = service.create_character(
            "Yara", fictional_age=29, kinship="girlfriend", biography="A poet."
        )
        files = service.character_files(bundle.id)
        profile = files.read_profile()
        assert profile is not None
        assert profile.name == "Yara"
        assert profile.kinship == "girlfriend"


def test_a_turn_is_recorded_into_the_companions_own_learning():
    with _service(_Seam("I have missed you.")) as service:
        bundle = service.create_character("Nour", fictional_age=27)
        conversation = service.start_conversation(bundle.id)
        service.send(conversation.record_id, "Do you remember me?")
        exchanges = service.character_files(bundle.id).exchanges()
        assert exchanges and exchanges[0]["question"] == "Do you remember me?"


def test_the_companions_own_knowledge_reaches_the_prompt():
    seam = _Seam("Of course.")
    with _service(seam) as service:
        bundle = service.create_character("Rana", fictional_age=31)
        service.add_character_knowledge(
            bundle.id, "Her work", "She is a marine biologist studying the Red Sea reefs."
        )
        conversation = service.start_conversation(bundle.id)
        service.send(conversation.record_id, "Tell me about the reefs")
        # The prompt the model was handed really carried the companion's own
        # knowledge block -- the file is not merely stored, it is used.
        assert seam.calls
        assert "marine biologist" in seam.calls[-1]


# -- the reach dials ------------------------------------------------------


def test_the_reach_dials_render_with_the_floor():
    from core.romance.personality import CharacterPersonalityEngine

    engine = CharacterPersonalityEngine()
    block = engine.reach_block(_Personality(boldness=0.95, passion=0.9, sensuality=0.9, escalation=0.8))
    assert "consent" in block.lower()
    assert "never pressure" in block.lower() or "never the floor" in block.lower()
    # The general floor names adults only.
    assert "18" in SAFETY_FLOOR


def test_every_reach_dial_is_a_known_personality_field():
    from core.romance.personality import CharacterPersonalityEngine

    engine = CharacterPersonalityEngine()
    clean = engine.validate(dict.fromkeys(REACH_DIALS, 0.7))
    assert set(clean) == set(REACH_DIALS)
    assert all(0.0 <= value <= 1.0 for value in clean.values())


def test_the_service_reports_the_reach_dials():
    with _service() as service:
        bundle = service.create_character("Dalia", fictional_age=33)
        values = service.personality(bundle.id)
        for name in REACH_DIALS:
            assert name in values


class _Personality:
    """A minimal stand-in carrying only the dials a block reads."""

    def __init__(self, **values: float) -> None:
        for name in (
            "introversion", "playfulness", "warmth", "expressiveness", "confidence",
            "humour", "initiative", "affection", "romantic_intensity", "formality",
            *REACH_DIALS,
        ):
            setattr(self, name, values.get(name, 0.5))
        self.response_length = "balanced"
        self.dialect = ""
        self.traits = ()
        self.instructions = ""


# -- the library ----------------------------------------------------------


def test_every_library_character_is_adult_eligible_at_the_source():
    assert all(c.draft.adult_eligible for c in LIBRARY_CHARACTERS)


def test_every_library_character_carries_a_kinship():
    assert all(c.draft.kinship for c in LIBRARY_CHARACTERS)


def test_the_parents_archetype_ships_a_mother_and_a_father():
    parents = library_by_category(CATEGORY_PARENTS)
    assert len(parents) == 2
    kin = {c.draft.kinship for c in parents}
    assert kin == {"mother", "father"}


def test_an_adult_eligible_library_character_needs_a_stated_adult_age():
    # The manager refuses an adult-eligible character with an unknown or minor age,
    # so the library's own eligibility rests on a real, stated adult age.
    assert all(c.age is not None and c.age >= 18 for c in LIBRARY_CHARACTERS)


def test_every_library_character_carries_a_reach():
    for character in LIBRARY_CHARACTERS:
        for name in REACH_DIALS:
            assert name in character.draft.personality, f"{character.name} lacks {name}"


def test_a_lover_reaches_further_than_a_parent():
    lover = library_by_category("lovers")[0]
    parent = library_by_category(CATEGORY_PARENTS)[0]
    assert lover.draft.personality["passion"] > parent.draft.personality["passion"]
    assert lover.draft.personality["sensuality"] > parent.draft.personality["sensuality"]


def test_an_explicit_reach_dial_overrides_the_archetype_default():
    from core.romance.library import _char

    custom = _char(
        "lovers", "Test", "تست", 30, "woman",
        biography="A test character.", dials={"passion": 0.1},
    )
    assert custom.draft.personality["passion"] == 0.1
    # The other reach dials keep the archetype's own default.
    assert custom.draft.personality["boldness"] > 0.0
