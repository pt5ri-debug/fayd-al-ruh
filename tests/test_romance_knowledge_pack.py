"""Tests for each companion's **own knowledge base** (its voice and world).

Covers :mod:`core.romance.knowledge_pack`: the generator that renders a
companion's own words, sentences, self-knowledge, and domain material into the
character's own knowledge bank, and the wiring that reaches a conversation.

Everything runs with no model, no display, and no network.
"""

from __future__ import annotations

import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import pytest
from core.romance import knowledge_pack as kp
from core.romance.character_files import (
    CharacterFiles,
    CharacterFilesError,
    CharacterProfileDocument,
)
from core.romance.library import LIBRARY_CHARACTERS
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


def _profile(character_id: str = "CON-a", name: str = "Layla") -> CharacterProfileDocument:
    return CharacterProfileDocument(
        character_id=character_id,
        name=name,
        nickname="dear",
        fictional_age=27,
        adult_eligible=True,
        gender="woman",
        biography="She grew up by the sea and never quite left it.",
        nationality="Egyptian",
        languages=("Arabic", "English"),
        appearance="Tall, with a habit of laughing before the joke lands.",
        relationship_style="Warm and direct.",
        kinship="partner",
        default_mode="romantic",
        interests=("cooking", "old films", "walking by the sea"),
        boundaries=("Adults only.",),
        personality={
            "warmth": 0.8,
            "playfulness": 0.6,
            "humour": 0.7,
            "romantic_intensity": 0.7,
            "dialect": "Egyptian",
            "traits": ("warm", "observant", "funny"),
        },
    )


# -- the generator ---------------------------------------------------------


def test_the_generator_is_deterministic():
    first = list(kp.iter_documents(_profile(), "en"))
    second = list(kp.iter_documents(_profile(), "en"))
    assert first == second
    assert first  # non-empty


def test_documents_are_character_specific():
    a = {title for _s, title, _t in kp.iter_documents(_profile("CON-a", "Layla"), "en")}
    b = {title for _s, title, _t in kp.iter_documents(_profile("CON-b", "Noor"), "en")}
    assert a and b
    assert a != b


def test_every_section_is_generated():
    sections = {section for section, _t, _x in kp.iter_documents(_profile(), "en")}
    assert {
        "voice_card",
        "biography",
        "personality",
        "phrasebook",
        "interests",
        "knowledge",
        "dialogue",
        "scenes",
    } <= sections


def test_both_languages_are_generated():
    en = "\n".join(text for _s, _t, text in kp.iter_documents(_profile(), "en"))
    ar = "\n".join(text for _s, _t, text in kp.iter_documents(_profile(), "ar"))
    assert "Layla" in en
    # The Arabic pack names the character and carries Arabic letters.
    assert any("\u0600" <= ch <= "\u06ff" for ch in ar)


def test_the_voice_card_names_the_character():
    body = kp.voice_block(_profile(), language="en")
    assert body.startswith(kp.VOICE_HEADER)
    assert "Layla" in body


def test_the_voice_card_is_bounded():
    body = kp.voice_block(_profile(), language="en", max_chars=200)
    assert len(body) <= 200
    # A bound smaller than the header keeps the header whole rather than cutting it.
    tiny = kp.voice_block(_profile(), language="en", max_chars=5)
    assert tiny.startswith(kp.VOICE_HEADER)


def test_a_minimal_profile_still_builds_a_pack(tmp_path: Path):
    report = kp.build_knowledge_pack(
        tmp_path, CharacterProfileDocument(character_id="CON-min", name="Min"), target_mb=1.0
    )
    assert report.chunks > 0
    assert report.size_mb > 0


# -- the builder -----------------------------------------------------------


def test_a_build_reaches_its_target(tmp_path: Path):
    report = kp.build_knowledge_pack(tmp_path, _profile(), target_mb=2.0, languages=("en",))
    assert report.complete is True
    assert report.size_mb >= 2.0
    assert report.index_built is True
    assert report.chunks > 0


def test_a_build_writes_a_real_retrievable_store(tmp_path: Path):
    kp.build_knowledge_pack(tmp_path, _profile(), target_mb=1.0, languages=("en",))
    stats = kp.pack_stats(tmp_path)
    assert stats["present"] is True
    assert stats["source"] == kp.PACK_SOURCE
    assert stats["indexed"] is True
    # A real search finds the character's own material.
    from core.knowledge_base import KnowledgeBaseStore
    from core.knowledge_index import KnowledgeTextIndex

    store = KnowledgeBaseStore(tmp_path)
    index = KnowledgeTextIndex(tmp_path, store=store)
    hits = index.search("hello", limit=3)
    assert hits


def test_a_rebuild_adds_nothing(tmp_path: Path):
    first = kp.build_knowledge_pack(tmp_path, _profile(), target_mb=1.0, languages=("en",))
    second = kp.build_knowledge_pack(tmp_path, _profile(), target_mb=1.0, languages=("en",))
    assert second.size_mb == first.size_mb
    assert second.chunks == first.chunks


def test_reset_rebuilds_from_empty(tmp_path: Path):
    kp.build_knowledge_pack(tmp_path, _profile(), target_mb=3.0, languages=("en",))
    smaller = kp.build_knowledge_pack(
        tmp_path, _profile(), target_mb=1.0, languages=("en",), reset=True
    )
    assert smaller.complete is True
    assert 1.0 <= smaller.size_mb < 3.0


def test_the_target_is_clamped(tmp_path: Path):
    low = kp.build_knowledge_pack(tmp_path, _profile(), target_mb=-5.0, languages=("en",))
    assert low.target_mb == kp.TARGET_MB_RANGE[0]


def test_an_exhausted_space_is_reported_honestly(tmp_path: Path):
    # A tiny profile cannot fill 200 MB; the report says so rather than padding.
    report = kp.build_knowledge_pack(
        tmp_path,
        CharacterProfileDocument(character_id="CON-tiny", name="Tiny"),
        target_mb=200.0,
        languages=("en",),
    )
    assert report.complete is False
    assert report.note


def test_pack_stats_is_empty_without_a_pack(tmp_path: Path):
    assert kp.pack_stats(tmp_path) == {}


# -- CharacterFiles --------------------------------------------------------


def test_character_files_builds_its_own_pack(tmp_path: Path):
    files = CharacterFiles("CON-f", root=tmp_path)
    files.write_profile(_profile("CON-f"))
    report = files.build_knowledge_pack(target_mb=1.0, languages=("en",))
    assert report.chunks > 0
    assert files.knowledge_pack_stats()["present"] is True
    assert files.voice_block()


def test_character_files_refuses_a_pack_with_no_profile(tmp_path: Path):
    with pytest.raises(CharacterFilesError):
        CharacterFiles("CON-np", root=tmp_path).build_knowledge_pack()


def test_a_pack_lives_under_the_characters_own_folder(tmp_path: Path):
    files = CharacterFiles("CON-iso", root=tmp_path)
    files.write_profile(_profile("CON-iso"))
    files.build_knowledge_pack(target_mb=1.0, languages=("en",))
    assert files.knowledge_dir.is_relative_to(files.directory)


# -- the Studio service ----------------------------------------------------


def test_the_service_builds_and_reports_a_pack():
    with _service() as service:
        char = service.create_character("Layla", fictional_age=27)
        report = service.build_character_knowledge_pack(char.id, target_mb=1.0)
        assert report.chunks > 0
        stats = service.character_knowledge_pack_stats(char.id)
        assert stats["present"] is True
        assert service.character_voice_block(char.id)


def test_the_service_renders_a_library_character_and_builds_its_pack():
    with _service() as service:
        name = LIBRARY_CHARACTERS[0].name
        char = service.open_library_character(name)
        report = service.build_character_knowledge_pack(char.id, target_mb=1.0)
        assert report.chunks > 0
        block = service.character_voice_block(char.id)
        assert name.split()[0] in block


def test_the_memory_block_carries_the_voice_card_and_knowledge():
    with _service() as service:
        char = service.create_character("Layla", fictional_age=27)
        service.build_character_knowledge_pack(char.id, target_mb=1.0)
        conversation = service.start_conversation(char.id)
        block = service._memory_block(char.id, conversation.record_id)
        assert "Who you are" in block  # the voice card
        assert block.strip()  # not empty


def test_the_memory_block_is_bounded_and_safe_without_a_pack():
    with _service() as service:
        char = service.create_character("Noor", fictional_age=30)
        conversation = service.start_conversation(char.id)
        block = service._memory_block(char.id, conversation.record_id)
        # No pack yet: the voice card still reaches the prompt.
        assert "Who you are" in block


def test_a_turn_still_works_with_a_pack_built():
    seam = _Seam("I am here.")
    with _service(seam) as service:
        char = service.create_character("Layla", fictional_age=27)
        service.build_character_knowledge_pack(char.id, target_mb=1.0)
        conversation = service.start_conversation(char.id)
        turn = service.send(conversation.record_id, "Hello there.")
        assert turn.ok
        assert turn.replies
        assert seam.calls


# -- the CLI ---------------------------------------------------------------


def _run(argv: list[str]) -> int:
    from cli.romance import main

    return main(argv)


def test_cli_builds_and_reports_a_pack(tmp_path: Path, capsys: pytest.CaptureFixture[str]):
    root = str(tmp_path)
    assert _run(["new", "Noor", "--age", "30", "--data-root", root, "--json"]) == 0
    import json

    created = json.loads(capsys.readouterr().out)
    cid = created["character_id"]
    assert _run(["knowledge-pack", cid, "--target-mb", "1", "--lang", "en",
                 "--data-root", root, "--json"]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["chunks"] > 0
    assert report["complete"] is True
    assert _run(["knowledge-status", cid, "--data-root", root, "--json"]) == 0
    stats = json.loads(capsys.readouterr().out)
    assert stats["present"] is True


def test_cli_knowledge_status_is_honest_without_a_pack(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
):
    root = str(tmp_path)
    assert _run(["new", "Noor", "--age", "30", "--data-root", root, "--json"]) == 0
    import json

    cid = json.loads(capsys.readouterr().out)["character_id"]
    assert _run(["knowledge-status", cid, "--data-root", root, "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["present"] is False
