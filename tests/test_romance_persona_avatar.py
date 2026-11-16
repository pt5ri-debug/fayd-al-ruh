"""The user's own persona and each character's own avatar.

Two additions to the Romance & Companion Studio, tested at the level they live:

* **The user's persona.** A character record with ``role=ROLE_USER`` is who the
  *user* is in the fiction -- the name the companions address them by, and the
  identity block the prompt renders. Exactly one persona is active at a time, a
  persona is never listed as a companion, and a companion is never a persona.
* **The avatar.** Each character (companion *or* persona) carries its own
  picture, uploaded or generated, kept in its own folder so one character can
  never reach another's.

The service is the Tk-free half the window and the CLI share, so everything here
runs headless with an injected generation seam (no model) and a throwaway data
root -- except the CLI, which is driven through its own ``main``.
"""

from __future__ import annotations

import json
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import pytest
from cli import romance as cli
from core.romance.avatar import (
    AVATAR_SUFFIXES,
    AvatarError,
    CharacterAvatarManager,
    avatar_prompt,
    is_image_name,
)
from core.romance.model import ROLE_COMPANION, ROLE_USER
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


def _png(path: Path) -> Path:
    path.write_bytes(b"\x89PNG\r\n\x1a\n" + b"0" * 48)
    return path


# -- the persona ----------------------------------------------------------


def test_a_persona_carries_the_user_role_and_is_the_only_active_one():
    with _service() as service:
        first = service.create_persona("Omar", fictional_age=31)
        assert first.character.role == ROLE_USER
        # The first persona becomes the one the user speaks as.
        assert first.character.is_active is True
        assert service.active_persona_name() == "Omar"

        second = service.create_persona("Sara", fictional_age=29)
        service.activate_persona(second.id)
        assert service.active_persona_name() == "Sara"
        # Switching deactivates the other, so exactly one is ever active.
        assert service.characters.get(first.id).character.is_active is False


def test_a_companion_is_never_a_persona_and_a_persona_is_never_a_companion():
    with _service() as service:
        service.create_character("Layla", fictional_age=27)
        service.create_persona("Omar", fictional_age=31)
        assert {row.name for row in service.character_rows()} == {"Layla"}
        assert {row.name for row in service.persona_rows()} == {"Omar"}


def test_activating_a_companion_is_refused():
    from core.romance.characters import CharacterError

    with _service() as service:
        companion = service.create_character("Layla", fictional_age=27)
        with pytest.raises(CharacterError):
            service.activate_persona(companion.id)


def test_the_persona_reaches_the_prompt_the_companion_reads():
    seam = _Seam("Nice to meet you.")
    with _service(seam) as service:
        companion = service.create_character("Layla", fictional_age=27)
        service.create_persona(
            "Omar", fictional_age=31, biography="a petroleum engineer",
            appearance="tall, dark hair", nationality="Egypt",
        )
        conversation = service.start_conversation(companion.id)
        turn = service.send(conversation.record_id, "Hello.")
        assert turn.ok
        # The prompt names the user's persona and carries its own identity block.
        prompt = seam.calls[-1]
        assert "Omar" in prompt
        assert "petroleum engineer" in prompt


def test_a_user_with_no_persona_keeps_the_plain_you_label():
    with _service() as service:
        service.create_character("Layla", fictional_age=27)
        assert service.persona_label() == "You"
        assert service.active_persona_name() == ""


# -- the avatar -----------------------------------------------------------


def test_an_avatar_is_generated_from_the_characters_own_profile():
    with _service() as service:
        bundle = service.create_character(
            "Layla", fictional_age=27, appearance="silver hair"
        )
        result = service.generate_avatar(bundle.id)
        assert result.ok
        path = service.avatar_path(bundle.id)
        assert path is not None and path.is_file()
        # The record points at it, relative to the character's own folder.
        assert service.character(bundle.id).character.avatar_path


def test_an_avatar_is_uploaded_byte_for_byte():
    with tempfile.TemporaryDirectory() as folder:
        root = Path(folder)
        service = RomanceStudioService(root=root, generator=_Seam())
        bundle = service.create_character("Layla", fictional_age=27)
        source = _png(root / "chosen.png")
        result = service.set_avatar_from_file(bundle.id, source)
        assert result.ok
        stored = service.avatar_path(bundle.id)
        assert stored is not None
        assert stored.read_bytes() == source.read_bytes()


def test_clearing_an_avatar_removes_the_picture():
    with _service() as service:
        bundle = service.create_character("Layla", fictional_age=27)
        service.generate_avatar(bundle.id)
        assert service.avatar_path(bundle.id) is not None
        service.clear_avatar(bundle.id)
        assert service.avatar_path(bundle.id) is None


def test_one_character_never_reads_anothers_avatar():
    with _service() as service:
        first = service.create_character("Layla", fictional_age=27)
        second = service.create_character("Noor", fictional_age=30)
        service.generate_avatar(first.id)
        assert service.avatar_path(second.id) is None
        manager = service.avatar_manager()
        assert manager.folder(first.id) != manager.folder(second.id)


def test_a_non_image_upload_is_refused():
    with tempfile.TemporaryDirectory() as folder:
        root = Path(folder)
        service = RomanceStudioService(root=root, generator=_Seam())
        bundle = service.create_character("Layla", fictional_age=27)
        stray = root / "notes.txt"
        stray.write_text("not a picture", encoding="utf-8")
        with pytest.raises(AvatarError):
            service.set_avatar_from_file(bundle.id, stray)


def test_an_unknown_character_is_refused_by_the_avatar_engine():
    with tempfile.TemporaryDirectory() as folder:
        root = Path(folder)
        service = RomanceStudioService(root=root, generator=_Seam())
        with pytest.raises(AvatarError):
            service.generate_avatar("CHR-missing")


def test_the_avatar_engine_refuses_an_unknown_backend():
    with tempfile.TemporaryDirectory() as folder:
        root = Path(folder)
        service = RomanceStudioService(root=root, generator=_Seam())
        bundle = service.create_character("Layla", fictional_age=27)
        with pytest.raises(AvatarError):
            service.generate_avatar(bundle.id, backend="nope")


def test_the_image_name_check_and_the_prompt_builder():
    assert is_image_name("photo.PNG") is True
    assert is_image_name("notes.txt") is False
    assert ".png" in AVATAR_SUFFIXES
    prompt = avatar_prompt("Layla", appearance="silver hair", gender="woman")
    assert "Layla" in prompt and "silver hair" in prompt and "woman" in prompt
    # A character with nothing recorded still yields a usable prompt.
    assert avatar_prompt("") != ""


def test_the_avatar_manager_is_isolated_per_character_folder():
    with tempfile.TemporaryDirectory() as folder:
        root = Path(folder)
        service = RomanceStudioService(root=root, generator=_Seam())
        manager = service.avatar_manager()
        assert isinstance(manager, CharacterAvatarManager)
        assert manager.folder("CHR-1") != manager.folder("CHR-2")


def test_a_persona_can_carry_its_own_avatar_too():
    with _service() as service:
        persona = service.create_persona("Omar", fictional_age=31)
        result = service.generate_avatar(persona.id)
        assert result.ok
        assert service.avatar_path(persona.id) is not None


# -- the CLI --------------------------------------------------------------


def _run(argv: list[str]) -> int:
    return cli.main(argv)


def test_the_cli_creates_and_activates_a_persona(tmp_path: Path, capsys):
    root = str(tmp_path)
    assert _run(["new-persona", "Omar", "--age", "31", "--activate", "--data-root", root,
                 "--json"]) == 0
    created = json.loads(capsys.readouterr().out)
    assert created["active"] is True
    assert _run(["personas", "--data-root", root, "--json"]) == 0
    listing = json.loads(capsys.readouterr().out)
    assert listing["active"] == "Omar"
    assert listing["personas"][0]["role"] == ROLE_USER


def test_the_cli_keeps_personas_out_of_the_character_list(tmp_path: Path, capsys):
    root = str(tmp_path)
    assert _run(["new", "Layla", "--age", "27", "--data-root", root, "--json"]) == 0
    capsys.readouterr()
    assert _run(["new-persona", "Omar", "--age", "31", "--data-root", root, "--json"]) == 0
    capsys.readouterr()
    assert _run(["characters", "--data-root", root, "--json"]) == 0
    characters = json.loads(capsys.readouterr().out)
    assert [row["name"] for row in characters] == ["Layla"]


def test_the_cli_switches_the_active_persona(tmp_path: Path, capsys):
    root = str(tmp_path)
    assert _run(["new-persona", "Omar", "--age", "31", "--activate", "--data-root", root,
                 "--json"]) == 0
    first = json.loads(capsys.readouterr().out)["character_id"]
    assert _run(["new-persona", "Sara", "--age", "29", "--data-root", root, "--json"]) == 0
    second = json.loads(capsys.readouterr().out)["character_id"]
    assert _run(["use-persona", second, "--data-root", root, "--json"]) == 0
    capsys.readouterr()
    assert _run(["personas", "--data-root", root, "--json"]) == 0
    listing = json.loads(capsys.readouterr().out)
    assert listing["active"] == "Sara"
    by_id = {row["character_id"]: row for row in listing["personas"]}
    assert by_id[second]["is_active"] is True
    assert by_id[first]["is_active"] is False


def test_the_cli_generates_and_clears_an_avatar(tmp_path: Path, capsys):
    root = str(tmp_path)
    assert _run(["new", "Layla", "--age", "27", "--data-root", root, "--json"]) == 0
    cid = json.loads(capsys.readouterr().out)["character_id"]
    assert _run(["avatar", cid, "--generate", "--data-root", root, "--json"]) == 0
    generated = json.loads(capsys.readouterr().out)
    assert Path(generated["path"]).is_file()
    assert _run(["avatar", cid, "--clear", "--data-root", root, "--json"]) == 0
    capsys.readouterr()
    # After clearing, generating again is the honest path (no picture is a no-op).
    assert _run(["avatar", cid, "--generate", "--data-root", root, "--json"]) == 0
    capsys.readouterr()


def test_the_cli_show_reports_the_role(tmp_path: Path, capsys):
    root = str(tmp_path)
    assert _run(["new", "Layla", "--age", "27", "--data-root", root, "--json"]) == 0
    cid = json.loads(capsys.readouterr().out)["character_id"]
    assert _run(["show", cid, "--data-root", root, "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["role"] == ROLE_COMPANION
