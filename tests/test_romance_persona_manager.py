"""Tests for the ready-made persona library and the two manager windows.

The user asked for three things in the Studio's left sidebar: ready-made
**personas** to talk as at once, a **My Persona Manager** that controls every
attribute of the user's own personas (including their pictures), and a
**Character Manager** that does the same for the companions. These tests pin the
persona data (every one a stated adult, every name distinct, no clash with the
companion library), the service and CLI that expose it, and the two windows --
the Studio's new sidebar sections and the manager windows themselves.
"""

from __future__ import annotations

import gc
import json
import tempfile
from pathlib import Path

import pytest
from cli import romance as cli
from core.romance.library import (
    LIBRARY_CHARACTERS,
    PERSONA_CATEGORIES,
    PERSONA_LIBRARY_CHARACTERS,
    find_persona_library_character,
    library_by_category_for,
    persona_category_keys,
    seed_library_for,
)
from core.romance.model import ROLE_COMPANION, ROLE_USER
from core.romance.personality import CharacterPersonalityEngine
from gui.romance import RomanceStudioService

# -- the persona data -----------------------------------------------------


def test_the_persona_library_is_a_real_library() -> None:
    assert len(PERSONA_LIBRARY_CHARACTERS) >= 10
    keys = set(persona_category_keys())
    assert {"anyone", "professional", "academic", "artist", "explorer", "seeker"} <= keys
    for category in PERSONA_CATEGORIES:
        assert category.title and category.icon and category.summary


def test_every_persona_is_a_fully_specified_adult_for_the_user() -> None:
    engine = CharacterPersonalityEngine()
    for character in PERSONA_LIBRARY_CHARACTERS:
        draft = character.draft
        assert draft.role == ROLE_USER, character.name
        assert character.age is not None and character.age >= 18, character.name
        assert draft.name.strip() and draft.biography.strip() and draft.gender.strip()
        assert character.blurb.strip()
        assert draft.default_mode
        assert draft.profile_items
        engine.validate(draft.personality)


def test_every_persona_name_is_distinct_and_does_not_clash_with_the_companions() -> None:
    persona_names = [c.name.casefold() for c in PERSONA_LIBRARY_CHARACTERS]
    assert len(set(persona_names)) == len(persona_names)
    companion_names = {c.name.casefold() for c in LIBRARY_CHARACTERS}
    # A persona and a companion sharing a name would make "add by name" ambiguous.
    assert not (set(persona_names) & companion_names)


def test_the_persona_library_is_separate_from_the_companion_library() -> None:
    personas = library_by_category_for(ROLE_USER)
    companions = library_by_category_for(ROLE_COMPANION)
    assert personas and companions
    assert all(c.draft.role == ROLE_USER for c in personas)
    assert all(c.draft.role == ROLE_COMPANION for c in companions)
    # A category name that exists in one catalogue does not leak into the other.
    assert all(c.category in set(persona_category_keys()) for c in personas)


def test_find_persona_library_character_is_exact_and_honest() -> None:
    assert find_persona_library_character("You") is not None
    assert find_persona_library_character("No Such Persona") is None
    # The companion lookup never returns a persona.
    from core.romance.library import find_library_character

    assert find_library_character("You") is None


def test_seed_library_for_creates_only_the_personas() -> None:
    from core.romance.characters import CharacterManager
    from core.romance.storage import CompanionStorageService

    with tempfile.TemporaryDirectory() as folder:
        manager = CharacterManager(CompanionStorageService(root=Path(folder)))
        created = seed_library_for(manager, role=ROLE_USER)
        assert len(created) == len(PERSONA_LIBRARY_CHARACTERS)
        rows = manager.list_characters(include_archived=True)
        assert all(row.character.role == ROLE_USER for row in rows)
        # Seeding twice never duplicates.
        assert seed_library_for(manager, role=ROLE_USER) == []


# -- the service ----------------------------------------------------------


def test_the_service_lists_the_persona_library(tmp_path: Path) -> None:
    service = RomanceStudioService(root=tmp_path)
    assert len(service.library_rows(role=ROLE_USER)) == len(PERSONA_LIBRARY_CHARACTERS)
    assert service.persona_library_categories()
    # The companion catalogue is untouched.
    assert len(service.library_rows()) == len(LIBRARY_CHARACTERS)


def test_adding_a_ready_made_persona_makes_it_active_and_speaks_as_it(tmp_path: Path) -> None:
    service = RomanceStudioService(root=tmp_path)
    bundle = service.add_persona_library_character("Dr. Salma")
    assert bundle.character.role == ROLE_USER
    service.activate_persona(bundle.id)
    assert service.active_persona_name() == "Dr. Salma"
    assert [row.name for row in service.persona_rows()] == ["Dr. Salma"]


def test_adding_a_ready_made_persona_twice_does_not_duplicate(tmp_path: Path) -> None:
    service = RomanceStudioService(root=tmp_path)
    first = service.add_persona_library_character("Dr. Salma")
    second = service.add_persona_library_character("Dr. Salma")
    assert first.id == second.id
    assert len(service.persona_rows()) == 1


def test_the_companion_library_still_adds_companions(tmp_path: Path) -> None:
    service = RomanceStudioService(root=tmp_path)
    name = service.library_rows()[0].name
    bundle = service.add_library_character(name)
    assert bundle.character.role == ROLE_COMPANION
    assert service.persona_rows() == []


def test_seeding_a_persona_archetype_creates_only_that_group(tmp_path: Path) -> None:
    service = RomanceStudioService(root=tmp_path)
    created = service.seed_persona_library_category("academic")
    assert created >= 2
    assert all(row.role == ROLE_USER for row in service.persona_rows())
    assert len(service.persona_rows()) == created


# -- the CLI --------------------------------------------------------------


def test_cli_persona_categories_and_library(capsys, tmp_path: Path) -> None:
    assert cli.main(["--data-root", str(tmp_path), "persona-categories", "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert any(entry["key"] == "academic" for entry in payload)

    assert cli.main(
        ["--data-root", str(tmp_path), "persona-library", "--category", "academic", "--json"]
    ) == 0
    rows = json.loads(capsys.readouterr().out)
    assert rows and all(row["name"] for row in rows)


def test_cli_add_persona_library_speaks_as_it(capsys, tmp_path: Path) -> None:
    service = RomanceStudioService(root=tmp_path)
    assert cli.main(
        ["--data-root", str(tmp_path), "add-persona-library", "Dr. Salma", "--json"]
    ) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["name"] == "Dr. Salma"
    assert service.active_persona_name() == "Dr. Salma"


def test_cli_seed_persona_library(capsys, tmp_path: Path) -> None:
    assert cli.main(
        ["--data-root", str(tmp_path), "seed-persona-library", "--category", "artist", "--json"]
    ) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["created"] >= 2


# -- the windows ----------------------------------------------------------


def _display_available() -> bool:
    try:
        import tkinter

        root = tkinter.Tk()
        root.withdraw()
        root.destroy()
    except Exception:
        return False
    return True


def _close_window(root, dialog) -> None:
    try:
        dialog._close()
    finally:
        root.destroy()
        gc.collect()


@pytest.mark.skipif(not _display_available(), reason="no display available")
def test_the_persona_manager_window_controls_and_avatars(tmp_path: Path) -> None:
    import tkinter as tk

    from gui.app import DARK_THEME
    from gui.romance_managers import TITLE_PERSONA_MANAGER, CharacterManagerDialog

    service = RomanceStudioService(root=tmp_path)
    root = tk.Tk()
    root.withdraw()
    try:
        dialog = CharacterManagerDialog(
            root, service, role=ROLE_USER,
            theme=DARK_THEME, font="TkDefaultFont", mono="TkFixedFont",
        )
        root.update()
        assert dialog.window.title() == TITLE_PERSONA_MANAGER
        # The ready-made personas are listed, and adding one selects it.
        assert len(dialog._library_rows) == len(PERSONA_LIBRARY_CHARACTERS)
        dialog._library_list.selection_set(0)
        dialog._add_library()
        root.update()
        assert dialog._selected
        assert service.persona_rows()
        # Every attribute the profile exposes is editable and saves.
        dialog._vars["name"].set("Aya")
        dialog._vars["gender"].set("woman")
        dialog._vars["age"].set("29")
        dialog._check_vars["persona_active"].set(True)
        dialog._save()
        root.update()
        assert service.active_persona_name() == "Aya"
        # The picture is set, generated, and cleared through the manager.
        dialog._generate_avatar()
        root.update()
        assert service.avatar_path(dialog._selected) is not None
        dialog._clear_avatar()
        root.update()
        assert service.avatar_path(dialog._selected) is None
    finally:
        _close_window(root, dialog)


@pytest.mark.skipif(not _display_available(), reason="no display available")
def test_the_character_manager_window_controls_companions(tmp_path: Path) -> None:
    import tkinter as tk

    from gui.app import DARK_THEME
    from gui.romance_managers import TITLE_CHARACTER_MANAGER, CharacterManagerDialog

    service = RomanceStudioService(root=tmp_path)
    root = tk.Tk()
    root.withdraw()
    try:
        dialog = CharacterManagerDialog(
            root, service, role=ROLE_COMPANION,
            theme=DARK_THEME, font="TkDefaultFont", mono="TkFixedFont",
        )
        root.update()
        assert dialog.window.title() == TITLE_CHARACTER_MANAGER
        assert len(dialog._library_rows) == len(LIBRARY_CHARACTERS)
        dialog._library_list.selection_set(0)
        dialog._add_library()
        root.update()
        assert dialog._selected
        bundle = service.character(dialog._selected)
        assert bundle.character.role == ROLE_COMPANION
        # A companion's profile saves like a persona's, and its picture is set.
        dialog._vars["nickname"].set("Habibi")
        dialog._save()
        root.update()
        assert service.character(dialog._selected).character.nickname == "Habibi"
        dialog._generate_avatar()
        root.update()
        assert service.avatar_path(dialog._selected) is not None
    finally:
        _close_window(root, dialog)


@pytest.mark.skipif(not _display_available(), reason="no display available")
def test_the_manager_controls_every_attribute_including_interests_and_boundaries(
    tmp_path: Path,
) -> None:
    import tkinter as tk

    from gui.app import DARK_THEME
    from gui.romance_managers import CharacterManagerDialog

    service = RomanceStudioService(root=tmp_path)
    root = tk.Tk()
    root.withdraw()
    try:
        dialog = CharacterManagerDialog(
            root, service, role=ROLE_USER,
            theme=DARK_THEME, font="TkDefaultFont", mono="TkFixedFont",
        )
        root.update()
        dialog._library_list.selection_set(0)
        dialog._add_library()
        root.update()
        # Every scalar field, the personality dials, and the reach dials save.
        dialog._vars["nickname"].set("Sally")
        dialog._vars["nationality"].set("Egyptian")
        dialog._vars["languages"].set("Arabic, English")
        dialog._vars["dialect"].set("Egyptian")
        dialog._combo_set("kinship", "partner")
        for scale in dialog._dial_scales.values():
            scale.set(0.7)
        dialog._save()
        root.update()
        bundle = service.character(dialog._selected)
        assert bundle.character.nickname == "Sally"
        assert bundle.character.kinship == "partner"
        assert bundle.character.languages == ("Arabic", "English")
        personality = service.personality(dialog._selected)
        assert personality["passion"] == pytest.approx(0.7)
        assert personality["dialect"] == "Egyptian"
        # Interests and boundaries are added and removed through the manager.
        dialog._interest_entry.insert(0, "poetry")
        dialog._add_interest()
        root.update()
        assert any(item.label == "poetry" for item in service.character(dialog._selected).profile_items)
        dialog._interest_list.selection_set(len(dialog._interest_ids) - 1)
        dialog._remove_interest()
        assert all(item.label != "poetry" for item in service.character(dialog._selected).profile_items)
        dialog._boundary_entry.insert(0, "no politics")
        dialog._add_boundary()
        root.update()
        assert [b.value for b in service.boundaries(dialog._selected)] == ["no politics"]
    finally:
        _close_window(root, dialog)


@pytest.mark.skipif(not _display_available(), reason="no display available")
def test_the_studio_sidebar_has_both_managers_and_the_persona_library(tmp_path: Path) -> None:
    import tkinter as tk

    from gui.app import DARK_THEME
    from gui.romance_dialog import RomanceStudioDialog

    service = RomanceStudioService(root=tmp_path)
    root = tk.Tk()
    root.withdraw()
    try:
        dialog = RomanceStudioDialog(
            root, service, theme=DARK_THEME, font="TkDefaultFont", mono="TkFixedFont",
        )
        root.update()
        # The new sidebar sections exist.
        assert dialog._persona_library_list is not None
        assert len(dialog._persona_library_rows) == len(PERSONA_LIBRARY_CHARACTERS)
        # Picking a ready-made persona speaks as it at once.
        dialog._persona_library_list.selection_set(0)
        dialog._add_persona_library()
        root.update()
        assert service.persona_rows()
        # Both managers open, and one window per role is reused.
        dialog._open_persona_manager()
        dialog._open_character_manager()
        root.update()
        assert set(dialog._managers) == {ROLE_USER, ROLE_COMPANION}
        assert dialog._managers[ROLE_USER].window.title() == "My Persona Manager"
        assert dialog._managers[ROLE_COMPANION].window.title() == "Character Manager"
        first = dialog._managers[ROLE_USER]
        dialog._open_persona_manager()
        assert dialog._managers[ROLE_USER] is first
        # A change made in a manager refreshes the Studio behind it.
        dialog._managers[ROLE_USER]._library_list.selection_set(1)
        dialog._managers[ROLE_USER]._add_library()
        root.update()
        assert len(service.persona_rows()) >= 2
    finally:
        for manager in getattr(dialog, "_managers", {}).values():
            try:
                manager._close()
            except Exception:
                pass
        _close_window(root, dialog)
