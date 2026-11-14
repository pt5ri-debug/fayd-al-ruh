"""Tests for the ready-made companion library and the redesigned Discover flow.

The library (:mod:`core.romance.library`) is *data*: a set of fully-specified
fictional adults grouped by relationship archetype. These tests pin its honesty
rules -- every character is a stated adult, every name is distinct, the dials are
valid -- and the wiring: browsing the library creates nothing, adding one creates
exactly one, and the service and CLI both expose it.
"""

from __future__ import annotations

import json
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

from cli import romance as cli
from core.romance.characters import CharacterManager
from core.romance.library import (
    CATEGORY_COURTESANS,
    CATEGORY_FAMILY,
    CATEGORY_INCLUSIVE,
    CATEGORY_LOVERS,
    LIBRARY_CATEGORIES,
    LIBRARY_CHARACTERS,
    category_keys,
    find_library_character,
    library_by_category,
    library_drafts,
    seed_library,
)
from core.romance.personality import CharacterPersonalityEngine
from core.romance.storage import CompanionStorageService
from gui.romance import RomanceStudioService


def _manager(root: Path) -> CharacterManager:
    return CharacterManager(CompanionStorageService(root=root))


@contextmanager
def _service() -> Iterator[RomanceStudioService]:
    with tempfile.TemporaryDirectory() as folder:
        yield RomanceStudioService(root=Path(folder))


# -- the library data -----------------------------------------------------


def test_the_library_is_large_and_covers_the_relationship_archetypes():
    # The user asked for a *library*, not a handful of examples.
    assert len(LIBRARY_CHARACTERS) >= 40
    keys = set(category_keys())
    # The archetypes the request named: family, neighbours, colleagues, lovers,
    # courtesans/sex workers, male escorts, and inclusive/LGBTQ+.
    assert {
        CATEGORY_FAMILY,
        CATEGORY_LOVERS,
        CATEGORY_COURTESANS,
        CATEGORY_INCLUSIVE,
        "neighbours",
        "colleagues",
        "escorts",
    } <= keys
    for category in LIBRARY_CATEGORIES:
        assert category.title and category.title_ar and category.icon and category.summary


def test_every_library_character_is_a_fully_specified_adult():
    engine = CharacterPersonalityEngine()
    for character in LIBRARY_CHARACTERS:
        draft = character.draft
        # A stated, real adult age: none is age-ambiguous.
        assert character.age is not None and character.age >= 18, character.name
        assert draft.name.strip(), "a character needs a name"
        assert draft.biography.strip(), f"{draft.name} needs a biography"
        assert draft.gender.strip(), f"{draft.name} needs a gender"
        assert character.blurb.strip(), f"{draft.name} needs a blurb"
        # The personality dials are valid, so the engine accepts them unchanged.
        engine.validate(draft.personality)
        # Every character carries a mode and at least one interest.
        assert draft.default_mode
        assert draft.profile_items


def test_every_library_name_is_distinct():
    names = [c.name.casefold() for c in LIBRARY_CHARACTERS]
    assert len(set(names)) == len(names)


def test_the_library_spans_both_men_and_women_and_inclusive_identities():
    genders = {c.draft.gender for c in LIBRARY_CHARACTERS}
    assert any("woman" in g for g in genders)
    assert any("man" in g for g in genders)
    assert any("non-binary" in g or "trans" in g for g in genders)


def test_the_library_marks_its_own_characters():
    from core.romance.library import LIBRARY_MARKER

    for character in LIBRARY_CHARACTERS:
        assert LIBRARY_MARKER in character.draft.tags
        assert character.draft.metadata.get("seed") == LIBRARY_MARKER


def test_category_filtering_and_lookup():
    family = library_by_category(CATEGORY_FAMILY)
    assert family and all(c.category == CATEGORY_FAMILY for c in family)
    assert library_by_category("") == LIBRARY_CHARACTERS
    assert find_library_character("ruby") is not None
    assert find_library_character("Nobody") is None


def test_library_drafts_are_read_only_reads():
    drafts = library_drafts()
    assert len(drafts) == len(LIBRARY_CHARACTERS)


def test_adult_characters_default_to_the_adult_mode():
    adult = library_by_category(CATEGORY_COURTESANS) + library_by_category("escorts")
    assert adult and all(c.draft.default_mode == "adult" for c in adult)


# -- seeding --------------------------------------------------------------


def test_seed_library_creates_adults_and_is_idempotent(tmp_path: Path):
    manager = _manager(tmp_path)
    created = seed_library(manager, category=CATEGORY_FAMILY)
    assert len(created) == len(library_by_category(CATEGORY_FAMILY))
    # Every created character is a known adult.
    for summary in manager.list_characters():
        age = summary.character.fictional_age
        assert age is not None and age >= 18
    # Seeding again creates nothing new.
    assert seed_library(manager, category=CATEGORY_FAMILY) == []
    assert len(manager.list_characters()) == len(created)


def test_seed_library_with_no_category_creates_every_character(tmp_path: Path):
    manager = _manager(tmp_path)
    seed_library(manager)
    assert len(manager.list_characters()) == len(LIBRARY_CHARACTERS)


def test_seed_library_honours_an_owner_override(tmp_path: Path):
    manager = _manager(tmp_path)
    created = seed_library(manager, owner="natasha", category=CATEGORY_LOVERS)
    assert created and all(c.character.owner == "natasha" for c in created)


# -- the service ----------------------------------------------------------


def test_the_service_browses_the_library_without_creating_anything():
    with _service() as service:
        rows = service.library_rows()
        assert len(rows) == len(LIBRARY_CHARACTERS)
        # Browsing is a pure read.
        assert service.character_rows() == []
        # Every row is honest about its archetype and adult age.
        assert all(row.age is not None and row.age >= 18 for row in rows)
        assert all(row.category_title for row in rows)


def test_the_service_filters_the_library_by_category_and_search():
    with _service() as service:
        family = service.library_rows(category=CATEGORY_FAMILY)
        assert family and all(row.category == CATEGORY_FAMILY for row in family)
        assert len(family) < len(service.library_rows())
        found = service.library_rows(search="ruby")
        assert found and found[0].name == "Ruby"


def test_adding_one_library_character_creates_exactly_one():
    with _service() as service:
        rows = service.library_rows(category=CATEGORY_LOVERS)
        name = rows[0].name
        bundle = service.add_library_character(name)
        assert len(service.character_rows()) == 1
        # Adding it again returns the same character, never a duplicate.
        again = service.add_library_character(name)
        assert again.id == bundle.id
        assert len(service.character_rows()) == 1


def test_adding_an_unknown_library_character_is_refused():
    with _service() as service:
        try:
            service.add_library_character("Nobody")
        except ValueError as error:
            assert "Nobody" in str(error)
        else:  # pragma: no cover - the guard must fire
            raise AssertionError("an unknown library name must be refused")


def test_seeding_a_category_through_the_service_is_idempotent():
    with _service() as service:
        created = service.seed_library_category(CATEGORY_INCLUSIVE)
        assert created == len(library_by_category(CATEGORY_INCLUSIVE))
        assert service.seed_library_category(CATEGORY_INCLUSIVE) == 0


def test_the_library_categories_are_exposed_to_a_surface():
    with _service() as service:
        rows = service.library_categories()
        assert len(rows) == len(LIBRARY_CATEGORIES)
        assert all(key and title and icon and summary for key, title, icon, summary in rows)


# -- the CLI --------------------------------------------------------------


def test_cli_lists_the_library_categories(capsys, tmp_path: Path):
    code = cli.main(["--data-root", str(tmp_path), "library-categories"])
    assert code == 0
    out = capsys.readouterr().out
    assert "family" in out and "courtesans" in out


def test_cli_lists_the_library_as_json(capsys, tmp_path: Path):
    code = cli.main(["--data-root", str(tmp_path), "library", "--category", "family", "--json"])
    assert code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload and all(row["category"] == "family" for row in payload)
    assert all(row["age"] >= 18 for row in payload)


def test_cli_adds_a_library_character(capsys, tmp_path: Path):
    code = cli.main(["--data-root", str(tmp_path), "add-library", "Ruby", "--json"])
    assert code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["name"] == "Ruby"
    # The character really exists in the store now.
    service = RomanceStudioService(root=tmp_path)
    assert any(row.name == "Ruby" for row in service.character_rows())


def test_cli_seeds_a_category(capsys, tmp_path: Path):
    code = cli.main(["--data-root", str(tmp_path), "seed-library", "--category", "escorts"])
    assert code == 0
    assert "Created" in capsys.readouterr().out
    service = RomanceStudioService(root=tmp_path)
    assert len(service.character_rows()) == len(library_by_category("escorts"))
