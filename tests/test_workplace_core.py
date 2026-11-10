"""Tests for the Workplace foundation: model, database, repository, seed, taxonomy.

These exercise the real code with a real SQLite file in a temporary folder, so
the migrations, the foreign keys, the soft-delete rule, the backup/restore, and
the idempotent seeding are all proven rather than assumed. No display, no model,
no network.
"""

from __future__ import annotations

import sqlite3

import pytest
from core.workplace.database import WorkplaceDatabase, WorkplaceDatabaseError
from core.workplace.model import (
    Division,
    Institution,
    OrganisationalUnit,
    make_id,
    utc_now,
)
from core.workplace.repository import DIVISIONS, WorkplaceRepository
from core.workplace.schema import SCHEMA_VERSION
from core.workplace.seed import (
    DIVISIONS_SEED,
    EPRI_KEY,
    TSTC_KEY,
    default_records,
    seed_workplace,
)
from core.workplace.taxonomy import (
    TAXONOMY,
    area_for,
    discipline_count,
    find_discipline,
    taxonomy_keys,
)


@pytest.fixture()
def repo(tmp_path) -> WorkplaceRepository:
    db = WorkplaceDatabase(tmp_path / "workplace.db")
    return WorkplaceRepository(db)


# -- model ----------------------------------------------------------------


def test_make_id_is_prefixed_and_stable_format():
    inst = make_id("institution")
    assert inst.startswith("INST-")
    assert make_id("division").startswith("DIV-")
    assert make_id("unit").startswith("UNIT-")
    # An unknown kind still mints a readable id rather than raising.
    assert make_id("widget").startswith("WIDGET-")


def test_records_are_frozen_and_touch_is_a_copy():
    record = Institution(record_id="INST-1", name="EPRI")
    with pytest.raises(Exception):
        record.name = "changed"  # type: ignore[misc]
    updated = record.touch(name="EPRI (new)")
    assert updated.name == "EPRI (new)"
    assert record.name == "EPRI"  # the original is untouched
    assert updated.created_at == record.created_at


def test_deactivate_and_activate_are_copies():
    record = Division(record_id="DIV-1", name="Welding")
    inactive = record.deactivate()
    assert inactive.active is False and record.active is True
    assert inactive.activate().active is True


def test_utc_now_is_iso():
    stamp = utc_now()
    assert "T" in stamp


# -- database & migrations ------------------------------------------------


def test_schema_version_is_recorded(tmp_path):
    db = WorkplaceDatabase(tmp_path / "wp.db")
    assert db.schema_version() == SCHEMA_VERSION
    assert db.integrity_check() is True


def test_migrate_is_idempotent(tmp_path):
    db = WorkplaceDatabase(tmp_path / "wp.db")
    assert db.migrate() == SCHEMA_VERSION
    assert db.migrate() == SCHEMA_VERSION


def test_newer_schema_is_refused(tmp_path):
    path = tmp_path / "wp.db"
    WorkplaceDatabase(path)  # create the file at the current schema first
    # Simulate a database written by a newer build.
    connection = sqlite3.connect(path)
    connection.execute("UPDATE meta SET value=? WHERE key='schema_version'", (str(SCHEMA_VERSION + 5),))
    connection.commit()
    connection.close()
    with pytest.raises(WorkplaceDatabaseError):
        WorkplaceDatabase(path)


def test_foreign_keys_are_on_and_restrict(tmp_path):
    db = WorkplaceDatabase(tmp_path / "wp.db")
    repo = WorkplaceRepository(db)
    # A unit that references a non-existent institution must be refused.
    with pytest.raises(sqlite3.IntegrityError):
        repo.create_unit(institution_id="INST-missing", name="Orphan")


def test_backup_and_restore(tmp_path):
    db = WorkplaceDatabase(tmp_path / "wp.db")
    repo = WorkplaceRepository(db)
    repo.create_institution(key="epri", name="EPRI")
    backup = db.backup(tmp_path / "copy.db")
    assert backup.ok
    # Change the live database, then restore the backup.
    repo.create_institution(key="other", name="Other")
    assert len(repo.list_institutions()) == 2
    report = db.restore(tmp_path / "copy.db")
    assert report.ok
    assert len(repo.list_institutions()) == 1
    assert db.integrity_check() is True


def test_restore_refuses_a_missing_backup(tmp_path):
    db = WorkplaceDatabase(tmp_path / "wp.db")
    report = db.restore(tmp_path / "nope.db")
    assert report.ok is False


def test_restore_refuses_a_corrupt_backup(tmp_path):
    db = WorkplaceDatabase(tmp_path / "wp.db")
    bad = tmp_path / "bad.db"
    bad.write_bytes(b"not a database at all")
    report = db.restore(bad)
    assert report.ok is False


# -- repository -----------------------------------------------------------


def test_create_and_read_institution(repo):
    inst = repo.create_institution(key="epri", name="EPRI", name_ar="معهد بحوث البترول")
    assert repo.get_institution(inst.record_id) == inst
    assert repo.institution_by_key("epri") == inst
    assert repo.institution_by_key("nope") is None
    assert [i.key for i in repo.list_institutions()] == ["epri"]


def test_units_form_a_tree(repo):
    inst = repo.create_institution(key="epri", name="EPRI")
    root = repo.create_unit(institution_id=inst.record_id, key="tstc", name="TSTC")
    child = repo.create_unit(
        institution_id=inst.record_id, parent_id=root.record_id, name="Lab"
    )
    assert repo.children_of(root.record_id) == [child]
    assert repo.list_units(institution_id=inst.record_id) == [root, child]


def test_top_level_unit_has_empty_parent(repo):
    inst = repo.create_institution(key="epri", name="EPRI")
    unit = repo.create_unit(institution_id=inst.record_id, name="Top")
    assert unit.parent_id == ""
    assert repo.get_unit(unit.record_id).parent_id == ""


def test_division_soft_delete_preserves_the_row(repo):
    inst = repo.create_institution(key="epri", name="EPRI")
    unit = repo.create_unit(institution_id=inst.record_id, name="TSTC")
    div = repo.create_division(unit_id=unit.record_id, key="welding", name="Welding")
    assert repo.list_divisions() == [div]
    repo.deactivate_division(div.record_id)
    assert repo.list_divisions() == []
    kept = repo.list_divisions(include_inactive=True)
    assert len(kept) == 1 and kept[0].active is False
    # The row still exists, so history that references it is intact.
    assert repo.get_division(div.record_id) is not None


def test_division_can_be_moved_between_units(repo):
    inst = repo.create_institution(key="epri", name="EPRI")
    a = repo.create_unit(institution_id=inst.record_id, name="A")
    b = repo.create_unit(institution_id=inst.record_id, name="B")
    div = repo.create_division(unit_id=a.record_id, name="D")
    moved = repo.save_division(div.touch(unit_id=b.record_id))
    assert moved.unit_id == b.record_id


def test_rename_and_reorder_division(repo):
    inst = repo.create_institution(key="epri", name="EPRI")
    unit = repo.create_unit(institution_id=inst.record_id, name="TSTC")
    div = repo.create_division(unit_id=unit.record_id, name="Welding")
    renamed = repo.rename_division(div.record_id, name="Welding & NDT", name_ar="اللحام")
    assert renamed.name == "Welding & NDT" and renamed.name_ar == "اللحام"
    reordered = repo.reorder_division(div.record_id, order=7)
    assert reordered.order == 7


def test_unknown_records_return_none(repo):
    assert repo.get_institution("NOPE") is None
    assert repo.get_unit("NOPE") is None
    assert repo.get_division("NOPE") is None
    assert repo.deactivate_division("NOPE") is None
    assert repo.rename_division("NOPE", name="x") is None


def test_bulk_insert(repo):
    inst = repo.create_institution(key="epri", name="EPRI")
    unit = repo.create_unit(institution_id=inst.record_id, name="TSTC")
    records = [
        Division(record_id=make_id("division"), unit_id=unit.record_id, name=f"D{i}")
        for i in range(3)
    ]
    assert repo.bulk_insert(DIVISIONS, records) == 3
    assert len(repo.list_divisions()) == 3


def test_corrupt_metadata_row_is_skipped_not_fatal(repo):
    inst = repo.create_institution(key="epri", name="EPRI")
    unit = repo.create_unit(institution_id=inst.record_id, name="TSTC")
    div = repo.create_division(unit_id=unit.record_id, name="D")
    # Write invalid JSON into the metadata column behind the repository's back.
    connection = sqlite3.connect(repo.db.path)
    connection.execute(
        "UPDATE divisions SET metadata='{not json' WHERE record_id=?", (div.record_id,)
    )
    connection.commit()
    connection.close()
    report = repo.load_report()
    rows = repo.db.list_records(DIVISIONS, report=report)
    assert len(rows) == 1  # the row is still returned
    assert report.skipped == 1  # and the bad metadata is counted


# -- seed -----------------------------------------------------------------


def test_seed_installs_the_defaults(repo):
    report = seed_workplace(repo)
    assert report.institutions == 1
    assert report.units == 1
    assert report.divisions == len(DIVISIONS_SEED) == 10
    institution = repo.institution_by_key(EPRI_KEY)
    assert institution is not None and institution.official is True
    divisions = repo.list_divisions()
    assert len(divisions) == 10
    assert all(d.proposed for d in divisions)


def test_seed_is_idempotent(repo):
    seed_workplace(repo)
    again = seed_workplace(repo)
    assert again.total == 0
    assert again.skipped > 0
    assert len(repo.list_divisions()) == 10


def test_seed_does_not_overwrite_a_user_edit(repo):
    seed_workplace(repo)
    division = repo.division_by_key("welding-engineering")
    repo.rename_division(division.record_id, name="Renamed by the user")
    seed_workplace(repo)
    assert repo.division_by_key("welding-engineering").name == "Renamed by the user"


def test_seed_force_reinstalls(repo):
    seed_workplace(repo)
    division = repo.division_by_key("welding-engineering")
    repo.rename_division(division.record_id, name="Renamed")
    seed_workplace(repo, force=True)
    # force adds a *new* row for the key rather than editing the existing one.
    assert len(repo.list_divisions()) > 10


def test_default_records_preview_matches_the_seed():
    institutions, units, divisions = default_records()
    assert len(institutions) == 1
    assert units[0].key == TSTC_KEY
    assert len(divisions) == 10


# -- taxonomy -------------------------------------------------------------


def test_taxonomy_covers_the_briefs_areas():
    keys = taxonomy_keys()
    for expected in ("petroleum", "mechanical", "welding", "coating", "civil",
                     "electrical", "scientific", "quality"):
        assert expected in keys
    assert discipline_count() >= 40


def test_taxonomy_lookup_helpers():
    assert area_for("welding").name == "Welding Engineering"
    assert area_for("nope") is None
    assert find_discipline("WPS / PQR / WPQ").key == "welding"
    assert find_discipline("no such thing") is None


def test_every_taxonomy_area_has_bilingual_labels():
    for area in TAXONOMY:
        assert area.name and area.name_ar
        assert area.disciplines


def test_organisational_unit_defaults_are_safe():
    unit = OrganisationalUnit(record_id="UNIT-1")
    assert unit.parent_id == "" and unit.institution_id == ""
