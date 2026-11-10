"""Tests for the Workplace Phase 3: the configurable hierarchy and taxonomy.

Phase 3 makes the institutional hierarchy *fully* configurable and gives each
division a technical profile: divisional detail items (the brief's §3), a stored
and editable taxonomy (the brief's §4), and the interface settings. Everything is
exercised on the real service and the real database -- no model, no network, no
display.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from cli.workplace import main as workplace_main
from core.workplace.database import WorkplaceDatabase
from core.workplace.model import DIVISION_DETAIL_KINDS
from core.workplace.schema import SCHEMA_VERSION
from core.workplace.settings import (
    WorkplaceSettings,
    load_settings,
    save_settings,
)
from gui.workplace import (
    DIVISION_DETAIL_LABELS,
    WorkplaceError,
    WorkplaceService,
    division_detail_kind_label,
)


@pytest.fixture()
def service(tmp_path: Path) -> WorkplaceService:
    return WorkplaceService(root=tmp_path)


# -- divisional profile (the brief's §3) ----------------------------------


def test_every_detail_kind_has_a_bilingual_label():
    assert set(DIVISION_DETAIL_LABELS) == set(DIVISION_DETAIL_KINDS)
    for kind, (english, arabic) in DIVISION_DETAIL_LABELS.items():
        assert english and arabic
        assert division_detail_kind_label(kind) == english
        assert division_detail_kind_label(kind, language="ar") == arabic
    assert division_detail_kind_label("unknown-kind") == "unknown-kind"


def test_a_division_starts_with_its_scope_as_a_profile_item(service: WorkplaceService):
    division = service.divisions()[0]
    profile = service.division_profile(division.record_id)
    assert profile.division.record_id == division.record_id
    assert not profile.empty
    assert profile.count("capability") >= 1


def test_add_and_remove_a_profile_item(service: WorkplaceService):
    division = service.divisions()[0]
    before = len(service.division_profile(division.record_id).details)
    item = service.add_division_detail(
        division_id=division.record_id,
        kind="standard",
        title="API 1104",
        detail="Welding of pipelines and related facilities.",
        reference="API 1104, 22nd edition",
    )
    assert item.kind == "standard"
    assert item.kind_label == "Applicable standards"
    assert "API 1104" in item.describe()
    profile = service.division_profile(division.record_id)
    assert len(profile.details) == before + 1
    assert profile.count("standard") == 1

    service.remove_division_detail(item.record_id)
    profile = service.division_profile(division.record_id)
    assert len(profile.details) == before  # a removed item is deactivated, not listed


def test_a_profile_item_needs_a_known_kind(service: WorkplaceService):
    division = service.divisions()[0]
    with pytest.raises(WorkplaceError):
        service.add_division_detail(division_id=division.record_id, kind="not-a-kind", title="x")


def test_a_profile_item_needs_a_title_or_a_detail(service: WorkplaceService):
    division = service.divisions()[0]
    with pytest.raises(WorkplaceError):
        service.add_division_detail(division_id=division.record_id, kind="standard")


def test_a_profile_item_needs_a_real_division(service: WorkplaceService):
    with pytest.raises(WorkplaceError):
        service.add_division_detail(division_id="DIV-nope", kind="standard", title="x")


def test_profile_groups_by_kind(service: WorkplaceService):
    division = service.divisions()[0]
    service.add_division_detail(division_id=division.record_id, kind="discipline", title="Welding")
    service.add_division_detail(division_id=division.record_id, kind="equipment", title="UT set")
    grouped = service.division_profile(division.record_id).by_kind()
    assert "discipline" in grouped and "equipment" in grouped and "capability" in grouped


# -- the stored taxonomy (the brief's §4) ---------------------------------


def test_the_taxonomy_is_stored_on_first_use(service: WorkplaceService):
    areas = service.taxonomy_tree()
    assert len(areas) == 8
    assert all(area.kind == "area" for area in areas)
    # Every area carries its disciplines in its metadata, ready for a surface.
    total = sum(len(area.metadata["disciplines"]) for area in areas)
    assert total >= 40


def test_add_and_edit_a_taxonomy_area(service: WorkplaceService):
    before = len(service.taxonomy_tree())
    area = service.add_taxonomy_area(name="Corrosion Science", name_ar="علم التآكل")
    assert area.key == "corrosion-science"
    assert not area.proposed  # a user's own area is not marked proposed
    assert len(service.taxonomy_tree()) == before + 1

    renamed = service.rename_taxonomy_node(area.record_id, name="Corrosion Engineering")
    assert renamed.name == "Corrosion Engineering"

    service.remove_taxonomy_node(area.record_id)
    assert len(service.taxonomy_tree()) == before  # a removed node is deactivated


def test_add_a_discipline_under_an_area(service: WorkplaceService):
    area = service.taxonomy_tree()[0]
    before = len(area.metadata["disciplines"])
    service.add_taxonomy_discipline(area_id=area.record_id, name="Flow Assurance")
    refreshed = next(a for a in service.taxonomy_tree() if a.record_id == area.record_id)
    assert "Flow Assurance" in refreshed.metadata["disciplines"]
    assert len(refreshed.metadata["disciplines"]) == before + 1


def test_a_discipline_needs_a_real_area(service: WorkplaceService):
    with pytest.raises(WorkplaceError):
        service.add_taxonomy_discipline(area_id="TAX-nope", name="Nope")


def test_a_taxonomy_area_needs_a_unique_key(service: WorkplaceService):
    service.add_taxonomy_area(name="Unique Area")
    with pytest.raises(WorkplaceError):
        service.add_taxonomy_area(name="Unique Area", key="unique-area")


# -- settings -------------------------------------------------------------


def test_settings_default_then_persist(tmp_path: Path):
    service = WorkplaceService(root=tmp_path)
    assert service.settings().language == "en"
    service.update_settings(language="ar")
    reopened = WorkplaceService(root=tmp_path)
    assert reopened.settings().language == "ar"
    assert reopened.settings().rtl is True  # Arabic implies RTL


def test_settings_survive_a_corrupt_store(tmp_path: Path):
    store = tmp_path / "workplace"
    store.mkdir(parents=True, exist_ok=True)
    (store / "settings.json").write_text("{ not json", encoding="utf-8")
    assert load_settings(tmp_path).language == "en"


def test_settings_round_trip_a_primitive():
    settings = WorkplaceSettings(language="ar", rtl=True, show_proposed=False)
    restored = WorkplaceSettings.from_primitive(settings.to_primitive())
    assert restored == settings
    assert WorkplaceSettings.from_primitive("nonsense") == WorkplaceSettings()


def test_save_settings_reports_a_read_only_folder(tmp_path: Path, monkeypatch):
    import core.workplace.settings as module

    class Broken:
        def __init__(self, *_a, **_k) -> None:
            pass

        def save(self, *_a, **_k) -> None:
            raise OSError("read-only")

    monkeypatch.setattr(module, "JsonStore", Broken)
    assert save_settings(WorkplaceSettings()) is False


# -- migration ------------------------------------------------------------


def test_a_version_1_database_migrates_to_version_2(tmp_path: Path):
    import sqlite3

    db_path = tmp_path / "old.db"
    connection = sqlite3.connect(db_path)
    connection.execute("CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT NOT NULL)")
    connection.execute(
        "CREATE TABLE institutions (record_id TEXT PRIMARY KEY, created_at TEXT, "
        "updated_at TEXT, active INTEGER, key TEXT, name TEXT, name_ar TEXT, kind TEXT, "
        "country TEXT, website TEXT, summary TEXT, official INTEGER, metadata TEXT)"
    )
    connection.execute("INSERT INTO meta VALUES('schema_version','1')")
    connection.execute(
        "INSERT INTO institutions VALUES('INST-1','t','t',1,'epri','EPRI','معهد',"
        "'institute','Egypt','','','1','{}')"
    )
    connection.commit()
    connection.close()

    database = WorkplaceDatabase(db_path)
    assert database.schema_version() == SCHEMA_VERSION == 2
    assert len(database.list_records("institutions")) == 1  # nothing was lost
    assert database.integrity_check()
    # The new tables exist and are empty.
    assert database.list_records("division_details") == []
    assert database.list_records("taxonomy_nodes") == []


def test_a_newer_database_is_refused(tmp_path: Path):
    import sqlite3

    from core.workplace.database import WorkplaceDatabaseError

    db_path = tmp_path / "future.db"
    connection = sqlite3.connect(db_path)
    connection.execute("CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT NOT NULL)")
    connection.execute("INSERT INTO meta VALUES('schema_version','99')")
    connection.commit()
    connection.close()
    with pytest.raises(WorkplaceDatabaseError):
        WorkplaceDatabase(db_path)


# -- CLI ------------------------------------------------------------------


def test_cli_profile_lists_items(tmp_path: Path, capsys):
    service = WorkplaceService(root=tmp_path)
    division = service.divisions()[0]
    service.add_division_detail(division_id=division.record_id, kind="standard", title="API 1104")
    assert workplace_main(["--data-root", str(tmp_path), "profile", division.record_id]) == 0
    out = capsys.readouterr().out
    assert "Applicable standards" in out
    assert "API 1104" in out


def test_cli_add_detail_and_kinds(tmp_path: Path, capsys):
    service = WorkplaceService(root=tmp_path)
    division = service.divisions()[0]
    code = workplace_main([
        "--data-root", str(tmp_path), "add-detail", division.record_id, "equipment",
        "--title", "Ultrasonic set",
    ])
    assert code == 0
    assert "Ultrasonic set" in capsys.readouterr().out
    assert workplace_main(["--data-root", str(tmp_path), "kinds", "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert any(row["kind"] == "equipment" for row in payload)


def test_cli_taxonomy_edit(tmp_path: Path, capsys):
    assert workplace_main([
        "--data-root", str(tmp_path), "add-taxonomy-area", "New Area", "--name-ar", "مجال",
    ]) == 0
    out = capsys.readouterr().out
    assert "New Area" in out
    assert workplace_main(["--data-root", str(tmp_path), "taxonomy", "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert any(area["name"] == "New Area" for area in payload)


def test_cli_settings_round_trip(tmp_path: Path, capsys):
    assert workplace_main(["--data-root", str(tmp_path), "settings", "--language", "ar", "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["language"] == "ar" and payload["rtl"] is True
    assert workplace_main(["--data-root", str(tmp_path), "settings", "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["language"] == "ar"


def test_cli_move_division(tmp_path: Path, capsys):
    service = WorkplaceService(root=tmp_path)
    view = service.hierarchy()
    unit_id = view.units[0].record_id
    division = service.divisions()[0]
    assert workplace_main([
        "--data-root", str(tmp_path), "move-division", division.record_id, "--unit", unit_id,
    ]) == 0
    assert "Moved" in capsys.readouterr().out
