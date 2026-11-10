"""Tests for the Workplace service and its CLI twin.

The service is exercised with a real database in a temporary folder, so the
seeding, the division editing, the hierarchy, the status, and the backup are all
proven. The CLI is driven through its ``main`` with ``--data-root``, so the CLI
and the service share one database and the exit codes are pinned.
"""

from __future__ import annotations

import json

import pytest
from cli.workplace import main as workplace_main
from gui.workplace import WorkplaceError, WorkplaceService


@pytest.fixture()
def service(tmp_path) -> WorkplaceService:
    return WorkplaceService(root=tmp_path)


# -- service --------------------------------------------------------------


def test_status_reports_real_counts(service):
    status = service.status()
    assert status.ready is True
    assert status.institutions == 1
    assert status.units == 1
    assert status.divisions == 10
    assert status.active_divisions == 10
    assert status.taxonomy_areas == 8
    assert "10 division" in status.describe()


def test_status_describes_in_arabic(service):
    described = service.status().describe(language="ar")
    assert "قاعدة بيانات" in described


def test_hierarchy_groups_divisions_under_their_unit(service):
    view = service.hierarchy()
    assert view.institution is not None
    assert len(view.units) == 1
    unit = view.units[0]
    assert len(view.divisions_by_unit[unit.record_id]) == 10
    assert view.unassigned == []


def test_divisions_are_ordered_and_labelled(service):
    rows = service.divisions()
    assert len(rows) == 10
    assert rows[0].order <= rows[-1].order
    assert rows[0].unit_name  # the unit's name is resolved


def test_add_rename_and_deactivate_division(service):
    created = service.add_division(name="Metallurgy Lab", name_ar="مختبر المعادن")
    assert created.key == "metallurgy-lab"
    assert created.name_ar == "مختبر المعادن"
    renamed = service.rename_division(created.record_id, name="Metallurgy & Corrosion")
    assert renamed.name == "Metallurgy & Corrosion"
    deactivated = service.deactivate_division(created.record_id)
    assert deactivated.active is False
    assert len(service.divisions()) == 10
    assert len(service.divisions(include_inactive=True)) == 11
    reactivated = service.activate_division(created.record_id)
    assert reactivated.active is True


def test_add_division_needs_a_name(service):
    with pytest.raises(WorkplaceError):
        service.add_division(name="   ")


def test_add_division_refuses_a_duplicate_key(service):
    with pytest.raises(WorkplaceError):
        service.add_division(name="Welding Again", key="welding-engineering")


def test_unknown_division_operations_raise(service):
    with pytest.raises(WorkplaceError):
        service.rename_division("DIV-nope", name="x")
    with pytest.raises(WorkplaceError):
        service.deactivate_division("DIV-nope")
    assert service.division("DIV-nope") is None


def test_set_division_order_and_unit(service):
    division = service.divisions()[3]
    reordered = service.set_division_order(division.record_id, order=1)
    assert reordered.order == 1
    unit = service.hierarchy().units[0]
    moved = service.set_division_unit(division.record_id, unit_id=unit.record_id)
    assert moved.unit_id == unit.record_id


def test_taxonomy_is_exposed(service):
    assert len(service.taxonomy()) == 8
    assert "areas" in service.taxonomy_summary()


def test_backup_writes_a_file(service, tmp_path):
    path = service.backup(tmp_path / "backup.db")
    assert (tmp_path / "backup.db").exists()
    assert path.endswith("backup.db")


def test_seed_is_idempotent_through_the_service(service):
    report = service.seed()
    assert report.total == 0
    assert len(service.divisions()) == 10


def test_service_opens_a_broken_database_honestly(tmp_path):
    path = tmp_path / "workplace.db"
    path.write_bytes(b"definitely not a database")
    service = WorkplaceService(db_path=path, auto_seed=False)
    status = service.status()
    assert status.ready is False
    assert status.message  # it names the real reason


# -- CLI ------------------------------------------------------------------


def test_cli_status(tmp_path, capsys):
    assert workplace_main(["--data-root", str(tmp_path), "status"]) == 0
    out = capsys.readouterr().out
    assert "10 division" in out


def test_cli_status_json(tmp_path, capsys):
    assert workplace_main(["--data-root", str(tmp_path), "status", "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["ready"] is True
    assert payload["divisions"] == 10


def test_cli_divisions_and_hierarchy(tmp_path, capsys):
    assert workplace_main(["--data-root", str(tmp_path), "divisions"]) == 0
    assert "Engineering Services" in capsys.readouterr().out
    assert workplace_main(["--data-root", str(tmp_path), "hierarchy"]) == 0
    out = capsys.readouterr().out
    assert "Egyptian Petroleum Research Institute" in out
    assert "Welding Engineering" in out


def test_cli_taxonomy(tmp_path, capsys):
    assert workplace_main(["--data-root", str(tmp_path), "taxonomy"]) == 0
    out = capsys.readouterr().out
    assert "Welding Engineering" in out
    assert "WPS / PQR / WPQ" in out


def test_cli_add_rename_deactivate(tmp_path, capsys):
    assert workplace_main(["--data-root", str(tmp_path), "add-division", "Metallurgy Lab"]) == 0
    out = capsys.readouterr().out
    assert "Metallurgy Lab" in out
    record_id = out.split()[1].rstrip(":")
    assert workplace_main(
        ["--data-root", str(tmp_path), "rename-division", record_id, "Metallurgy & NDT"]
    ) == 0
    assert "Metallurgy & NDT" in capsys.readouterr().out
    assert workplace_main(
        ["--data-root", str(tmp_path), "deactivate-division", record_id]
    ) == 0
    assert "Deactivated" in capsys.readouterr().out


def test_cli_refuses_an_unknown_division(tmp_path, capsys):
    code = workplace_main(["--data-root", str(tmp_path), "deactivate-division", "DIV-nope"])
    assert code == 4
    assert "refused" in capsys.readouterr().out.lower()


def test_cli_seed_reset(tmp_path, capsys):
    assert workplace_main(["--data-root", str(tmp_path), "seed", "--reset", "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["divisions"] >= 10


def test_cli_backup(tmp_path, capsys):
    target = tmp_path / "out.db"
    assert workplace_main(
        ["--data-root", str(tmp_path), "backup", "--output", str(target)]
    ) == 0
    assert target.exists()
    assert "Backed up" in capsys.readouterr().out
