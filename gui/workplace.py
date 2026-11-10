"""The Workplace service: the Tk-free half of the Workplace window.

This is where the Workplace's work actually happens -- resolve the database,
seed the defaults, read and edit the institutional hierarchy, report a status --
with no display, so the whole workflow is testable on a fresh clone. The window
and the CLI both drive *this* service, so they can never disagree about what the
Workplace holds.

It is deliberately a thin, honest layer over the domain package:

* the relational store is :class:`core.workplace.database.WorkplaceDatabase`;
* typed access is :class:`core.workplace.repository.WorkplaceRepository`;
* the defaults are :func:`core.workplace.seed.seed_workplace`;
* the taxonomy is :mod:`core.workplace.taxonomy`.

Everything is local: nothing here opens a network connection or loads a model. A
missing database or a refused edit is reported as a :class:`WorkplaceError`
carrying the real reason, never a silent failure. A read-only data folder is
handled honestly: the service still reports status, and an edit that cannot be
written says so.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from core.paths import workplace_db_path
from core.workplace.database import WorkplaceDatabase
from core.workplace.model import Division, Institution, OrganisationalUnit
from core.workplace.repository import WorkplaceRepository
from core.workplace.seed import SeedReport, seed_workplace
from core.workplace.taxonomy import TAXONOMY, TaxonomyArea


class WorkplaceError(RuntimeError):
    """A Workplace operation could not be completed; the message says why."""


@dataclass
class DivisionSummary:
    """One division as a surface shows it: the readable fields plus its place."""

    record_id: str
    key: str
    name: str
    name_ar: str
    summary: str
    scope: str
    proposed: bool
    order: int
    active: bool
    unit_id: str
    unit_name: str = ""

    def label(self, *, language: str = "en") -> str:
        if language == "ar" and self.name_ar:
            return self.name_ar
        return self.name

    def describe(self, *, language: str = "en") -> str:
        parts = [self.label(language=language)]
        if self.proposed:
            parts.append("(proposed)" if language == "en" else "(مقترح)")
        if not self.active:
            parts.append("(inactive)" if language == "en" else "(غير نشط)")
        if self.summary:
            parts.append(f"— {self.summary}")
        return " ".join(parts)


@dataclass
class WorkplaceStatus:
    """An honest snapshot of the Workplace for the window's dashboard header."""

    db_path: str
    schema_version: int
    institutions: int
    units: int
    divisions: int
    active_divisions: int
    taxonomy_areas: int
    taxonomy_disciplines: int
    ready: bool
    message: str = ""

    def describe(self, *, language: str = "en") -> str:
        if language == "ar":
            return (
                f"قاعدة بيانات المكتب: {self.db_path} (الإصدار {self.schema_version}) — "
                f"{self.institutions} مؤسسة، {self.units} وحدة، {self.divisions} قسم "
                f"({self.active_divisions} نشط)."
            )
        return (
            f"Workplace database: {self.db_path} (schema {self.schema_version}) — "
            f"{self.institutions} institution(s), {self.units} unit(s), "
            f"{self.divisions} division(s) ({self.active_divisions} active)."
        )


@dataclass
class HierarchyView:
    """The institution tree as the sidebar draws it: units and their divisions."""

    institution: Institution | None
    units: list[OrganisationalUnit] = field(default_factory=list)
    divisions_by_unit: dict[str, list[DivisionSummary]] = field(default_factory=dict)
    unassigned: list[DivisionSummary] = field(default_factory=list)

    @property
    def empty(self) -> bool:
        return self.institution is None and not self.units and not self.unassigned


class WorkplaceService:
    """The one engine the Workplace window and the CLI both drive."""

    def __init__(
        self,
        *,
        root: Path | None = None,
        db_path: Path | None = None,
        auto_seed: bool = True,
    ) -> None:
        # ``root`` is the data root the caller uses (``data_dir()`` by default);
        # ``db_path`` overrides the file itself, which is what a test injects.
        self.root = root
        self._db_path = Path(db_path) if db_path is not None else (
            Path(root) / "workplace" / "workplace.db" if root is not None else workplace_db_path()
        )
        self._db: WorkplaceDatabase | None = None
        self._repo: WorkplaceRepository | None = None
        self._seed_report: SeedReport | None = None
        self._auto_seed = auto_seed

    # -- lifecycle --------------------------------------------------------

    @property
    def db_path(self) -> Path:
        return self._db_path

    def open(self) -> WorkplaceDatabase:
        """Open (and migrate) the database, seeding the defaults on first use."""
        if self._db is not None:
            return self._db
        try:
            self._db = WorkplaceDatabase(self._db_path)
        except Exception as error:  # a broken file is reported, never swallowed
            raise WorkplaceError(f"Could not open the Workplace database: {error}") from error
        self._repo = WorkplaceRepository(self._db)
        if self._auto_seed:
            self._seed_report = seed_workplace(self._repo)
        return self._db

    @property
    def repository(self) -> WorkplaceRepository:
        if self._repo is None:
            self.open()
        assert self._repo is not None
        return self._repo

    def seed(self, *, force: bool = False) -> SeedReport:
        """Install (or reset) the Workplace defaults."""
        report = seed_workplace(self.repository, force=force)
        self._seed_report = report
        return report

    # -- status -----------------------------------------------------------

    def status(self, *, language: str = "en") -> WorkplaceStatus:
        """An honest snapshot: real counts, the schema, and whether it is ready."""
        try:
            self.open()
        except WorkplaceError as error:
            return WorkplaceStatus(
                db_path=str(self._db_path), schema_version=0, institutions=0, units=0,
                divisions=0, active_divisions=0, taxonomy_areas=len(TAXONOMY),
                taxonomy_disciplines=sum(a.discipline_count for a in TAXONOMY),
                ready=False, message=str(error),
            )
        repo = self.repository
        divisions = repo.list_divisions(include_inactive=True)
        return WorkplaceStatus(
            db_path=str(self._db_path),
            schema_version=self.repository.db.schema_version(),
            institutions=len(repo.list_institutions(include_inactive=True)),
            units=len(repo.list_units(include_inactive=True)),
            divisions=len(divisions),
            active_divisions=sum(1 for d in divisions if d.active),
            taxonomy_areas=len(TAXONOMY),
            taxonomy_disciplines=sum(a.discipline_count for a in TAXONOMY),
            ready=True,
            message="",
        )

    # -- institution hierarchy -------------------------------------------

    def hierarchy(self, *, include_inactive: bool = False) -> HierarchyView:
        """The institution tree: units and the divisions that belong to each."""
        self.open()
        repo = self.repository
        institutions = repo.list_institutions(include_inactive=include_inactive)
        institution = institutions[0] if institutions else None
        units = (
            repo.list_units(institution_id=institution.record_id, include_inactive=include_inactive)
            if institution is not None
            else []
        )
        by_unit: dict[str, list[DivisionSummary]] = {unit.record_id: [] for unit in units}
        unassigned: list[DivisionSummary] = []
        for division in repo.list_divisions(include_inactive=include_inactive):
            summary = self._summarise(division, units)
            if division.unit_id in by_unit:
                by_unit[division.unit_id].append(summary)
            else:
                unassigned.append(summary)
        for items in by_unit.values():
            items.sort(key=lambda d: (d.order, d.name))
        unassigned.sort(key=lambda d: (d.order, d.name))
        return HierarchyView(
            institution=institution, units=units, divisions_by_unit=by_unit, unassigned=unassigned
        )

    def divisions(self, *, include_inactive: bool = False) -> list[DivisionSummary]:
        """Every division, in display order, with its unit's name resolved."""
        self.open()
        repo = self.repository
        units = repo.list_units(include_inactive=True)
        summaries = [
            self._summarise(division, units)
            for division in repo.list_divisions(include_inactive=include_inactive)
        ]
        summaries.sort(key=lambda d: (d.order, d.name))
        return summaries

    def division(self, record_id: str) -> DivisionSummary | None:
        """One division by id, or ``None``."""
        self.open()
        division = self.repository.get_division(record_id)
        if division is None:
            return None
        return self._summarise(division, self.repository.list_units(include_inactive=True))

    # -- division editing (no source change) ------------------------------

    def add_division(
        self,
        *,
        name: str,
        name_ar: str = "",
        key: str = "",
        unit_id: str = "",
        summary: str = "",
        scope: str = "",
        proposed: bool = True,
    ) -> DivisionSummary:
        """Add a division. ``name`` is required; the rest are optional."""
        if not name.strip():
            raise WorkplaceError("A division needs a name.")
        self.open()
        repo = self.repository
        if not unit_id:
            institutions = repo.list_institutions()
            units = repo.list_units(institution_id=institutions[0].record_id) if institutions else []
            unit_id = units[0].record_id if units else ""
        if key and repo.division_by_key(key) is not None:
            raise WorkplaceError(f"A division with key {key!r} already exists.")
        order = len(repo.list_divisions(include_inactive=True)) + 1
        record = repo.create_division(
            unit_id=unit_id,
            key=key or _slugify(name),
            name=name.strip(),
            name_ar=name_ar.strip(),
            summary=summary.strip(),
            scope=scope.strip(),
            proposed=proposed,
            order=order,
            metadata={"source": "user", "note": "Added by the user in the Workplace."},
        )
        return self._summarise(record, repo.list_units(include_inactive=True))

    def rename_division(self, record_id: str, *, name: str, name_ar: str = "") -> DivisionSummary:
        """Rename a division without touching source code."""
        if not name.strip():
            raise WorkplaceError("A division needs a name.")
        self.open()
        updated = self.repository.rename_division(record_id, name=name.strip(), name_ar=name_ar)
        if updated is None:
            raise WorkplaceError(f"No division with id {record_id!r}.")
        return self._summarise(updated, self.repository.list_units(include_inactive=True))

    def set_division_unit(self, record_id: str, *, unit_id: str) -> DivisionSummary:
        """Move a division to another unit (a reorganisation, never a delete)."""
        self.open()
        division = self.repository.get_division(record_id)
        if division is None:
            raise WorkplaceError(f"No division with id {record_id!r}.")
        updated = self.repository.save_division(division.touch(unit_id=unit_id))
        return self._summarise(updated, self.repository.list_units(include_inactive=True))

    def set_division_order(self, record_id: str, *, order: int) -> DivisionSummary:
        """Reorder a division within its unit."""
        self.open()
        updated = self.repository.reorder_division(record_id, order=order)
        if updated is None:
            raise WorkplaceError(f"No division with id {record_id!r}.")
        return self._summarise(updated, self.repository.list_units(include_inactive=True))

    def deactivate_division(self, record_id: str) -> DivisionSummary:
        """Deactivate a division (the soft delete); historical records survive."""
        self.open()
        updated = self.repository.deactivate_division(record_id)
        if updated is None:
            raise WorkplaceError(f"No division with id {record_id!r}.")
        return self._summarise(updated, self.repository.list_units(include_inactive=True))

    def activate_division(self, record_id: str) -> DivisionSummary:
        """Reactivate a deactivated division."""
        self.open()
        division = self.repository.get_division(record_id)
        if division is None:
            raise WorkplaceError(f"No division with id {record_id!r}.")
        updated = self.repository.save_division(division.activate())
        return self._summarise(updated, self.repository.list_units(include_inactive=True))

    # -- taxonomy ---------------------------------------------------------

    def taxonomy(self) -> tuple[TaxonomyArea, ...]:
        """The professional knowledge taxonomy (declarative; no database needed)."""
        return TAXONOMY

    def taxonomy_summary(self) -> str:
        areas = len(TAXONOMY)
        disciplines = sum(area.discipline_count for area in TAXONOMY)
        return f"{areas} areas, {disciplines} disciplines"

    # -- backup -----------------------------------------------------------

    def backup(self, destination: Path | None = None) -> str:
        """Write a consistent copy of the database and return its path."""
        self.open()
        from core.paths import workplace_backups_dir

        if destination is None:
            from core.workplace.model import utc_now

            stamp = utc_now().replace(":", "").replace("-", "")[:15]
            destination = workplace_backups_dir() / f"workplace-{stamp}.db"
        report = self.repository.db.backup(destination)
        if not report.ok:
            raise WorkplaceError(report.message or "backup failed")
        return report.path

    # -- helpers ----------------------------------------------------------

    @staticmethod
    def _summarise(
        division: Division, units: list[OrganisationalUnit]
    ) -> DivisionSummary:
        unit_name = next((u.name for u in units if u.record_id == division.unit_id), "")
        return DivisionSummary(
            record_id=division.record_id,
            key=division.key,
            name=division.name,
            name_ar=division.name_ar,
            summary=division.summary,
            scope=division.scope,
            proposed=division.proposed,
            order=division.order,
            active=division.active,
            unit_id=division.unit_id,
            unit_name=unit_name,
        )


def _slugify(text: str) -> str:
    """A stable key from a division name (lower-case, hyphenated)."""
    cleaned = [character if character.isalnum() else " " for character in text.strip().casefold()]
    parts = "".join(cleaned).split()
    return "-".join(parts) or "division"


__all__ = [
    "DivisionSummary",
    "HierarchyView",
    "WorkplaceError",
    "WorkplaceService",
    "WorkplaceStatus",
]
