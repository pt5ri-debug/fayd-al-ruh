"""The Workplace's seeded defaults, and the honesty rules that govern them.

Two things are seeded here, and the difference between them is the whole point of
this module:

* **The institution** -- the Egyptian Petroleum Research Institute (EPRI) -- is a
  real, publicly documented body, so its record is ``official=True``. Its
  *internal* structure is **not** fully public, so no official departmental tree
  is invented: the units seeded are marked ``official=False`` and carry a
  ``metadata`` note naming them as a working starting point to be completed from
  the institute's own approved structure.
* **The Technical Support & Technology Center (TSTC)** and its **ten divisions**
  are shipped as **proposed, configurable** defaults (``proposed=True``). They are
  a starting point an administrator edits, renames, reorganises, and deactivates
  without touching source code -- never a claim of an officially approved
  structure.

Every seeded record carries a ``metadata`` ``source`` note, so a reader can always
tell where a fact came from. :func:`seed_workplace` is idempotent: seeding twice
does not duplicate a record, and it never overwrites a row the user has edited.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from core.workplace.model import Division, Institution, OrganisationalUnit, make_id
from core.workplace.repository import WorkplaceRepository
from core.workplace.taxonomy import TAXONOMY

#: A stable key for the institution and the centre, so seeding is idempotent and
#: other modules can look them up without a magic id.
EPRI_KEY = "epri"
TSTC_KEY = "tstc"

#: The honest note attached to every proposed/internal record.
PROPOSED_NOTE = (
    "Proposed, configurable default -- not an officially approved structure. "
    "Complete it from the institute's own approved organisational documents."
)

#: The institution's publicly documented facts (no invented departments).
EPRI_INSTITUTION: dict[str, Any] = {
    "key": EPRI_KEY,
    "name": "Egyptian Petroleum Research Institute",
    "name_ar": "معهد بحوث البترول المصري",
    "kind": "institute",
    "country": "Egypt",
    "summary": (
        "A national research institute for petroleum and related sciences, "
        "affiliated with Egypt's Ministry of Higher Education and Scientific "
        "Research, conducting research, laboratory analysis, and technical "
        "services for the petroleum industry."
    ),
    "official": True,
    "metadata": {
        "source": "public documentation (institute website; Ministry of Higher Education and Scientific Research)",
        "note": (
            "Only publicly documented facts are recorded here. The institute's "
            "full internal structure must be completed from its own approved "
            "documents."
        ),
    },
}

#: The centre itself (a unit under the institution), shipped as a proposed entry.
TSTC_UNIT: dict[str, Any] = {
    "key": TSTC_KEY,
    "name": "Technical Support & Technology Center",
    "name_ar": "مركز الدعم الفني والتكنولوجي",
    "kind": "center",
    "summary": (
        "The institute's technical support and technology centre serving the "
        "petroleum industry (بترول - صناعة): engineering services, inspection and "
        "quality control, technical consultancy, and project technical support."
    ),
    "official": False,
    "order": 1,
    "metadata": {"source": "proposed default", "note": PROPOSED_NOTE},
}

#: The ten divisions the brief lists, as proposed, configurable defaults.
DIVISIONS_SEED: tuple[dict[str, Any], ...] = (
    {
        "key": "engineering-services",
        "name": "Engineering Services",
        "name_ar": "الخدمات الهندسية",
        "summary": "Multi-discipline engineering support across the centre's projects.",
        "scope": "Design review, technical studies, and engineering support for oil and gas facilities.",
    },
    {
        "key": "painting-coating",
        "name": "Painting and Coating",
        "name_ar": "الدهانات والطلاء",
        "summary": "Surface preparation, protective coating application, and coating inspection.",
        "scope": "Blasting, coating systems, field joint coating, and coating condition assessment.",
    },
    {
        "key": "welding-engineering",
        "name": "Welding Engineering",
        "name_ar": "هندسة اللحام",
        "summary": "Welding procedures, qualification, and weld quality.",
        "scope": "WPS/PQR/WPQ, welder qualification, welding inspection, and metallurgy.",
    },
    {
        "key": "electrical-engineering",
        "name": "Electrical Engineering",
        "name_ar": "الهندسة الكهربائية",
        "summary": "Electrical installations, instrumentation, and commissioning.",
        "scope": "Electrical inspection, instrumentation, testing, and commissioning support.",
    },
    {
        "key": "civil-engineering",
        "name": "Civil Engineering",
        "name_ar": "الهندسة المدنية",
        "summary": "Structures, concrete technology, and construction works.",
        "scope": "Structural works, concrete technology, construction inspection, and material testing.",
    },
    {
        "key": "concrete-weight-coating",
        "name": "Concrete Weight Coating (CWC)",
        "name_ar": "الطلاء الخرساني للوزن (CWC)",
        "summary": "Concrete weight coating for offshore and submarine pipelines.",
        "scope": "CWC application, density and thickness control, and inspection.",
    },
    {
        "key": "technical-surveillance",
        "name": "Technical Surveillance",
        "name_ar": "الإشراف الفني",
        "summary": "Independent technical surveillance of projects and suppliers.",
        "scope": "Witnessing, surveillance visits, and technical compliance monitoring.",
    },
    {
        "key": "inspection-quality-control",
        "name": "Inspection and Quality Control",
        "name_ar": "الفحص ومراقبة الجودة",
        "summary": "Inspection, NDT, testing, and quality control services.",
        "scope": "Inspection planning, NDT, testing and evaluation, QA/QC, and NCR/CAR handling.",
    },
    {
        "key": "technical-consultancy",
        "name": "Technical Consultancy",
        "name_ar": "الاستشارات الفنية",
        "summary": "Specialist technical advice and studies.",
        "scope": "Failure analysis, technical studies, specification review, and expert advice.",
    },
    {
        "key": "project-technical-support",
        "name": "Project Technical Support",
        "name_ar": "الدعم الفني للمشروعات",
        "summary": "Technical support embedded in client and project teams.",
        "scope": "Project technical support, documentation, and coordination.",
    },
)


@dataclass
class SeedReport:
    """What seeding did, so it is visible rather than assumed."""

    institutions: int = 0
    units: int = 0
    divisions: int = 0
    taxonomy_areas: int = 0
    taxonomy_disciplines: int = 0
    details: int = 0
    skipped: int = 0

    @property
    def total(self) -> int:
        return (
            self.institutions
            + self.units
            + self.divisions
            + self.taxonomy_areas
            + self.taxonomy_disciplines
            + self.details
        )

    def describe(self) -> str:
        if self.total == 0:
            return "Workplace defaults are already present; nothing was seeded."
        return (
            f"Seeded {self.institutions} institution(s), {self.units} unit(s), "
            f"{self.divisions} division(s), {self.taxonomy_areas} taxonomy area(s) "
            f"with {self.taxonomy_disciplines} discipline(s), and {self.details} "
            f"divisional profile item(s); {self.skipped} already present."
        )


def seed_workplace(repository: WorkplaceRepository, *, force: bool = False) -> SeedReport:
    """Install the Workplace defaults, idempotently.

    A record whose ``key`` already exists is left exactly as it is (so a user's
    edit is never overwritten) unless ``force`` is set, which is only for a
    deliberate reset. Returns a :class:`SeedReport` of what happened.
    """
    report = SeedReport()

    institution = repository.institution_by_key(EPRI_KEY)
    if institution is None or force:
        institution = repository.create_institution(**EPRI_INSTITUTION)
        report.institutions += 1
    else:
        report.skipped += 1

    unit = _unit_by_key(repository, institution.record_id, TSTC_KEY)
    if unit is None or force:
        unit = repository.create_unit(institution_id=institution.record_id, **TSTC_UNIT)
        report.units += 1
    else:
        report.skipped += 1

    existing = {division.key for division in repository.list_divisions(include_inactive=True)}
    division_ids: dict[str, str] = {
        division.key: division.record_id
        for division in repository.list_divisions(include_inactive=True)
    }
    for order, spec in enumerate(DIVISIONS_SEED, start=1):
        if spec["key"] in existing and not force:
            report.skipped += 1
            continue
        created = repository.create_division(
            unit_id=unit.record_id,
            proposed=True,
            order=order,
            metadata={"source": "proposed default", "note": PROPOSED_NOTE},
            **spec,
        )
        division_ids[created.key] = created.record_id
        report.divisions += 1

    _seed_taxonomy(repository, report, force=force)
    _seed_division_details(repository, report, division_ids, force=force)

    return report


def _seed_taxonomy(
    repository: WorkplaceRepository, report: SeedReport, *, force: bool
) -> None:
    """Store the shipped taxonomy as editable nodes (idempotent by key)."""
    existing_keys = {
        node.key for node in repository.list_taxonomy_nodes(include_inactive=True)
    }
    for order, area in enumerate(TAXONOMY, start=1):
        if area.key in existing_keys and not force:
            report.skipped += 1
            continue
        node = repository.create_taxonomy_node(
            kind="area",
            key=area.key,
            name=area.name,
            name_ar=area.name_ar,
            summary=area.summary,
            order=order,
            proposed=True,
            metadata={"source": "proposed default", "note": PROPOSED_NOTE},
        )
        report.taxonomy_areas += 1
        for discipline_order, discipline in enumerate(area.disciplines, start=1):
            repository.create_taxonomy_node(
                kind="discipline",
                parent_id=node.record_id,
                key=f"{area.key}:{_slugify(discipline)}",
                name=discipline,
                order=discipline_order,
                proposed=True,
                metadata={"source": "proposed default", "note": PROPOSED_NOTE},
            )
            report.taxonomy_disciplines += 1


def _seed_division_details(
    repository: WorkplaceRepository,
    report: SeedReport,
    division_ids: dict[str, str],
    *,
    force: bool,
) -> None:
    """Give each division a starting technical profile (the brief's §3).

    Every division gets its own scope recorded as a ``discipline``/``capability``
    seed only when it has no profile yet, so a user's own edits are never
    overwritten. The detail is deliberately thin: the point is that the *shape*
    exists and is editable, not that a full profile is invented.
    """
    for spec in DIVISIONS_SEED:
        division_id = division_ids.get(spec["key"])
        if not division_id:
            continue
        existing = repository.list_details(division_id=division_id, include_inactive=True)
        if existing and not force:
            report.skipped += 1
            continue
        scope = str(spec.get("scope", "") or "")
        if scope:
            repository.create_detail(
                division_id=division_id,
                kind="capability",
                title="Scope of services",
                detail=scope,
                order=1,
                metadata={"source": "proposed default", "note": PROPOSED_NOTE},
            )
            report.details += 1


def _unit_by_key(
    repository: WorkplaceRepository, institution_id: str, key: str
) -> OrganisationalUnit | None:
    for unit in repository.list_units(institution_id=institution_id, include_inactive=True):
        if unit.key == key:
            return unit
    return None


def _slugify(text: str) -> str:
    """A stable key fragment from a name (lower-case, hyphenated)."""
    cleaned = [character if character.isalnum() else " " for character in text.strip().casefold()]
    parts = "".join(cleaned).split()
    return "-".join(parts) or "item"


def default_records() -> tuple[list[Institution], list[OrganisationalUnit], list[Division]]:
    """The seeded records as plain objects (no database), for a preview or a test.

    Ids are generated here so the preview matches what seeding would write; the
    caller can show the defaults before committing them.
    """
    institution = Institution(record_id=make_id("institution"), **EPRI_INSTITUTION)
    unit = OrganisationalUnit(
        record_id=make_id("unit"), institution_id=institution.record_id, **TSTC_UNIT
    )
    divisions = [
        Division(
            record_id=make_id("division"),
            unit_id=unit.record_id,
            proposed=True,
            order=order,
            metadata={"source": "proposed default", "note": PROPOSED_NOTE},
            **spec,
        )
        for order, spec in enumerate(DIVISIONS_SEED, start=1)
    ]
    return [institution], [unit], divisions


__all__ = [
    "DIVISIONS_SEED",
    "EPRI_INSTITUTION",
    "EPRI_KEY",
    "PROPOSED_NOTE",
    "TSTC_KEY",
    "TSTC_UNIT",
    "SeedReport",
    "default_records",
    "seed_workplace",
]
