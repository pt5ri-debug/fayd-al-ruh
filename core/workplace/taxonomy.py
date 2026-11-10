"""The professional knowledge taxonomy of the Workplace.

This is a **declarative** catalogue, in the same spirit as the project's other
seeds (``core.domain_knowledge``, ``core.curriculum_knowledge``): a tuple of
:class:`TaxonomyArea`, each with its sub-disciplines, covering the domains the
brief lists. It is deliberately plain data with no I/O, so it can be shown,
searched, and extended without a database, and a later phase stores it in the
Workplace database.

The taxonomy is **expandable and configurable**: it is the *default* set, not a
closed list. An administrator adds an area or a sub-discipline without touching
source code once it is stored (later phase); here it is the shipped default.

English labels are primary (the project's convention); Arabic labels are kept as
secondary, as everywhere else in Fayd al-Ruh.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TaxonomyArea:
    """One top-level area of the professional taxonomy and its sub-disciplines."""

    key: str
    name: str
    name_ar: str
    summary: str
    disciplines: tuple[str, ...]

    @property
    def discipline_count(self) -> int:
        return len(self.disciplines)


#: The taxonomy the Workplace ships with (the brief's §4, as declarative data).
TAXONOMY: tuple[TaxonomyArea, ...] = (
    TaxonomyArea(
        key="petroleum",
        name="Petroleum Engineering",
        name_ar="هندسة البترول",
        summary="The exploration, drilling, production, and surface handling of oil and gas.",
        disciplines=(
            "Drilling Engineering",
            "Production Engineering",
            "Reservoir Engineering",
            "Petroleum Geology",
            "Well Integrity",
            "Oil and Gas Processing",
            "Pipelines and Flowlines",
        ),
    ),
    TaxonomyArea(
        key="mechanical",
        name="Mechanical Engineering",
        name_ar="الهندسة الميكانيكية",
        summary="Static and rotating equipment, pressure vessels, and piping.",
        disciplines=(
            "Static Equipment",
            "Rotating Equipment",
            "Pressure Vessels",
            "Piping Engineering",
            "Mechanical Inspection",
        ),
    ),
    TaxonomyArea(
        key="welding",
        name="Welding Engineering",
        name_ar="هندسة اللحام",
        summary="Welding processes, qualification, inspection, and metallurgy.",
        disciplines=(
            "Welding Processes",
            "Welding Procedures",
            "WPS / PQR / WPQ",
            "Welder Qualification",
            "Welding Inspection",
            "Metallurgy and Heat Treatment",
            "Weld Defects and Acceptance Criteria",
        ),
    ),
    TaxonomyArea(
        key="coating",
        name="Coating and Corrosion",
        name_ar="الطلاء والحماية من التآكل",
        summary="Surface preparation, protective coatings, and corrosion control.",
        disciplines=(
            "Surface Preparation",
            "Protective Coatings",
            "Field Joint Coating",
            "Internal and External Coatings",
            "Concrete Weight Coating",
            "Insulation and Protective Systems",
            "Cathodic Protection",
            "Corrosion Monitoring",
            "Coating Inspection and Testing",
        ),
    ),
    TaxonomyArea(
        key="civil",
        name="Civil Engineering",
        name_ar="الهندسة المدنية",
        summary="Structures, concrete technology, and construction works.",
        disciplines=(
            "Structural Engineering",
            "Concrete Technology",
            "Construction Inspection",
            "Material Testing",
            "Offshore and Onshore Civil Works",
        ),
    ),
    TaxonomyArea(
        key="electrical",
        name="Electrical Engineering",
        name_ar="الهندسة الكهربائية",
        summary="Electrical installations, instrumentation, and commissioning.",
        disciplines=(
            "Electrical Installations",
            "Electrical Inspection",
            "Instrumentation",
            "Testing and Commissioning",
        ),
    ),
    TaxonomyArea(
        key="scientific",
        name="Scientific Disciplines",
        name_ar="التخصصات العلمية",
        summary="The laboratory sciences behind the institute's research.",
        disciplines=(
            "Analytical Chemistry",
            "Industrial Chemistry",
            "Physical Chemistry",
            "Geochemistry",
            "Geology",
            "Materials Science",
            "Polymer Science",
            "Environmental Sciences",
        ),
    ),
    TaxonomyArea(
        key="quality",
        name="Quality and Inspection",
        name_ar="الجودة والفحص",
        summary="QA/QC, non-destructive testing, surveillance, and auditing.",
        disciplines=(
            "QA/QC",
            "NDT",
            "Technical Surveillance",
            "Inspection Planning",
            "Quality Auditing",
            "NCR / CAR",
            "Testing and Evaluation",
        ),
    ),
)


def area_for(key: str) -> TaxonomyArea | None:
    """Look up one area by its stable key, or ``None``."""
    for area in TAXONOMY:
        if area.key == key:
            return area
    return None


def taxonomy_keys() -> tuple[str, ...]:
    """Every area key, in order."""
    return tuple(area.key for area in TAXONOMY)


def discipline_count() -> int:
    """How many sub-disciplines the whole taxonomy carries."""
    return sum(area.discipline_count for area in TAXONOMY)


def find_discipline(name: str) -> TaxonomyArea | None:
    """The area that owns a sub-discipline (case-insensitive), or ``None``."""
    wanted = name.strip().casefold()
    for area in TAXONOMY:
        if any(discipline.casefold() == wanted for discipline in area.disciplines):
            return area
    return None


__all__ = [
    "TAXONOMY",
    "TaxonomyArea",
    "area_for",
    "discipline_count",
    "find_discipline",
    "taxonomy_keys",
]
