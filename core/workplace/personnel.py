"""Personnel and expertise management for the Workplace (the brief's §8).

The Workplace holds one :class:`~core.workplace.records.Person` row per member of
staff and one :class:`~core.workplace.records.Qualification` row per certification,
plus an :class:`~core.workplace.records.Assignment` per project role. This module
assembles them into the three things a technical office actually asks for:

* a **personnel directory** -- who works here, their qualifications, and their
  project history, grouped by division and discipline;
* a **competency matrix** -- which people hold which qualifications, as a readable
  grid, so a coverage gap is visible rather than assumed; and
* **project-specific selection** -- given the skills and certifications a task
  requires, rank the people who can do it and name, for each, what they hold and
  what they are missing.

The selection is deliberately **explainable and honest**: it matches on the words a
person's own record carries (role, discipline, and the titles of their
qualifications), it never invents a qualification, and it returns the people who
matched with the reason -- never a bare, unexplained ranking. A task no one can
fully cover still returns the closest people, each with the missing items named.
"""

from __future__ import annotations

from dataclasses import dataclass

from core.workplace.records import Assignment, Person, Project, Qualification
from core.workplace.repository import WorkplaceRepository


@dataclass(frozen=True)
class PersonProfile:
    """One person with their qualifications and their project history."""

    person: Person
    qualifications: tuple[Qualification, ...] = ()
    assignments: tuple[Assignment, ...] = ()
    projects: tuple[Project, ...] = ()

    @property
    def is_available(self) -> bool:
        return self.person.available

    @property
    def certification_titles(self) -> tuple[str, ...]:
        return tuple(q.title for q in self.qualifications if q.title)

    def describe(self, *, language: str = "en") -> str:
        person = self.person
        lines = [f"{person.name} — {person.role or person.discipline or '—'}"]
        if person.division_id:
            lines.append(f"Division: {person.division_id}")
        if self.qualifications:
            lines.append(
                "Qualifications: "
                + "; ".join(q.title or q.kind for q in self.qualifications)
            )
        if self.projects:
            lines.append(
                "Projects: "
                + ", ".join(f"{p.number} {p.title}".strip() for p in self.projects)
            )
        if not person.available:
            lines.append("(not currently available)")
        return "\n".join(lines)


@dataclass(frozen=True)
class Candidate:
    """One ranked candidate for a task, with what they hold and what is missing."""

    person: Person
    score: float
    matched: tuple[str, ...] = ()
    missing: tuple[str, ...] = ()

    def describe(self, *, language: str = "en") -> str:
        parts = [f"{self.person.name} ({self.score:.2f})"]
        if self.matched:
            parts.append("matches: " + ", ".join(self.matched))
        if self.missing:
            parts.append("missing: " + ", ".join(self.missing))
        return " — ".join(parts)


@dataclass(frozen=True)
class CompetencyMatrix:
    """Which people hold which qualifications, as a readable grid."""

    columns: tuple[str, ...] = ()
    rows: tuple[tuple[str, tuple[bool, ...]], ...] = ()

    @property
    def empty(self) -> bool:
        return not self.columns or not self.rows

    def coverage(self) -> dict[str, int]:
        """How many people hold each qualification."""
        totals = dict.fromkeys(self.columns, 0)
        for _name, marks in self.rows:
            for column, held in zip(self.columns, marks, strict=True):
                if held:
                    totals[column] += 1
        return totals

    def describe(self, *, language: str = "en") -> str:
        if self.empty:
            return "No competency data yet."
        lines = ["Competency matrix (● held, · not held):"]
        header = "  " + "  ".join(f"{c[:14]:<14}" for c in self.columns)
        lines.append(header)
        for name, marks in self.rows:
            cells = "  ".join(f"{('●' if m else '·'):<14}" for m in marks)
            lines.append(f"{name[:20]:<20}{cells}")
        return "\n".join(lines)


class PersonnelDirectory:
    """Read-and-write personnel views over the Workplace repository.

    Reads resolve references and group records; the write helpers go through the
    :class:`~core.workplace.domain.WorkplaceDomain` service so every edit is
    audited exactly like any other Workplace change.
    """

    def __init__(self, repository: WorkplaceRepository, domain: object | None = None) -> None:
        self.repository = repository
        # The domain service is the audited writer; it is optional so the directory
        # can be read-only (the evaluation and the CLI both build it either way).
        self._domain = domain

    # -- reads ------------------------------------------------------------

    def people(
        self,
        *,
        division_id: str | None = None,
        discipline: str | None = None,
        available_only: bool = False,
    ) -> list[Person]:
        where = []
        params: list[str] = []
        if division_id:
            where.append("division_id = ?")
            params.append(division_id)
        if discipline:
            where.append("discipline = ?")
            params.append(discipline)
        clause = " AND ".join(where)
        rows = self.repository.list_domain("people", where=clause, params=tuple(params))
        if available_only:
            rows = [row for row in rows if row.available]
        rows.sort(key=lambda row: row.name.casefold())
        return rows

    def qualifications(self, person_id: str) -> list[Qualification]:
        return self.repository.list_domain(
            "qualifications", where="person_id = ?", params=(person_id,)
        )

    def profile(self, person_id: str) -> PersonProfile | None:
        person = self.repository.get_domain("people", person_id)
        if not isinstance(person, Person):
            return None
        assignments = self.repository.list_domain(
            "assignments", where="person_id = ?", params=(person_id,)
        )
        projects: list[Project] = []
        for assignment in assignments:
            project = self.repository.get_domain("projects", assignment.project_id)
            if isinstance(project, Project):
                projects.append(project)
        return PersonProfile(
            person=person,
            qualifications=tuple(self.qualifications(person_id)),
            assignments=tuple(assignments),
            projects=tuple(projects),
        )

    def disciplines(self) -> list[str]:
        """The distinct disciplines the personnel records actually carry."""
        seen = {
            person.discipline
            for person in self.repository.list_domain("people")
            if person.discipline
        }
        return sorted(seen)

    # -- the competency matrix -------------------------------------------

    def competency_matrix(
        self, *, division_id: str | None = None, limit: int = 20
    ) -> CompetencyMatrix:
        """A person × qualification grid over the people the Workplace holds.

        The columns are the qualification titles that at least one person in the
        set holds; a cell is filled when that person holds that qualification.
        Nothing is invented: a person with no qualifications is a row of blanks,
        which is itself the honest picture of the coverage.
        """
        people = self.people(division_id=division_id)
        per_person: list[tuple[str, set[str]]] = []
        columns: list[str] = []
        seen: set[str] = set()
        for person in people:
            titles = {
                q.title for q in self.qualifications(person.record_id) if q.title
            }
            per_person.append((person.name, titles))
            for title in sorted(titles):
                if title not in seen:
                    seen.add(title)
                    columns.append(title)
        columns = columns[:limit]
        rows = tuple(
            (name, tuple(column in titles for column in columns))
            for name, titles in per_person
        )
        return CompetencyMatrix(columns=tuple(columns), rows=rows)

    # -- project-specific selection --------------------------------------

    def select(
        self,
        *,
        skills: tuple[str, ...] = (),
        qualifications: tuple[str, ...] = (),
        division_id: str | None = None,
        limit: int = 5,
        available_only: bool = True,
    ) -> list[Candidate]:
        """Rank the people who can cover ``skills``/``qualifications``.

        A person's *holdings* are the words their own record carries -- their role,
        their discipline, and the titles of their qualifications -- matched
        case-insensitively as whole phrases. The score is the share of the required
        items the person holds (``(matched) / (required)``), so a full match is
        ``1.0``; the ranking is by score, then by name, so it is deterministic. The
        matched and missing items are named on each candidate, so the choice is
        explainable and a gap is never hidden.
        """
        required = [item for item in (*skills, *qualifications) if item.strip()]
        people = self.people(division_id=division_id, available_only=available_only)
        candidates: list[Candidate] = []
        for person in people:
            held = self._holdings(person)
            matched = tuple(item for item in required if self._holds(held, item))
            missing = tuple(item for item in required if item not in matched)
            if required:
                score = len(matched) / len(required)
            else:
                score = 1.0  # nothing was required: everyone is equally suitable
            candidates.append(
                Candidate(person=person, score=score, matched=matched, missing=missing)
            )
        # A person holding nothing required still appears, but below the rest.
        candidates.sort(key=lambda c: (-c.score, c.person.name.casefold()))
        return candidates[:limit]

    def _holdings(self, person: Person) -> str:
        parts = [person.role, person.discipline]
        parts.extend(q.title for q in self.qualifications(person.record_id))
        return " \n ".join(part for part in parts if part).casefold()

    @staticmethod
    def _holds(holdings: str, item: str) -> bool:
        return item.strip().casefold() in holdings

    # -- writes (audited through the domain service) ---------------------

    def add_person(self, *, name: str, **fields: object) -> Person:
        if self._domain is None:
            raise RuntimeError("This directory is read-only (no domain service).")
        return self._domain.create_person(name=name, **fields)  # type: ignore[attr-defined]

    def add_qualification(self, *, person_id: str, title: str, **fields: object) -> Qualification:
        if self._domain is None:
            raise RuntimeError("This directory is read-only (no domain service).")
        return self._domain.add_qualification(  # type: ignore[attr-defined]
            person_id=person_id, title=title, **fields
        )

    def assign(self, *, project_id: str, person_id: str, role: str = "", **fields: object) -> Assignment:
        if self._domain is None:
            raise RuntimeError("This directory is read-only (no domain service).")
        return self._domain.assign(  # type: ignore[attr-defined]
            project_id=project_id, person_id=person_id, role=role, **fields
        )


__all__ = [
    "Candidate",
    "CompetencyMatrix",
    "PersonProfile",
    "PersonnelDirectory",
]
