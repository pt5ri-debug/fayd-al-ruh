"""The Workplace record model: stable ids, audit columns, and typed records.

Every Workplace record is a small frozen dataclass with a **stable string id**, a
``created_at``/``updated_at`` audit pair, and an ``active`` flag. The id is
generated once (``make_id``) and never reused, so a record that is *deactivated*
keeps its identity and every foreign key that points at it still resolves -- which
is what lets an organisational unit be reorganised or retired without destroying
the historical project records that reference it.

The records are deliberately plain data: no behaviour, no database, no Tk. The
:mod:`core.workplace.repository` maps them to and from SQL, and the later phases
add their own record types (projects, personnel, standards, inspections, ...)
beside these. Keeping the model free of I/O is what makes the whole Workplace
testable with no database file and no display.
"""

from __future__ import annotations

import uuid
from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime
from typing import Any

#: Prefixes keep ids readable in a log and let a bare id be recognised.
_PREFIXES = {
    "institution": "INST",
    "unit": "UNIT",
    "division": "DIV",
}

#: The kinds whose records carry the institutional hierarchy (this phase).
INSTITUTION_KIND = "institution"
UNIT_KIND = "unit"
DIVISION_KIND = "division"


def utc_now() -> str:
    """The current UTC time as an ISO-8601 string (the one timestamp format)."""
    return datetime.now(UTC).isoformat()


def make_id(kind: str, *, prefix: str | None = None) -> str:
    """Return a stable, human-readable id for a record of ``kind``.

    The id is ``<PREFIX>-<hex>`` (e.g. ``DIV-3f9a1c2b``). It is generated once
    and never reused, so a record can be deactivated and still be referenced by
    history. An unknown ``kind`` uses its own upper-cased name as the prefix, so a
    later phase can mint ids without editing this table.
    """
    tag = prefix or _PREFIXES.get(kind, kind.upper()[:6] or "REC")
    return f"{tag}-{uuid.uuid4().hex[:8]}"


@dataclass(frozen=True)
class RecordBase:
    """The columns every Workplace record carries.

    ``record_id`` is the stable primary key; ``created_at``/``updated_at`` are the
    audit trail; ``active`` is the **soft** deletion flag -- a record is
    deactivated, never deleted, so history survives.
    """

    record_id: str
    created_at: str = field(default_factory=utc_now)
    updated_at: str = field(default_factory=utc_now)
    active: bool = True

    def touch(self, **changes: Any) -> Any:
        """Return a copy with ``changes`` applied and ``updated_at`` refreshed.

        Frozen records are never mutated in place, so a correction is a new value
        -- the same "a revision is a new record" discipline the learning records
        use. Subclasses inherit this and keep their own fields.
        """
        return replace(self, updated_at=utc_now(), **changes)

    def deactivate(self) -> Any:
        """Return a copy marked inactive (the soft delete)."""
        return self.touch(active=False)

    def activate(self) -> Any:
        """Return a copy marked active again."""
        return self.touch(active=True)


#: Anything with the base columns -- what the repository can store generically.
WorkplaceRecord = RecordBase


@dataclass(frozen=True)
class Institution(RecordBase):
    """An institution: EPRI itself, or another body the Workplace tracks.

    ``official`` records whether the entry is publicly documented (EPRI) or an
    internal/proposed entry, so a reader can always tell a documented fact from a
    working assumption.
    """

    key: str = ""
    name: str = ""
    name_ar: str = ""
    kind: str = "institute"
    country: str = "Egypt"
    website: str = ""
    summary: str = ""
    official: bool = True
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class OrganisationalUnit(RecordBase):
    """A node in an institution's hierarchy (department, centre, lab, ...).

    ``parent_id`` is empty for a top-level unit and otherwise the id of the
    containing unit, so the hierarchy is a tree. ``institution_id`` links the
    whole tree to its institution. A unit is reorganised by editing its
    ``parent_id``/``name`` or deactivating it -- never by deleting it.
    """

    institution_id: str = ""
    parent_id: str = ""
    key: str = ""
    name: str = ""
    name_ar: str = ""
    kind: str = "department"
    summary: str = ""
    official: bool = True
    order: int = 0
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Division(RecordBase):
    """A professional division of the Technical Support & Technology Center.

    The ten divisions shipped in :mod:`core.workplace.seed` are **proposed,
    configurable** defaults (``proposed=True``) -- not a claim of an officially
    approved structure. An administrator adds, renames, reorganises, and
    deactivates divisions without touching source code: every division is a row,
    and a division's ``unit_id`` places it in the hierarchy.
    """

    unit_id: str = ""
    key: str = ""
    name: str = ""
    name_ar: str = ""
    summary: str = ""
    scope: str = ""
    proposed: bool = True
    order: int = 0
    metadata: Mapping[str, Any] = field(default_factory=dict)


#: The record classes this phase's repository knows how to store, by table name.
RECORD_TYPES: dict[str, type[RecordBase]] = {
    "institutions": Institution,
    "organisational_units": OrganisationalUnit,
    "divisions": Division,
}


__all__ = [
    "DIVISION_KIND",
    "INSTITUTION_KIND",
    "RECORD_TYPES",
    "UNIT_KIND",
    "Division",
    "Institution",
    "OrganisationalUnit",
    "RecordBase",
    "WorkplaceRecord",
    "make_id",
    "utc_now",
]
