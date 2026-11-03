"""Entity lifecycle over the graph: create, version, detect duplicates, merge.

:class:`~core.entity.Entity` is the *record*; this module is the *manager*. It
adds the lifecycle operations the task requires that a bare dataclass cannot own
by itself -- because they touch the whole graph:

* **stable identity** -- an entity keeps its id for life; a merge does not
  renumber the survivor;
* **versioning** -- every mutation records a bounded :class:`EntityVersion`
  snapshot, so an entity's history can be read back;
* **duplicate detection** -- two entities are candidates when their *normalised*
  names and types agree (case, whitespace, and Arabic letter forms normalised
  through the project's own :mod:`core.tokenize`, so "Porosity" and "porosity"
  are one concept);
* **merging with preserved history** -- the loser's relations are re-pointed to
  the survivor (never dropped), its properties are folded in without overwriting
  the survivor's, its version history is retained under the survivor, and the
  merge itself is recorded.

Read-only with respect to *meaning*: a merge never invents a property, and a
duplicate is a *candidate* a caller confirms -- ``detect_duplicates`` reports, it
does not merge on its own.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

from core.entity import Entity, EntityType
from core.knowledge_graph import KnowledgeGraph
from core.tokenize import normalize_phrase

#: How many versions of one entity are kept (bounded, newest last).
DEFAULT_VERSION_LIMIT = 64


@dataclass(frozen=True)
class EntityVersion:
    """A snapshot of one entity at a moment, and why it changed."""

    entity_id: str
    version: int
    snapshot: dict[str, Any]
    reason: str
    at: datetime = field(default_factory=lambda: datetime.now(UTC))

    def describe(self) -> str:
        return f"v{self.version} ({self.reason}) at {self.at.isoformat()}"


@dataclass(frozen=True)
class DuplicateCandidate:
    """Two entities that look like the same thing, and how sure the check is."""

    keeper: Entity
    duplicate: Entity
    similarity: float
    reason: str

    def describe(self) -> str:
        return f"'{self.duplicate.name}' ~ '{self.keeper.name}' ({self.similarity:.2f}: {self.reason})"


@dataclass(frozen=True)
class MergeResult:
    """What one merge did, so it is auditable."""

    survivor_id: str
    merged_id: str
    relations_moved: int
    properties_added: tuple[str, ...]
    at: datetime = field(default_factory=lambda: datetime.now(UTC))

    def describe(self) -> str:
        return (
            f"merged {self.merged_id} into {self.survivor_id} "
            f"({self.relations_moved} relation(s) moved)"
        )


def normalize_name(name: str) -> str:
    """A comparison key for an entity name: case, space, and Arabic normalised."""
    return normalize_phrase(name).strip()


class EntityManager:
    """Create, version, de-duplicate, and merge the entities in one graph."""

    def __init__(self, graph: KnowledgeGraph, *, version_limit: int = DEFAULT_VERSION_LIMIT) -> None:
        if version_limit < 0:
            raise ValueError("Version limit cannot be negative.")
        self.graph = graph
        self.version_limit = version_limit
        self._versions: dict[str, list[EntityVersion]] = {}

    # -- create / update ---------------------------------------------------

    def create(
        self,
        name: str,
        *,
        entity_type: EntityType = EntityType.CONCEPT,
        confidence: float = 0.5,
        properties: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> Entity:
        """Create an entity (assigning its stable id) and record version 1.

        A name that already exists yields the *existing* entity rather than a
        second one -- creation is idempotent by normalised name, which is what
        keeps duplicate resolution honest at the source.
        """
        existing = self.find_by_name(name, entity_type=entity_type)
        if existing is not None:
            return existing
        entity = Entity(
            entity_type=entity_type,
            name=name,
            confidence=confidence,
            properties=dict(properties or {}),
            metadata=dict(metadata or {}),
        )
        self.graph.add_entity(entity)
        self._record_version(entity, "created")
        return entity

    def update(
        self,
        entity_id: str,
        *,
        name: str | None = None,
        confidence: float | None = None,
        properties: dict[str, Any] | None = None,
        reason: str = "updated",
    ) -> Entity:
        """Mutate an entity and record a new version; returns the entity.

        An unknown id raises ``KeyError`` (a caller asked to update something that
        is not there -- silently doing nothing would hide the mistake).
        """
        entity = self.graph.get_entity(entity_id)
        if entity is None:
            raise KeyError(f"Unknown entity: {entity_id}")
        if name is not None and name.strip():
            entity.name = name
        if confidence is not None:
            entity.set_confidence(confidence)
        for key, value in (properties or {}).items():
            entity.set_property(key, value)
        entity.touch()
        self._record_version(entity, reason)
        return entity

    # -- lookup ------------------------------------------------------------

    def find_by_name(self, name: str, *, entity_type: EntityType | None = None) -> Entity | None:
        """The entity whose normalised name matches, or ``None``."""
        key = normalize_name(name)
        for entity in self.graph.entities.values():
            if normalize_name(entity.name) == key and (entity_type is None or entity.entity_type == entity_type):
                return entity
        return None

    def versions(self, entity_id: str) -> tuple[EntityVersion, ...]:
        """The recorded versions of one entity, oldest first."""
        return tuple(self._versions.get(entity_id, ()))

    def latest_version(self, entity_id: str) -> EntityVersion | None:
        items = self._versions.get(entity_id)
        return items[-1] if items else None

    # -- duplicates --------------------------------------------------------

    def detect_duplicates(self, *, threshold: float = 1.0) -> tuple[DuplicateCandidate, ...]:
        """Pairs of entities that look like the same thing.

        ``threshold`` is the minimum similarity (1.0 = identical normalised name
        and type; lower values include a shared normalised name across types).
        The check is deterministic and ordered, so the same graph yields the same
        candidates.
        """
        entities = list(self.graph.entities.values())
        found: list[DuplicateCandidate] = []
        for index, first in enumerate(entities):
            for second in entities[index + 1 :]:
                candidate = self._compare(first, second, threshold)
                if candidate is not None:
                    found.append(candidate)
        found.sort(key=lambda item: (-item.similarity, item.keeper.name, item.duplicate.name))
        return tuple(found)

    @staticmethod
    def _compare(first: Entity, second: Entity, threshold: float) -> DuplicateCandidate | None:
        if normalize_name(first.name) != normalize_name(second.name):
            return None
        same_type = first.entity_type == second.entity_type
        similarity = 1.0 if same_type else 0.7
        if similarity < threshold:
            return None
        reason = "same name and type" if same_type else "same name, different type"
        # The older entity is the keeper, so a merge never renumbers the survivor.
        keeper, duplicate = (first, second) if first.created_at <= second.created_at else (second, first)
        return DuplicateCandidate(keeper=keeper, duplicate=duplicate, similarity=similarity, reason=reason)

    # -- merge -------------------------------------------------------------

    def merge(self, survivor_id: str, merged_id: str) -> MergeResult:
        """Merge ``merged_id`` into ``survivor_id``, preserving both histories.

        Relations pointing at the duplicate are re-pointed to the survivor (a
        self-loop that would result from a relation *between* the two is dropped,
        since an entity cannot relate to itself). The duplicate's properties are
        folded in **without overwriting** the survivor's, its version history is
        appended to the survivor's, and the duplicate entity is then removed.
        """
        if survivor_id == merged_id:
            raise ValueError("Cannot merge an entity into itself.")
        survivor = self.graph.get_entity(survivor_id)
        duplicate = self.graph.get_entity(merged_id)
        if survivor is None:
            raise KeyError(f"Unknown survivor: {survivor_id}")
        if duplicate is None:
            raise KeyError(f"Unknown duplicate: {merged_id}")

        moved = 0
        for relation in list(self.graph.relations.values()):
            touched = False
            if relation.source == merged_id:
                relation.source = survivor_id
                touched = True
            if relation.target == merged_id:
                relation.target = survivor_id
                touched = True
            if not touched:
                continue
            if relation.source == relation.target:
                # The relation was between the two duplicates; it is now a
                # self-loop, which is meaningless, so it is removed.
                self.graph.remove_relation(relation.id)
                continue
            relation._touch()
            moved += 1

        added: list[str] = []
        for key, value in duplicate.properties.items():
            if key not in survivor.properties:
                survivor.properties[key] = value
                added.append(key)

        # Fold the duplicate's version history under the survivor, renumbered.
        prior = self._versions.pop(merged_id, [])
        base = len(self._versions.get(survivor_id, ()))
        for offset, version in enumerate(prior, start=1):
            self._versions.setdefault(survivor_id, []).append(
                EntityVersion(
                    entity_id=survivor_id,
                    version=base + offset,
                    snapshot=version.snapshot,
                    reason=f"from merged {merged_id}: {version.reason}",
                    at=version.at,
                )
            )

        survivor.touch()
        self._record_version(survivor, f"merged {merged_id}")
        self.graph.remove_entity(merged_id)
        self._trim(survivor_id)

        return MergeResult(
            survivor_id=survivor_id,
            merged_id=merged_id,
            relations_moved=moved,
            properties_added=tuple(added),
        )

    def merge_duplicates(self, *, threshold: float = 1.0) -> tuple[MergeResult, ...]:
        """Merge every exact duplicate (threshold 1.0 by default), oldest kept.

        Returns the merges performed, in order. Only pairs at or above the
        threshold are merged, so the default merges exact duplicates and leaves a
        same-name-different-type pair for a human to decide.
        """
        results: list[MergeResult] = []
        for candidate in self.detect_duplicates(threshold=threshold):
            if candidate.keeper.id not in self.graph.entities or candidate.duplicate.id not in self.graph.entities:
                continue
            results.append(self.merge(candidate.keeper.id, candidate.duplicate.id))
        return tuple(results)

    # -- internals ---------------------------------------------------------

    def _record_version(self, entity: Entity, reason: str) -> None:
        items = self._versions.setdefault(entity.id, [])
        items.append(
            EntityVersion(
                entity_id=entity.id,
                version=len(items) + 1,
                snapshot={
                    "name": entity.name,
                    "type": entity.entity_type.value,
                    "confidence": entity.confidence,
                    "properties": dict(entity.properties),
                },
                reason=reason,
            )
        )
        self._trim(entity.id)

    def _trim(self, entity_id: str) -> None:
        items = self._versions.get(entity_id)
        if items and self.version_limit and len(items) > self.version_limit:
            del items[: len(items) - self.version_limit]

    def describe(self) -> str:
        return (
            f"EntityManager({len(self.graph.entities)} entity/entities, "
            f"{sum(len(v) for v in self._versions.values())} version record(s))"
        )


__all__ = [
    "DEFAULT_VERSION_LIMIT",
    "DuplicateCandidate",
    "EntityManager",
    "EntityVersion",
    "MergeResult",
    "normalize_name",
]
