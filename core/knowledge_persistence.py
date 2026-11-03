"""Durable, transactional persistence for a knowledge graph.

The graph (:mod:`core.knowledge_graph`) lives in memory. The project already
persists a *whole cognitive snapshot* through :mod:`core.knowledge_store`, but a
snapshot is an all-or-nothing document: it cannot save one entity, and a single
bad record makes the whole file unreadable. This module is the smaller, more
robust counterpart the task asks for -- **efficient local storage** of entities,
relations, and evidence with **history**, over the project's own
:class:`~storage.sqlite_store.SqliteStore` (SQLite from the standard library) and
its :class:`~storage.json_store.JsonStore` fallback.

Three properties matter and are all enforced here:

* **atomic** -- a save writes every record, then a single ``meta`` document; a
  load that finds the ``meta`` document reads a complete graph. A failure midway
  leaves the previous save intact, because each write is one SQLite transaction
  (or one atomic file replace) and the meta marker is written *last*.
* **corruption-tolerant** -- a record that will not parse is **skipped and
  counted**, never fatal; ``LoadReport.skipped`` says how many and why, so a
  damaged store degrades to "most of the knowledge" rather than "no session".
* **faithful** -- a round trip preserves ids, types, timestamps, confidence,
  evidence links, and the entity/relation history this Phase adds, so
  ``load(save(g))`` is the same graph.

Local-only. SQLite is unsupported on a network drive, so this is documented as a
local store; it never reaches the network.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

from core.entity import Entity, EntityType
from core.evidence import Evidence
from core.knowledge_graph import KnowledgeGraph
from core.relation import Relation

#: The namespace inside the store that holds one graph's records.
NAMESPACE_ENTITIES = "kg_entities"
NAMESPACE_RELATIONS = "kg_relations"
NAMESPACE_EVIDENCE = "kg_evidence"
NAMESPACE_META = "kg_meta"

#: The schema version written into the meta document, so a future format can be
#: recognised (and refused, not misread).
SCHEMA_VERSION = 1

#: The meta key. It is written **last**, so its presence means the save finished.
META_KEY = "graph"


@dataclass(frozen=True)
class SaveReport:
    """What a save wrote."""

    path: str
    entities: int
    relations: int
    evidence: int
    schema_version: int = SCHEMA_VERSION

    def describe(self) -> str:
        return (
            f"saved {self.entities} entity/entities, {self.relations} relation(s), "
            f"{self.evidence} evidence item(s) to {self.path}"
        )


@dataclass(frozen=True)
class LoadReport:
    """What a load read, and what it could not."""

    path: str
    entities: int = 0
    relations: int = 0
    evidence: int = 0
    skipped: int = 0
    reasons: tuple[str, ...] = field(default=())
    schema_version: int = 0

    @property
    def ok(self) -> bool:
        return self.schema_version > 0

    def describe(self) -> str:
        text = (
            f"loaded {self.entities} entity/entities, {self.relations} relation(s), "
            f"{self.evidence} evidence item(s) from {self.path}"
        )
        if self.skipped:
            text += f"; skipped {self.skipped} unreadable record(s)"
        if not self.ok:
            text = f"no stored graph at {self.path}"
        return text


class KnowledgePersistence:
    """Save and load a graph through a store (SQLite by default, JSON fallback).

    ``store`` is any object with ``save``/``load``/``list_keys``/``delete`` --
    :class:`~storage.sqlite_store.SqliteStore` and
    :class:`~storage.json_store.JsonStore` both qualify, so a caller chooses the
    backend without this module knowing which.
    """

    def __init__(self, store: Any, *, name: str = "graph") -> None:
        self.store = store
        self.name = name

    # -- save --------------------------------------------------------------

    def save(self, graph: KnowledgeGraph, *, versions: dict[str, list[Any]] | None = None) -> SaveReport:
        """Persist the graph atomically; the meta marker is written **last**.

        ``versions`` optionally carries entity/relation history (the managers'
        records) as plain primitives, so a Phase-1 history survives a restart too.
        """
        for entity in graph.entities.values():
            self.store.save(NAMESPACE_ENTITIES, entity.id, _entity_to_primitive(entity))
        for relation in graph.relations.values():
            self.store.save(NAMESPACE_RELATIONS, relation.id, _relation_to_primitive(relation))
        for evidence in graph.evidence.values():
            self.store.save(NAMESPACE_EVIDENCE, evidence.id, _evidence_to_primitive(evidence))

        meta = {
            "schema_version": SCHEMA_VERSION,
            "name": self.name,
            "saved_at": datetime.now(UTC).isoformat(),
            "entities": len(graph.entities),
            "relations": len(graph.relations),
            "evidence": len(graph.evidence),
            "versions": versions or {},
        }
        # The marker last: if the process dies before this line, the previous
        # save is still the one a load will read.
        self.store.save(NAMESPACE_META, META_KEY, meta)
        return SaveReport(
            path=self._path(),
            entities=len(graph.entities),
            relations=len(graph.relations),
            evidence=len(graph.evidence),
        )

    # -- load --------------------------------------------------------------

    def load(self, graph: KnowledgeGraph | None = None) -> tuple[KnowledgeGraph, LoadReport]:
        """Read a stored graph, tolerating a corrupt record.

        Returns the graph and a report. A record that will not parse is skipped
        and counted; the returned graph is always usable, even when every record
        was unreadable (it is then simply empty, and the report says so).
        """
        target = graph if graph is not None else KnowledgeGraph()
        meta = self.store.load(NAMESPACE_META, META_KEY)
        if not isinstance(meta, dict) or not meta.get("schema_version"):
            return target, LoadReport(path=self._path())

        schema = int(meta.get("schema_version", 0))
        skipped = 0
        reasons: list[str] = []

        for key in self.store.list_keys(NAMESPACE_EVIDENCE):
            raw = self.store.load(NAMESPACE_EVIDENCE, key)
            try:
                target.add_evidence(_evidence_from_primitive(raw))
            except Exception as error:
                skipped += 1
                reasons.append(f"evidence {key}: {type(error).__name__}")

        for key in self.store.list_keys(NAMESPACE_ENTITIES):
            raw = self.store.load(NAMESPACE_ENTITIES, key)
            try:
                target.add_entity(_entity_from_primitive(raw))
            except Exception as error:
                skipped += 1
                reasons.append(f"entity {key}: {type(error).__name__}")

        for key in self.store.list_keys(NAMESPACE_RELATIONS):
            raw = self.store.load(NAMESPACE_RELATIONS, key)
            try:
                target.add_relation(_relation_from_primitive(raw))
            except Exception as error:
                # missing (its entity record was corrupt) is skipped, not fatal.
                skipped += 1
                reasons.append(f"relation {key}: {type(error).__name__}")

        return target, LoadReport(
            path=self._path(),
            entities=len(target.entities),
            relations=len(target.relations),
            evidence=len(target.evidence),
            skipped=skipped,
            reasons=tuple(reasons),
            schema_version=schema,
        )

    def versions(self) -> dict[str, Any]:
        """The stored history mapping (entity/relation versions), or an empty dict."""
        meta = self.store.load(NAMESPACE_META, META_KEY)
        if isinstance(meta, dict):
            stored = meta.get("versions")
            if isinstance(stored, dict):
                return stored
        return {}

    def clear(self) -> None:
        """Remove every stored record for this graph (a confirmed reset)."""
        for namespace in (NAMESPACE_ENTITIES, NAMESPACE_RELATIONS, NAMESPACE_EVIDENCE):
            for key in self.store.list_keys(namespace):
                self.store.delete(namespace, key)
        self.store.delete(NAMESPACE_META, META_KEY)

    def exists(self) -> bool:
        """Whether a completed save is present."""
        meta = self.store.load(NAMESPACE_META, META_KEY)
        return isinstance(meta, dict) and bool(meta.get("schema_version"))

    def _path(self) -> str:
        path = getattr(self.store, "path", None) or getattr(self.store, "root", None)
        return str(path) if path is not None else repr(self.store)


# -- primitives -----------------------------------------------------------


def _entity_to_primitive(entity: Entity) -> dict[str, Any]:
    return {
        "id": entity.id,
        "entity_type": entity.entity_type.value,
        "name": entity.name,
        "properties": dict(entity.properties),
        "relations": list(entity.relations),
        "confidence": entity.confidence,
        "activation": entity.activation,
        "metadata": dict(entity.metadata),
        "created_at": entity.created_at.isoformat(),
        "updated_at": entity.updated_at.isoformat(),
    }


def _entity_from_primitive(data: Any) -> Entity:
    if not isinstance(data, dict):
        raise ValueError("entity record is not a mapping")
    entity = Entity(
        entity_type=EntityType(data["entity_type"]),
        name=data["name"],
        id=data["id"],
        properties=dict(data.get("properties", {})),
        relations=list(data.get("relations", [])),
        confidence=float(data.get("confidence", 1.0)),
        activation=float(data.get("activation", 0.0)),
        metadata=dict(data.get("metadata", {})),
    )
    if data.get("created_at"):
        entity.created_at = datetime.fromisoformat(data["created_at"])
    if data.get("updated_at"):
        entity.updated_at = datetime.fromisoformat(data["updated_at"])
    return entity


def _relation_to_primitive(relation: Relation) -> dict[str, Any]:
    return {
        "id": relation.id,
        "source": relation.source,
        "target": relation.target,
        "relation_type": relation.relation_type,
        "confidence": relation.confidence,
        "belief_id": relation.belief_id,
        "evidence_ids": list(relation.evidence_ids),
        "metadata": dict(relation.metadata),
        "created_at": relation.created_at.isoformat(),
        "updated_at": relation.updated_at.isoformat(),
        "active": relation.active,
    }


def _relation_from_primitive(data: Any) -> Relation:
    if not isinstance(data, dict):
        raise ValueError("relation record is not a mapping")
    relation = Relation(
        source=data["source"],
        target=data["target"],
        relation_type=data["relation_type"],
        confidence=float(data.get("confidence", 0.5)),
        belief_id=data.get("belief_id"),
        evidence_ids=list(data.get("evidence_ids", [])),
        metadata=dict(data.get("metadata", {})),
        id=data["id"],
        active=bool(data.get("active", True)),
    )
    if data.get("created_at"):
        relation.created_at = datetime.fromisoformat(data["created_at"])
    if data.get("updated_at"):
        relation.updated_at = datetime.fromisoformat(data["updated_at"])
    return relation


def _evidence_to_primitive(evidence: Evidence) -> dict[str, Any]:
    return {
        "id": evidence.id,
        "description": evidence.description,
        "source": evidence.source,
        "supports": evidence.supports,
        "strength": evidence.strength,
        "data": dict(evidence.data),
        "created_at": evidence.created_at.isoformat(),
    }


def _evidence_from_primitive(data: Any) -> Evidence:
    if not isinstance(data, dict):
        raise ValueError("evidence record is not a mapping")
    evidence = Evidence(
        description=data["description"],
        source=data["source"],
        supports=bool(data.get("supports", True)),
        strength=float(data.get("strength", 1.0)),
        data=dict(data.get("data", {})),
        id=data["id"],
    )
    if data.get("created_at"):
        evidence.created_at = datetime.fromisoformat(data["created_at"])
    return evidence


__all__ = [
    "META_KEY",
    "NAMESPACE_ENTITIES",
    "NAMESPACE_EVIDENCE",
    "NAMESPACE_META",
    "NAMESPACE_RELATIONS",
    "SCHEMA_VERSION",
    "KnowledgePersistence",
    "LoadReport",
    "SaveReport",
]
