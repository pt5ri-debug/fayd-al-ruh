from __future__ import annotations

from collections import deque

from core.belief import Belief
from core.entity import Entity
from core.evidence import Evidence
from core.relation import Relation


class KnowledgeGraph:
    """
    Knowledge Graph 2.0

    Manages:
        Entity
        Evidence
        Belief
        Relation

    The graph guarantees referential integrity:
        - Relations must reference existing entities.
        - Beliefs must exist before relations reference them.
        - Evidence must exist before relations or beliefs reference it.
    """

    def __init__(self) -> None:
        self.entities: dict[str, Entity] = {}
        self.evidence: dict[str, Evidence] = {}
        self.beliefs: dict[str, Belief] = {}
        self.relations: dict[str, Relation] = {}

    # =========================================================
    # Entity management
    # =========================================================

    def add_entity(self, entity: Entity) -> None:
        if entity.id in self.entities:
            raise ValueError(
                f"Entity '{entity.id}' already exists."
            )

        self.entities[entity.id] = entity

    def get_entity(self, entity_id: str) -> Entity | None:
        return self.entities.get(entity_id)

    def remove_entity(self, entity_id: str) -> None:
        if entity_id not in self.entities:
            return

        relation_ids = [
            relation.id
            for relation in self.relations.values()
            if relation.source == entity_id
            or relation.target == entity_id
        ]

        for relation_id in relation_ids:
            self.remove_relation(relation_id)

        del self.entities[entity_id]

    # =========================================================
    # Evidence management
    # =========================================================

    def add_evidence(self, evidence: Evidence) -> None:
        if evidence.id in self.evidence:
            raise ValueError(
                f"Evidence '{evidence.id}' already exists."
            )

        self.evidence[evidence.id] = evidence

    def get_evidence(self, evidence_id: str) -> Evidence | None:
        return self.evidence.get(evidence_id)

    def remove_evidence(self, evidence_id: str) -> None:
        if evidence_id not in self.evidence:
            return

        # Remove evidence references from beliefs.
        for belief in self.beliefs.values():
            if evidence_id in belief.supporting_evidence:
                belief.supporting_evidence.remove(evidence_id)

            if evidence_id in belief.contradicting_evidence:
                belief.contradicting_evidence.remove(evidence_id)

        # Remove evidence references from relations.
        for relation in self.relations.values():
            if evidence_id in relation.evidence_ids:
                relation.remove_evidence(evidence_id)

        del self.evidence[evidence_id]

    # =========================================================
    # Belief management
    # =========================================================

    def add_belief(self, belief: Belief) -> None:
        if belief.id in self.beliefs:
            raise ValueError(
                f"Belief '{belief.id}' already exists."
            )

        self._validate_belief_evidence(belief)

        self.beliefs[belief.id] = belief

    def get_belief(self, belief_id: str) -> Belief | None:
        return self.beliefs.get(belief_id)

    def remove_belief(self, belief_id: str) -> None:
        if belief_id not in self.beliefs:
            return

        for relation in self.relations.values():
            if relation.belief_id == belief_id:
                relation.belief_id = None
                relation._touch()

        del self.beliefs[belief_id]

    # =========================================================
    # Relation management
    # =========================================================

    def add_relation(self, relation: Relation) -> None:

        if relation.id in self.relations:
            raise ValueError(
                f"Relation '{relation.id}' already exists."
            )

        if relation.source not in self.entities:
            raise ValueError(
                f"Source entity '{relation.source}' does not exist."
            )

        if relation.target not in self.entities:
            raise ValueError(
                f"Target entity '{relation.target}' does not exist."
            )

        if relation.belief_id is not None:
            if relation.belief_id not in self.beliefs:
                raise ValueError(
                    f"Belief '{relation.belief_id}' does not exist."
                )

        for evidence_id in relation.evidence_ids:
            if evidence_id not in self.evidence:
                raise ValueError(
                    f"Evidence '{evidence_id}' does not exist."
                )

        self.relations[relation.id] = relation

    def get_relation(self, relation_id: str) -> Relation | None:
        return self.relations.get(relation_id)

    def remove_relation(self, relation_id: str) -> None:
        self.relations.pop(relation_id, None)

    # =========================================================
    # Query
    # =========================================================

    def find_relations(
        self,
        entity_id: str,
        relation_type: str | None = None,
    ) -> list[Relation]:

        if entity_id not in self.entities:
            return []

        result = []

        for relation in self.relations.values():

            if not relation.active:
                continue

            connected = (
                relation.source == entity_id
                or relation.target == entity_id
            )

            if not connected:
                continue

            if (
                relation_type is not None
                and relation.relation_type != relation_type
            ):
                continue

            result.append(relation)

        return result

    def get_neighbors(self, entity_id: str) -> list[Entity]:

        neighbors: list[Entity] = []
        seen: set[str] = set()

        for relation in self.find_relations(entity_id):

            if relation.source == entity_id:
                neighbor_id = relation.target
            else:
                neighbor_id = relation.source

            if neighbor_id in seen:
                continue

            neighbor = self.entities.get(neighbor_id)

            if neighbor is not None:
                neighbors.append(neighbor)
                seen.add(neighbor_id)

        return neighbors

    def find_by_name(self, name: str) -> list[Entity]:
        return [
            entity
            for entity in self.entities.values()
            if entity.name == name
        ]

    # =========================================================
    # Belief / evidence queries
    # =========================================================

    def get_relation_belief(
        self,
        relation_id: str,
    ) -> Belief | None:

        relation = self.get_relation(relation_id)

        if relation is None:
            return None

        if relation.belief_id is None:
            return None

        return self.get_belief(relation.belief_id)

    def get_relation_evidence(
        self,
        relation_id: str,
    ) -> list[Evidence]:

        relation = self.get_relation(relation_id)

        if relation is None:
            return []

        return [
            self.evidence[evidence_id]
            for evidence_id in relation.evidence_ids
            if evidence_id in self.evidence
        ]

    def get_belief_evidence(
        self,
        belief_id: str,
    ) -> list[Evidence]:

        belief = self.get_belief(belief_id)

        if belief is None:
            return []

        evidence_ids = (
            belief.supporting_evidence
            + belief.contradicting_evidence
        )

        return [
            self.evidence[evidence_id]
            for evidence_id in evidence_ids
            if evidence_id in self.evidence
        ]

    # =========================================================
    # Graph traversal
    # =========================================================

    def find_path(
        self,
        start_entity_id: str,
        target_entity_id: str,
    ) -> list[str]:

        if start_entity_id not in self.entities:
            return []

        if target_entity_id not in self.entities:
            return []

        if start_entity_id == target_entity_id:
            return [start_entity_id]

        queue = deque([start_entity_id])

        visited = {start_entity_id}

        previous: dict[str, str | None] = {
            start_entity_id: None
        }

        while queue:

            current = queue.popleft()

            for neighbor in self.get_neighbors(current):

                if neighbor.id in visited:
                    continue

                visited.add(neighbor.id)

                previous[neighbor.id] = current

                if neighbor.id == target_entity_id:
                    return self._reconstruct_path(
                        previous,
                        target_entity_id,
                    )

                queue.append(neighbor.id)

        return []

    @staticmethod
    def _reconstruct_path(
        previous: dict[str, str | None],
        target: str,
    ) -> list[str]:

        path = []

        current: str | None = target

        while current is not None:
            path.append(current)
            current = previous[current]

        path.reverse()

        return path

    # =========================================================
    # Validation
    # =========================================================

    def validate(self) -> list[str]:
        """
        Returns a list of integrity errors.
        An empty list means the graph is consistent.
        """

        errors: list[str] = []

        for relation in self.relations.values():

            if relation.source not in self.entities:
                errors.append(
                    f"Relation {relation.id}: "
                    f"missing source entity."
                )

            if relation.target not in self.entities:
                errors.append(
                    f"Relation {relation.id}: "
                    f"missing target entity."
                )

            if (
                relation.belief_id is not None
                and relation.belief_id not in self.beliefs
            ):
                errors.append(
                    f"Relation {relation.id}: "
                    f"missing belief."
                )

            for evidence_id in relation.evidence_ids:
                if evidence_id not in self.evidence:
                    errors.append(
                        f"Relation {relation.id}: "
                        f"missing evidence {evidence_id}."
                    )

        for belief in self.beliefs.values():

            for evidence_id in (
                belief.supporting_evidence
                + belief.contradicting_evidence
            ):
                if evidence_id not in self.evidence:
                    errors.append(
                        f"Belief {belief.id}: "
                        f"missing evidence {evidence_id}."
                    )

        return errors

    # =========================================================
    # Internal validation
    # =========================================================

    def _validate_belief_evidence(
        self,
        belief: Belief,
    ) -> None:

        evidence_ids = (
            belief.supporting_evidence
            + belief.contradicting_evidence
        )

        for evidence_id in evidence_ids:

            if evidence_id not in self.evidence:
                raise ValueError(
                    f"Evidence '{evidence_id}' "
                    f"does not exist."
                )

    # =========================================================
    # Statistics
    # =========================================================

    @property
    def entity_count(self) -> int:
        return len(self.entities)

    @property
    def relation_count(self) -> int:
        return len(self.relations)

    @property
    def evidence_count(self) -> int:
        return len(self.evidence)

    @property
    def belief_count(self) -> int:
        return len(self.beliefs)

    def clear(self) -> None:
        self.entities.clear()
        self.evidence.clear()
        self.beliefs.clear()
        self.relations.clear()
