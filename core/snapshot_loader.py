from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from core.belief import Belief, BeliefStatus
from core.causal_graph import CausalGraph
from core.causal_link import CausalLink
from core.cognitive_context import ContextRegistry
from core.cognitive_state import CognitiveState
from core.entity import Entity, EntityType
from core.evidence import Evidence
from core.knowledge_graph import KnowledgeGraph
from core.memory import CognitiveMemory
from core.observation import Observation
from core.pattern_detector import Pattern
from core.prediction import Prediction
from core.prediction_error import PredictionError
from core.provenance import ProvenanceLedger
from core.relation import Relation
from core.snapshot import SNAPSHOT_SCHEMA_VERSION, Snapshot
from core.snapshot_migrations import migrate_payload
from core.cognitive_trace import CognitiveTrace
from core.hypothesis import Hypothesis


@dataclass(frozen=True)
class RestoredSnapshot:
    graph: KnowledgeGraph
    memory: CognitiveMemory
    state: CognitiveState
    contexts: ContextRegistry
    causal_graph: CausalGraph
    provenance: ProvenanceLedger
    version: int = SNAPSHOT_SCHEMA_VERSION
    migrations_applied: tuple[int, ...] = ()


class SnapshotLoader:
    """Restores data-only snapshots into verified cognitive objects.

    Loading is a three-stage safe rebuild: (1) validate the record shape,
    (2) upgrade older payloads through registered migrations, and (3) rebuild
    every object and check referential integrity before returning. A snapshot
    that cannot be fully rebuilt is rejected rather than partially loaded.
    """

    def restore(self, record: dict) -> RestoredSnapshot:
        payload, version = self._read_record(record)
        payload, migrations_applied = self._upgrade(payload, version)
        return self._rebuild(payload, migrations_applied)

    def _read_record(self, record: dict) -> tuple[dict, int]:
        payload = record.get("payload")
        checksum = record.get("checksum")
        if not isinstance(payload, dict) or not isinstance(checksum, str):
            raise ValueError("Invalid snapshot record.")
        snapshot = Snapshot(record.get("name", "restored"), payload, checksum, record.get("version", 1))
        if not snapshot.verify():
            raise ValueError("Snapshot checksum verification failed.")
        return payload, snapshot.version

    def _upgrade(self, payload: dict, version: int) -> tuple[dict, tuple[int, ...]]:
        applied = tuple(range(version, SNAPSHOT_SCHEMA_VERSION))
        return migrate_payload(payload, version), applied

    def _rebuild(self, payload: dict, migrations_applied: tuple[int, ...]) -> RestoredSnapshot:
        self._validate_payload(payload)
        graph = self._restore_graph(payload.get("graph", {}))
        memory = self._restore_memory(payload.get("memory", {}))
        state = self._restore_state(payload.get("state", {}), graph)
        contexts = ContextRegistry.restore(payload.get("contexts", {}))
        causal_graph = self._restore_causal_graph(payload.get("causal_graph", {}))
        provenance = ProvenanceLedger.restore(payload.get("provenance", {}))

        errors = graph.validate()
        errors += contexts.validate(graph, state, causal_graph)
        if errors:
            raise ValueError("Invalid restored snapshot: " + "; ".join(errors))

        return RestoredSnapshot(
            graph, memory, state, contexts, causal_graph, provenance,
            SNAPSHOT_SCHEMA_VERSION, migrations_applied,
        )

    @staticmethod
    def _validate_payload(payload: dict) -> None:
        for section in ("graph", "memory", "state", "contexts", "causal_graph", "provenance"):
            value = payload.get(section, {})
            if not isinstance(value, dict):
                raise ValueError(f"Snapshot section '{section}' must be an object.")

    def _restore_causal_graph(self, data: dict) -> CausalGraph:
        graph = CausalGraph()
        for item in data.get("links", {}).values():
            graph.add_link(CausalLink(
                cause=item["cause"], effect=item["effect"],
                confidence=item.get("confidence", .5), evidence_ids=item.get("evidence_ids", []),
                mechanism=item.get("mechanism"), metadata=item.get("metadata", {}),
                id=item["id"], created_at=self._time(item["created_at"]), active=item.get("active", True),
            ))
        return graph

    def _restore_graph(self, data: dict) -> KnowledgeGraph:
        graph = KnowledgeGraph()
        for item in data.get("entities", {}).values():
            graph.add_entity(Entity(
                EntityType(item["entity_type"]), item["name"], item["id"],
                item.get("properties", {}), item.get("relations", []),
                item.get("confidence", 1.0), item.get("activation", 0.0),
                item.get("metadata", {}), self._time(item["created_at"]), self._time(item["updated_at"]),
            ))
        for item in data.get("evidence", {}).values():
            graph.add_evidence(self._evidence(item))
        for item in data.get("beliefs", {}).values():
            graph.add_belief(Belief(
                item["statement"], item["confidence"], BeliefStatus(item["status"]),
                item.get("source_hypothesis_id"), item["id"],
                item.get("supporting_evidence", []), item.get("contradicting_evidence", []),
                item.get("metadata", {}), self._time(item["created_at"]), self._time(item["updated_at"]),
            ))
        for item in data.get("relations", {}).values():
            graph.add_relation(Relation(
                item["source"], item["target"], item["relation_type"], item.get("confidence", .5),
                item.get("belief_id"), item.get("evidence_ids", []), item.get("metadata", {}), item["id"],
                self._time(item["created_at"]), self._time(item["updated_at"]), item.get("active", True),
            ))
        return graph

    def _restore_memory(self, data: dict) -> CognitiveMemory:
        memory = CognitiveMemory()
        for item in data.get("items", {}).values():
            content = self._memory_content(item["item_type"], item["content"])
            restored = memory.store(item["item_id"], item["item_type"], content, item.get("importance", .5))
            restored.created_at = self._time(item["created_at"])
            restored.access_count = item.get("access_count", 0)
        return memory

    def _restore_state(self, data: dict, graph: KnowledgeGraph) -> CognitiveState:
        state = CognitiveState(cycle_count=data.get("cycle_count", 0))
        for belief in graph.beliefs.values():
            state.add_belief(belief)
        for item in data.get("patterns", {}).values():
            state.add_pattern(Pattern(
                item["key"], item["occurrences"], item["confidence"], item.get("memory_ids", []),
            ))
        for item in data.get("hypotheses", {}).values():
            state.add_hypothesis(Hypothesis(
                item["statement"], item["confidence"], item["source_pattern"], item["id"],
                self._time(item["created_at"]),
            ))
        for item in data.get("predictions", {}).values():
            prediction = Prediction(item["statement"], item["confidence"], item["hypothesis_id"], item["id"], self._time(item["created_at"]), item.get("fulfilled"))
            state.add_prediction(prediction)
        for item in data.get("prediction_errors", []):
            state.add_prediction_error(PredictionError(
                item["prediction_id"], item["expected_confidence"], item["actual_result"], item["error"], item["id"],
            ))
        for item in data.get("traces", {}).values():
            state.add_trace(CognitiveTrace(
                item["observation_id"], item["id"], item.get("pattern_keys", []),
                item.get("hypothesis_ids", []), item.get("prediction_ids", []),
                item.get("prediction_error_ids", []), item.get("belief_id"),
                self._time(item["created_at"]),
            ))
        return state

    def _memory_content(self, item_type: str, content):
        if item_type == "observation" and isinstance(content, dict):
            return Observation(content["content"], content["source"], content.get("confidence", 1.0), content.get("data", {}), content["id"], self._time(content["created_at"]))
        if item_type == "evidence" and isinstance(content, dict):
            return self._evidence(content)
        return content

    def _evidence(self, item: dict) -> Evidence:
        return Evidence(item["description"], item["source"], item.get("supports", True), item.get("strength", 1.0), item.get("data", {}), item["id"], self._time(item["created_at"]))

    @staticmethod
    def _time(value: str) -> datetime:
        return datetime.fromisoformat(value)
