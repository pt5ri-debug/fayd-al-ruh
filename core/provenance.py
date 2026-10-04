from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any
from uuid import uuid4


class ProvenanceStage(str, Enum):
    """The cognitive layer that produced an artifact."""

    OBSERVATION = "observation"
    EVIDENCE = "evidence"
    BELIEF = "belief"
    PREDICTION = "prediction"
    PREDICTION_ERROR = "prediction_error"
    LEARNING = "learning"
    CAUSAL_LINK = "causal_link"
    MODEL_PROPOSAL = "model_proposal"


def _unique(values: tuple[str, ...] | list[str]) -> tuple[str, ...]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        if value not in seen:
            seen.add(value)
            result.append(value)
    return tuple(result)


@dataclass
class ProvenanceRecord:
    """Central provenance envelope carried by every cognitive artifact.

    One record captures the identity (``entity_id``/``id``), time
    (``recorded_at``), source, evidence (``evidence_ids``), confidence, and
    context (``context_id``) of an artifact. Because every transition records
    one, context cannot be lost as data moves from observation to evidence to
    belief to prediction to learning.
    """

    entity_id: str
    stage: ProvenanceStage
    source: str
    confidence: float
    context_id: str | None = None
    evidence_ids: tuple[str, ...] = ()
    parent_ids: tuple[str, ...] = ()
    recorded_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    id: str = field(default_factory=lambda: f"PV-{uuid4().hex[:8]}")
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.entity_id.strip():
            raise ValueError("Provenance entity id cannot be empty.")
        if not self.source.strip():
            raise ValueError("Provenance source cannot be empty.")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("Provenance confidence must be between 0 and 1.")
        self.evidence_ids = _unique(self.evidence_ids)
        self.parent_ids = _unique(self.parent_ids)

    @classmethod
    def from_artifact(
        cls,
        artifact: Any,
        stage: ProvenanceStage,
        *,
        source: str | None = None,
        confidence: float | None = None,
        context_id: str | None = None,
        evidence_ids: tuple[str, ...] | list[str] = (),
        parent_ids: tuple[str, ...] | list[str] = (),
        metadata: dict[str, Any] | None = None,
    ) -> "ProvenanceRecord":
        """Build a record for any object exposing an ``id`` attribute."""
        entity_id = getattr(artifact, "id", None)
        if not isinstance(entity_id, str) or not entity_id.strip():
            raise ValueError("Artifact must expose a non-empty 'id'.")

        resolved_source = source or getattr(artifact, "source", None) or "internal"

        if confidence is None:
            raw = getattr(artifact, "confidence", None)
            if raw is None:
                raw = getattr(artifact, "strength", None)
            if raw is None:
                raw = 1.0
            confidence = float(raw)

        return cls(
            entity_id=entity_id,
            stage=stage,
            source=str(resolved_source),
            confidence=float(confidence),
            context_id=context_id,
            evidence_ids=tuple(evidence_ids),
            parent_ids=tuple(parent_ids),
            metadata=dict(metadata or {}),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "entity_id": self.entity_id,
            "stage": self.stage.value,
            "source": self.source,
            "confidence": self.confidence,
            "context_id": self.context_id,
            "evidence_ids": list(self.evidence_ids),
            "parent_ids": list(self.parent_ids),
            "recorded_at": self.recorded_at.isoformat(),
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ProvenanceRecord":
        record = cls(
            entity_id=data["entity_id"],
            stage=ProvenanceStage(data["stage"]),
            source=data["source"],
            confidence=float(data["confidence"]),
            context_id=data.get("context_id"),
            evidence_ids=tuple(data.get("evidence_ids", [])),
            parent_ids=tuple(data.get("parent_ids", [])),
            metadata=data.get("metadata", {}),
        )
        if data.get("recorded_at"):
            record.recorded_at = datetime.fromisoformat(data["recorded_at"])
        if data.get("id"):
            record.id = data["id"]
        return record


class ProvenanceLedger:
    """Central, ordered audit trail of provenance records, keyed by entity id.

    Several records may exist for one entity (for example a belief recorded at
    creation and again after learning); ``get`` returns the latest while
    ``history`` returns the full chain.
    """

    def __init__(self) -> None:
        self._records: dict[str, list[ProvenanceRecord]] = {}

    def record(self, record: ProvenanceRecord) -> ProvenanceRecord:
        self._records.setdefault(record.entity_id, []).append(record)
        return record

    def get(self, entity_id: str) -> ProvenanceRecord | None:
        items = self._records.get(entity_id)
        return items[-1] if items else None

    def require(self, entity_id: str) -> ProvenanceRecord:
        record = self.get(entity_id)
        if record is None:
            raise KeyError(f"No provenance recorded for entity: {entity_id}")
        return record

    def history(self, entity_id: str) -> list[ProvenanceRecord]:
        return list(self._records.get(entity_id, ()))

    def for_context(self, context_id: str) -> list[ProvenanceRecord]:
        return [
            record
            for records in self._records.values()
            for record in records
            if record.context_id == context_id
        ]

    def for_stage(self, stage: ProvenanceStage) -> list[ProvenanceRecord]:
        return [
            record
            for records in self._records.values()
            for record in records
            if record.stage == stage
        ]

    def __len__(self) -> int:
        return sum(len(records) for records in self._records.values())

    def __contains__(self, entity_id: str) -> bool:
        return entity_id in self._records

    def dump(self) -> dict[str, list[dict[str, Any]]]:
        return {
            entity_id: [record.to_dict() for record in records]
            for entity_id, records in self._records.items()
        }

    @classmethod
    def restore(cls, data: dict[str, list[dict[str, Any]]]) -> "ProvenanceLedger":
        ledger = cls()
        for records in data.values():
            for item in records:
                ledger.record(ProvenanceRecord.from_dict(item))
        return ledger
