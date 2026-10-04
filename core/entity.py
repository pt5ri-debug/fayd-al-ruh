from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List
from uuid import uuid4


class EntityType(str, Enum):
    OBJECT = "Object"
    PERSON = "Person"
    CONCEPT = "Concept"
    GOAL = "Goal"
    EVENT = "Event"
    MEMORY = "Memory"
    QUESTION = "Question"


@dataclass
class Entity:
    entity_type: EntityType
    name: str

    id: str = field(
        default_factory=lambda: f"E-{uuid4().hex[:8]}"
    )

    properties: Dict[str, Any] = field(
        default_factory=dict
    )

    relations: List[str] = field(
        default_factory=list
    )

    confidence: float = 1.0

    activation: float = 0.0

    metadata: Dict[str, Any] = field(
        default_factory=dict
    )

    created_at: datetime = field(
        default_factory=lambda: datetime.now(timezone.utc)
    )

    updated_at: datetime = field(
        default_factory=lambda: datetime.now(timezone.utc)
    )

    def __post_init__(self) -> None:
        self._validate()

    def _validate(self) -> None:
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(
                "Confidence must be between 0 and 1."
            )

        if not 0.0 <= self.activation <= 1.0:
            raise ValueError(
                "Activation must be between 0 and 1."
            )

        if not self.name.strip():
            raise ValueError(
                "Entity name cannot be empty."
            )

    def touch(self) -> None:
        self.updated_at = datetime.now(
            timezone.utc
        )

    def set_property(
        self,
        key: str,
        value: Any,
    ) -> None:
        self.properties[key] = value
        self.touch()

    def get_property(
        self,
        key: str,
    ) -> Any:
        return self.properties.get(key)

    def remove_property(
        self,
        key: str,
    ) -> None:
        if key in self.properties:
            del self.properties[key]
            self.touch()

    def add_relation(
        self,
        relation_id: str,
    ) -> None:
        if relation_id not in self.relations:
            self.relations.append(relation_id)
            self.touch()

    def remove_relation(
        self,
        relation_id: str,
    ) -> None:
        if relation_id in self.relations:
            self.relations.remove(relation_id)
            self.touch()

    def set_confidence(
        self,
        value: float,
    ) -> None:
        if not 0.0 <= value <= 1.0:
            raise ValueError(
                "Confidence out of range."
            )

        self.confidence = value
        self.touch()

    def activate(
        self,
        amount: float = 1.0,
    ) -> None:
        self.activation = min(
            1.0,
            amount,
        )
        self.touch()

    def deactivate(self) -> None:
        self.activation = 0.0
        self.touch()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "type": self.entity_type.value,
            "name": self.name,
            "properties": self.properties,
            "relations": self.relations,
            "confidence": self.confidence,
            "activation": self.activation,
            "metadata": self.metadata,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
        }
