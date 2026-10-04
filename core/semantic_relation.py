from __future__ import annotations

from dataclasses import dataclass, field
from uuid import uuid4


@dataclass(frozen=True)
class SemanticRelation:
    source_id: str
    target_id: str
    relation_type: str
    confidence: float = .5
    id: str = field(default_factory=lambda: f"SEM-{uuid4().hex[:8]}")

    def __post_init__(self) -> None:
        if not self.source_id.strip() or not self.target_id.strip() or not self.relation_type.strip():
            raise ValueError("Semantic relation fields cannot be empty.")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("Semantic relation confidence must be between 0 and 1.")
