from __future__ import annotations

from dataclasses import dataclass

from core.evidence import Evidence


@dataclass(frozen=True)
class EvidenceTrace:
    evidence_ids: tuple[str, ...]
    supporting_ids: tuple[str, ...]
    contradicting_ids: tuple[str, ...]
    sources: tuple[str, ...]

    @classmethod
    def from_evidence(cls, evidence: list[Evidence]) -> "EvidenceTrace":
        return cls(
            tuple(item.id for item in evidence),
            tuple(item.id for item in evidence if item.supports),
            tuple(item.id for item in evidence if not item.supports),
            tuple(dict.fromkeys(item.source for item in evidence)),
        )
