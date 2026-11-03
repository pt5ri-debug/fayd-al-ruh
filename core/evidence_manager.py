"""Evidence lifecycle: record, classify, and weigh the support for a claim.

:class:`~core.evidence.Evidence` is the *record*; this module is the *manager*. It
adds the fields and operations the task requires that the bare dataclass leaves to
convention -- and, crucially, keeps them **separate from the claim**:

* **direct vs inferred** -- evidence that was *observed* (a measurement, a cited
  sentence) is kept apart from evidence that was *inferred* (a derivation the
  program drew). An inferred item is never presented as an observation, which is
  the same honesty rule the reasoning layer already applies.
* **acquisition method** -- *how* the item was obtained (``manual``, ``ingest``,
  ``web``, ``inference``, ``system``), recorded so a reader can weigh it.
* **source reference** -- a stable pointer to where it came from (a document id,
  a URL, a session), kept verbatim.
* **reliability** -- a property of the *source*, kept distinct from the item's
  strength and from any claim's confidence, exactly as the confidence engine
  insists.

It stores its metadata **inside the existing** ``Evidence.data`` mapping, so it
needs no change to the dataclass or to the graph's persistence -- an item written
here round-trips through ``to_dict``/the store with everything intact, and an item
written by older code (no ``kind``) is read as ``DIRECT`` with an honest note.

Read-only with respect to claims: attaching evidence to a claim is the
integration engine's job; this manager only produces and classifies the items.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import Enum
from typing import Any

from core.evidence import Evidence
from core.evidence_weighting import EvidenceWeighting
from core.knowledge_graph import KnowledgeGraph

#: The metadata keys this manager writes into ``Evidence.data``. Declared as
#: constants so a reader and a writer cannot drift on a literal.
KEY_KIND = "kind"
KEY_METHOD = "acquisition"
KEY_REFERENCE = "source_reference"
KEY_RELIABILITY = "source_reliability"
KEY_ACQUIRED_AT = "acquired_at"

#: The reliability the manager assigns a source it has no calibration for. It is
#: deliberately mid-range, never 1.0 -- an uncalibrated source is not a certain one.
DEFAULT_RELIABILITY = 0.5


class EvidenceKind(str, Enum):
    """Whether evidence was observed or derived -- kept apart on purpose."""

    DIRECT = "direct"
    INFERRED = "inferred"


class AcquisitionMethod(str, Enum):
    """How an evidence item was obtained -- a closed vocabulary."""

    MANUAL = "manual"
    INGEST = "ingest"
    WEB = "web"
    INFERENCE = "inference"
    SYSTEM = "system"


@dataclass(frozen=True)
class EvidenceRecord:
    """A readable view of one stored evidence item, with its classification."""

    id: str
    description: str
    source: str
    supports: bool
    strength: float
    kind: EvidenceKind
    method: AcquisitionMethod
    reference: str = ""
    reliability: float = DEFAULT_RELIABILITY
    acquired_at: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def is_direct(self) -> bool:
        return self.kind is EvidenceKind.DIRECT

    def describe(self) -> str:
        stance = "supports" if self.supports else "contradicts"
        origin = "observed" if self.is_direct else "inferred"
        return (
            f"[{origin}/{self.method.value}] {self.source} {stance} "
            f"(strength {self.strength:.2f}, reliability {self.reliability:.2f}): {self.description}"
        )


class EvidenceManager:
    """Record evidence into a graph with its classification and reliability.

    ``reliability`` is an optional per-source calibration; a source absent from it
    gets :data:`DEFAULT_RELIABILITY`. The manager writes the classification into
    ``Evidence.data`` and reads it back, so nothing depends on a side table that
    could be lost.
    """

    def __init__(
        self,
        graph: KnowledgeGraph,
        *,
        reliability: dict[str, float] | None = None,
        weighting: EvidenceWeighting | None = None,
    ) -> None:
        self.graph = graph
        self.reliability = dict(reliability or {})
        if any(not 0.0 <= value <= 1.0 for value in self.reliability.values()):
            raise ValueError("Source reliability must be between 0 and 1.")
        self.weighting = weighting or EvidenceWeighting(self.reliability)

    # -- record ------------------------------------------------------------

    def record(
        self,
        description: str,
        source: str,
        *,
        supports: bool = True,
        strength: float = 1.0,
        kind: EvidenceKind = EvidenceKind.DIRECT,
        method: AcquisitionMethod | None = None,
        reference: str = "",
        reliability: float | None = None,
        at: datetime | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> Evidence:
        """Create and store one evidence item; return it.

        The method defaults sensibly from the kind: an inferred item is recorded
        as ``inference``, a direct one as ``manual``. A caller may override it
        (an ingested document is direct but its method is ``ingest``).
        """
        moment = at or datetime.now(UTC)
        resolved_method = method or (AcquisitionMethod.INFERENCE if kind is EvidenceKind.INFERRED else AcquisitionMethod.MANUAL)
        resolved_reliability = reliability if reliability is not None else self.reliability.get(source, DEFAULT_RELIABILITY)
        if not 0.0 <= resolved_reliability <= 1.0:
            raise ValueError("Evidence reliability must be between 0 and 1.")

        data: dict[str, Any] = {
            KEY_KIND: kind.value,
            KEY_METHOD: resolved_method.value,
            KEY_REFERENCE: reference,
            KEY_RELIABILITY: resolved_reliability,
            KEY_ACQUIRED_AT: moment.isoformat(),
        }
        data.update(metadata or {})

        evidence = Evidence(
            description=description,
            source=source,
            supports=supports,
            strength=strength,
            data=data,
        )
        evidence.created_at = moment
        self.graph.add_evidence(evidence)
        return evidence

    def record_inference(self, description: str, *, source: str = "inference", supports: bool = True, strength: float = 0.6, reference: str = "", metadata: dict[str, Any] | None = None) -> Evidence:
        """A convenience for *derived* evidence -- always classified as inferred."""
        return self.record(
            description,
            source,
            supports=supports,
            strength=strength,
            kind=EvidenceKind.INFERRED,
            method=AcquisitionMethod.INFERENCE,
            reference=reference,
            metadata=metadata,
        )

    # -- read --------------------------------------------------------------

    def describe(self, evidence_id: str) -> EvidenceRecord | None:
        """A readable view of one stored item, or ``None`` for an unknown id."""
        evidence = self.graph.get_evidence(evidence_id)
        if evidence is None:
            return None
        return self.as_record(evidence)

    @staticmethod
    def as_record(evidence: Evidence) -> EvidenceRecord:
        """Read an :class:`Evidence` into an :class:`EvidenceRecord`.

        An item written before this manager existed has no ``kind``; it is read as
        ``DIRECT`` with method ``manual`` -- the honest default, since the older
        code only ever recorded what it had in hand.
        """
        data = evidence.data or {}
        kind = _as_enum(EvidenceKind, data.get(KEY_KIND), EvidenceKind.DIRECT)
        method = _as_enum(AcquisitionMethod, data.get(KEY_METHOD), AcquisitionMethod.MANUAL)
        reliability = data.get(KEY_RELIABILITY, DEFAULT_RELIABILITY)
        if not isinstance(reliability, (int, float)):
            reliability = DEFAULT_RELIABILITY
        return EvidenceRecord(
            id=evidence.id,
            description=evidence.description,
            source=evidence.source,
            supports=evidence.supports,
            strength=evidence.strength,
            kind=kind,
            method=method,
            reference=str(data.get(KEY_REFERENCE, "")),
            reliability=float(reliability),
            acquired_at=str(data.get(KEY_ACQUIRED_AT, "")),
            metadata={k: v for k, v in data.items() if k not in {KEY_KIND, KEY_METHOD, KEY_REFERENCE, KEY_RELIABILITY, KEY_ACQUIRED_AT}},
        )

    def for_claim(self, relation_id: str) -> tuple[EvidenceRecord, ...]:
        """Every evidence item attached to a relation, as records."""
        return tuple(self.as_record(item) for item in self.graph.get_relation_evidence(relation_id))

    def by_kind(self, kind: EvidenceKind, *, limit: int = 100) -> tuple[EvidenceRecord, ...]:
        """Every stored item of one kind (direct or inferred)."""
        found = [
            self.as_record(item)
            for item in self.graph.evidence.values()
            if _as_enum(EvidenceKind, (item.data or {}).get(KEY_KIND), EvidenceKind.DIRECT) is kind
        ]
        return tuple(found[:limit])

    def direct_count(self) -> int:
        return len(self.by_kind(EvidenceKind.DIRECT, limit=10_000))

    def inferred_count(self) -> int:
        return len(self.by_kind(EvidenceKind.INFERRED, limit=10_000))

    def weight(self, evidence_id: str) -> float:
        """The calibrated weight of one item (strength x source reliability)."""
        evidence = self.graph.get_evidence(evidence_id)
        if evidence is None:
            raise KeyError(f"Unknown evidence: {evidence_id}")
        return self.weighting.weight(evidence)

    def describe_all(self) -> str:
        return (
            f"EvidenceManager({len(self.graph.evidence)} item(s): "
            f"{self.direct_count()} direct, {self.inferred_count()} inferred)"
        )


def _as_enum(enum_type: Any, value: Any, default: Any) -> Any:
    """Parse an enum from a stored string, falling back to ``default``."""
    if isinstance(value, enum_type):
        return value
    if isinstance(value, str):
        try:
            return enum_type(value)
        except ValueError:
            return default
    return default


__all__ = [
    "DEFAULT_RELIABILITY",
    "KEY_ACQUIRED_AT",
    "KEY_KIND",
    "KEY_METHOD",
    "KEY_REFERENCE",
    "KEY_RELIABILITY",
    "AcquisitionMethod",
    "EvidenceKind",
    "EvidenceManager",
    "EvidenceRecord",
]
