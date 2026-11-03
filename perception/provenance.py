"""The perception provenance manager (section 3, section 9).

Provenance is the answer to *where did this come from, and how was it read?* It
is what lets a perceived sentence be cited, audited, and corrected later, and it
is the property the whole project is built around -- nothing is believed without
a traceable origin.

The manager wraps the project's own :class:`core.provenance.ProvenanceLedger` and
:class:`core.provenance.ProvenanceRecord` (reused, not reimplemented) and adds the
perception-specific facts: the source path and its content digest, the extraction
method and content type, the engine name and version, and the parent artifact ids
so a corrected transcript points back at the original audio.

A :class:`PerceptionProvenance` is a small, frozen view of one result's lineage;
:meth:`PerceptionProvenanceManager.record` builds it *and* files a record in the
ledger, so the same lineage is both readable in a result and auditable in the
ledger. Nothing here loads a model or touches the network.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from core.provenance import ProvenanceLedger, ProvenanceRecord, ProvenanceStage

from perception.schema import ContentType, ExtractionMethod, PerceptionResult

#: The extra metadata keys a perception provenance record writes, so a reader and
#: a writer cannot drift on a literal.
KEY_SOURCE_DIGEST = "source_digest"
KEY_EXTRACTION_METHOD = "extraction_method"
KEY_CONTENT_TYPE = "content_type"
KEY_ENGINE = "engine"
KEY_ENGINE_VERSION = "engine_version"


@dataclass(frozen=True)
class PerceptionProvenance:
    """The lineage of one perception result, readable on its own."""

    result_id: str
    source_reference: str
    source_type: str
    extraction_method: str
    content_type: str
    engine: str = ""
    engine_version: str = ""
    source_digest: str = ""
    parent_ids: tuple[str, ...] = ()
    confidence: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)

    def describe(self) -> str:
        bits = [f"{self.source_type} via {self.extraction_method}"]
        if self.engine:
            bits.append(f"engine {self.engine}")
            if self.engine_version:
                bits.append(f"v{self.engine_version}")
        if self.source_digest:
            bits.append(f"digest {self.source_digest[:12]}")
        if self.parent_ids:
            bits.append(f"from {', '.join(self.parent_ids)}")
        return f"{self.source_reference or '(no source)'} — " + ", ".join(bits)

    def to_primitive(self) -> dict[str, Any]:
        return {
            "result_id": self.result_id,
            "source_reference": self.source_reference,
            "source_type": self.source_type,
            "extraction_method": self.extraction_method,
            "content_type": self.content_type,
            "engine": self.engine,
            "engine_version": self.engine_version,
            "source_digest": self.source_digest,
            "parent_ids": list(self.parent_ids),
            "confidence": self.confidence,
            "metadata": dict(self.metadata),
        }


class PerceptionProvenanceManager:
    """Build and file perception provenance, backed by the project's ledger.

    The ledger is optional: a manager with no ledger still builds the readable
    :class:`PerceptionProvenance` (a read-only session touches nothing), and a
    manager with one also files a :class:`ProvenanceRecord` at the ``OBSERVATION``
    stage so the lineage is auditable beside every other artifact.
    """

    def __init__(self, ledger: ProvenanceLedger | None = None) -> None:
        self.ledger = ledger or ProvenanceLedger()

    def record(
        self,
        result: PerceptionResult,
        *,
        engine: str = "",
        engine_version: str = "",
        source_digest: str = "",
        parent_ids: tuple[str, ...] = (),
        context_id: str | None = None,
        file_into_ledger: bool = True,
    ) -> PerceptionProvenance:
        """Build the provenance for ``result`` and (optionally) file it in the ledger."""
        provenance = PerceptionProvenance(
            result_id=result.result_id,
            source_reference=result.source_reference,
            source_type=result.source_type.value,
            extraction_method=result.extraction_method.value,
            content_type=result.content_type.value,
            engine=engine or result.model_name,
            engine_version=engine_version or result.model_version,
            source_digest=source_digest or str(result.provenance.get(KEY_SOURCE_DIGEST, "")),
            parent_ids=tuple(parent_ids) or tuple(result.provenance.get("parent_ids", ())),
            confidence=result.confidence,
            metadata=dict(result.provenance),
        )
        if file_into_ledger and result.source_reference:
            record = ProvenanceRecord(
                entity_id=result.result_id,
                stage=ProvenanceStage.OBSERVATION,
                source=result.source_reference,
                confidence=result.confidence,
                context_id=context_id,
                parent_ids=provenance.parent_ids,
                metadata={
                    KEY_EXTRACTION_METHOD: provenance.extraction_method,
                    KEY_CONTENT_TYPE: provenance.content_type,
                    KEY_SOURCE_DIGEST: provenance.source_digest,
                    KEY_ENGINE: provenance.engine,
                    KEY_ENGINE_VERSION: provenance.engine_version,
                },
            )
            self.ledger.record(record)
        return provenance

    def get(self, result_id: str) -> PerceptionProvenance | None:
        """Read a filed provenance record back as a readable view."""
        record = self.ledger.get(result_id)
        if record is None:
            return None
        metadata = record.metadata
        return PerceptionProvenance(
            result_id=record.entity_id,
            source_reference=record.source,
            source_type=str(metadata.get("source_type", "")),
            extraction_method=str(metadata.get(KEY_EXTRACTION_METHOD, "")),
            content_type=str(metadata.get(KEY_CONTENT_TYPE, "")),
            engine=str(metadata.get(KEY_ENGINE, "")),
            engine_version=str(metadata.get(KEY_ENGINE_VERSION, "")),
            source_digest=str(metadata.get(KEY_SOURCE_DIGEST, "")),
            parent_ids=record.parent_ids,
            confidence=record.confidence,
            metadata=dict(metadata),
        )

    def attach(
        self,
        result: PerceptionResult,
        *,
        engine: str = "",
        engine_version: str = "",
        source_digest: str = "",
        parent_ids: tuple[str, ...] = (),
        context_id: str | None = None,
        file_into_ledger: bool = True,
    ) -> PerceptionResult:
        """Return a copy of ``result`` with its provenance attached and filed.

        A frozen dataclass cannot be mutated in place, so this builds a replacement
        whose ``provenance`` mapping is filled in -- the shape a caller stores.
        ``file_into_ledger=False`` builds the readable lineage without touching the
        ledger, which is what a read-only session asks for.
        """
        provenance = self.record(
            result,
            engine=engine,
            engine_version=engine_version,
            source_digest=source_digest,
            parent_ids=parent_ids,
            context_id=context_id,
            file_into_ledger=file_into_ledger,
        )
        from dataclasses import replace

        merged = dict(result.provenance)
        merged.update(provenance.to_primitive())
        return replace(result, provenance=merged)


def content_type_for(method: ExtractionMethod) -> ContentType:
    """The content type an extraction method implies, for a caller with only a method."""
    if method is ExtractionMethod.PDF_OCR or method is ExtractionMethod.OCR:
        return ContentType.OCR
    if method is ExtractionMethod.SPEECH:
        return ContentType.ASR
    if method is ExtractionMethod.MODEL:
        return ContentType.MODEL
    if method is ExtractionMethod.METADATA:
        return ContentType.METADATA
    if method is ExtractionMethod.NONE:
        return ContentType.NONE
    return ContentType.NATIVE


__all__ = [
    "KEY_CONTENT_TYPE",
    "KEY_ENGINE",
    "KEY_ENGINE_VERSION",
    "KEY_EXTRACTION_METHOD",
    "KEY_SOURCE_DIGEST",
    "PerceptionProvenance",
    "PerceptionProvenanceManager",
    "content_type_for",
]
