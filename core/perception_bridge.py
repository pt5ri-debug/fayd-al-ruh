"""The perception bridge: a perceived artifact becomes evidence for reasoning.

This is the module the whole phase exists for. It turns a
:class:`perception.schema.PerceptionResult` into the **same** kinds of artifact the
text path already uses, so perception and reasoning share one provenance path
instead of a parallel one:

1. **Ingest** a source through :class:`perception.coordinator.PerceptionCoordinator`.
2. **Extract** its content (already normalized by the coordinator).
3. **Preserve provenance** -- the source path, digest, extraction method, and
   content type travel onto every artifact this bridge creates.
4. **Validate extraction quality** -- the result is checked by the one validator;
   a result that cannot be used is reported, never published.
5. **Identify candidate** entities, relations, and claims -- typed relations are
   read with the project's own :class:`core.relation_extraction.RelationExtractor`.
6. **Submit candidates for knowledge validation** -- each becomes a
   :class:`core.knowledge_validation.KnowledgeClaim` and goes through the
   controller's own validator/integrator.
7. **Store accepted information** through the existing Cognitive Core APIs.
8. **Preserve rejected or uncertain content** for review.

Two honesty rules are structural here, not advisory:

* **Extracted content is not trusted knowledge.** A perceived sentence is turned
  into *evidence* (which a later layer may weigh) and into *candidate* claims
  (which must pass validation). Nothing is written to the graph merely because a
  document said it.
* **A model description is not ground truth.** Content typed ``MODEL`` becomes an
  evidence item that *says so*, and its strength is bounded well below a native
  text layer's -- so a generated description can never outweigh what a source
  actually states.

The bridge is dependency-light and total: every method reports what it did and
never raises for a merely unusable input. It reuses the project's own
``Document``, ``Evidence``, ``EvidenceManager``, ``KnowledgeClaim``,
``RelationExtractor``, and memory APIs -- it reimplements none of them.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any

from perception.schema import (
    ContentType,
    PerceptionResult,
    PerceptionStatus,
)

#: The namespace prefix for documents the bridge adds, so they are easy to spot
#: and to detach again (matching the attachment namespace convention).
PERCEPTION_SOURCE = "perception"
_PERCEPTION_PREFIX = "perception:"

#: A perceived artifact is chunked at this size before it becomes citable
#: documents. Smaller than the bulk corpus default so one relevant passage is not
#: diluted by a whole document.
_CHUNK_CHARS = 900
_CHUNK_OVERLAP = 120

#: The evidence strength assigned to a *native* text layer (a document's own
#: words), to an OCR/ASR reading (a machine's reading of a picture or a voice),
#: and to a model description (an interpretation). The ordering is the honesty
#: rule: a generated description can never outweigh a stated fact.
STRENGTH_NATIVE = 0.8
STRENGTH_OCR = 0.6
STRENGTH_ASR = 0.5
STRENGTH_MODEL = 0.3
STRENGTH_DEFAULT = 0.6


def _slug(text: str) -> str:
    """A short, filesystem-safe token used inside a document id."""
    cleaned = "".join(ch if ch.isalnum() else "-" for ch in text.casefold())
    collapsed = "-".join(part for part in cleaned.split("-") if part)
    return collapsed[:48] or "source"


def _digest(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def evidence_strength(result: PerceptionResult) -> float:
    """The evidence strength a result's content warrants, by how it was obtained.

    Native text is the source's own words; OCR and ASR are a machine's reading;
    a model description is an interpretation. The scale makes that ordering
    explicit so a generated description is never treated as equal to a stated fact.
    """
    base = {
        ContentType.NATIVE: STRENGTH_NATIVE,
        ContentType.STRUCTURED: STRENGTH_NATIVE,
        ContentType.OCR: STRENGTH_OCR,
        ContentType.ASR: STRENGTH_ASR,
        ContentType.MODEL: STRENGTH_MODEL,
    }.get(result.content_type, STRENGTH_DEFAULT)
    # A partial read (a skipped page, a capped file) is a little weaker still.
    if result.status is PerceptionStatus.PARTIAL:
        base = round(base * 0.85, 4)
    return base


def describe_source(result: PerceptionResult) -> str:
    """A readable, honest description of where content came from and how."""
    method = result.extraction_method.value
    origin = result.content_type.value
    bits = [f"{result.source_type.value} ({method})"]
    if origin != "native":
        bits.append(f"{origin}-derived")
    if result.status is not PerceptionStatus.OK:
        bits.append(result.status.value)
    return ", ".join(bits)


@dataclass(frozen=True)
class BridgeOutcome:
    """What one bridge run did, as data (so a caller can report it honestly)."""

    result_id: str
    documents: tuple[str, ...] = ()
    evidence_ids: tuple[str, ...] = ()
    candidate_claims: int = 0
    accepted_claims: int = 0
    rejected_claims: int = 0
    review_claims: int = 0
    memory_ids: tuple[str, ...] = ()
    skipped_reason: str = ""
    notes: tuple[str, ...] = ()

    @property
    def changed(self) -> bool:
        return bool(self.documents or self.evidence_ids or self.accepted_claims or self.memory_ids)

    def describe(self) -> str:
        if self.skipped_reason:
            return f"{self.result_id}: not ingested — {self.skipped_reason}"
        bits = []
        if self.documents:
            bits.append(f"{len(self.documents)} document(s)")
        if self.evidence_ids:
            bits.append(f"{len(self.evidence_ids)} evidence item(s)")
        if self.candidate_claims:
            bits.append(
                f"{self.candidate_claims} candidate(s): {self.accepted_claims} accepted, "
                f"{self.review_claims} for review, {self.rejected_claims} rejected"
            )
        if self.memory_ids:
            bits.append(f"{len(self.memory_ids)} memory record(s)")
        return f"{self.result_id}: " + (", ".join(bits) if bits else "nothing to ingest")


# --------------------------------------------------------------------------- #
# Pure converters: a result -> documents / evidence / claims
# --------------------------------------------------------------------------- #


def to_documents(result: PerceptionResult, *, chunk_chars: int = _CHUNK_CHARS) -> tuple[Any, ...]:
    """Turn a result's content into citable :class:`core.documents.Document` objects.

    The text is chunked with the project's own ``chunk_text`` so it is retrievable
    exactly like the built-in corpus, and each document's metadata carries the
    provenance a citation needs: the source reference, the extraction method, and
    the content type. Returns ``()`` when the result carries no usable content.
    """
    if not result.has_content or result.status in (
        PerceptionStatus.FAILED,
        PerceptionStatus.EMPTY,
        PerceptionStatus.UNSUPPORTED,
    ):
        return ()
    from core.documents import Document
    from core.knowledge_base import chunk_text

    text = result.content.strip()
    if not text:
        text = "\n\n".join(segment.text for segment in result.segments if segment.has_text)
    if not text.strip():
        return ()
    chunks = chunk_text(text, chunk_chars=chunk_chars, overlap=_CHUNK_OVERLAP) or [text]
    slug = _slug(result.source_reference or result.result_id)
    multiple = len(chunks) > 1
    documents = []
    for index, chunk in enumerate(chunks):
        doc_id = f"{_PERCEPTION_PREFIX}{slug}-{index}" if multiple else f"{_PERCEPTION_PREFIX}{slug}"
        title = _document_title(result, index, len(chunks))
        documents.append(
            Document(
                doc_id=doc_id,
                title=title,
                text=chunk,
                source=PERCEPTION_SOURCE,
                metadata={
                    "kind": "perception",
                    "result_id": result.result_id,
                    "source_reference": result.source_reference,
                    "extraction_method": result.extraction_method.value,
                    "content_type": result.content_type.value,
                    "source_digest": result.provenance.get("source_digest", ""),
                    "part": index + 1,
                    "parts": len(chunks),
                },
            )
        )
    return tuple(documents)


def _document_title(result: PerceptionResult, index: int, total: int) -> str:
    """A readable, honest title: the source's name plus how it was read."""
    name = result.source_reference.rsplit("/", 1)[-1] or result.result_id
    label = {
        ContentType.OCR: " (OCR)",
        ContentType.ASR: " (transcript)",
        ContentType.MODEL: " (model description)",
    }.get(result.content_type, "")
    title = f"{name}{label}"
    if total > 1:
        title += f" — part {index + 1}/{total}"
    return title


def to_evidence(result: PerceptionResult, *, source_label: str = "") -> tuple[Any, ...]:
    """Turn a result's content into :class:`core.evidence.Evidence` items.

    One evidence item per content-bearing segment (a page, a transcript, an image
    reading), each carrying its provenance in ``data`` so a reader can see exactly
    which page/segment a claim rests on. The strength comes from
    :func:`evidence_strength`, so a model description is never as strong as a
    native text layer. Returns ``()`` for a result with no usable content.
    """
    if not result.has_content or result.status in (
        PerceptionStatus.FAILED,
        PerceptionStatus.EMPTY,
        PerceptionStatus.UNSUPPORTED,
    ):
        return ()
    from core.evidence import Evidence

    strength = evidence_strength(result)
    source = source_label or describe_source(result)
    segments = [segment for segment in result.segments if segment.has_text]
    items = []
    if not segments and result.content.strip():
        items.append(
            Evidence(
                description=result.content.strip(),
                source=source,
                strength=strength,
                data={
                    "result_id": result.result_id,
                    "source_reference": result.source_reference,
                    "extraction_method": result.extraction_method.value,
                    "content_type": result.content_type.value,
                    "source_digest": result.provenance.get("source_digest", ""),
                    "confidence": result.confidence,
                },
            )
        )
        return tuple(items)
    for segment in segments:
        items.append(
            Evidence(
                description=segment.text.strip(),
                source=source,
                strength=strength,
                data={
                    "result_id": result.result_id,
                    "source_reference": result.source_reference,
                    "extraction_method": result.extraction_method.value,
                    "content_type": segment.content_type.value,
                    "segment": segment.label or f"segment {segment.index}",
                    "segment_index": segment.index,
                    "source_digest": result.provenance.get("source_digest", ""),
                    "confidence": segment.confidence if segment.confidence is not None else result.confidence,
                },
            )
        )
    return tuple(items)


def extract_candidate_claims(result: PerceptionResult, ontology: Any, learner: Any = None):
    """Read typed candidate relations out of a result with the project's extractor.

    Returns a :class:`core.relation_extraction.RelationExtraction` (the same type
    the text path produces), so a perceived document teaches the graph *how* two
    concepts relate, not merely that they co-occur. Read-only: nothing is written
    here; the caller submits each candidate for validation.
    """
    from core.relation_extraction import RelationExtraction, RelationExtractor

    if ontology is None or not result.has_content:
        return RelationExtraction((), 0, 0, 0)
    extractor = RelationExtractor(ontology, learner) if learner is not None else RelationExtractor(ontology)
    return extractor.extract(result.content)


def candidate_claims(result: PerceptionResult, ontology: Any, learner: Any = None):
    """Build :class:`core.knowledge_validation.KnowledgeClaim` candidates from a result.

    Each typed relation the extractor reads becomes a claim carrying the result's
    *evidence* (so the validator weighs it) and the extraction's own confidence.
    The claims are candidates: the caller submits them to the controller, which
    validates before anything is stored. A result with no readable relation yields
    ``()`` -- an honest "nothing to learn" rather than a guess.
    """
    extraction = extract_candidate_claims(result, ontology, learner)
    if not getattr(extraction, "relations", ()):
        return ()
    from core.knowledge_validation import KnowledgeClaim

    evidence = to_evidence(result)
    claims = []
    for relation in extraction.relations:
        claims.append(
            KnowledgeClaim(
                source=relation.source,
                relation_type=relation.relation_type,
                target=relation.target,
                evidence=evidence,
                confidence=float(getattr(relation, "confidence", 0.5)),
                metadata={
                    "result_id": result.result_id,
                    "source_reference": result.source_reference,
                    "extraction_method": result.extraction_method.value,
                    "content_type": result.content_type.value,
                    "pattern": getattr(relation, "pattern", ""),
                },
            )
        )
    return tuple(claims)


# --------------------------------------------------------------------------- #
# The bridge itself
# --------------------------------------------------------------------------- #


class PerceptionBridge:
    """Submits perceived content to the Cognitive Core, Memory, and Learning.

    A bridge holds optional references to the project's own stores; each target is
    independent, so a caller may wire only what it has:

    * ``controller`` -- a :class:`core.cognitive_controller.CognitiveController`
      (for knowledge validation/integration and long-term memory);
    * ``retriever`` -- a :class:`core.rag.RagRetriever` (so a perceived document is
      immediately retrievable and citable);
    * ``graph`` -- a :class:`core.knowledge_graph.KnowledgeGraph` with an
      :class:`core.evidence_manager.EvidenceManager` (to record evidence).

    Nothing is written merely because a document said it: documents become
    retrievable text, evidence is recorded with its provenance, and typed
    relations become **candidates** that the controller validates before storage.
    """

    def __init__(
        self,
        *,
        controller: Any = None,
        retriever: Any = None,
        graph: Any = None,
        evidence_manager: Any = None,
        memory: Any = None,
        ontology: Any = None,
    ) -> None:
        self.controller = controller
        self.retriever = retriever
        self.graph = graph
        self.evidence_manager = evidence_manager
        self.memory = memory
        self.ontology = ontology
        if self.ontology is None:
            self.ontology = getattr(controller, "ontology", None)
        if self.graph is None:
            self.graph = getattr(controller, "graph", None)
        if self.evidence_manager is None:
            self.evidence_manager = getattr(controller, "evidence_manager", None)
        if self.memory is None:
            self.memory = getattr(controller, "long_term_memory", None)

    # -- the one entry point ----------------------------------------------- #

    def ingest(
        self,
        result: PerceptionResult,
        *,
        index: bool = True,
        record_evidence: bool = True,
        submit_claims: bool = True,
        remember: bool = False,
        memory_kind: str = "episodic",
    ) -> BridgeOutcome:
        """Submit a perceived result to every wired target; report what happened.

        ``index`` adds the content to the retriever (making it citable),
        ``record_evidence`` records it in the graph's evidence store,
        ``submit_claims`` sends the typed-relation candidates through validation,
        and ``remember`` captures an episodic memory of the perception (off by
        default: perception is not automatically a memory the user asked for).

        Never raises for a merely unusable input: an unusable result is reported
        through :attr:`BridgeOutcome.skipped_reason`.
        """
        notes: list[str] = []
        if result.status in (PerceptionStatus.FAILED, PerceptionStatus.UNSUPPORTED):
            return BridgeOutcome(
                result_id=result.result_id,
                skipped_reason="; ".join(result.errors) or result.status.value,
            )
        if not result.has_content:
            return BridgeOutcome(
                result_id=result.result_id,
                skipped_reason="the result carries no usable content.",
            )

        # 4. validate extraction quality before publishing anything.
        from perception.validator import validate_result

        verdict = validate_result(result)
        if not verdict.ok:
            return BridgeOutcome(
                result_id=result.result_id,
                skipped_reason=verdict.describe(),
            )
        notes.extend(finding.message for finding in verdict.warnings)

        documents: tuple[str, ...] = ()
        if index and self.retriever is not None:
            documents = self._index(result)
            if not documents:
                notes.append("the retriever was present but accepted no document.")

        evidence_ids: tuple[str, ...] = ()
        if record_evidence and self.evidence_manager is not None:
            evidence_ids = self._record_evidence(result)

        accepted = rejected = review = candidates = 0
        if submit_claims and self.controller is not None:
            accepted, rejected, review, candidates = self._submit_claims(result)

        memory_ids: tuple[str, ...] = ()
        if remember:
            memory_ids = self._remember(result, kind=memory_kind)

        return BridgeOutcome(
            result_id=result.result_id,
            documents=documents,
            evidence_ids=evidence_ids,
            candidate_claims=candidates,
            accepted_claims=accepted,
            rejected_claims=rejected,
            review_claims=review,
            memory_ids=memory_ids,
            notes=tuple(notes),
        )

    # -- targets ----------------------------------------------------------- #

    def _index(self, result: PerceptionResult) -> tuple[str, ...]:
        documents = to_documents(result)
        added: list[str] = []
        for document in documents:
            try:
                self.retriever.index_document(document)
                added.append(document.doc_id)
            except Exception:  # a store that refuses a duplicate must not stop the rest
                continue
        return tuple(added)

    def _record_evidence(self, result: PerceptionResult) -> tuple[str, ...]:
        items = to_evidence(result)
        recorded: list[str] = []
        for item in items:
            try:
                stored = self.evidence_manager.record(
                    item.description,
                    item.source,
                    supports=item.supports,
                    strength=item.strength,
                    kind=_evidence_kind(result),
                    method=_evidence_method(result),
                    reference=result.source_reference,
                    metadata={
                        "result_id": result.result_id,
                        "content_type": result.content_type.value,
                        "extraction_method": result.extraction_method.value,
                    },
                )
                recorded.append(stored.id)
            except Exception:
                continue
        return tuple(recorded)

    def _submit_claims(self, result: PerceptionResult) -> tuple[int, int, int, int]:
        claims = candidate_claims(result, self.ontology)
        accepted = rejected = review = 0
        for claim in claims:
            try:
                record = self.controller.integrate_knowledge(claim)
            except Exception:
                rejected += 1
                continue
            verdict = getattr(record, "verdict", "")
            if getattr(record, "stored", False):
                accepted += 1
            elif verdict in ("needs_review", "needs-review"):
                review += 1
            else:
                rejected += 1
        return accepted, rejected, review, len(claims)

    def _remember(self, result: PerceptionResult, *, kind: str) -> tuple[str, ...]:
        if self.memory is None:
            return ()
        try:
            from core.memory_record import MemoryKind

            memory_kind: MemoryKind | str = MemoryKind(str(kind))
        except Exception:
            memory_kind = "episodic"
        content = result.content.strip()
        if not content:
            return ()
        try:
            record = self.memory.capture(
                content,
                kind=memory_kind,
                summary=f"perceived {describe_source(result)}",
                source="perception",
                source_reference=result.source_reference,
                confidence=result.confidence,
                metadata={
                    "result_id": result.result_id,
                    "content_type": result.content_type.value,
                    "extraction_method": result.extraction_method.value,
                    "source_digest": result.provenance.get("source_digest", ""),
                },
            )
        except Exception:
            return ()
        record_id = getattr(record, "record_id", "")
        return (record_id,) if record_id else ()


def _evidence_kind(result: PerceptionResult):
    from core.evidence_manager import EvidenceKind

    return EvidenceKind.DIRECT


def _evidence_method(result: PerceptionResult):
    """The acquisition method an evidence item records, by how it was obtained."""
    from core.evidence_manager import AcquisitionMethod

    return {
        ContentType.OCR: AcquisitionMethod.OCR,
        ContentType.ASR: AcquisitionMethod.SPEECH,
        ContentType.MODEL: AcquisitionMethod.MODEL,
        ContentType.NATIVE: AcquisitionMethod.INGEST,
    }.get(result.content_type, AcquisitionMethod.MANUAL)


def ingest_result(result: PerceptionResult, **options: Any) -> BridgeOutcome:
    """Build a bridge from the project's default assistant and ingest one result.

    A convenience for the CLI and a quick script: it wires the default grounded
    assistant's retriever and ontology (so a perceived document is immediately
    citable) and the assistant's own learner for candidate relations. A caller with
    a controller in hand should build :class:`PerceptionBridge` directly.
    """
    bridge = default_bridge()
    return bridge.ingest(result, **options)


def default_bridge() -> PerceptionBridge:
    """A bridge over the project's default assistant (retriever + ontology).

    The default assistant is the project's own knowledge corpus, so a perceived
    document becomes citable beside it. The graph/evidence targets are left unset
    (the assistant has none); wire a controller to also validate claims.
    """
    from core.assistant import build_default_assistant

    assistant = build_default_assistant()
    retriever = getattr(assistant, "retriever", None)
    ontology = getattr(retriever, "ontology", None)
    return PerceptionBridge(retriever=retriever, ontology=ontology)


__all__ = [
    "PERCEPTION_SOURCE",
    "STRENGTH_ASR",
    "STRENGTH_DEFAULT",
    "STRENGTH_MODEL",
    "STRENGTH_NATIVE",
    "STRENGTH_OCR",
    "BridgeOutcome",
    "PerceptionBridge",
    "candidate_claims",
    "default_bridge",
    "describe_source",
    "evidence_strength",
    "extract_candidate_claims",
    "ingest_result",
    "to_documents",
    "to_evidence",
]
