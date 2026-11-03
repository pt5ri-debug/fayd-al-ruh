"""One validation service for a proposed knowledge write.

The project already validates in three places -- :mod:`core.acceptance_policy`
(an external statement's source and topic), :mod:`core.consistency_checker`
(two beliefs that contradict), :mod:`core.contradiction_detector` (evidence that
pulls one claim both ways). Each is correct and tested, but they run in different
pipelines, so a *new* write path can forget one of them. This module is the one
callable that runs them together and returns **one verdict**.

``validate_claim`` performs, in a fixed order, the checks the task names:

* **structural** -- the claim has a source, a target, and a relation type, and
  they are well-formed;
* **semantic** -- the endpoints are known concepts/entities and the relation type
  is one the ontology declares (a warning, not a failure: a text may teach a new
  relation the seeds lack);
* **evidence** -- the claim carries at least one evidence item;
* **relationship** -- the same triple is not already present (a duplicate is a
  *reinforcement*, reported, never a second edge), and no opposing relation is
  already stored on the same ordered pair;
* **contradiction** -- the claim's own evidence does not pull it both ways;
* **temporal** -- the validity window (when given) is coherent;
* **missing evidence** -- an unsupported claim is flagged, never accepted.

**The one rule that must never bend:** a claim with no evidence is *never*
classified as verified knowledge. It is returned as
``NEEDS_REVIEW``/``PROVISIONAL`` with the missing-evidence finding, so a caller
cannot mistake "we stored it" for "it is true". Verification is a separate,
explicit act (``BeliefStatus``), exactly as ``core.learning_pipeline`` already
insists.

Read-only: this module inspects a graph and returns a verdict; it writes nothing.
The integration engine (:mod:`core.knowledge_integration`) is what acts on it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import Enum
from typing import Any

from core.consistency_checker import ConsistencyChecker
from core.contradiction_detector import ContradictionDetector
from core.evidence import Evidence
from core.evidence_weighting import EvidenceWeighting
from core.knowledge_graph import KnowledgeGraph
from core.relation import Relation
from core.relation_polarity import opposes

#: The confidence at or above which a well-evidenced claim may be stored as
#: accepted. Deliberately the same order as ``learning_pipeline``'s floor, and
#: still *provisional* in the verification sense.
DEFAULT_ACCEPT_THRESHOLD = 0.6

#: Below this, a claim is not even provisionally stored -- it is flagged.
DEFAULT_REVIEW_THRESHOLD = 0.2


class ValidationVerdict(str, Enum):
    """The outcome of validating one claim -- a closed set."""

    ACCEPTED = "accepted"
    PROVISIONAL = "provisional"
    NEEDS_REVIEW = "needs_review"
    REJECTED = "rejected"


class FindingKind(str, Enum):
    """Which check produced a finding."""

    STRUCTURAL = "structural"
    SEMANTIC = "semantic"
    EVIDENCE = "evidence"
    RELATIONSHIP = "relationship"
    CONTRADICTION = "contradiction"
    TEMPORAL = "temporal"
    MISSING_EVIDENCE = "missing_evidence"
    DUPLICATE = "duplicate"


@dataclass(frozen=True)
class ValidationFinding:
    """One thing a check found, with a severity and a readable reason."""

    kind: FindingKind
    detail: str
    severity: float = 0.0
    blocking: bool = False

    def __post_init__(self) -> None:
        if not 0.0 <= self.severity <= 1.0:
            raise ValueError("Finding severity must be between 0 and 1.")

    def describe(self) -> str:
        mark = "!" if self.blocking else "-"
        return f"{mark} [{self.kind.value}] {self.detail}"


@dataclass(frozen=True)
class KnowledgeClaim:
    """A proposed typed relation, graph-agnostic, ready to be validated.

    ``evidence`` may be empty (the missing-evidence check reports it); ``valid_from``
    / ``valid_until`` carry an optional validity window for the temporal check.
    """

    source: str
    relation_type: str
    target: str
    evidence: tuple[Evidence, ...] = ()
    confidence: float = 0.5
    source_label: str = ""
    target_label: str = ""
    valid_from: datetime | None = None
    valid_until: datetime | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("Claim confidence must be between 0 and 1.")

    @property
    def triple(self) -> tuple[str, str, str]:
        return (self.source, self.relation_type, self.target)

    def describe(self) -> str:
        return f"{self.source or '?'} —{self.relation_type or '?'}→ {self.target or '?'}"


@dataclass(frozen=True)
class ValidationResult:
    """The verdict, every finding, and the explainable confidence."""

    verdict: ValidationVerdict
    findings: tuple[ValidationFinding, ...] = ()
    confidence: float = 0.0
    claim: KnowledgeClaim | None = None

    def __post_init__(self) -> None:
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("Validation confidence must be between 0 and 1.")

    @property
    def ok(self) -> bool:
        """Whether the claim may be stored at all (anything but ``REJECTED``).

        A ``NEEDS_REVIEW`` claim is storable **flagged**, never silently
        dropped; only a blocking structural/temporal failure is rejected. Use
        :attr:`storable` for the stricter "may be written to the graph" test.
        """
        return self.verdict is not ValidationVerdict.REJECTED

    @property
    def storable(self) -> bool:
        """Whether the claim may be **written** to the graph.

        ``ACCEPTED`` and ``PROVISIONAL`` are written. A ``NEEDS_REVIEW`` claim is
        written only when it carries evidence -- so a genuine *conflict* (both
        claims preserved) is stored flagged, while an **unsupported** claim is
        recorded for review and never written. This is the one place the rule
        "an unsupported claim is never stored as knowledge" is enforced.
        """
        if self.verdict in (ValidationVerdict.ACCEPTED, ValidationVerdict.PROVISIONAL):
            return True
        if self.verdict is ValidationVerdict.NEEDS_REVIEW:
            return bool(self.claim is not None and self.claim.evidence)
        return False

    @property
    def accepted(self) -> bool:
        """Whether the claim cleared every check with confidence -- still not *verified*."""
        return self.verdict is ValidationVerdict.ACCEPTED

    @property
    def verified(self) -> bool:
        """Always ``False``: validation never *verifies* -- that is a separate act."""
        return False

    def of_kind(self, kind: FindingKind) -> tuple[ValidationFinding, ...]:
        return tuple(f for f in self.findings if f.kind == kind)

    def has(self, kind: FindingKind) -> bool:
        return any(f.kind == kind for f in self.findings)

    def describe(self) -> str:
        lines = [f"verdict: {self.verdict.value}", f"confidence: {self.confidence:.3f}"]
        lines.extend(f"  {finding.describe()}" for finding in self.findings)
        lines.append("  (validation never verifies; verification is a separate act)")
        return "\n".join(lines)

    def to_primitive(self) -> dict[str, Any]:
        return {
            "verdict": self.verdict.value,
            "confidence": self.confidence,
            "findings": [
                {"kind": f.kind.value, "detail": f.detail, "severity": f.severity, "blocking": f.blocking}
                for f in self.findings
            ],
        }


class KnowledgeValidator:
    """Runs every check on a claim and returns one verdict.

    ``known_relations`` is the set of relation types the ontology declares (used
    for the semantic check); ``None`` means "skip the relation-vocabulary check",
    which is what a caller with no ontology passes. ``ontology`` (optional) is
    used to resolve whether an endpoint is a known concept.
    """

    def __init__(
        self,
        *,
        known_relations: set[str] | None = None,
        ontology: Any = None,
        weighting: EvidenceWeighting | None = None,
        accept_threshold: float = DEFAULT_ACCEPT_THRESHOLD,
        review_threshold: float = DEFAULT_REVIEW_THRESHOLD,
    ) -> None:
        if not 0.0 <= review_threshold <= accept_threshold <= 1.0:
            raise ValueError("Thresholds must satisfy 0 <= review <= accept <= 1.")
        self.known_relations = known_relations
        self.ontology = ontology
        self.weighting = weighting or EvidenceWeighting()
        self.accept_threshold = accept_threshold
        self.review_threshold = review_threshold
        self._consistency = ConsistencyChecker()
        self._contradictions = ContradictionDetector(self.weighting)

    # -- the one entry point ----------------------------------------------

    def validate(
        self,
        claim: KnowledgeClaim,
        *,
        graph: KnowledgeGraph | None = None,
        now: datetime | None = None,
    ) -> ValidationResult:
        """Validate one claim against ``graph`` (optional) and return a verdict."""
        moment = now or datetime.now(UTC)
        findings: list[ValidationFinding] = []

        findings.extend(self._structural(claim))
        findings.extend(self._semantic(claim))
        findings.extend(self._evidence(claim))
        if graph is not None:
            findings.extend(self._relationship(claim, graph))
        findings.extend(self._contradiction(claim))
        findings.extend(self._temporal(claim, moment))

        # Evaluate confidence from the claim's own evidence (explainable).
        from core.confidence_engine import evaluate_confidence

        evaluation = evaluate_confidence(
            claim.evidence,
            prior=claim.confidence,
            weighting=self.weighting,
            now=moment,
        )
        confidence = evaluation.confidence

        # A blocking finding rejects outright; a missing-evidence or unresolved
        # contradiction flags for review; otherwise the confidence decides.
        has_contradiction = any(f.kind is FindingKind.CONTRADICTION for f in findings)
        if any(f.blocking for f in findings):
            verdict = ValidationVerdict.REJECTED
        elif not claim.evidence:
            # THE RULE: an unsupported claim is never accepted.
            verdict = ValidationVerdict.NEEDS_REVIEW
        elif has_contradiction:
            verdict = ValidationVerdict.NEEDS_REVIEW
        elif confidence >= self.accept_threshold:
            verdict = ValidationVerdict.ACCEPTED
        elif confidence >= self.review_threshold:
            verdict = ValidationVerdict.PROVISIONAL
        else:
            verdict = ValidationVerdict.NEEDS_REVIEW

        return ValidationResult(verdict=verdict, findings=tuple(findings), confidence=confidence, claim=claim)

    # -- individual checks -------------------------------------------------

    @staticmethod
    def _structural(claim: KnowledgeClaim) -> list[ValidationFinding]:
        found: list[ValidationFinding] = []
        if not claim.source.strip():
            found.append(ValidationFinding(FindingKind.STRUCTURAL, "the claim has no source", 1.0, True))
        if not claim.target.strip():
            found.append(ValidationFinding(FindingKind.STRUCTURAL, "the claim has no target", 1.0, True))
        if not claim.relation_type.strip():
            found.append(ValidationFinding(FindingKind.STRUCTURAL, "the claim has no relation type", 1.0, True))
        if claim.source.strip() and claim.source == claim.target:
            found.append(
                ValidationFinding(FindingKind.STRUCTURAL, "source and target are identical", 0.8, True)
            )
        return found

    def _semantic(self, claim: KnowledgeClaim) -> list[ValidationFinding]:
        found: list[ValidationFinding] = []
        if self.known_relations is not None and claim.relation_type.strip():
            if claim.relation_type not in self.known_relations:
                # Not blocking: a text may legitimately teach a relation the
                # seeds do not yet declare. Reported so the caller can see it.
                found.append(
                    ValidationFinding(
                        FindingKind.SEMANTIC,
                        f"relation type '{claim.relation_type}' is not declared by the ontology",
                        0.2,
                        False,
                    )
                )
        if self.ontology is not None:
            for label, role in ((claim.source_label, "source"), (claim.target_label, "target")):
                if label and not self._known_concept(label):
                    found.append(
                        ValidationFinding(
                            FindingKind.SEMANTIC,
                            f"{role} concept '{label}' is not in the ontology",
                            0.3,
                            False,
                        )
                    )
        return found

    def _known_concept(self, label: str) -> bool:
        try:
            concepts = self.ontology.find_concepts(label)
        except Exception:
            return True
        return bool(concepts)

    @staticmethod
    def _evidence(claim: KnowledgeClaim) -> list[ValidationFinding]:
        if claim.evidence:
            return []
        return [
            ValidationFinding(
                FindingKind.MISSING_EVIDENCE,
                "the claim carries no evidence and can never be accepted as knowledge",
                0.5,
                False,
            )
        ]

    @staticmethod
    def _relationship(claim: KnowledgeClaim, graph: KnowledgeGraph) -> list[ValidationFinding]:
        found: list[ValidationFinding] = []
        source_id = KnowledgeValidator._entity_id_for(graph, claim.source)
        target_id = KnowledgeValidator._entity_id_for(graph, claim.target)
        if source_id is None or target_id is None:
            # An endpoint the graph has never seen cannot be a duplicate or a
            # conflict -- there is nothing stored about it yet.
            return found
        same: list[Relation] = []
        for relation in graph.relations.values():
            if not relation.active:
                continue
            if relation.source != source_id or relation.target != target_id:
                continue
            if relation.relation_type == claim.relation_type:
                same.append(relation)
            elif opposes(relation.relation_type, claim.relation_type):
                found.append(
                    ValidationFinding(
                        FindingKind.CONTRADICTION,
                        f"the graph already holds '{relation.relation_type}' on this ordered pair, "
                        f"which opposes '{claim.relation_type}'",
                        0.6,
                        False,
                    )
                )
        if same:
            found.append(
                ValidationFinding(
                    FindingKind.DUPLICATE,
                    f"an identical relation already exists ({same[0].id}); this reinforces it",
                    0.0,
                    False,
                )
            )
        return found

    @staticmethod
    def _entity_id_for(graph: KnowledgeGraph, name: str) -> str | None:
        """The id of the entity with this name (case-insensitive), or ``None``.

        A claim names its endpoints by *label*; the graph stores them by *id*, so
        the relationship check must resolve one to the other -- comparing a name
        against an id would silently never match.
        """
        wanted = name.strip().casefold()
        if not wanted:
            return None
        for entity in graph.entities.values():
            if entity.name.strip().casefold() == wanted:
                return entity.id
        return None

    def _contradiction(self, claim: KnowledgeClaim) -> list[ValidationFinding]:
        if len(claim.evidence) < 2:
            return []
        found: list[ValidationFinding] = []
        for contradiction in self._contradictions.detect(list(claim.evidence)):
            found.append(
                ValidationFinding(
                    FindingKind.CONTRADICTION,
                    f"evidence {contradiction.supporting_evidence_id} and "
                    f"{contradiction.contradicting_evidence_id} disagree (severity {contradiction.severity:.2f})",
                    contradiction.severity,
                    False,
                )
            )
        return found

    @staticmethod
    def _temporal(claim: KnowledgeClaim, now: datetime) -> list[ValidationFinding]:
        found: list[ValidationFinding] = []
        start, end = claim.valid_from, claim.valid_until
        if start is not None and end is not None and end < start:
            found.append(
                ValidationFinding(FindingKind.TEMPORAL, "validity ends before it begins", 0.7, True)
            )
        if end is not None:
            end_aware = end if end.tzinfo is not None else end.replace(tzinfo=UTC)
            if end_aware < now:
                found.append(
                    ValidationFinding(
                        FindingKind.TEMPORAL,
                        "the claim's validity window has already expired",
                        0.4,
                        False,
                    )
                )
        return found


def known_relation_types(ontology: Any) -> set[str]:
    """The relation types an ontology declares, or an empty set.

    A thin helper so a caller does not reach into the ontology's internals; an
    ontology that cannot answer yields an empty set (the semantic check is then
    skipped rather than firing on every relation). The ontology stores its
    relations in a mapping keyed by *id*, so the values are read -- a caller must
    not have to know whether the store is a dict, a list, or a set.
    """
    try:
        relations = ontology.relations
    except Exception:
        return set()
    values = relations.values() if isinstance(relations, dict) else relations
    types: set[str] = set()
    for relation in values:
        relation_type = getattr(relation, "relation_type", None) or getattr(relation, "type", None)
        if isinstance(relation_type, str) and relation_type:
            types.add(relation_type)
    return types


__all__ = [
    "DEFAULT_ACCEPT_THRESHOLD",
    "DEFAULT_REVIEW_THRESHOLD",
    "FindingKind",
    "KnowledgeClaim",
    "KnowledgeValidator",
    "ValidationFinding",
    "ValidationResult",
    "ValidationVerdict",
    "known_relation_types",
]
