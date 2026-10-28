"""Carry learned graph knowledge across sessions.

The project carries a whole persistence layer -- ``KnowledgeStore``,
``Snapshot``, ``Persistence``, ``JsonStore`` -- but it was never wired to the
answer path: the graph an answer *reads* is rebuilt from the seeds on every
process start, and every relation the architecture **learned** during a session
(a typed relation read from a user's sentence, a reinforced link, a belief that
was revised) is thrown away when that process ends. So knowledge Jarvis gained
today was gone tomorrow. That is Gap 8 of ``docs/REASONING_ASSESSMENT.md``.

This module is the bridge. :class:`LearnedKnowledgeStore` captures the graph's
*learned* relations -- and only those, never a curated seed -- as text-free
records keyed by the two concepts' **labels**, and replays them into a later
session's ontology through the project's own :class:`OntologyLearner`, so a
replayed relation is bounded by the same ceiling and the same known-concept rule
as a freshly learned one. Restoring is idempotent: a relation that already exists
is left alone, so a restart never duplicates a link.

Honesty is the whole point:

* Only *learned* relations are carried. A seeded relation is already in every
  session, so capturing it would be dead weight and restoring it could never
  change an answer -- and would blur the line between the curated corpus and what
  the system taught itself.
* Records store **labels**, not internal ids (a ``CON-...`` id is regenerated on
  every start), so a record written by one run resolves in another.
* A record whose concept is unknown to the ontology is **reported as skipped**,
  never invented as a new concept.
* A stored relation can never *override* a seeded one: replay goes through the
  learner, which refuses to touch an explicit relation.
* Nothing here runs on its own. Capturing and restoring are explicit calls, so a
  read-only session is untouched.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from core.ontology_learner import OntologyLearner

if TYPE_CHECKING:
    from core.ontology import Ontology
    from core.persistence import Persistence

#: The one namespace learned relations live under, so a store and a reader can
#: never disagree about where they are.
CARRY_NAMESPACE = "learned-knowledge"

#: The single key holding every carried relation. One document (not one per
#: relation) keeps a restore to a single read, and the record set is small by
#: construction -- only what the system actually learned.
CARRY_KEY = "relations"

#: The format version written with the records, so a future change can migrate
#: rather than misread an old file.
CARRY_VERSION = 1


@dataclass(frozen=True)
class LearnedRelation:
    """One learned relation, keyed by the labels it connects -- never by ids."""

    source: str
    relation_type: str
    target: str
    confidence: float

    def __post_init__(self) -> None:
        if not self.source.strip() or not self.target.strip():
            raise ValueError("A carried relation needs two concept labels.")
        if not self.relation_type.strip():
            raise ValueError("A carried relation needs a relation type.")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("A carried relation confidence must be between 0 and 1.")

    @property
    def key(self) -> tuple[str, str, str]:
        """The case-folded (source, type, target) identity used to de-duplicate."""
        return (self.source.casefold(), self.relation_type.casefold(), self.target.casefold())

    def to_primitive(self) -> dict[str, object]:
        return {
            "source": self.source,
            "relation_type": self.relation_type,
            "target": self.target,
            "confidence": self.confidence,
        }

    @classmethod
    def from_primitive(cls, data: dict[str, object]) -> LearnedRelation:
        confidence = data.get("confidence", 0.5)
        return cls(
            source=str(data["source"]),
            relation_type=str(data["relation_type"]),
            target=str(data["target"]),
            confidence=float(confidence) if isinstance(confidence, (int, float, str)) else 0.5,
        )

    def describe(self) -> str:
        return f"{self.source} --{self.relation_type}--> {self.target} ({self.confidence:.2f})"


@dataclass(frozen=True)
class CarryReport:
    """What one restore did: how many links were replayed, and what was skipped."""

    restored: tuple[LearnedRelation, ...]
    already_present: int = 0
    skipped: tuple[LearnedRelation, ...] = ()

    @property
    def changed(self) -> bool:
        return bool(self.restored)

    @property
    def total(self) -> int:
        return len(self.restored) + self.already_present + len(self.skipped)

    def describe(self) -> str:
        if not self.total:
            return "No learned relations were carried."
        return (
            f"Carried {len(self.restored)} learned relation(s); "
            f"{self.already_present} already present, {len(self.skipped)} skipped."
        )


def capture_learned(
    ontology: Ontology,
    learner: OntologyLearner,
) -> tuple[LearnedRelation, ...]:
    """The graph's learned relations as label-keyed records, seeds excluded.

    A relation counts as learned only when the learner created it (``is_learned``)
    **and** its type is not the seed marker, so a curated relation is never
    captured -- it is already in every session and restoring it could not change an
    answer. Relations whose endpoints are no longer concepts are dropped: a
    record that could not be replayed is not written.
    """
    records: list[LearnedRelation] = []
    seen: set[tuple[str, str, str]] = set()
    for relation_id, relation in ontology.relations.items():
        if not learner.is_learned(relation_id):
            continue
        source = ontology.concepts.get(relation.source_id)
        target = ontology.concepts.get(relation.target_id)
        if source is None or target is None:
            continue
        record = LearnedRelation(
            source=source.label,
            relation_type=relation.relation_type,
            target=target.label,
            confidence=relation.confidence,
        )
        if record.key in seen:
            continue
        seen.add(record.key)
        records.append(record)
    records.sort(key=lambda record: record.key)
    return tuple(records)


def restore_learned(
    ontology: Ontology,
    learner: OntologyLearner,
    records: tuple[LearnedRelation, ...],
) -> CarryReport:
    """Replay label-keyed records into an ontology through the learner.

    Each record is resolved to the concepts its labels name and handed to
    :meth:`OntologyLearner.observe_relation`, so the replay is bounded by the
    learner's own ceiling and refuses to touch a seeded relation. A record whose
    label is unknown to the ontology, or whose pair already has an explicit
    relation, is **skipped and reported** rather than invented. Replaying the same
    records twice restores nothing the second time (the link already exists), so a
    restart never duplicates a relation.
    """
    restored: list[LearnedRelation] = []
    skipped: list[LearnedRelation] = []
    already = 0
    for record in records:
        source = ontology.find_concepts(record.source)
        target = ontology.find_concepts(record.target)
        if not source or not target:
            skipped.append(record)
            continue
        source_id = source[0].id
        target_id = target[0].id
        if source_id == target_id:
            skipped.append(record)
            continue
        existing = learner.relation_by_triple(source_id, record.relation_type, target_id)
        if existing is not None:
            already += 1
            continue
        learning = learner.observe_relation(
            source[0].label,
            target[0].label,
            record.relation_type,
            record.confidence,
        )
        if learning.added or learning.reinforced:
            restored.append(record)
        else:
            # The learner refused it (e.g. an explicit relation on the same pair
            # in the other direction) -- report it rather than pretend.
            skipped.append(record)
    return CarryReport(restored=tuple(restored), already_present=already, skipped=tuple(skipped))


class LearnedKnowledgeStore:
    """Durable storage for the graph's learned relations.

    A thin, explicit wrapper over the project's :class:`Persistence` protocol
    (``save``/``load``): it writes one versioned document of label-keyed records
    and reads it back. It never touches the ontology itself -- ``capture`` and
    ``restore`` do, through the caller's own learner -- so a store can be opened
    without changing anything.
    """

    def __init__(self, persistence: Persistence) -> None:
        self.persistence = persistence

    def save(self, records: tuple[LearnedRelation, ...]) -> None:
        self.persistence.save(
            CARRY_NAMESPACE,
            CARRY_KEY,
            {
                "version": CARRY_VERSION,
                "relations": [record.to_primitive() for record in records],
            },
        )

    def load(self) -> tuple[LearnedRelation, ...]:
        """The stored records, or ``()`` when nothing was stored.

        A record that cannot be read back (a missing field, a bad confidence) is
        dropped rather than crashing the session -- the store is a cache of what
        was learned, not the source of truth for a seed.
        """
        payload = self.persistence.load(CARRY_NAMESPACE, CARRY_KEY)
        if not isinstance(payload, dict):
            return ()
        raw = payload.get("relations")
        if not isinstance(raw, list):
            return ()
        records: list[LearnedRelation] = []
        for item in raw:
            if not isinstance(item, dict):
                continue
            try:
                records.append(LearnedRelation.from_primitive(item))
            except (KeyError, TypeError, ValueError):
                continue
        return tuple(records)

    def capture(self, ontology: Ontology, learner: OntologyLearner) -> tuple[LearnedRelation, ...]:
        """Capture the ontology's learned relations and persist them."""
        records = capture_learned(ontology, learner)
        self.save(records)
        return records

    def restore(self, ontology: Ontology, learner: OntologyLearner) -> CarryReport:
        """Load the stored records and replay them into the ontology."""
        return restore_learned(ontology, learner, self.load())


@dataclass(frozen=True)
class CarryJudgement:
    """One held-out expectation about the knowledge-carry bridge.

    ``kind`` selects what is checked: ``"round-trip"`` that a learned relation is
    captured and replayed into a *fresh* session's ontology, ``"idempotent"`` that
    restoring the same records twice does not duplicate a relation, ``"seed-kept"``
    that a curated seed relation is never captured (and so is never mistaken for
    learned knowledge), and ``"skip-unknown"`` that a record naming an unknown
    concept is skipped rather than invented -- the negative controls that prove
    nothing is fabricated.
    """

    name: str
    kind: str
    source: str = ""
    relation_type: str = ""
    target: str = ""

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("A carry judgement needs a name.")
        if self.kind not in ("round-trip", "idempotent", "seed-kept", "skip-unknown"):
            raise ValueError(f"Unknown carry judgement kind: {self.kind}")


#: The held-out carry catalogue -- with **negative controls**: a curated seed must
#: never be captured as learned, and a record naming an unknown concept must be
#: skipped rather than turned into a concept.
CARRY_JUDGEMENTS: tuple[CarryJudgement, ...] = (
    CarryJudgement(
        "a learned relation survives into a fresh session",
        "round-trip",
        source="waterflood",
        relation_type="increases",
        target="water_saturation",
    ),
    CarryJudgement(
        "restoring the same records twice does not duplicate",
        "idempotent",
        source="waterflood",
        relation_type="increases",
        target="water_saturation",
    ),
    CarryJudgement(
        "a curated seed relation is never captured as learned",
        "seed-kept",
        source="porosity",
        relation_type="increases",
        target="permeability",
    ),
    CarryJudgement(
        "a record naming an unknown concept is skipped",
        "skip-unknown",
        source="waterflood",
        relation_type="increases",
        target="not_a_real_concept_xyz",
    ),
)


def _fresh_ontology() -> Ontology:
    """A seeded ontology, the way the answer path builds one.

    Mirrors ``core.assistant.build_default_assistant``: the domain frame, the
    curriculum, and the petroleum corpus all seed the one graph, so a concept the
    answer path knows (``waterflood``, ``water_saturation``) is known here too.
    """
    from core.curriculum_knowledge import seed_curriculum
    from core.domain_knowledge import seed_ontology
    from core.ontology import Ontology
    from core.petroleum_corpus import seed_petroleum_corpus

    ontology = Ontology()
    seed_ontology(ontology)
    seed_curriculum(ontology)
    seed_petroleum_corpus(ontology)
    return ontology


def knowledge_carry_accuracy(
    judgements: tuple[CarryJudgement, ...] = CARRY_JUDGEMENTS,
) -> float:
    """The share of held-out carry judgements the bridge gets right.

    A judgement counts only when the bridge does exactly what it must: carry a
    learned relation into a fresh session, keep a second restore from duplicating,
    never capture a curated seed, or skip a record naming an unknown concept. So a
    bridge that lost the learned relation, duplicated on restore, captured a seed,
    or invented a concept fails the relevant judgement. Returns ``0.0`` for an
    empty catalogue.
    """
    catalogue = tuple(judgements)
    if not catalogue:
        return 0.0
    correct = 0
    for judgement in catalogue:
        if _judgement_holds(judgement):
            correct += 1
    return correct / len(catalogue)


def _judgement_holds(judgement: CarryJudgement) -> bool:
    """Run one judgement against the real capture/restore path."""
    if judgement.kind == "seed-kept":
        ontology = _fresh_ontology()
        learner = OntologyLearner(ontology)
        records = capture_learned(ontology, learner)
        captured = {(record.source, record.relation_type, record.target) for record in records}
        return (
            judgement.source,
            judgement.relation_type,
            judgement.target,
        ) not in captured

    if judgement.kind == "skip-unknown":
        ontology = _fresh_ontology()
        learner = OntologyLearner(ontology)
        record = LearnedRelation(
            source=judgement.source,
            relation_type=judgement.relation_type,
            target=judgement.target,
            confidence=0.5,
        )
        before = len(ontology.concepts)
        report = restore_learned(ontology, learner, (record,))
        return (
            not report.restored
            and record in report.skipped
            and len(ontology.concepts) == before
        )

    # round-trip and idempotent share the same setup.
    first = _fresh_ontology()
    first_learner = OntologyLearner(first)
    first_learner.observe_relation(
        judgement.source, judgement.target, judgement.relation_type, 0.5
    )
    records = capture_learned(first, first_learner)
    if judgement.kind == "round-trip":
        second = _fresh_ontology()
        second_learner = OntologyLearner(second)
        report = restore_learned(second, second_learner, records)
        if not report.restored:
            return False
        source = second.find_concepts(judgement.source)
        target = second.find_concepts(judgement.target)
        if not source or not target:
            return False
        return (
            second_learner.relation_by_triple(
                source[0].id, judgement.relation_type, target[0].id
            )
            is not None
        )

    # idempotent: the second restore must add nothing.
    second = _fresh_ontology()
    second_learner = OntologyLearner(second)
    restore_learned(second, second_learner, records)
    count = len(second.relations)
    again = restore_learned(second, second_learner, records)
    return not again.restored and len(second.relations) == count


__all__ = [
    "CARRY_JUDGEMENTS",
    "CARRY_KEY",
    "CARRY_NAMESPACE",
    "CARRY_VERSION",
    "CarryJudgement",
    "CarryReport",
    "LearnedKnowledgeStore",
    "LearnedRelation",
    "capture_learned",
    "knowledge_carry_accuracy",
    "restore_learned",
]
