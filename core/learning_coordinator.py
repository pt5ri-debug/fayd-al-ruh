"""The learning coordinator: one governed entry point for the whole learning system.

Phase 3 requires a **learning coordinator** and a **learning event processor**,
and it requires the seven personalities to keep separate experience-based
learning records, separate procedural learning, and separate feedback history --
with Natasha excluded from any aggregated dataset the primary model would use.

This module is that coordinator. It wires the four pipelines
(:mod:`core.knowledge_learning`, :mod:`core.experience_learning`,
:mod:`core.feedback_processing`, :mod:`core.pattern_learning`) over one durable
:class:`~core.learning_store.LearningStore` per owner and exposes one API. It does
**not** re-implement any of them.

**Isolation is structural.** A coordinator is built for exactly one ``owner`` and
resolves that owner's store through :mod:`core.learning_isolation`, so a
cross-personality call is refused *before* any path is resolved. ``learn_from``
(the controlled sharing interface) refuses Natasha for the primary model, and
``dataset_owners`` -- the owners whose records may enter an aggregated dataset --
always excludes her.

**Honest by construction.** Nothing here trusts generated text, promotes a weak
pattern, or claims improvement without a measured comparison; and nothing here
modifies model weights or personality training datasets.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from core.experience_learning import (
    ExperienceLearningPipeline,
    ExperienceLearningResult,
    ExperienceRecord,
)
from core.feedback_processing import FeedbackOutcome, FeedbackProcessor
from core.knowledge_graph import KnowledgeGraph
from core.knowledge_learning import (
    KnowledgeLearningPipeline,
    KnowledgeLearningResult,
)
from core.knowledge_validation import KnowledgeClaim
from core.learning_events import LearningEvent
from core.learning_isolation import (
    MAIN_MODEL_OWNER,
    PRIVATE_OWNERS,
    SHAREABLE_OWNERS,
    LearningAccessError,
    assert_owner,
    is_main_model,
    is_private,
    owner_for,
    store_dir_for,
)
from core.learning_store import LearningStore
from core.long_term_memory import LongTermMemory
from core.memory_management import PersonalityMemoryRegistry
from core.memory_record import MemoryKind
from core.pattern_learning import PatternLearning, PatternReport

#: The file name the durable learning store uses inside an owner's folder.
LEARNING_DB_NAME = "learning.sqlite3"

#: The file name the owner's long-term memory uses (shared with Phase 2).
MEMORY_DB_NAME = "memory.sqlite3"


@dataclass
class CoordinatorStatus:
    """An honest snapshot of one owner's learning state."""

    owner: str
    is_main_model: bool
    private: bool
    events: int = 0
    counts: dict[str, int] = field(default_factory=dict)
    procedures: int = 0
    trusted_procedures: int = 0
    lessons: int = 0
    feedback: int = 0
    version: int = 0

    def describe(self) -> str:
        parts = ", ".join(f"{kind}={count}" for kind, count in sorted(self.counts.items()))
        return (
            f"learning for {self.owner} ({'main model' if self.is_main_model else 'character'}"
            f"{', private' if self.private else ''}): {parts}; "
            f"procedures={self.procedures} (trusted={self.trusted_procedures}); "
            f"lessons={self.lessons}; feedback={self.feedback}; version={self.version}"
        )


class LearningCoordinator:
    """The one owner's learning system: knowledge, experience, feedback, patterns."""

    def __init__(
        self,
        owner: str,
        *,
        graph: KnowledgeGraph | None = None,
        ontology: Any = None,
        learner: Any = None,
        memory: LongTermMemory | None = None,
        store: LearningStore | None = None,
        data_root: str | Path | None = None,
        registry: PersonalityMemoryRegistry | None = None,
    ) -> None:
        self.owner = owner_for(owner)
        assert_owner(self.owner)
        self._registry = registry or PersonalityMemoryRegistry(data_root=data_root)
        base = self._base_dir(data_root)
        self.memory = memory or self._registry.owner_memory(self.owner)
        self.store = store or LearningStore(self.owner, base / LEARNING_DB_NAME)
        self.knowledge = KnowledgeLearningPipeline(
            graph if graph is not None else KnowledgeGraph(),
            ontology=ontology,
            learner=learner,
            store=self.store,
            owner=self.owner,
        )
        self.experience = ExperienceLearningPipeline(self.memory, store=self.store, owner=self.owner)
        self.feedback = FeedbackProcessor(self.memory, store=self.store, owner=self.owner)
        self.patterns = PatternLearning(self.memory, store=self.store, owner=self.owner)

    def _base_dir(self, data_root: str | Path | None) -> Path:
        """The owner's own folder, resolved through the isolation policy."""
        return store_dir_for(self.owner, data_root)

    # -- the four pipelines, through one API ------------------------------
    def learn_knowledge(
        self,
        text: str,
        *,
        source: str = "system",
        source_reference: str = "",
        confidence: float = 0.5,
        claims: Iterable[KnowledgeClaim] | None = None,
    ) -> KnowledgeLearningResult:
        """Learn a knowledge item through the Phase-1 core."""
        return self.knowledge.learn(
            text, source=source, source_reference=source_reference, confidence=confidence, claims=claims
        )

    def learn_claim(
        self, claim: KnowledgeClaim, *, source: str | None = None, source_reference: str = ""
    ) -> KnowledgeLearningResult:
        """Learn one structured claim."""
        return self.knowledge.learn_claim(claim, source=source, source_reference=source_reference)

    def learn_experience(self, experience: ExperienceRecord) -> ExperienceLearningResult:
        """Capture and evaluate one experience through the Phase-2 memory."""
        return self.experience.learn(experience)

    def record_feedback(self, kind: str, subject: str, **kwargs: Any) -> FeedbackOutcome:
        """Process one feedback record (explicit or inferred)."""
        return self.feedback.process(kind, subject, **kwargs)

    def extract_patterns(self, *, limit: int = 500) -> PatternReport:
        """Extract versioned procedures from the stored experiences."""
        return self.patterns.extract(limit=limit)

    # -- controlled sharing ------------------------------------------------
    def learn_from(self, source_owner: str, *, reader: str | None = None) -> tuple[LearningEvent, ...]:
        """Learn from another owner's **trusted** records, under the policy.

        Refuses a private owner (Natasha) *before* resolving any path, and copies
        only the **knowledge** -- the statement and its provenance -- never the
        private exchange (a question/answer pair). Each accepted record becomes a
        **new**, verified memory owned by this coordinator's owner, plus a learning
        event, so the two stores stay separate and the copy is traceable.
        """
        target = owner_for(source_owner)
        actor = reader or self.owner
        if target == self.owner:
            raise LearningAccessError("An owner cannot learn from itself.")
        if is_main_model(actor) and target in PRIVATE_OWNERS:
            raise LearningAccessError(
                f"{target} is private; the main model may not learn from it."
            )
        if not is_main_model(actor):
            raise LearningAccessError(
                "Only the main model may learn from another owner through this interface."
            )
        source_memory = self._registry.owner_memory(target)
        records, _ = source_memory.page(limit=200)
        manager = self._registry.manager_for(self.owner)
        events: list[LearningEvent] = []
        for record in records:
            if not record.trusted:
                continue
            copied = self.memory.capture(
                record.content,
                kind=MemoryKind.SEMANTIC,
                summary=record.summary,
                source="character",
                source_reference=target,
                confidence=record.confidence,
                importance=record.importance,
                metadata={
                    "shared_from": target,
                    "source_record": record.record_id,
                    "shared_knowledge_only": True,
                },
            )
            manager.verify(copied.record_id)
            events.append(
                LearningEvent(
                    kind="knowledge",
                    target=record.summary or record.content[:120],
                    owner=self.owner,
                    source="character",
                    source_reference=target,
                    summary=f"learned from {target}",
                    after=copied.record_id,
                    applied=True,
                    confidence=record.confidence,
                    metadata={"source_owner": target, "source_record": record.record_id},
                )
            )
        if events and self.store is not None:
            version = self.store.next_version()
            from dataclasses import replace

            events = [replace(event, version=str(version)) for event in events]
            self.store.record_many(events)
        return tuple(events)

    # -- reading -----------------------------------------------------------
    def status(self) -> CoordinatorStatus:
        """An honest snapshot of this owner's learning state."""
        procedures = self.patterns.procedures()
        return CoordinatorStatus(
            owner=self.owner,
            is_main_model=is_main_model(self.owner),
            private=is_private(self.owner),
            events=self.store.count() if self.store is not None else 0,
            counts=self.store.counts_by_kind() if self.store is not None else {},
            procedures=len(procedures),
            trusted_procedures=len(self.patterns.trusted_procedures()),
            lessons=len(self.experience.lessons()),
            feedback=len(self.feedback.active()),
            version=self.store.version if self.store is not None else 0,
        )

    def history(self, *, kind: str | None = None, limit: int = 100) -> tuple[LearningEvent, ...]:
        """This owner's learning history, newest first."""
        if self.store is None:
            return ()
        events, _ = self.store.events(kind=kind, limit=limit)
        return tuple(events)

    def rollback(self, version: int, *, reason: str = "") -> tuple[LearningEvent, ...]:
        """Roll this owner's learning back to ``version`` (append-only)."""
        if self.store is None:
            return ()
        return tuple(self.store.rollback(version, reason=reason))

    def dataset_owners(self, *, include_private: bool = False) -> tuple[str, ...]:
        """The owners whose learning may enter an aggregated dataset.

        The primary model's dataset is the main model **plus the shareable
        characters**; Natasha is excluded unless ``include_private`` is explicitly
        passed, and even then this method reports the exclusion honestly through
        :meth:`dataset_summary`.
        """
        owners = [MAIN_MODEL_OWNER, *SHAREABLE_OWNERS]
        if include_private:
            owners.extend(sorted(PRIVATE_OWNERS))
        return tuple(owners)

    def dataset_summary(self) -> str:
        """The one honest line about what an aggregated dataset would contain."""
        excluded = ", ".join(sorted(PRIVATE_OWNERS))
        return (
            f"aggregated learning dataset: {len(self.dataset_owners())} owner(s) "
            f"({MAIN_MODEL_OWNER} + {len(SHAREABLE_OWNERS)} shareable characters); "
            f"excluded: {excluded}"
        )

    def describe(self) -> str:
        return self.status().describe()


class LearningEventProcessor:
    """Process the queued learning events of one or more owners, incrementally.

    The processor is what makes learning *incremental* rather than a full rescan:
    it keeps a cursor per owner (the last event id it processed) and only acts on
    the events after it. Nothing here runs on its own -- a caller drains the queue
    explicitly.
    """

    def __init__(self) -> None:
        self._cursors: dict[str, str] = {}

    def pending(self, coordinator: LearningCoordinator, *, limit: int = 100) -> tuple[LearningEvent, ...]:
        """The events this processor has not yet seen for ``coordinator``."""
        cursor = self._cursors.get(coordinator.owner, "")
        events = coordinator.history(limit=limit)
        if not cursor:
            return tuple(reversed(events))  # oldest first on the first drain
        seen = False
        pending: list[LearningEvent] = []
        for event in reversed(events):  # oldest first
            if seen:
                pending.append(event)
            elif event.event_id == cursor:
                seen = True
        return tuple(pending) if seen else tuple(reversed(events))

    def drain(
        self, coordinator: LearningCoordinator, *, limit: int = 100
    ) -> tuple[LearningEvent, ...]:
        """Advance the cursor past everything currently queued and return it."""
        pending = self.pending(coordinator, limit=limit)
        if pending:
            self._cursors[coordinator.owner] = pending[-1].event_id
        return pending

    def cursor(self, owner: str) -> str:
        return self._cursors.get(owner_for(owner), "")

    def reset(self, owner: str | None = None) -> None:
        """Forget a cursor (or every cursor) so the next drain starts over."""
        if owner is None:
            self._cursors.clear()
        else:
            self._cursors.pop(owner_for(owner), None)


def coordinator_for(
    owner: str,
    *,
    graph: KnowledgeGraph | None = None,
    ontology: Any = None,
    learner: Any = None,
    data_root: str | Path | None = None,
    store: LearningStore | None = None,
) -> LearningCoordinator:
    """Build one owner's coordinator (its own store and memory, resolved safely)."""
    return LearningCoordinator(
        owner,
        graph=graph,
        ontology=ontology,
        learner=learner,
        data_root=data_root,
        store=store,
    )


def build_default_coordinator(
    *,
    graph: KnowledgeGraph | None = None,
    ontology: Any = None,
    learner: Any = None,
    data_root: str | Path | None = None,
) -> LearningCoordinator:
    """The primary model's coordinator (the main model owner)."""
    return LearningCoordinator(
        MAIN_MODEL_OWNER,
        graph=graph,
        ontology=ontology,
        learner=learner,
        data_root=data_root,
    )


__all__ = [
    "LEARNING_DB_NAME",
    "MEMORY_DB_NAME",
    "CoordinatorStatus",
    "LearningCoordinator",
    "LearningEventProcessor",
    "build_default_coordinator",
    "coordinator_for",
]
