"""Controlled knowledge sharing: the main model learning from six characters.

The main Fayd Al-Ruh model may learn from the knowledge stored in six characters --
Jarvis, Ultron, Vision, Banner, Hakeem, and Cloud -- and **never** from Natasha.
When it learns from a character it does **not** copy that character's record: it
creates a **new record in its own store**, carrying the source character, the
source record id, the source date, and the source's confidence, so the acquisition
is fully traceable and the original file is never written to.

This module is the only component that reads across owners, and it goes through
:func:`core.learning_isolation.assert_read` *before* it builds a path. The
exclusion is therefore enforced in the backend: a caller that asks for Natasha
gets :class:`~core.learning_isolation.LearningAccessError`, the attempt is logged
as a refusal, and no directory is even resolved. The user interface can hide the
button, but hiding it is not what protects the data.

Nothing here imports Tk or loads a model.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from core.character_learning import persona_for
from core.character_learning_store import CharacterLearningStore, main_model_store
from core.learning_audit import EVENT_SHARED, LearningAudit, default_audit
from core.learning_isolation import (
    MAIN_MODEL_OWNER,
    PRIVATE_OWNERS,
    PRIVATE_REASON,
    SHAREABLE_OWNERS,
    LearningAccessError,
    assert_read,
)
from core.learning_records import (
    SOURCE_CHARACTER,
    STATUS_PROVISIONAL,
    LearningRecord,
    new_record,
)

#: How many records one share reads by default, so a large store is bounded.
DEFAULT_SHARE_LIMIT = 100

#: The confidence a shared record starts at when the source record carried none.
DEFAULT_SHARE_CONFIDENCE = 0.5


@dataclass(frozen=True)
class ShareResult:
    """What one share did: the new records the main model wrote, and what it skipped."""

    source_character: str
    imported: tuple[LearningRecord, ...] = ()
    skipped_private: bool = False
    skipped_duplicates: int = 0
    reason: str = ""

    @property
    def count(self) -> int:
        return len(self.imported)

    def describe(self) -> str:
        """One honest line naming what was learned and from whom."""
        if self.skipped_private:
            return f"Refused: {self.source_character} is private ({self.reason})."
        if not self.imported:
            return f"Nothing new to learn from {self.source_character}."
        return (
            f"Learned {self.count} record(s) from {self.source_character} "
            f"into the main model's own store."
        )


@dataclass
class ShareReport:
    """The outcome of sharing from several characters at once."""

    results: list[ShareResult] = field(default_factory=list)

    @property
    def total(self) -> int:
        return sum(result.count for result in self.results)

    def describe(self) -> str:
        lines = [result.describe() for result in self.results]
        lines.append(f"Total: {self.total} record(s) learned into the main model's store.")
        return "\n".join(lines)


class KnowledgeSharing:
    """The controlled knowledge-sharing interface for the main model.

    ``data_root`` lets a caller keep everything under its own data folder (a test,
    or a portable build); the store layout beneath it is still the isolation
    policy's own, so sharing cannot be pointed at a stray directory.
    """

    def __init__(
        self,
        *,
        data_root: Path | None = None,
        audit: LearningAudit | None = None,
    ) -> None:
        self._root = data_root
        self._audit = audit if audit is not None else default_audit()

    # ---------------------------------------------------------------- helpers

    def _store(self, owner: str) -> CharacterLearningStore:
        return CharacterLearningStore(owner, root=self._root, audit=self._audit)

    @property
    def main_store(self) -> CharacterLearningStore:
        """The main model's own store (built for it, under the same root)."""
        if self._root is None:
            return main_model_store(audit=self._audit)
        return CharacterLearningStore(MAIN_MODEL_OWNER, root=self._root, audit=self._audit)

    def shareable_characters(self) -> tuple[str, ...]:
        """The characters the main model may learn from, in the project's order."""
        return SHAREABLE_OWNERS

    def private_characters(self) -> tuple[str, ...]:
        """The characters whose records are never shared (Natasha)."""
        return tuple(sorted(PRIVATE_OWNERS))

    def can_read(self, character: str) -> bool:
        """Whether the main model may read ``character``'s records.

        A pure check (it opens nothing), so a menu can ask before offering an
        action -- but the *enforcement* is in :meth:`learn_from`, which refuses and
        logs regardless of what a caller believes.
        """
        if character in PRIVATE_OWNERS:
            return False
        return character in SHAREABLE_OWNERS

    # -------------------------------------------------------------- learning

    def _share_history(self, character: str) -> bool:
        """Whether ``character`` permits sharing its *conversation history*.

        The isolation policy is the authority: a private character never shares
        anything, and even a shareable one exposes its question/answer exchange
        **only** when its configuration explicitly says so. The check fails
        **closed** -- an unreadable configuration means "do not expose the
        exchange", never "expose it" -- so the knowledge can always be shared while
        the private conversation stays private by default.
        """
        if character in PRIVATE_OWNERS:
            return False
        try:
            from core.character_registry import CharacterRegistry

            config = CharacterRegistry(data_root=self._root).load(character)
        except (ValueError, OSError, KeyError):
            return False
        return bool(config.effective_sharing and config.learning.share_conversation_history)

    def learn_from(
        self,
        character: str,
        *,
        limit: int = DEFAULT_SHARE_LIMIT,
        min_confidence: float = 0.0,
        trusted_only: bool = True,
    ) -> ShareResult:
        """Learn from ``character``'s records into the main model's own store.

        Refuses a private character **before** resolving any path, so Natasha's
        records are never read -- the refusal is logged and returned as a
        ``ShareResult`` with ``skipped_private=True``.

        Each permitted record becomes a **new** record owned by the main model,
        carrying the source character, id, date, and confidence. A record already
        imported (the same source id) is skipped, so sharing twice does not
        duplicate. The source store is opened read-only and is never written.

        The **knowledge** is always shared; the **question/answer exchange** is
        carried over only when the source character's configuration explicitly
        permits it (``share_conversation_history``), so private conversation
        history is never exposed by a plain share.
        """
        if character in PRIVATE_OWNERS:
            self._audit.log_denied(MAIN_MODEL_OWNER, character, PRIVATE_REASON)
            return ShareResult(
                source_character=character, skipped_private=True, reason=PRIVATE_REASON
            )
        try:
            assert_read(MAIN_MODEL_OWNER, character)
        except LearningAccessError as error:
            self._audit.log_denied(MAIN_MODEL_OWNER, character, str(error))
            return ShareResult(source_character=character, skipped_private=True, reason=str(error))

        share_history = self._share_history(character)
        source = self._store(character)
        destination = self.main_store
        already = {
            record.source_record_id for record in destination.all() if record.source_record_id
        }

        imported: list[LearningRecord] = []
        skipped = 0
        for record in source.recent(limit=limit):
            if trusted_only and not record.trusted:
                continue
            if record.confidence < min_confidence:
                continue
            if record.record_id in already:
                skipped += 1
                continue
            imported.append(self._copy(character, record, share_history=share_history))

        if imported:
            destination.append_many(tuple(imported))
            self._audit.log(
                EVENT_SHARED,
                actor=MAIN_MODEL_OWNER,
                owner=character,
                detail=f"learned {len(imported)} record(s) into the main model's own store.",
            )
        return ShareResult(
            source_character=character,
            imported=tuple(imported),
            skipped_duplicates=skipped,
        )

    def learn_from_all(
        self,
        *,
        limit: int = DEFAULT_SHARE_LIMIT,
        min_confidence: float = 0.0,
        trusted_only: bool = True,
    ) -> ShareReport:
        """Learn from every shareable character. Natasha is never included."""
        report = ShareReport()
        for character in SHAREABLE_OWNERS:
            report.results.append(
                self.learn_from(
                    character,
                    limit=limit,
                    min_confidence=min_confidence,
                    trusted_only=trusted_only,
                )
            )
        return report

    def _copy(
        self, character: str, record: LearningRecord, *, share_history: bool = False
    ) -> LearningRecord:
        """A new main-model record carrying the source's full provenance.

        The knowledge is kept verbatim (never paraphrased), the confidence is the
        source's own, and the record starts **provisional** -- learning from a
        character is not the same as verifying, so the main model's copy is never
        born ``verified``.

        The **question/answer exchange** is carried over only when the source
        explicitly permits sharing its conversation history (``share_history``);
        otherwise the copy keeps the extracted knowledge and the provenance while
        the private exchange stays behind. This is what stops a plain share from
        exposing a private conversation.
        """
        persona = persona_for(character)
        return new_record(
            MAIN_MODEL_OWNER,
            source_type=SOURCE_CHARACTER,
            source_reference=f"{persona.name} ({character}) record {record.record_id}",
            question=record.question if share_history else "",
            answer=record.answer if share_history else "",
            knowledge=record.knowledge or record.answer,
            confidence=record.confidence or DEFAULT_SHARE_CONFIDENCE,
            verification=STATUS_PROVISIONAL,
            learning_status=STATUS_PROVISIONAL,
            statements=record.statements,
            evidence=record.evidence,
            source_character=character,
            source_record_id=record.record_id,
            source_date=record.created_at,
            note=(
                f"learned from {persona.name} through the knowledge-sharing interface"
                + (", with the exchange." if share_history else "; the private exchange was not shared.")
            ),
            metadata={
                "source_status": record.learning_status,
                "source_type": record.source_type,
                "history_shared": share_history,
            },
        )


def shareable_characters() -> tuple[str, ...]:
    """The characters the main model may learn from (a module-level convenience)."""
    return SHAREABLE_OWNERS


def private_characters() -> tuple[str, ...]:
    """The characters whose records are never shared."""
    return tuple(sorted(PRIVATE_OWNERS))


__all__ = [
    "DEFAULT_SHARE_CONFIDENCE",
    "DEFAULT_SHARE_LIMIT",
    "KnowledgeSharing",
    "ShareReport",
    "ShareResult",
    "private_characters",
    "shareable_characters",
]
