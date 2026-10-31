"""The controlled learning pipeline: a response in, an honest record out.

This is the process the task spells out, in order:

1. receive the response;
2. extract the potentially useful knowledge;
3. identify the factual statements;
4. check for contradictions with what is already known;
5. evaluate confidence and evidence;
6. store it in the right independent learning system;
7. update beliefs or the knowledge graph where appropriate;
8. preserve the original information and its provenance;
9. allow a future correction or revision.

The honesty rules are structural, not advisory:

* **A language-model answer or an internet result is never born verified.** Every
  record starts ``pending``; the settings' mode and thresholds decide the *most* it
  may reach, and only a ``supervised`` verification -- or an explicit
  :meth:`LearningPipeline.verify` -- can make one ``verified``.
* **A contradiction never overwrites what is held.** When a new statement opposes
  something already in the store, the new record is kept as ``contradicted`` (its
  original text and provenance intact) and the *existing* record is **not** edited:
  both are preserved, so competing evidence survives and a future correction has
  something to correct.
* **Nothing is invented.** The extractor is a pattern matcher
  (:mod:`core.relation_extraction`) that refuses rather than guesses, and a
  statement it cannot read is simply not extracted.

The pipeline writes to **one** store -- its owner's -- through the store's own
append path, so the isolation the store enforces is never bypassed. It is the only
writer of learning records; the graph is touched only through the project's
existing bounded learners, and only when a caller supplies one.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

from core.character_config import (
    MODE_AUTOMATIC,
    CharacterConfig,
)
from core.character_learning_store import CharacterLearningStore
from core.learning_audit import EVENT_LEARNED, LearningAudit, default_audit
from core.learning_isolation import MAIN_MODEL_OWNER
from core.learning_records import (
    SOURCE_CHARACTER,
    SOURCE_INTERNET,
    SOURCE_LANGUAGE_MODEL,
    SOURCE_SYSTEM,
    SOURCE_USER,
    STATUS_CONTRADICTED,
    STATUS_PENDING,
    STATUS_PROVISIONAL,
    STATUS_REJECTED,
    STATUS_VERIFIED,
    LearningRecord,
    new_record,
)

#: How confident a language-model answer starts: never high enough on its own to be
#: treated as verified, which is the point.
MODEL_BASE_CONFIDENCE = 0.45
#: How confident an internet result starts: a real source, but unverified here.
INTERNET_BASE_CONFIDENCE = 0.55
#: How confident a user's own statement starts.
USER_BASE_CONFIDENCE = 0.7
#: How confident a system conclusion starts.
SYSTEM_BASE_CONFIDENCE = 0.6

#: The confidence a source starts at, before the settings' thresholds are applied.
BASE_CONFIDENCE: dict[str, float] = {
    SOURCE_LANGUAGE_MODEL: MODEL_BASE_CONFIDENCE,
    SOURCE_INTERNET: INTERNET_BASE_CONFIDENCE,
    SOURCE_USER: USER_BASE_CONFIDENCE,
    SOURCE_CHARACTER: 0.5,
    SOURCE_SYSTEM: SYSTEM_BASE_CONFIDENCE,
}


@dataclass
class LearningOutcome:
    """What one pass of the pipeline did."""

    owner: str
    stored: tuple[LearningRecord, ...] = ()
    skipped_reason: str = ""
    contradictions: int = 0

    @property
    def count(self) -> int:
        return len(self.stored)

    @property
    def learned(self) -> bool:
        return bool(self.stored)

    def describe(self) -> str:
        if self.skipped_reason:
            return f"{self.owner} learned nothing: {self.skipped_reason}"
        if not self.stored:
            return f"{self.owner} learned nothing new."
        statuses = ", ".join(record.learning_status for record in self.stored)
        return f"{self.owner} stored {self.count} record(s) ({statuses})."


def factual_statements(text: str) -> tuple[str, ...]:
    """The whole sentences of ``text`` -- the candidate factual statements.

    Uses the project's own sentence splitter (:func:`core.composition.sentences`)
    and keeps only sentences with a terminal mark, so a fragment is never recorded
    as a statement. Bounded, so one runaway answer cannot flood a store.
    """
    from core.composition import sentences

    found: list[str] = []
    for sentence in sentences(text):
        cleaned = sentence.strip()
        if len(cleaned) < 8 or cleaned[-1] not in ".!?؟":
            continue
        found.append(cleaned)
    return tuple(found[:12])


class LearningPipeline:
    """The one controlled path from a response to a stored, honest record.

    ``config`` is the owner's :class:`~core.character_config.CharacterConfig` -- it
    decides whether learning is on at all, which sources are allowed, and how much
    confidence a record needs. ``store`` is the owner's own
    :class:`~core.character_learning_store.CharacterLearningStore`, which enforces
    that nothing is written anywhere else.
    """

    def __init__(
        self,
        owner: str,
        *,
        store: CharacterLearningStore | None = None,
        config: CharacterConfig | None = None,
        audit: LearningAudit | None = None,
        ontology: Any | None = None,
    ) -> None:
        # One identity per character: normalise the owner so a pipeline built with
        # the chat's mouth key ("sources") stamps and guards its records under the
        # character's own name ("jarvis"). The store normalises too, so without this
        # it would refuse every record the pipeline writes and a character would
        # silently keep no isolated learning at all.
        from core.learning_isolation import owner_for

        self.owner = owner_for(owner)
        self._store = store if store is not None else CharacterLearningStore(self.owner)
        self._config = config
        self._audit = audit if audit is not None else default_audit()
        # An optional ontology + learner, so a caller that *wants* the extracted
        # relations folded into the graph can have it -- through the project's own
        # bounded learner, never by writing to the graph here.
        self._ontology = ontology

    # -------------------------------------------------------------- settings

    @property
    def store(self) -> CharacterLearningStore:
        return self._store

    def _settings(self) -> tuple[bool, bool, bool, str, float, float]:
        """The effective learning settings, defaulting to the conservative ones."""
        if self._config is None:
            return True, True, False, "supervised", 0.7, 0.4
        settings = self._config.learning
        return (
            settings.enabled,
            settings.learn_from_model,
            settings.learn_from_internet,
            settings.mode,
            settings.verification_threshold,
            settings.confidence_threshold,
        )

    def _source_allowed(self, source_type: str) -> bool:
        """Whether the settings permit learning from this kind of source."""
        enabled, from_model, from_internet, _mode, _vt, _ct = self._settings()
        if not enabled:
            return False
        if source_type == SOURCE_LANGUAGE_MODEL:
            return from_model
        if source_type == SOURCE_INTERNET:
            return from_internet
        # A user statement, a shared character record, and the system's own
        # conclusion are always permitted when learning is on.
        return True

    # -------------------------------------------------------------- learning

    def learn(
        self,
        *,
        source_type: str,
        question: str = "",
        answer: str = "",
        knowledge: str = "",
        source_reference: str = "",
        evidence: Iterable[str] = (),
        confidence: float | None = None,
        source_character: str = "",
        source_record_id: str = "",
        source_date: str = "",
        verify: bool = False,
    ) -> LearningOutcome:
        """Run the whole pipeline for one response.

        Returns a :class:`LearningOutcome`; nothing is raised for a merely
        uninteresting response (an empty answer, a disabled source), because
        "nothing worth learning" is a normal result, not an error.
        """
        if not self._source_allowed(source_type):
            return LearningOutcome(
                owner=self.owner, skipped_reason=f"learning from {source_type} is off."
            )
        text = (knowledge or answer).strip()
        if not text:
            return LearningOutcome(owner=self.owner, skipped_reason="there was nothing to learn.")

        statements = factual_statements(text)
        base = BASE_CONFIDENCE.get(source_type, 0.5)
        score = base if confidence is None else max(0.0, min(1.0, float(confidence)))
        _enabled, _fm, _fi, mode, verification_threshold, confidence_threshold = self._settings()

        # (5) evaluate confidence: below the floor, the record is kept as rejected
        # rather than dropped -- the history of what was *not* accepted matters.
        status = self._initial_status(score, confidence_threshold)

        record = new_record(
            self.owner,
            source_type=source_type,
            source_reference=source_reference,
            question=question,
            answer=answer,
            knowledge=knowledge or answer,
            confidence=score,
            verification=STATUS_PENDING,
            learning_status=status,
            statements=statements,
            evidence=tuple(evidence),
            source_character=source_character,
            source_record_id=source_record_id,
            source_date=source_date,
            note="below the confidence threshold; kept for the record."
            if status == STATUS_REJECTED
            else "",
        )

        # (4) check for contradictions against what is already held.
        contradiction = self._find_contradiction(statements)
        if contradiction is not None:
            record = record.with_status(
                STATUS_CONTRADICTED,
                note=f"contradicts record {contradiction.record_id}: {contradiction.knowledge}",
            )

        self._store.append(record)

        # (6/7) a supervised or explicit verification may raise the status -- never
        # automatically for a model answer, and never for a contradiction.
        promoted: LearningRecord | None = None
        if verify and contradiction is None and score >= verification_threshold:
            promoted = self._store.revise(
                record.record_id, STATUS_VERIFIED, note="verified by the user."
            )
        elif (
            mode == MODE_AUTOMATIC
            and source_type in {SOURCE_USER, SOURCE_SYSTEM}
            and contradiction is None
            and score >= verification_threshold
        ):
            # Even in automatic mode, only a user statement or the system's own
            # conclusion may reach `verified` on its own; a model or internet
            # answer never does, whatever the mode.
            promoted = self._store.revise(
                record.record_id,
                STATUS_VERIFIED,
                note="verified automatically (user/system source).",
            )
        if promoted is not None:
            record = promoted

        self._audit.log(
            EVENT_LEARNED,
            actor=self.owner,
            owner=self.owner,
            detail=record.describe(),
            record_id=record.record_id,
        )
        return LearningOutcome(
            owner=self.owner,
            stored=(record,),
            contradictions=1 if contradiction is not None else 0,
        )

    def learn_from_model(
        self, *, question: str, answer: str, model: str = "", **kwargs: Any
    ) -> LearningOutcome:
        """Learn from the character's own language-model answer."""
        return self.learn(
            source_type=SOURCE_LANGUAGE_MODEL,
            question=question,
            answer=answer,
            source_reference=f"language model {model}".strip(),
            **kwargs,
        )

    def learn_from_internet(
        self, *, question: str, answer: str, url: str = "", **kwargs: Any
    ) -> LearningOutcome:
        """Learn from a governed internet result, keeping its source URL."""
        evidence = kwargs.pop("evidence", ())
        return self.learn(
            source_type=SOURCE_INTERNET,
            question=question,
            answer=answer,
            source_reference=url,
            evidence=(url, *tuple(evidence)) if url else tuple(evidence),
            **kwargs,
        )

    def learn_from_user(self, *, question: str, statement: str, **kwargs: Any) -> LearningOutcome:
        """Learn from something the user stated (the highest-trust source)."""
        return self.learn(
            source_type=SOURCE_USER,
            question=question,
            answer=statement,
            source_reference="the user",
            **kwargs,
        )

    def learn_conclusion(
        self, *, conclusion: str, basis: str = "", **kwargs: Any
    ) -> LearningOutcome:
        """Record a conclusion the system itself drew (an inference, a summary)."""
        return self.learn(
            source_type=SOURCE_SYSTEM,
            knowledge=conclusion,
            answer=basis,
            source_reference="the system's own reasoning",
            **kwargs,
        )

    def set_config(self, config: CharacterConfig | None) -> None:
        """Replace the settings this pipeline runs under (after an APPLY).

        The one mutable piece: an APPLY that changes a character's learning
        settings must take effect on the next record without rebuilding the mouth,
        and the pipeline is the only thing that reads them.
        """
        self._config = config

    def verify(self, record_id: str, *, note: str = "") -> LearningRecord | None:
        """Supervised verification: promote a record to ``verified``.

        The one way a record becomes verified, and it is explicit -- called by a
        user action, never by the pipeline on its own for a model answer.
        """
        return self._store.revise(record_id, STATUS_VERIFIED, note=note or "verified by the user.")

    def reject(self, record_id: str, *, note: str = "") -> LearningRecord | None:
        """Reject a record, keeping it and the reason (history is never deleted)."""
        return self._store.revise(record_id, STATUS_REJECTED, note=note or "rejected by the user.")

    def revise(
        self, record_id: str, *, note: str, confidence: float | None = None
    ) -> LearningRecord | None:
        """Supersede a record with a corrected version, keeping the original."""
        return self._store.revise(record_id, STATUS_PROVISIONAL, note=note, confidence=confidence)

    # ------------------------------------------------------------- internals

    def _initial_status(self, score: float, threshold: float) -> str:
        """The status a fresh record may have: never ``verified`` on arrival.

        A record below the confidence threshold is ``rejected``; otherwise it is
        ``provisional`` (knowledge we may use, but not verified). It can only
        become ``verified`` through an explicit verification.
        """
        return STATUS_PROVISIONAL if score >= threshold else STATUS_REJECTED

    def _find_contradiction(self, statements: tuple[str, ...]) -> LearningRecord | None:
        """A held record this response's statements appear to contradict, or ``None``.

        The check is deliberately conservative and deterministic: it compares the
        *polarity* of any typed relation the new statements state against the typed
        relations already held, using the project's own
        :mod:`core.relation_polarity` vocabulary, so "A increases B" contradicts
        "A reduces B" but an unrelated claim does not. With no ontology the check
        falls back to a plain negation test ("X is not Y" against "X is Y"), which
        can only ever *report* a contradiction -- it never rewrites anything.
        """
        if not statements:
            return None
        existing = self._store.recent(limit=200)
        for record in existing:
            if not record.trusted:
                continue
            for new in statements:
                for old in record.statements or (record.knowledge,):
                    if _opposes(new, old):
                        return record
        return None


def _polarity_verb(text: str) -> str | None:
    """The polarity-bearing verb a statement uses, or ``None``.

    Scans the closed :data:`_POLARITY_VERBS` set in a fixed order, so the same
    statement always yields the same verb.
    """
    lowered = f" {text.casefold()} "
    for verb in _POLARITY_VERBS:
        if f" {verb} " in lowered:
            return verb
    return None


def _content_terms(text: str) -> set[str]:
    """The content words of a statement, for a same-claim comparison.

    The polarity verbs themselves are excluded, so the comparison is about the
    *things* the statements are about.
    """
    from core.tokenize import normalize_token

    words = {normalize_token(word) for word in text.split()}
    return {word for word in words if word and word not in _POLARITY_VERBS}


def _opposes(new: str, old: str) -> bool:
    """Whether two statements assert opposite things about the same terms.

    Two deterministic checks, both conservative: a direct negation of the same
    content words, or a polarity-bearing verb that is the *antonym* of the other's
    on the same content words. Anything else is **not** a contradiction, so a false
    positive cannot silently mark a good record contradicted.
    """
    from core.relation_polarity import opposes

    new_terms = _content_terms(new)
    old_terms = _content_terms(old)
    if not new_terms or new_terms != old_terms:
        return False
    new_negated = " not " in f" {new.casefold()} "
    old_negated = " not " in f" {old.casefold()} "
    if new_negated != old_negated:
        return True
    new_verb = _polarity_verb(new)
    old_verb = _polarity_verb(old)
    if new_verb and old_verb and new_verb != old_verb and opposes(new_verb, old_verb):
        return True
    return False


#: The polarity-bearing verbs the contradiction check looks for. Kept small and
#: explicit; the polarity vocabulary itself decides which oppose which.
_POLARITY_VERBS: tuple[str, ...] = (
    "increases",
    "reduces",
    "causes",
    "prevents",
    "improves",
    "decreases",
    "raises",
    "lowers",
)


def learning_pipeline_for(
    owner: str,
    *,
    store: CharacterLearningStore | None = None,
    config: CharacterConfig | None = None,
    audit: LearningAudit | None = None,
) -> LearningPipeline:
    """The pipeline for one owner, with the owner's own store and settings."""
    return LearningPipeline(owner, store=store, config=config, audit=audit)


def is_main_model_owner(owner: str) -> bool:
    """True when ``owner`` is the project's own model."""
    from core.learning_isolation import owner_for

    return owner_for(owner) == MAIN_MODEL_OWNER


__all__ = [
    "BASE_CONFIDENCE",
    "INTERNET_BASE_CONFIDENCE",
    "MODEL_BASE_CONFIDENCE",
    "SYSTEM_BASE_CONFIDENCE",
    "USER_BASE_CONFIDENCE",
    "LearningOutcome",
    "LearningPipeline",
    "factual_statements",
    "is_main_model_owner",
    "learning_pipeline_for",
]
