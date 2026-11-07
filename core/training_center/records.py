"""The small durable records the Training Center keeps.

Two records live here, both plain and serialisable so a window, a CLI, and a
test can share them:

* :class:`LearningExample` -- one question/answer pair the Training Center
  gathered, with the **review status** that governs whether it may ever reach a
  dataset. Nothing is trained on an example whose status is not approved, and the
  status is the task's own vocabulary (pending / approved / rejected / needs
  review / corrected / exported / used in training).
* :class:`HistoryEntry` -- one line of the durable history: what happened, when,
  to what, and with what outcome.

Both are dependency-free and JSON-safe. The store wrappers persist them through
the project's own :class:`storage.json_store.JsonStore`, so a corrupt file reads
as empty rather than crashing a window.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

#: The review status of one learning example. The order is the lifecycle a
#: reviewer moves an example through, but any transition is allowed -- the point
#: is that the *status is explicit*, so generated data is never silently trained
#: on.
STATUS_PENDING = "pending"
STATUS_APPROVED = "approved"
STATUS_REJECTED = "rejected"
STATUS_NEEDS_REVIEW = "needs_review"
STATUS_CORRECTED = "corrected"
STATUS_EXPORTED = "exported"
STATUS_USED_IN_TRAINING = "used_in_training"

LEARNING_STATUSES: tuple[str, ...] = (
    STATUS_PENDING,
    STATUS_APPROVED,
    STATUS_REJECTED,
    STATUS_NEEDS_REVIEW,
    STATUS_CORRECTED,
    STATUS_EXPORTED,
    STATUS_USED_IN_TRAINING,
)

#: The statuses that may be exported to a dataset. A corrected example counts as
#: approved (a human fixed it); an exported or already-used one may be exported
#: again without harm.
APPROVED_STATUSES: frozenset[str] = frozenset(
    {STATUS_APPROVED, STATUS_CORRECTED, STATUS_EXPORTED, STATUS_USED_IN_TRAINING}
)

STATUS_LABELS: dict[str, str] = {
    STATUS_PENDING: "Pending",
    STATUS_APPROVED: "Approved",
    STATUS_REJECTED: "Rejected",
    STATUS_NEEDS_REVIEW: "Needs review",
    STATUS_CORRECTED: "Corrected",
    STATUS_EXPORTED: "Exported",
    STATUS_USED_IN_TRAINING: "Used in training",
}

#: The same status names in Arabic, so the bilingual surface never invents one.
STATUS_LABELS_AR: dict[str, str] = {
    STATUS_PENDING: "قيد الانتظار",
    STATUS_APPROVED: "معتمد",
    STATUS_REJECTED: "مرفوض",
    STATUS_NEEDS_REVIEW: "يحتاج مراجعة",
    STATUS_CORRECTED: "مُصحَّح",
    STATUS_EXPORTED: "تم التصدير",
    STATUS_USED_IN_TRAINING: "استُخدم في التدريب",
}


def _now() -> str:
    """An ISO-8601 UTC timestamp, second precision."""
    return datetime.now(UTC).replace(microsecond=0).isoformat()


def _digest(text: str) -> str:
    return hashlib.blake2b(text.encode("utf-8"), digest_size=8).hexdigest()


def detect_language(text: str) -> str:
    """``ar`` when the text carries Arabic letters, else ``en``.

    Reuses the project's own script check so the two agree; a text with no
    letters at all is reported as English, which is the project's default.
    """
    for char in text:
        if "\u0600" <= char <= "\u06ff":
            return "ar"
    return "en"


@dataclass
class LearningExample:
    """One question/answer pair with its provenance and its review status."""

    question: str
    answer: str
    status: str = STATUS_PENDING
    #: Where the answer came from: ``gemini``, ``local_model``, ``character``,
    #: ``document``, ``manual``. Never invented -- the caller states it.
    provider: str = ""
    model: str = ""
    topic: str = ""
    language: str = ""
    confidence: float = 0.0
    #: Citation labels (a document title, a URL) the answer rests on.
    citations: tuple[str, ...] = ()
    #: True only when a human explicitly marked the answer verified.
    verified: bool = False
    created_at: str = ""
    updated_at: str = ""
    note: str = ""
    example_id: str = ""

    def __post_init__(self) -> None:
        if not self.question.strip() and not self.answer.strip():
            raise ValueError("A learning example needs a question or an answer.")
        if self.status not in LEARNING_STATUSES:
            raise ValueError(
                f"Unknown learning status {self.status!r}; expected one of {LEARNING_STATUSES}."
            )
        if not self.language:
            self.language = detect_language(f"{self.question} {self.answer}")
        if not self.created_at:
            self.created_at = _now()
        if not self.updated_at:
            self.updated_at = self.created_at
        if not self.example_id:
            self.example_id = _digest(f"{self.provider}\x00{self.question}\x00{self.answer}")

    @property
    def approved(self) -> bool:
        """True when the example may be exported to a dataset."""
        return self.status in APPROVED_STATUSES

    @property
    def training_text(self) -> str:
        """The example rendered as one training-text line (question then answer)."""
        question = self.question.strip()
        answer = self.answer.strip()
        if question and answer:
            return f"{question}\n{answer}"
        return question or answer

    def with_status(self, status: str, *, note: str = "") -> LearningExample:
        """A *copy* with a new status, so the original is never mutated."""
        return LearningExample(
            question=self.question,
            answer=self.answer,
            status=status,
            provider=self.provider,
            model=self.model,
            topic=self.topic,
            language=self.language,
            confidence=self.confidence,
            citations=self.citations,
            verified=self.verified or (status == STATUS_APPROVED),
            created_at=self.created_at,
            updated_at=_now(),
            note=note or self.note,
            example_id=self.example_id,
        )

    def with_correction(self, answer: str, *, note: str = "") -> LearningExample:
        """A *copy* carrying a corrected answer, marked ``corrected``."""
        return LearningExample(
            question=self.question,
            answer=answer,
            status=STATUS_CORRECTED,
            provider=self.provider,
            model=self.model,
            topic=self.topic,
            language=self.language,
            confidence=self.confidence,
            citations=self.citations,
            verified=self.verified,
            created_at=self.created_at,
            updated_at=_now(),
            note=note or "corrected by a reviewer",
            example_id=self.example_id,
        )

    def to_primitive(self) -> dict[str, Any]:
        return {
            "answer": self.answer,
            "citations": list(self.citations),
            "confidence": self.confidence,
            "created_at": self.created_at,
            "example_id": self.example_id,
            "language": self.language,
            "model": self.model,
            "note": self.note,
            "provider": self.provider,
            "question": self.question,
            "status": self.status,
            "topic": self.topic,
            "updated_at": self.updated_at,
            "verified": self.verified,
        }

    @classmethod
    def from_primitive(cls, data: object) -> LearningExample:
        if not isinstance(data, dict):
            raise ValueError("A learning example must be a mapping.")
        return cls(
            question=str(data.get("question", "")),
            answer=str(data.get("answer", "")),
            status=str(data.get("status", STATUS_PENDING)),
            provider=str(data.get("provider", "")),
            model=str(data.get("model", "")),
            topic=str(data.get("topic", "")),
            language=str(data.get("language", "")),
            confidence=float(data.get("confidence", 0.0) or 0.0),
            citations=tuple(str(c) for c in data.get("citations", ()) or ()),
            verified=bool(data.get("verified", False)),
            created_at=str(data.get("created_at", "")),
            updated_at=str(data.get("updated_at", "")),
            note=str(data.get("note", "")),
            example_id=str(data.get("example_id", "")),
        )

    def describe(self) -> str:
        status = STATUS_LABELS.get(self.status, self.status)
        source = self.model or self.provider or "unknown source"
        return f"[{status}] {self.question[:60] or '(no question)'} · {source}"


@dataclass
class HistoryEntry:
    """One durable line of the Training Center's history."""

    kind: str
    summary: str
    outcome: str = ""
    detail: dict[str, Any] = field(default_factory=dict)
    at: str = ""

    def __post_init__(self) -> None:
        if not self.kind.strip():
            raise ValueError("A history entry needs a kind.")
        if not self.summary.strip():
            raise ValueError("A history entry needs a summary.")
        if not self.at:
            self.at = _now()

    def to_primitive(self) -> dict[str, Any]:
        return {
            "at": self.at,
            "detail": dict(self.detail),
            "kind": self.kind,
            "outcome": self.outcome,
            "summary": self.summary,
        }

    @classmethod
    def from_primitive(cls, data: object) -> HistoryEntry:
        if not isinstance(data, dict):
            raise ValueError("A history entry must be a mapping.")
        return cls(
            kind=str(data.get("kind", "")),
            summary=str(data.get("summary", "")),
            outcome=str(data.get("outcome", "")),
            detail=dict(data.get("detail", {}) or {}),
            at=str(data.get("at", "")),
        )

    def describe(self) -> str:
        when = self.at.replace("T", " ")[:19]
        suffix = f" ({self.outcome})" if self.outcome else ""
        return f"{when}  {self.kind}: {self.summary}{suffix}"


__all__ = [
    "APPROVED_STATUSES",
    "LEARNING_STATUSES",
    "STATUS_APPROVED",
    "STATUS_CORRECTED",
    "STATUS_EXPORTED",
    "STATUS_LABELS",
    "STATUS_LABELS_AR",
    "STATUS_NEEDS_REVIEW",
    "STATUS_PENDING",
    "STATUS_REJECTED",
    "STATUS_USED_IN_TRAINING",
    "HistoryEntry",
    "LearningExample",
    "detect_language",
]
