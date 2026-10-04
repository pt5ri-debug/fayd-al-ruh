from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

from core.question import Question


class InquiryStatus(str, Enum):
    OPEN = "open"
    ANSWERED = "answered"
    BLOCKED = "blocked"


@dataclass
class Inquiry:
    question: Question
    status: InquiryStatus = InquiryStatus.OPEN
    answer: str | None = None
    evidence_ids: list[str] = field(default_factory=list)

    def answer_with(self, answer: str, evidence_ids: list[str]) -> None:
        if not answer.strip():
            raise ValueError("Inquiry answer cannot be empty.")
        if not evidence_ids or any(not evidence_id.strip() for evidence_id in evidence_ids):
            raise ValueError("An answered inquiry needs evidence IDs.")
        self.answer = answer
        self.evidence_ids = list(dict.fromkeys(evidence_ids))
        self.status = InquiryStatus.ANSWERED

    def block(self) -> None:
        self.status = InquiryStatus.BLOCKED
