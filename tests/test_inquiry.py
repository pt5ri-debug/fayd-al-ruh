from core.inquiry import Inquiry, InquiryStatus
from core.question import Question


def test_inquiry_answer_requires_evidence():
    inquiry = Inquiry(Question("What happened?", "event", .5))
    inquiry.answer_with("Rain occurred.", ["EV-1", "EV-1"])
    assert inquiry.status is InquiryStatus.ANSWERED
    assert inquiry.evidence_ids == ["EV-1"]
