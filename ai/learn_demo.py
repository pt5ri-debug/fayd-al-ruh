"""Demonstrate continuous learning and the regression gate.

    python -m ai.learn_demo

Feeds a few explicit feedback events into the assistant's corpus, prints what
changed, then compares the current evaluation snapshot against the stored
baseline and reports any regression. Learning only ever happens from explicit
feedback -- never from the model's own output.
"""

from __future__ import annotations

import json
from pathlib import Path

from core.continuous_learning import ContinuousLearner, Feedback, RegressionMonitor
from core.documents import build_document_store
from core.knowledge_update import KnowledgeUpdater

_BASELINE = Path(__file__).resolve().parent.parent / "evaluation" / "baseline.json"

_FEEDBACK = (
    Feedback(
        question="ما علاقة المسامية بالنفاذية؟",
        answer="المسامية تقيس الفراغات، وترتبط بالنفاذية التي تقيس سهولة التدفق.",
        confirmed=True,
        confidence=0.8,
    ),
    Feedback(
        question="ما هي النفاذية؟",
        answer="النفاذية تقيس سهولة تدفق الموائع عبر الصخر.",
        confirmed=True,
        confidence=0.7,
    ),
    Feedback(
        question="ما هو السجل الجذبي؟",
        answer="جواب غير دقيق.",
        confirmed=False,
        correction="السجل الجذبي يقيس الإشعاع الطبيعي للصخور.",
        confidence=0.5,
    ),
)


def run_demo() -> int:
    updater = KnowledgeUpdater(build_document_store())
    learner = ContinuousLearner(updater)
    for feedback in _FEEDBACK:
        step = learner.learn(feedback)
        action = step.update.action if step.update else "no corpus change"
        print(f"سؤال: {feedback.question}")
        print(f"  المكافأة: {step.reward:.2f} | تغيير المصدر: {action}")

    print("ملخّص التعلّم:", json.dumps(learner.summary(), ensure_ascii=False, sort_keys=True))
    print(f"عدد المستندات بعد التعلّم: {len(updater.store)}")

    baseline = json.loads(_BASELINE.read_text(encoding="utf-8"))["metrics"]
    current = _current_metrics()
    if current is None:
        print("(تعذّر حساب مقاييس التقييم الآن؛ تخطّي فحص الانحدار.)")
        return 0
    report = RegressionMonitor(baseline, tolerance=0.0).check(current)
    if report.passed:
        print("فحص الانحدار: نجح (لا انحدار).")
        return 0
    print("فحص الانحدار: فشل")
    for check in report.regressions:
        print(f"  - {check.name}: {check.baseline:.4f} -> {check.current:.4f}")
    return 1


def _current_metrics() -> dict[str, float] | None:
    """Run the evaluation runner in-process to get the current metric snapshot."""
    try:
        from evaluation.baseline import default_report
        from evaluation.benchmark import report_to_dict

        return dict(report_to_dict(default_report())["metrics"])
    except Exception:
        return None


def main(argv: list[str] | None = None) -> int:
    if argv:
        print("The learning demo takes no arguments.")
        return 1
    return run_demo()


if __name__ == "__main__":
    raise SystemExit(main())
