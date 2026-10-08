from __future__ import annotations

from core.composition import evidence_summary_sentence
from core.decision_explainer import DecisionExplainer
from core.decision_trace import DecisionTrace
from core.evidence import Evidence
from core.evidence_trace import EvidenceTrace
from core.explanation import Explanation
from core.reasoning_explainer import ReasoningExplainer
from core.reasoning_trace import ReasoningTrace


class ExplanationEngine:
    """Composes concise, source-based explanations from explicit subsystem traces."""
    def __init__(self) -> None:
        self.reasoning = ReasoningExplainer()
        self.decision = DecisionExplainer()

    def explain(self, conclusion: str, evidence: list[Evidence] | None = None, reasoning_trace: ReasoningTrace | None = None, decision_trace: DecisionTrace | None = None, causal: tuple[str, ...] = ()) -> Explanation:
        items = evidence or []
        evidence_trace = EvidenceTrace.from_evidence(items)
        reasoning = self.reasoning.explain(reasoning_trace) if reasoning_trace is not None else ()
        decision = self.decision.explain(decision_trace) if decision_trace is not None else ()
        summary = evidence_summary_sentence(
            conclusion,
            len(evidence_trace.supporting_ids),
            len(evidence_trace.contradicting_ids),
        )
        return Explanation(conclusion, summary, evidence_trace.evidence_ids, reasoning, causal, decision)
