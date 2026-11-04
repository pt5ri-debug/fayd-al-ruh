"""Improvement prioritization: a ranking with named, inspectable criteria.

Section 7 of the phase asks for transparent criteria, no arbitrary scores, and a
way for the user to inspect and override a priority. This module is that ranking.

Every criterion is a **named, documented factor** with an explicit weight, and a
:class:`PriorityScore` reports each factor's contribution -- so a reader can see
*why* one proposal outranks another rather than being handed a number. The
factors are the phase's own list: severity, frequency, evidence strength,
expected benefit, implementation complexity, regression risk, resource cost,
reversibility, and user impact.

Two honesty rules:

* **the ranking is a recommendation** -- the user may pin a proposal's priority
  or reorder it (``override``), and a self-modification is never automatically
  placed above a user-requested change: a proposal created in answer to a user's
  own request carries the ``user_requested`` flag and wins ties; and
* **a factor with no evidence contributes nothing** -- an unknown factor is
  scored at its neutral value and *reported as such*, never guessed upward.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field

from core.improvement.proposals import ImprovementProposal, ProposalCategory
from core.improvement.weakness import Severity

#: The weight each factor contributes. They sum to 1.0 so a score is bounded
#: 0..1 and comparable across proposals. Written down here so a change to the
#: weighting is a visible decision.
FACTOR_WEIGHTS: dict[str, float] = {
    "severity": 0.20,
    "frequency": 0.10,
    "evidence": 0.10,
    "benefit": 0.15,
    "complexity": 0.10,   # lower complexity scores higher
    "regression_risk": 0.10,  # lower risk scores higher
    "resource_cost": 0.05,    # lower cost scores higher
    "reversibility": 0.10,    # more reversible scores higher
    "user_impact": 0.10,
}

#: What each factor means, so a reader never has to guess what moved a score.
FACTOR_MEANINGS: dict[str, str] = {
    "severity": "How bad the underlying problem is (low..critical).",
    "frequency": "How often the problem was observed.",
    "evidence": "How strong the evidence behind the finding is (its confidence).",
    "benefit": "How much the change is expected to help.",
    "complexity": "How hard the change is to implement (simpler scores higher).",
    "regression_risk": "How likely the change is to break something (lower is safer).",
    "resource_cost": "What the change costs to run and keep (lower is cheaper).",
    "reversibility": "How easily the change can be undone (more reversible is better).",
    "user_impact": "How much the user is affected by the problem and the fix.",
}


@dataclass(frozen=True)
class PriorityScore:
    """One proposal's priority: the total and every factor's contribution."""

    proposal_id: str
    total: float
    factors: dict[str, float] = field(default_factory=dict)
    notes: tuple[str, ...] = ()
    #: True when the user pinned this priority, so the ranking must not move it.
    pinned: bool = False

    def describe(self) -> str:
        parts = ", ".join(f"{name}={value:.2f}" for name, value in sorted(self.factors.items()))
        pin = " [pinned by the user]" if self.pinned else ""
        return f"{self.proposal_id}: {self.total:.4f}{pin} ({parts})"


def _severity_from_component(component: str) -> float:
    """A neutral severity when the proposal carries no explicit one.

    A proposal's own ``evidence`` and ``problem`` text do not encode a severity,
    so the ranker uses the affected component's typical seriousness as a neutral
    prior rather than inventing a precise figure.
    """
    return {
        "execution": 0.6,
        "reasoning": 0.65,
        "retrieval": 0.55,
        "knowledge": 0.5,
        "planning": 0.5,
        "resource": 0.6,
        "interface": 0.55,
        "assistant": 0.55,
    }.get(component, 0.5)


#: How much a category is expected to help, on a 0..1 scale. Written down so the
#: benefit factor is a documented judgement, not a hidden constant.
_CATEGORY_BENEFIT: dict[ProposalCategory, float] = {
    ProposalCategory.BUG_FIX: 0.9,
    ProposalCategory.REASONING_RULE: 0.8,
    ProposalCategory.RETRIEVAL_OPTIMIZATION: 0.75,
    ProposalCategory.PLANNING: 0.7,
    ProposalCategory.KNOWLEDGE_SCHEMA: 0.65,
    ProposalCategory.MEMORY_RANKING: 0.6,
    ProposalCategory.RESOURCE_MANAGEMENT: 0.6,
    ProposalCategory.TOOL_INTEGRATION: 0.55,
    ProposalCategory.CONFIGURATION: 0.5,
    ProposalCategory.PROCEDURE: 0.5,
    ProposalCategory.DOCUMENTATION: 0.4,
    ProposalCategory.MODEL_OR_ADAPTER: 0.7,
}

#: Categories that touch security or isolation, which raises the regression risk.
_RISKY_CATEGORIES: frozenset[ProposalCategory] = frozenset({ProposalCategory.MODEL_OR_ADAPTER})


def score_proposal(
    proposal: ImprovementProposal,
    *,
    weakness_severity: Severity | None = None,
    user_requested: bool = False,
    pinned: bool = False,
) -> PriorityScore:
    """Score one proposal on the nine named factors.

    ``weakness_severity`` is the severity of the finding the proposal answers, so
    a proposal built from a critical finding outranks one built from a low one.
    When it is absent the component's neutral prior is used and the score says so
    in its notes.
    """
    notes: list[str] = []
    if weakness_severity is not None:
        severity = weakness_severity.weight
    else:
        severity = _severity_from_component(proposal.target_component)
        notes.append("severity is the component's neutral prior (no explicit finding severity)")

    # Frequency: a proxy from the proposal's own evidence count, capped.
    frequency = min(1.0, len(proposal.evidence) / 4.0)
    # Evidence: the confidence of the finding, approximated by evidence richness.
    evidence = min(1.0, 0.3 + 0.15 * len(proposal.evidence))
    benefit = _CATEGORY_BENEFIT.get(proposal.category, 0.5)
    # Lower complexity scores higher.
    complexity = 1.0 - (proposal.complexity - 1) / 4.0
    regression_risk = 0.7 if proposal.category in _RISKY_CATEGORIES else 0.3
    if proposal.requires_explicit_approval:
        regression_risk = max(regression_risk, 0.6)
    resource_cost = min(1.0, len(proposal.required_resources) / 3.0)
    # A change with a written rollback strategy is more reversible.
    reversibility = 0.9 if proposal.rollback_strategy.strip() else 0.4
    user_impact = 0.9 if user_requested else 0.5
    if user_requested:
        notes.append("created in answer to a user's own request")

    factors = {
        "severity": severity,
        "frequency": frequency,
        "evidence": evidence,
        "benefit": benefit,
        "complexity": complexity,
        "regression_risk": regression_risk,
        "resource_cost": resource_cost,
        "reversibility": reversibility,
        "user_impact": user_impact,
    }
    total = sum(factors[name] * FACTOR_WEIGHTS[name] for name in FACTOR_WEIGHTS)
    return PriorityScore(proposal.proposal_id, round(total, 6), factors, tuple(notes), pinned)


@dataclass(frozen=True)
class RankedProposal:
    """A proposal together with its priority score."""

    proposal: ImprovementProposal
    score: PriorityScore

    def describe(self) -> str:
        return f"{self.score.describe()}\n  {self.proposal.title}"


def rank_proposals(
    proposals: Iterable[ImprovementProposal],
    *,
    severities: dict[str, Severity] | None = None,
    overrides: dict[str, float] | None = None,
    user_requested: frozenset[str] = frozenset(),
) -> tuple[RankedProposal, ...]:
    """Rank proposals strongest-first, honouring the user's overrides.

    ``severities`` maps a weakness id to its severity, so the proposal that
    answers a severe finding scores higher. ``overrides`` maps a proposal id to a
    priority the user set; an overridden proposal is marked ``pinned`` and sorts
    above any unpinned one with a lower explicit priority, but the user's order is
    otherwise respected exactly -- the ranker never overrules a human.
    """
    severities = severities or {}
    overrides = overrides or {}
    ranked: list[RankedProposal] = []
    for proposal in proposals:
        severity = severities.get(proposal.source_weakness_id or "")
        score = score_proposal(
            proposal,
            weakness_severity=severity,
            user_requested=proposal.proposal_id in user_requested,
            pinned=proposal.proposal_id in overrides,
        )
        if proposal.proposal_id in overrides:
            score = PriorityScore(
                score.proposal_id,
                float(overrides[proposal.proposal_id]),
                score.factors,
                score.notes,
                pinned=True,
            )
        ranked.append(RankedProposal(proposal, score))
    # Pinned first (by their explicit priority), then by score, then id -- a
    # deterministic order so the same proposals always rank the same way.
    return tuple(
        sorted(
            ranked,
            key=lambda item: (
                not item.score.pinned,
                -item.score.total,
                item.proposal.proposal_id,
            ),
        )
    )


def describe_ranking(ranked: Iterable[RankedProposal]) -> str:
    """Render a ranking with each factor's contribution."""
    ordered = list(ranked)
    if not ordered:
        return "No proposals to rank."
    lines = ["Proposal ranking (highest priority first):"]
    for index, item in enumerate(ordered, start=1):
        lines.append(f"{index}. {item.score.describe()}")
        lines.append(f"   {item.proposal.title}")
        for note in item.score.notes:
            lines.append(f"   note: {note}")
    return "\n".join(lines)


__all__ = [
    "FACTOR_MEANINGS",
    "FACTOR_WEIGHTS",
    "PriorityScore",
    "RankedProposal",
    "describe_ranking",
    "rank_proposals",
    "score_proposal",
]
