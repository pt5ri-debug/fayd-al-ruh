"""Conflict resolution between personalities: preserve, compare, never vote.

When two personalities answer the same question differently, the task is explicit
about what to do and what **not** to do:

1. preserve each result separately;
2. identify differences in assumptions, evidence, goals, or reasoning;
3. compare supporting evidence;
4. identify unresolved disagreements;
5. present a structured comparison;
6. use an explicit aggregation policy only when a final answer is required.

And explicitly: "Do not resolve conflicts solely through personality priority or
arbitrary voting. The system must be allowed to report that no consensus exists."

This module does exactly that. It keeps every position, computes an **evidence**
comparison when the positions carry it, and returns a :class:`ConflictReport` that
can say ``consensus=False``. The only "winner" it will name is the one an
**explicit, named policy** selects -- and the default policy is to *preserve*, not
to choose.

Dependency-free and Tk-free: it holds no model and touches no network.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class AggregationPolicy(str, Enum):
    """How, *if at all*, to reduce several positions to one answer.

    ``PRESERVE`` (the default) names no winner. ``EVIDENCE`` selects the position
    with the strongest stated evidence. ``USER_CHOICE`` selects the one the user
    named. There is deliberately **no** "majority" or "priority" policy: the task
    forbids resolving by count or by rank alone.
    """

    PRESERVE = "preserve"
    EVIDENCE = "evidence"
    USER_CHOICE = "user_choice"


@dataclass(frozen=True)
class Position:
    """One personality's answer, kept whole with what supports it."""

    owner: str
    text: str
    evidence: tuple[str, ...] = ()
    assumptions: tuple[str, ...] = ()
    confidence: float = 0.0
    reasoning: str = ""

    def describe(self) -> str:
        return f"{self.owner}: {self.text[:160]}"


@dataclass(frozen=True)
class PositionDifference:
    """One named way two positions differ."""

    dimension: str
    a: str
    b: str

    def describe(self) -> str:
        return f"{self.dimension}: {self.a} vs {self.b}"


@dataclass(frozen=True)
class ConflictReport:
    """The structured comparison of several positions -- never a bare winner."""

    question: str
    positions: tuple[Position, ...]
    differences: tuple[PositionDifference, ...] = ()
    consensus: bool = False
    shared: tuple[str, ...] = ()
    unresolved: tuple[str, ...] = ()
    policy: AggregationPolicy = AggregationPolicy.PRESERVE
    selected: str = ""
    selection_reason: str = ""

    @property
    def has_conflict(self) -> bool:
        """True when the positions are not unanimous."""
        return not self.consensus

    def describe(self) -> str:
        lines = [f"conflict on {self.question!r}: {len(self.positions)} position(s)"]
        if self.consensus:
            lines.append("  consensus reached")
        else:
            lines.append("  no consensus: the positions differ")
        if self.shared:
            lines.append(f"  shared: {'; '.join(self.shared)}")
        for difference in self.differences:
            lines.append(f"  differs — {difference.describe()}")
        if self.unresolved:
            lines.append(f"  unresolved: {'; '.join(self.unresolved)}")
        if self.selected:
            lines.append(f"  selected by {self.policy.value}: {self.selected} ({self.selection_reason})")
        for position in self.positions:
            lines.append(f"  · {position.describe()}")
        return "\n".join(lines)

    def to_primitive(self) -> dict[str, object]:
        return {
            "question": self.question,
            "consensus": self.consensus,
            "positions": [
                {"owner": position.owner, "text": position.text, "confidence": position.confidence}
                for position in self.positions
            ],
            "differences": [
                {"dimension": difference.dimension, "a": difference.a, "b": difference.b}
                for difference in self.differences
            ],
            "shared": list(self.shared),
            "unresolved": list(self.unresolved),
            "policy": self.policy.value,
            "selected": self.selected,
        }


def _normalise(text: str) -> str:
    """A comparison key: casefolded, whitespace-collapsed, trailing stop dropped."""
    return " ".join(text.casefold().split()).rstrip(".!?،؟ ")


def compare_positions(positions: tuple[Position, ...]) -> tuple[PositionDifference, ...]:
    """The named differences between the first position and each other one.

    Only *material* differences are reported: a difference in the answer text, in
    the stated evidence, or in the assumptions. Confidence is compared too, since a
    reader needs to know one position is more confident than another -- but a
    confidence gap alone is not treated as a disagreement about the facts.
    """
    if len(positions) < 2:
        return ()
    base = positions[0]
    differences: list[PositionDifference] = []
    for other in positions[1:]:
        if _normalise(base.text) != _normalise(other.text):
            differences.append(
                PositionDifference("answer", f"{base.owner}: {base.text[:80]}", f"{other.owner}: {other.text[:80]}")
            )
        base_evidence = {_normalise(item) for item in base.evidence}
        other_evidence = {_normalise(item) for item in other.evidence}
        if base_evidence != other_evidence:
            differences.append(
                PositionDifference(
                    "evidence",
                    f"{base.owner}: {len(base.evidence)} item(s)",
                    f"{other.owner}: {len(other.evidence)} item(s)",
                )
            )
        base_assumptions = {_normalise(item) for item in base.assumptions}
        other_assumptions = {_normalise(item) for item in other.assumptions}
        if base_assumptions != other_assumptions:
            differences.append(
                PositionDifference(
                    "assumptions",
                    f"{base.owner}: {len(base.assumptions)} assumption(s)",
                    f"{other.owner}: {len(other.assumptions)} assumption(s)",
                )
            )
        if abs(base.confidence - other.confidence) >= 0.15:
            differences.append(
                PositionDifference(
                    "confidence",
                    f"{base.owner}: {base.confidence:.2f}",
                    f"{other.owner}: {other.confidence:.2f}",
                )
            )
    return tuple(differences)


@dataclass
class ConflictResolver:
    """Preserve positions, compare them, and only then -- maybe -- aggregate.

    ``policy`` is the caller's explicit choice; the default preserves. Nothing
    here mutates a position or discards one: the report always carries every
    position, so a reader sees the disagreement even when a winner is named.
    """

    policy: AggregationPolicy = AggregationPolicy.PRESERVE
    #: The personality a ``USER_CHOICE`` policy should select.
    user_choice: str = ""
    #: How much more evidence a position needs to win under the ``EVIDENCE`` policy
    #: than the runner-up, so a near-tie is *reported as a tie* rather than decided.
    evidence_margin: int = 1

    def resolve(
        self,
        question: str,
        positions: tuple[Position, ...],
        *,
        policy: AggregationPolicy | None = None,
        user_choice: str = "",
    ) -> ConflictReport:
        """Compare ``positions`` and return the structured report."""
        active_policy = policy or self.policy
        chosen_user = user_choice or self.user_choice
        if not positions:
            return ConflictReport(question=question, positions=())

        texts = {_normalise(position.text) for position in positions if position.text.strip()}
        consensus = len(texts) <= 1
        differences = compare_positions(positions)
        shared = _shared_evidence(positions)
        unresolved = _unresolved(positions, consensus, differences)

        selected = ""
        reason = ""
        if not consensus and active_policy is AggregationPolicy.EVIDENCE:
            selected, reason = self._by_evidence(positions)
        elif not consensus and active_policy is AggregationPolicy.USER_CHOICE:
            selected, reason = self._by_user_choice(positions, chosen_user)
        elif active_policy is AggregationPolicy.PRESERVE:
            reason = "no policy asked for a single answer; every position is preserved."

        return ConflictReport(
            question=question,
            positions=positions,
            differences=differences,
            consensus=consensus,
            shared=shared,
            unresolved=unresolved,
            policy=active_policy,
            selected=selected,
            selection_reason=reason,
        )

    def _by_evidence(self, positions: tuple[Position, ...]) -> tuple[str, str]:
        ranked = sorted(positions, key=lambda position: (-len(position.evidence), -position.confidence))
        best = ranked[0]
        if len(ranked) > 1:
            runner_up = ranked[1]
            if len(best.evidence) - len(runner_up.evidence) < self.evidence_margin:
                return "", (
                    f"the two strongest positions are within {self.evidence_margin} "
                    "evidence item(s): no consensus, not decided."
                )
        return best.owner, f"{best.owner} rests on the most stated evidence ({len(best.evidence)})."

    def _by_user_choice(self, positions: tuple[Position, ...], user_choice: str) -> tuple[str, str]:
        from core.learning_isolation import owner_for

        wanted = owner_for(user_choice)
        for position in positions:
            if owner_for(position.owner) == wanted:
                return position.owner, f"the user selected {position.owner}."
        return "", f"the user named {user_choice!r}, which is not among the positions."


def _shared_evidence(positions: tuple[Position, ...]) -> tuple[str, ...]:
    """The evidence items every position states (the common ground)."""
    if not positions:
        return ()
    sets = [{_normalise(item) for item in position.evidence} for position in positions]
    common = set.intersection(*sets) if sets else set()
    return tuple(sorted(common))


def _unresolved(
    positions: tuple[Position, ...], consensus: bool, differences: tuple[PositionDifference, ...]
) -> tuple[str, ...]:
    """The disagreements that remain open, named for the reader."""
    if consensus:
        return ()
    open_items: list[str] = []
    for difference in differences:
        if difference.dimension in {"answer", "evidence", "assumptions"}:
            open_items.append(difference.dimension)
    return tuple(dict.fromkeys(open_items))


__all__ = [
    "AggregationPolicy",
    "ConflictReport",
    "ConflictResolver",
    "Position",
    "PositionDifference",
    "compare_positions",
]
