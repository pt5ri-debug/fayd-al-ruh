"""Curiosity: find what is worth finding out, from the graph and memory.

This wraps the project's own :class:`core.curiosity.Curiosity` (which turns
low-confidence beliefs into :class:`~core.information_gap.InformationGap`s) and
adds the gaps the graph itself implies but no belief covers:

* a concept with **no outgoing relation** (an isolated node is a hole);
* a concept reached by **only one** relation (thinly evidenced);
* a **contradicted** pair (the graph holds opposing poles on the same pair).

Every gap is an :class:`~core.information_gap.InformationGap`, so the rest of the
system (the question generator, the scheduler) needs no new type. Nothing here
asserts an answer; it only says what is missing.
"""

from __future__ import annotations

from collections.abc import Iterable

from core.belief import Belief
from core.curiosity import Curiosity
from core.information_gap import InformationGap
from core.ontology import Ontology


class CuriosityEngine:
    """Produce prioritised information gaps from beliefs, the graph, and memory."""

    def __init__(
        self,
        ontology: Ontology,
        *,
        curiosity: Curiosity | None = None,
        min_gap_confidence: float = 0.8,
    ) -> None:
        self.ontology = ontology
        self.curiosity = curiosity or Curiosity()
        self.min_gap_confidence = min_gap_confidence

    def gaps_from_beliefs(self, beliefs: Iterable[Belief]) -> list[InformationGap]:
        """The project's own curiosity over low-confidence beliefs."""
        return self.curiosity.discover(beliefs, self.min_gap_confidence)

    def gaps_from_graph(self, *, max_gaps: int = 20) -> list[InformationGap]:
        """Structural holes: isolated concepts and thinly-connected concepts.

        A concept with no relation at all, or with exactly one, is a place the
        graph is thin; the gap names the concept so a question can be asked about
        it. Deterministic: concepts are visited in label order.
        """
        if max_gaps < 1:
            raise ValueError("max_gaps must be at least 1.")
        degrees = self._degrees()
        gaps: list[InformationGap] = []
        for label in sorted(self._labels()):
            degree = degrees.get(label, 0)
            if degree == 0:
                gaps.append(
                    InformationGap(
                        topic=label,
                        known_confidence=0.0,
                        required_confidence=self.min_gap_confidence,
                        importance=0.8,
                    )
                )
            elif degree == 1:
                gaps.append(
                    InformationGap(
                        topic=label,
                        known_confidence=0.3,
                        required_confidence=self.min_gap_confidence,
                        importance=0.5,
                    )
                )
        gaps.sort(key=lambda g: (-g.priority, g.topic))
        return gaps[:max_gaps]

    def gaps_from_conflicts(self, conflicts: Iterable[object], *, max_gaps: int = 10) -> list[InformationGap]:
        """A contradiction is a high-priority gap: the graph disagrees with itself.

        Each conflict object (a :class:`core.relation_consistency.RelationConflict`)
        is expected to expose ``source``/``target`` labels; anything else is
        skipped rather than guessed at.
        """
        gaps: list[InformationGap] = []
        for conflict in conflicts:
            source = str(getattr(conflict, "source", "")).strip()
            target = str(getattr(conflict, "target", "")).strip()
            if not source or not target:
                continue
            severity = float(getattr(conflict, "severity", 0.5) or 0.5)
            gaps.append(
                InformationGap(
                    topic=f"{source} vs {target}",
                    known_confidence=max(0.0, 0.6 - severity),
                    required_confidence=self.min_gap_confidence,
                    importance=min(1.0, 0.6 + severity),
                )
            )
        gaps.sort(key=lambda g: (-g.priority, g.topic))
        return gaps[:max_gaps]

    def discover(
        self,
        beliefs: Iterable[Belief] = (),
        *,
        conflicts: Iterable[object] = (),
        max_gaps: int = 20,
    ) -> list[InformationGap]:
        """All gaps, de-duplicated by topic, ordered by priority, bounded."""
        combined = (
            self.gaps_from_beliefs(beliefs)
            + self.gaps_from_graph(max_gaps=max_gaps)
            + self.gaps_from_conflicts(conflicts)
        )
        seen: dict[str, InformationGap] = {}
        for gap in combined:
            current = seen.get(gap.topic)
            if current is None or gap.priority > current.priority:
                seen[gap.topic] = gap
        ordered = sorted(seen.values(), key=lambda g: (-g.priority, g.topic))
        return ordered[:max_gaps]

    def _labels(self) -> list[str]:
        return [c.label for c in self.ontology.concepts.values()]

    def _degrees(self) -> dict[str, int]:
        labels = {c.id: c.label for c in self.ontology.concepts.values()}
        degrees: dict[str, int] = dict.fromkeys(labels.values(), 0)
        for relation in self.ontology.relations.values():
            source = labels.get(relation.source_id)
            target = labels.get(relation.target_id)
            if source:
                degrees[source] = degrees.get(source, 0) + 1
            if target:
                degrees[target] = degrees.get(target, 0) + 1
        return degrees


__all__ = ["CuriosityEngine"]
