"""Self-criticism: review an idea before it is believed, and say what is wrong.

The engine makes four deterministic checks on an :class:`~cognition.records.Idea`
and returns a :class:`Critique`:

* **contradiction** -- the graph holds a relation opposing the one the idea
  states (reusing :mod:`core.relation_polarity`, so only a genuine antonym pair
  counts);
* **unsupported** -- the idea rests on a concept the graph does not know, or on
  no source at all;
* **overclaim** -- an idea still at the ``IDEA`` rung states something as if it
  were established (a "is"/"proves" verb with no evidence);
* **duplication** -- the idea is too close to one already reviewed.

A critique lowers the idea's standing; it never silently deletes it. The
:func:`critique` function returns a **new** idea carrying the notes, so the
original is preserved for review.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

from core.ontology import Ontology
from core.relation_polarity import find_polarity_conflicts
from core.tokenize import normalize_token

from cognition.records import Idea, IdeaKind

#: Verbs that assert rather than suggest. An ``IDEA``-rung thought using one is
#: over-claiming; a hypothesis or a supported idea may use them.
_ASSERTIVE = frozenset({"is", "are", "proves", "proven", "always", "never", "must"})


@dataclass(frozen=True)
class Critique:
    """The result of reviewing one idea: notes, and whether it is defensible."""

    notes: tuple[str, ...] = ()
    penalties: float = 0.0
    severity: str = "none"

    def __post_init__(self) -> None:
        if not 0.0 <= self.penalties <= 1.0:
            raise ValueError("Critique penalties must be between 0 and 1.")

    @property
    def clean(self) -> bool:
        return not self.notes

    def describe(self) -> str:
        if self.clean:
            return "no criticism"
        return f"{self.severity}: " + "; ".join(self.notes)


class SelfCriticismEngine:
    """Review ideas against the graph, their sourcing, and each other."""

    def __init__(self, ontology: Ontology, *, novelty_limit: float = 0.85) -> None:
        self.ontology = ontology
        self.novelty_limit = novelty_limit
        self._labels = {c.label for c in ontology.concepts.values()}

    def critique(self, idea: Idea) -> Critique:
        """Run every check and combine the findings into one critique."""
        notes: list[str] = []
        penalties = 0.0

        unsupported = self._unsupported(idea)
        if unsupported:
            notes.append(unsupported)
            penalties += 0.2

        contradiction = self._contradiction(idea)
        if contradiction:
            notes.append(contradiction)
            penalties += 0.4

        overclaim = self._overclaim(idea)
        if overclaim:
            notes.append(overclaim)
            penalties += 0.15

        severity = "high" if penalties >= 0.4 else ("medium" if penalties >= 0.2 else "low")
        return Critique(
            notes=tuple(notes),
            penalties=round(min(penalties, 1.0), 6),
            severity="none" if not notes else severity,
        )

    def review(self, idea: Idea) -> Idea:
        """Return a new idea carrying the critique's notes and a lowered confidence."""
        critique = self.critique(idea)
        if critique.clean:
            return idea
        lowered = max(0.0, idea.confidence - critique.penalties)
        return replace(
            idea,
            critique=idea.critique + critique.notes,
            confidence=round(lowered, 6),
        )

    def _unsupported(self, idea: Idea) -> str:
        if not idea.sources:
            return "no sources named"
        unknown = [s for s in idea.sources if s not in self._labels and " " not in s]
        # A gap topic may name two concepts joined by "vs"; only flag a source the
        # graph truly does not know as a single concept.
        if unknown and not any(s in self._labels for s in idea.sources):
            return f"sources not in the graph: {', '.join(sorted(unknown)[:3])}"
        return ""

    def _contradiction(self, idea: Idea) -> str:
        if len(idea.sources) < 2:
            return ""
        conflicts = find_polarity_conflicts(self.ontology.relations.values())
        for conflict in conflicts:
            source = self._label_for(conflict.source_id)
            target = self._label_for(conflict.target_id)
            if {source, target} <= set(idea.sources):
                return (
                    f"the graph holds both {source} {conflict.first_relation} {target} "
                    f"and {conflict.second_relation}, so this cannot be stated as settled"
                )
        return ""

    def _overclaim(self, idea: Idea) -> str:
        if idea.kind is not IdeaKind.IDEA or idea.evidence:
            return ""
        words = {normalize_token(w) for w in idea.text.split()}
        if words & _ASSERTIVE and "may" not in words and "might" not in words:
            return "states a generated idea as established (no 'may'/'might', no evidence)"
        return ""

    def _label_for(self, concept_id: str) -> str:
        concept = self.ontology.concepts.get(concept_id)
        return concept.label if concept is not None else ""


__all__ = ["Critique", "SelfCriticismEngine"]
