"""Idea generation: recombine knowledge into new statements, never into facts.

An idea is produced by a **named strategy** over real graph facts -- a direct
relation, a multi-hop path, or a meaningful analogy -- so it is traceable, and
its confidence is derived from the evidence rather than from a dice roll. The
generator never raises an idea above :class:`~cognition.records.IdeaKind.IDEA`;
promotion is a separate, recorded act (see :func:`cognition.records.promote`).

Determinism matters: the same graph and the same strategy produce the same
ideas, so a reviewer can reproduce what the mind thought. Novelty is *not*
randomness -- it is measured by how far an idea's concepts are from what has
already been thought (:class:`NoveltyTracker`), which is what makes "do not rely
on randomness alone" true.
"""

from __future__ import annotations

from collections import Counter

from core.information_gap import InformationGap
from core.ontology import Ontology
from core.tokenize import normalize_token

from cognition.engines.association import Association, AssociationEngine
from cognition.records import Idea, IdeaKind, Source, SourceKind

#: How far two ideas must differ in wording to count as different.
_JACCARD_LIMIT = 0.7


class NoveltyTracker:
    """Keeps what has already been thought, and scores how new a statement is.

    Novelty is ``1 - (shared tokens / union tokens)`` against the closest earlier
    statement, so a re-wording of something already said scores low and a
    genuinely different combination scores high. It is bounded and deterministic.
    """

    def __init__(self, *, max_items: int = 500) -> None:
        self.max_items = max_items
        self._seen: list[frozenset[str]] = []

    def novelty(self, text: str) -> float:
        tokens = self._tokens(text)
        if not tokens or not self._seen:
            return 1.0 if tokens else 0.0
        closest = 0.0
        for other in self._seen:
            union = tokens | other
            if not union:
                continue
            overlap = len(tokens & other) / len(union)
            closest = max(closest, overlap)
        return round(max(0.0, 1.0 - closest), 6)

    def remember(self, text: str) -> None:
        tokens = self._tokens(text)
        if tokens:
            self._seen.append(tokens)
            del self._seen[:-self.max_items]

    def _tokens(self, text: str) -> frozenset[str]:
        return frozenset(
            t for t in (normalize_token(w) for w in text.split()) if t and len(t) > 2
        )


class IdeaEngine:
    """Generate ideas by recombining the graph's own facts."""

    def __init__(
        self,
        ontology: Ontology,
        *,
        associations: AssociationEngine | None = None,
        novelty: NoveltyTracker | None = None,
    ) -> None:
        self.ontology = ontology
        self.associations = associations or AssociationEngine(ontology)
        self.novelty = novelty or NoveltyTracker()

    def from_relation(self, association: Association) -> Idea | None:
        """State a relation the graph stores, as an idea to examine."""
        if association.kind == "path":
            text = (
                f"{association.source} and {association.target} are connected "
                f"through {association.evidence.split('(')[0].strip()}"
            )
            engine = "association.path"
            rationale = "a multi-hop connection the graph supports"
        elif association.kind == "analogy":
            text = (
                f"{association.source} may work like {association.target}: "
                f"they share a relational shape ({association.evidence})"
            )
            engine = "association.analogy"
            rationale = "a structural analogy, a candidate not a fact"
        else:
            text = (
                f"{association.source} {association.kind.replace('_', ' ')} "
                f"{association.target}"
            )
            engine = "association.direct"
            rationale = "a relation the graph already stores"
        return self._make(
            text=text,
            engine=engine,
            sources=(association.source, association.target),
            rationale=rationale,
            confidence=min(0.6, association.confidence),
        )

    def from_gap(self, gap: InformationGap) -> Idea:
        """Turn a gap into a stated thing-to-find-out (still only an idea)."""
        return self._make(
            text=f"We may not yet know enough about {gap.topic}",
            engine="curiosity.gap",
            sources=(gap.topic,),
            rationale=f"the graph is thin here (priority {gap.priority:.2f})",
            confidence=min(0.4, 1.0 - gap.known_confidence),
        )

    def from_pair(self, source: str, target: str) -> Idea | None:
        """Recombine two concepts into one new statement, if the graph links them."""
        found = self.associations.between(source, target)
        if not found:
            return None
        best = found[0]
        return self.from_relation(best)

    def generate(
        self,
        *,
        labels: tuple[str, ...] = (),
        gaps: tuple[InformationGap, ...] = (),
        limit: int = 5,
    ) -> tuple[Idea, ...]:
        """Produce up to ``limit`` ideas, newest-idea-wins, de-duplicated.

        Pairs are ranked by the strength of their association, gaps by priority.
        An idea whose wording is too close to one already produced (Jaccard above
        :data:`_JACCARD_LIMIT`) is dropped, so the output is a set of distinct
        thoughts rather than near-copies.
        """
        if limit < 1:
            raise ValueError("limit must be at least 1.")
        ideas: list[Idea] = []
        if labels:
            for association in self.associations.rank_pairs(labels, limit=limit * 2):
                candidate = self.from_relation(association)
                if candidate is not None:
                    ideas.append(candidate)
        for gap in gaps[:limit]:
            ideas.append(self.from_gap(gap))

        accepted: list[Idea] = []
        for idea in ideas:
            if self.novelty.novelty(idea.text) < (1.0 - _JACCARD_LIMIT):
                continue
            self.novelty.remember(idea.text)
            accepted.append(idea)
            if len(accepted) >= limit:
                break
        return tuple(accepted)

    def _make(
        self,
        *,
        text: str,
        engine: str,
        sources: tuple[str, ...],
        rationale: str,
        confidence: float,
    ) -> Idea:
        return Idea(
            text=text,
            kind=IdeaKind.IDEA,
            engine=engine,
            sources=sources,
            rationale=rationale,
            confidence=round(min(max(confidence, 0.0), 1.0), 6),
            novelty=self.novelty.novelty(text),
            origin=Source(kind=SourceKind.SELF, reference="idea-engine"),
        )


def token_profile(ideas: tuple[Idea, ...]) -> dict[str, int]:
    """A readable token frequency over a set of ideas (for the UI)."""
    counts: Counter[str] = Counter()
    for idea in ideas:
        for word in idea.text.split():
            token = normalize_token(word)
            if token and len(token) > 2:
                counts[token] += 1
    return dict(counts.most_common(20))


__all__ = ["IdeaEngine", "NoveltyTracker", "token_profile"]
