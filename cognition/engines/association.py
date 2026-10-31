"""Association: discover how concepts are related, without inventing a link.

Three real, existing signals are read and reported as :class:`Association`
records:

* a **direct typed relation** the graph already stores;
* a **multi-hop connection** found by :func:`core.relation_paths.find_connection`
  (a walked path, labelled as walked, never stored);
* a **structural analogy** found by :class:`core.analogy.AnalogyEngine`
  (a shared relational shape between two concepts).

Nothing here writes to the graph. An association is a *report*, with the exact
evidence (a relation type, a path, or a mapping) attached so a reviewer can
check it.
"""

from __future__ import annotations

from dataclasses import dataclass

from core.analogy import AnalogyEngine
from core.ontology import Ontology
from core.relation_paths import find_connection


@dataclass(frozen=True)
class Association:
    """A discovered link between two concepts, with the evidence that shows it."""

    source: str
    target: str
    kind: str
    confidence: float
    evidence: str = ""

    def __post_init__(self) -> None:
        if not self.source.strip() or not self.target.strip():
            raise ValueError("Association endpoints cannot be empty.")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("Association confidence must be between 0 and 1.")

    def describe(self) -> str:
        return f"{self.source} ~ {self.target} [{self.kind}] ({self.confidence:.2f})"


class AssociationEngine:
    """Find direct, multi-hop, and analogical associations in the graph."""

    def __init__(self, ontology: Ontology, *, analogy: AnalogyEngine | None = None) -> None:
        self.ontology = ontology
        self.analogy_engine = analogy or AnalogyEngine(ontology)
        self._labels = {c.id: c.label for c in ontology.concepts.values()}

    def direct(self, source: str, target: str) -> tuple[Association, ...]:
        """The typed relations the graph stores between two named concepts."""
        source_id = self._id_for(source)
        target_id = self._id_for(target)
        if source_id is None or target_id is None:
            return ()
        found: list[Association] = []
        for relation in self.ontology.relations.values():
            if relation.source_id == source_id and relation.target_id == target_id:
                found.append(
                    Association(
                        source=source,
                        target=target,
                        kind=relation.relation_type,
                        confidence=float(relation.confidence),
                        evidence=f"stored relation {relation.relation_type}",
                    )
                )
        return tuple(sorted(found, key=lambda a: (-a.confidence, a.kind)))

    def path(self, source: str, target: str, *, max_hops: int = 3) -> Association | None:
        """A multi-hop connection, reported as *walked* (never as stored)."""
        path = find_connection(self.ontology, source, target, max_hops=max_hops)
        if path is None:
            return None
        return Association(
            source=source,
            target=target,
            kind="path",
            confidence=float(path.confidence),
            evidence=path.describe().replace("\n", " "),
        )

    def analogy(self, source: str, target: str, *, hops: int = 1) -> Association | None:
        """A structural analogy between two concepts, only when it is meaningful."""
        result = self.analogy_engine.between_concepts(source, target, hops=hops)
        if not result.is_meaningful:
            return None
        coverage = float(getattr(result.mapping, "relation_coverage", 0.0) or 0.0)
        return Association(
            source=source,
            target=target,
            kind="analogy",
            confidence=round(min(1.0, coverage), 6),
            evidence=f"{len(result.mapping.matched_edges)} matched relation(s)",
        )

    def between(self, source: str, target: str, *, max_hops: int = 3) -> tuple[Association, ...]:
        """Every association between two concepts: direct, then path, then analogy."""
        found: list[Association] = list(self.direct(source, target))
        path = self.path(source, target, max_hops=max_hops)
        if path is not None:
            found.append(path)
        analogue = self.analogy(source, target)
        if analogue is not None:
            found.append(analogue)
        return tuple(found)

    def neighbours(self, label: str) -> tuple[str, ...]:
        """The labels one relation away from ``label`` (in either direction)."""
        concept_id = self._id_for(label)
        if concept_id is None:
            return ()
        neighbours: list[str] = []
        for relation in self.ontology.relations.values():
            if relation.source_id == concept_id:
                other = self._labels.get(relation.target_id)
            elif relation.target_id == concept_id:
                other = self._labels.get(relation.source_id)
            else:
                continue
            if other and other not in neighbours:
                neighbours.append(other)
        return tuple(sorted(neighbours))

    def rank_pairs(self, labels: tuple[str, ...], *, limit: int = 10) -> tuple[Association, ...]:
        """Rank the strongest associations among a set of concepts.

        Used to pick a promising pair to recombine into an idea. Deterministic:
        ordered by confidence, then by the two labels.
        """
        if limit < 1:
            raise ValueError("limit must be at least 1.")
        pairs: list[Association] = []
        seen: set[tuple[str, str]] = set()
        ordered = tuple(sorted(set(labels)))
        for i, source in enumerate(ordered):
            for target in ordered[i + 1 :]:
                key = (source, target)
                if key in seen:
                    continue
                seen.add(key)
                best = self.between(source, target)
                if best:
                    pairs.append(best[0])
        pairs.sort(key=lambda a: (-a.confidence, a.source, a.target))
        return tuple(pairs[:limit])

    def _id_for(self, label: str) -> str | None:
        for concept in self.ontology.concepts.values():
            if concept.label == label:
                return concept.id
        return None


__all__ = ["Association", "AssociationEngine"]
