"""Ontology-aware retrieval: follow relations, not just shared words.

Keyword and hashed-semantic retrieval only reach memory items that literally
share terms with the query. This module grounds each item in ontology concepts
and follows explicit *and derived* relations outward from the query's concepts,
so an item that mentions a merely related concept becomes reachable.

Example: a query about a well log reaches a reservoir item, even though the two
share no words, because ``well_log measures reservoir`` is derived by
``core.ontology_reasoner`` and the item grounds the reservoir concept.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from core.ontology import Ontology
from core.ontology_reasoner import OntologyReasoner


@dataclass(frozen=True)
class OntologyMatch:
    """A memory item ranked by how its grounded concepts connect to the query."""

    item_id: str
    score: float
    hops: int
    concepts: tuple[str, ...]
    link: tuple[str, str, str] | None = None
    """The relation that connected the item, as (source_id, relation_type, target_id)."""


class OntologyGuidedRetriever:
    """Rank memory items by relational proximity to the query's concepts.

    Concepts are matched by term occurrence. Proximity is a bounded breadth-first
    walk over explicit and derived relations in their declared direction, with a
    geometric decay per hop, so direct matches outrank one-hop and two-hop ones.
    """

    def __init__(
        self,
        ontology: Ontology,
        reasoner: OntologyReasoner | None = None,
        max_hops: int = 2,
        hop_decay: float = 0.6,
    ) -> None:
        if max_hops < 1:
            raise ValueError("Maximum hops must be at least 1.")
        if not 0.0 < hop_decay <= 1.0:
            raise ValueError("Hop decay must be in (0, 1].")
        self.ontology = ontology
        self.reasoner = reasoner or OntologyReasoner()
        self.max_hops = max_hops
        self.hop_decay = hop_decay
        self._adjacency: dict[str, list[tuple[str, float, tuple[str, str, str]]]] | None = None

    def concepts_in(self, text: str) -> set[str]:
        """Return the concept IDs whose terms occur in the text."""
        folded = text.casefold()
        return {
            concept.id
            for concept in self.ontology.concepts.values()
            if any(term in folded for term in concept.terms)
        }

    def invalidate(self) -> None:
        """Drop the cached adjacency after the ontology's relations change."""
        self._adjacency = None

    def reachable(self, seeds: set[str]) -> dict[str, tuple[int, float, tuple[str, str, str] | None]]:
        """Map each concept reachable from ``seeds`` to (hops, weight, linking relation).

        ``link`` is ``None`` for the seed concepts themselves; for a concept reached
        in one hop it is the relation that connected it, which lets a caller credit
        the exact link that made an item reachable.
        """
        adjacency = self._edges()
        best: dict[str, tuple[int, float, tuple[str, str, str] | None]] = dict.fromkeys(seeds, (0, 1.0, None))
        frontier = list(seeds)
        for hop in range(1, self.max_hops + 1):
            next_frontier: list[str] = []
            for node in frontier:
                for target, _confidence, link in adjacency.get(node, ()):
                    if target in best:
                        continue
                    best[target] = (hop, self.hop_decay ** hop, link)
                    next_frontier.append(target)
            if not next_frontier:
                break
            frontier = next_frontier
        return best

    def rank(
        self,
        items: Mapping[str, str],
        query: str,
        base_scores: Mapping[str, float] | None = None,
        limit: int = 5,
    ) -> list[OntologyMatch]:
        """Rank items by the better of their base score and relational score."""
        if not query.strip():
            raise ValueError("Ontology retrieval query cannot be empty.")
        if limit < 1:
            raise ValueError("Retrieval limit must be at least 1.")
        base_scores = base_scores or {}
        reachable = self.reachable(self.concepts_in(query))

        matches: list[OntologyMatch] = []
        for item_id, text in items.items():
            item_concepts = self.concepts_in(text)
            ontology_score = 0.0
            best_hops = -1
            best_link: tuple[str, str, str] | None = None
            for concept_id in item_concepts:
                if concept_id in reachable:
                    hops, weight, link = reachable[concept_id]
                    if weight > ontology_score:
                        ontology_score = weight
                        best_hops = hops
                        best_link = link
            score = max(base_scores.get(item_id, 0.0), ontology_score)
            if score <= 0.0:
                continue
            matches.append(
                OntologyMatch(
                    item_id=item_id,
                    score=round(score, 6),
                    hops=best_hops,
                    concepts=tuple(sorted(item_concepts)),
                    link=best_link if best_hops > 0 else None,
                )
            )

        matches.sort(key=lambda match: (-match.score, match.item_id))
        return matches[:limit]

    def _edges(self) -> dict[str, list[tuple[str, float, tuple[str, str, str]]]]:
        if self._adjacency is None:
            edges: dict[str, list[tuple[str, float, tuple[str, str, str]]]] = {}
            for relation in self.ontology.relations.values():
                link = (relation.source_id, relation.relation_type, relation.target_id)
                edges.setdefault(relation.source_id, []).append((relation.target_id, relation.confidence, link))
            for inference in self.reasoner.derive(self.ontology):
                link = (inference.source_id, inference.relation_type, inference.target_id)
                edges.setdefault(inference.source_id, []).append((inference.target_id, inference.confidence, link))
            self._adjacency = edges
        return self._adjacency


__all__ = ["OntologyGuidedRetriever", "OntologyMatch"]
