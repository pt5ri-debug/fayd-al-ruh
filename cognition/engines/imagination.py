"""Imagination: future scenarios and counterfactuals, clearly labelled as such.

The project already has a real counterfactual reasoner
(:class:`core.counterfactual_reasoner.CounterfactualReasoner`) and a causal graph.
This engine uses them to build *scenarios* -- what would happen if a cause were
removed or a link changed -- and states each as a candidate, never as a fact.

It also builds **speculative** scenarios that are honest about being
speculation: it names the concepts involved and says plainly that the connection
is imagined. That is the difference between imagination and a claim.
"""

from __future__ import annotations

from dataclasses import dataclass

from core.causal_graph import CausalGraph
from core.counterfactual_reasoner import CounterfactualReasoner
from core.ontology import Ontology

from cognition.engines.association import AssociationEngine


@dataclass(frozen=True)
class Scenario:
    """A future possibility or counterfactual, with how it was produced."""

    description: str
    kind: str
    confidence: float
    evidence: str = ""

    def __post_init__(self) -> None:
        if not self.description.strip():
            raise ValueError("Scenario description cannot be empty.")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("Scenario confidence must be between 0 and 1.")

    def describe(self) -> str:
        return f"[{self.kind}] {self.description} ({self.confidence:.2f})"


class ImaginationEngine:
    """Build counterfactuals from the causal graph and speculative scenarios."""

    def __init__(
        self,
        ontology: Ontology,
        *,
        causal_graph: CausalGraph | None = None,
        associations: AssociationEngine | None = None,
        reasoner: CounterfactualReasoner | None = None,
    ) -> None:
        self.ontology = ontology
        self.associations = associations or AssociationEngine(ontology)
        self.causal_graph = causal_graph
        self.reasoner = reasoner or CounterfactualReasoner()

    def counterfactuals(self, *, limit: int = 5) -> tuple[Scenario, ...]:
        """For each causal link, what if the cause were removed?

        Only links the causal graph actually holds are used, and each scenario is
        labelled ``counterfactual`` with the link as its evidence. The confidence
        is the reasoner's own counterfactual figure, discounted by the link's
        confidence, so a scenario resting on a weak link is honestly weaker.
        Returns ``()`` when no causal graph was supplied or it holds no link.
        """
        if self.causal_graph is None:
            return ()
        scenarios: list[Scenario] = []
        for link in list(self.causal_graph.links.values())[:limit]:
            if not getattr(link, "active", True):
                continue
            confidence = 0.0
            try:
                result = self.reasoner.what_if_not(
                    self.causal_graph, link.cause, link.effect
                )
                confidence = float(getattr(result, "counterfactual_confidence", 0.0))
            except Exception:
                confidence = 0.0
            description = (
                f"If {link.cause} were removed, {link.effect} may fall to "
                f"{confidence:.2f} of its factual strength"
            )
            scenarios.append(
                Scenario(
                    description=description,
                    kind="counterfactual",
                    confidence=min(
                        0.7, confidence * float(getattr(link, "confidence", 0.5)) or 0.3
                    ),
                    evidence=f"{link.cause} -> {link.effect}",
                )
            )
        return tuple(scenarios)

    def speculate(self, source: str, target: str) -> Scenario | None:
        """An honestly-labelled *speculative* link between two concepts.

        This is imagination, not inference: it is offered only when the two
        concepts are at least reachable in the graph (so it is not arbitrary), and
        it says plainly that the connection is imagined.
        """
        found = self.associations.between(source, target)
        if not found:
            return None
        return Scenario(
            description=(
                f"Imagine a study that treats {source} and {target} together; "
                f"the graph already relates them ({found[0].kind})"
            ),
            kind="speculative",
            confidence=0.3,
            evidence=found[0].describe(),
        )

    def generate(
        self,
        *,
        labels: tuple[str, ...] = (),
        limit: int = 5,
    ) -> tuple[Scenario, ...]:
        """Counterfactuals first (real evidence), then speculation, bounded."""
        scenarios: list[Scenario] = list(self.counterfactuals(limit=limit))
        remaining = max(0, limit - len(scenarios))
        if remaining and len(labels) >= 2:
            for association in self.associations.rank_pairs(labels, limit=remaining):
                scenario = self.speculate(association.source, association.target)
                if scenario is not None:
                    scenarios.append(scenario)
                if len(scenarios) >= limit:
                    break
        return tuple(scenarios[:limit])


__all__ = ["ImaginationEngine", "Scenario"]
