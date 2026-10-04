from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from core.causal_chain import CausalChain
from core.causal_graph import CausalGraph


@dataclass(frozen=True)
class CausalValidation:
    valid: bool
    errors: tuple[str, ...]


class CausalValidator:
    """Validates evidence references and directed chain continuity."""

    def validate_graph(self, graph: CausalGraph, evidence_ids: Iterable[str] | None = None) -> CausalValidation:
        known_evidence = set(evidence_ids) if evidence_ids is not None else None
        errors: list[str] = []
        for link in graph.links.values():
            if known_evidence is not None:
                for evidence_id in link.evidence_ids:
                    if evidence_id not in known_evidence:
                        errors.append(f"Link {link.id}: missing evidence {evidence_id}.")
        return CausalValidation(not errors, tuple(errors))

    def validate_chain(self, chain: CausalChain) -> CausalValidation:
        errors = []
        if len(set(chain.nodes)) != len(chain.nodes):
            errors.append("Causal chain contains a cycle.")
        return CausalValidation(not errors, tuple(errors))
