from __future__ import annotations

import re

from core.concept import Concept
from core.ontology import Ontology


class MeaningModel:
    """Resolves lexical forms to known concepts using labels and aliases."""
    def resolve(self, text: str, ontology: Ontology) -> list[Concept]:
        tokens = set(re.findall(r"\w+", text.casefold(), flags=re.UNICODE))
        return [concept for concept in ontology.concepts.values() if any(set(re.findall(r"\w+", term, flags=re.UNICODE)) <= tokens for term in concept.terms)]

    def similarity(self, first: Concept, second: Concept) -> float:
        overlap = first.terms & second.terms
        union = first.terms | second.terms
        return round(len(overlap) / len(union), 6) if union else 0.0
