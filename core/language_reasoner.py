from __future__ import annotations

from core.ontology import Ontology


class LanguageReasoner:
    """Performs explicit ontology reasoning, currently transitive is_a inference."""
    def is_a(self, ontology: Ontology, child_id: str, parent_id: str) -> bool:
        visited: set[str] = set()

        def visit(current: str) -> bool:
            if current in visited:
                return False
            visited.add(current)
            for relation in ontology.outgoing(current, "is_a"):
                if relation.target_id == parent_id or visit(relation.target_id):
                    return True
            return False

        return visit(child_id)

    def related_meanings(self, ontology: Ontology, concept_id: str) -> list[str]:
        return [relation.target_id for relation in ontology.outgoing(concept_id)]
