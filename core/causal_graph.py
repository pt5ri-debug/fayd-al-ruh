from __future__ import annotations

from core.causal_link import CausalLink


class CausalGraph:
    """Stores causal claims separately from ordinary correlations/relations."""

    def __init__(self) -> None:
        self.links: dict[str, CausalLink] = {}

    def add_link(self, link: CausalLink) -> None:
        if link.id in self.links:
            raise ValueError(f"Causal link '{link.id}' already exists.")
        self.links[link.id] = link

    def get_link(self, link_id: str) -> CausalLink | None:
        return self.links.get(link_id)

    def remove_link(self, link_id: str) -> None:
        self.links.pop(link_id, None)

    def causes_of(self, effect: str) -> list[CausalLink]:
        return [link for link in self.links.values() if link.active and link.effect == effect]

    def effects_of(self, cause: str) -> list[CausalLink]:
        return [link for link in self.links.values() if link.active and link.cause == cause]

    @property
    def link_count(self) -> int:
        return len(self.links)
