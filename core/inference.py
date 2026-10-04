from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Inference:
    """A bounded request to derive or explain a statement."""
    facts: tuple[str, ...]
    query: str | None = None

    def __post_init__(self) -> None:
        if any(not fact.strip() for fact in self.facts):
            raise ValueError("Inference facts cannot be empty.")
        if self.query is not None and not self.query.strip():
            raise ValueError("Inference query cannot be blank.")
