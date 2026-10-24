"""Hakeem's durable memory: a thin, named layer over the companion's own.

Memory is the program's job, not the model's, and the companion layer already
implements that job well: it keeps a memory's *kind* (a preference, a fact the
user asked to keep, a summary, or a detail about the character), its *source and
confidence*, and the **fiction/reality** split, refuses sensitive personal data by
default, and retrieves only the few most relevant items.

Hakeem is a different character on the same *model*, not a different memory
engine, so this module does not re-implement any of that: it re-exports the
companion's own :class:`~companion.memory_manager.MemoryManager` and its kinds,
under Hakeem's namespace. Hakeem's memories live in **his own database** (a
separate file from the companion's), so the two characters never share a memory
even though they share the code that stores it.

Nothing here writes to the project's entities, relations, evidence, beliefs, or
knowledge graph.
"""

from __future__ import annotations

from companion.memory_manager import (
    DEFAULT_CONFIDENCE,
    KIND_COMPANION,
    KIND_FACT,
    KIND_PREFERENCE,
    KIND_SUMMARY,
    KINDS,
    Memory,
    MemoryCandidate,
    MemoryManager,
    is_save_request,
    looks_sensitive,
)

__all__ = [
    "DEFAULT_CONFIDENCE",
    "KINDS",
    "KIND_COMPANION",
    "KIND_FACT",
    "KIND_PREFERENCE",
    "KIND_SUMMARY",
    "Memory",
    "MemoryCandidate",
    "MemoryManager",
    "is_save_request",
    "looks_sensitive",
]
