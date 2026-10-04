from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class ParsedSemanticStatement:
    subject: str
    relation: str
    object: str
    confidence: float = .5


class SemanticParser:
    """Minimal transparent parser for simple English and Arabic copular statements."""
    def parse(self, text: str) -> ParsedSemanticStatement | None:
        value = text.strip().rstrip(".")
        for pattern, relation in ((r"^(.+?)\s+is a[n]?\s+(.+)$", "is_a"), (r"^(.+?)\s+causes\s+(.+)$", "causes"), (r"^(.+?)\s+هو\s+(.+)$", "is_a"), (r"^(.+?)\s+يسبب\s+(.+)$", "causes")):
            match = re.match(pattern, value, flags=re.IGNORECASE)
            if match:
                return ParsedSemanticStatement(match.group(1).strip(), relation, match.group(2).strip())
        return None
