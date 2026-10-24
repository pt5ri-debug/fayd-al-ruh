from __future__ import annotations

from core.contradiction import Contradiction
from core.evidence import Evidence
from core.evidence_weighting import EvidenceWeighting
from core.relation_polarity import opposes


class ContradictionDetector:
    """Finds evidence that pulls the same belief in opposing directions.

    The original rule pairs every supporting record with every contradicting one.
    Two refinements keep it honest when evidence carries *relation* metadata
    (``data["relation_type"]``, and optionally ``data["source_id"]`` /
    ``data["target_id"]``):

    * two records that assert *antonymous* relations on the same ordered pair
      (``increases`` vs ``reduces``) are a contradiction even when both set
      ``supports=True`` -- the relation itself says they disagree;
    * two records that assert the *same* relation type (or a non-antonymous one)
      are not a contradiction merely because one is flagged as supporting and the
      other as not, because they may be about different claims.
    """

    def __init__(self, weighting: EvidenceWeighting | None = None) -> None:
        self.weighting = weighting or EvidenceWeighting()

    def detect(self, evidence_items: list[Evidence]) -> list[Contradiction]:
        found: list[Contradiction] = []
        for index, first in enumerate(evidence_items):
            for second in evidence_items[index + 1 :]:
                if self._contradicts(first, second):
                    found.append(
                        Contradiction(
                            first.id,
                            second.id,
                            round(min(self.weighting.weight(first), self.weighting.weight(second)), 6),
                        )
                    )
        return found

    @classmethod
    def _contradicts(cls, first: Evidence, second: Evidence) -> bool:
        """Whether two evidence records genuinely oppose each other."""
        first_axis, second_axis = cls._axis(first), cls._axis(second)
        if first_axis is not None and second_axis is not None:
            # Both carry a typed relation: they oppose exactly when the pair and
            # direction match and the two types are antonyms. The support flag is
            # not consulted -- a typed relation is the stronger statement.
            return first_axis[:2] == second_axis[:2] and opposes(first_axis[2], second_axis[2])
        # Otherwise fall back to the original support-flag rule.
        return first.supports != second.supports

    @staticmethod
    def _axis(evidence: Evidence) -> tuple[str, str, str] | None:
        data = evidence.data or {}
        relation_type = str(data.get("relation_type", "")).strip()
        source_id = str(data.get("source_id", "")).strip()
        target_id = str(data.get("target_id", "")).strip()
        if not relation_type or not source_id or not target_id:
            return None
        return source_id, target_id, relation_type
