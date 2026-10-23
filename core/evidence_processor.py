from __future__ import annotations

from enum import Enum

from core.evidence import Evidence
from core.observation import Observation


class EvidencePolarity(str, Enum):
    SUPPORTS = "supports"
    CONTRADICTS = "contradicts"
    NEUTRAL = "neutral"


class EvidenceProcessor:
    """
    Converts an observation into evidence with an explicit polarity.
    """

    def process(
        self,
        observation: Observation,
        polarity: EvidencePolarity,
    ) -> Evidence:

        if polarity == EvidencePolarity.SUPPORTS:
            supports = True

        elif polarity == EvidencePolarity.CONTRADICTS:
            supports = False

        else:
            # Neutral evidence is represented separately.
            # It does not directly support or contradict a belief.
            supports = True

        return Evidence(
            description=observation.content,
            source=observation.source,
            supports=supports,
            strength=(
                observation.confidence
                if polarity != EvidencePolarity.NEUTRAL
                else 0.0
            ),
            data={
                **observation.data,
                "polarity": polarity.value,
                "observation_id": observation.id,
            },
        )
