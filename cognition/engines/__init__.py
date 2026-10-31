"""The engines of the Proactive Mind, one module per cognitive faculty.

Each engine is small, dependency-free, and does one job over the project's own
knowledge graph, memory, and goals. They are composed by
:class:`cognition.proactive_mind.ProactiveMind`, which is the only object that
runs them together.

* :mod:`cognition.engines.curiosity` -- what is worth finding out.
* :mod:`cognition.engines.association` -- what concepts are related.
* :mod:`cognition.engines.ideas` -- recombination into new statements.
* :mod:`cognition.engines.imagination` -- future scenarios, counterfactuals.
* :mod:`cognition.engines.criticism` -- self-review, contradictions, weak spots.
* :mod:`cognition.engines.motivation` -- which topics to pursue, and why.
* :mod:`cognition.engines.questions` -- turning gaps into questions.
"""

from __future__ import annotations

__all__ = [
    "association",
    "criticism",
    "curiosity",
    "ideas",
    "imagination",
    "motivation",
    "questions",
]
