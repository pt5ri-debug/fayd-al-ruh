"""Polarity of a relation type: which typed relations can oppose each other.

The graph can already hold a *typed* relation (``porosity --increases-->
permeability``, read from text by :mod:`core.relation_extraction` or seeded in
:mod:`core.domain_knowledge`). What nothing knew until now is that some relation
types *oppose* others: ``increases`` is the opposite of ``reduces``, ``causes``
the opposite of ``prevents``. Without that, two sentences that say the opposite
thing about the same pair -- "waterflood increases recovery" and "waterflood
reduces recovery" -- looked like two unrelated links rather than a contradiction.

This module is the one declarative place that fact lives. It is deliberately
small and deterministic: a relation type carries a **polarity** drawn from a fixed
vocabulary, and two types oppose when they are the two poles of the same
*axis* -- the same subject-to-object direction. It never guesses: an unknown
relation type has no polarity and therefore opposes nothing.

Honest by construction: a *symmetric* relation (``associated_with``,
``depends_on``) is not on an axis, so it can never be reported as opposing
another; only an explicitly antonymous pair is.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Protocol


class _Relation(Protocol):
    """The shape :func:`find_polarity_conflicts` needs (a SemanticRelation fits).

    The members are read-only properties so a frozen ``SemanticRelation`` is
    structurally compatible.
    """

    @property
    def source_id(self) -> str: ...

    @property
    def target_id(self) -> str: ...

    @property
    def relation_type(self) -> str: ...

    @property
    def confidence(self) -> float: ...

#: The polarity an antonymous relation type carries. A type not listed has no
#: polarity (``None``) and opposes nothing -- the honest default for a relation
#: the model was never told is directional.
POLARITY: dict[str, str] = {
    # A positive/negative effect on the object's quantity or quality.
    "increases": "positive",
    "reduces": "negative",
    # Bringing something about vs. holding it off.
    "causes": "positive",
    "prevents": "negative",
    # A relation that asserts a link vs. one that denies it.
    "supports": "positive",
    "contradicts": "negative",
}

#: The two relation types that are antonyms of each other, as unordered pairs.
#: Only a pair listed here can be reported as a polarity conflict; two unrelated
#: relation types are never called contradictory.
OPPOSING_PAIRS: tuple[frozenset[str], ...] = (
    frozenset({"increases", "reduces"}),
    frozenset({"causes", "prevents"}),
    frozenset({"supports", "contradicts"}),
)

#: Relation types that are symmetric: ``A --X--> B`` says the same as
#: ``B --X--> A``. These are never on a polarity axis, so they can neither oppose
#: nor be opposed -- recorded explicitly so the refusal is a fact, not an
#: omission.
SYMMETRIC_RELATIONS: frozenset[str] = frozenset({"associated_with", "related_to"})


@dataclass(frozen=True)
class PolarityConflict:
    """Two relations on the same pair that assert opposite things."""

    source_id: str
    target_id: str
    first_relation: str
    second_relation: str
    severity: float

    def __post_init__(self) -> None:
        if not 0.0 <= self.severity <= 1.0:
            raise ValueError("Polarity conflict severity must be between 0 and 1.")

    @property
    def pair(self) -> tuple[str, str]:
        return self.source_id, self.target_id


def polarity_of(relation_type: str) -> str | None:
    """The polarity of a relation type, or ``None`` when it has none."""
    return POLARITY.get(relation_type.strip().casefold())


def is_symmetric(relation_type: str) -> bool:
    """Whether a relation type is symmetric (and therefore on no polarity axis)."""
    return relation_type.strip().casefold() in SYMMETRIC_RELATIONS


def opposes(first: str, second: str) -> bool:
    """Whether two relation types are antonyms of each other.

    True only for an explicitly listed antonymous pair, so ``increases`` opposes
    ``reduces`` while ``increases`` and ``measures`` do not oppose. A type is
    never its own opposite.
    """
    left, right = first.strip().casefold(), second.strip().casefold()
    if left == right or is_symmetric(left) or is_symmetric(right):
        return False
    return frozenset({left, right}) in OPPOSING_PAIRS


def conflicts(
    source_id: str,
    target_id: str,
    relation_a: str,
    relation_b: str,
    *,
    confidence_a: float = 1.0,
    confidence_b: float = 1.0,
) -> PolarityConflict | None:
    """A :class:`PolarityConflict` for two opposing relations on the same pair.

    The severity is the weaker of the two confidences -- a conflict is only as
    strong as the evidence for the claim that is more weakly held. Returns
    ``None`` when the two relations are not antonyms, or when they run in
    opposite directions (``A increases B`` does not conflict with
    ``B reduces A``: those are two different claims).
    """
    if not opposes(relation_a, relation_b):
        return None
    severity = round(min(max(0.0, confidence_a), max(0.0, confidence_b)), 6)
    # Canonical order: the two types are reported sorted, so the same conflict is
    # the same value whichever relation was seen first.
    first_relation, second_relation = sorted((relation_a.strip().casefold(), relation_b.strip().casefold()))
    return PolarityConflict(source_id, target_id, first_relation, second_relation, min(severity, 1.0))


def find_polarity_conflicts(relations: Iterable[_Relation]) -> list[PolarityConflict]:
    """Every opposing pair among ``relations`` that runs the same direction.

    ``relations`` is any iterable of objects carrying ``source_id``, ``target_id``,
    ``relation_type`` and ``confidence`` (a :class:`core.semantic_relation.
    SemanticRelation`, or any duck-typed equivalent). Two relations conflict only
    when they share the *same ordered pair* and carry antonymous types, so
    ``porosity --increases--> permeability`` conflicts with
    ``porosity --reduces--> permeability`` but not with
    ``permeability --reduces--> porosity`` (a different claim) nor with an
    unrelated type. Deterministic: ordered by the pair, then the two types.
    """
    by_pair: dict[tuple[str, str], list[_Relation]] = {}
    for relation in relations:
        by_pair.setdefault((relation.source_id, relation.target_id), []).append(relation)
    found: list[PolarityConflict] = []
    for (source_id, target_id), group in by_pair.items():
        for index, first in enumerate(group):
            for second in group[index + 1 :]:
                conflict = conflicts(
                    source_id,
                    target_id,
                    first.relation_type,
                    second.relation_type,
                    confidence_a=first.confidence,
                    confidence_b=second.confidence,
                )
                if conflict is not None:
                    found.append(conflict)
    found.sort(key=lambda item: (item.source_id, item.target_id, item.first_relation, item.second_relation))
    return found


class PolarityJudgement:
    """One held-out pair of relations and whether they should be a conflict.

    ``expected`` is the pair the two relation types are asserted on -- the
    ``(source_id, target_id, relation_a, relation_b)`` -- or ``None`` when the two
    must *not* conflict (a negative control: an unrelated type, a symmetric
    relation, or the same type twice). ``note`` records why.
    """

    __slots__ = ("expected", "note", "relation_a", "relation_b", "source_id", "target_id")

    def __init__(
        self,
        source_id: str,
        target_id: str,
        relation_a: str,
        relation_b: str,
        expected: tuple[str, str, str, str] | None,
        note: str = "",
    ) -> None:
        if not source_id.strip() or not target_id.strip():
            raise ValueError("A polarity judgement needs a source and a target.")
        if not relation_a.strip() or not relation_b.strip():
            raise ValueError("A polarity judgement needs two relation types.")
        self.source_id = source_id
        self.target_id = target_id
        self.relation_a = relation_a
        self.relation_b = relation_b
        self.expected = expected
        self.note = note


#: The held-out polarity catalogue -- external to the model (written by hand,
#: with negative controls), so a change that makes the polarity model over-fire
#: or miss is a measured regression rather than a silent one.
POLARITY_JUDGEMENTS: tuple[PolarityJudgement, ...] = (
    PolarityJudgement(
        "porosity", "permeability", "increases", "reduces",
        ("porosity", "permeability", "increases", "reduces"),
        "the canonical antonym pair",
    ),
    PolarityJudgement(
        "waterflood", "recovery_factor", "causes", "prevents",
        ("waterflood", "recovery_factor", "causes", "prevents"),
        "bringing about vs holding off",
    ),
    PolarityJudgement(
        "porosity", "permeability", "increases", "measures", None,
        "two unrelated relation types never conflict",
    ),
    PolarityJudgement(
        "porosity", "permeability", "increases", "increases", None,
        "a relation type is never its own opposite",
    ),
    PolarityJudgement(
        "porosity", "permeability", "associated_with", "associated_with", None,
        "a symmetric relation is on no polarity axis",
    ),
)


def polarity_accuracy(judgements: Iterable[PolarityJudgement] = POLARITY_JUDGEMENTS) -> float:
    """The share of held-out judgements the polarity model gets right.

    A judgement counts when :func:`conflicts` returns a conflict exactly when the
    judgement expects one and names exactly the expected pair and the two
    expected types (in the model's canonical order) -- so an over-firing model
    fails as surely as one that misses, and a negative control is a real test.
    Returns ``0.0`` for an empty catalogue.
    """
    catalogue = tuple(judgements)
    if not catalogue:
        return 0.0
    correct = 0
    for judgement in catalogue:
        found = conflicts(
            judgement.source_id, judgement.target_id, judgement.relation_a, judgement.relation_b
        )
        if judgement.expected is None:
            if found is None:
                correct += 1
        elif found is not None and (found.source_id, found.target_id, found.first_relation, found.second_relation) == judgement.expected:
            correct += 1
    return correct / len(catalogue)


__all__ = [
    "OPPOSING_PAIRS",
    "POLARITY",
    "POLARITY_JUDGEMENTS",
    "SYMMETRIC_RELATIONS",
    "PolarityConflict",
    "PolarityJudgement",
    "conflicts",
    "find_polarity_conflicts",
    "is_symmetric",
    "opposes",
    "polarity_accuracy",
    "polarity_of",
]
