"""Dataset isolation for Phase 6: who a dataset may draw from, and the Natasha rule.

Phase 5 already made the isolation policy structural (a capability ceiling, a
pre-path guard, a fixed owner list). This module is the **training-side** of that
policy: it answers, for a dataset, whether its membership is legal, and it proves
the two hard requirements of Phase 6:

* a personality's training dataset is **independently scoped** -- it may only
  contain examples from owners it is allowed to read;
* the project model's **aggregated** dataset contains **no** private owner
  (Natasha), by construction and by an explicit check.

Nothing here trusts a manifest. :func:`audit_dataset` re-derives the answer from
the examples themselves and the live isolation policy, so a hand-edited manifest
cannot make an illegal dataset legal.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from core.agents.learning_scope import (
    aggregated_dataset_is_clean,
    aggregated_dataset_owners,
    learning_scope,
)
from core.learning_isolation import (
    is_owner,
    is_private,
)

from ml.training.datasets import DatasetSplit

#: The reason a private owner is excluded, shown wherever the rule is explained.
NATASHA_EXCLUSION_REASON = (
    "A private personality's experience, memory, feedback, and training records are "
    "never an aggregated training source for the project model."
)


@dataclass(frozen=True)
class IsolationVerdict:
    """Whether a dataset's membership is legal, and why not when it is not."""

    ok: bool
    reason: str = ""
    owners: tuple[str, ...] = ()
    foreign: tuple[str, ...] = ()
    private: tuple[str, ...] = ()

    def describe(self) -> str:
        if self.ok:
            scope = ", ".join(self.owners) or "project"
            return f"isolation ok: members from {scope}"
        return f"isolation refused: {self.reason}"


def dataset_owners(split: DatasetSplit) -> tuple[str, ...]:
    """Every owner that contributed at least one example, in a stable order."""
    seen: dict[str, None] = {}
    for example in split.all_examples():
        if example.owner:
            seen.setdefault(example.owner, None)
    return tuple(seen)


def audit_dataset(
    split: DatasetSplit,
    *,
    declared_scope: Sequence[str] = (),
    for_main_model: bool = False,
) -> IsolationVerdict:
    """Check a dataset's membership against the isolation policy.

    A dataset is refused when:

    * an example names an **unknown** owner;
    * an example's owner is outside the **declared scope**;
    * it is destined for the **project model** and contains a **private** owner;
    * it is destined for a **single character** and contains another character's
      examples (a personality may not absorb another's experience).
    """
    owners = dataset_owners(split)
    scope = tuple(dict.fromkeys(declared_scope))

    unknown = tuple(o for o in owners if not is_owner(o))
    if unknown:
        return IsolationVerdict(
            ok=False,
            reason=f"unknown owner(s): {', '.join(unknown)}",
            owners=owners,
            foreign=unknown,
        )

    if scope:
        foreign = tuple(o for o in owners if o not in scope)
        if foreign:
            return IsolationVerdict(
                ok=False,
                reason=(
                    f"member(s) {', '.join(foreign)} are outside the declared scope "
                    f"{scope}"
                ),
                owners=owners,
                foreign=foreign,
            )
    elif len(owners) > 1:
        # No declared scope means a single-personality dataset; two characters'
        # examples may not be silently mixed.
        return IsolationVerdict(
            ok=False,
            reason=(
                f"a single-personality dataset may not mix {', '.join(owners)}; "
                "declare a scope to build an aggregate"
            ),
            owners=owners,
            foreign=owners[1:],
        )

    private = tuple(o for o in owners if is_private(o))
    if for_main_model and private:
        return IsolationVerdict(
            ok=False,
            reason=f"the project model's dataset contains private owner(s): "
            f"{', '.join(private)}. {NATASHA_EXCLUSION_REASON}",
            owners=owners,
            private=private,
        )
    return IsolationVerdict(ok=True, owners=owners)


def main_model_dataset_owners() -> tuple[str, ...]:
    """The owners the project model may aggregate -- Natasha is never among them."""
    return aggregated_dataset_owners()


def main_model_dataset_is_clean() -> bool:
    """True when no private owner is present in the aggregate (the Phase-6 gate)."""
    return aggregated_dataset_is_clean()


def can_learn_from(reader: str, owner: str) -> bool:
    """Whether ``reader`` may use ``owner``'s records as a training source."""
    if not is_owner(reader) or not is_owner(owner):
        return False
    if reader == owner:
        return True
    scope = learning_scope(reader)
    return owner in scope.learns_from


def assert_main_model_excludes_private(split: DatasetSplit) -> None:
    """Raise when a dataset destined for the project model contains a private owner."""
    verdict = audit_dataset(split, for_main_model=True)
    if not verdict.ok:
        raise PermissionError(verdict.reason)


__all__ = [
    "NATASHA_EXCLUSION_REASON",
    "IsolationVerdict",
    "assert_main_model_excludes_private",
    "audit_dataset",
    "can_learn_from",
    "dataset_owners",
    "main_model_dataset_is_clean",
    "main_model_dataset_owners",
]
