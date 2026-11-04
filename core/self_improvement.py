"""Controlled self-improvement: the reviewable plan, and the gated loop.

This module began as the smallest possible thing the task allowed -- package
:class:`~core.architecture_learning.ArchitectureRecommendation` objects into a
plan that **requires human review** -- and it keeps that behaviour exactly, so
every existing caller and test is untouched.

Phase 9 extends it with the one facade the rest of the project uses:
:func:`improvement_loop` returns the composed
:class:`~core.improvement.loop.ImprovementLoop`, and :class:`SelfImprovement`
gains thin delegations to it. The guarantee the original class was written for is
preserved and made structural: ``requires_human_review`` stays ``True``, and the
loop it drives stops at the human approval step before anything is deployed.

Nothing here imports Tk, loads a model, or reaches the network.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from core.architecture_learning import ArchitectureRecommendation
from core.improvement.loop import ImprovementLoop, default_loop


@dataclass(frozen=True)
class ImprovementPlan:
    recommendations: tuple[ArchitectureRecommendation, ...]
    requires_human_review: bool = True


class SelfImprovement:
    """Packages recommendations for review rather than self-modifying code or policy.

    Phase 9 adds :meth:`loop` (the composed, gated loop) and :meth:`analyse`
    (detect → propose → rank) as thin delegations, so a caller can reach the full
    framework through the same object it already had. The class itself still
    changes nothing: it produces a plan, and a plan is data.
    """

    def propose(self, recommendations: list[ArchitectureRecommendation]) -> ImprovementPlan:
        return ImprovementPlan(tuple(recommendations))

    # -- Phase 9: the controlled loop --------------------------------------
    def loop(
        self,
        root: Path | str | None = None,
        *,
        state_root: Path | str | None = None,
        **kwargs: object,
    ) -> ImprovementLoop:
        """Build the controlled self-improvement loop over a checkout.

        ``root`` is the checkout the loop may isolate (a git worktree is created
        there); ``state_root`` is where the loop keeps its own data. Both default
        to the project's own locations.
        """
        if root is None and state_root is None:
            return default_loop(**kwargs)  # type: ignore[arg-type]
        resolved = Path(root) if root is not None else Path(__file__).resolve().parents[1]
        return ImprovementLoop(root=resolved, state_root=state_root, **kwargs)  # type: ignore[arg-type]

    def analyse(
        self,
        loop: ImprovementLoop,
        *,
        failures: tuple[object, ...] = (),
    ) -> tuple[tuple[object, ...], tuple[object, ...]]:
        """Detect weaknesses and rank the proposals that answer them."""
        return loop.analyse(failures=failures)  # type: ignore[arg-type]


def improvement_loop(
    root: Path | str | None = None,
    *,
    state_root: Path | str | None = None,
    **kwargs: object,
) -> ImprovementLoop:
    """The one entry point to the controlled self-improvement loop.

    Kept as a module-level function as well as a method so a caller can reach the
    loop without constructing :class:`SelfImprovement` first.
    """
    return SelfImprovement().loop(root, state_root=state_root, **kwargs)


__all__ = [
    "ImprovementPlan",
    "SelfImprovement",
    "improvement_loop",
]
