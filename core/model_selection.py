"""Choosing a model for a task, and falling back when it will not run.

The manager already answers "which models can run here?" (:meth:`ModelManager.
ready_models`). This module answers the next two questions, and keeps them apart:

* **Which of them should this task use?** :func:`rank_for_task` scores every
  candidate against a :class:`TaskProfile` and returns a **ranked list with the
  reasons** -- a score without its reasoning is a number a user cannot argue with,
  so :class:`RankedModel` carries the factors that produced it. The ranking is
  deterministic: the same inputs always give the same order, which is what makes it
  testable and what stops a choice from wobbling between two equal models.
* **What happens when the chosen one fails?** :func:`plan_fallback` builds a
  *bounded* ladder: the preferred model, then each remaining candidate that could
  answer, then -- for a task with no local model at all -- an honest statement that
  nothing here can, rather than a hidden dependency on a cloud service. Every rung
  names what would be tried and, when the ladder cannot reach the task, why not.

The one rule throughout: **nothing is invented.** A candidate is a model the
manager already knows is present, a task is one the build declares, and a fallback
is only ever another real candidate. Nothing here downloads, loads, or starts
anything -- it returns a plan for a caller to follow.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

#: The three dimensions a task profile scores, in the order they are described.
FACTOR_ENGINE = "engine"
FACTOR_TASK_MATCH = "task-match"
FACTOR_CAPABILITY = "capability"
FACTOR_FIT = "fit"
FACTOR_PREFERENCE = "preference"
FACTOR_HEALTH = "health"

#: How much each factor contributes. The weights sum to 1.0, so a score is a
#: fraction and two scores are comparable across tasks.
FACTOR_WEIGHTS: dict[str, float] = {
    FACTOR_ENGINE: 0.30,
    FACTOR_TASK_MATCH: 0.25,
    FACTOR_CAPABILITY: 0.15,
    FACTOR_FIT: 0.15,
    FACTOR_PREFERENCE: 0.10,
    FACTOR_HEALTH: 0.05,
}

#: What a task needs a model to be able to do, by capability name. A task with an
#: empty requirement is satisfied by any ready model.
TASK_CAPABILITIES: dict[str, tuple[str, ...]] = {
    "chat": (),
    "code": ("code",),
    "presentation": ("text-generation",),
    "petroleum": ("text-generation",),
    "companion": ("chat",),
    "religion": ("chat",),
}

#: How much context a task is assumed to want, in tokens. Used only for the *fit*
#: factor -- a model that cannot hold this much context is not disqualified, it is
#: merely ranked lower, because a shorter window still answers.
TASK_CONTEXT_PREFERENCE: dict[str, int] = {
    "chat": 4096,
    "code": 8192,
    "presentation": 4096,
    "petroleum": 4096,
    "companion": 4096,
    "religion": 4096,
}

#: A capability a task requires but no model can be *proven* to have scores this
#: fraction rather than zero: an unstated capability is unknown, not absent.
UNKNOWN_CAPABILITY_SCORE = 0.5


@dataclass(frozen=True)
class TaskProfile:
    """What a task needs from a model.

    ``engines`` are the mouth engines the task names (``banner`` for a
    presentation, ``vision`` for petroleum); ``capabilities`` are the abilities it
    needs; ``context`` is the window it would like; ``resident`` says whether the
    task wants a model kept loaded. All of it is **data**, so a caller can add a
    profile without changing the ranking code, and a test can rank against a
    profile of its own.
    """

    task: str
    engines: tuple[str, ...] = ()
    capabilities: tuple[str, ...] = ()
    context: int = 4096
    #: True when the task benefits from a model that stays resident between calls.
    resident: bool = True
    description: str = ""

    def to_primitive(self) -> dict[str, Any]:
        return {
            "capabilities": list(self.capabilities),
            "context": self.context,
            "description": self.description,
            "engines": list(self.engines),
            "resident": self.resident,
            "task": self.task,
        }


def task_profile(task: str, *, engines: tuple[str, ...] | None = None) -> TaskProfile:
    """The profile for a task, or ``None``-shaped data for an unknown one.

    An unknown task yields no profile rather than a default one: a caller asking
    about a task this build does not declare is told so, and no ranking is
    invented to fill the gap.
    """
    from core.model_manager import TASKS

    key = str(task or "").strip()
    if key not in TASKS:
        return TaskProfile(task=key, description="This build does not declare this task.")
    return TaskProfile(
        task=key,
        engines=tuple(engines) if engines is not None else tuple(TASKS[key]),
        capabilities=TASK_CAPABILITIES.get(key, ()),
        context=TASK_CONTEXT_PREFERENCE.get(key, 4096),
        description=f"The {key} task.",
    )


def is_known_task(task: str) -> bool:
    """Whether this build declares ``task``."""
    from core.model_manager import TASKS

    return str(task or "").strip() in TASKS


@dataclass(frozen=True)
class RankedModel:
    """One candidate, with the score and the reasons behind it.

    ``factors`` is the per-factor contribution, so a reader can see *why* one model
    outranked another; ``notes`` says what could not be scored. Neither is derived
    after the fact -- both are produced by the same pass that made the score.
    """

    model_id: str
    display_name: str
    score: float
    factors: dict[str, float] = field(default_factory=dict)
    notes: tuple[str, ...] = field(default_factory=tuple)

    def describe(self) -> str:
        line = f"{self.display_name} — {self.score:.3f}"
        if self.factors:
            parts = ", ".join(
                f"{name} {value:.2f}" for name, value in sorted(self.factors.items())
            )
            line = f"{line} ({parts})"
        if self.notes:
            line = f"{line} — {'; '.join(self.notes)}"
        return line

    def to_primitive(self) -> dict[str, Any]:
        return {
            "display_name": self.display_name,
            "factors": {name: round(value, 4) for name, value in self.factors.items()},
            "model_id": self.model_id,
            "notes": list(self.notes),
            "score": round(self.score, 4),
        }


@dataclass(frozen=True)
class FallbackPlan:
    """A bounded ladder of models to try, and what to say when it runs out.

    ``steps`` are ordered best-first and each names a real, present model. When
    ``steps`` is empty, ``reason`` says why -- "nothing here can answer this task"
    is the honest answer, and ``cloud_hint`` names a cloud provider as an *option*
    for the user to consider, never as a silent dependency.
    """

    task: str
    steps: tuple[RankedModel, ...] = field(default=())
    reason: str = ""
    cloud_hint: str = ""

    @property
    def available(self) -> bool:
        return bool(self.steps)

    def describe(self) -> str:
        if not self.steps:
            line = f"No local model can answer the {self.task} task. {self.reason}".strip()
            if self.cloud_hint:
                line = f"{line} {self.cloud_hint}"
            return line
        lines = [f"Fallback order for the {self.task} task:"]
        lines.extend(f"  {index}. {step.describe()}" for index, step in enumerate(self.steps, 1))
        return "\n".join(lines)

    def to_primitive(self) -> dict[str, Any]:
        return {
            "available": self.available,
            "cloud_hint": self.cloud_hint,
            "reason": self.reason,
            "steps": [step.to_primitive() for step in self.steps],
            "task": self.task,
        }


def rank_for_task(
    manager: Any,
    task: str,
    *,
    profile: TaskProfile | None = None,
    preferred_model_id: str = "",
    ready_only: bool = True,
    limit: int = 0,
    resources: Any = None,
) -> tuple[RankedModel, ...]:
    """Rank the models the manager knows against a task, best first.

    Only models that can run **now** are ranked by default (``ready_only``), which
    is what makes a fallback ladder honest: every rung is a model that is present.
    A caller that wants to see *every* model's score -- including one whose file is
    missing -- passes ``ready_only=False``, and the score is unchanged; the model
    is simply not a candidate a fallback could use.

    ``preferred_model_id`` is the user's remembered choice: it is scored like the
    others but given the preference weight, so a preference breaks a tie rather
    than overriding a genuinely better fit.
    """
    resolved = profile if profile is not None else task_profile(task)
    if not resolved.task or not is_known_task(resolved.task):
        return ()

    entries = manager.list_models()
    if ready_only:
        entries = tuple(entry for entry in entries if entry.ready)

    ranked: list[RankedModel] = []
    for entry in entries:
        ranked.append(
            _score_entry(
                entry,
                resolved,
                preferred_model_id=preferred_model_id,
                manager=manager,
                resources=resources,
            )
        )
    # A total order, so the same inputs always give the same sequence: score first,
    # then the display name, then the id -- never the registry's insertion order.
    ranked.sort(key=lambda item: (-item.score, item.display_name.casefold(), item.model_id))
    if limit > 0:
        ranked = ranked[:limit]
    return tuple(ranked)


def _score_entry(
    entry: Any,
    profile: TaskProfile,
    *,
    preferred_model_id: str,
    manager: Any,
    resources: Any,
) -> RankedModel:
    """One candidate's score, computed factor by factor."""
    record = entry.record
    engine = str(getattr(record, "engine", "") or "")
    factors: dict[str, float] = {}
    notes: list[str] = []

    # --- engine: the task's own engine first, then anything else -------------
    if profile.engines:
        engine_score = 1.0 if engine in profile.engines else 0.0
        if not engine:
            notes.append("This model names no engine.")
    else:
        engine_score = 1.0
    factors[FACTOR_ENGINE] = engine_score

    # --- task match: does the record carry the task as a tag/capability? -----
    factors[FACTOR_TASK_MATCH] = _task_match(record, profile)

    # --- capability: what the task needs the model to be able to do ----------
    factors[FACTOR_CAPABILITY] = _capability_score(record, profile, notes)

    # --- fit: can it hold the context the task wants? ------------------------
    factors[FACTOR_FIT] = _fit_score(entry, profile, manager, resources, notes)

    # --- preference: the user's own remembered choice ------------------------
    factors[FACTOR_PREFERENCE] = (
        1.0 if preferred_model_id and entry.model_id == preferred_model_id else 0.0
    )

    # --- health: is the file whole? -----------------------------------------
    integrity = str(getattr(record, "integrity", "") or "")
    if integrity and not getattr(record, "verified", False):
        factors[FACTOR_HEALTH] = 0.0
        notes.append("The file failed its integrity check.")
    else:
        factors[FACTOR_HEALTH] = 1.0

    score = sum(FACTOR_WEIGHTS[name] * value for name, value in factors.items())
    return RankedModel(
        model_id=entry.model_id,
        display_name=entry.display_name,
        score=round(score, 4),
        factors=factors,
        notes=tuple(notes),
    )


def _task_match(record: Any, profile: TaskProfile) -> float:
    """How directly the record names the task.

    A record that lists the task in its own ``capabilities`` or ``tags`` scores
    highest; one that lists the task's engine already scored on the engine factor,
    so this is a separate signal rather than a repeat of it.
    """
    wanted = {profile.task, *profile.capabilities}
    if not wanted:
        return 0.5
    tags = {str(item).casefold() for item in getattr(record, "tags", ()) or ()}
    capabilities = {
        str(item).casefold() for item in getattr(record, "capabilities", ()) or ()
    }
    present = {str(item).casefold() for item in wanted}
    if present & (tags | capabilities):
        return 1.0
    if tags or capabilities:
        # It states *some* abilities, just not this task's -- a partial match.
        return 0.25
    # It states none, so this factor cannot discriminate; neither reward nor punish.
    return 0.5


def _capability_score(
    record: Any, profile: TaskProfile, notes: list[str]
) -> float:
    """Whether the model can do what the task needs, when that can be told."""
    if not profile.capabilities:
        return 1.0
    stated = {
        str(item).casefold() for item in getattr(record, "capabilities", ()) or ()
    }
    if not stated:
        notes.append("The record states no capabilities, so the match cannot be confirmed.")
        return UNKNOWN_CAPABILITY_SCORE
    wanted = {str(item).casefold() for item in profile.capabilities}
    matched = len(wanted & stated)
    return round(matched / len(wanted), 4) if wanted else 1.0


def _fit_score(
    entry: Any,
    profile: TaskProfile,
    manager: Any,
    resources: Any,
    notes: list[str],
) -> float:
    """Whether the model's context can hold what the task wants, and it fits memory.

    A shorter context is not a disqualification -- it is a lower score, because a
    smaller window still answers. A model whose *estimated* memory need is larger
    than what is free is scored down and the reason is noted, never excluded: an
    estimate is not a guarantee, in either direction.
    """
    record = entry.record
    declared = getattr(record, "context_length", None)
    if not declared:
        notes.append("The model declares no context length.")
        context_score = 0.5
    elif profile.context <= 0:
        context_score = 1.0
    else:
        context_score = min(1.0, declared / profile.context)

    memory_score = 1.0
    try:
        resolved = resources if resources is not None else manager.resources()
    except Exception:  # a resource read must never fail a ranking
        resolved = None
    if resolved is not None:
        available = getattr(resolved, "available_memory_bytes", None)
        if available:
            try:
                estimate = manager.memory_estimate(entry.model_id, resources=resolved)
            except Exception:
                estimate = None
            if estimate is not None and estimate.high > available:
                memory_score = 0.25
                notes.append(
                    "The estimated memory need is larger than what the OS reports free."
                )
    return round((context_score + memory_score) / 2, 4)


def plan_fallback(
    manager: Any,
    task: str,
    *,
    preferred_model_id: str = "",
    limit: int = 3,
    resources: Any = None,
) -> FallbackPlan:
    """The bounded ladder of models to try for ``task``, best first.

    The ladder is the ranking, capped at ``limit``, so a caller tries at most that
    many before reporting a failure -- a fallback that walks every model on the
    machine is not a fallback, it is a hunt. When the ladder is empty the plan says
    why, and names a cloud provider only as an *option* the user may choose; the
    program itself never reaches the network here.
    """
    ranked = rank_for_task(
        manager, task,
        preferred_model_id=preferred_model_id,
        ready_only=True,
        resources=resources,
    )
    if ranked:
        return FallbackPlan(task=task, steps=ranked[: max(1, limit)])

    reason = "No model that can run here is registered for it."
    if not is_known_task(task):
        reason = "This build does not declare that task."
    else:
        every = rank_for_task(
            manager, task,
            preferred_model_id=preferred_model_id,
            ready_only=False,
            resources=resources,
        )
        if every:
            reason = (
                "A model for it is known, but its file is not present or not usable "
                "yet; inspect it or download it first."
            )
    return FallbackPlan(
        task=task,
        reason=reason,
        cloud_hint=(
            "A remote (cloud) model could answer instead, but that is the user's own "
            "choice and needs their approval; nothing is sent off this machine by "
            "default."
        ),
    )


def describe_ranking(ranked: tuple[RankedModel, ...], *, task: str = "") -> str:
    """A readable listing of a ranking, for a diagnostics surface."""
    head = f"Model ranking for the {task} task" if task else "Model ranking"
    if not ranked:
        return f"{head}: no model can answer it here."
    lines = [head]
    lines.extend(f"  {index}. {item.describe()}" for index, item in enumerate(ranked, 1))
    return "\n".join(lines)


__all__ = [
    "FACTOR_CAPABILITY",
    "FACTOR_ENGINE",
    "FACTOR_FIT",
    "FACTOR_HEALTH",
    "FACTOR_PREFERENCE",
    "FACTOR_TASK_MATCH",
    "FACTOR_WEIGHTS",
    "TASK_CAPABILITIES",
    "TASK_CONTEXT_PREFERENCE",
    "UNKNOWN_CAPABILITY_SCORE",
    "FallbackPlan",
    "RankedModel",
    "TaskProfile",
    "describe_ranking",
    "is_known_task",
    "plan_fallback",
    "rank_for_task",
    "task_profile",
]
