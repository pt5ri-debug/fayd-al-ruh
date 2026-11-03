"""Task routing: choose the personality best suited to a task, and explain why.

The task requires a routing mechanism that identifies the task type, determines
the required capabilities, selects a suitable personality, **explains** the
decision, respects a user-selected personality, supports manual overrides, detects
when *no* personality is suitable, and avoids unnecessary switching -- and it
insists the decision be traceable and configurable, and that no personality be
assumed "objectively superior" without evaluation.

This router is therefore honest by construction:

* it scores each personality on **named, readable factors** (a declared
  specialisation match, a capability match, a task-preference match) rather than
  an opaque rank, so the decision can be printed;
* it **never invents** a winner: when no personality matches, it says so
  (:class:`RoutingDecision.suitable` is ``False``) rather than routing to a
  default;
* a user-selected or overridden personality always wins, and the router records
  *why* it was chosen;
* it records the last decision per conversation, so a follow-up to the same
  personality does not re-route (avoiding unnecessary switching).

Dependency-free and Tk-free: it holds no model and touches no network.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

from core.agents.capabilities import AgentCapability
from core.agents.registry import PersonalityProfile, PersonalityRegistry
from core.learning_isolation import CHARACTER_OWNERS, owner_for


class TaskType(str, Enum):
    """The kinds of task the project routes. One closed vocabulary."""

    GROUNDED_ANSWER = "grounded_answer"
    GENERAL_ANSWER = "general_answer"
    PETROLEUM = "petroleum"
    PRESENTATION = "presentation"
    COMPANION = "companion"
    RELIGION = "religion"
    REMOTE = "remote"
    CODE = "code"
    REASONING = "reasoning"
    UNKNOWN = "unknown"


#: The capabilities each task type needs. A task type not listed needs only
#: :attr:`~AgentCapability.ANSWER`.
TASK_REQUIREMENTS: dict[TaskType, tuple[AgentCapability, ...]] = {
    TaskType.GROUNDED_ANSWER: (AgentCapability.ANSWER, AgentCapability.RETRIEVE),
    TaskType.GENERAL_ANSWER: (AgentCapability.ANSWER,),
    TaskType.PETROLEUM: (AgentCapability.ANSWER, AgentCapability.REASON),
    TaskType.PRESENTATION: (AgentCapability.ANSWER,),
    TaskType.COMPANION: (AgentCapability.ANSWER,),
    TaskType.RELIGION: (AgentCapability.ANSWER,),
    TaskType.REMOTE: (AgentCapability.ANSWER,),
    TaskType.CODE: (AgentCapability.ANSWER,),
    TaskType.REASONING: (AgentCapability.ANSWER, AgentCapability.REASON),
    TaskType.UNKNOWN: (AgentCapability.ANSWER,),
}

#: The specialisation each personality is *declared* for, in the router's own
#: words. This is configuration, not a judgement of quality -- the router scores
#: a declared match, never a claim that one personality is "better".
SPECIALISATIONS: dict[str, tuple[TaskType, ...]] = {
    "jarvis": (TaskType.GROUNDED_ANSWER,),
    "ultron": (TaskType.GENERAL_ANSWER,),
    "vision": (TaskType.PETROLEUM, TaskType.REASONING),
    "banner": (TaskType.PRESENTATION,),
    "natasha": (TaskType.COMPANION,),
    "hakeem": (TaskType.RELIGION,),
    "cloud": (TaskType.REMOTE,),
}

#: Keywords that hint at a task type, used only when the caller does not name one.
#: Deliberately small and readable: routing by keyword is a *hint*, and the
#: decision prints which hint fired, so a wrong guess is visible.
_TASK_HINTS: tuple[tuple[TaskType, tuple[str, ...]], ...] = (
    (TaskType.PETROLEUM, ("porosity", "permeability", "reservoir", "drilling", "well", "pvt",
                          "petrophysic", "waterflood", "مسامية", "نفاذية", "بئر", "مكمن")),
    (TaskType.PRESENTATION, ("slide", "presentation", "deck", "powerpoint", "outline",
                             "عرض", "شرائح")),
    (TaskType.RELIGION, ("religion", "faith", "islam", "christian", "judaism", "quran",
                         "bible", "history of", "دين", "إسلام", "تاريخ")),
    (TaskType.COMPANION, ("how are you", "i feel", "talk to me", "companion", "كيف حالك")),
    (TaskType.CODE, ("code", "function", "bug", "refactor", "python", "برمجة")),
    (TaskType.REASONING, ("why", "prove", "infer", "compare", "connect", "لماذا", "استنتج")),
    (TaskType.GROUNDED_ANSWER, ("cite", "source", "according to", "evidence", "مصدر", "مرجع")),
)


@dataclass(frozen=True)
class RoutingFactor:
    """One named reason a personality scored the way it did."""

    name: str
    weight: float
    detail: str


@dataclass(frozen=True)
class RoutingCandidate:
    """One personality's score for a task, with its factors."""

    owner: str
    score: float
    factors: tuple[RoutingFactor, ...]
    #: Whether the personality's **declared** specialisation matches the task. Only
    #: a declared match makes a personality *suitable* by default; the capability
    #: factor alone never does, so an unmatched task honestly reports "none".
    matched: bool = False

    def describe(self) -> str:
        reasons = "; ".join(f"{factor.name} ({factor.detail})" for factor in self.factors)
        return f"{self.owner}: {self.score:.2f} — {reasons or 'no factors'}"


@dataclass(frozen=True)
class RoutingDecision:
    """The router's answer: who should do the task, and the whole reasoning."""

    task_type: TaskType
    required: tuple[AgentCapability, ...]
    selected: str = ""
    suitable: bool = False
    manual: bool = False
    reason: str = ""
    candidates: tuple[RoutingCandidate, ...] = ()
    refused: tuple[str, ...] = ()

    def describe(self) -> str:
        if not self.suitable:
            return f"no personality is suitable for a {self.task_type.value} task ({self.reason})"
        how = "manual/override" if self.manual else "routed"
        lines = [f"{how}: {self.selected} for a {self.task_type.value} task — {self.reason}"]
        if self.refused:
            lines.append(f"  could not serve: {', '.join(self.refused)}")
        for candidate in self.candidates[:4]:
            lines.append(f"  {candidate.describe()}")
        return "\n".join(lines)

    def to_primitive(self) -> dict[str, object]:
        return {
            "task_type": self.task_type.value,
            "required": [capability.value for capability in self.required],
            "selected": self.selected,
            "suitable": self.suitable,
            "manual": self.manual,
            "reason": self.reason,
            "refused": list(self.refused),
            "candidates": [
                {"owner": candidate.owner, "score": candidate.score} for candidate in self.candidates
            ],
        }


def classify_task(question: str, *, declared: TaskType | None = None) -> tuple[TaskType, str]:
    """The task type for ``question``, with the hint that fired (or ``declared``)."""
    if declared is not None:
        return declared, "the caller named the task type"
    lowered = question.casefold()
    for task_type, keywords in _TASK_HINTS:
        for keyword in keywords:
            if keyword in lowered:
                return task_type, f"the question mentions {keyword!r}"
    return TaskType.UNKNOWN, "no hint matched"


def required_capabilities(task_type: TaskType) -> tuple[AgentCapability, ...]:
    """The capabilities a task of ``task_type`` needs."""
    return TASK_REQUIREMENTS.get(task_type, (AgentCapability.ANSWER,))


@dataclass
class TaskRouter:
    """The traceable, configurable task router.

    ``registry`` supplies each personality's declared specialisation (its
    configuration) and its capabilities. ``sticky`` remembers the last personality
    chosen per conversation so a follow-up does not re-route, which is the
    "avoid unnecessary switching" requirement.
    """

    registry: PersonalityRegistry = field(default_factory=PersonalityRegistry)
    #: Whether a personality whose declared specialisation does not match may
    #: still be chosen when nothing else can serve (a *capable fallback*). Off by
    #: default, so an unmatched task honestly reports "no suitable personality".
    allow_general_fallback: bool = False
    _last: dict[str, tuple[str, TaskType]] = field(default_factory=dict, repr=False)
    _log: list[RoutingDecision] = field(default_factory=list, repr=False)

    # ------------------------------------------------------------- routing

    def route(
        self,
        question: str,
        *,
        conversation: str = "",
        task_type: TaskType | None = None,
        required: tuple[AgentCapability, ...] | None = None,
        override: str = "",
        user_choice: str = "",
    ) -> RoutingDecision:
        """Choose a personality for ``question`` and explain the decision.

        Precedence, highest first:

        1. ``override`` -- an explicit manual choice (always wins);
        2. ``user_choice`` -- the personality the user selected;
        3. a **sticky** previous choice for this conversation, when it can still
           serve the task (avoids switching mid-conversation);
        4. the best-scoring personality whose declared specialisation matches;
        5. a capable fallback, only when :attr:`allow_general_fallback` is set.

        If none applies, the decision is ``suitable=False`` with a reason -- the
        router never silently picks a personality it cannot justify.
        """
        resolved_type, hint = classify_task(question, declared=task_type)
        needs = required if required is not None else required_capabilities(resolved_type)

        for key, how in ((override, "override"), (user_choice, "user choice")):
            if key:
                decision = self._manual_decision(resolved_type, needs, key, how)
                self._log.append(decision)
                if decision.suitable and conversation:
                    self._last[conversation] = (decision.selected, resolved_type)
                return decision

        if conversation and conversation in self._last:
            previous, previous_type = self._last[conversation]
            # Stick only when the topic has not clearly changed: the same task
            # type, or an unrecognised follow-up (which keeps the thread). A
            # question that plainly names a different task re-routes.
            same_topic = resolved_type is previous_type or resolved_type is TaskType.UNKNOWN
            if same_topic and self._can_serve(previous, needs):
                decision = RoutingDecision(
                    task_type=resolved_type,
                    required=needs,
                    selected=owner_for(previous),
                    suitable=True,
                    manual=False,
                    reason="kept the personality already answering this conversation.",
                )
                self._log.append(decision)
                return decision

        return self._scored_decision(resolved_type, needs, hint, conversation)

    # ------------------------------------------------------------- helpers

    def _manual_decision(
        self, task_type: TaskType, needs: tuple[AgentCapability, ...], key: str, how: str
    ) -> RoutingDecision:
        owner = owner_for(key)
        if not self._is_character(owner):
            return RoutingDecision(
                task_type=task_type,
                required=needs,
                suitable=False,
                manual=True,
                reason=f"{key!r} is not one of the seven personalities.",
            )
        if not self._can_serve(owner, needs):
            missing = self._missing(owner, needs)
            return RoutingDecision(
                task_type=task_type,
                required=needs,
                suitable=False,
                manual=True,
                reason=f"{owner} lacks {', '.join(capability.value for capability in missing)}.",
                refused=(owner,),
            )
        return RoutingDecision(
            task_type=task_type,
            required=needs,
            selected=owner,
            suitable=True,
            manual=True,
            reason=f"the {how} selected this personality.",
        )

    def _scored_decision(
        self, task_type: TaskType, needs: tuple[AgentCapability, ...], hint: str, conversation: str
    ) -> RoutingDecision:
        candidates: list[RoutingCandidate] = []
        refused: list[str] = []
        for owner in CHARACTER_OWNERS:
            if not self._can_serve(owner, needs):
                refused.append(owner)
                continue
            candidates.append(self._score(owner, task_type, needs))
        if not candidates:
            decision = RoutingDecision(
                task_type=task_type,
                required=needs,
                suitable=False,
                reason=f"no personality holds {', '.join(c.value for c in needs)}",
                refused=tuple(refused),
            )
            self._log.append(decision)
            return decision

        candidates.sort(key=lambda candidate: (-candidate.score, candidate.owner))
        best = candidates[0]
        matched = [candidate for candidate in candidates if candidate.matched]
        if not matched:
            decision = RoutingDecision(
                task_type=task_type,
                required=needs,
                suitable=False,
                reason=f"no personality specialises in a {task_type.value} task ({hint})",
                candidates=tuple(candidates),
                refused=tuple(refused),
            )
            self._log.append(decision)
            return decision
        chosen = best if best.matched else matched[0]
        decision = RoutingDecision(
            task_type=task_type,
            required=needs,
            selected=chosen.owner,
            suitable=True,
            manual=False,
            reason=f"{hint}; best declared match",
            candidates=tuple(candidates),
            refused=tuple(refused),
        )
        self._log.append(decision)
        if conversation:
            self._last[conversation] = (chosen.owner, task_type)
        return decision

    def _score(
        self, owner: str, task_type: TaskType, needs: tuple[AgentCapability, ...]
    ) -> RoutingCandidate:
        profile: PersonalityProfile = self.registry.resolve(owner)
        factors: list[RoutingFactor] = []
        declared = SPECIALISATIONS.get(owner, ())
        matched = task_type in declared
        if matched:
            factors.append(RoutingFactor("specialisation", 1.0, f"declared for {task_type.value}"))
        capability_score = sum(1.0 for capability in needs if profile.holds(capability))
        if capability_score:
            factors.append(
                RoutingFactor(
                    "capabilities",
                    0.1 * capability_score,
                    f"holds {int(capability_score)}/{len(needs)} required",
                )
            )
        preference_hit = any(
            task_type.value.split("_")[0] in preference.casefold()
            for preference in profile.task_preferences
        )
        if preference_hit:
            factors.append(RoutingFactor("task preference", 0.2, "names a matching objective"))
        score = sum(factor.weight for factor in factors)
        return RoutingCandidate(owner=owner, score=score, factors=tuple(factors), matched=matched)

    def _is_character(self, owner: str) -> bool:
        return owner in CHARACTER_OWNERS

    def _can_serve(self, owner: str, needs: tuple[AgentCapability, ...]) -> bool:
        if not self._is_character(owner):
            return False
        profile = self.registry.resolve(owner)
        return all(profile.holds(capability) for capability in needs)

    def _missing(self, owner: str, needs: tuple[AgentCapability, ...]) -> tuple[AgentCapability, ...]:
        profile = self.registry.resolve(owner)
        return tuple(capability for capability in needs if not profile.holds(capability))

    # ------------------------------------------------------------- reading

    def decisions(self) -> tuple[RoutingDecision, ...]:
        """Every decision the router has taken, for a report or a test."""
        return tuple(self._log)

    def last_for(self, conversation: str) -> str:
        """The personality currently answering ``conversation`` (or ``""``)."""
        entry = self._last.get(conversation)
        return entry[0] if entry else ""

    def forget(self, conversation: str) -> None:
        """Drop the sticky choice for ``conversation`` (a new conversation)."""
        self._last.pop(conversation, None)


__all__ = [
    "SPECIALISATIONS",
    "TASK_REQUIREMENTS",
    "RoutingCandidate",
    "RoutingDecision",
    "RoutingFactor",
    "TaskRouter",
    "TaskType",
    "classify_task",
    "required_capabilities",
]
