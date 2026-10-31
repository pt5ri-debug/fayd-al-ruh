"""The Proactive Mind: the eight engines, composed into one thinking cycle.

:class:`ProactiveMind` is the only object that runs the engines together. One
:meth:`cycle` is a complete, bounded unit of autonomous thought:

1. **Curiosity** finds the information gaps (from beliefs, the graph, conflicts).
2. **Motivation** ranks the topics worth pursuing.
3. **Self-Questioning** turns the top gaps into answerable questions.
4. **Association** finds how the concepts involved are related.
5. **Idea** generation recombines those associations into new statements.
6. **Imagination** builds counterfactuals and speculative scenarios.
7. **Self-Criticism** reviews each idea and lowers its confidence where it is
   weak, recording the reasons.
8. **Thought Scheduler** decides *when* all of this may run at all.

The cycle returns a :class:`~cognition.records.ThoughtCycle` and persists the
ideas, the questions, and the cycle itself. **No generated idea is ever stored as
a fact**: the store keeps every idea at its own rung, and only an explicit
promotion (or the verification path) can raise one.

The mind never writes into a character's learning files, and never writes into
the ontology or the belief graph. It only *reads* the graph and *writes* its own
ideas, questions, and cycles -- which is the isolation the task requires.
"""

from __future__ import annotations

import time as _time

from core.belief import Belief
from core.causal_graph import CausalGraph
from core.goal import Goal
from core.ontology import Ontology

from cognition.engines.association import AssociationEngine
from cognition.engines.criticism import SelfCriticismEngine
from cognition.engines.curiosity import CuriosityEngine
from cognition.engines.ideas import IdeaEngine, NoveltyTracker
from cognition.engines.imagination import ImaginationEngine, Scenario
from cognition.engines.motivation import Drive, MotivationEngine
from cognition.engines.questions import SelfQuestioningEngine
from cognition.records import Idea, IdeaKind, ThoughtCycle
from cognition.scheduler import Decision, SchedulerSettings, SchedulerState, ThoughtScheduler
from cognition.storage import CycleStore, IdeaStore, meta_get, meta_set


class ProactiveMind:
    """Run one bounded, self-directed thinking cycle over the project's knowledge."""

    def __init__(
        self,
        ontology: Ontology,
        *,
        causal_graph: CausalGraph | None = None,
        idea_store: IdeaStore | None = None,
        cycle_store: CycleStore | None = None,
        scheduler: ThoughtScheduler | None = None,
        novelty: NoveltyTracker | None = None,
    ) -> None:
        self.ontology = ontology
        self.novelty = novelty or NoveltyTracker()
        self.curiosity = CuriosityEngine(ontology)
        self.association = AssociationEngine(ontology)
        self.ideas = IdeaEngine(ontology, associations=self.association, novelty=self.novelty)
        self.imagination = ImaginationEngine(
            ontology, causal_graph=causal_graph, associations=self.association
        )
        self.criticism = SelfCriticismEngine(ontology)
        self.motivation = MotivationEngine(ontology, novelty=self.novelty)
        self.questions = SelfQuestioningEngine(ontology)
        self.scheduler = scheduler or ThoughtScheduler()
        self.idea_store = idea_store or IdeaStore()
        self.cycle_store = cycle_store or CycleStore()

    # -- scheduling ---------------------------------------------------------
    def due(self, *, cpu_load: float = 0.0, free_memory_fraction: float = 1.0) -> Decision:
        """Whether a cycle may run now (delegates to the scheduler)."""
        return self.scheduler.due(cpu_load=cpu_load, free_memory_fraction=free_memory_fraction)

    def settings(self) -> SchedulerSettings:
        return self.scheduler.settings

    def pause(self) -> None:
        self.scheduler.pause()
        self._persist_scheduler()

    def resume(self) -> None:
        self.scheduler.resume()
        self._persist_scheduler()

    # -- one cycle ----------------------------------------------------------
    def cycle(
        self,
        *,
        beliefs: tuple[Belief, ...] = (),
        goals: tuple[Goal, ...] = (),
        conflicts: tuple[object, ...] = (),
        max_ideas: int | None = None,
        max_questions: int | None = None,
    ) -> ThoughtCycle:
        """Run one complete thinking cycle and persist its results.

        Bounded by the scheduler's ``ideas_per_cycle`` unless ``max_ideas`` is
        given. Every engine runs in a fixed order, so a cycle is reproducible for
        the same inputs. The cycle is recorded even when it finds little, so the
        UI can always show that the mind ran.
        """
        started = _time.monotonic()
        idea_limit = max_ideas if max_ideas is not None else self.scheduler.settings.ideas_per_cycle
        question_limit = max_questions if max_questions is not None else max(1, idea_limit)

        gaps = tuple(self.curiosity.discover(beliefs, conflicts=conflicts))
        drives = self.motivation.rank(gaps=gaps, goals=goals, limit=idea_limit * 2)
        questions = self.questions.questions(gaps, limit=question_limit)

        labels = self._concept_labels(gaps)
        raw_ideas = list(self.ideas.generate(labels=labels, gaps=gaps, limit=idea_limit))

        scenarios = self.imagination.generate(labels=labels, limit=idea_limit)
        for scenario in scenarios:
            raw_ideas.append(self._scenario_idea(scenario))

        reviewed: list[Idea] = []
        for idea in raw_ideas:
            reviewed_idea = self.criticism.review(idea)
            self.idea_store.add(reviewed_idea)
            self.novelty.remember(reviewed_idea.text)
            reviewed.append(reviewed_idea)

        duration_ms = round((_time.monotonic() - started) * 1000, 3)
        cycle = ThoughtCycle(
            questions=tuple(q.id for q in questions),
            ideas=tuple(i.id for i in reviewed),
            hypotheses=tuple(i.id for i in reviewed if i.kind is IdeaKind.HYPOTHESIS),
            gaps=tuple(d.topic for d in drives),
            notes=tuple(d.describe() for d in drives[:3]),
            duration_ms=duration_ms,
        )
        self.cycle_store.add(cycle)
        meta_set("last_cycle", cycle.to_primitive())
        self.scheduler.record_cycle()
        self._persist_scheduler()
        return cycle

    # -- review helpers -----------------------------------------------------
    def pending_ideas(self, *, limit: int = 20) -> tuple[Idea, ...]:
        """Ideas that are not yet facts, newest first -- what awaits review."""
        pending = [i for i in self.idea_store.all() if i.kind is not IdeaKind.VERIFIED]
        return tuple(reversed(pending))[:limit]

    def top_drives(
        self, *, beliefs: tuple[Belief, ...] = (), goals: tuple[Goal, ...] = (), limit: int = 10
    ) -> tuple[Drive, ...]:
        """The topics the mind would pursue next (for the UI)."""
        gaps = tuple(self.curiosity.discover(beliefs))
        return self.motivation.rank(gaps=gaps, goals=goals, limit=limit)

    def counts(self) -> dict[str, int]:
        return self.idea_store.counts()

    def last_cycle(self) -> ThoughtCycle | None:
        return self.cycle_store.last()

    def status(self) -> dict[str, object]:
        """A small, honest snapshot for the UI and the CLI."""
        last = self.cycle_store.last()
        return {
            "enabled": self.scheduler.settings.enabled,
            "paused": self.scheduler.state.paused,
            "cycles": self.cycle_store.count(),
            "ideas": self.counts(),
            "last_cycle": None if last is None else last.to_primitive(),
            "interval_seconds": self.scheduler.settings.interval_seconds,
            "ideas_per_cycle": self.scheduler.settings.ideas_per_cycle,
        }

    # -- internals ----------------------------------------------------------
    def _concept_labels(self, gaps: tuple[object, ...]) -> tuple[str, ...]:
        """The graph concepts the current gaps name (for recombination)."""
        labels = {c.label for c in self.ontology.concepts.values()}
        found: list[str] = []
        for gap in gaps:
            topic = getattr(gap, "topic", "")
            if topic in labels:
                found.append(topic)
            else:
                for token in str(topic).replace(" vs ", " ").split():
                    if token in labels:
                        found.append(token)
        return tuple(dict.fromkeys(found))

    def _scenario_idea(self, scenario: Scenario) -> Idea:
        return Idea(
            text=scenario.description,
            kind=IdeaKind.IDEA,
            engine=f"imagination.{scenario.kind}",
            sources=(scenario.evidence,) if scenario.evidence else (),
            rationale=f"an imagined {scenario.kind}, not a fact",
            confidence=scenario.confidence,
            novelty=self.novelty.novelty(scenario.description),
        )

    def _persist_scheduler(self) -> None:
        meta_set("scheduler", self.scheduler.state.to_primitive())


def load_scheduler_state() -> SchedulerState:
    """Load the persisted scheduler state, or a fresh one."""
    data = meta_get("scheduler")
    if isinstance(data, dict):
        return SchedulerState.from_primitive(data)
    return SchedulerState()


__all__ = ["ProactiveMind", "load_scheduler_state"]
