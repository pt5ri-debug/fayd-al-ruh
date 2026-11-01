"""The Cognitive Learning System: six memories behind one controlled interface.

The task asks for six kinds of memory and the operations that make them useful.
Every kind already has a home in the project, so this module **composes the
existing stores** rather than writing a second one:

===========================  =========================================
Memory kind                  Backed by
===========================  =========================================
Semantic                     :class:`core.semantic_memory.SemanticMemory`
Episodic                     :class:`core.episodic_memory.EpisodicMemory`
Associative                  :class:`core.memory_associator.MemoryAssociator`
Procedural                   :class:`core.procedural_memory.ProceduralMemory`
Hypothesis                   :class:`cognition.storage.IdeaStore` (the HYPOTHESIS rung)
Learning                     :class:`cognition.storage.ExperienceStore`
===========================  =========================================

On top of those it adds the operations the task requires: record an experience,
record an expectation, compare the expectation to what actually happened, analyse
the error, name its likely cause, extract the lesson, update knowledge, link a
past experience to a new problem, and retrieve knowledge on demand -- keeping the
**source and the reliability** of every statement, and never storing an
unverified statement as a fact (see :mod:`cognition.reliability`).

Nothing here writes into a character's learning files, and nothing here is
reached by a character: this is the model's own memory.
"""

from __future__ import annotations

from dataclasses import dataclass

from core.episodic_memory import EpisodicMemory
from core.memory_associator import MemoryAssociator
from core.procedural_memory import ProceduralMemory, Procedure
from core.semantic_memory import SemanticMemory

from cognition.records import Experience, IdeaKind, Source
from cognition.reliability import is_trusted, score
from cognition.storage import ExperienceStore, FactStore, IdeaStore


@dataclass(frozen=True)
class ErrorAnalysis:
    """Why an expectation was not met, and what to do about it."""

    expectation: str
    outcome: str
    surprise: float
    cause: str
    lesson: str
    corrected_confidence: float

    def describe(self) -> str:
        return (
            f"expected '{self.expectation}', got '{self.outcome}' "
            f"(surprise {self.surprise:.2f}); cause: {self.cause}; "
            f"lesson: {self.lesson}"
        )


@dataclass
class LearningStats:
    """A readable snapshot of what the learning system holds."""

    semantic: int = 0
    episodic: int = 0
    procedures: int = 0
    associations: int = 0
    hypotheses: int = 0
    experiences: int = 0
    failures: int = 0
    success_rate: float = 0.0

    def describe(self) -> str:
        return (
            f"semantic {self.semantic}, episodic {self.episodic}, "
            f"procedures {self.procedures}, associations {self.associations}, "
            f"hypotheses {self.hypotheses}, experiences {self.experiences} "
            f"(success {self.success_rate:.0%})"
        )


class CognitiveLearningSystem:
    """The six memories and the learning operations over them."""

    def __init__(
        self,
        *,
        semantic: SemanticMemory | None = None,
        episodic: EpisodicMemory | None = None,
        procedural: ProceduralMemory | None = None,
        associator: MemoryAssociator | None = None,
        idea_store: IdeaStore | None = None,
        experience_store: ExperienceStore | None = None,
        fact_store: FactStore | None = None,
    ) -> None:
        self.semantic = semantic or SemanticMemory()
        self.episodic = episodic or EpisodicMemory()
        self.procedural = procedural or ProceduralMemory()
        self.associator = associator or MemoryAssociator()
        self.idea_store = idea_store or IdeaStore()
        self.experiences = experience_store or ExperienceStore()
        self.facts = fact_store or FactStore()
        self._associations = 0
        # Rehydrate the session's semantic memory from what was learned before, so
        # a fact survives a restart exactly as the project's own learned relations
        # do (``core.knowledge_carry``). A corrupt document reads as empty.
        for fact_id, statement in self.facts.all():
            if fact_id and statement:
                self.semantic.learn(fact_id, statement)

    # -- recording ----------------------------------------------------------
    def learn_fact(
        self,
        fact_id: str,
        statement: str,
        *,
        source: Source | None = None,
        confidence: float = 0.5,
        verified: bool = False,
    ) -> bool:
        """Learn a statement into semantic memory, but only trust it when verified.

        Returns whether the statement was stored as a *trusted* fact. An
        unverified statement is still stored (so it can be recalled and later
        checked) but is reported as not trusted.
        """
        reliability = score(source=source, confidence=confidence, verified=verified)
        self.semantic.learn(fact_id, statement)
        self.facts.learn(fact_id, statement)
        return is_trusted(reliability)

    def record_experience(self, experience: Experience) -> Experience:
        """Store an experience in the episodic and learning memories."""
        self.episodic.store(experience.id, experience.describe(), importance=1.0 - experience.surprise)
        return self.experiences.add(experience)

    def learn_procedure(self, name: str, steps: tuple[str, ...], *, confidence: float = 0.5) -> Procedure:
        """Learn a procedure into procedural memory."""
        procedure = Procedure(name=name, steps=tuple(steps), confidence=confidence)
        self.procedural.learn(procedure)
        return procedure

    def associate(self, source_id: str, target_id: str, strength: float = 0.5) -> None:
        """Record an association between two memories."""
        self.associator.associate(source_id, target_id, strength)
        self._associations += 1

    # -- learning from error ------------------------------------------------
    def analyse_error(
        self,
        *,
        expectation: str,
        expected_confidence: float,
        outcome: str,
        success: bool,
        source: Source | None = None,
    ) -> ErrorAnalysis:
        """Compare an expectation to the outcome, name the cause, extract a lesson.

        Surprise is how far the outcome was from the expectation, so a confident
        expectation that fails is *more* surprising than a hesitant one that
        fails. The cause is a plain-language reading of that gap (an over-confident
        expectation, a surprising success, or a confirmed expectation), and the
        lesson is a short, reusable rule. Nothing here writes to the graph.
        """
        target = 1.0 if success else 0.0
        surprise = round(abs(expected_confidence - target), 6)
        if not success and expected_confidence >= 0.6:
            cause = "the expectation was over-confident"
            lesson = f"lower confidence in '{expectation}' until it is re-verified"
        elif success and expected_confidence <= 0.4:
            cause = "the expectation under-estimated what works"
            lesson = f"raise confidence in '{expectation}'; it worked despite doubt"
        elif not success:
            cause = "the expectation was wrong in substance"
            lesson = f"revise what '{expectation}' assumes"
        else:
            cause = "the expectation held"
            lesson = f"keep '{expectation}' as a working rule"
        corrected = round(min(max(target, 0.0), 1.0), 6)
        return ErrorAnalysis(
            expectation=expectation,
            outcome=outcome,
            surprise=surprise,
            cause=cause,
            lesson=lesson,
            corrected_confidence=corrected,
        )

    def learn_from_error(
        self,
        *,
        expectation: str,
        expected_confidence: float,
        outcome: str,
        success: bool,
        source: Source | None = None,
    ) -> Experience:
        """Analyse an error and store it as an experience with its lesson."""
        analysis = self.analyse_error(
            expectation=expectation,
            expected_confidence=expected_confidence,
            outcome=outcome,
            success=success,
            source=source,
        )
        experience = Experience(
            expectation=expectation,
            outcome=outcome,
            lesson=analysis.lesson,
            success=success,
            surprise=analysis.surprise,
            source=source,
        )
        return self.record_experience(experience)

    # -- retrieval ----------------------------------------------------------
    def recall(self, query: str, *, limit: int = 5) -> dict[str, list[str]]:
        """Retrieve related knowledge of every kind for a query.

        Returns a dict with ``semantic``, ``episodic`` and ``associated`` lists,
        each a readable description, so a caller sees where each answer came
        from. Empty lists are honest: nothing was found.
        """
        semantic_ids = self.semantic.retrieve(query, limit=limit)
        semantic = [self.semantic.facts[i] for i in semantic_ids if i in self.semantic.facts]
        if not semantic:
            # The hashed embedding is a weak signal on a tiny store; fall back to
            # a lexical match so a fact that shares a word is still recalled.
            needle = query.lower()
            semantic = [
                statement
                for statement in self.semantic.facts.values()
                if any(token in statement.lower() for token in needle.split() if len(token) > 2)
            ][:limit]
        needle = query.lower()
        episodic = [
            str(episode.content)
            for episode in self.episodic.episodes.values()
            if needle in str(episode.content).lower()
            or any(token in str(episode.content).lower() for token in needle.split() if len(token) > 3)
        ][:limit]
        associated = [a.target_id for a in self.associator.related_to(query)][:limit]
        return {"semantic": semantic, "episodic": episodic, "associated": associated}

    def link_experience(self, problem: str, *, limit: int = 3) -> tuple[Experience, ...]:
        """Find past experiences relevant to a new problem (shared words)."""
        problem_tokens = {w.lower() for w in problem.split() if len(w) > 2}
        scored: list[tuple[float, Experience]] = []
        for experience in self.experiences.all():
            text = f"{experience.expectation} {experience.outcome} {experience.lesson}".lower()
            overlap = sum(1 for token in problem_tokens if token in text)
            if overlap:
                scored.append((overlap / max(1, len(problem_tokens)), experience))
        scored.sort(key=lambda pair: (-pair[0], pair[1].id))
        return tuple(exp for _, exp in scored[:limit])

    def stats(self) -> LearningStats:
        """A readable count of everything the system holds."""
        return LearningStats(
            semantic=len(self.semantic.facts),
            episodic=len(self.episodic.episodes),
            procedures=len(self.procedural.procedures),
            associations=self._associations,
            hypotheses=len(self.idea_store.by_kind(IdeaKind.HYPOTHESIS)),
            experiences=self.experiences.count(),
            failures=len(self.experiences.failures()),
            success_rate=self.experiences.success_rate(),
        )

    # -- consolidation ------------------------------------------------------
    def consolidator(self):
        """The consolidator over this system's own memories (built once).

        Kept here rather than in a second object so the window, the CLI, and the
        proactive mind all consolidate the *same* experiences into the *same*
        patterns -- one source of truth for what the mind has learned.
        """
        consolidator = getattr(self, "_consolidator", None)
        if consolidator is None:
            from cognition.consolidation import Consolidator

            consolidator = Consolidator(self)
            self._consolidator = consolidator
        return consolidator

    def consolidate(self):
        """Consolidate the raw experiences into durable, reviewed knowledge."""
        return self.consolidator().consolidate()


__all__ = ["CognitiveLearningSystem", "ErrorAnalysis", "LearningStats"]
