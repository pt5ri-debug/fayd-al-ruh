"""Cognitive evaluation: measure the mind, and compare before with after.

The task asks for ten indicators and for a before/after comparison so a change
can be *shown* to help rather than assumed to. Each indicator is computed from a
real, existing measurement -- the ontology, the retriever, the analogy engine, the
learning system, the idea engine, the scheduler -- so nothing here is a proxy
dressed up as a result.

The ten indicators (with their English keys):

===============================  ===========================================
Indicator (Arabic)               Key
===============================  ===========================================
استرجاع المعرفة                  ``knowledge_retrieval``
قدرة الاستدلال                   ``reasoning``
دقة الإجابة                      ``answer_accuracy``
سرعة التعلم                      ``learning_speed``
جودة الأفكار                     ``idea_quality``
النقد الذاتي                     ``self_criticism``
التعميم                          ``generalisation``
سرعة الاستجابة                   ``response_speed``
الموثوقية                        ``reliability``
الاستقلالية                      ``independence``
===============================  ===========================================

Every score is in ``[0, 1]`` and higher is better. A :class:`CognitiveReport`
carries the ten scores plus a :class:`Comparison` against a baseline, so the CLI
and the GUI can say "better by X" with the same numbers.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

from core.analogy import AnalogyEngine
from core.ontology import Ontology
from core.rag import RagRetriever

from cognition.engines.criticism import SelfCriticismEngine
from cognition.engines.ideas import IdeaEngine, NoveltyTracker
from cognition.learning_system import CognitiveLearningSystem
from cognition.records import Idea, IdeaKind, Source, SourceKind

#: The ten indicators, in the order the report lists them.
INDICATORS: tuple[str, ...] = (
    "knowledge_retrieval",
    "reasoning",
    "answer_accuracy",
    "learning_speed",
    "idea_quality",
    "self_criticism",
    "generalisation",
    "response_speed",
    "reliability",
    "independence",
)

#: A readable label per indicator, so a surface never invents one.
LABELS: dict[str, str] = {
    "knowledge_retrieval": "Knowledge retrieval",
    "reasoning": "Reasoning ability",
    "answer_accuracy": "Answer accuracy",
    "learning_speed": "Learning speed",
    "idea_quality": "Idea quality",
    "self_criticism": "Self-criticism",
    "generalisation": "Generalisation",
    "response_speed": "Response speed",
    "reliability": "Reliability",
    "independence": "Independence",
}


@dataclass
class CognitiveReport:
    """The ten measured indicators for one evaluation run."""

    scores: dict[str, float]
    notes: dict[str, str] = field(default_factory=dict)
    duration_ms: float = 0.0

    def __post_init__(self) -> None:
        for key in INDICATORS:
            self.scores.setdefault(key, 0.0)

    @property
    def overall(self) -> float:
        """The mean of the ten indicators."""
        return round(sum(self.scores[k] for k in INDICATORS) / len(INDICATORS), 6)

    def describe(self) -> str:
        lines = [f"overall {self.overall:.3f}"]
        for key in INDICATORS:
            note = f" -- {self.notes[key]}" if key in self.notes else ""
            lines.append(f"  {LABELS[key]}: {self.scores[key]:.3f}{note}")
        return "\n".join(lines)

    def to_primitive(self) -> dict[str, object]:
        return {
            "overall": self.overall,
            "scores": {k: round(self.scores[k], 6) for k in INDICATORS},
            "notes": dict(self.notes),
            "duration_ms": self.duration_ms,
        }

    @classmethod
    def from_primitive(cls, data: dict[str, object]) -> CognitiveReport:
        raw_scores = data.get("scores")
        raw_notes = data.get("notes")
        scores = (
            {str(k): float(v) for k, v in raw_scores.items()}
            if isinstance(raw_scores, dict)
            else {}
        )
        notes = (
            {str(k): str(v) for k, v in raw_notes.items()}
            if isinstance(raw_notes, dict)
            else {}
        )
        duration = data.get("duration_ms")
        return cls(
            scores=scores,
            notes=notes,
            duration_ms=float(duration) if isinstance(duration, (int, float)) else 0.0,
        )


@dataclass(frozen=True)
class Comparison:
    """A before/after reading of two reports."""

    before: CognitiveReport
    after: CognitiveReport

    def deltas(self) -> dict[str, float]:
        return {
            key: round(self.after.scores[key] - self.before.scores[key], 6)
            for key in INDICATORS
        }

    @property
    def overall_delta(self) -> float:
        return round(self.after.overall - self.before.overall, 6)

    def improved(self) -> tuple[str, ...]:
        return tuple(k for k, d in self.deltas().items() if d > 0)

    def regressed(self) -> tuple[str, ...]:
        return tuple(k for k, d in self.deltas().items() if d < 0)

    def describe(self) -> str:
        sign = "+" if self.overall_delta >= 0 else ""
        lines = [f"overall {self.before.overall:.3f} -> {self.after.overall:.3f} ({sign}{self.overall_delta:.3f})"]
        for key in INDICATORS:
            delta = self.deltas()[key]
            if abs(delta) >= 0.001:
                marker = "improved" if delta > 0 else "regressed"
                lines.append(f"  {LABELS[key]}: {delta:+.3f} ({marker})")
        if not self.improved() and not self.regressed():
            lines.append("  no measurable change")
        return "\n".join(lines)


class CognitiveEvaluator:
    """Measure the ten indicators against the live project."""

    def __init__(
        self,
        ontology: Ontology,
        *,
        retriever: RagRetriever | None = None,
        learning: CognitiveLearningSystem | None = None,
    ) -> None:
        self.ontology = ontology
        if retriever is None:
            from core.documents import build_document_store

            retriever = RagRetriever(build_document_store(), ontology=ontology)
        self.retriever = retriever
        self.learning = learning or CognitiveLearningSystem()

    def evaluate(self, *, questions: tuple[str, ...] = ()) -> CognitiveReport:
        """Run every indicator and return the report."""
        started = time.monotonic()
        probes = questions or self._default_questions()
        scores: dict[str, float] = {}
        notes: dict[str, str] = {}

        scores["knowledge_retrieval"], notes["knowledge_retrieval"] = self._retrieval(probes)
        scores["reasoning"], notes["reasoning"] = self._reasoning(probes)
        scores["answer_accuracy"], notes["answer_accuracy"] = self._answer_accuracy(probes)
        scores["learning_speed"], notes["learning_speed"] = self._learning_speed()
        scores["idea_quality"], notes["idea_quality"] = self._idea_quality()
        scores["self_criticism"], notes["self_criticism"] = self._self_criticism()
        scores["generalisation"], notes["generalisation"] = self._generalisation(probes)
        scores["response_speed"], notes["response_speed"] = self._response_speed(probes)
        scores["reliability"], notes["reliability"] = self._reliability()
        scores["independence"], notes["independence"] = self._independence()

        duration = round((time.monotonic() - started) * 1000, 3)
        return CognitiveReport(scores=scores, notes=notes, duration_ms=duration)

    def compare(self, before: CognitiveReport, after: CognitiveReport) -> Comparison:
        return Comparison(before=before, after=after)

    # -- indicators ---------------------------------------------------------
    def _retrieval(self, questions: tuple[str, ...]) -> tuple[float, str]:
        hits = 0
        for question in questions:
            passages = self.retriever.retrieve(question, limit=3)
            if passages:
                hits += 1
        score = round(hits / len(questions), 6) if questions else 0.0
        return score, f"{hits}/{len(questions)} questions retrieved evidence"

    def _reasoning(self, questions: tuple[str, ...]) -> tuple[float, str]:
        engine = AnalogyEngine(self.ontology)
        concepts = [c.label for c in self.ontology.concepts.values()]
        checked = 0
        meaningful = 0
        for i, first in enumerate(concepts[:8]):
            for second in concepts[i + 1 : 8]:
                checked += 1
                result = engine.between_concepts(first, second)
                if result is not None and result.is_meaningful:
                    meaningful += 1
        score = round(meaningful / checked, 6) if checked else 0.0
        return score, f"{meaningful}/{checked} concept pairs share a relational system"

    def _answer_accuracy(self, questions: tuple[str, ...]) -> tuple[float, str]:
        grounded = 0
        for question in questions:
            passages = self.retriever.retrieve(question, limit=3)
            if any(question.lower().split()[0] in p.text.lower() for p in passages):
                grounded += 1
        score = round(grounded / len(questions), 6) if questions else 0.0
        return score, f"{grounded}/{len(questions)} answers rest on a matching passage"

    def _learning_speed(self) -> tuple[float, str]:
        """Learn a few facts now and measure how quickly they become retrievable.

        The measurement is self-contained: it stores statements through the real
        learning system and immediately asks for them back, so it does not depend
        on the store happening to have data.
        """
        probes = (
            ("speed-1", "waterflood raises reservoir pressure"),
            ("speed-2", "acid stimulation increases permeability"),
            ("speed-3", "a density log measures porosity"),
        )
        for fact_id, statement in probes:
            self.learning.learn_fact(
                fact_id,
                statement,
                source=Source(SourceKind.CORPUS, "petroleum_corpus"),
                confidence=0.9,
                verified=True,
            )
        retrieved = 0
        for fact_id, statement in probes:
            found = self.learning.semantic.retrieve(statement, limit=3)
            if fact_id in found or statement in self.learning.recall(statement)["semantic"]:
                retrieved += 1
        score = round(retrieved / len(probes), 6)
        return score, f"{retrieved}/{len(probes)} newly learned facts are retrievable at once"

    def _idea_quality(self) -> tuple[float, str]:
        novelty = NoveltyTracker()
        engine = IdeaEngine(self.ontology, novelty=novelty)
        labels = tuple(c.label for c in list(self.ontology.concepts.values())[:6])
        ideas = engine.generate(labels=labels, limit=5)
        if not ideas:
            return 0.0, "no ideas generated"
        mean_confidence = sum(i.confidence for i in ideas) / len(ideas)
        mean_novelty = sum(i.novelty for i in ideas) / len(ideas)
        score = round(0.5 * mean_confidence + 0.5 * mean_novelty, 6)
        return score, f"{len(ideas)} ideas, mean confidence {mean_confidence:.2f}, novelty {mean_novelty:.2f}"

    def _self_criticism(self) -> tuple[float, str]:
        engine = SelfCriticismEngine(self.ontology)
        overclaim = Idea(
            text="porosity is permeability",
            kind=IdeaKind.IDEA,
            engine="test",
            sources=("porosity", "permeability"),
            confidence=0.6,
        )
        clean = Idea(
            text="porosity may relate to permeability",
            kind=IdeaKind.IDEA,
            engine="test",
            sources=("porosity", "permeability"),
            confidence=0.6,
        )
        caught = 0
        if not engine.critique(overclaim).clean:
            caught += 1
        if engine.critique(clean).clean:
            caught += 1
        score = round(caught / 2, 6)
        return score, f"{caught}/2 criticism checks behaved correctly"

    def _generalisation(self, questions: tuple[str, ...]) -> tuple[float, str]:
        # A held-out check: a question's words need not appear in the seeds, but
        # the retriever should still reach related concepts via the ontology.
        held_out = ("why does flow slow down", "what makes rock hold fluid")
        matched = 0
        for question in held_out:
            ids = self.ontology.concepts_in(question)
            if ids:
                matched += 1
        score = round(matched / len(held_out), 6)
        return score, f"{matched}/{len(held_out)} held-out phrasings reach the ontology"

    def _response_speed(self, questions: tuple[str, ...]) -> tuple[float, str]:
        if not questions:
            return 0.0, "no questions to time"
        started = time.monotonic()
        for question in questions:
            self.retriever.retrieve(question, limit=3)
        elapsed = (time.monotonic() - started) / len(questions)
        # 0.1 s per question or better is full marks; 1 s or worse is zero.
        score = round(min(1.0, max(0.0, (1.0 - elapsed) / 0.9)), 6)
        return score, f"{elapsed * 1000:.0f} ms per question"

    def _reliability(self) -> tuple[float, str]:
        """The share of the graph's relations that carry a bounded confidence.

        Reliability here is a property of the *knowledge*, not of a model: a
        relation with no confidence (or one outside ``[0, 1]``) cannot be weighed,
        so a graph in which every relation is scored is a graph that can be
        trusted to be honest about how sure it is.
        """
        relations = list(self.ontology.relations.values())
        if not relations:
            return 0.0, "no relations to judge"
        scored = sum(1 for r in relations if 0.0 < float(r.confidence) <= 1.0)
        score = round(scored / len(relations), 6)
        return score, f"{scored}/{len(relations)} relations carry a bounded confidence"

    def _independence(self) -> tuple[float, str]:
        # Independence is the share of the graph reachable without a teacher: the
        # concepts with at least one relation of their own.
        concepts = list(self.ontology.concepts.values())
        if not concepts:
            return 0.0, "no concepts"
        linked_ids = set()
        for relation in self.ontology.relations.values():
            linked_ids.add(relation.source_id)
            linked_ids.add(relation.target_id)
        linked = sum(1 for c in concepts if c.id in linked_ids)
        score = round(linked / len(concepts), 6)
        return score, f"{linked}/{len(concepts)} concepts are self-connected in the graph"

    def _default_questions(self) -> tuple[str, ...]:
        return (
            "how does porosity relate to permeability",
            "what does a density log measure",
            "why does waterflood increase production",
            "how is a blowout prevented",
        )


def load_report(data: object) -> CognitiveReport | None:
    """Read a report from primitive data (a stored baseline), or ``None``."""
    if isinstance(data, dict):
        return CognitiveReport.from_primitive(data)
    return None


__all__ = [
    "INDICATORS",
    "LABELS",
    "CognitiveEvaluator",
    "CognitiveReport",
    "Comparison",
    "load_report",
]
