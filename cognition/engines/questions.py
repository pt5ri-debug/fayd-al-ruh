"""Questions: turn gaps and drives into answerable, evidence-seeking questions.

The project's own :class:`core.question_generator.QuestionGenerator` produces one
generic question per gap ("What evidence would establish X?"). That is correct
but monotonous, so this engine adds *variety* without inventing content: it reads
the graph's own relation types around a topic and asks the question the missing
knowledge actually calls for --

* a **degree** gap -> "How does X relate to the concepts around it?";
* a **contradiction** gap -> "Which of these claims about X holds?";
* a **mechanism** gap -> "Why does X change?".

Every question is a real :class:`core.question.Question`, so it flows through the
existing inquiry machinery unchanged.
"""

from __future__ import annotations

from core.information_gap import InformationGap
from core.ontology import Ontology
from core.question import Question
from core.question_generator import QuestionGenerator

#: Relation types whose absence is a *mechanism* question, not a link question.
_MECHANISM = frozenset({"causes", "drives", "increases", "reduces", "prevents", "controls"})


class SelfQuestioningEngine:
    """Compose varied, answerable questions from gaps and the graph's own types."""

    def __init__(
        self,
        ontology: Ontology,
        *,
        generator: QuestionGenerator | None = None,
    ) -> None:
        self.ontology = ontology
        self.generator = generator or QuestionGenerator()
        self._labels = {c.label for c in ontology.concepts.values()}
        self._outgoing_types = self._index_relation_types()

    def questions(
        self,
        gaps: tuple[InformationGap, ...],
        *,
        limit: int = 5,
    ) -> tuple[Question, ...]:
        """Return up to ``limit`` distinct questions, highest priority first."""
        if limit < 1:
            raise ValueError("limit must be at least 1.")
        ordered = sorted(gaps, key=lambda g: (-g.priority, g.topic))
        questions: list[Question] = []
        seen: set[str] = set()
        for gap in ordered:
            question = self._question_for(gap)
            if question.text in seen:
                continue
            seen.add(question.text)
            questions.append(question)
            if len(questions) >= limit:
                break
        return tuple(questions)

    def _question_for(self, gap: InformationGap) -> Question:
        topic = gap.topic
        if " vs " in topic:
            return Question(
                text=f"Which of the conflicting claims about {topic} is supported?",
                topic=topic,
                priority=round(gap.priority, 6),
            )
        types = self._outgoing_types.get(topic, frozenset())
        if not types:
            text = f"What is {topic} connected to in the knowledge graph?"
        elif types & _MECHANISM:
            text = f"Why does {topic} change, and through what mechanism?"
        else:
            text = f"How does {topic} relate to the concepts around it?"
        return Question(text=text, topic=topic, priority=round(gap.priority, 6))

    def _index_relation_types(self) -> dict[str, frozenset[str]]:
        labels = {c.id: c.label for c in self.ontology.concepts.values()}
        index: dict[str, set[str]] = {}
        for relation in self.ontology.relations.values():
            source = labels.get(relation.source_id)
            if source:
                index.setdefault(source, set()).add(relation.relation_type)
        return {label: frozenset(types) for label, types in index.items()}


__all__ = ["SelfQuestioningEngine"]
