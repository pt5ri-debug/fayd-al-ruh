"""The Mind's question bank: many distinct, non-repeating questions per run.

The Proactive Mind asks questions, but it drew them from a *tiny* pool -- the
same handful of gaps every cycle -- so the same questions came up again and
again. This module gives it a deep, deterministic bank of questions built from
the project's **own** knowledge (the ontology's concepts and relation types, the
corpus entries, the disciplines, the courses), so the mind can ask **two hundred**
distinct questions in one run and never repeat one across runs.

Two sources, both honest by construction:

* **Graph questions** -- a template per relation type ("What does {a} {relation}
  {b}?", "Why does {a} change?", ...) filled from the graph's *own* edges, so a
  question is always about a relation the project actually states;
* **Curated topic questions** -- the authored question sets the characters are
  seeded from (:data:`core.character_seeding.PETROLEUM_QUESTIONS` and
  :data:`core.character_seeding.PROJECT_QUESTIONS`), which name topics the corpus
  really covers.

Nothing is invented: a graph question names an edge the ontology holds, and a
topic question names a topic the corpus states. The bank is ordered
deterministically (by a stable key), and a persisted record of what has already
been asked lets :func:`next_questions` hand back the next *unasked* batch, so the
mind works through the bank rather than looping over its first few entries.

The store is a single bounded document under the project's own ``JsonStore``
(:data:`cognition.storage.NAMESPACE_QUESTIONS`); a corrupt record reads as empty
and a read-only folder is tolerated, so the bank always yields questions.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from hashlib import blake2b

from cognition.storage import NAMESPACE_QUESTIONS, _load_list, _save_list

#: The default size of one run's question set -- the number the task asks for.
DEFAULT_QUESTION_COUNT = 200

#: How many questions the store remembers as already asked, so the bank never
#: runs dry on a long-lived install and the document stays small.
MAX_ASKED = 4000

#: A question's stable identity, so "have we asked this?" is a set lookup and the
#: same question text always maps to the same key.
def question_key(text: str) -> str:
    """A stable key for a question's normalised text."""
    normalised = " ".join(text.split()).casefold()
    return blake2b(normalised.encode("utf-8"), digest_size=8).hexdigest()


@dataclass(frozen=True)
class BankQuestion:
    """One question in the bank, with where it came from."""

    text: str
    topic: str
    kind: str
    """``graph`` for a question built from a stated edge, ``topic`` for a curated one."""

    @property
    def key(self) -> str:
        return question_key(self.text)

    def describe(self) -> str:
        return f"[{self.kind}] {self.text}"

    def to_primitive(self) -> dict[str, str]:
        return {"text": self.text, "topic": self.topic, "kind": self.kind}

    @classmethod
    def from_primitive(cls, data: dict) -> BankQuestion:
        return cls(
            text=str(data.get("text", "")),
            topic=str(data.get("topic", "")),
            kind=str(data.get("kind", "topic")),
        )


#: A template per relation type, filled with the edge's own endpoints. ``{a}`` is
#: the source and ``{b}`` the target; a template may use either or both. The
#: relation's own word is inserted for the generic form.
_RELATION_TEMPLATES: dict[str, tuple[str, ...]] = {
    "is_a": ("What kind of thing is {a}?", "Is every {a} a {b}?"),
    "measures": ("What does {a} measure?", "How is {b} measured?"),
    "causes": ("Why does {a} cause {b}?", "What happens when {a} acts on {b}?"),
    "increases": ("Why does {a} increase {b}?", "How can {b} be increased?"),
    "reduces": ("Why does {a} reduce {b}?", "How can {b} be reduced?"),
    "prevents": ("How does {a} prevent {b}?", "What prevents {b}?"),
    "contains": ("What does {a} contain?", "Is {b} part of {a}?"),
    "depends_on": ("What does {a} depend on?", "How does {a} depend on {b}?"),
    "affects": ("How does {a} affect {b}?", "What does {b} depend on?"),
    "controls": ("How does {a} control {b}?", "What controls {b}?"),
    "drives": ("What does {a} drive?", "What is driven by {a}?"),
    "enables": ("What does {a} enable?", "What makes {b} possible?"),
    "requires": ("What does {a} require?", "What is needed for {b}?"),
    "produces": ("What does {a} produce?", "Where does {b} come from?"),
    "influences": ("How does {a} influence {b}?", "What influences {b}?"),
    "governs": ("What does {a} govern?", "What governs {b}?"),
    "determines": ("What does {a} determine?", "What determines {b}?"),
}

#: Templates that use only the *source* of an edge, so an edge whose target is a
#: bare value still yields a question about the source.
_SOURCE_TEMPLATES: tuple[str, ...] = (
    "What is {a}?",
    "What role does {a} play?",
    "What is important to know about {a}?",
)

#: Templates that use only the *target* of an edge.
_TARGET_TEMPLATES: tuple[str, ...] = (
    "What is {b}?",
    "What determines {b}?",
    "What should be known about {b}?",
)

#: Arabic templates for the relation types the graph uses most, so the mind asks
#: some of its questions in Arabic -- the language the project is Arabic-first for.
#: Only types named here yield an Arabic question; an unlisted type falls back to
#: the generic pair below, so no edge is silently dropped and the English bank is
#: unchanged.
ARABIC_RELATION_TEMPLATES: dict[str, tuple[str, ...]] = {
    "is_a": ("ما هو {a}؟", "هل {a} نوع من {b}؟"),
    "measures": ("كيف يقيس {a} {b}؟", "ما الذي يقيس {b}؟"),
    "increases": ("كيف يزيد {a} {b}؟", "ما الذي يزيد {b}؟"),
    "reduces": ("كيف يقلّل {a} {b}؟", "ما الذي يقلّل {b}؟"),
    "causes": ("كيف يسبب {a} {b}؟", "ما الذي يسبب {b}؟"),
    "controls": ("كيف يتحكم {a} في {b}؟", "ما الذي يتحكم في {b}؟"),
    "prevents": ("كيف يمنع {a} {b}؟", "ما الذي يمنع {b}؟"),
    "affects": ("كيف يؤثر {a} في {b}؟", "ما الذي يؤثر في {b}؟"),
    "drives": ("كيف يقود {a} إلى {b}؟", "ما الذي يقود إلى {b}؟"),
    "depends_on": ("على ماذا يعتمد {a}؟", "كيف يعتمد {a} على {b}؟"),
    "contains": ("ماذا يحتوي {a}؟", "هل {b} جزء من {a}؟"),
    "requires": ("ماذا يتطلب {a}؟", "ما المطلوب من أجل {b}؟"),
    "produces": ("ماذا ينتج {a}؟", "من أين يأتي {b}؟"),
    "enables": ("ماذا يمكّن {a}؟", "ما الذي يجعل {b} ممكنًا؟"),
    "supports": ("كيف يدعم {a} {b}؟", "ما الذي يدعم {b}؟"),
    "indicates": ("إلامَ يشير {a}؟", "ما الذي يشير إلى {b}؟"),
    "maintains": ("كيف يحافظ {a} على {b}؟", "ما الذي يحافظ على {b}؟"),
    "verifies": ("كيف يتحقق {a} من {b}؟", "ما الذي يتحقق من {b}؟"),
    "predicts": ("كيف يتنبأ {a} بـ{b}؟", "ما الذي يتنبأ بـ{b}؟"),
    "implements": ("كيف ينفّذ {a} {b}؟", "ما الذي ينفّذ {b}؟"),
    "forms": ("كيف يشكّل {a} {b}؟", "ما الذي يشكّل {b}؟"),
    "has": ("ماذا يمتلك {a}؟", "ما الذي يمتلك {b}؟"),
    "governs": ("كيف يحكم {a} {b}؟", "ما الذي يحكم {b}؟"),
    "determines": ("كيف يحدّد {a} {b}؟", "ما الذي يحدّد {b}؟"),
    "protects": ("كيف يحمي {a} {b}؟", "ما الذي يحمي {b}؟"),
    "influences": ("كيف يؤثر {a} في {b}؟", "ما الذي يؤثر في {b}؟"),
    "creates": ("كيف ينشئ {a} {b}؟", "ما الذي ينشئ {b}؟"),
    "estimates": ("كيف يقدّر {a} {b}؟", "ما الذي يقدّر {b}؟"),
    "balances": ("كيف يوازن {a} {b}؟", "ما الذي يوازن {b}؟"),
    "separates": ("كيف يفصل {a} {b}؟", "ما الذي يفصل {b}؟"),
    "improves": ("كيف يحسّن {a} {b}؟", "ما الذي يحسّن {b}؟"),
    "uses": ("كيف يستخدم {a} {b}؟", "ما الذي يستخدم {b}؟"),
    "defines": ("كيف يعرّف {a} {b}؟", "ما الذي يعرّف {b}؟"),
    "provides": ("ماذا يوفّر {a}؟", "ما الذي يوفّر {b}؟"),
    "has_property": ("ما خصائص {a}؟", "ما خصائص {b}؟"),
    "seals": ("كيف يسدّ {a} {b}؟", "ما الذي يسدّ {b}؟"),
    "connects": ("كيف يربط {a} بـ{b}؟", "ما الذي يربط {b}؟"),
    "generates": ("ماذا يولّد {a}؟", "ما الذي يولّد {b}؟"),
    "threatens": ("كيف يهدّد {a} {b}؟", "ما الذي يهدّد {b}؟"),
    "hosts": ("ماذا يحتضن {a}؟", "أين يوجد {b}؟"),
    "used_in": ("أين يُستخدم {a}؟", "ما الذي يُستخدم فيه {b}؟"),
    "part_of": ("هل {a} جزء من {b}؟", "ما الذي يكون {b} جزءًا منه؟"),
    "describes": ("كيف يصف {a} {b}؟", "ما الذي يصف {b}؟"),
    "identifies": ("كيف يحدّد {a} {b}؟", "ما الذي يحدّد {b}؟"),
    "manages": ("كيف يدير {a} {b}؟", "ما الذي يدير {b}؟"),
    "limits": ("كيف يحدّ {a} {b}؟", "ما الذي يحدّ {b}؟"),
    "exists_in": ("أين يوجد {a}؟", "ما الذي يوجد في {b}؟"),
    "drills": ("كيف يحفر {a} {b}؟", "ما الذي يحفر {b}؟"),
    "evaluates": ("كيف يقيّم {a} {b}؟", "ما الذي يقيّم {b}؟"),
    "stabilizes": ("كيف يثبّت {a} {b}؟", "ما الذي يثبّت {b}؟"),
    "processes": ("كيف يعالج {a} {b}؟", "ما الذي يعالج {b}؟"),
    "transports": ("كيف ينقل {a} {b}؟", "ما الذي ينقل {b}؟"),
    "stores": ("ماذا يخزّن {a}؟", "ما الذي يخزّن {b}؟"),
    "mitigates": ("كيف يخفّف {a} {b}؟", "ما الذي يخفّف {b}؟"),
    "guides": ("كيف يوجّه {a} {b}؟", "ما الذي يوجّه {b}؟"),
    "builds": ("ماذا يبني {a}؟", "ما الذي يبني {b}؟"),
}

#: The generic Arabic pair for a relation type with no template of its own, so
#: every stated edge contributes at least one Arabic question. The relation word
#: is left out on purpose (the Arabic relation vocabulary is not one-to-one with
#: the seeds'), so the question names the two concepts rather than guessing a verb.
ARABIC_GENERIC_TEMPLATES: tuple[str, ...] = (
    "ما العلاقة بين {a} و{b}؟",
)

#: Arabic questions that use only the source, for an edge whose target is a value.
ARABIC_SOURCE_TEMPLATES: tuple[str, ...] = (
    "ما هو {a}؟",
    "ما دور {a}؟",
)

#: Arabic questions that use only the target.
ARABIC_TARGET_TEMPLATES: tuple[str, ...] = (
    "ما هو {b}؟",
    "ما الذي يحدّد {b}؟",
)

#: How many Arabic graph questions one bank may hold. Bounded so the Arabic
#: addition widens the bank without swamping it, and deterministic (edges are
#: taken in the graph's own order) so the same graph yields the same bank.
MAX_ARABIC_GRAPH_QUESTIONS = 1000


def _arabic_graph_questions(ontology) -> list[BankQuestion]:
    """Questions about the graph's stated edges, asked in Arabic.

    Bounded by :data:`MAX_ARABIC_GRAPH_QUESTIONS` and built from the same edges as
    the English questions, so an Arabic question always names a relation the graph
    really states. A type with no template of its own uses the generic pair, so no
    edge is silently dropped.
    """
    from core.arabic_prose import arabic_label

    questions: list[BankQuestion] = []
    seen: set[str] = set()
    for relation in ontology.relations.values():
        if len(questions) >= MAX_ARABIC_GRAPH_QUESTIONS:
            break
        source = getattr(relation, "source_id", "") or getattr(relation, "source", "")
        target = getattr(relation, "target_id", "") or getattr(relation, "target", "")
        rtype = str(getattr(relation, "relation_type", "") or getattr(relation, "type", ""))
        a = arabic_label(_label_of(ontology, source), ontology)
        b = arabic_label(_label_of(ontology, target), ontology)
        if not a:
            continue
        templates = ARABIC_RELATION_TEMPLATES.get(rtype) or ARABIC_GENERIC_TEMPLATES
        for template in templates:
            if "{b}" in template and not b:
                continue
            text = template.format(a=a, b=b)
            key = question_key(text)
            if key in seen:
                continue
            seen.add(key)
            questions.append(BankQuestion(text=text, topic=a, kind="graph"))
        for template in ARABIC_SOURCE_TEMPLATES:
            text = template.format(a=a)
            key = question_key(text)
            if key not in seen:
                seen.add(key)
                questions.append(BankQuestion(text=text, topic=a, kind="graph"))
        if b:
            for template in ARABIC_TARGET_TEMPLATES:
                text = template.format(b=b)
                key = question_key(text)
                if key not in seen:
                    seen.add(key)
                    questions.append(BankQuestion(text=text, topic=b, kind="graph"))
    return questions


def _graph_questions(ontology) -> list[BankQuestion]:
    """Questions built from the graph's own stated edges.

    Each edge contributes the templates its relation type names; an unknown
    relation type contributes the generic pair (source-only and target-only), so
    no edge is silently dropped. De-duplicated by question key.
    """
    questions: list[BankQuestion] = []
    seen: set[str] = set()
    for relation in ontology.relations.values():
        source = getattr(relation, "source_id", "") or getattr(relation, "source", "")
        target = getattr(relation, "target_id", "") or getattr(relation, "target", "")
        rtype = str(getattr(relation, "relation_type", "") or getattr(relation, "type", ""))
        a = _label_of(ontology, source)
        b = _label_of(ontology, target)
        if not a:
            continue
        templates = _RELATION_TEMPLATES.get(rtype, ())
        for template in templates:
            if "{b}" in template and not b:
                continue
            text = template.format(a=a, b=b)
            key = question_key(text)
            if key in seen:
                continue
            seen.add(key)
            questions.append(BankQuestion(text=text, topic=a, kind="graph"))
        # A source-only and a target-only question per edge, so an edge always
        # contributes at least one question whatever its relation type.
        for template in _SOURCE_TEMPLATES:
            text = template.format(a=a)
            key = question_key(text)
            if key not in seen:
                seen.add(key)
                questions.append(BankQuestion(text=text, topic=a, kind="graph"))
        if b:
            for template in _TARGET_TEMPLATES:
                text = template.format(b=b)
                key = question_key(text)
                if key not in seen:
                    seen.add(key)
                    questions.append(BankQuestion(text=text, topic=b, kind="graph"))
    return questions


def _label_of(ontology, concept_id: str) -> str:
    """The readable label of a concept id, or the id itself when unknown."""
    concept = ontology.concepts.get(concept_id) if concept_id else None
    if concept is not None:
        return str(getattr(concept, "label", concept_id))
    return str(concept_id) if concept_id and not concept_id.startswith("CON-") else ""


def _topic_questions() -> list[BankQuestion]:
    """The curated, authored question sets the characters are seeded from."""
    from core.character_seeding import PETROLEUM_QUESTIONS, PROJECT_QUESTIONS

    questions: list[BankQuestion] = []
    seen: set[str] = set()
    for text in (*PETROLEUM_QUESTIONS, *PROJECT_QUESTIONS):
        key = question_key(text)
        if key in seen:
            continue
        seen.add(key)
        questions.append(BankQuestion(text=text, topic=text, kind="topic"))
    return questions


def build_bank(ontology) -> tuple[BankQuestion, ...]:
    """The whole question bank for an ontology, ordered and de-duplicated.

    Topic questions come first (they are authored and always valid), then the
    English graph questions, then the Arabic graph questions, each group ordered by
    a stable key so the same ontology yields the same bank every time.
    """
    topics = _topic_questions()
    graph = _graph_questions(ontology)
    arabic = _arabic_graph_questions(ontology)
    ordered = (
        sorted(topics, key=lambda q: q.text)
        + sorted(graph, key=lambda q: (q.topic, q.text))
        + sorted(arabic, key=lambda q: (q.topic, q.text))
    )
    seen: set[str] = set()
    unique: list[BankQuestion] = []
    for question in ordered:
        if question.key in seen:
            continue
        seen.add(question.key)
        unique.append(question)
    return tuple(unique)


@dataclass
class QuestionLog:
    """A bounded record of the questions the mind has already asked.

    Backed by the project's own ``JsonStore`` under
    :data:`cognition.storage.NAMESPACE_QUESTIONS`; a corrupt document reads as
    empty and a read-only folder is tolerated, so the mind always has a log.
    """

    max_asked: int = MAX_ASKED
    _asked: list[str] = field(default_factory=list, repr=False)

    def __post_init__(self) -> None:
        if not self._asked:
            self._asked = [
                str(d.get("key", "")) for d in _load_list(NAMESPACE_QUESTIONS) if d.get("key")
            ][-self.max_asked :]

    def asked_keys(self) -> frozenset[str]:
        return frozenset(self._asked)

    def asked_count(self) -> int:
        return len(self._asked)

    def has(self, question: BankQuestion) -> bool:
        return question.key in self._asked

    def record(self, questions: tuple[BankQuestion, ...]) -> None:
        """Mark questions as asked, keeping the log bounded and newest-last."""
        known = set(self._asked)
        for question in questions:
            if question.key not in known:
                self._asked.append(question.key)
                known.add(question.key)
        del self._asked[: -self.max_asked]
        self._flush()

    def clear(self) -> None:
        self._asked = []
        self._flush()

    def _flush(self) -> None:
        _save_list(NAMESPACE_QUESTIONS, [{"key": key} for key in self._asked])


def next_questions(
    ontology,
    *,
    count: int = DEFAULT_QUESTION_COUNT,
    log: QuestionLog | None = None,
    record: bool = False,
) -> tuple[BankQuestion, ...]:
    """The next ``count`` questions not yet asked, wrapping when the bank is spent.

    When fewer than ``count`` unasked questions remain, the ones already asked
    are re-offered *after* the fresh ones -- so a run always gets ``count``
    questions, but fresh material is always preferred and the batch never repeats
    within itself. With ``record=True`` the returned questions are marked asked.
    """
    if count < 1:
        raise ValueError("count must be at least 1.")
    bank = build_bank(ontology)
    active_log = log if log is not None else QuestionLog()
    asked = active_log.asked_keys()
    fresh = [q for q in bank if q.key not in asked]
    stale = [q for q in bank if q.key in asked]
    chosen = (fresh + stale)[:count]
    if record:
        active_log.record(tuple(chosen))
    return tuple(chosen)


def bank_size(ontology) -> int:
    """How many distinct questions the bank holds for an ontology."""
    return len(build_bank(ontology))


__all__ = [
    "DEFAULT_QUESTION_COUNT",
    "MAX_ASKED",
    "BankQuestion",
    "QuestionLog",
    "bank_size",
    "build_bank",
    "next_questions",
    "question_key",
]
