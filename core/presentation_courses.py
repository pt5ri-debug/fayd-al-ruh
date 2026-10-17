"""Match a presentation's content to the project's own curriculum.

The Presentation Studio develops scientific and technical material, and the
project already organizes that same material as a degree: science branches, the
courses inside them, and the concepts inside each course
(:mod:`core.knowledge_navigator`, reading :mod:`core.curriculum_knowledge`).

This module is the bridge between the two. It answers one question -- *which of
the project's courses does this deck actually cover, and how well?* -- so a front
end can offer the next real step: open the matching course in the interactive
learner, or ground the deck in the course's own concepts.

Two rules keep it honest, matching the rest of the project:

* it invents nothing. Every course, concept, and label comes from the curriculum
  seed; this module only scores and ranks what is already there.
* it does not guess. A deck that matches nothing returns no courses, rather than
  the "closest" one, so a caller never sends the user to the wrong course.

Scoring reuses :mod:`core.rag`'s own Arabic-aware normaliser, so a concept written
in Arabic matches the same way a query does everywhere else in the project.
"""

from __future__ import annotations

from dataclasses import dataclass

from core.rag import _query_terms  # the project's one Arabic-aware normaliser

#: A course is only *suggested* when at least this many of its own concepts are
#: present in the deck. One accidental word ("pressure") must not nominate a whole
#: discipline, and the threshold is deliberately low enough that a genuinely
#: on-topic deck always clears it.
DEFAULT_MIN_MATCHES = 2

#: The share of a course's concepts a deck must cover to count as a *strong*
#: match. Below this the course is offered as related, not as the deck's subject.
STRONG_COVERAGE = 0.25


@dataclass(frozen=True)
class CourseMatch:
    """One curriculum course the deck covers, with the evidence for the match."""

    key: str
    label: str
    branch: str
    #: The course's concepts found in the deck, in the curriculum's own order.
    matched: tuple[str, ...]
    #: How many concepts the course declares in total.
    total: int

    @property
    def match_count(self) -> int:
        """How many of the course's concepts the deck mentions."""
        return len(self.matched)

    @property
    def coverage(self) -> float:
        """The share of the course's concepts the deck mentions (0.0 - 1.0)."""
        return self.match_count / self.total if self.total else 0.0

    @property
    def strong(self) -> bool:
        """True when the deck covers enough of the course to be *about* it."""
        return self.coverage >= STRONG_COVERAGE

    def describe(self) -> str:
        """One honest line naming the course and the concepts that matched."""
        shown = ", ".join(self.matched[:4])
        more = f" (+{self.match_count - 4} more)" if self.match_count > 4 else ""
        return f"{self.label} — {self.branch}: {shown}{more}"


def _concept_phrases(concepts: tuple[tuple[str, tuple[str, ...]], ...]) -> tuple[str, ...]:
    """The searchable phrases for a course's concepts.

    A concept's aliases are its own human-readable names ("effective porosity"),
    so they are what a deck would actually say; the underscored label is the
    fallback when a concept declares none. Each phrase's *terms* are what get
    matched, so a multi-word concept counts as covered when its words are present.
    """
    phrases: list[str] = []
    for label, aliases in concepts:
        for alias in aliases or (label,):
            cleaned = alias.strip()
            if cleaned:
                phrases.append(cleaned)
    return tuple(phrases)


def project_text(project: object) -> str:
    """All the user-facing text of a project: title, outline, and every slide.

    The deck's *own* words are what decide which course it covers -- not the
    discussion that produced it, which may wander -- so this gathers the title,
    subtitle, the outline's headings and subsections, and each slide's title,
    bullets, table cells, and equations.
    """
    parts: list[str] = []
    for attribute in ("title", "subtitle", "audience"):
        value = getattr(project, attribute, "")
        if value:
            parts.append(str(value))

    outline = getattr(project, "outline", None)
    if outline is not None:
        parts.append(str(getattr(outline, "heading", "") or ""))
        for subsection in getattr(outline, "subsections", ()) or ():
            parts.append(str(subsection))

    for slide in getattr(project, "ordered_slides", lambda: [])():
        parts.append(str(getattr(slide, "title", "") or ""))
        parts.append(str(getattr(slide, "section", "") or ""))
        parts.append(str(getattr(slide, "subsection", "") or ""))
        parts.extend(str(line) for line in (getattr(slide, "content", ()) or ()))
        parts.extend(str(line) for line in (getattr(slide, "equations", ()) or ()))
        for row in getattr(slide, "table", ()) or ():
            if isinstance(row, (list, tuple)):
                parts.extend(str(cell) for cell in row)
            else:
                parts.append(str(row))
    return "\n".join(part for part in parts if part)


def match_courses(
    text: str,
    *,
    min_matches: int = DEFAULT_MIN_MATCHES,
    limit: int | None = None,
) -> tuple[CourseMatch, ...]:
    """The curriculum courses ``text`` covers, strongest first.

    A course qualifies when at least ``min_matches`` of its own concepts appear in
    the text. Ties are broken by coverage, then by the course label, so the order
    is stable across runs and platforms.
    """
    terms = _query_terms(text)
    if not terms:
        return ()

    from core.curriculum_knowledge import course_categories, courses

    categories = course_categories()
    matches: list[CourseMatch] = []
    for course in courses():
        phrases = _concept_phrases(course.concepts)
        # A concept counts as present when every content term of one of its
        # phrases is in the deck; so "effective porosity" needs both words, while
        # a one-word concept needs only its own.
        found: list[str] = []
        for phrase in phrases:
            phrase_terms = _query_terms(phrase)
            if phrase_terms and phrase_terms <= terms:
                found.append(phrase)
        # Distinct concepts, not aliases: one concept with two spellings is one match.
        unique = tuple(dict.fromkeys(found))
        if len(unique) >= min_matches:
            matches.append(
                CourseMatch(
                    key=course.key,
                    label=course.label,
                    branch=categories.get(course.key, "Other"),
                    matched=unique,
                    total=len(course.concepts),
                )
            )

    matches.sort(key=lambda match: (-match.match_count, -match.coverage, match.label))
    if limit is not None and limit >= 0:
        matches = matches[:limit]
    return tuple(matches)


def match_project(
    project: object,
    *,
    min_matches: int = DEFAULT_MIN_MATCHES,
    limit: int | None = None,
) -> tuple[CourseMatch, ...]:
    """The courses a project's own content covers (see :func:`project_text`)."""
    return match_courses(project_text(project), min_matches=min_matches, limit=limit)


def best_course(matches: tuple[CourseMatch, ...]) -> CourseMatch | None:
    """The strongest *strong* match, or ``None`` when nothing clears the bar.

    A caller uses this to decide whether to *open* a course window: a weak,
    incidental overlap is not a course the deck is about, so it is not offered as
    one.
    """
    for match in matches:
        if match.strong:
            return match
    return None


@dataclass(frozen=True)
class ConceptInfo:
    """One concept of a course, and whether the deck already mentions it."""

    label: str
    #: The human-readable name(s) the concept is known by.
    phrases: tuple[str, ...]
    present: bool

    @property
    def name(self) -> str:
        """The concept's readable name (its first alias, else its label)."""
        return self.phrases[0] if self.phrases else self.label.replace("_", " ")


@dataclass(frozen=True)
class RelationInfo:
    """One curriculum relation between two concepts, in readable words."""

    source: str
    target: str
    relation: str
    confidence: float

    def describe(self) -> str:
        return f"{self.source} —{self.relation}→ {self.target}"


@dataclass(frozen=True)
class CourseDetail:
    """A course's own structure, with the deck's coverage marked per concept.

    This is what the Course window shows: the course's concepts (each marked
    *covered* or not, decided by the same matcher that nominated the course), and
    the relations the curriculum declares among them -- so the user sees not just
    which course the deck belongs to but which of its ideas are still missing.
    """

    key: str
    label: str
    branch: str
    concepts: tuple[ConceptInfo, ...]
    relations: tuple[RelationInfo, ...]

    @property
    def covered(self) -> tuple[ConceptInfo, ...]:
        return tuple(concept for concept in self.concepts if concept.present)

    @property
    def missing(self) -> tuple[ConceptInfo, ...]:
        return tuple(concept for concept in self.concepts if not concept.present)


def _readable(label: str) -> str:
    return label.replace("_", " ").strip()


def course_detail(key: str, *, text: str = "") -> CourseDetail:
    """One course's concepts and relations, marking what ``text`` already covers.

    ``key`` is a curriculum course key (``CourseMatch.key``). An unknown key
    raises ``KeyError`` rather than returning an empty course, so a caller never
    shows a blank window for a typo.
    """
    from core.curriculum_knowledge import course_categories, courses

    course = next((c for c in courses() if c.key == key), None)
    if course is None:
        raise KeyError(f"unknown course: {key}")

    terms = _query_terms(text)
    concepts: list[ConceptInfo] = []
    for label, aliases in course.concepts:
        phrases = tuple(alias.strip() for alias in (aliases or (label,)) if alias.strip())
        present = any(
            (phrase_terms := _query_terms(phrase)) and phrase_terms <= terms
            for phrase in phrases
        )
        concepts.append(ConceptInfo(label=label, phrases=phrases, present=present))

    readable = {label: (aliases[0] if aliases else _readable(label))
                for label, aliases in course.concepts}
    relations = tuple(
        RelationInfo(
            source=readable.get(source, _readable(source)),
            target=readable.get(target, _readable(target)),
            relation=relation,
            confidence=float(confidence),
        )
        for source, target, relation, confidence in course.relations
    )
    return CourseDetail(
        key=course.key,
        label=course.label,
        branch=course_categories().get(course.key, "Other"),
        concepts=tuple(concepts),
        relations=relations,
    )


__all__ = [
    "DEFAULT_MIN_MATCHES",
    "STRONG_COVERAGE",
    "ConceptInfo",
    "CourseDetail",
    "CourseMatch",
    "RelationInfo",
    "best_course",
    "course_detail",
    "match_courses",
    "match_project",
    "project_text",
]
