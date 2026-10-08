"""Navigate the project's science branches, courses, and concepts.

The project already organizes its knowledge as a degree: science branches
(categories such as *Petroleum Disciplines* or *Earth Sciences*), the courses
inside each branch, and the concepts inside each course. This module exposes
that structure as a single, ordered tree so a caller -- the Continuous Internet
Learner's interactive mode, in particular -- can walk it one level at a time.

Nothing new is invented here: it reads `core.curriculum_knowledge` and presents
its categories, courses, and concepts in a stable order, with human-readable
concept labels (the spaced alias when one exists, otherwise the underscored
label with its underscores replaced by spaces).
"""

from __future__ import annotations

from dataclasses import dataclass

# Petroleum first, then the sciences that support it, then the foundations, so
# the most relevant branch is also the first choice.
_CATEGORY_ORDER = (
    "Petroleum Disciplines",
    "Engineering Science",
    "Earth Sciences",
    "Foundations",
)


@dataclass(frozen=True)
class CourseNode:
    """One course and the searchable topics it contributes."""

    key: str
    label: str
    topics: tuple[str, ...]


@dataclass(frozen=True)
class BranchNode:
    """One science branch (a curriculum category) and its courses."""

    label: str
    courses: tuple[CourseNode, ...]

    @property
    def topic_count(self) -> int:
        return sum(len(course.topics) for course in self.courses)


def _concept_topic(label: str, aliases: tuple[str, ...]) -> str:
    """A readable, searchable phrase for one concept."""
    if aliases:
        return aliases[0].strip()
    return label.replace("_", " ").strip()


def _category_key(label: str) -> tuple[int, str]:
    try:
        return (_CATEGORY_ORDER.index(label), "")
    except ValueError:
        return (len(_CATEGORY_ORDER), label)


def science_branches() -> tuple[BranchNode, ...]:
    """The science branches, each with its courses and searchable topics."""
    from core.curriculum_knowledge import course_categories, courses

    categories = course_categories()
    grouped: dict[str, list[CourseNode]] = {}
    for course in courses():
        category = categories.get(course.key, "Other")
        topics = tuple(_concept_topic(label, aliases) for label, aliases in course.concepts)
        grouped.setdefault(category, []).append(
            CourseNode(key=course.key, label=course.label, topics=topics)
        )
    ordered = sorted(grouped.items(), key=lambda item: _category_key(item[0]))
    return tuple(BranchNode(label=label, courses=tuple(courses_)) for label, courses_ in ordered)


__all__ = ["BranchNode", "CourseNode", "science_branches"]
