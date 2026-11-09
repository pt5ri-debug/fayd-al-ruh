"""Turn a pasted training schedule (days and topics) into a structured plan.

A user often has a schedule -- "Day 1: Porosity and permeability", "Day 2:
Well logging", ... -- and wants a professional deck built from it. This module
is the deterministic reader for that text: it recognises a *day* (or session,
week, module, lecture) line in English or Arabic, keeps the topic that follows
it, and gathers any sub-topic lines beneath it. It invents nothing: a line that
is not a day and not under one is kept as a plain topic, so a bare topic list
still parses, and the order the user wrote is the order returned.

The result is a :class:`SchedulePlan` the Studio turns into an outline (one
section per day) and then into numbered slides. Dependency-free and deterministic,
so the whole path is testable with no model.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from core.presentation import OutlineSection

#: Bounds so a runaway paste cannot make an unbounded plan.
MAX_TOPICS = 60
MAX_DETAILS = 12
MAX_TITLE_CHARS = 120

#: A worked example schedule, so the expected shape is obvious and the reader can
#: be demonstrated with no input at all.
SCHEDULE_EXAMPLE = (
    "Day 1: Introduction to reservoir rock properties\n"
    "- Porosity and its measurement\n"
    "- Permeability and Darcy's law\n"
    "Day 2: Well logging and petrophysics\n"
    "- Gamma ray and resistivity logs\n"
    "Day 3: Well testing and production"
)

#: The words that open a day/session line, in both languages.
_EN_DAY_NOUNS = (
    "day",
    "session",
    "week",
    "module",
    "unit",
    "lecture",
    "lesson",
    "class",
    "part",
)
_AR_DAY_NOUNS = (
    "اليوم",
    "يوم",
    "الجلسة",
    "جلسة",
    "المحاضرة",
    "محاضرة",
    "الأسبوع",
    "أسبوع",
    "الوحدة",
    "وحدة",
    "الحلقة",
    "حلقة",
    "الدرس",
    "درس",
)

#: English number words (one .. twenty) and Arabic ordinals/cardinals.
_EN_NUMBERS = {
    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7,
    "eight": 8, "nine": 9, "ten": 10, "eleven": 11, "twelve": 12, "thirteen": 13,
    "fourteen": 14, "fifteen": 15, "sixteen": 16, "seventeen": 17, "eighteen": 18,
    "nineteen": 19, "twenty": 20,
}
_AR_NUMBERS = {
    "الأول": 1, "الاول": 1, "الأولى": 1, "الاولى": 1, "واحد": 1,
    "الثاني": 2, "الثانية": 2, "اثنان": 2, "اثنين": 2,
    "الثالث": 3, "الثالثة": 3, "ثلاثة": 3,
    "الرابع": 4, "الرابعة": 4, "أربعة": 4,
    "الخامس": 5, "الخامسة": 5, "خمسة": 5,
    "السادس": 6, "السادسة": 6, "ستة": 6,
    "السابع": 7, "السابعة": 7, "سبعة": 7,
    "الثامن": 8, "الثامنة": 8, "ثمانية": 8,
    "التاسع": 9, "التاسعة": 9, "تسعة": 9,
    "العاشر": 10, "العاشرة": 10, "عشرة": 10,
}

_TATWEEL = "\u0640"
_DIACRITICS = re.compile(r"[\u0617-\u061A\u064B-\u0652\u0670\u0640]")
_SEPARATORS = ").:\u2013\u2014\u00b7\u060c-"
_NUMBER_WORD_RE = re.compile(r"^[0-9]{1,2}$")


def _leading_number(rest: str) -> tuple[int, str, str]:
    """Read a leading day number from ``rest``; return ``(number, remainder, word)``.

    Accepts a digit ("1", "12"), an English word ("one"), or an Arabic
    ordinal/cardinal ("الأول"). ``word`` is the ordinal word as written (empty
    for a plain digit), so an Arabic day keeps its own wording. When no number is
    present the number is 0 and the whole ``rest`` is the remainder.
    """
    tokens = rest.split(" ", 1)
    if not tokens:
        return 0, "", ""
    head = tokens[0].strip(_SEPARATORS).strip()
    tail = tokens[1] if len(tokens) > 1 else ""
    if _NUMBER_WORD_RE.match(head):
        return int(head), tail.strip(_SEPARATORS).strip(), ""
    lowered = head.casefold()
    if lowered in _EN_NUMBERS:
        return _EN_NUMBERS[lowered], tail.strip(_SEPARATORS).strip(), ""
    if head in _AR_NUMBERS:
        return _AR_NUMBERS[head], tail.strip(_SEPARATORS).strip(), head
    return 0, rest.strip(_SEPARATORS).strip(), ""


def _day_line(raw: str) -> tuple[str, int, str] | None:
    """Split a line into ``(label, number, title)`` when it opens a day.

    The label is the day noun as written ("Day", "اليوم"), plus the ordinal word
    when one was used ("اليوم الأول"); the number is the parsed day number (0 when
    none was given); the title is the topic that follows. Returns ``None`` for a
    line that does not open a day.
    """
    line = raw.strip().lstrip("#-*\u2022\u00b7 \t").strip()
    if not line:
        return None
    head, _, remainder = line.partition(" ")
    head_clean = head.strip(_SEPARATORS).strip()
    lowered = head_clean.casefold()
    if lowered in _EN_DAY_NOUNS:
        number, title, _word = _leading_number(remainder.strip())
        return head_clean, number, title
    if head_clean in _AR_DAY_NOUNS:
        number, title, word = _leading_number(remainder.strip())
        label = f"{head_clean} {word}".strip() if word else head_clean
        return label, number, title
    return None


@dataclass(frozen=True)
class ScheduleTopic:
    """One day/session of the schedule: its label, number, title, and details."""

    label: str
    number: int
    title: str
    details: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.title.strip():
            raise ValueError("A schedule topic needs a title.")

    @property
    def heading(self) -> str:
        """A readable heading, e.g. ``"Day 1 — Porosity and permeability"``.

        An Arabic label that already carries its ordinal ("اليوم الأول") is used
        as written; an English label appends its number ("Day 1").
        """
        if " " in self.label:
            prefix = self.label
        else:
            prefix = f"{self.label} {self.number}".strip() if self.number else self.label
        return f"{prefix} — {self.title}" if prefix else self.title

    def to_primitive(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "number": self.number,
            "title": self.title,
            "details": list(self.details),
        }

    @classmethod
    def from_primitive(cls, data: dict[str, Any]) -> ScheduleTopic:
        return cls(
            label=str(data.get("label", "")),
            number=int(data.get("number", 0)),
            title=str(data.get("title", "")),
            details=tuple(str(item) for item in data.get("details", ())),
        )


@dataclass(frozen=True)
class SchedulePlan:
    """A parsed training schedule: a title and an ordered list of topics."""

    title: str = ""
    topics: tuple[ScheduleTopic, ...] = field(default_factory=tuple)

    @property
    def is_empty(self) -> bool:
        return not self.topics

    def topic_headings(self) -> tuple[str, ...]:
        """The day headings, in order (used for the outline sections)."""
        return tuple(topic.heading for topic in self.topics)

    def outline_sections(self) -> tuple[OutlineSection, ...]:
        """The outline the schedule implies: one section per day.

        A day with sub-topics yields them as the section's subsections (so the
        slide drafting develops each piece); a day without them yields one
        section, which the existing pipeline turns into one topic.
        """
        sections: list[OutlineSection] = []
        for topic in self.topics:
            sections.append(
                OutlineSection(
                    heading=topic.heading,
                    subsections=topic.details[:MAX_DETAILS],
                    estimated_slides=max(1, len(topic.details) or 1),
                )
            )
        return tuple(sections)

    def as_prompt(self) -> str:
        """The plan as a compact block for a model prompt."""
        lines = [f"- {topic.heading}" for topic in self.topics]
        for topic in self.topics:
            for detail in topic.details:
                lines.append(f"    - {detail}")
        return "\n".join(lines)

    def describe(self) -> str:
        """A readable summary of the parsed plan."""
        if self.is_empty:
            return "No schedule topics were found."
        header = f"Schedule: {self.title}" if self.title else "Schedule"
        lines = [f"{header} ({len(self.topics)} topic(s))"]
        for topic in self.topics:
            lines.append(f"  • {topic.heading}")
            for detail in topic.details:
                lines.append(f"      - {detail}")
        return "\n".join(lines)

    def to_primitive(self) -> dict[str, Any]:
        return {
            "title": self.title,
            "topics": [topic.to_primitive() for topic in self.topics],
        }

    @classmethod
    def from_primitive(cls, data: dict[str, Any]) -> SchedulePlan:
        return cls(
            title=str(data.get("title", "")),
            topics=tuple(
                ScheduleTopic.from_primitive(item) for item in data.get("topics", ())
            ),
        )


def parse_schedule(text: str, *, title: str = "") -> SchedulePlan:
    """Parse a pasted schedule into a :class:`SchedulePlan`.

    A day line (``Day 1: ...``, ``اليوم الأول ...``, ``Session 3 ...``) opens a
    topic; the lines under it, until the next day, are its sub-topics. When no
    day line is found at all, every non-blank line becomes its own topic, so a
    bare list of topics still yields a usable plan. Order is preserved and blank
    lines are ignored. Returns an empty plan when nothing parses.
    """
    topics: list[ScheduleTopic] = []
    current: ScheduleTopic | None = None
    saw_day = False
    fallback: list[ScheduleTopic] = []

    for raw in (text or "").splitlines():
        stripped = raw.strip()
        if not stripped:
            continue
        day = _day_line(raw)
        if day is not None:
            saw_day = True
            label, number, topic_title = day
            topic_title = topic_title.strip()[:MAX_TITLE_CHARS]
            if not topic_title:
                # "Day 3" with nothing after it: the day is the topic.
                topic_title = f"{label} {number}".strip() if number else label
            if current is not None and topics:
                topics[-1] = current
            current = ScheduleTopic(label=label, number=number, title=topic_title)
            topics.append(current)
            if len(topics) >= MAX_TOPICS:
                break
            continue
        if current is not None:
            detail = stripped.lstrip("#-*•· \t").strip()[:MAX_TITLE_CHARS]
            if detail and len(current.details) < MAX_DETAILS:
                current = ScheduleTopic(
                    label=current.label,
                    number=current.number,
                    title=current.title,
                    details=(*current.details, detail),
                )
                topics[-1] = current
            continue
        # No day opened yet: remember the line as a fallback topic.
        line = stripped.lstrip("#-*•· \t").strip()
        if line:
            fallback.append(
                ScheduleTopic(label="", number=len(fallback) + 1, title=line[:MAX_TITLE_CHARS])
            )

    if not saw_day:
        topics = fallback[:MAX_TOPICS]
    plan_title = title.strip() or (topics[0].title if len(topics) == 1 else "")
    return SchedulePlan(title=plan_title, topics=tuple(topics))


def outline_from_schedule(plan: SchedulePlan) -> tuple[OutlineSection, ...]:
    """The outline a plan implies (a thin, readable alias of the method)."""
    return plan.outline_sections()


__all__ = [
    "MAX_DETAILS",
    "MAX_TOPICS",
    "SCHEDULE_EXAMPLE",
    "SchedulePlan",
    "ScheduleTopic",
    "outline_from_schedule",
    "parse_schedule",
]
