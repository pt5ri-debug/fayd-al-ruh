"""Consolidation: turn raw experience into durable knowledge, and keep it.

The cognitive learning system *records* -- facts, experiences, procedures -- but
recording is not learning. This module is what happens between: it reads the raw
records back and consolidates them, the way sleep is said to consolidate a day.

Five operations, each one deterministic and honest:

* :meth:`Consolidator.consolidate` -- read the experiences, find the ones that
  repeat, and extract a **pattern** (a rule with a support count and a
  confidence). A rule resting on one anecdote is never stated; a rule needs real
  support.
* **Semantic upgrade** -- a pattern that is strong enough is written into semantic
  memory as a learned fact, with its provenance, so a lesson becomes recallable
  knowledge. It is stored *unverified* (a pattern is an inference, not a fact),
  which is exactly the epistemic rule the rest of the package keeps.
* :class:`ReviewSchedule` -- **spaced review**: every consolidated item is given a
  review date that grows each time it is recalled correctly and resets when it is
  forgotten, so durable knowledge is kept durable.
* :class:`SkillRegistry` -- **skill refinement**: procedures are practised and
  scored, and a skill's proficiency is its measured success rate, not a guess.
* :class:`ConsolidationReport` -- what one pass actually did, so the window and
  the CLI can show it without inventing anything.

Nothing here writes to the graph or to a character; it writes only to the
learning system's own memories and to its own bounded stores.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta

from core.tokenize import normalize_token

from cognition.records import Experience, Source, SourceKind
from cognition.storage import (
    NAMESPACE_PATTERNS,
    NAMESPACE_REVIEWS,
    doc_get,
    doc_set,
)

#: A rule needs at least this much support before it is stated as a pattern.
MIN_PATTERN_SUPPORT = 2

#: A pattern at or above this confidence may be written into semantic memory.
SEMANTIC_UPGRADE_CONFIDENCE = 0.6

#: The spaced-review intervals, in days, by Leitner box. The box advances on a
#: correct recall and resets to 0 on a miss.
REVIEW_INTERVALS_DAYS: tuple[int, ...] = (1, 3, 7, 16, 35)

#: Stop-words that carry no topic signal, so a pattern is not built on them.
_STOPWORDS = frozenset(
    {
        "the", "and", "for", "that", "with", "this", "from", "into", "were", "was",
        "are", "not", "but", "its", "has", "had", "have", "will", "would", "can",
        "may", "might", "does", "did", "than", "then", "when", "where", "which",
    }
)

MAX_PATTERNS = 400
MAX_REVIEWS = 800
MAX_SKILLS = 200


def _now() -> datetime:
    return datetime.now(UTC)


def _tokens(text: str) -> frozenset[str]:
    return frozenset(
        t
        for t in (normalize_token(w) for w in text.split())
        if t and len(t) > 2 and t not in _STOPWORDS
    )


# ---------------------------------------------------------------------------
# Patterns
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Pattern:
    """A recurring regularity the mind extracted from its own experience."""

    condition: str
    consequent: str
    kind: str
    support: int
    confidence: float
    id: str = ""
    created_at: datetime = field(default_factory=_now)

    def __post_init__(self) -> None:
        if not self.condition.strip() or not self.consequent.strip():
            raise ValueError("A pattern needs both a condition and a consequent.")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("Pattern confidence must be between 0 and 1.")
        if not self.id:
            # Content-addressed, so the same regularity keeps the same id across
            # runs (a per-process ``hash()`` would change every restart and the
            # same pattern would be stored twice).
            from hashlib import blake2b

            digest = blake2b(
                f"{self.condition}|{self.consequent}".encode(), digest_size=6
            )
            object.__setattr__(self, "id", f"PAT-{digest.hexdigest()}")

    def statement(self) -> str:
        """The pattern as one readable sentence."""
        connector = {
            "success_rule": "usually works",
            "failure_rule": "usually fails",
            "association_rule": "usually appears together",
        }.get(self.kind, "usually holds")
        return f"When {self.condition}, {self.consequent} {connector}"

    def to_primitive(self) -> dict[str, object]:
        return {
            "id": self.id,
            "condition": self.condition,
            "consequent": self.consequent,
            "kind": self.kind,
            "support": self.support,
            "confidence": self.confidence,
            "created_at": self.created_at.isoformat(),
        }

    @classmethod
    def from_primitive(cls, data: dict[str, object]) -> Pattern:
        def _num(key: str, fallback: float) -> float:
            value = data.get(key, fallback)
            return float(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else fallback

        def _int(key: str, fallback: int) -> int:
            value = data.get(key, fallback)
            return int(value) if isinstance(value, int) and not isinstance(value, bool) else fallback

        created = data.get("created_at")
        try:
            created_at = datetime.fromisoformat(str(created)) if created else _now()
        except ValueError:
            created_at = _now()
        return cls(
            id=str(data.get("id", "")),
            condition=str(data.get("condition", "")),
            consequent=str(data.get("consequent", "")),
            kind=str(data.get("kind", "success_rule")),
            support=_int("support", 1),
            confidence=_num("confidence", 0.5),
            created_at=created_at,
        )

    def describe(self) -> str:
        return f"{self.statement()} (support {self.support}, confidence {self.confidence:.2f})"


# ---------------------------------------------------------------------------
# Spaced review
# ---------------------------------------------------------------------------


@dataclass
class ReviewItem:
    """One piece of knowledge under spaced review."""

    item_id: str
    statement: str
    box: int = 0
    reviews: int = 0
    lapses: int = 0
    due_at: datetime = field(default_factory=_now)

    @property
    def interval_days(self) -> int:
        return REVIEW_INTERVALS_DAYS[min(self.box, len(REVIEW_INTERVALS_DAYS) - 1)]

    def schedule(self, *, remembered: bool, now: datetime | None = None) -> None:
        """Advance the box on a correct recall, reset it on a miss."""
        now = now or _now()
        self.reviews += 1
        if remembered:
            self.box = min(self.box + 1, len(REVIEW_INTERVALS_DAYS) - 1)
        else:
            self.lapses += 1
            self.box = 0
        self.due_at = now + timedelta(days=self.interval_days)

    def is_due(self, *, now: datetime | None = None) -> bool:
        return (now or _now()) >= self.due_at

    def to_primitive(self) -> dict[str, object]:
        return {
            "item_id": self.item_id,
            "statement": self.statement,
            "box": self.box,
            "reviews": self.reviews,
            "lapses": self.lapses,
            "due_at": self.due_at.isoformat(),
        }

    @classmethod
    def from_primitive(cls, data: dict[str, object]) -> ReviewItem:
        def _int(key: str, fallback: int = 0) -> int:
            value = data.get(key, fallback)
            return int(value) if isinstance(value, int) and not isinstance(value, bool) else fallback

        due = data.get("due_at")
        try:
            due_at = datetime.fromisoformat(str(due)) if due else _now()
        except ValueError:
            due_at = _now()
        return cls(
            item_id=str(data.get("item_id", "")),
            statement=str(data.get("statement", "")),
            box=_int("box"),
            reviews=_int("reviews"),
            lapses=_int("lapses"),
            due_at=due_at,
        )

    def describe(self) -> str:
        state = "due" if self.is_due() else f"due {self.due_at.date().isoformat()}"
        return f"[box {self.box}] {self.statement} ({state}, {self.reviews} review(s))"


class ReviewSchedule:
    """A bounded, persistent spaced-review queue over consolidated knowledge."""

    def __init__(self, *, items: dict[str, ReviewItem] | None = None, max_items: int = MAX_REVIEWS) -> None:
        self.max_items = max_items
        self._items: dict[str, ReviewItem] = dict(items or {})

    @classmethod
    def load(cls, *, max_items: int = MAX_REVIEWS) -> ReviewSchedule:
        data = doc_get(NAMESPACE_REVIEWS)
        items: dict[str, ReviewItem] = {}
        if isinstance(data, dict):
            raw = data.get("items")
            if isinstance(raw, dict):
                for key, entry in raw.items():
                    if isinstance(entry, dict):
                        item = ReviewItem.from_primitive(entry)
                        items[str(key)] = item
        return cls(items=items, max_items=max_items)

    def save(self) -> None:
        doc_set(
            NAMESPACE_REVIEWS,
            {"items": {k: v.to_primitive() for k, v in self._items.items()}},
        )

    def add(self, item_id: str, statement: str, *, now: datetime | None = None) -> ReviewItem:
        """Put a statement under review (idempotent by id)."""
        existing = self._items.get(item_id)
        if existing is not None:
            existing.statement = statement
            return existing
        item = ReviewItem(item_id=item_id, statement=statement, due_at=now or _now())
        self._items[item_id] = item
        self._trim()
        return item

    def review(self, item_id: str, *, remembered: bool, now: datetime | None = None) -> ReviewItem | None:
        item = self._items.get(item_id)
        if item is None:
            return None
        item.schedule(remembered=remembered, now=now)
        self.save()
        return item

    def due(self, *, now: datetime | None = None, limit: int = 20) -> tuple[ReviewItem, ...]:
        now = now or _now()
        ordered = sorted(
            (i for i in self._items.values() if i.is_due(now=now)),
            key=lambda i: (i.due_at, i.item_id),
        )
        return tuple(ordered[:limit])

    def all(self) -> tuple[ReviewItem, ...]:
        return tuple(sorted(self._items.values(), key=lambda i: i.item_id))

    def count(self) -> int:
        return len(self._items)

    def _trim(self) -> None:
        if len(self._items) <= self.max_items:
            return
        ordered = sorted(self._items.values(), key=lambda i: (i.box, i.due_at, i.item_id))
        for item in ordered[: len(self._items) - self.max_items]:
            self._items.pop(item.item_id, None)


# ---------------------------------------------------------------------------
# Skills
# ---------------------------------------------------------------------------


@dataclass
class Skill:
    """A procedure the mind has practised, with its measured proficiency."""

    name: str
    steps: tuple[str, ...] = ()
    attempts: int = 0
    successes: int = 0

    @property
    def proficiency(self) -> float:
        return round(self.successes / self.attempts, 6) if self.attempts else 0.0

    def practise(self, *, success: bool, steps: tuple[str, ...] = ()) -> None:
        self.attempts += 1
        if success:
            self.successes += 1
        if steps:
            # Refinement: keep the longer, more detailed step list, which is the
            # one that carries more procedure -- never a shorter, vaguer one.
            if len(steps) >= len(self.steps):
                self.steps = tuple(steps)

    def to_primitive(self) -> dict[str, object]:
        return {
            "name": self.name,
            "steps": list(self.steps),
            "attempts": self.attempts,
            "successes": self.successes,
        }

    @classmethod
    def from_primitive(cls, data: dict[str, object]) -> Skill:
        raw_steps = data.get("steps")
        steps = tuple(str(s) for s in raw_steps) if isinstance(raw_steps, list) else ()

        def _int(key: str) -> int:
            value = data.get(key, 0)
            return int(value) if isinstance(value, int) and not isinstance(value, bool) else 0

        return cls(
            name=str(data.get("name", "")),
            steps=steps,
            attempts=_int("attempts"),
            successes=_int("successes"),
        )

    def describe(self) -> str:
        return (
            f"{self.name}: {self.successes}/{self.attempts} "
            f"(proficiency {self.proficiency:.0%}, {len(self.steps)} step(s))"
        )


class SkillRegistry:
    """The persistent record of practised procedures and their proficiency."""

    def __init__(self, *, skills: dict[str, Skill] | None = None, max_items: int = MAX_SKILLS) -> None:
        self.max_items = max_items
        self._skills: dict[str, Skill] = dict(skills or {})

    @classmethod
    def load(cls, *, max_items: int = MAX_SKILLS) -> SkillRegistry:
        data = doc_get(NAMESPACE_PATTERNS, "skills")
        skills: dict[str, Skill] = {}
        if isinstance(data, list):
            for entry in data:
                if isinstance(entry, dict):
                    skill = Skill.from_primitive(entry)
                    skills[skill.name] = skill
        return cls(skills=skills, max_items=max_items)

    def save(self) -> None:
        doc_set(NAMESPACE_PATTERNS, [s.to_primitive() for s in self._skills.values()], "skills")

    def practise(self, name: str, *, success: bool, steps: tuple[str, ...] = ()) -> Skill:
        skill = self._skills.get(name) or Skill(name=name)
        skill.practise(success=success, steps=steps)
        self._skills[name] = skill
        self._trim()
        self.save()
        return skill

    def skills(self) -> tuple[Skill, ...]:
        return tuple(sorted(self._skills.values(), key=lambda s: (-s.proficiency, s.name)))

    def count(self) -> int:
        return len(self._skills)

    def _trim(self) -> None:
        if len(self._skills) <= self.max_items:
            return
        ordered = sorted(self._skills.values(), key=lambda s: (s.proficiency, s.name))
        for skill in ordered[: len(self._skills) - self.max_items]:
            self._skills.pop(skill.name, None)


# ---------------------------------------------------------------------------
# Consolidation
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ConsolidationReport:
    """What one consolidation pass actually did."""

    patterns: tuple[Pattern, ...] = ()
    upgraded: tuple[str, ...] = ()
    reviewed: int = 0
    scheduled: int = 0

    def describe(self) -> str:
        lines = [
            f"consolidation: {len(self.patterns)} pattern(s), "
            f"{len(self.upgraded)} fact(s) upgraded, {self.scheduled} under review"
        ]
        lines.extend(f"  pattern: {p.describe()}" for p in self.patterns)
        lines.extend(f"  upgraded: {statement}" for statement in self.upgraded)
        return "\n".join(lines)


class Consolidator:
    """Consolidate the learning system's raw records into durable knowledge.

    It reads the experiences the mind has accumulated, groups the ones that share
    a topic, and -- only when a group has real support -- states a pattern and,
    when it is strong enough, writes it into semantic memory as a learned fact.
    Every consolidated statement is also put under spaced review, so it is kept.
    A pattern resting on a single anecdote is never stated: a rule needs support.
    """

    def __init__(
        self,
        learning,
        *,
        review: ReviewSchedule | None = None,
        skills: SkillRegistry | None = None,
        patterns: tuple[Pattern, ...] | None = None,
    ) -> None:
        self.learning = learning
        self.review = review or ReviewSchedule.load()
        self.skills = skills or SkillRegistry.load()
        self._patterns: list[Pattern] = list(patterns) if patterns is not None else self._load_patterns()

    # -- persistence --------------------------------------------------------
    @staticmethod
    def _load_patterns() -> list[Pattern]:
        data = doc_get(NAMESPACE_PATTERNS, "all")
        if isinstance(data, list):
            return [Pattern.from_primitive(d) for d in data if isinstance(d, dict)]
        return []

    def _save_patterns(self) -> None:
        doc_set(NAMESPACE_PATTERNS, [p.to_primitive() for p in self._patterns], "all")

    def patterns(self) -> tuple[Pattern, ...]:
        return tuple(self._patterns)

    # -- the pass -----------------------------------------------------------
    def consolidate(self, *, now: datetime | None = None) -> ConsolidationReport:
        """Extract patterns from the experiences, upgrade the strong ones, review."""
        experiences = self.learning.experiences.all()
        groups: dict[frozenset[str], list[Experience]] = {}
        for experience in experiences:
            key = self._topic(experience)
            if key:
                groups.setdefault(key, []).append(experience)

        found: list[Pattern] = []
        upgraded: list[str] = []
        for key, group in sorted(groups.items(), key=lambda pair: (-len(pair[1]), sorted(pair[0]))):
            pattern = self._pattern_from(key, group)
            if pattern is None:
                continue
            found.append(pattern)
            self._remember_pattern(pattern)
            if pattern.confidence >= SEMANTIC_UPGRADE_CONFIDENCE:
                statement = pattern.statement()
                self.learning.learn_fact(
                    pattern.id,
                    statement,
                    source=Source(SourceKind.INFERENCE, "consolidation"),
                    confidence=pattern.confidence,
                    verified=False,
                )
                self.review.add(pattern.id, statement, now=now)
                upgraded.append(statement)

        # Also keep every learned fact under review, so recall stays durable.
        for fact_id, statement in self.learning.facts.all():
            self.review.add(fact_id, statement, now=now)

        self._save_patterns()
        self.review.save()
        return ConsolidationReport(
            patterns=tuple(found),
            upgraded=tuple(upgraded),
            reviewed=0,
            scheduled=self.review.count(),
        )

    def review_due(self, *, remembered: bool = True, now: datetime | None = None, limit: int = 5) -> tuple[ReviewItem, ...]:
        """Settle the items that are due, advancing or resetting their box."""
        due = self.review.due(now=now, limit=limit)
        for item in due:
            self.review.review(item.item_id, remembered=remembered, now=now)
        return due

    def practise_skill(self, name: str, *, success: bool, steps: tuple[str, ...] = ()) -> Skill:
        """Practise a procedure and record its outcome (skill refinement)."""
        return self.skills.practise(name, success=success, steps=steps)

    # -- internals ----------------------------------------------------------
    def _topic(self, experience: Experience) -> frozenset[str]:
        return _tokens(f"{experience.expectation} {experience.outcome}")

    def _pattern_from(self, key: frozenset[str], group: list[Experience]) -> Pattern | None:
        if len(group) < MIN_PATTERN_SUPPORT:
            return None
        successes = [e for e in group if e.success]
        failures = [e for e in group if not e.success]
        if len(successes) >= len(failures) and len(successes) >= MIN_PATTERN_SUPPORT:
            chosen, kind = successes, "success_rule"
        elif len(failures) >= MIN_PATTERN_SUPPORT:
            chosen, kind = failures, "failure_rule"
        else:
            # Mixed evidence with no clear side: state the association only, which
            # is honest about not knowing which way it goes.
            chosen, kind = group, "association_rule"
        condition = " and ".join(sorted(key)[:3])
        consequent = chosen[0].outcome.strip() or chosen[0].expectation.strip()
        confidence = round(len(chosen) / len(group), 6)
        return Pattern(
            condition=condition,
            consequent=consequent,
            kind=kind,
            support=len(chosen),
            confidence=confidence,
        )

    def _remember_pattern(self, pattern: Pattern) -> None:
        for index, existing in enumerate(self._patterns):
            if existing.id == pattern.id:
                self._patterns[index] = pattern
                return
        self._patterns.append(pattern)
        del self._patterns[:-MAX_PATTERNS]

    def describe(self) -> str:
        lines = [f"{len(self._patterns)} consolidated pattern(s)"]
        lines.extend(f"  {p.describe()}" for p in self._patterns[-10:])
        return "\n".join(lines)


__all__ = [
    "MIN_PATTERN_SUPPORT",
    "REVIEW_INTERVALS_DAYS",
    "SEMANTIC_UPGRADE_CONFIDENCE",
    "ConsolidationReport",
    "Consolidator",
    "Pattern",
    "ReviewItem",
    "ReviewSchedule",
    "Skill",
    "SkillRegistry",
]
