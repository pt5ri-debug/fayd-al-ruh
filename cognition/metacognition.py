"""Metacognition: the mind's model of its own thinking, so it can improve it.

The Proactive Mind's first eight engines all think *about the world*. Nothing in
it thinks about **the thinking** -- so the mind could not tell which of its own
strategies actually paid off, could not notice it was going in circles, and could
not change how it searched when a topic stopped yielding. This module is that
missing faculty, and it is what turns a fixed loop into a *developing* one.

Five parts, each one honest and deterministic:

* :class:`SelfModel` -- the durable record of how each **strategy** (each engine
  that produced ideas) has performed: how many ideas it produced, how many
  survived self-criticism, their mean confidence and novelty, and how many were
  promoted up the epistemic ladder. From those facts it computes a
  :class:`StrategyYield` and a **bias** per strategy, so the next cycle leans on
  what works and still explores what it has neglected (an explicit
  explore/exploit balance, never a random choice).
* :class:`StagnationDetector` -- reads the recent cycles and says whether the mind
  is *stagnating*: producing less novelty, promoting nothing, and circling the
  same topics. A stagnation is a measurement, not a mood.
* :class:`EscalationPolicy` -- when stagnating, decides how to **escalate**: widen
  the search (more concepts, more hops), shift the strategy focus, or raise the
  novelty floor so the mind stops repeating itself. Every escalation is bounded
  by explicit ceilings, so "escalate" means *harder*, never *unbounded*.
* :class:`PersistentNovelty` -- the novelty memory, made durable. The in-RAM
  :class:`~cognition.engines.ideas.NoveltyTracker` forgets everything on exit, so
  a restarted mind re-thinks its own old thoughts; this keeps them.
* :class:`FocusBuffer` -- a small, bounded working memory of the topics in focus,
  with decay, so the mind deepens a line of thought instead of thrashing between
  unrelated topics every cycle.

Nothing here widens the model's authority: it changes *how* the mind searches,
never *what* it is allowed to believe. A strategy bias is a search hint, not a
fact, and an escalated cycle still passes through the same self-criticism and the
same epistemic ladder.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime

from cognition.records import Idea, IdeaKind, ThoughtCycle
from cognition.storage import (
    NAMESPACE_NOVELTY,
    NAMESPACE_SELF_MODEL,
    doc_get,
    doc_set,
)

#: The explore/exploit balance: how much of a strategy bias comes from a
#: strategy's own record versus a flat prior that keeps an untried one alive.
EXPLORE_WEIGHT = 0.35

#: A strategy that has produced nothing for this many cycles is "cold" and gets a
#: fresh look even when its past record is poor -- the mind must not lock itself
#: out of a strategy that the graph has only just grown able to support.
COLD_AFTER_CYCLES = 8

#: Bounds, so the self-model and the buffers stay a fixed size.
MAX_STRATEGIES = 40
MAX_FOCUS = 12
MAX_NOVELTY_ITEMS = 800


def _now() -> datetime:
    return datetime.now(UTC)


def _clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return round(min(max(value, low), high), 6)


# ---------------------------------------------------------------------------
# Self-model
# ---------------------------------------------------------------------------


@dataclass
class StrategyRecord:
    """How one strategy (engine) has actually performed, as running totals.

    Only facts the mind can observe are kept: ideas produced, ideas that survived
    criticism, the confidence and novelty they carried, and how many reached a
    higher rung. Nothing is estimated.
    """

    name: str
    produced: int = 0
    kept: int = 0
    promoted: int = 0
    confidence_sum: float = 0.0
    novelty_sum: float = 0.0
    last_cycle: int = 0

    @property
    def keep_rate(self) -> float:
        return round(self.kept / self.produced, 6) if self.produced else 0.0

    @property
    def mean_confidence(self) -> float:
        return round(self.confidence_sum / self.produced, 6) if self.produced else 0.0

    @property
    def mean_novelty(self) -> float:
        return round(self.novelty_sum / self.produced, 6) if self.produced else 0.0

    def observe(self, idea: Idea, *, kept: bool, cycle: int) -> None:
        """Fold one produced idea into the running totals."""
        self.produced += 1
        self.confidence_sum += idea.confidence
        self.novelty_sum += idea.novelty
        if kept:
            self.kept += 1
        if idea.kind.rank > IdeaKind.IDEA.rank:
            self.promoted += 1
        self.last_cycle = cycle

    def to_primitive(self) -> dict[str, object]:
        return {
            "name": self.name,
            "produced": self.produced,
            "kept": self.kept,
            "promoted": self.promoted,
            "confidence_sum": self.confidence_sum,
            "novelty_sum": self.novelty_sum,
            "last_cycle": self.last_cycle,
        }

    @classmethod
    def from_primitive(cls, data: dict[str, object]) -> StrategyRecord:
        def _num(key: str, fallback: float = 0.0) -> float:
            value = data.get(key, fallback)
            return float(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else fallback

        def _int(key: str, fallback: int = 0) -> int:
            value = data.get(key, fallback)
            return int(value) if isinstance(value, int) and not isinstance(value, bool) else fallback

        return cls(
            name=str(data.get("name", "")),
            produced=_int("produced"),
            kept=_int("kept"),
            promoted=_int("promoted"),
            confidence_sum=_num("confidence_sum"),
            novelty_sum=_num("novelty_sum"),
            last_cycle=_int("last_cycle"),
        )


@dataclass(frozen=True)
class StrategyYield:
    """A strategy's payoff, and how much the next cycle should lean on it."""

    name: str
    produced: int
    kept: int
    promoted: int
    keep_rate: float
    mean_confidence: float
    mean_novelty: float
    bias: float

    def describe(self) -> str:
        return (
            f"{self.name}: {self.kept}/{self.produced} kept, {self.promoted} promoted, "
            f"conf {self.mean_confidence:.2f}, novelty {self.mean_novelty:.2f}, "
            f"bias {self.bias:.2f}"
        )


class SelfModel:
    """The durable record of which strategies pay off, and the bias they imply.

    The *value* of a strategy is a bounded blend of three observable facts -- how
    often its ideas survive criticism, how confident they are, and how novel they
    are -- so a strategy that produces a lot of weak, unoriginal ideas does not
    win by volume. The *bias* mixes that value with a flat prior
    (:data:`EXPLORE_WEIGHT`), so a proven strategy is favoured without ever
    locking out an untried one; and a strategy that has been cold for
    :data:`COLD_AFTER_CYCLES` cycles is given the full prior again, because the
    graph may have grown able to support it since it last ran.
    """

    def __init__(self, *, records: dict[str, StrategyRecord] | None = None, cycle: int = 0) -> None:
        self._records: dict[str, StrategyRecord] = dict(records or {})
        self.cycle = cycle

    # -- persistence --------------------------------------------------------
    @classmethod
    def load(cls) -> SelfModel:
        """Read the self-model from disk, or start a fresh one."""
        data = doc_get(NAMESPACE_SELF_MODEL)
        if not isinstance(data, dict):
            return cls()
        raw_records = data.get("records")
        records: dict[str, StrategyRecord] = {}
        if isinstance(raw_records, dict):
            for name, entry in raw_records.items():
                if isinstance(entry, dict):
                    records[str(name)] = StrategyRecord.from_primitive(entry)
        cycle = data.get("cycle")
        return cls(records=records, cycle=int(cycle) if isinstance(cycle, int) else 0)

    def save(self) -> None:
        doc_set(
            NAMESPACE_SELF_MODEL,
            {
                "records": {name: r.to_primitive() for name, r in self._records.items()},
                "cycle": self.cycle,
            },
        )

    # -- observation --------------------------------------------------------
    def advance_cycle(self) -> int:
        """Move the mind's own cycle counter on, returning the new value."""
        self.cycle += 1
        return self.cycle

    def observe(self, engine: str, idea: Idea, *, kept: bool) -> None:
        """Fold one produced idea into its strategy's record."""
        name = engine or "unknown"
        record = self._records.get(name)
        if record is None:
            if len(self._records) >= MAX_STRATEGIES:
                # Drop the least-productive strategy so a long-running mind is
                # bounded; a dropped strategy simply starts fresh if it returns.
                weakest = min(self._records.values(), key=lambda r: (r.kept, r.produced, r.name))
                self._records.pop(weakest.name, None)
            record = StrategyRecord(name=name)
            self._records[name] = record
        record.observe(idea, kept=kept, cycle=self.cycle)

    # -- reading ------------------------------------------------------------
    def records(self) -> tuple[StrategyRecord, ...]:
        return tuple(sorted(self._records.values(), key=lambda r: r.name))

    def yield_for(self, name: str) -> StrategyYield:
        """The payoff of one strategy, with its search bias."""
        record = self._records.get(name) or StrategyRecord(name=name)
        return self._yield(record)

    def yields(self) -> tuple[StrategyYield, ...]:
        """Every known strategy's payoff, strongest first."""
        computed = [self._yield(r) for r in self._records.values()]
        computed.sort(key=lambda y: (-y.bias, y.name))
        return tuple(computed)

    def bias(self, name: str) -> float:
        """How much to lean on a strategy: ``0.5`` for an untried one (neutral)."""
        record = self._records.get(name)
        if record is None or record.produced == 0:
            return 0.5
        return self._yield(record).bias

    def _yield(self, record: StrategyRecord) -> StrategyYield:
        # Value: kept-rate dominates (did the idea survive review?), with
        # confidence and novelty as smaller, bounded contributions. Promotions
        # are the strongest evidence a strategy is worth running.
        promotion_term = min(1.0, record.promoted / max(1, record.produced))
        value = (
            0.5 * record.keep_rate
            + 0.2 * record.mean_confidence
            + 0.15 * record.mean_novelty
            + 0.15 * promotion_term
        )
        cold = (self.cycle - record.last_cycle) >= COLD_AFTER_CYCLES
        if record.produced == 0 or cold:
            bias = 0.5
        else:
            bias = _clamp((1.0 - EXPLORE_WEIGHT) * value + EXPLORE_WEIGHT * 0.5)
        return StrategyYield(
            name=record.name,
            produced=record.produced,
            kept=record.kept,
            promoted=record.promoted,
            keep_rate=record.keep_rate,
            mean_confidence=record.mean_confidence,
            mean_novelty=record.mean_novelty,
            bias=bias,
        )

    def describe(self) -> str:
        yields = self.yields()
        if not yields:
            return "The mind has not yet learned which of its strategies pay off."
        lines = [f"Strategy self-model after {self.cycle} cycle(s):"]
        lines.extend(f"  {y.describe()}" for y in yields)
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Persistent novelty
# ---------------------------------------------------------------------------


class PersistentNovelty:
    """The novelty memory, made durable and shared with the idea engine.

    It keeps the token sets of the statements the mind has already thought, so a
    restarted mind does not re-think its own old thoughts. It exposes the same
    :meth:`novelty`/:meth:`remember` surface as
    :class:`~cognition.engines.ideas.NoveltyTracker`, so the idea engine accepts
    it unchanged -- the mind simply hands it a durable one.
    """

    def __init__(self, *, items: tuple[frozenset[str], ...] = (), max_items: int = MAX_NOVELTY_ITEMS) -> None:
        self.max_items = max_items
        self._seen: list[frozenset[str]] = list(items)

    @classmethod
    def load(cls, *, max_items: int = MAX_NOVELTY_ITEMS) -> PersistentNovelty:
        data = doc_get(NAMESPACE_NOVELTY)
        items: list[frozenset[str]] = []
        if isinstance(data, dict):
            raw = data.get("items")
            if isinstance(raw, list):
                for entry in raw:
                    if isinstance(entry, list):
                        items.append(frozenset(str(t) for t in entry))
        return cls(items=tuple(items), max_items=max_items)

    def save(self) -> None:
        doc_set(NAMESPACE_NOVELTY, {"items": [sorted(tokens) for tokens in self._seen]})

    def novelty(self, text: str) -> float:
        tokens = self._tokens(text)
        if not tokens or not self._seen:
            return 1.0 if tokens else 0.0
        closest = 0.0
        for other in self._seen:
            union = tokens | other
            if not union:
                continue
            overlap = len(tokens & other) / len(union)
            closest = max(closest, overlap)
        return round(max(0.0, 1.0 - closest), 6)

    def remember(self, text: str) -> None:
        tokens = self._tokens(text)
        if tokens:
            self._seen.append(tokens)
            del self._seen[:-self.max_items]

    def count(self) -> int:
        return len(self._seen)

    def _tokens(self, text: str) -> frozenset[str]:
        from core.tokenize import normalize_token

        return frozenset(
            t for t in (normalize_token(w) for w in text.split()) if t and len(t) > 2
        )


# ---------------------------------------------------------------------------
# Focus buffer
# ---------------------------------------------------------------------------


@dataclass
class FocusEntry:
    """One topic the mind is holding in mind, with its decayed weight."""

    topic: str
    weight: float = 1.0
    cycles: int = 0

    def to_primitive(self) -> dict[str, object]:
        return {"topic": self.topic, "weight": self.weight, "cycles": self.cycles}

    @classmethod
    def from_primitive(cls, data: dict[str, object]) -> FocusEntry:
        weight = data.get("weight", 1.0)
        cycles = data.get("cycles", 0)
        return cls(
            topic=str(data.get("topic", "")),
            weight=float(weight) if isinstance(weight, (int, float)) and not isinstance(weight, bool) else 1.0,
            cycles=int(cycles) if isinstance(cycles, int) and not isinstance(cycles, bool) else 0,
        )


class FocusBuffer:
    """A small, bounded working memory of the topics in focus, with decay.

    Holding a topic in focus for more than one cycle is what lets the mind
    *deepen* a line of thought instead of jumping to a fresh topic every time. A
    topic's weight decays each cycle it is not reinforced, and the buffer is
    bounded, so a stale focus is forgotten rather than accumulated forever.
    """

    def __init__(self, *, entries: tuple[FocusEntry, ...] = (), decay: float = 0.6, max_items: int = MAX_FOCUS) -> None:
        self.decay = decay
        self.max_items = max_items
        self._entries: dict[str, FocusEntry] = {e.topic: e for e in entries}

    @classmethod
    def load(cls, *, decay: float = 0.6, max_items: int = MAX_FOCUS) -> FocusBuffer:
        data = doc_get(NAMESPACE_SELF_MODEL, "focus")
        entries: list[FocusEntry] = []
        if isinstance(data, list):
            entries = [FocusEntry.from_primitive(d) for d in data if isinstance(d, dict)]
        return cls(entries=tuple(entries), decay=decay, max_items=max_items)

    def save(self) -> None:
        doc_set(NAMESPACE_SELF_MODEL, [e.to_primitive() for e in self._entries.values()], "focus")

    def reinforce(self, topic: str, *, amount: float = 1.0) -> None:
        """Raise a topic's weight (cap it at 1.0) and mark it current."""
        if not topic.strip():
            return
        entry = self._entries.get(topic) or FocusEntry(topic=topic, weight=0.0, cycles=0)
        entry.weight = _clamp(entry.weight + amount)
        entry.cycles += 1
        self._entries[topic] = entry
        self._trim()

    def decay_all(self) -> None:
        """Fade every topic that was not reinforced this cycle."""
        for entry in list(self._entries.values()):
            entry.weight = round(entry.weight * self.decay, 6)
            if entry.weight < 0.05:
                self._entries.pop(entry.topic, None)
        self._trim()

    def topics(self, *, limit: int = 5) -> tuple[str, ...]:
        """The topics in focus, strongest first."""
        ordered = sorted(self._entries.values(), key=lambda e: (-e.weight, e.topic))
        return tuple(e.topic for e in ordered[:limit])

    def weight(self, topic: str) -> float:
        entry = self._entries.get(topic)
        return entry.weight if entry is not None else 0.0

    def clear(self) -> None:
        self._entries.clear()

    def _trim(self) -> None:
        if len(self._entries) <= self.max_items:
            return
        ordered = sorted(self._entries.values(), key=lambda e: (e.weight, e.topic))
        for entry in ordered[: len(self._entries) - self.max_items]:
            self._entries.pop(entry.topic, None)


# ---------------------------------------------------------------------------
# Stagnation and escalation
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class StagnationReading:
    """Whether the mind is going in circles, and the facts that show it."""

    stagnating: bool
    novelty_trend: float
    promotions: int
    repeated_topics: int
    reason: str = ""

    def describe(self) -> str:
        verdict = "stagnating" if self.stagnating else "making progress"
        return (
            f"{verdict}: novelty trend {self.novelty_trend:+.2f}, "
            f"{self.promotions} promotion(s), {self.repeated_topics} repeated topic(s)"
            + (f" -- {self.reason}" if self.reason else "")
        )


class StagnationDetector:
    """Read the recent cycles and decide whether the mind is stagnating.

    Stagnation is defined by *three* observable facts together, never one alone,
    so a single quiet cycle is not mistaken for a rut: novelty must be flat or
    falling, nothing may have been promoted, and the recent cycles must be
    circling a small set of topics. Any one of these being healthy means the mind
    is still learning.
    """

    def __init__(self, *, window: int = 6, novelty_floor: float = 0.35) -> None:
        self.window = window
        self.novelty_floor = novelty_floor

    def read(
        self,
        cycles: tuple[ThoughtCycle, ...],
        ideas_by_id: dict[str, Idea],
    ) -> StagnationReading:
        recent = cycles[-self.window :]
        if len(recent) < max(2, self.window // 2):
            return StagnationReading(
                stagnating=False,
                novelty_trend=0.0,
                promotions=0,
                repeated_topics=0,
                reason="not enough cycles to judge yet",
            )
        # Novelty trend: the mean novelty of the older half vs the newer half.
        novelty_by_cycle = [self._mean_novelty(c, ideas_by_id) for c in recent]
        half = len(novelty_by_cycle) // 2
        older = sum(novelty_by_cycle[:half]) / max(1, half)
        newer = sum(novelty_by_cycle[half:]) / max(1, len(novelty_by_cycle) - half)
        trend = round(newer - older, 6)
        promotions = sum(
            1
            for c in recent
            for idea_id in c.ideas
            if (idea := ideas_by_id.get(idea_id)) is not None and idea.kind.rank > IdeaKind.IDEA.rank
        )
        topics = [t for c in recent for t in c.gaps]
        repeated = len(topics) - len(set(topics))
        flat = newer <= self.novelty_floor or trend <= 0.0
        stagnating = flat and promotions == 0 and repeated > 0
        reason = ""
        if stagnating:
            reason = "novelty is flat, nothing is being promoted, and the topics repeat"
        return StagnationReading(
            stagnating=stagnating,
            novelty_trend=trend,
            promotions=promotions,
            repeated_topics=repeated,
            reason=reason,
        )

    def _mean_novelty(self, cycle: ThoughtCycle, ideas_by_id: dict[str, Idea]) -> float:
        values = [
            idea.novelty
            for idea_id in cycle.ideas
            if (idea := ideas_by_id.get(idea_id)) is not None
        ]
        return sum(values) / len(values) if values else 0.0


@dataclass(frozen=True)
class Escalation:
    """How the mind will search harder after a stagnation, within hard ceilings."""

    widen_concepts: int
    widen_ideas: int
    novelty_floor: float
    strategy_focus: str
    reason: str = ""

    def describe(self) -> str:
        return (
            f"escalate: +{self.widen_concepts} concept(s), +{self.widen_ideas} idea(s), "
            f"novelty floor {self.novelty_floor:.2f}, focus {self.strategy_focus or 'none'}"
            + (f" -- {self.reason}" if self.reason else "")
        )


class EscalationPolicy:
    """Decide how to search harder when the mind stagnates -- never unbounded.

    An escalation widens the search (more concepts, more ideas) and raises the
    novelty floor so the mind stops repeating itself, and it may shift the
    strategy focus to the strategy with the best record. Each field is capped, so
    an escalated cycle is *bigger*, not *unlimited*; the scheduler's own per-cycle
    budget still applies on top.
    """

    def __init__(
        self,
        *,
        max_widen_concepts: int = 4,
        max_widen_ideas: int = 3,
        max_novelty_floor: float = 0.6,
    ) -> None:
        self.max_widen_concepts = max_widen_concepts
        self.max_widen_ideas = max_widen_ideas
        self.max_novelty_floor = max_novelty_floor
        self._steps = 0

    @property
    def steps(self) -> int:
        """How many escalations have been applied since the last reset."""
        return self._steps

    def plan(self, reading: StagnationReading, self_model: SelfModel) -> Escalation | None:
        """Plan an escalation, or ``None`` when the mind is not stagnating."""
        if not reading.stagnating:
            self.reset()
            return None
        self._steps += 1
        # Each stagnation steps the escalation up, bounded by the ceilings.
        widen_concepts = min(self.max_widen_concepts, self._steps)
        widen_ideas = min(self.max_widen_ideas, self._steps)
        novelty_floor = _clamp(min(self.max_novelty_floor, 0.3 + 0.1 * self._steps))
        yields = self_model.yields()
        focus = yields[0].name if yields else ""
        return Escalation(
            widen_concepts=widen_concepts,
            widen_ideas=widen_ideas,
            novelty_floor=novelty_floor,
            strategy_focus=focus,
            reason=reading.reason,
        )

    def reset(self) -> None:
        """Forget the escalation ladder (called when the mind is progressing)."""
        self._steps = 0


# ---------------------------------------------------------------------------
# The controller
# ---------------------------------------------------------------------------


@dataclass
class CyclePlan:
    """The metacognitive adjustments the mind applies to its next cycle."""

    concept_budget: int
    idea_budget: int
    novelty_floor: float
    strategy_focus: str = ""
    strategy_bias: dict[str, float] = field(default_factory=dict)
    escalation: Escalation | None = None
    stagnation: StagnationReading | None = None
    focus_topics: tuple[str, ...] = ()

    def describe(self) -> str:
        lines = [
            f"plan: {self.concept_budget} concept(s), {self.idea_budget} idea(s), "
            f"novelty floor {self.novelty_floor:.2f}"
        ]
        if self.strategy_focus:
            lines.append(f"focus strategy: {self.strategy_focus}")
        if self.escalation is not None:
            lines.append(self.escalation.describe())
        if self.stagnation is not None:
            lines.append(self.stagnation.describe())
        if self.focus_topics:
            lines.append("in focus: " + ", ".join(self.focus_topics))
        return "\n".join(lines)


class MetacognitiveController:
    """The mind's self-awareness: plan a cycle, then learn from how it went.

    One object holds the four parts together and is the only thing the
    :class:`~cognition.proactive_mind.ProactiveMind` talks to. It plans the next
    cycle *before* it runs (strategy bias, an escalation when stagnating) and
    records what actually happened *after* it runs (which strategies paid off,
    what came into focus), so the mind's next cycle is informed by its last.
    """

    def __init__(
        self,
        *,
        self_model: SelfModel | None = None,
        novelty: PersistentNovelty | None = None,
        focus: FocusBuffer | None = None,
        detector: StagnationDetector | None = None,
        policy: EscalationPolicy | None = None,
        base_concepts: int = 6,
        base_ideas: int = 3,
    ) -> None:
        self.self_model = self_model or SelfModel.load()
        self.novelty = novelty or PersistentNovelty.load()
        self.focus = focus or FocusBuffer.load()
        self.detector = detector or StagnationDetector()
        self.policy = policy or EscalationPolicy()
        self.base_concepts = base_concepts
        self.base_ideas = base_ideas

    # -- planning -----------------------------------------------------------
    def plan(
        self,
        *,
        cycles: tuple[ThoughtCycle, ...] = (),
        ideas_by_id: dict[str, Idea] | None = None,
        idea_limit: int | None = None,
    ) -> CyclePlan:
        """Decide how the next cycle should search, from the mind's own history."""
        reading = self.detector.read(cycles, ideas_by_id or {})
        escalation = self.policy.plan(reading, self.self_model)
        idea_budget = idea_limit if idea_limit is not None else self.base_ideas
        concept_budget = self.base_concepts
        novelty_floor = 0.0
        focus = ""
        if escalation is not None:
            idea_budget = idea_budget + escalation.widen_ideas
            concept_budget = concept_budget + escalation.widen_concepts
            novelty_floor = escalation.novelty_floor
            focus = escalation.strategy_focus
        bias = {y.name: y.bias for y in self.self_model.yields()}
        return CyclePlan(
            concept_budget=concept_budget,
            idea_budget=idea_budget,
            novelty_floor=novelty_floor,
            strategy_focus=focus,
            strategy_bias=bias,
            escalation=escalation,
            stagnation=reading,
            focus_topics=self.focus.topics(),
        )

    # -- learning from the cycle -------------------------------------------
    def observe_cycle(self, *, produced: tuple[tuple[str, Idea, bool], ...], topics: tuple[str, ...]) -> int:
        """Fold a finished cycle's outcome into the self-model and the focus.

        ``produced`` is ``(engine, idea, kept)`` per idea the cycle produced, where
        ``kept`` is whether the idea survived self-criticism. Returns the mind's
        new cycle number.
        """
        number = self.self_model.advance_cycle()
        for engine, idea, kept in produced:
            self.self_model.observe(engine, idea, kept=kept)
        self.focus.decay_all()
        for topic in topics[:3]:
            self.focus.reinforce(topic)
        self.self_model.save()
        self.novelty.save()
        self.focus.save()
        return number

    # -- reading ------------------------------------------------------------
    def status(self) -> dict[str, object]:
        """A small, honest snapshot for the UI and the CLI."""
        yields = self.self_model.yields()
        return {
            "cycle": self.self_model.cycle,
            "strategies": [
                {
                    "name": y.name,
                    "produced": y.produced,
                    "kept": y.kept,
                    "promoted": y.promoted,
                    "keep_rate": y.keep_rate,
                    "mean_novelty": y.mean_novelty,
                    "bias": y.bias,
                }
                for y in yields
            ],
            "escalation_steps": self.policy.steps,
            "focus": list(self.focus.topics()),
            "novelty_items": self.novelty.count(),
        }

    def describe(self) -> str:
        return self.self_model.describe()


__all__ = [
    "COLD_AFTER_CYCLES",
    "EXPLORE_WEIGHT",
    "CyclePlan",
    "Escalation",
    "EscalationPolicy",
    "FocusBuffer",
    "FocusEntry",
    "MetacognitiveController",
    "PersistentNovelty",
    "SelfModel",
    "StagnationDetector",
    "StagnationReading",
    "StrategyRecord",
    "StrategyYield",
]
