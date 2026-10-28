"""Surface the *timeline* for a question that asks what changed over time.

The project carries a whole temporal layer -- ``TemporalReasoner``, ``Timeline``,
``TemporalEvent``, ``TemporalPattern``, ``TimePrediction`` -- but it was reachable
only from tests, never from an answer. So a question like *"what changed in this
session?"* got a definition back, never the ordered account of what actually
happened. That is the temporal half of Gap 7 of ``docs/REASONING_ASSESSMENT.md``.

This module reads the one real timeline the answer path already holds -- the
conversation's own **timestamped turns** -- turns them into ``TemporalEvent``s, and
runs them through the project's real ``TemporalReasoner``: the ordering is the
``Timeline``'s own, the relation between two turns is the reasoner's own
(``before``/``during``/``simultaneous``...), and any repeating pattern and its
next-time prediction are the ones ``find_pattern``/``predict_next`` actually
produced. It is **read-only** and **honest by construction**: it only builds a
timeline when the question actually *asks about change over time* (a definitional
"what is porosity?" gets none), it never invents an event the conversation did not
contain, it attributes every entry to the turn it came from, and it returns
``None`` rather than a guess when there is no timeline to read (fewer than two
turns, or no conversation at all).

The account is a *reported* artifact: nothing here changes the conversation or the
knowledge graph.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import TYPE_CHECKING

from core.temporal_event import TemporalEvent
from core.temporal_pattern import TemporalPattern
from core.temporal_reasoner import TemporalReasoner
from core.time_prediction import TimePrediction
from core.timeline import Timeline
from core.voices import detect_language

if TYPE_CHECKING:
    from core.conversation import Conversation

#: The phrases that make a question *temporal* -- a request for what changed over
#: time, in either language. A timeline is only produced for such a question, so a
#: definitional question never drags an account of the session into its answer. The
#: cues are deliberately *unambiguous*: a bare "changed" (which matches the causal
#: "what changed the permeability?") and the Arabic "اتجاه" (which means the flow
#: *direction* in petroleum) are **not** cues, because a false positive would put a
#: conversation timeline into an answer that asked for something else.
TEMPORAL_CUES: tuple[str, ...] = (
    "changed over",
    "change over",
    "changes over",
    "changed in this",
    "what changed over",
    "what has changed over",
    "how has it changed",
    "how has this changed",
    "what is the trend",
    "the trend",
    "over the session",
    "in this session",
    "this session",
    "this conversation",
    "so far in",
    "timeline",
    "تغيّر خلال",
    "تغير خلال",
    "تغيّر في هذه",
    "تغير في هذه",
    "ما الذي تغيّر في",
    "ما الذي تغير في",
    "على مر",
    "هذه الجلسة",
    "هذه المحادثة",
    "خلال الجلسة",
    "خلال المحادثة",
    "سجل المحادثة",
)

#: How many characters of a turn are shown in the timeline, so a long turn reads
#: as a topic rather than filling the account.
_MAX_ENTRY_CHARS = 60

#: The label for a user turn and an assistant turn in the timeline.
_ROLE_LABEL = {"user": "Q", "assistant": "A"}


@dataclass(frozen=True)
class TimelineEntry:
    """One turn of the conversation as a point on the timeline."""

    role: str
    """``"user"`` or ``"assistant"`` -- who produced the turn."""

    text: str
    """The turn's own text, trimmed to a readable length."""

    at: datetime
    """When the turn was made (its real ``created_at``)."""


@dataclass(frozen=True)
class AnswerTemporal:
    """The ordered account of what changed over the conversation."""

    entries: tuple[TimelineEntry, ...]
    """Every turn, in the order the timeline put it."""

    relation: str
    """The temporal relation between the last two turns, as the reasoner named it
    (``before``/``after``/``during``/``contains``/``overlaps``/``simultaneous``)."""

    change: str
    """A readable statement of the most recent change: how the latest turn relates
    to the one before it."""

    pattern: TemporalPattern | None
    """The repeating pattern the reasoner found, or ``None`` when nothing repeats
    -- an honest "no pattern in this conversation" rather than a fabricated one."""

    prediction: TimePrediction | None
    """When the reasoner expects the repeating pattern to recur, or ``None`` when
    there is no pattern to extrapolate from."""

    language: str
    """``"ar"`` or ``"en"`` -- the language the account was written in."""

    @property
    def span_seconds(self) -> float:
        """The wall-clock span the timeline covers, in seconds."""
        if len(self.entries) < 2:
            return 0.0
        return (self.entries[-1].at - self.entries[0].at).total_seconds()

    def describe(self) -> str:
        """The account as a heading followed by the timeline and the change."""
        if self.language == "ar":
            head = f"ما تغيّر خلال هذه المحادثة (امتداد {self.span_seconds:.0f} ثانية):"
            change = f"آخر تغيّر: {self.change} (العلاقة الزمنية: {self.relation})."
        else:
            head = f"What changed over this conversation (span {self.span_seconds:.0f} s):"
            change = f"Latest change: {self.change} (temporal relation: {self.relation})."
        lines = [head]
        lines.extend(f"  - {entry.role}: {entry.text}" for entry in self.entries)
        lines.append(f"  {change}")
        if self.pattern is not None:
            if self.language == "ar":
                lines.append(
                    f"  نمط متكرر: «{self.pattern.description}» تكرّر "
                    f"{self.pattern.occurrences} مرات (اتجاه {self.pattern.direction})."
                )
            else:
                lines.append(
                    f"  Repeating pattern: \"{self.pattern.description}\" occurred "
                    f"{self.pattern.occurrences} times (direction {self.pattern.direction})."
                )
        if self.prediction is not None:
            if self.language == "ar":
                lines.append(f"  التوقع التالي: {self.prediction.predicted_time.isoformat()}.")
            else:
                lines.append(f"  Next expected: {self.prediction.predicted_time.isoformat()}.")
        return "\n".join(lines)


def is_temporal(question: str) -> bool:
    """Whether a question asks what changed over time, so a timeline is warranted.

    Whole-phrase and case-insensitive: the cue must appear in the question, so a
    definitional question ("what is porosity?") is never read as temporal.
    """
    lowered = question.casefold()
    return any(cue in lowered for cue in TEMPORAL_CUES)


def _trim(text: str) -> str:
    """A turn's text as a readable topic, with a visible marker when cut."""
    collapsed = " ".join(text.split())
    if len(collapsed) <= _MAX_ENTRY_CHARS:
        return collapsed
    return collapsed[:_MAX_ENTRY_CHARS].rstrip() + "..."


def _events_from_conversation(conversation: Conversation) -> list[TemporalEvent]:
    """The conversation's own turns as timestamped events, oldest first.

    Only the exchange is read -- ``SYSTEM`` instructions and ``CONTEXT`` reference
    material are skipped, because they are not something that happened in the
    conversation. Each event keeps its turn's real ``created_at``, so the timeline
    is the actual wall-clock order, not an invented one.
    """
    from core.conversation import Role

    events: list[TemporalEvent] = []
    for message in conversation.messages:
        if message.role not in (Role.USER, Role.ASSISTANT):
            continue
        label = _ROLE_LABEL.get(message.role.value, message.role.value)
        events.append(
            TemporalEvent(
                description=f"{label}: {_trim(message.text)}",
                start=message.created_at,
            )
        )
    return events


def _ordered_events(events: list[TemporalEvent]) -> list[TemporalEvent]:
    """The events in the order the project's own ``Timeline`` puts them.

    A fresh id is assigned so the timeline never refuses a caller's duplicate id;
    the ordering, not the id, is what this bridge reports.
    """
    timeline = Timeline()
    for index, event in enumerate(events):
        timeline.add(
            TemporalEvent(
                description=event.description,
                start=event.start,
                end=event.end,
                value=event.value,
                id=f"T{index}",
            )
        )
    return timeline.ordered()


def _entries_from_events(ordered: list[TemporalEvent]) -> tuple[TimelineEntry, ...]:
    """The events as timeline entries, in the order the ``Timeline`` put them."""
    return tuple(
        TimelineEntry(
            role=event.description.split(":", 1)[0].strip(),
            text=event.description.split(":", 1)[1].strip() if ":" in event.description else event.description,
            at=event.start,
        )
        for event in ordered
    )


def _repeating(events: list[TemporalEvent], reasoner: TemporalReasoner):
    """The pattern the reasoner finds for the most-repeated description, if any.

    A description that occurs once is not a pattern, so this returns ``(None,
    None)`` when nothing in the timeline repeats -- the honest answer for a
    conversation of distinct questions.
    """
    counts: dict[str, int] = {}
    for event in events:
        counts[event.description] = counts.get(event.description, 0) + 1
    repeated = [description for description, count in counts.items() if count >= 2]
    if not repeated:
        return None, None
    # Deterministic: the most frequent, ties broken by the description itself.
    description = sorted(repeated, key=lambda item: (-counts[item], item))[0]
    pattern = reasoner.find_pattern(events, description)
    prediction = reasoner.predict_next(events, description)
    return pattern, prediction


def _change_statement(ordered: list[TemporalEvent], language: str) -> str:
    """How the latest turn relates to the one before it, in the question's language."""
    last, previous = ordered[-1], ordered[-2]
    if language == "ar":
        return f"انتقل الحديث من «{previous.description}» إلى «{last.description}»"
    return f"the discussion moved from \"{previous.description}\" to \"{last.description}\""


def answer_temporal(
    question: str,
    *,
    conversation: Conversation | None = None,
    events: list[TemporalEvent] | None = None,
    language: str | None = None,
) -> AnswerTemporal | None:
    """The timeline for a question, or ``None`` when no timeline is warranted.

    A timeline is produced only when the question is *temporal* (``is_temporal``)
    **and** there are at least two events to order. Otherwise ``None`` -- a
    definitional or unrelated question gets no timeline rather than a guess.
    """
    if not is_temporal(question):
        return None
    if events is None:
        if conversation is None:
            return None
        events = _events_from_conversation(conversation)
    if len(events) < 2:
        return None
    lang = language or detect_language(question)
    reasoner = TemporalReasoner()
    ordered = _ordered_events(events)
    entries = _entries_from_events(ordered)
    relation = reasoner.relate(ordered[-2], ordered[-1])
    pattern, prediction = _repeating(ordered, reasoner)
    return AnswerTemporal(
        entries=entries,
        relation=relation.value,
        change=_change_statement(ordered, lang),
        pattern=pattern,
        prediction=prediction,
        language=lang,
    )


@dataclass(frozen=True)
class TemporalJudgement:
    """One held-out expectation about the temporal bridge.

    ``kind`` selects what is checked: ``"order"`` that a set of events comes back
    in chronological order with the expected relation, ``"pattern"`` that a
    repeating description is found with the expected occurrence count, and
    ``"no-timeline"`` that a question warrants no timeline at all -- the negative
    control that proves nothing is invented for a definitional question or a
    conversation too short to have a timeline.
    """

    name: str
    kind: str
    question: str = ""
    offsets: tuple[int, ...] = ()
    descriptions: tuple[str, ...] = ()
    expected_relation: str = ""
    expected_occurrences: int = 0

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("A temporal judgement needs a name.")
        if self.kind not in ("order", "pattern", "no-timeline"):
            raise ValueError(f"Unknown temporal judgement kind: {self.kind}")


def _judgement_events(judgement: TemporalJudgement) -> list[TemporalEvent]:
    """Build timestamped events from the judgement's offsets and descriptions."""
    from datetime import UTC, datetime, timedelta

    base = datetime(2026, 1, 1, tzinfo=UTC)
    descriptions = judgement.descriptions or tuple(
        f"Q: question {index}" for index in range(len(judgement.offsets))
    )
    return [
        TemporalEvent(description=description, start=base + timedelta(seconds=offset))
        for offset, description in zip(judgement.offsets, descriptions, strict=False)
    ]


#: The held-out temporal catalogue -- hand-written with **negative controls**: a
#: definitional or unrelated question must yield no timeline, and a conversation
#: with fewer than two turns must yield none either.
TEMPORAL_JUDGEMENTS: tuple[TemporalJudgement, ...] = (
    TemporalJudgement(
        "events given out of order are ordered chronologically",
        "order",
        question="what changed over the session?",
        offsets=(30, 0, 60),
        descriptions=("Q: third", "Q: first", "Q: second"),
        expected_relation="before",
    ),
    TemporalJudgement(
        "a repeating description is found as a pattern",
        "pattern",
        question="what is the trend?",
        offsets=(0, 60, 120, 180),
        descriptions=("Q: rate", "Q: rate", "Q: rate", "Q: rate"),
        expected_occurrences=4,
    ),
    # Negative controls: no timeline is invented for a question that asks none.
    TemporalJudgement("a definitional question gets no timeline", "no-timeline", question="what is porosity?"),
    TemporalJudgement(
        "an unrelated question gets no timeline", "no-timeline", question="what is the capital of France?"
    ),
    TemporalJudgement(
        "a temporal question with one event gets no timeline",
        "no-timeline",
        question="what changed over the session?",
        offsets=(0,),
        descriptions=("Q: only",),
    ),
)


def temporal_accuracy(
    judgements: tuple[TemporalJudgement, ...] = TEMPORAL_JUDGEMENTS,
) -> float:
    """The share of held-out temporal judgements the bridge gets right.

    A judgement counts only when the bridge does exactly what it must: order a
    timeline chronologically and name the right relation, find the repeating
    pattern with its real occurrence count, or produce **no** timeline for a
    question that warrants none. So a bridge that invents a timeline for a
    definitional question, misorders the events, or fabricates a pattern fails the
    relevant judgement. Returns ``0.0`` for an empty catalogue.
    """
    catalogue = tuple(judgements)
    if not catalogue:
        return 0.0
    correct = 0
    for judgement in catalogue:
        if judgement.kind == "no-timeline":
            events = _judgement_events(judgement)
            got = answer_temporal(judgement.question, events=events)
            ok = got is None
        else:
            events = _judgement_events(judgement)
            got = answer_temporal(judgement.question, events=events)
            if got is None:
                ok = False
            elif judgement.kind == "order":
                ordered_at = [entry.at for entry in got.entries]
                ok = ordered_at == sorted(ordered_at) and got.relation == judgement.expected_relation
            else:  # pattern
                ok = got.pattern is not None and got.pattern.occurrences == judgement.expected_occurrences
        if ok:
            correct += 1
    return correct / len(catalogue)


__all__ = [
    "TEMPORAL_CUES",
    "TEMPORAL_JUDGEMENTS",
    "AnswerTemporal",
    "TemporalJudgement",
    "TimelineEntry",
    "answer_temporal",
    "is_temporal",
    "temporal_accuracy",
]
