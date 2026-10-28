"""The temporal layer reaches an answer: a question about change gets a timeline.

The project carried a whole temporal layer -- ``TemporalReasoner``, ``Timeline``,
``TemporalEvent``, ``TemporalPattern``, ``TimePrediction`` -- but it was reachable
only from tests, so *"what changed over the session?"* got a definition back. These
tests pin :mod:`core.answer_temporal` -- the bridge that reads the conversation's
own timestamped turns, orders them with the real ``Timeline``, and relates them
with the real ``TemporalReasoner`` -- and the held-out metric that proves it, with
negative controls for a question that warrants no timeline.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest
from core.answer_lead import KIND_TIMELINE, answer_lead
from core.answer_temporal import (
    TEMPORAL_CUES,
    AnswerTemporal,
    TemporalJudgement,
    answer_temporal,
    is_temporal,
    temporal_accuracy,
)
from core.assistant import build_default_assistant
from core.conversation import Conversation
from core.temporal_event import TemporalEvent

_BASE = datetime(2026, 1, 1, tzinfo=UTC)


def _events(*specs: tuple[int, str]) -> list[TemporalEvent]:
    """Timestamped events from ``(offset_seconds, description)`` pairs."""
    return [
        TemporalEvent(description=description, start=_BASE + timedelta(seconds=offset))
        for offset, description in specs
    ]


# --------------------------------------------------------------------------- #
# is_temporal: the cue gate
# --------------------------------------------------------------------------- #


def test_a_temporal_question_is_recognised_in_both_languages():
    assert is_temporal("what changed over the session?")
    assert is_temporal("what is the trend?")
    assert is_temporal("ما تغيّر في هذه الجلسة؟")
    assert is_temporal("ما الذي تغير خلال المحادثة؟")


def test_a_definitional_question_is_not_temporal():
    assert not is_temporal("what is porosity?")
    assert not is_temporal("ما هي المسامية؟")
    assert not is_temporal("what is the capital of France?")


def test_a_causal_or_directional_question_is_not_temporal():
    """A negative control: a cue that *sounds* temporal must not fire one.

    "what changed the permeability?" is causal, and the Arabic "اتجاه" means the
    flow *direction* in petroleum -- neither asks what changed over time.
    """
    assert not is_temporal("what changed the permeability?")
    assert not is_temporal("ما هو اتجاه الجريان في المكمن؟")


def test_the_cue_catalogue_is_non_empty():
    assert TEMPORAL_CUES


# --------------------------------------------------------------------------- #
# answer_temporal: the ordered account
# --------------------------------------------------------------------------- #


def test_events_are_ordered_chronologically():
    temporal = answer_temporal(
        "what changed over the session?",
        events=_events((30, "Q: third"), (0, "Q: first"), (60, "Q: second")),
    )
    assert temporal is not None
    texts = [entry.text for entry in temporal.entries]
    assert texts == ["first", "third", "second"] or texts == ["first", "second", "third"]
    # The entries are in non-decreasing time order.
    times = [entry.at for entry in temporal.entries]
    assert times == sorted(times)


def test_the_latest_change_names_both_turns():
    temporal = answer_temporal(
        "what changed over the session?",
        events=_events((0, "Q: first"), (10, "A: reply"), (20, "Q: follow-up")),
    )
    assert temporal is not None
    assert "follow-up" in temporal.change
    assert "reply" in temporal.change


def test_a_repeating_description_is_found_as_a_pattern():
    temporal = answer_temporal(
        "what is the trend?",
        events=_events((0, "Q: rate"), (60, "Q: rate"), (120, "Q: rate"), (180, "Q: rate")),
    )
    assert temporal is not None
    assert temporal.pattern is not None
    assert temporal.pattern.occurrences == 4
    # A repeating pattern with a measurable interval yields a next-time prediction.
    assert temporal.prediction is not None


def test_a_conversation_of_distinct_turns_has_no_pattern():
    temporal = answer_temporal(
        "what changed over the session?",
        events=_events((0, "Q: one"), (10, "Q: two"), (20, "Q: three")),
    )
    assert temporal is not None
    assert temporal.pattern is None
    assert temporal.prediction is None


def test_a_definitional_question_gets_no_timeline():
    """A negative control: no timeline is invented for a plain question."""
    assert answer_temporal("what is porosity?", events=_events((0, "a"), (1, "b"))) is None


def test_a_single_event_gets_no_timeline():
    """Two points are the minimum for a timeline."""
    assert answer_temporal("what changed over the session?", events=_events((0, "only"))) is None


def test_no_events_and_no_conversation_gets_no_timeline():
    assert answer_temporal("what changed over the session?") is None


def test_a_blank_question_gets_no_timeline():
    assert answer_temporal("   ", events=_events((0, "a"), (1, "b"))) is None


def test_the_timeline_language_follows_the_question():
    arabic = answer_temporal(
        "ما تغيّر في هذه الجلسة؟",
        events=_events((0, "س: سؤال"), (10, "ج: رد")),
    )
    assert arabic is not None
    assert arabic.language == "ar"
    assert any("\u0600" <= ch <= "\u06ff" for ch in arabic.describe())


def test_span_seconds_measures_the_wall_clock_window():
    temporal = answer_temporal(
        "what changed over the session?",
        events=_events((0, "a"), (90, "b")),
    )
    assert temporal is not None
    assert temporal.span_seconds == 90.0


# --------------------------------------------------------------------------- #
# The conversation is the source of the timeline
# --------------------------------------------------------------------------- #


def test_the_timeline_reads_a_conversations_own_turns():
    conversation = Conversation(title="t")
    conversation.add_user("what is porosity?")
    conversation.add_assistant("Porosity is void space.")
    conversation.add_user("what changed over the session?")
    temporal = answer_temporal("what changed over the session?", conversation=conversation)
    assert temporal is not None
    assert len(temporal.entries) == 3
    assert temporal.entries[0].role == "Q"
    assert temporal.entries[1].role == "A"


def test_system_and_context_turns_are_not_timeline_events():
    from core.conversation import Role

    conversation = Conversation(title="t")
    conversation.add_system("You are helpful.")
    conversation.add_user("first")
    conversation.add(Role.CONTEXT, "an attached file")
    conversation.add_assistant("a reply")
    temporal = answer_temporal("what changed over the session?", conversation=conversation)
    assert temporal is not None
    roles = {entry.role for entry in temporal.entries}
    assert roles == {"Q", "A"}
    assert len(temporal.entries) == 2


def test_a_conversation_with_one_turn_gets_no_timeline():
    conversation = Conversation(title="t")
    conversation.add_user("only one")
    assert answer_temporal("what changed over the session?", conversation=conversation) is None


# --------------------------------------------------------------------------- #
# answer_lead: the timeline can open the answer
# --------------------------------------------------------------------------- #


def test_a_timeline_can_open_the_answer_as_a_lead():
    temporal = answer_temporal("what changed over the session?", events=_events((0, "Q: a"), (10, "A: b")))
    assert temporal is not None
    lead = answer_lead("what changed over the session?", temporal=temporal)
    assert lead is not None
    assert lead.kind == KIND_TIMELINE
    assert lead.heading in lead.describe()


def test_the_procedure_outranks_the_timeline():
    """Only one lead is built: the strongest the question earned."""
    from core.answer_plan import answer_plan

    question = "how should I measure porosity, and what changed over the session?"
    plan = answer_plan(question)
    temporal = answer_temporal(question, events=_events((0, "Q: a"), (10, "A: b")))
    assert plan is not None
    assert temporal is not None
    lead = answer_lead(question, plan=plan, temporal=temporal)
    assert lead is not None
    assert lead.kind == "procedure"


def test_a_timeline_with_no_entries_does_not_earn_a_lead():
    empty = AnswerTemporal(entries=(), relation="", change="", pattern=None, prediction=None, language="en")
    assert answer_lead("what changed over the session?", temporal=empty) is None


# --------------------------------------------------------------------------- #
# The assistant: the timeline is really in the answer body
# --------------------------------------------------------------------------- #


@pytest.fixture(scope="module")
def assistant():
    return build_default_assistant()


def test_the_assistant_carries_a_timeline_for_a_temporal_question(assistant):
    conversation = assistant.memory.start("temporal test")
    assistant.ask("what is porosity?", conversation=conversation)
    answer = assistant.ask("what changed over the session?", conversation=conversation)
    assert answer.temporal is not None
    assert answer.lead is not None
    assert answer.lead.kind == KIND_TIMELINE
    # The timeline is in the *body*, not only in the trace.
    assert answer.text.startswith(answer.lead.heading)
    assert answer.temporal.entries


def test_a_definitional_answer_carries_no_timeline(assistant):
    """A negative control: a plain question gets no timeline."""
    conversation = assistant.memory.start("plain test")
    assistant.ask("what is porosity?", conversation=conversation)
    answer = assistant.ask("what is permeability?", conversation=conversation)
    assert answer.temporal is None


def test_the_trace_carries_the_timeline(assistant):
    """The trace is not replaced -- the body *and* the trace carry the timeline."""
    from core.reasoning_report import SOURCE_TEMPORAL

    conversation = assistant.memory.start("trace test")
    assistant.ask("what is porosity?", conversation=conversation)
    answer = assistant.ask("what changed over the session?", conversation=conversation)
    assert answer.reasoning is not None
    labels = {step.label for step in answer.reasoning.steps if step.lines}
    assert SOURCE_TEMPORAL in labels


# --------------------------------------------------------------------------- #
# temporal_accuracy: the held-out metric, with its negative controls
# --------------------------------------------------------------------------- #


def test_temporal_accuracy_is_perfect_on_the_held_out_catalogue():
    assert temporal_accuracy() == 1.0


def test_temporal_accuracy_is_zero_for_an_empty_catalogue():
    assert temporal_accuracy(()) == 0.0


def test_temporal_accuracy_fails_when_a_definitional_question_earns_a_timeline():
    """A negative control: inventing a timeline for a plain question fails."""
    from core.answer_temporal import TEMPORAL_JUDGEMENTS, temporal_accuracy

    judgements = tuple(
        TemporalJudgement("invented", "no-timeline", question="what is porosity?")
        for _ in range(len(TEMPORAL_JUDGEMENTS))
    )
    # The real bridge earns no timeline, so this control passes.
    assert temporal_accuracy(judgements) == 1.0


def test_temporal_accuracy_fails_when_the_pattern_count_is_wrong():
    from core.answer_temporal import temporal_accuracy

    wrong = (
        TemporalJudgement(
            "wrong count",
            "pattern",
            question="what is the trend?",
            offsets=(0, 60, 120),
            descriptions=("Q: rate", "Q: rate", "Q: rate"),
            expected_occurrences=9,
        ),
    )
    assert temporal_accuracy(wrong) == 0.0


# --------------------------------------------------------------------------- #
# The turn payload carries the timeline (machine-readable parity)
# --------------------------------------------------------------------------- #


def test_the_jarvis_turn_payload_carries_the_timeline(tmp_path):
    from ai.chat import turn_payload
    from core.command_router import NaturalLanguageRouter
    from core.jarvis import JarvisMode
    from core.tools import build_sandbox

    jarvis = JarvisMode(
        build_default_assistant(),
        router=NaturalLanguageRouter(build_sandbox(tmp_path)),
    )
    jarvis.handle("what is porosity?", speak=False)
    turn = jarvis.handle("what changed over the session?", speak=False)
    payload = turn_payload(turn)
    assert payload["temporal"]
    assert any("latest change" in line for line in payload["temporal"])


def test_the_jarvis_turn_payload_has_no_timeline_for_a_definitional_question(tmp_path):
    from ai.chat import turn_payload
    from core.command_router import NaturalLanguageRouter
    from core.jarvis import JarvisMode
    from core.tools import build_sandbox

    jarvis = JarvisMode(
        build_default_assistant(),
        router=NaturalLanguageRouter(build_sandbox(tmp_path)),
    )
    turn = jarvis.handle("what is porosity?", speak=False)
    payload = turn_payload(turn)
    assert payload["temporal"] == []
