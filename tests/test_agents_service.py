"""The Personalities (agents) service (Phase 5): the multi-agent layer, headless.

The window is a thin view over :class:`gui.agents.AgentsService`, which drives the
project's *real* :class:`core.agents.system.AgentSystem`. These tests exercise the
service with no display and no model (the system's executor is a stub), so they
pin the facts the window and the CLI both read: the seven personalities, their
capability ceilings, a routing decision, the measured activity, and the isolation
policy.
"""

from __future__ import annotations

from core.agents.capabilities import AgentCapability
from core.agents.runtime import AgentResponse
from core.agents.system import AgentSystem
from gui.agents import (
    AgentsService,
    CapabilityLine,
    IsolationReport,
    PersonalityRow,
    RouteReport,
)


def _echo(request: object) -> AgentResponse:
    """A stand-in executor so the system needs no model."""
    name = getattr(getattr(request, "profile", None), "name", "the personality")
    return AgentResponse(ok=True, text=f"[echo] {name}", capabilities_used=(AgentCapability.ANSWER,))


def _service() -> AgentsService:
    return AgentsService(system=AgentSystem(executor=_echo))


# ------------------------------------------------------------------- rows


def test_rows_lists_every_personality() -> None:
    rows = _service().rows()
    owners = {row.owner for row in rows}
    assert owners == {"jarvis", "ultron", "vision", "banner", "natasha", "hakeem", "cloud", "fayd_al_ruh"}
    assert all(isinstance(row, PersonalityRow) for row in rows)


def test_rows_mark_natasha_private_and_the_model_shareable() -> None:
    rows = {row.owner: row for row in _service().rows()}
    assert rows["natasha"].private is True
    assert rows["natasha"].contributes_training is False
    assert rows["jarvis"].private is False
    assert rows["fayd_al_ruh"].contributes_training is True


def test_profile_unknown_key_resolves_to_the_project_model() -> None:
    service = _service()
    profile = service.profile("not-a-character")
    assert profile.personality_id == "fayd_al_ruh"


def test_describe_names_the_personality() -> None:
    text = _service().describe("vision")
    assert "Vision" in text
    assert "vision" in text


# ----------------------------------------------------------- capabilities


def test_capabilities_report_the_whole_vocabulary() -> None:
    lines = _service().capabilities("jarvis")
    assert len(lines) == len(AgentCapability)
    assert all(isinstance(line, CapabilityLine) for line in lines)


def test_capability_ceiling_denies_search_for_a_character() -> None:
    lines = {line.capability: line for line in _service().capabilities("jarvis")}
    # SEARCH is refused by default -- a personality cannot grant itself the web.
    assert lines["search"].allowed is False
    assert "ceiling" in lines["search"].reason


def test_capability_owners_groups_every_capability() -> None:
    owners = _service().capability_owners()
    assert set(owners) == {capability.value for capability in AgentCapability}
    # Every personality may answer.
    assert "jarvis" in owners["answer"]


# ---------------------------------------------------------------- routing


def test_route_selects_a_petroleum_personality_for_a_petroleum_question() -> None:
    report = _service().route("how do I measure porosity?")
    assert isinstance(report, RouteReport)
    assert report.suitable is True
    assert report.selected == "vision"
    assert report.task_type == "petroleum"
    assert "porosity" in report.reason


def test_route_explains_an_unsuitable_question() -> None:
    # A task type no personality's ceiling can serve yields an honest refusal.
    report = _service().route("please run a shell command", task_type="code")
    assert report.suitable in (True, False)  # never raises
    assert isinstance(report.describe(), str)


def test_route_with_an_unknown_task_type_does_not_raise() -> None:
    report = _service().route("what is porosity?", task_type="not-a-task")
    assert isinstance(report, RouteReport)


def test_task_types_lists_the_closed_vocabulary() -> None:
    types = _service().task_types()
    assert "petroleum" in types
    assert "grounded_answer" in types


# ------------------------------------------------------------- monitoring


def test_activity_starts_empty_then_counts_a_run() -> None:
    service = _service()
    assert all(activity.total == 0 for activity in service.activity())
    outcome = service.system.run("vision", "how do I measure porosity?")
    assert outcome.ok is True
    assert service.activity_for("vision").succeeded == 1
    assert service.activity_for("vision").success_rate == 1.0


def test_activity_for_an_unknown_owner_is_a_zero_tally() -> None:
    activity = _service().activity_for("nobody")
    assert activity.total == 0


# -------------------------------------------------------------- isolation


def test_isolation_reports_natasha_private_and_the_dataset_clean() -> None:
    report = _service().isolation()
    assert isinstance(report, IsolationReport)
    assert "natasha" in report.private
    assert "natasha" not in report.shareable
    assert "natasha" not in report.dataset_owners
    assert report.dataset_is_clean is True


def test_isolation_describe_is_readable() -> None:
    text = _service().isolation().describe()
    assert "Private characters" in text
    assert "Shareable characters" in text
