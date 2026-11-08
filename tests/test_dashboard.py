"""The Home / Dashboard model (Phase 2).

The dashboard is the main window's front page: what is answering, the session,
the recent conversations, the knowledge/learning activity, and the machine's own
figures -- plus quick actions. These tests exercise the Tk-free model, so they
need no display; the window's own wiring is pinned in ``test_sidebar.py``.
"""

from __future__ import annotations

from datetime import datetime

from gui.dashboard import (
    QUICK_ACTION_KEYS,
    Dashboard,
    DashboardRow,
    DashboardSection,
    QuickAction,
    build_dashboard,
    dashboard_for,
    knowledge_base_summary,
)


class _Service:
    """A duck-typed stand-in for :class:`gui.service.ChatService`."""

    source = "ultron"
    session = "abc123"

    class _Mode:
        label = "Fayd Al-Ruh Model"

    operating_mode = _Mode()

    def describe_mouth_model(self, mouth: str) -> str:
        return f"{mouth}: qwen2.5-7b (local)"


class _Bare:
    """A service missing every optional member -- the model must not raise."""


class _Item:
    def __init__(self, label: str, when: datetime | None) -> None:
        self.label = label
        self.created_at = when
        self.session_id = "s1"


# --------------------------------------------------------------- the shape


def test_build_dashboard_includes_the_session_section() -> None:
    dashboard = build_dashboard(
        mouth_label="Ultron (General)",
        model_description="qwen2.5-7b",
        session="abc",
        mode_label="Fayd Al-Ruh Model",
    )
    assert isinstance(dashboard, Dashboard)
    assert dashboard.sections
    session = dashboard.sections[0]
    assert session.key == "home:session"
    values = {row.label: row.value for row in session.rows}
    assert values["Model"] == "Ultron (General)"
    assert values["Session"] == "abc"
    assert values["Mode"] == "Fayd Al-Ruh Model"


def test_an_absent_optional_value_is_named_unknown_never_faked() -> None:
    dashboard = build_dashboard(mouth_label="", session="", mode_label="")
    rows = {row.label: row.value for row in dashboard.sections[0].rows}
    assert rows["Model"] == "unknown"
    assert rows["Session"] == "unsaved"
    assert rows["Mode"] == "unknown"


def test_sections_keep_a_fixed_order() -> None:
    dashboard = build_dashboard(
        mouth_label="M",
        session="s",
        mode_label="mode",
        recent_chats=(_Item("A chat", datetime(2026, 1, 2)),),
        knowledge_summary="12 chunks",
        resources=type(
            "R",
            (),
            {
                "system": "Linux",
                "machine": "x86_64",
                "cpu_count": 8,
                "total_memory_bytes": 16 * 1024**3,
                "available_memory_bytes": 4 * 1024**3,
                "disk_free_bytes": 100 * 1024**3,
                "disk_path": "/data",
            },
        )(),
    )
    keys = [section.key for section in dashboard.sections]
    assert keys == ["home:session", "home:recent", "home:knowledge", "home:resources"]


# ------------------------------------------------------------- the sections


def test_an_empty_recent_list_draws_no_section() -> None:
    dashboard = build_dashboard(mouth_label="M", session="s", mode_label="m")
    assert all(section.key != "home:recent" for section in dashboard.sections)


def test_recent_conversations_carry_their_date() -> None:
    dashboard = build_dashboard(
        mouth_label="M",
        session="s",
        mode_label="m",
        recent_chats=(_Item("Yesterday", datetime(2026, 3, 4)),),
    )
    recent = next(s for s in dashboard.sections if s.key == "home:recent")
    assert recent.rows[0].label == "Yesterday"
    assert recent.rows[0].value == "2026-03-04"


def test_a_recent_item_without_a_date_reads_saved() -> None:
    dashboard = build_dashboard(
        mouth_label="M", session="s", mode_label="m", recent_chats=(_Item("No date", None),)
    )
    recent = next(s for s in dashboard.sections if s.key == "home:recent")
    assert recent.rows[0].value == "saved"


def test_knowledge_section_needs_at_least_one_figure() -> None:
    dashboard = build_dashboard(mouth_label="M", session="s", mode_label="m")
    assert all(section.key != "home:knowledge" for section in dashboard.sections)
    dashboard = build_dashboard(
        mouth_label="M", session="s", mode_label="m", knowledge_summary="1.2 MB"
    )
    knowledge = next(s for s in dashboard.sections if s.key == "home:knowledge")
    assert knowledge.rows[0].value == "1.2 MB"


def test_no_resources_draws_no_machine_section() -> None:
    dashboard = build_dashboard(mouth_label="M", session="s", mode_label="m", resources=None)
    assert all(section.key != "home:resources" for section in dashboard.sections)


def test_resource_figures_are_readable_and_honest() -> None:
    resources = type(
        "R",
        (),
        {
            "system": "Windows",
            "machine": "AMD64",
            "cpu_count": 12,
            "total_memory_bytes": 32 * 1024**3,
            "available_memory_bytes": None,
            "disk_free_bytes": None,
            "disk_path": "C:/data",
        },
    )()
    dashboard = build_dashboard(
        mouth_label="M", session="s", mode_label="m", resources=resources
    )
    machine = next(s for s in dashboard.sections if s.key == "home:resources")
    rows = {row.label: row for row in machine.rows}
    assert rows["System"].value == "Windows (AMD64)"
    assert rows["CPU cores"].value == "12"
    assert "32.0 GB" in rows["Memory"].value
    assert rows["Memory"].detail == "unknown available"  # an honest unknown, not 0
    assert rows["Disk free"].value == "unknown"


# ------------------------------------------------------------- the service


def test_dashboard_for_reads_the_live_service() -> None:
    dashboard = dashboard_for(_Service(), mouth_label="Ultron (General)")
    rows = {row.label: row.value for row in dashboard.sections[0].rows}
    assert rows["Model"] == "Ultron (General)"
    assert rows["Session"] == "abc123"
    assert rows["Mode"] == "Fayd Al-Ruh Model"
    # The description comes from the service's own describe_mouth_model.
    model_row = next(r for r in dashboard.sections[0].rows if r.label == "Model")
    assert "qwen2.5-7b" in model_row.detail


def test_dashboard_for_tolerates_a_service_missing_every_member() -> None:
    dashboard = dashboard_for(_Bare())
    assert dashboard.sections  # the session section is always drawn
    rows = {row.label: row.value for row in dashboard.sections[0].rows}
    assert rows["Model"] == "unknown"
    assert rows["Session"] == "unsaved"


def test_quick_actions_are_carried_through() -> None:
    actions = (QuickAction("insight", "Ask the AI"),)
    dashboard = build_dashboard(
        mouth_label="M", session="s", mode_label="m", quick_actions=actions
    )
    assert dashboard.quick_actions == actions


def test_quick_action_keys_resolve_in_the_real_catalogue() -> None:
    from gui.actions import action_by_key

    for key in QUICK_ACTION_KEYS:
        assert action_by_key(key) is not None


# ---------------------------------------------------- the knowledge summary


def test_knowledge_base_summary_is_empty_for_a_missing_store(tmp_path) -> None:
    assert knowledge_base_summary(tmp_path / "nope") == ""


def test_dashboard_types_are_plain_data() -> None:
    row = DashboardRow("Label", "value", "detail")
    section = DashboardSection("k", "T", "i", (row,))
    dashboard = Dashboard(sections=(section,), quick_actions=())
    assert section.rows == (row,)
    assert dashboard.sections == (section,)
