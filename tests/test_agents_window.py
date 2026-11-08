"""The Personalities (agents) window: the seven characters as agents.

The window parts are skipped unless a display (or Xvfb) is available, exactly
like ``tests/test_gui.py``. The engine the window drives is exercised headless in
``tests/test_agents_service.py``; here the *view* is what is checked, so the window
is opened with a stand-in service and no real model is ever loaded.
"""

from __future__ import annotations

import time

import pytest

from tests.test_gui_design import _display_available


class StubService:
    """A stand-in service: fixed profiles, no model, no display of its own."""

    def __init__(self) -> None:
        self.routed: list[str] = []

    def rows(self):
        from gui.agents import PersonalityRow

        return (
            PersonalityRow(
                owner="jarvis", name="Jarvis", role="grounded assistant",
                model="(mouth default)", private=False, contributes_training=True,
                capabilities=("answer", "retrieve"),
            ),
            PersonalityRow(
                owner="natasha", name="Natasha", role="companion",
                model="qwen3:4b", private=True, contributes_training=False,
                capabilities=("answer",),
            ),
        )

    def profile(self, key: str):
        class _Profile:
            def __init__(self, name: str) -> None:
                self.name = name

            def describe(self) -> str:
                return f"{self.name} profile\n  role: something"

        return _Profile("Natasha" if key == "natasha" else "Jarvis")

    def capabilities(self, key: str):
        from gui.agents import CapabilityLine

        return (
            CapabilityLine(capability="answer", allowed=True, reason="granted within this personality's ceiling"),
            CapabilityLine(capability="search", allowed=False, reason="outside this personality's ceiling (denied by default)"),
        )

    def route(self, question: str, **kwargs):
        from gui.agents import RouteReport

        self.routed.append(question)
        return RouteReport(
            question=question, task_type="petroleum", selected="vision",
            suitable=True, manual=False, reason="the question mentions 'porosity'",
            candidates=(("vision", 1.4),),
        )

    def activity(self):
        from core.agents.monitoring import PersonalityActivity

        return (PersonalityActivity(owner="vision", total=2, succeeded=2),)

    def isolation(self):
        from gui.agents import IsolationReport

        return IsolationReport(
            private=("natasha",), shareable=("jarvis", "ultron", "vision"),
            dataset_owners=("fayd_al_ruh", "jarvis"), dataset_is_clean=True,
        )


def _dialog(root, service):
    from gui.agents_dialog import AgentsDialog
    from gui.app import DARK_THEME

    return AgentsDialog(  # type: ignore[arg-type]
        root, service, theme=DARK_THEME, font="DejaVu Sans", mono="DejaVu Sans Mono"
    )


def _drain(root, dialog, seconds: float = 10.0) -> None:
    end = time.time() + seconds
    while time.time() < end:
        root.update()
        if not dialog._busy:
            return
        time.sleep(0.02)


@pytest.mark.skipif(not _display_available(), reason="no display available")
def test_window_builds_with_every_sidebar_section() -> None:
    import tkinter as tk

    root = tk.Tk()
    try:
        dialog = _dialog(root, StubService())
        for key in ("people", "routing", "isolation", "about"):
            assert key in dialog._sidebar_column.sections
    finally:
        root.destroy()


@pytest.mark.skipif(not _display_available(), reason="no display available")
def test_the_people_list_names_every_personality() -> None:
    import tkinter as tk

    root = tk.Tk()
    try:
        dialog = _dialog(root, StubService())
        labels = [dialog._people_list.get(i) for i in range(dialog._people_list.size())]
        assert any("Jarvis" in label for label in labels)
        assert any("Natasha" in label for label in labels)
    finally:
        root.destroy()


@pytest.mark.skipif(not _display_available(), reason="no display available")
def test_picking_a_personality_shows_its_profile_and_capabilities() -> None:
    import tkinter as tk

    root = tk.Tk()
    try:
        dialog = _dialog(root, StubService())
        dialog._people_list.selection_clear(0, tk.END)
        dialog._people_list.selection_set(1)
        dialog._on_pick_personality()
        assert "Natasha" in dialog._profile_label.cget("text")
        # The capability block reports the denied one too.
        assert "✕ search" in dialog._capability_label.cget("text")
    finally:
        root.destroy()


@pytest.mark.skipif(not _display_available(), reason="no display available")
def test_routing_a_question_posts_the_decision() -> None:
    import tkinter as tk

    root = tk.Tk()
    try:
        service = StubService()
        dialog = _dialog(root, service)
        dialog._route_var.set("how do I measure porosity?")
        dialog._on_route()
        _drain(root, dialog)
        assert service.routed == ["how do I measure porosity?"]
        texts = [card.get("1.0", "end-1c") for card in dialog._cards]
        assert any("vision" in text for text in texts)
    finally:
        root.destroy()


@pytest.mark.skipif(not _display_available(), reason="no display available")
def test_the_isolation_policy_is_shown() -> None:
    import tkinter as tk

    root = tk.Tk()
    try:
        dialog = _dialog(root, StubService())
        assert "natasha" in dialog._isolation_label.cget("text")
    finally:
        root.destroy()


@pytest.mark.skipif(not _display_available(), reason="no display available")
def test_closing_releases_the_pump_and_clears_the_callback() -> None:
    import tkinter as tk

    root = tk.Tk()
    closed: list[bool] = []
    try:
        service = StubService()
        from gui.agents_dialog import AgentsDialog
        from gui.app import DARK_THEME

        dialog = AgentsDialog(
            root, service, theme=DARK_THEME, font="DejaVu Sans", mono="DejaVu Sans Mono",  # type: ignore[arg-type]
            on_close=lambda: closed.append(True),
        )
        dialog._close()
        assert closed == [True]
        assert dialog._closed is True
    finally:
        root.destroy()


def test_the_agents_action_is_grouped_and_opens_the_window() -> None:
    """The action exists, is visible, and lives in the Personalities group."""
    from gui.actions import ACTIONS, action_group_by_key, visible_actions

    keys = {action.key for action in visible_actions()}
    assert "agents" in keys
    group = action_group_by_key("characters")
    assert "agents" in group.keys
    assert any(action.key == "agents" for action in ACTIONS)
