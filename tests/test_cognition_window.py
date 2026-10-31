"""The Autonomous Cognitive System window: sidebar, transcript, governed actions.

The window parts are skipped unless a display (or Xvfb) is available, exactly
like ``tests/test_gui.py``. The engine the window drives is exercised headless in
``tests/test_cognition_service.py``; here the *view* is what is checked, so the
window is opened with a stand-in service and no real mind, model, or training run
ever happens.
"""

from __future__ import annotations

import time

import pytest
from gui.character_evaluation import CharacterEvaluation, DimensionScore, EvaluationReport
from gui.cognition import OperationReport, ProgressionReport, StageRow

from tests.test_gui_design import _display_available


class StubService:
    """A stand-in service: fixed reports, no controller, no model, no training."""

    def __init__(self) -> None:
        self.mode_value = "balanced"
        self.learned: list[str] = []
        self.taught: list[str] = []
        self.trained: list[tuple[str, bool]] = []
        self.mode_set: list[str] = []

    # -- proactive mind ----------------------------------------------------
    def status(self):
        return {
            "enabled": True, "paused": False, "cycles": 3,
            "ideas": {"idea": 2}, "last_cycle": None,
            "interval_seconds": 300, "ideas_per_cycle": 3,
        }

    def think(self, max_ideas=None):
        return OperationReport(ok=True, title="Thinking cycle: 1 idea", lines=("drive: fill a gap",))

    def ideas(self):
        from gui.cognition import IdeaRow

        return (
            IdeaRow(text="An idea", kind="idea", engine="imagination", confidence=0.3, novelty=0.6),
        )

    def drives(self):
        return ("gap: porosity vs permeability",)

    def prove(self):
        return (OperationReport(ok=True, title="Thinking cycle"),)

    # -- learning ----------------------------------------------------------
    def learn(self, statement, **kwargs):
        self.learned.append(statement)
        return OperationReport(ok=True, title=f"Learned: {statement}", lines=("trusted fact",))

    def recall(self, query, limit=5):
        from gui.cognition import RecallReport

        return RecallReport(query=query, found={"semantic": ["a fact"]})

    def learning_stats(self):
        return "semantic 1, episodic 0"

    # -- training ----------------------------------------------------------
    def stages(self):
        from cognition.training import TrainingStage

        return tuple(TrainingStage)

    def train(self, *, stage, approved=False, progress=None):
        self.trained.append((stage.value, approved))
        return OperationReport(ok=True, title="smoke: accepted")

    def run_gradual(self, *, up_to=None, approved=False, progress=None):
        self.trained.append((getattr(up_to, "value", ""), approved))
        return OperationReport(ok=True, title="gradual complete")

    def progression(self):
        return ProgressionReport(
            rows=(StageRow(stage="smoke", accepted=1, attempts=1, done=True, description="smoke: done"),),
            next_stage="quick",
        )

    def versions_report(self):
        return "v1 (base)"

    def rollback(self):
        return OperationReport(ok=True, title="No accepted version to roll back to; the base model is in use.")

    # -- evaluation --------------------------------------------------------
    def evaluate(self):
        from cognition.evaluation import INDICATORS, CognitiveReport

        return CognitiveReport(scores=dict.fromkeys(INDICATORS, 0.5))

    def evaluate_characters(self):
        dims = tuple(
            DimensionScore(k, 0.5, (f"{k} fact",))
            for k in ("training", "memory", "traits", "development")
        )
        return EvaluationReport(
            characters=(
                CharacterEvaluation(
                    key="jarvis", name="Jarvis", role="cited assistant", private=False,
                    dimensions=dims, strengths=("grounded",), shortcomings=("memory empty",),
                    next_steps=("keep talking",),
                ),
            )
        )

    # -- system ------------------------------------------------------------
    def mode(self):
        from cognition.runner import ResourceMode

        return ResourceMode(self.mode_value)

    def set_mode(self, mode):
        self.mode_set.append(mode.value)
        return OperationReport(ok=True, title=f"Resource mode set to {mode.value}.")

    def tick(self):
        return OperationReport(ok=True, title="No cycle this tick: not due")

    def teach(self, question):
        self.taught.append(question)
        return OperationReport(ok=True, title=f"taught: {question}")


def _dialog(root, service):
    from gui.app import DARK_THEME
    from gui.cognition_dialog import CognitionDialog

    return CognitionDialog(
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
        for key in ("mind", "learning", "training", "characters", "indicators", "system"):
            assert key in dialog._sidebar_column.sections
    finally:
        root.destroy()


@pytest.mark.skipif(not _display_available(), reason="no display available")
def test_the_honesty_line_is_in_the_transcript() -> None:
    import tkinter as tk

    root = tk.Tk()
    try:
        dialog = _dialog(root, StubService())
        _drain(root, dialog)
        first = dialog._cards[0].get("1.0", "end")
        assert "never a fact" in first
    finally:
        root.destroy()


@pytest.mark.skipif(not _display_available(), reason="no display available")
def test_running_a_thinking_cycle_posts_a_card() -> None:
    import tkinter as tk

    root = tk.Tk()
    try:
        dialog = _dialog(root, StubService())
        _drain(root, dialog)
        before = len(dialog._cards)
        dialog._on_think()
        _drain(root, dialog)
        assert len(dialog._cards) > before
        assert "Thinking cycle" in dialog._cards[-1].get("1.0", "end")
    finally:
        root.destroy()


@pytest.mark.skipif(not _display_available(), reason="no display available")
def test_learning_a_statement_posts_a_card_and_clears_the_box() -> None:
    import tkinter as tk

    root = tk.Tk()
    try:
        service = StubService()
        dialog = _dialog(root, service)
        _drain(root, dialog)
        dialog._statement_var.set("porosity measures pore space")
        dialog._on_learn()
        _drain(root, dialog)
        assert service.learned == ["porosity measures pore space"]
        assert dialog._statement_var.get() == ""
    finally:
        root.destroy()


@pytest.mark.skipif(not _display_available(), reason="no display available")
def test_training_asks_for_confirmation_first() -> None:
    import tkinter as tk

    root = tk.Tk()
    try:
        service = StubService()
        dialog = _dialog(root, service)
        _drain(root, dialog)
        # A declined confirmation runs nothing.
        dialog._confirm = lambda *a, **k: False  # type: ignore[method-assign]
        dialog._on_train()
        _drain(root, dialog)
        assert service.trained == []
        # An accepted confirmation runs the chosen stage, approved.
        dialog._confirm = lambda *a, **k: True  # type: ignore[method-assign]
        dialog._on_train()
        _drain(root, dialog)
        assert service.trained == [("smoke", True)]
    finally:
        root.destroy()


@pytest.mark.skipif(not _display_available(), reason="no display available")
def test_gradual_training_is_governed_too() -> None:
    import tkinter as tk

    root = tk.Tk()
    try:
        service = StubService()
        dialog = _dialog(root, service)
        _drain(root, dialog)
        dialog._confirm = lambda *a, **k: True  # type: ignore[method-assign]
        dialog._on_train_gradual()
        _drain(root, dialog)
        assert service.trained == [("smoke", True)]
    finally:
        root.destroy()


@pytest.mark.skipif(not _display_available(), reason="no display available")
def test_evaluating_the_characters_lists_them() -> None:
    import tkinter as tk

    root = tk.Tk()
    try:
        dialog = _dialog(root, StubService())
        _drain(root, dialog)
        before = len(dialog._cards)
        dialog._on_characters()
        _drain(root, dialog)
        assert len(dialog._cards) > before
        body = "\n".join(c.get("1.0", "end") for c in dialog._cards[before:])
        assert "Jarvis" in body
        assert "Shortcomings" in body
        assert "Next steps" in body
    finally:
        root.destroy()


@pytest.mark.skipif(not _display_available(), reason="no display available")
def test_indicators_are_measured_and_shown() -> None:
    import tkinter as tk

    root = tk.Tk()
    try:
        dialog = _dialog(root, StubService())
        _drain(root, dialog)
        before = len(dialog._cards)
        dialog._on_evaluate()
        _drain(root, dialog)
        assert len(dialog._cards) > before
        assert "Knowledge retrieval" in dialog._cards[-1].get("1.0", "end")
    finally:
        root.destroy()


@pytest.mark.skipif(not _display_available(), reason="no display available")
def test_resource_mode_applies() -> None:
    import tkinter as tk

    root = tk.Tk()
    try:
        service = StubService()
        dialog = _dialog(root, service)
        _drain(root, dialog)
        dialog._mode_var.set("intensive")
        dialog._on_mode()
        _drain(root, dialog)
        assert service.mode_set == ["intensive"]
    finally:
        root.destroy()


@pytest.mark.skipif(not _display_available(), reason="no display available")
def test_the_command_line_teaches_the_student() -> None:
    import tkinter as tk

    root = tk.Tk()
    try:
        service = StubService()
        dialog = _dialog(root, service)
        _drain(root, dialog)
        dialog._command_var.set("what is permeability?")
        dialog._on_command()
        _drain(root, dialog)
        assert service.taught == ["what is permeability?"]
    finally:
        root.destroy()


@pytest.mark.skipif(not _display_available(), reason="no display available")
def test_closing_releases_the_pump_and_clears_the_callback() -> None:
    import tkinter as tk

    root = tk.Tk()
    try:
        closed: list[bool] = []
        dialog = _dialog(root, StubService())
        dialog._on_close = lambda: closed.append(True)  # type: ignore[method-assign]
        dialog._close()
        assert dialog._closed is True
        assert closed == [True]
        assert dialog._pump_after is None
    finally:
        root.destroy()


def test_the_cognition_action_is_grouped_and_opens_the_window() -> None:
    # The sidebar entry is a real action in the Assistant group, and pressing it
    # opens the cognition window rather than running a tool.
    from gui.actions import ACTIONS, grouped_visible_actions

    by_key = {action.key: action for action in ACTIONS}
    assert "cognition" in by_key
    groups = {group.key: tuple(a.key for a in actions) for group, actions in grouped_visible_actions()}
    assert "cognition" in groups["assistant"]


def test_the_cognition_cli_is_registered() -> None:
    from cli import cognition as cognition_cli
    from cli.__main__ import _COMMANDS
    from cli.menu import MENU, run_cognition

    assert _COMMANDS["cognition"] is run_cognition
    assert any(entry[1] is run_cognition for entry in MENU)
    # The CLI exposes the characters subcommand the window also drives.
    parser = cognition_cli._parser()
    subactions = next(
        a for a in parser._actions if isinstance(a, __import__("argparse")._SubParsersAction)
    )
    assert "characters" in subactions.choices

