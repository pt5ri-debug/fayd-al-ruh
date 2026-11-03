"""Phase 3 CLI: ``python -m cli learning ...``.

The CLI is the headless twin of the learning system: it drives the *same*
coordinator the controller uses, so the CLI and any window share one learning
store per personality and one isolation policy. These tests run the real CLI in
a temporary data folder (never the user's own data).
"""

from __future__ import annotations

import json

import pytest
from cli import learning as cli_learning


@pytest.fixture
def env(tmp_path, monkeypatch):
    monkeypatch.setenv("FAYD_DATA_DIR", str(tmp_path))
    return tmp_path


def _run(*argv: str) -> int:
    return cli_learning.main(list(argv))


class TestLearningCli:
    def test_status_reports_honestly_when_empty(self, env, capsys):
        assert _run("status") == 0
        assert "learning for fayd_al_ruh" in capsys.readouterr().out

    def test_knowledge_is_learned(self, env, capsys):
        code = _run("knowledge", "porosity increases permeability", "--source", "document", "--confidence", "0.8")
        assert code == 0
        assert "1 integrated" in capsys.readouterr().out

    def test_unsupported_knowledge_is_not_integrated(self, env, capsys):
        # No concept resolution -> no claim -> nothing integrated (honest).
        code = _run("knowledge", "zzz qqq www", "--source", "model")
        assert code == 0
        assert "0 integrated" in capsys.readouterr().out

    def test_experience_is_recorded(self, env, capsys):
        code = _run("experience", "increase permeability", "acid worked", "--success", "--approach", "acid")
        assert code == 0
        assert "captured" in capsys.readouterr().out

    def test_feedback_is_processed(self, env, capsys):
        code = _run("feedback", "correct", "answer-1")
        assert code == 0
        assert "stored" in capsys.readouterr().out

    def test_correction_needs_the_text(self, env, capsys):
        code = _run("feedback", "correction", "answer-1")
        assert code == 2  # invalid arguments
        assert "invalid" in capsys.readouterr().out

    def test_patterns_run(self, env, capsys):
        _run("experience", "goal", "worked", "--success", "--approach", "acid")
        _run("experience", "goal", "worked again", "--success", "--approach", "acid")
        assert _run("patterns", "--min-support", "2") == 0
        assert "patterns" in capsys.readouterr().out

    def test_history_json_is_machine_readable(self, env, capsys):
        _run("knowledge", "porosity increases permeability", "--source", "document", "--confidence", "0.8")
        capsys.readouterr()  # discard the first command's output
        assert _run("history", "--json") == 0
        payload = json.loads(capsys.readouterr().out)
        assert payload and payload[0]["kind"] == "knowledge"

    def test_metrics_are_measured(self, env, capsys):
        _run("knowledge", "porosity increases permeability", "--source", "document", "--confidence", "0.8")
        capsys.readouterr()  # discard the first command's output
        assert _run("metrics", "--json") == 0
        metrics = json.loads(capsys.readouterr().out)
        assert metrics["knowledge_events"] == 1.0

    def test_evaluate_measures_the_learning(self, env, capsys):
        assert _run("evaluate", "porosity increases permeability", "--source", "document") == 0
        assert "improved" in capsys.readouterr().out

    def test_rollback_keeps_the_audit_trail(self, env, capsys):
        _run("knowledge", "porosity increases permeability", "--source", "document", "--confidence", "0.8")
        assert _run("rollback", "1", "--approve") == 0
        assert "audit trail is kept" in capsys.readouterr().out

    def test_rollback_without_approval_is_refused(self, env, capsys):
        _run("knowledge", "porosity increases permeability", "--source", "document", "--confidence", "0.8")
        capsys.readouterr()
        assert _run("rollback", "1") == 4
        assert "governed action" in capsys.readouterr().out

    def test_unknown_owner_is_refused(self, env, capsys):
        assert _run("status", "--owner", "nope") == 4
        assert "refused" in capsys.readouterr().out

    def test_no_command_prints_help(self, env, capsys):
        assert _run() == 0
        assert "learning" in capsys.readouterr().out.lower()

    def test_command_is_wired_in_the_dispatch_table(self):
        from cli.__main__ import _COMMANDS

        assert "learning" in _COMMANDS
        assert "learn-phase3" in _COMMANDS

    def test_menu_carries_the_entry(self):
        from cli.menu import MENU, run_learning

        assert any(action is run_learning for _, action, _ in MENU)
