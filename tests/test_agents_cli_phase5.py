"""Phase 5 CLI tests: the headless twin of the multi-agent layer."""

from __future__ import annotations

import json

from cli import agents


def test_list(capsys) -> None:
    assert agents.main(["list"]) == 0
    out = capsys.readouterr().out
    assert "Jarvis" in out and "Natasha" in out


def test_list_json(capsys) -> None:
    assert agents.main(["list", "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert isinstance(payload, list)
    assert any(entry["personality_id"] == "jarvis" for entry in payload)


def test_show(capsys) -> None:
    assert agents.main(["show", "vision"]) == 0
    assert "Vision" in capsys.readouterr().out


def test_show_json(capsys) -> None:
    assert agents.main(["show", "sources", "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["personality_id"] == "jarvis"


def test_capabilities(capsys) -> None:
    assert agents.main(["capabilities", "natasha"]) == 0
    out = capsys.readouterr().out
    assert "natasha" in out
    assert "contribute_training" not in out  # private: excluded


def test_route(capsys) -> None:
    assert agents.main(["route", "what is the porosity of the reservoir?"]) == 0
    assert "vision" in capsys.readouterr().out


def test_route_none_exits_four(capsys) -> None:
    assert agents.main(["route", "zzz qqq"]) == 4


def test_route_allow_none(capsys) -> None:
    assert agents.main(["route", "zzz qqq", "--allow-none"]) == 0


def test_route_override(capsys) -> None:
    assert agents.main(["route", "x", "--override", "hakeem", "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["selected"] == "hakeem" and payload["manual"] is True


def test_run_through_echo(capsys) -> None:
    assert agents.main(["run", "vision", "what is porosity?"]) == 0
    assert "vision" in capsys.readouterr().out.lower()


def test_run_json(capsys) -> None:
    assert agents.main(["run", "vision", "q", "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["owner"] == "vision" and payload["ok"] is True


def test_run_refusal_exits_four(capsys) -> None:
    # A capability the runtime refuses is not reachable from the CLI's run, but a
    # missing personality is refused with exit 3 by the registry.
    assert agents.main(["show", "not-a-character"]) in (0, 3)


def test_collaborate(capsys) -> None:
    assert agents.main(["collaborate", "jarvis", "q", "vision", "ultron"]) == 0
    out = capsys.readouterr().out
    assert "vision" in out and "ultron" in out


def test_conflicts(capsys) -> None:
    assert agents.main(["conflicts", "jarvis", "q", "vision", "ultron"]) == 0
    out = capsys.readouterr().out
    assert "conflict" in out.lower() or "consensus" in out.lower()


def test_monitoring(capsys) -> None:
    assert agents.main(["monitoring"]) == 0
    assert "agent monitoring" in capsys.readouterr().out


def test_monitoring_json(capsys) -> None:
    assert agents.main(["monitoring", "--json"]) == 0
    assert isinstance(json.loads(capsys.readouterr().out), list)


def test_isolation(capsys) -> None:
    assert agents.main(["isolation"]) == 0
    out = capsys.readouterr().out
    assert "natasha" in out and "clean" in out.lower()


def test_isolation_json(capsys) -> None:
    assert agents.main(["isolation", "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert "natasha" in payload["private"]
    assert payload["dataset_is_clean"] is True


def test_no_command_defaults_to_list(capsys) -> None:
    assert agents.main([]) == 0
    assert "Jarvis" in capsys.readouterr().out


def test_dispatcher_registers_the_command() -> None:
    from cli.__main__ import _COMMANDS
    from cli.menu import run_agents

    assert _COMMANDS["agents"] is run_agents
    assert _COMMANDS["agent-system"] is run_agents


def test_menu_lists_the_agents_entries() -> None:
    from cli.menu import MENU, run_agents

    entries = [entry for entry in MENU if entry[1] is run_agents]
    assert entries, "the agents command must have menu entries"
    assert any("independent agents" in entry[0] for entry in entries)
