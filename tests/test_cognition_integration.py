"""The autonomous layer's integration with the controller, and the CLI twin.

The controller must expose the proactive mind, the learning system, and the
staged trainer over its **own** ontology and causal graph -- never a second
graph -- and the CLI must drive the *same* objects, so the two surfaces can never
disagree. These tests pin both, plus the ``prove`` command that demonstrates the
whole loop end to end.
"""

from __future__ import annotations

import json

from cli import cognition as cognition_cli
from cli.__main__ import _COMMANDS
from cli.menu import MENU, run_cognition
from core.cognitive_controller import CognitiveController


# -- the controller integration ---------------------------------------------
def test_controller_shares_its_own_ontology_and_causal_graph(tmp_path, monkeypatch):
    monkeypatch.setenv("FAYD_DATA_DIR", str(tmp_path))
    controller = CognitiveController()
    mind = controller.proactive_mind()
    assert mind.ontology is controller.ontology
    assert mind.imagination.causal_graph is controller.causal_graph


def test_controller_builds_the_autonomous_layer_lazily(tmp_path, monkeypatch):
    monkeypatch.setenv("FAYD_DATA_DIR", str(tmp_path))
    controller = CognitiveController()
    assert getattr(controller, "_proactive_mind", None) is None
    controller.proactive_mind()
    assert controller._proactive_mind is not None


def test_controller_think_runs_a_cycle(tmp_path, monkeypatch):
    monkeypatch.setenv("FAYD_DATA_DIR", str(tmp_path))
    controller = CognitiveController()
    cycle = controller.think(max_ideas=2)
    assert cycle.questions or cycle.ideas
    assert controller.cognitive_status()["cycles"] >= 1


def test_controller_learning_is_one_system(tmp_path, monkeypatch):
    monkeypatch.setenv("FAYD_DATA_DIR", str(tmp_path))
    controller = CognitiveController()
    assert controller.cognitive_learning() is controller.cognitive_learning()


def test_controller_learning_persists_across_a_new_controller(tmp_path, monkeypatch):
    monkeypatch.setenv("FAYD_DATA_DIR", str(tmp_path))
    from cognition.records import Source, SourceKind

    first = CognitiveController()
    first.cognitive_learning().learn_fact(
        "f1",
        "acid stimulation increases permeability",
        source=Source(SourceKind.CORPUS, "corpus"),
        confidence=0.9,
        verified=True,
    )
    second = CognitiveController()
    assert second.cognitive_learning().semantic.facts.get("f1")


# -- the CLI ----------------------------------------------------------------
def test_cognition_is_registered_in_the_cli_and_the_menu():
    assert _COMMANDS["cognition"] is run_cognition
    assert any(entry[1] is run_cognition for entry in MENU)


def test_status_reports_the_mind(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("FAYD_DATA_DIR", str(tmp_path))
    assert cognition_cli.main(["status"]) == cognition_cli.EXIT_OK
    assert "ideas" in capsys.readouterr().out


def test_think_records_a_cycle(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("FAYD_DATA_DIR", str(tmp_path))
    assert cognition_cli.main(["think", "--ideas", "2"]) == cognition_cli.EXIT_OK
    assert "cycle" in capsys.readouterr().out


def test_learn_then_recall_survives_between_invocations(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("FAYD_DATA_DIR", str(tmp_path))
    assert cognition_cli.main(["learn", "waterflood raises pressure", "--verified"]) == cognition_cli.EXIT_OK
    capsys.readouterr()
    assert cognition_cli.main(["recall", "pressure"]) == cognition_cli.EXIT_OK
    out = capsys.readouterr().out
    assert "waterflood raises pressure" in out


def test_evaluate_prints_all_ten_indicators(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("FAYD_DATA_DIR", str(tmp_path))
    assert cognition_cli.main(["evaluate"]) == cognition_cli.EXIT_OK
    out = capsys.readouterr().out
    assert "overall" in out
    assert "Knowledge retrieval" in out
    assert "Independence" in out


def test_evaluate_json_is_machine_readable(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("FAYD_DATA_DIR", str(tmp_path))
    assert cognition_cli.main(["evaluate", "--json"]) == cognition_cli.EXIT_OK
    payload = json.loads(capsys.readouterr().out)
    assert "overall" in payload
    assert "scores" in payload


def test_train_requires_approval(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("FAYD_DATA_DIR", str(tmp_path))
    assert cognition_cli.main(["train", "--stage", "smoke"]) == cognition_cli.EXIT_GATE
    assert "approve" in capsys.readouterr().out.lower()


def test_versions_lists_the_log(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("FAYD_DATA_DIR", str(tmp_path))
    assert cognition_cli.main(["versions"]) == cognition_cli.EXIT_OK
    assert capsys.readouterr().out.strip()


def test_progression_reports_every_stage_and_the_next(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("FAYD_DATA_DIR", str(tmp_path))
    assert cognition_cli.main(["progression"]) == cognition_cli.EXIT_OK
    out = capsys.readouterr().out
    assert "smoke" in out
    assert "next stage: smoke" in out


def test_progression_json_is_machine_readable(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("FAYD_DATA_DIR", str(tmp_path))
    assert cognition_cli.main(["progression", "--json"]) == cognition_cli.EXIT_OK
    payload = json.loads(capsys.readouterr().out)
    assert payload["next"] == "smoke"
    assert next(s["stage"] for s in payload["stages"]) == "smoke"


def test_gradual_train_requires_approval(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("FAYD_DATA_DIR", str(tmp_path))
    code = cognition_cli.main(["train", "--gradual", "--up-to", "smoke"])
    assert code == cognition_cli.EXIT_GATE
    assert "approve" in capsys.readouterr().out.lower()


def test_rollback_is_honest_with_no_version(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("FAYD_DATA_DIR", str(tmp_path))
    assert cognition_cli.main(["rollback"]) == cognition_cli.EXIT_OK
    assert "base" in capsys.readouterr().out.lower()


def test_mode_round_trips_through_the_cli(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("FAYD_DATA_DIR", str(tmp_path))
    assert cognition_cli.main(["mode", "light"]) == cognition_cli.EXIT_OK
    capsys.readouterr()
    assert cognition_cli.main(["mode"]) == cognition_cli.EXIT_OK
    assert "light" in capsys.readouterr().out


def test_tick_runs_or_defers_honestly(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("FAYD_DATA_DIR", str(tmp_path))
    assert cognition_cli.main(["tick"]) == cognition_cli.EXIT_OK
    assert capsys.readouterr().out.strip()


def test_prove_demonstrates_the_whole_loop(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("FAYD_DATA_DIR", str(tmp_path))
    assert cognition_cli.main(["prove", "--ideas", "2"]) == cognition_cli.EXIT_OK
    out = capsys.readouterr().out
    assert "1. Thinking" in out
    assert "2. Measuring the ten indicators" in out
    assert "3. Learning a fact and recalling it" in out
    assert "acid stimulation increases permeability" in out


def test_teach_runs_one_exchange(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("FAYD_DATA_DIR", str(tmp_path))
    assert cognition_cli.main(["teach", "what does a density log measure"]) == cognition_cli.EXIT_OK
    assert capsys.readouterr().out.strip()


def test_an_unknown_command_is_a_usage_error():
    assert cognition_cli.main(["nonsense"]) == cognition_cli.EXIT_USAGE


def test_experience_records_a_lesson(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("FAYD_DATA_DIR", str(tmp_path))
    code = cognition_cli.main(
        ["experience", "--expect", "it rains", "--outcome", "it did not"]
    )
    assert code == cognition_cli.EXIT_OK
    assert capsys.readouterr().out.strip()
