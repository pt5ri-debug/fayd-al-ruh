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


def test_characters_evaluates_the_eight(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("FAYD_DATA_DIR", str(tmp_path))
    assert cognition_cli.main(["characters"]) == cognition_cli.EXIT_OK
    out = capsys.readouterr().out
    assert "Character evaluation" in out
    assert "8 characters" in out


def test_characters_json_is_machine_readable(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("FAYD_DATA_DIR", str(tmp_path))
    assert cognition_cli.main(["characters", "--json"]) == cognition_cli.EXIT_OK
    payload = json.loads(capsys.readouterr().out)
    assert len(payload["characters"]) == 8
    assert "dimension_averages" in payload


def test_characters_single_key_reports_its_gaps(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("FAYD_DATA_DIR", str(tmp_path))
    assert cognition_cli.main(["characters", "--key", "natasha"]) == cognition_cli.EXIT_OK
    out = capsys.readouterr().out
    assert "Natasha" in out
    assert "shortcomings" in out
    assert "next steps" in out


def test_characters_refuses_an_unknown_key(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("FAYD_DATA_DIR", str(tmp_path))
    assert cognition_cli.main(["characters", "--key", "nobody"]) == cognition_cli.EXIT_GATE
    assert "not a known character" in capsys.readouterr().out


def test_characters_is_registered_in_the_menu():
    assert any(entry[2] == ("characters",) for entry in MENU)


# -- the new faculties, through the CLI --------------------------------------
def test_metacognition_reports_the_self_model(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("FAYD_DATA_DIR", str(tmp_path))
    assert cognition_cli.main(["think", "--ideas", "2"]) == cognition_cli.EXIT_OK
    capsys.readouterr()
    assert cognition_cli.main(["metacognition"]) == cognition_cli.EXIT_OK
    assert capsys.readouterr().out.strip()


def test_metacognition_json_is_machine_readable(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("FAYD_DATA_DIR", str(tmp_path))
    assert cognition_cli.main(["metacognition", "--json"]) == cognition_cli.EXIT_OK
    payload = json.loads(capsys.readouterr().out)
    assert "cycle" in payload
    assert "strategies" in payload


def test_plan_reports_how_the_mind_will_search(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("FAYD_DATA_DIR", str(tmp_path))
    assert cognition_cli.main(["plan"]) == cognition_cli.EXIT_OK
    assert "concept(s)" in capsys.readouterr().out


def test_plan_json_is_machine_readable(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("FAYD_DATA_DIR", str(tmp_path))
    assert cognition_cli.main(["plan", "--json"]) == cognition_cli.EXIT_OK
    payload = json.loads(capsys.readouterr().out)
    assert payload["idea_budget"] >= 1
    assert "focus_topics" in payload


def test_consolidate_then_patterns_round_trips(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("FAYD_DATA_DIR", str(tmp_path))
    assert cognition_cli.main(["consolidate"]) == cognition_cli.EXIT_OK
    assert "pattern" in capsys.readouterr().out
    assert cognition_cli.main(["patterns"]) == cognition_cli.EXIT_OK
    assert capsys.readouterr().out.strip()


def test_consolidate_upgrades_a_repeated_experience(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("FAYD_DATA_DIR", str(tmp_path))
    for _ in range(3):
        cognition_cli.main(
            ["experience", "--expect", "acid on tight rock", "--outcome", "permeability improves"]
        )
    capsys.readouterr()
    assert cognition_cli.main(["consolidate", "--json"]) == cognition_cli.EXIT_OK
    payload = json.loads(capsys.readouterr().out)
    assert payload["patterns"]
    assert payload["upgraded"]


def test_review_and_skills_are_honest_when_empty(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("FAYD_DATA_DIR", str(tmp_path))
    assert cognition_cli.main(["review"]) == cognition_cli.EXIT_OK
    assert "due" in capsys.readouterr().out
    assert cognition_cli.main(["skills"]) == cognition_cli.EXIT_OK
    assert "No skills" in capsys.readouterr().out


def test_curriculum_reports_the_training_memory(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("FAYD_DATA_DIR", str(tmp_path))
    assert cognition_cli.main(["curriculum"]) == cognition_cli.EXIT_OK
    assert capsys.readouterr().out.strip()


def test_autonomous_train_requires_approval(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("FAYD_DATA_DIR", str(tmp_path))
    assert cognition_cli.main(["train", "--autonomous"]) == cognition_cli.EXIT_GATE
    assert "governed" in capsys.readouterr().out


def test_the_new_commands_are_registered_in_the_menu():
    for command in ("metacognition", "plan", "consolidate", "patterns", "skills", "review", "curriculum"):
        assert any(entry[2] == (command,) for entry in MENU), command


# -- knowledge depth (the widest reach) -------------------------------------


def test_depth_reports_the_graphs_reach(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("FAYD_DATA_DIR", str(tmp_path))
    assert cognition_cli.main(["depth"]) == cognition_cli.EXIT_OK
    out = capsys.readouterr().out
    assert "concept" in out
    assert "connectivity" in out


def test_depth_json_carries_the_counts(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("FAYD_DATA_DIR", str(tmp_path))
    assert cognition_cli.main(["depth", "--json"]) == cognition_cli.EXIT_OK
    payload = json.loads(capsys.readouterr().out)
    assert payload["concepts"] > 0
    assert payload["relations"] > 0
    assert 0.0 <= payload["connectivity"] <= 1.0


def test_coverage_lists_disciplines_and_courses(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("FAYD_DATA_DIR", str(tmp_path))
    assert cognition_cli.main(["coverage"]) == cognition_cli.EXIT_OK
    out = capsys.readouterr().out
    assert "Disciplines:" in out
    assert "Courses:" in out


def test_arabic_reach_is_measured(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("FAYD_DATA_DIR", str(tmp_path))
    assert cognition_cli.main(["arabic"]) == cognition_cli.EXIT_OK
    out = capsys.readouterr().out
    assert "Arabic alias" in out
    assert "questions of the bank in Arabic" in out


def test_harvest_previews_without_writing(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("FAYD_DATA_DIR", str(tmp_path))
    assert cognition_cli.main(["harvest", "--limit", "5"]) == cognition_cli.EXIT_OK
    out = capsys.readouterr().out
    assert "chunk" in out.lower()


def test_harvest_refuses_to_write_without_approval(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("FAYD_DATA_DIR", str(tmp_path))
    code = cognition_cli.main(["harvest", "--write", "--limit", "5"])
    assert code == cognition_cli.EXIT_GATE
    assert "governed" in capsys.readouterr().out.lower()


def test_bridges_are_readable(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("FAYD_DATA_DIR", str(tmp_path))
    assert cognition_cli.main(["bridges", "--from", "reservoir", "--to", "drilling"]) == cognition_cli.EXIT_OK
    assert capsys.readouterr().out.strip()


def test_catalog_lists_every_control_with_its_command(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("FAYD_DATA_DIR", str(tmp_path))
    assert cognition_cli.main(["catalog"]) == cognition_cli.EXIT_OK
    out = capsys.readouterr().out
    assert "Proactive Mind" in out
    assert "command: python -m cli cognition" in out
    assert "[governed]" in out


def test_catalog_json_is_machine_readable(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("FAYD_DATA_DIR", str(tmp_path))
    assert cognition_cli.main(["catalog", "--json"]) == cognition_cli.EXIT_OK
    payload = json.loads(capsys.readouterr().out)
    assert payload
    keys = {entry["key"] for entry in payload}
    assert {"think", "study", "depth", "harvest", "rollback", "catalog"} <= keys
    for entry in payload:
        assert entry["title"] and entry["summary"] and entry["outcome"]
        assert entry["arabic_title"] and entry["arabic_summary"]


def test_rollback_is_refused_without_approval_when_a_version_exists(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("FAYD_DATA_DIR", str(tmp_path))
    from cognition.versions import ModelRegistry, ModelVersion

    registry = ModelRegistry()
    version = registry.record(
        ModelVersion(
            number=1,
            path=str(tmp_path / "model.npz"),
            parent=0,
            dataset="test",
            examples=1,
            steps=1,
            accepted=True,
            reason="test",
        )
    )
    assert registry.accepted_versions()
    code = cognition_cli.main(["rollback"])
    assert code == cognition_cli.EXIT_GATE
    assert "governed" in capsys.readouterr().out.lower()
    assert cognition_cli.main(["rollback", "--approve"]) == cognition_cli.EXIT_OK
    assert f"v{version.number}" in capsys.readouterr().out


def test_the_depth_commands_are_registered_in_the_menu():
    for command in ("depth", "coverage", "harvest", "bridges", "arabic", "catalog"):
        assert any(entry[2] == (command,) for entry in MENU), command
