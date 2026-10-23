"""The analogy CLI: the headless twin of the concept-analogy engine."""

import json

import pytest
from cli import analogy as analogy_cli
from cli.__main__ import _COMMANDS
from cli.menu import MENU, run_analogy


def test_analogy_is_registered_in_the_cli_and_the_menu():
    assert _COMMANDS["analogy"] is run_analogy
    assert any(entry[1] is run_analogy for entry in MENU)


def test_disciplines_lists_the_seeded_ones(capsys):
    assert analogy_cli.main(["disciplines"]) == analogy_cli.EXIT_OK
    out = capsys.readouterr().out
    assert "reservoir" in out
    assert "drilling" in out


def test_between_prints_the_mapping_and_the_gaps(capsys):
    assert analogy_cli.main(["between", "porosity", "permeability"]) == analogy_cli.EXIT_OK
    out = capsys.readouterr().out
    assert "Analogy: porosity ~ permeability" in out
    assert "relation coverage" in out


def test_between_json_carries_the_honest_fields(capsys):
    assert analogy_cli.main(["between", "porosity", "permeability", "--json"]) == analogy_cli.EXIT_OK
    payload = json.loads(capsys.readouterr().out)
    assert payload["base"] == "porosity"
    assert payload["target"] == "permeability"
    assert payload["meaningful"] is True
    assert payload["matched_relations"]
    assert "candidate_inferences" in payload
    assert payload["exact_share"] > 0.0


def test_like_ranks_analogous_concepts(capsys):
    assert analogy_cli.main(["like", "porosity", "--limit", "3"]) == analogy_cli.EXIT_OK
    out = capsys.readouterr().out
    assert "structurally analogous" in out
    assert "permeability" in out


def test_like_json_is_a_list(capsys):
    assert analogy_cli.main(["like", "porosity", "--json"]) == analogy_cli.EXIT_OK
    payload = json.loads(capsys.readouterr().out)
    assert isinstance(payload, list)
    assert all(item["meaningful"] for item in payload)


def test_unknown_concept_is_reported_with_its_exit_code(capsys):
    assert analogy_cli.main(["between", "porosity", "not-a-concept"]) == analogy_cli.EXIT_UNKNOWN
    assert "Could not align" in capsys.readouterr().out


def test_unknown_discipline_is_reported(capsys):
    assert analogy_cli.main(["discipline", "drilling", "nope"]) == analogy_cli.EXIT_UNKNOWN
    assert "Unknown discipline" in capsys.readouterr().out


def test_discipline_aligns_two_seeded_disciplines(capsys):
    assert analogy_cli.main(["discipline", "reservoir", "process_safety"]) == analogy_cli.EXIT_OK
    assert "Analogy:" in capsys.readouterr().out


def test_a_bad_limit_is_rejected(capsys):
    assert analogy_cli.main(["like", "porosity", "--limit", "0"]) == analogy_cli.EXIT_UNKNOWN
    assert "Could not rank" in capsys.readouterr().out


def test_unknown_subcommand_is_a_usage_error():
    with pytest.raises(SystemExit):
        analogy_cli.main(["not-a-command"])
