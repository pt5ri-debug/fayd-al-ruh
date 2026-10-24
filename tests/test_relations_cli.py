"""The relations CLI: the headless twin of the relation extractor."""

import json

from cli import relations as relations_cli
from cli.__main__ import _COMMANDS
from cli.menu import MENU, run_relations


def test_relations_is_registered_in_the_cli_and_the_menu():
    assert _COMMANDS["relations"] is run_relations
    assert any(entry[1] is run_relations for entry in MENU)


def test_types_lists_the_extractable_relations(capsys):
    assert relations_cli.main(["types"]) == relations_cli.EXIT_OK
    out = capsys.readouterr().out
    assert "increases" in out
    assert "measures" in out


def test_read_prints_the_relation_and_its_pattern(capsys):
    code = relations_cli.main(["read", "Porosity increases permeability."])
    assert code == relations_cli.EXIT_OK
    out = capsys.readouterr().out
    assert "porosity --increases--> permeability" in out
    assert "pattern:" in out


def test_read_json_carries_the_honest_fields(capsys):
    code = relations_cli.main(["read", "Porosity increases permeability.", "--json"])
    assert code == relations_cli.EXIT_OK
    payload = json.loads(capsys.readouterr().out)
    assert payload["extracted"] == 1
    relation = payload["relations"][0]
    assert relation["source"] == "porosity"
    assert relation["relation_type"] == "increases"
    assert relation["target"] == "permeability"
    assert relation["sentence"]


def test_read_reports_when_nothing_matches(capsys):
    code = relations_cli.main(["read", "Baking bread is unrelated."])
    assert code == relations_cli.EXIT_OK
    assert "No relations matched" in capsys.readouterr().out


def test_read_with_no_input_is_a_usage_error(capsys):
    assert relations_cli.main(["read"]) == relations_cli.EXIT_NOTHING
    assert "Nothing to read" in capsys.readouterr().out


def test_read_from_a_file(tmp_path, capsys):
    path = tmp_path / "notes.txt"
    path.write_text("The density log measures porosity.", encoding="utf-8")
    code = relations_cli.main(["read", "--file", str(path)])
    assert code == relations_cli.EXIT_OK
    assert "density_log --measures--> porosity" in capsys.readouterr().out


def test_learn_previews_without_writing(capsys):
    """The default is a preview: nothing is learned without --write."""
    code = relations_cli.main(["learn", "Wettability affects flow."])
    assert code == relations_cli.EXIT_OK
    out = capsys.readouterr().out
    assert "Preview only" in out
    assert "wettability --affects--> flow" in out


def test_learn_write_reports_what_it_added(capsys):
    code = relations_cli.main(["learn", "Wettability affects flow.", "--write"])
    assert code == relations_cli.EXIT_OK
    assert "added: wettability --affects--> flow" in capsys.readouterr().out


def test_learn_write_json_is_machine_readable(capsys):
    code = relations_cli.main(["learn", "Wettability affects flow.", "--write", "--json"])
    assert code == relations_cli.EXIT_OK
    payload = json.loads(capsys.readouterr().out)
    assert ["wettability", "affects", "flow"] in payload["added"]


def test_connect_explains_a_direct_relation(capsys):
    code = relations_cli.main(["connect", "porosity", "permeability"])
    assert code == relations_cli.EXIT_OK
    out = capsys.readouterr().out
    assert "porosity" in out and "permeability" in out
    assert "step" in out


def test_connect_json_is_machine_readable(capsys):
    code = relations_cli.main(["connect", "porosity", "permeability", "--json"])
    assert code == relations_cli.EXIT_OK
    payload = json.loads(capsys.readouterr().out)
    assert payload["connected"] is True
    assert payload["hops"] >= 1
    assert payload["steps"]


def test_connect_says_plainly_when_there_is_no_chain(capsys):
    code = relations_cli.main(["connect", "porosity", "zzz-not-a-concept"])
    assert code == relations_cli.EXIT_OK
    assert "No chain" in capsys.readouterr().out


def test_connect_rejects_a_bad_hop_bound(capsys):
    code = relations_cli.main(["connect", "porosity", "permeability", "--max-hops", "0"])
    assert code == relations_cli.EXIT_USAGE


def test_conflicts_reports_a_clean_graph(capsys):
    code = relations_cli.main(["conflicts"])
    assert code == relations_cli.EXIT_OK
    assert "No contradictory relations" in capsys.readouterr().out


def test_conflicts_json_is_machine_readable(capsys):
    code = relations_cli.main(["conflicts", "--json"])
    assert code == relations_cli.EXIT_OK
    payload = json.loads(capsys.readouterr().out)
    assert payload["conflicts"] == []


def test_conflicts_scopes_to_a_concept(capsys):
    code = relations_cli.main(["conflicts", "--concept", "porosity"])
    assert code == relations_cli.EXIT_OK
    assert "porosity" in capsys.readouterr().out
