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


# --- infer: what the graph entails ----------------------------------------


def test_infer_prints_a_derivation(capsys):
    code = relations_cli.main(["infer", "What does a well penetrate?"])
    assert code == relations_cli.EXIT_OK
    out = capsys.readouterr().out
    assert "Inferred from the knowledge graph" in out
    assert "penetrates" in out


def test_infer_json_is_machine_readable(capsys):
    code = relations_cli.main(["infer", "What does a well penetrate?", "--json"])
    assert code == relations_cli.EXIT_OK
    payload = json.loads(capsys.readouterr().out)
    assert payload["inferred"]
    first = payload["inferred"][0]
    assert {"source", "relation_type", "target", "confidence", "steps"} <= set(first)
    assert first["steps"]


def test_infer_says_nothing_for_an_unknown_question(capsys):
    code = relations_cli.main(["infer", "zzz-not-a-concept"])
    assert code == relations_cli.EXIT_OK
    assert "entails nothing" in capsys.readouterr().out


def test_infer_needs_a_question(capsys):
    code = relations_cli.main(["infer"])
    assert code == relations_cli.EXIT_NOTHING


def test_infer_rejects_a_bad_limit(capsys):
    assert relations_cli.main(["infer", "porosity", "--limit", "0"]) == relations_cli.EXIT_USAGE
    assert relations_cli.main(
        ["infer", "porosity", "--min-confidence", "2"]
    ) == relations_cli.EXIT_USAGE


# --- connections: how the concepts a question names are linked -------------


def test_connections_prints_a_chain(capsys):
    code = relations_cli.main(["connections", "how do casing and perforating relate?"])
    assert code == relations_cli.EXIT_OK
    out = capsys.readouterr().out
    assert "walked, not stored" in out
    assert "step(s)" in out


def test_connections_json_is_machine_readable(capsys):
    code = relations_cli.main(["connections", "casing perforating completion", "--json"])
    assert code == relations_cli.EXIT_OK
    payload = json.loads(capsys.readouterr().out)
    assert payload["connections"]
    first = payload["connections"][0]
    assert {"source", "target", "hops", "confidence", "steps"} <= set(first)
    assert first["steps"]


def test_connections_says_nothing_for_one_concept(capsys):
    code = relations_cli.main(["connections", "what is porosity?"])
    assert code == relations_cli.EXIT_OK
    assert "connects no two concepts" in capsys.readouterr().out


def test_connections_needs_a_question(capsys):
    assert relations_cli.main(["connections"]) == relations_cli.EXIT_NOTHING


def test_connections_rejects_a_bad_bound(capsys):
    assert relations_cli.main(["connections", "casing perforating", "--limit", "0"]) == relations_cli.EXIT_USAGE
    assert relations_cli.main(["connections", "casing perforating", "--max-hops", "0"]) == relations_cli.EXIT_USAGE


# --- causal: the directed causal paths the graph states --------------------


def test_causal_prints_a_chain(capsys):
    code = relations_cli.main(["causal", "why does waterflood increase production?"])
    assert code == relations_cli.EXIT_OK
    out = capsys.readouterr().out
    assert "directed, cause" in out
    assert "waterflood" in out and "production" in out


def test_causal_json_is_machine_readable(capsys):
    code = relations_cli.main(["causal", "why does waterflood increase production?", "--json"])
    assert code == relations_cli.EXIT_OK
    payload = json.loads(capsys.readouterr().out)
    assert payload["causal_paths"]
    first = payload["causal_paths"][0]
    assert {"cause", "effect", "hops", "confidence", "nodes", "mechanisms", "sentence"} <= set(first)
    assert first["nodes"]


def test_causal_says_nothing_for_one_concept(capsys):
    code = relations_cli.main(["causal", "what is porosity?"])
    assert code == relations_cli.EXIT_OK
    assert "no causal path" in capsys.readouterr().out


def test_causal_needs_a_question(capsys):
    assert relations_cli.main(["causal"]) == relations_cli.EXIT_NOTHING


def test_causal_rejects_bad_bounds(capsys):
    assert relations_cli.main(["causal", "porosity permeability", "--limit", "0"]) == relations_cli.EXIT_USAGE
    assert relations_cli.main(["causal", "porosity permeability", "--max-depth", "0"]) == relations_cli.EXIT_USAGE
