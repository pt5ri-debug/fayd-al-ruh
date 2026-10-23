"""The embeddings CLI: status, train, similar, compare, evaluate."""

from __future__ import annotations

import json

import pytest
from cli import embeddings as embeddings_cli
from core.embeddings import reset_semantic_cache


@pytest.fixture(autouse=True)
def _isolated_data(tmp_path, monkeypatch):
    """Every CLI test gets its own data folder and a clean in-process cache."""
    monkeypatch.setenv("FAYD_DATA_DIR", str(tmp_path))
    reset_semantic_cache()
    yield
    reset_semantic_cache()


def test_status_before_training_says_how_to_build_it(capsys):
    code = embeddings_cli.main(["status"])
    assert code == embeddings_cli.EXIT_OK
    out = capsys.readouterr().out
    assert "No semantic space has been trained yet" in out
    assert "python -m cli embeddings train" in out


def test_status_json_before_training(capsys):
    code = embeddings_cli.main(["status", "--json"])
    assert code == embeddings_cli.EXIT_OK
    payload = json.loads(capsys.readouterr().out)
    assert payload["trained"] is False
    assert payload["vocabulary_size"] == 0


def test_train_then_status_reports_the_space(capsys):
    assert embeddings_cli.main(["train"]) == embeddings_cli.EXIT_OK
    capsys.readouterr()
    code = embeddings_cli.main(["status", "--json"])
    assert code == embeddings_cli.EXIT_OK
    payload = json.loads(capsys.readouterr().out)
    assert payload["trained"] is True
    assert payload["vocabulary_size"] > 100
    assert payload["dimensions"] > 0
    assert payload["documents"] > 100


def test_train_json_is_small_and_does_not_dump_every_vector(capsys):
    assert embeddings_cli.main(["train", "--json"]) == embeddings_cli.EXIT_OK
    payload = json.loads(capsys.readouterr().out)
    assert set(payload) == {
        "path",
        "vocabulary_size",
        "dimensions",
        "documents",
        "tokens",
        "window",
        "min_count",
    }


def test_train_accepts_custom_settings(capsys):
    code = embeddings_cli.main(
        ["train", "--dimensions", "32", "--window", "2", "--min-count", "1", "--json"]
    )
    assert code == embeddings_cli.EXIT_OK
    payload = json.loads(capsys.readouterr().out)
    assert payload["dimensions"] == 32
    assert payload["window"] == 2
    assert payload["min_count"] == 1


def test_similar_finds_a_topic_word(capsys):
    assert embeddings_cli.main(["similar", "permeability", "--limit", "15"]) == embeddings_cli.EXIT_OK
    out = capsys.readouterr().out
    assert "Terms nearest 'permeability'" in out
    assert "porosity" in out


def test_similar_json_is_machine_readable(capsys):
    assert embeddings_cli.main(["similar", "porosity", "--json"]) == embeddings_cli.EXIT_OK
    payload = json.loads(capsys.readouterr().out)
    assert payload
    assert {"term", "score"} == set(payload[0])


def test_similar_reports_an_unknown_word_honestly(capsys):
    code = embeddings_cli.main(["similar", "zzzzznotaword", "qqqqqnotaword"])
    assert code == embeddings_cli.EXIT_OK
    assert "has never seen these words" in capsys.readouterr().out


def test_compare_prints_a_number(capsys):
    assert embeddings_cli.main(["compare", "porosity", "permeability"]) == embeddings_cli.EXIT_OK
    out = capsys.readouterr().out
    assert "similarity('porosity', 'permeability')" in out


def test_compare_json(capsys):
    assert embeddings_cli.main(["compare", "porosity", "permeability", "--json"]) == embeddings_cli.EXIT_OK
    payload = json.loads(capsys.readouterr().out)
    assert payload["left"] == "porosity"
    assert payload["right"] == "permeability"
    assert 0.0 <= payload["similarity"] <= 1.0


def test_compare_of_unknown_words_is_zero_and_says_why(capsys):
    code = embeddings_cli.main(["compare", "zzzzz", "qqqqq"])
    assert code == embeddings_cli.EXIT_OK
    out = capsys.readouterr().out
    assert "= 0.0000" in out
    assert "knows none of these words" in out


def test_evaluate_reports_the_held_out_numbers(capsys):
    assert embeddings_cli.main(["evaluate"]) == embeddings_cli.EXIT_OK
    out = capsys.readouterr().out
    assert "Held-out evaluation" in out
    assert "separation" in out
    assert "learned: True" in out


def test_evaluate_json(capsys):
    assert embeddings_cli.main(["evaluate", "--json"]) == embeddings_cli.EXIT_OK
    payload = json.loads(capsys.readouterr().out)
    assert payload["learned"] is True
    assert payload["pairs_measured"] > 0
    assert payload["related_cosine"] > payload["random_cosine"]


def test_evaluate_honours_a_custom_hold_out(capsys):
    assert embeddings_cli.main(["evaluate", "--hold-out", "3", "--json"]) == embeddings_cli.EXIT_OK
    payload = json.loads(capsys.readouterr().out)
    assert payload["pairs_measured"] >= 0


def test_main_returns_usage_code_for_a_bad_dimension():
    assert embeddings_cli.main(["train", "--dimensions", "0"]) == embeddings_cli.EXIT_USAGE


def test_cli_is_wired_into_the_menu_and_dispatcher():
    from cli.__main__ import _COMMANDS
    from cli.menu import MENU, run_embeddings

    assert _COMMANDS["embeddings"] is run_embeddings
    assert any(runner is run_embeddings for _label, runner, _args in MENU)
