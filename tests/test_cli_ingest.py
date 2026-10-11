"""Tests for the ingestion CLI: add, list, stats, index, and search."""

from __future__ import annotations

import json

from cli import ingest as ingest_cli


def _run(argv):
    return ingest_cli.main(argv)


def _docs(tmp_path, **files):
    for name, text in files.items():
        target = tmp_path / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text, encoding="utf-8")
    return tmp_path


def test_add_ingests_a_file(tmp_path, capsys):
    source = _docs(tmp_path / "src", **{"a.txt": "Pressure drives oil.\n"})
    store_dir = tmp_path / "kb"
    code = _run(["add", str(source), "--store-dir", str(store_dir), "--quiet"])
    assert code == 0
    assert "chunks 1" in capsys.readouterr().out


def test_add_json_output(tmp_path, capsys):
    source = _docs(tmp_path / "src", **{"a.txt": "content\n"})
    code = _run(["add", str(source), "--store-dir", str(tmp_path / "kb"), "--quiet", "--json"])
    assert code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["chunks"] == 1
    assert payload["errors"] == []


def test_add_missing_source_returns_error(tmp_path):
    code = _run(["add", str(tmp_path / "nope"), "--store-dir", str(tmp_path / "kb"), "--quiet"])
    assert code == 1


def test_add_unknown_connector_returns_error(tmp_path, capsys):
    code = _run(["add", "x.txt", "--connector", "nope", "--store-dir", str(tmp_path / "kb")])
    assert code == 1
    assert "no connector named" in capsys.readouterr().out


def test_add_web_without_approve_is_refused(tmp_path, capsys):
    code = _run(
        [
            "add",
            "https://example.com/a",
            "--allow-web",
            "--store-dir",
            str(tmp_path / "kb"),
        ]
    )
    assert code == 1
    assert "--approve" in capsys.readouterr().out


def test_list_connectors(tmp_path, capsys):
    assert _run(["list-connectors", "--store-dir", str(tmp_path / "kb")]) == 0
    out = capsys.readouterr().out
    assert "files" in out
    assert "sql" in out
    assert "web" not in out


def test_list_connectors_json(tmp_path, capsys):
    assert _run(["list-connectors", "--json", "--allow-web"]) == 0
    payload = json.loads(capsys.readouterr().out)
    names = {entry["name"] for entry in payload}
    assert {"files", "sql", "web"} <= names


def test_stats_and_search(tmp_path, capsys):
    source = _docs(
        tmp_path / "src",
        **{"oil.txt": "Permeability measures how easily fluid flows through rock."},
    )
    store_dir = tmp_path / "kb"
    _run(["add", str(source), "--store-dir", str(store_dir), "--quiet"])
    assert _run(["stats", "--store-dir", str(store_dir)]) == 0
    assert "Ingested items" in capsys.readouterr().out
    assert _run(["build-index", "--store-dir", str(store_dir)]) == 0
    capsys.readouterr()
    assert _run(["search", "permeability fluid", "--store-dir", str(store_dir)]) == 0
    assert "Permeability" in capsys.readouterr().out


def test_search_without_store_reports_helpfully(tmp_path, capsys):
    assert _run(["search", "q", "--store-dir", str(tmp_path / "empty")]) == 1
    assert "Ingest a source first" in capsys.readouterr().out
