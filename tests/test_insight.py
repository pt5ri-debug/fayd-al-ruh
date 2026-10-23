"""The insight command: asking the AI and reading what it has learned."""

from __future__ import annotations

import json

import pytest
from ai import insight


def _manifest() -> dict:
    return {
        "version": "9.9.9",
        "trained_at": "2026-01-02T03:04:05+00:00",
        "git_revision": "abc1234",
        "knowledge": {
            "disciplines": 17,
            "courses": 48,
            "ontology_concepts": 495,
            "ontology_relations": 474,
            "curriculum_coverage": 1.0,
            "corpus_coverage": 1.0,
        },
        "base": {
            "policies": {"a": {"held_out_success_rate": 1.0}, "b": {"held_out_success_rate": 0.5}},
            "brains": {"a": {"error_improvement": 0.01}},
            "supervised": {"val_accuracy": 0.9, "val_brier": 0.1},
            "general_knowledge": {"hit_at_1": 0.4, "held_out_accuracy": 0.6, "generalisation_gain": 0.15},
            "interaction_reward_model": {"brier_score": 0.02, "expected_calibration_error": 0.1},
        },
        "petroleum": {
            "overall_success": 1.0,
            "procedural_coverage": 1.0,
            "procedural_tasks": {"covered": 48, "total": 48},
            "tasks": [{"name": "t1", "held_out_success": 1.0}, {"name": "t2", "held_out_success": 0.0}],
        },
        "language_model": {
            "steps": 4000,
            "parameters": 391200,
            "preset": "release-1",
            "perplexity": {"validation": 4.675, "test": 4.265, "bits_per_byte": 2.125},
        },
    }


def test_load_manifest_returns_none_when_absent(tmp_path) -> None:
    assert insight.load_manifest(tmp_path / "missing.json") is None


def test_load_manifest_returns_none_when_unreadable(tmp_path) -> None:
    broken = tmp_path / "release_manifest.json"
    broken.write_text("{not json", encoding="utf-8")
    assert insight.load_manifest(broken) is None


def test_load_manifest_reads_a_dict(tmp_path) -> None:
    path = tmp_path / "release_manifest.json"
    path.write_text(json.dumps({"version": "1.0"}), encoding="utf-8")
    assert insight.load_manifest(path) == {"version": "1.0"}


def test_learning_rows_cover_every_layer() -> None:
    rows = insight.learning_rows(_manifest())
    layers = {row["layer"] for row in rows}
    assert "Sandbox policies" in layers
    assert "Knowledge-graph link prediction" in layers
    assert "Petroleum tasks" in layers
    assert "Procedural tasks" in layers
    assert "Language model" in layers


def test_learning_rows_include_language_model_perplexity_and_bits_per_byte() -> None:
    rows = insight.learning_rows(_manifest())
    metrics = {(row["layer"], row["metric"]) for row in rows}
    assert ("Language model", "capacity tier") in metrics
    assert ("Language model", "held-out perplexity") in metrics
    assert ("Language model", "test bits/byte") in metrics


def test_learning_rows_mean_aggregates_multiple_tasks() -> None:
    rows = insight.learning_rows(_manifest())
    held_out = next(
        row for row in rows if row["layer"] == "Sandbox policies" and row["metric"] == "held-out success"
    )
    assert held_out["value"] == pytest.approx(0.75)


def test_format_learning_report_without_manifest() -> None:
    text = insight.format_learning_report(None)
    assert "How well the project learns" in text
    assert "python -m ml.release" in text


def test_format_learning_report_marks_a_manifest_only_stub() -> None:
    text = insight.format_learning_report({"version": "2.3.0"})
    assert "no result sections" in text


def test_format_learning_report_shows_recorded_metrics() -> None:
    text = insight.format_learning_report(_manifest())
    assert "Release 9.9.9" in text
    assert "held-out Hit@1" in text
    assert "baseline 0.250" in text


def test_knowledge_report_lists_repositories() -> None:
    report = insight.knowledge_report()
    assert report["documents_seen"] > 0
    assert "curriculum_knowledge" in report["repositories"]
    assert "petroleum_corpus" in report["repositories"]
    assert "count" in report["promoted"]


def test_format_knowledge_report_mentions_promoted_and_runtime() -> None:
    text = insight.format_knowledge_report(_manifest())
    assert "What the project has learned" in text
    assert "Promoted learned statements" in text
    assert "Learned at runtime" in text


def test_ask_answers_with_citations() -> None:
    answer = insight.ask("What is porosity?", use_model=False)
    assert answer.text.strip()
    assert answer.citations
    assert answer.used_model is False


def test_format_answer_ends_with_sources_block() -> None:
    answer = insight.ask("What is porosity?", use_model=False)
    text = insight.format_answer(answer)
    assert "Sources:" in text
    assert text.rstrip().endswith(")")


def test_main_report_prints_both_sections(monkeypatch, capsys, tmp_path) -> None:
    monkeypatch.setattr(insight, "manifest_path", lambda: tmp_path / "release_manifest.json")
    path = tmp_path / "release_manifest.json"
    path.write_text(json.dumps(_manifest()), encoding="utf-8")
    assert insight.main([]) == 0
    out = capsys.readouterr().out
    assert "How well the project learns" in out
    assert "What the project has learned" in out


def test_main_json_is_machine_readable(monkeypatch, capsys, tmp_path) -> None:
    monkeypatch.setattr(insight, "manifest_path", lambda: tmp_path / "release_manifest.json")
    assert insight.main(["--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["manifest"]["present"] is False
    assert "repositories" in payload["knowledge"]


def test_main_question_prints_an_answer(capsys) -> None:
    assert insight.main(["--no-model", "What is porosity?"]) == 0
    out = capsys.readouterr().out
    assert out.strip()
    assert "Sources:" in out


def test_insight_is_wired_into_the_menu_and_cli() -> None:
    import cli.__main__ as cli_entry
    from cli import menu

    assert "insight" in cli_entry._COMMANDS
    titles = [title for title, _action, _args in menu.MENU]
    assert any("what has it learned" in title for title in titles)


def test_run_insight_dispatches_to_the_module(monkeypatch) -> None:
    from cli import menu

    captured: dict[str, object] = {}

    def fake_run_module(name: str, argv) -> int:
        captured["name"] = name
        captured["argv"] = list(argv)
        return 0

    monkeypatch.setattr(menu, "_run_module", fake_run_module)
    assert menu.run_insight(("--json",)) == 0
    assert captured == {"name": "ai.insight", "argv": ["--json"]}
