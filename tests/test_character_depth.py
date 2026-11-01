"""The character depth layer: memory view, consistency check, personality layer.

These tests exercise the three pieces added for a comprehensive character
evaluation -- a unified read-only *memory view*, a *consistency* check over a
standing configuration, and a *personality layer* (registers + initiative) for the
prompt-only mouths. Everything runs headless with no model, no display, and no
network, and each character store is isolated under a temporary data root.
"""

from __future__ import annotations

import pytest
from core.character_learning import CHARACTER_KEYS, PROJECT_MODEL_KEY, CharacterLearning
from core.character_memory_view import (
    apply_fact_decay,
    memory_view,
    memory_view_stats,
)
from core.personality_layer import (
    LAYER_MOUTHS,
    describe_layers,
    has_personality_layer,
    layer_for,
    layer_keys,
)


@pytest.fixture(autouse=True)
def _data_root(tmp_path, monkeypatch):
    monkeypatch.setenv("FAYD_DATA_DIR", str(tmp_path))
    return tmp_path


# -- the memory view --------------------------------------------------------


def test_memory_view_is_empty_before_anything_is_learned():
    view = memory_view("vision")
    assert view.entries() == ()
    stats = view.stats()
    assert stats.facts == 0 and stats.exchanges == 0


def test_memory_view_reads_back_a_fact():
    CharacterLearning("vision").add_fact("porosity is the void fraction")
    view = memory_view("vision")
    texts = [entry.text for entry in view.entries()]
    assert "porosity is the void fraction" in texts


def test_memory_view_reads_back_an_exchange():
    learning = CharacterLearning("ultron")
    learning.remember(question="what is porosity?", answer="the void fraction")
    entries = memory_view("ultron").entries()
    assert any(entry.kind == "exchange" for entry in entries)


def test_memory_view_searches_what_it_holds():
    learning = CharacterLearning("banner")
    learning.add_fact("waterflood increases recovery")
    learning.add_fact("acid stimulation increases permeability")
    hits = memory_view("banner").search("permeability")
    assert hits
    assert all("permeability" in hit.entry.text.lower() for hit in hits)


def test_memory_view_search_is_empty_for_an_unrelated_query():
    CharacterLearning("cloud").add_fact("porosity is the void fraction")
    assert memory_view("cloud").search("hydraulic fracturing") == ()


def test_memory_view_marks_a_private_character():
    assert memory_view("natasha").stats().private is True
    assert memory_view("jarvis").stats().private is False


def test_memory_view_stats_round_trip_through_the_helper():
    CharacterLearning("hakeem").add_fact("the treaty of hudaybiyyah was in 6 AH")
    assert memory_view_stats("hakeem").facts == 1


def test_memory_view_owner_is_resolved_from_the_key():
    # The owner is derived, never passed, so the view cannot be pointed at another
    # character's records.
    assert memory_view("jarvis").owner
    assert memory_view("jarvis").owner != memory_view("vision").owner


def test_fact_decay_ranks_without_deleting():
    learning = CharacterLearning("ultron")
    learning.add_fact("a")
    learning.add_fact("b")
    ranked = apply_fact_decay(learning.facts(), rate=0.01)
    assert len(ranked) == 2
    # It is a ranking, not a deletion.
    assert learning.facts()


# -- gradual decay with real fact ages --------------------------------------


def test_add_fact_stamps_an_age():
    learning = CharacterLearning("ultron")
    learning.add_fact("porosity is the void fraction")
    records = learning.fact_records()
    assert len(records) == 1
    fact, age = records[0]
    assert fact == "porosity is the void fraction"
    assert age  # a real ISO timestamp, so decay has a reference point


def test_older_fact_fades_below_a_newer_one():
    from datetime import UTC, datetime, timedelta

    learning = CharacterLearning("ultron")
    learning.add_fact("old fact")
    old = (datetime.now(UTC) - timedelta(days=400)).isoformat()
    learning.record.fact_ages["old fact"] = old
    learning.add_fact("new fact")
    ranked = dict(learning.fact_records())
    assert ranked["old fact"] == old

    scored = dict(apply_fact_decay(learning.fact_records(), rate=0.01))
    assert scored["old fact"] < scored["new fact"]


def test_decay_never_prunes_a_fact_with_no_age():
    from core.character_memory_view import CharacterMemoryView

    learning = CharacterLearning("ultron")
    learning.add_fact("ageless fact")
    # Simulate a record written before ages existed.
    learning.record.fact_ages.clear()
    learning._save(merge=False)
    view = CharacterMemoryView("ultron")
    assert view.prune_faded(floor=0.9) == ()
    assert learning.facts() == ("ageless fact",)


def test_prune_faded_reports_then_forgets_only_when_asked():
    from datetime import UTC, datetime, timedelta

    from core.character_memory_view import CharacterMemoryView

    learning = CharacterLearning("ultron")
    learning.add_fact("stale fact")
    learning.record.fact_ages["stale fact"] = (
        datetime.now(UTC) - timedelta(days=400)
    ).isoformat()
    learning._save(merge=False)
    view = CharacterMemoryView("ultron")

    reported = view.prune_faded(floor=0.5)
    assert reported == ("stale fact",)
    # Read-only by default: the fact is still there.
    assert CharacterLearning("ultron").facts() == ("stale fact",)

    applied = view.prune_faded(floor=0.5, apply=True)
    assert applied == ("stale fact",)
    assert CharacterLearning("ultron").facts() == ()


def test_forget_fact_reports_whether_it_was_there():
    learning = CharacterLearning("ultron")
    learning.add_fact("a fact")
    assert learning.forget_fact("a fact") is True
    assert learning.forget_fact("a fact") is False
    assert learning.facts() == ()


def test_fact_ages_survive_a_reload():
    from datetime import UTC, datetime, timedelta

    learning = CharacterLearning("ultron")
    learning.add_fact("a fact")
    old = (datetime.now(UTC) - timedelta(days=200)).isoformat()
    learning.record.fact_ages["a fact"] = old
    learning._save(merge=False)

    reopened = CharacterLearning("ultron")
    assert reopened.fact_records() == (("a fact", old),)


# -- semantic recall --------------------------------------------------------


def test_rank_exchanges_puts_the_relevant_one_first():
    from core.character_memory_view import CharacterMemoryView

    learning = CharacterLearning("ultron")
    learning.remember("How do I bake bread?", "Mix flour, water, and yeast.")
    learning.remember("What is permeability?", "The ability of rock to transmit fluid.")
    view = CharacterMemoryView("ultron")

    ranked = view.rank_exchanges("tell me about permeability in rock")
    assert ranked
    assert ranked[0].question == "What is permeability?"


def test_rank_exchanges_skips_the_question_on_screen():
    from core.character_memory_view import CharacterMemoryView

    learning = CharacterLearning("ultron")
    learning.remember("What is permeability?", "The ability to transmit fluid.")
    learning.remember("What is porosity?", "The void fraction.")
    view = CharacterMemoryView("ultron")

    ranked = view.rank_exchanges(
        "permeability", exclude_questions=("What is permeability?",)
    )
    assert all(exchange.question != "What is permeability?" for exchange in ranked)


def test_rank_exchanges_is_empty_for_a_blank_question():
    from core.character_memory_view import CharacterMemoryView

    learning = CharacterLearning("ultron")
    learning.remember("q", "a")
    assert CharacterMemoryView("ultron").rank_exchanges("   ") == ()


def test_recall_matches_search():
    from core.character_memory_view import CharacterMemoryView

    learning = CharacterLearning("ultron")
    learning.add_fact("porosity is the void fraction")
    view = CharacterMemoryView("ultron")
    assert [hit.entry.text for hit in view.recall("porosity")] == [
        hit.entry.text for hit in view.search("porosity")
    ]


def test_wrapper_uses_relevant_recall_in_the_prompt():
    from ai.characters import wrap_character
    from core.character_learning import CharacterLearning as CL

    class _Engine:
        system_prompt = "BASE"

    class _Mouth:
        def __init__(self) -> None:
            self.engine = _Engine()

        def ask(self, question, **_kwargs):
            class _A:
                text = "ok"

            return _A()

    CL("ultron").remember("How do I bake bread?", "Mix flour and yeast.")
    CL("ultron").remember("What is permeability?", "The ability to transmit fluid.")
    mouth = _Mouth()
    character = wrap_character("ultron", mouth)
    # A fresh wrapper renders the standing prompt on its first answer, choosing
    # the recalled window by relevance to the question asked.
    character.ask("explain permeability to me")
    assert "permeability" in mouth.engine.system_prompt.lower()


# -- the language-model catalog's ready model -------------------------------


def test_ready_model_falls_back_to_any_available_model():
    from core import language_models as lm

    class _Model:
        def __init__(self, key, available):
            self.key = key
            self.available = available

    models = (_Model("default", False), _Model("shared", True))
    original = lm.language_models
    lm.language_models = lambda mouth: models  # type: ignore[assignment]
    try:
        assert lm.ready_language_model_key("ultron") == "shared"
    finally:
        lm.language_models = original  # type: ignore[assignment]


def test_ready_model_prefers_the_default_when_it_is_installed():
    from core import language_models as lm

    class _Model:
        def __init__(self, key, available):
            self.key = key
            self.available = available

    models = (_Model("default", True), _Model("shared", True))
    original = lm.language_models
    lm.language_models = lambda mouth: models  # type: ignore[assignment]
    try:
        assert lm.ready_language_model_key("ultron") == "default"
    finally:
        lm.language_models = original  # type: ignore[assignment]


def test_ready_model_is_none_when_nothing_is_installed():
    from core import language_models as lm

    class _Model:
        def __init__(self, key, available):
            self.key = key
            self.available = available

    models = (_Model("default", False), _Model("shared", False))
    original = lm.language_models
    lm.language_models = lambda mouth: models  # type: ignore[assignment]
    try:
        assert lm.ready_language_model_key("ultron") is None
    finally:
        lm.language_models = original  # type: ignore[assignment]


# -- the consistency check --------------------------------------------------


def test_consistency_accepts_a_shipped_configuration():
    from core.character_consistency import check_consistency
    from core.character_registry import default_registry

    registry = default_registry()
    for key in registry.keys():
        report = check_consistency(registry.load(key))
        assert report.key
        # The shipped defaults are internally consistent.
        assert report.ok, report.describe()


def test_consistency_accepts_a_bare_object():
    from core.character_consistency import check_consistency

    report = check_consistency(object())
    assert report.ok


def test_consistency_flags_a_high_formality_with_a_casual_style():
    from core.character_config import default_config_for, replace_personality
    from core.character_consistency import check_consistency

    config = replace_personality(
        default_config_for("jarvis"),
        formality=0.9,
        communication_style="casual and chatty",
    )
    report = check_consistency(config)
    assert not report.ok
    assert any("formality" in w.code for w in report.warnings)


def test_consistency_report_round_trips():
    from core.character_consistency import ConsistencyReport

    report = ConsistencyReport(key="jarvis")
    assert report.ok
    assert "consistent" in report.describe()


def test_check_all_reports_every_character():
    from core.character_consistency import check_all
    from core.character_registry import default_registry

    registry = default_registry()
    reports = check_all(registry.load(k) for k in CHARACTER_KEYS)
    assert len(reports) == len(CHARACTER_KEYS)


# -- the personality layer --------------------------------------------------


def test_the_four_prompt_only_mouths_have_a_layer():
    assert set(layer_keys()) == set(LAYER_MOUTHS)
    for key in LAYER_MOUTHS:
        assert layer_for(key) is not None


def test_a_mouth_with_its_own_layer_gets_no_second_one():
    # Jarvis (grounding), Natasha (companion), Hakeem (scholar), and the project
    # model keep their own dedicated layers.
    for key in ("jarvis", "natasha", "hakeem", PROJECT_MODEL_KEY):
        assert layer_for(key) is None
        assert has_personality_layer(key) is True


def test_an_unknown_mouth_has_no_layer():
    assert layer_for("nobody") is None
    assert has_personality_layer("nobody") is False


def test_the_layer_renders_a_style_and_initiative_block():
    layer = layer_for("ultron")
    assert layer is not None
    assert layer.style_block()
    assert layer.initiative_block()
    assert 0.0 <= layer.depth <= 1.0


def test_the_layer_is_deterministic():
    first = layer_for("vision").register("show me the numbers")  # type: ignore[union-attr]
    second = layer_for("vision").register("show me the numbers")  # type: ignore[union-attr]
    assert first == second


def test_describe_layers_names_every_layered_mouth():
    lines = describe_layers()
    assert len(lines) == len(LAYER_MOUTHS)


def test_the_layer_never_claims_a_tool_ran():
    for key in LAYER_MOUTHS:
        layer = layer_for(key)
        assert layer is not None
        text = (layer.style_block() + layer.initiative_block()).lower()
        assert "tool ran" not in text
        assert "i have executed" not in text


def test_standing_block_states_its_own_limit():
    block = layer_for("ultron").standing_block()  # type: ignore[union-attr]
    assert "tone and initiative only" in block
    assert "honesty rules" in block
    assert "registers" in block
    assert "initiative" in block


def test_the_layer_reaches_the_prompt_of_the_four_prompt_only_mouths():
    from ai.characters import CharacterMouth

    class _Engine:
        pass

    for key in LAYER_MOUTHS:
        prompt = CharacterMouth(key, _Engine()).system_prompt()
        assert "standing manner" in prompt


def test_a_mouth_with_its_own_layer_gets_no_second_layer_in_its_prompt():
    from ai.characters import CharacterMouth

    class _Engine:
        pass

    for key in ("jarvis", "natasha", "hakeem"):
        prompt = CharacterMouth(key, _Engine()).system_prompt()
        assert "standing manner" not in prompt


# -- the character CLI ------------------------------------------------------


def test_character_cli_exposes_the_evaluation_commands():
    import argparse

    from cli.characters import build_parser

    parser = build_parser()
    subactions = next(
        a for a in parser._actions if isinstance(a, argparse._SubParsersAction)
    )
    assert {"memory", "consistency", "evaluate"} <= set(subactions.choices)


def test_character_cli_consistency_all(capsys):
    from cli.characters import main

    code = main(["consistency", "--all"])
    out = capsys.readouterr().out
    assert code == 0
    assert "consistent" in out


def test_character_cli_consistency_defaults_to_all(capsys):
    from cli.characters import main

    # With no character and no --all, it reports every character rather than
    # printing an empty row for the empty key.
    code = main(["consistency"])
    out = capsys.readouterr().out
    assert code == 0
    assert out.count("internally consistent") >= len(CHARACTER_KEYS)


def test_character_cli_fill_defaults_to_all(capsys):
    from cli.characters import main

    code = main(["fill"])
    out = capsys.readouterr().out
    assert code == 0
    for key in ("jarvis", "ultron", "vision"):
        assert f"{key}:" in out


def test_character_cli_consistency_json(capsys):
    import json as _json

    from cli.characters import main
    from core.character_registry import default_registry

    code = main(["consistency", "--all", "--json"])
    assert code == 0
    payload = _json.loads(capsys.readouterr().out)
    assert len(payload) == len(default_registry().keys())
    assert all({"character", "consistent", "issues"} <= set(row) for row in payload)


def test_character_cli_memory_is_honest_when_empty(capsys):
    from cli.characters import main

    code = main(["memory", "vision"])
    assert code == 0
    assert "vision" in capsys.readouterr().out


def test_character_cli_memory_reads_a_learned_fact(capsys):
    CharacterLearning("vision").add_fact("porosity is the void fraction")
    from cli.characters import main

    code = main(["memory", "vision", "--search", "porosity"])
    assert code == 0
    assert "porosity" in capsys.readouterr().out.lower()


def test_character_cli_evaluate_scores_the_eight(capsys):
    from cli.characters import main

    code = main(["evaluate"])
    assert code == 0
    out = capsys.readouterr().out
    assert "8 characters" in out
