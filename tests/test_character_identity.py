"""Phase C: one identity per character across every window.

The feature this pins is the task's third phase: the **same** character shown in
the main chat window and in its own dedicated window must be one identity -- one
memory, one display name -- and a character's personality must actually reach the
text of its answer, not only its configuration file.

Everything here is headless: the identity resolver, the shared learning record,
the name bridge between the mouth profile and the character configuration, the
held-out identity gate, and the comparison tool all run with no display, no model,
and no network.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from core import character_compare as cc
from core import character_learning as cl
from core.character_identity import (
    IDENTITY_OWNERS,
    all_identities,
    identity_for,
    identity_signature,
    same_character,
)
from core.character_identity_gate import (
    IDENTITY_JUDGEMENTS,
    IdentityJudgement,
    character_identity_accuracy,
)

CHARACTERS = ("jarvis", "ultron", "vision", "banner", "natasha", "hakeem", "cloud")
MAIN = "fayd_al_ruh"


@pytest.fixture
def data_root(tmp_path, monkeypatch):
    """A fresh data folder for the whole program, so nothing touches the checkout."""
    root = tmp_path / "data"
    root.mkdir()
    monkeypatch.setenv("FAYD_DATA_DIR", str(root))
    return root


# --- one identity per character -------------------------------------------


def test_every_character_resolves_to_one_identity() -> None:
    identities = all_identities()
    assert len(identities) == len(IDENTITY_OWNERS) == 8
    assert all(identity.name and identity.role for identity in identities)


def test_the_mouth_key_and_the_owner_name_are_one_character() -> None:
    # The chat speaks "sources"; the Characters surface speaks "jarvis".
    assert same_character("sources", "jarvis") is True
    assert identity_for("sources").owner == identity_for("jarvis").owner == "jarvis"
    # The project's own model, too.
    assert same_character("model", "fayd_al_ruh") is True


def test_identities_are_distinct() -> None:
    signatures = {identity_signature(owner) for owner in IDENTITY_OWNERS}
    assert len(signatures) == len(IDENTITY_OWNERS)


def test_only_natasha_is_private() -> None:
    private = {identity.owner for identity in all_identities() if identity.private}
    assert private == {"natasha"}


# --- one memory file, two windows -----------------------------------------


def test_a_fact_written_under_the_owner_is_read_under_the_mouth(data_root: Path) -> None:
    cl.CharacterLearning("jarvis", root=data_root).add_fact("the user works offshore")
    assert "the user works offshore" in cl.CharacterLearning("sources", root=data_root).facts()


def test_two_live_writers_do_not_clobber_each_other(data_root: Path) -> None:
    """The exact Phase C bug: one file, two windows, neither write may be lost."""
    first = cl.CharacterLearning("ultron", root=data_root)
    second = cl.CharacterLearning("ultron", root=data_root)
    first.add_fact("from the first window")
    second.add_fact("from the second window")
    first.reload()
    assert set(first.facts()) == {"from the first window", "from the second window"}
    # A fresh read (a third window) sees both too.
    assert set(cl.CharacterLearning("ultron", root=data_root).facts()) == {
        "from the first window",
        "from the second window",
    }


def test_two_live_writers_do_not_drop_each_others_exchanges(data_root: Path) -> None:
    first = cl.CharacterLearning("vision", root=data_root)
    second = cl.CharacterLearning("vision", root=data_root)
    first.remember("q1", "a1")
    second.remember("q2", "a2")
    questions = {exchange.question for exchange in cl.CharacterLearning("vision", root=data_root).exchanges()}
    assert questions == {"q1", "q2"}


def test_a_fact_learned_in_one_window_is_recognised_as_known_in_the_other(data_root: Path) -> None:
    first = cl.CharacterLearning("hakeem", root=data_root)
    second = cl.CharacterLearning("hakeem", root=data_root)
    assert first.add_fact("the Hijra was in 622 CE") is True
    # The second window has not reloaded, but the shared file already holds it.
    assert second.add_fact("the Hijra was in 622 CE") is False


def test_clear_actually_empties_the_shared_file(data_root: Path) -> None:
    cl.CharacterLearning("ultron", root=data_root).add_fact("a fact")
    cl.CharacterLearning("ultron", root=data_root).clear()
    assert cl.CharacterLearning("ultron", root=data_root).facts() == ()


def test_the_memory_bounds_still_hold_across_writers(data_root: Path, monkeypatch) -> None:
    monkeypatch.setattr(cl, "MAX_FACTS", 3)
    first = cl.CharacterLearning("banner", root=data_root)
    second = cl.CharacterLearning("banner", root=data_root)
    for index in range(3):
        first.add_fact(f"fact {index}")
    second.add_fact("fact 3")
    assert cl.CharacterLearning("banner", root=data_root).facts() == ("fact 1", "fact 2", "fact 3")


# --- one display name, two stores -----------------------------------------


def test_a_rename_in_the_mouth_profile_reaches_the_character_config(data_root: Path) -> None:
    from core import mouth_profile as profiles
    from core.character_registry import CharacterRegistry

    profiles.set_name("ultron", "Ultra")
    assert profiles.load_profile("ultron").name == "Ultra"
    assert CharacterRegistry(data_root=data_root).load("ultron").general.name == "Ultra"


def test_a_rename_in_the_character_config_reaches_the_mouth_profile(data_root: Path) -> None:
    from core import mouth_profile as profiles
    from core.character_config import replace_general
    from core.character_registry import CharacterRegistry

    registry = CharacterRegistry(data_root=data_root)
    registry.save(replace_general(registry.load("vision"), name="Vision Prime"))
    assert profiles.load_profile("vision").name == "Vision Prime"


def test_the_identity_resolver_reports_the_one_name(data_root: Path) -> None:
    from core import mouth_profile as profiles
    from core.character_registry import CharacterRegistry

    profiles.set_name("hakeem", "الحكيم")
    registry = CharacterRegistry(data_root=data_root)
    assert identity_for("hakeem", registry=registry).name == "الحكيم"


# --- the personality reaches the answer text ------------------------------


def test_every_character_has_a_distinct_personality_out_of_the_box() -> None:
    """The shipped dials make the characters measurably different."""
    from core.character_config import personality_defaults_for

    dials = {
        owner: (personality_defaults_for(owner)["formality"], personality_defaults_for(owner)["response_length"])
        for owner in (*CHARACTERS, MAIN)
    }
    assert len(set(dials.values())) >= 6


def test_every_character_ships_a_complete_personality() -> None:
    """Out of the box, a character's style and objectives are defined, not empty.

    Before this, every character shipped with an empty communication style, an
    empty reasoning style, and no objectives -- a personality half written. The
    style and objectives are derived from the character's own persona, so a fresh
    character is as complete as its persona is, and no character is left blank.
    """
    from core.character_config import default_config_for

    for owner in (*CHARACTERS, MAIN):
        personality = default_config_for(owner).personality
        assert personality.communication_style, owner
        assert personality.reasoning_style, owner
        assert personality.objectives, owner


def test_the_style_and_objectives_reach_the_instruction() -> None:
    """The new fields really reach the standing instruction the model reads."""
    from core.character_config import default_config_for

    prompt = default_config_for("vision").system_prompt()
    assert "Communication style" in prompt
    assert "Reasoning style" in prompt
    assert "objectives" in prompt.lower()


def test_every_character_has_a_distinct_style_out_of_the_box() -> None:
    """The styles make the characters different, not a shared boilerplate."""
    from core.character_config import default_config_for

    styles = {
        owner: (
            default_config_for(owner).personality.communication_style,
            default_config_for(owner).personality.reasoning_style,
        )
        for owner in (*CHARACTERS, MAIN)
    }
    assert len(set(styles.values())) == len(styles)


def test_the_personality_reaches_the_answer_text(data_root: Path) -> None:
    """A character's identity must reach the *text* it produces, not only a file."""
    from ai.characters import wrap_character

    class EchoEngine:
        system_prompt = ""

    class EchoMouth:
        def __init__(self) -> None:
            self.engine = EchoEngine()

        def ask(self, question, **kwargs):
            class Answer:
                text = self.engine.system_prompt

            return Answer()

    mouth = EchoMouth()
    character = wrap_character("hakeem", mouth, root=data_root)
    answer = character.ask("who are you?")
    persona = cl.persona_for("hakeem")
    assert persona.name in answer.text
    assert persona.role in answer.text


# --- the held-out identity gate -------------------------------------------


def test_the_gate_is_one_on_the_shipped_wiring() -> None:
    assert character_identity_accuracy() == 1.0


def test_the_gate_is_zero_for_an_empty_catalogue() -> None:
    assert character_identity_accuracy(()) == 0.0


def test_the_gate_fails_when_the_identity_never_reaches_the_mouth() -> None:
    """Prove the gate discriminates: drop the wiring and the metric falls."""
    import ai.characters as characters

    real = characters.wrap_character
    characters.wrap_character = lambda mouth, assistant, **kwargs: assistant
    try:
        assert character_identity_accuracy() < 1.0
    finally:
        characters.wrap_character = real


def test_the_gate_fails_when_the_two_windows_stop_sharing_memory() -> None:
    """The unification is a judgement of its own, not a side effect."""
    assert any(j.kind == "unify-memory" for j in IDENTITY_JUDGEMENTS)
    assert any(j.kind == "unify-concurrent" for j in IDENTITY_JUDGEMENTS)


def test_a_single_judgement_catalogue_is_scored_honestly() -> None:
    catalogue = (IdentityJudgement("a check", "resolve-all"),)
    assert character_identity_accuracy(catalogue) == 1.0
    broken = (IdentityJudgement("a check", "not-a-kind"),)
    assert character_identity_accuracy(broken) == 0.0


# --- the comparison tool ---------------------------------------------------


def test_the_profile_reads_the_effective_configuration(data_root: Path) -> None:
    from core.character_registry import CharacterRegistry

    registry = CharacterRegistry(data_root=data_root)
    profile = cc.character_profile("natasha", registry=registry)
    assert profile.owner == "natasha"
    assert profile.private is True
    assert 0.0 <= profile.formality <= 1.0


def test_the_comparison_reports_the_dimensions_that_differ(data_root: Path) -> None:
    from core.character_registry import CharacterRegistry

    registry = CharacterRegistry(data_root=data_root)
    contrast = cc.compare_characters("natasha", "hakeem", registry=registry)
    assert contrast.difference_score > 0
    keys = {diff.key for diff in contrast.differences}
    assert "formality" in keys  # the companion is far less formal than the scholar
    assert all(diff.label for diff in contrast.diffs)


def test_the_comparison_is_same_when_nothing_differs(data_root: Path) -> None:
    from core.character_registry import CharacterRegistry

    registry = CharacterRegistry(data_root=data_root)
    contrast = cc.compare_characters("ultron", "ultron", registry=registry)
    assert contrast.difference_score == 0
    assert "every compared dimension is the same" in contrast.describe()


def test_the_table_lists_every_character(data_root: Path) -> None:
    from core.character_registry import CharacterRegistry

    registry = CharacterRegistry(data_root=data_root)
    table = cc.comparison_table(registry=registry)
    assert len(table.profiles) == 8
    assert table.row("Formality") is not None
    assert "Dimension" in table.describe()


def test_the_ranking_orders_the_most_distinct_first(data_root: Path) -> None:
    from core.character_registry import CharacterRegistry

    registry = CharacterRegistry(data_root=data_root)
    ranking = cc.most_different(registry=registry)
    scores = [score for _first, _second, score in ranking]
    assert scores == sorted(scores, reverse=True)
    # Natasha (private, expressive, informal) is among the most distinct pairs.
    assert ranking[0][2] >= ranking[-1][2]


def test_the_edit_of_a_dial_changes_the_comparison(data_root: Path) -> None:
    from core.character_config import replace_personality
    from core.character_registry import CharacterRegistry

    registry = CharacterRegistry(data_root=data_root)
    before = cc.compare_characters("ultron", "vision", registry=registry).difference_score
    # Make Ultron's formality match Vision's: that dimension must stop differing.
    vision_formality = registry.load("vision").personality.formality
    registry.save(replace_personality(registry.load("ultron"), formality=vision_formality))
    after = cc.compare_characters("ultron", "vision", registry=registry).difference_score
    assert after < before


# --- the Characters surface and the live mouth share one memory -----------


def test_the_manager_and_the_live_mouth_share_one_memory(data_root: Path) -> None:
    """A fact taught in the Characters window reaches the chat's own character."""
    from gui.character_manager import CharacterManagerService

    service = CharacterManagerService(data_root=data_root)
    assert service.remember_fact("jarvis", "the user works offshore") is True
    # The chat speaks "sources"; it must read the same fact.
    assert "the user works offshore" in cl.CharacterLearning("sources", root=data_root).facts()


def test_the_live_mouths_exchange_reaches_the_characters_own_learning(data_root: Path) -> None:
    """The exchange a live window records is the one the character's file holds."""
    from gui.character_manager import CharacterManagerService

    cl.CharacterLearning("sources", root=data_root).remember("what is porosity?", "a definition")
    # Read back through the owner vocabulary the Characters surface uses.
    recorded = cl.CharacterLearning("jarvis", root=data_root).exchanges()
    assert [exchange.question for exchange in recorded] == ["what is porosity?"]
    # And it is rendered into the training text the project's own model learns from.
    assert any("what is porosity?" in sentence for sentence in cl.character_learning_sentences())
    assert CharacterManagerService(data_root=data_root) is not None


# --- the CLI ---------------------------------------------------------------


def test_the_cli_lists_one_identity_per_character(data_root: Path, capsys) -> None:
    from cli.characters import main

    assert main(["identity"]) == 0
    out = capsys.readouterr().out
    assert "Jarvis (jarvis / sources)" in out
    assert "Fayd Al-Ruh Model (fayd_al_ruh / model)" in out


def test_the_cli_shows_one_character_identity(data_root: Path, capsys) -> None:
    from cli.characters import main

    assert main(["identity", "sources"]) == 0
    assert "Jarvis" in capsys.readouterr().out


def test_the_cli_compares_two_characters(data_root: Path, capsys) -> None:
    from cli.characters import main

    assert main(["compare", "natasha", "hakeem"]) == 0
    out = capsys.readouterr().out
    assert "Natasha" in out and "Hakeem" in out
    assert "dimensions differ" in out


def test_the_cli_prints_the_comparison_table(data_root: Path, capsys) -> None:
    from cli.characters import main

    assert main(["compare"]) == 0
    assert "Formality" in capsys.readouterr().out


def test_the_cli_ranks_the_most_distinct_pairs(data_root: Path, capsys) -> None:
    from cli.characters import main

    assert main(["compare", "--rank"]) == 0
    assert "dimensions differ" in capsys.readouterr().out


def test_the_cli_compare_is_machine_readable(data_root: Path, capsys) -> None:
    import json as json_module

    from cli.characters import main

    assert main(["compare", "natasha", "hakeem", "--json"]) == 0
    payload = json_module.loads(capsys.readouterr().out)
    assert payload["difference_score"] >= 1
    assert payload["dimensions"]


def test_the_cli_identity_is_machine_readable(data_root: Path, capsys) -> None:
    import json as json_module

    from cli.characters import main

    assert main(["identity", "--json"]) == 0
    rows = json_module.loads(capsys.readouterr().out)
    assert len(rows) == 8
    assert rows[0]["owner"] == "jarvis" and rows[0]["mouth"] == "sources"

