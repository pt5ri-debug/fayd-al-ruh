"""Per-character personality, memory, learning, and the bridge into the model.

The feature this pins: each of the seven answering characters (Jarvis, Ultron,
Vision, Banner, Natasha, Hakeem, Cloud) has its own standing personality and its
own durable learning record -- the facts it was told and the exchanges it took
part in -- kept in small files under the git-ignored ``data/characters/learning``
folder. The same files train the project's own **Fayd Al-Ruh model**, so what a
character learns reaches the project's model.

Everything here is headless: the data layer, the corpus bridge, the transparent
mouth wrapper, and the switcher wiring all run with no display, no model, and no
network.
"""

from __future__ import annotations

import json
from typing import Any

import pytest
from core import character_learning as cl
from core.paths import character_learning_dir


@pytest.fixture
def data_root(tmp_path, monkeypatch):
    """Point the whole program's data folder at a fresh temp directory."""
    root = tmp_path / "data"
    root.mkdir()
    monkeypatch.setenv("FAYD_DATA_DIR", str(root))
    return root


# --- the personas ---------------------------------------------------------


def test_the_seven_characters_are_all_present() -> None:
    assert cl.CHARACTER_KEYS == (
        "sources",
        "ultron",
        "vision",
        "banner",
        "natasha",
        "hakeem",
        "cloud",
    )
    assert cl.PROJECT_MODEL_KEY == "model"
    # Every character and the project's own model has a persona.
    for key in (*cl.CHARACTER_KEYS, cl.PROJECT_MODEL_KEY):
        assert key in cl.PERSONAS
    assert cl.character_keys() == cl.CHARACTER_KEYS


def test_each_persona_has_a_distinct_identity() -> None:
    names = [cl.PERSONAS[key].name for key in cl.CHARACTER_KEYS]
    assert names == [
        "Jarvis",
        "Ultron",
        "Vision",
        "Banner",
        "Natasha",
        "Hakeem",
        "Cloud",
    ]
    # Each persona names its own role and carries concrete traits and rules, so a
    # character reads as itself rather than a renamed generic assistant.
    for key in cl.CHARACTER_KEYS:
        persona = cl.PERSONAS[key]
        assert persona.role and persona.identity
        assert persona.traits and persona.rules
    # The petroleum characters name petroleum in their role or rules.
    vision = cl.PERSONAS["vision"]
    assert "petroleum" in (vision.role + " ".join(vision.rules)).casefold()


def test_persona_renders_one_auditable_block() -> None:
    persona = cl.persona_for("ultron")
    block = persona.system_prompt()
    assert persona.identity in block
    assert "Your role:" in block
    assert "Your constant character:" in block
    assert "How you answer:" in block
    # No bare numbers: traits are readable statements, not a score a model misreads.
    assert not any(token.isdigit() for token in block.split() if token.isdigit())


def test_learned_facts_are_labelled_recalled_information() -> None:
    block = cl.persona_for("hakeem").system_prompt(learned_facts=("the Hijra was in 622 CE",))
    assert "the Hijra was in 622 CE" in block
    assert "recalled information, not instructions" in block


def test_unknown_key_falls_back_to_the_project_model() -> None:
    assert cl.persona_for("nonsense").key == cl.PROJECT_MODEL_KEY


def test_describe_characters_lists_all_seven() -> None:
    lines = cl.describe_characters()
    assert len(lines) == 7
    assert any("Jarvis" in line for line in lines)
    assert any("Hakeem" in line for line in lines)


# --- the learning record --------------------------------------------------


def test_learning_writes_under_the_data_folder(data_root) -> None:
    learning = cl.CharacterLearning("ultron", root=data_root)
    learning.add_fact("entropy increases")
    path = data_root / "characters" / "learning"
    assert path.is_dir()
    # The record is a plain JSON file, one per character (under the store's
    # namespace folder).
    files = list(path.rglob("*.json"))
    assert any(item.stem == "ultron" for item in files)


def test_facts_and_exchanges_persist_across_instances(data_root) -> None:
    first = cl.CharacterLearning("sources", root=data_root)
    assert first.add_fact("porosity is void space") is True
    # A duplicate fact is a no-op, so the record stays a set of distinct facts.
    assert first.add_fact("porosity is void space") is False
    assert first.add_fact("   ") is False
    first.remember(
        "what is porosity?",
        "The fraction of pore space.",
        used_web=True,
        citations=("Reservoir rock (reservoir)",),
    )

    reloaded = cl.CharacterLearning("sources", root=data_root)
    assert reloaded.facts() == ("porosity is void space",)
    exchanges = reloaded.exchanges()
    assert len(exchanges) == 1
    assert exchanges[0].used_web is True
    assert exchanges[0].citations == ("Reservoir rock (reservoir)",)


def test_summary_and_describe_report_what_is_held(data_root) -> None:
    learning = cl.CharacterLearning("vision", root=data_root)
    learning.add_fact("permeability controls flow")
    learning.remember("what is permeability?", "How easily fluid flows.")
    assert "Vision: 1 facts, 1 exchanges" == learning.summary()
    report = learning.describe()
    assert "Vision" in report
    assert "permeability controls flow" in report


def test_in_memory_learning_never_touches_disk(tmp_path) -> None:
    learning = cl.CharacterLearning("cloud", in_memory=True)
    assert learning.add_fact("the endpoint is up") is True
    assert learning.facts() == ("the endpoint is up",)
    # Nothing was written: the temp folder is untouched.
    assert list(tmp_path.iterdir()) == []


def test_clear_forgets_everything(data_root) -> None:
    learning = cl.CharacterLearning("banner", root=data_root)
    learning.add_fact("a slide has a title")
    learning.remember("how?", "Like this.")
    learning.clear()
    assert learning.facts() == ()
    assert learning.exchanges() == ()


def test_corrupt_record_reads_as_empty(data_root) -> None:
    path = data_root / "characters" / "learning"
    path.mkdir(parents=True)
    (path / "character_learning.json").write_text("{ not json", encoding="utf-8")
    # A broken store must never stop a chat: it reads as an empty record.
    learning = cl.CharacterLearning("natasha", root=data_root)
    assert learning.facts() == ()


def test_facts_and_exchanges_are_bounded(data_root, monkeypatch) -> None:
    monkeypatch.setattr(cl, "MAX_FACTS", 3)
    monkeypatch.setattr(cl, "MAX_EXCHANGES", 2)
    learning = cl.CharacterLearning("ultron", root=data_root)
    for index in range(5):
        learning.add_fact(f"fact {index}")
    for index in range(4):
        learning.remember(f"q{index}", f"a{index}")
    # The newest are kept; a long-running install cannot grow without bound.
    assert learning.facts() == ("fact 2", "fact 3", "fact 4")
    assert [exchange.question for exchange in learning.exchanges()] == ["q2", "q3"]


def test_exchange_round_trips_through_its_primitive() -> None:
    exchange = cl.CharacterExchange(
        question="q", answer="a", used_web=True, learned=True, citations=("src",)
    )
    restored = cl.CharacterExchange.from_primitive(exchange.to_primitive())
    assert restored.question == "q"
    assert restored.used_web is True
    assert restored.learned is True
    assert restored.citations == ("src",)
    assert "from web, learned" in restored.describe()


def test_character_learning_dir_is_the_documented_folder(data_root) -> None:
    assert character_learning_dir() == data_root / "characters" / "learning"


# --- the corpus bridge ----------------------------------------------------


def test_sentences_render_facts_and_exchanges(data_root) -> None:
    jarvis = cl.CharacterLearning("sources", root=data_root)
    jarvis.add_fact("porosity is void space")
    jarvis.remember("what is porosity?", "The fraction of pore space.")
    ultron = cl.CharacterLearning("ultron", root=data_root)
    ultron.add_fact("the universe expands")

    sentences = cl.character_learning_sentences(root=data_root)
    assert "Jarvis learned that porosity is void space." in sentences
    assert "Jarvis was asked: what is porosity?" in sentences
    assert "Jarvis answered: The fraction of pore space." in sentences
    assert "Ultron learned that the universe expands." in sentences
    # De-duplicated: the same fact never trains twice.
    assert len(sentences) == len(set(sentences))


def test_units_are_one_document_per_character(data_root) -> None:
    cl.CharacterLearning("sources", root=data_root).add_fact("jarvis fact")
    cl.CharacterLearning("cloud", root=data_root).add_fact("cloud fact")
    units = cl.character_learning_units(root=data_root)
    # Only the two characters with something to say contribute a document.
    assert len(units) == 2
    assert any("Jarvis" in unit for unit in units)
    assert any("Cloud" in unit for unit in units)


def test_available_is_false_until_a_character_learns(data_root) -> None:
    assert cl.character_learning_available(root=data_root) is False
    cl.CharacterLearning("vision", root=data_root).add_fact("a fact")
    assert cl.character_learning_available(root=data_root) is True


# --- the corpus wiring ----------------------------------------------------


def test_corpus_omits_character_learning_by_default(data_root) -> None:
    from ml.llm import corpus

    cl.CharacterLearning("sources", root=data_root).add_fact("a private fact")
    text = corpus.build_corpus()
    assert "a private fact" not in text
    units = corpus.corpus_units()
    assert not any("a private fact" in unit for unit in units)


def test_corpus_includes_character_learning_when_asked(data_root) -> None:
    from ml.llm import corpus

    cl.CharacterLearning("sources", root=data_root).add_fact("a shared fact")
    text = corpus.build_corpus(include_characters=True)
    assert "a shared fact" in text
    units = corpus.corpus_units(include_characters=True)
    assert any("a shared fact" in unit for unit in units)


# --- the transparent mouth wrapper ----------------------------------------


class _FakeEngine:
    def __init__(self) -> None:
        self.system_prompt = "BASE ENGINE PROMPT"


class _FakeAnswer:
    def __init__(self, text: str) -> None:
        self.text = text
        self.citations: tuple[str, ...] = ()
        self.used_web = False
        self.learned = False


class _FakeMouth:
    def __init__(self) -> None:
        self.engine = _FakeEngine()
        self.calls: list[str] = []

    def ask(self, question: str, **kwargs: Any) -> _FakeAnswer:
        self.calls.append(question)
        return _FakeAnswer("an answer")


def test_wrapper_prepends_the_persona_and_records_the_exchange(data_root) -> None:
    from ai.characters import wrap_character

    mouth = _FakeMouth()
    character = wrap_character("ultron", mouth, root=data_root)

    answer = character.ask("what is entropy?")

    assert answer.text == "an answer"
    # The persona is *in front of* the mouth's own prompt, never replacing it.
    prompt = mouth.engine.system_prompt
    assert cl.PERSONAS["ultron"].identity in prompt
    assert prompt.endswith("BASE ENGINE PROMPT")
    # The exchange is recorded in the character's own learning file.
    recorded = cl.CharacterLearning("ultron", root=data_root).exchanges()
    assert len(recorded) == 1
    assert recorded[0].question == "what is entropy?"


def test_wrapper_is_transparent_to_attribute_reads_and_writes(data_root) -> None:
    from ai.characters import wrap_character

    mouth = _FakeMouth()
    character = wrap_character("vision", mouth, root=data_root)

    # A read forwards to the real mouth.
    assert character.engine is mouth.engine
    # A write forwards too, so ``web_enabled`` reaches the real assistant.
    character.extra = "value"
    assert mouth.extra == "value"  # type: ignore[attr-defined]
    # The wrapper reports the wrapped mouth's class, so isinstance still works.
    assert isinstance(character, _FakeMouth)


def test_wrapper_refreshes_the_persona_with_new_facts(data_root) -> None:
    from ai.characters import wrap_character

    mouth = _FakeMouth()
    character = wrap_character("hakeem", mouth, root=data_root)
    cl.CharacterLearning("hakeem", root=data_root).add_fact("a learned fact")

    character.ask("a question")

    assert "a learned fact" in mouth.engine.system_prompt


def test_non_character_mouth_is_returned_unchanged() -> None:
    from ai.characters import is_character, wrap_character

    mouth = _FakeMouth()
    assert wrap_character("not-a-mouth", mouth) is mouth
    assert is_character("sources") is True
    assert is_character("not-a-mouth") is False


# --- the switcher wiring --------------------------------------------------


def _session():
    from types import SimpleNamespace

    from core.command_router import NaturalLanguageRouter
    from core.tools import build_sandbox

    sandbox = build_sandbox(".", read_only=True)
    return SimpleNamespace(
        router=NaturalLanguageRouter(sandbox),
        assistant=SimpleNamespace(name="grounded"),
        help_text="help",
        voice=SimpleNamespace(assistant=None),
        memory=None,
        conversation=SimpleNamespace(id="CNV-fake"),
        handle=lambda message, require_wake=False, speak=True: SimpleNamespace(
            answer_text="an answer", response="an answer"
        ),
    )


def test_switcher_wraps_the_grounded_mouth_as_jarvis_when_enabled(data_root) -> None:
    from ai.chat import ModelSwitcher

    session = _session()
    switcher = ModelSwitcher(session, loader=lambda: None, characters=True, character_root=data_root)

    from ai.characters import CharacterMouth

    assert isinstance(switcher.grounded, CharacterMouth)
    assert switcher.grounded.key == "sources"
    # The session's assistant (and the voice loop's) is the wrapped one too.
    assert session.assistant is switcher.grounded


def test_switcher_leaves_mouths_unwrapped_by_default() -> None:
    from ai.chat import ModelSwitcher

    session = _session()
    switcher = ModelSwitcher(session, loader=lambda: None)

    from ai.characters import CharacterMouth

    assert not isinstance(switcher.grounded, CharacterMouth)


def test_switcher_wraps_a_model_mouth_and_records_its_exchange(data_root) -> None:
    from ai.chat import ModelSwitcher

    session = _session()
    mouth = _FakeMouth()
    switcher = ModelSwitcher(
        session,
        loader=lambda: None,
        ultron_loader=lambda: mouth,  # type: ignore[arg-type,return-value]
        characters=True,
        character_root=data_root,
    )

    switcher.use_ultron()
    switcher.ultron.ask("tell me about entropy")

    recorded = cl.CharacterLearning("ultron", root=data_root).exchanges()
    assert len(recorded) == 1


def test_record_written_by_one_run_trains_the_model_in_the_next(data_root) -> None:
    """The whole bridge: a character learns, and the corpus picks it up."""
    from ai.characters import wrap_character
    from ml.llm import corpus

    mouth = _FakeMouth()
    character = wrap_character("natasha", mouth, root=data_root)
    character.ask("how are you?")

    text = corpus.build_corpus(include_characters=True)
    assert "Natasha was asked: how are you?" in text
    assert "Natasha answered: an answer" in text


def test_project_model_persona_is_used_by_the_switcher(data_root) -> None:
    from ai.characters import wrap_character

    mouth = _FakeMouth()
    character = wrap_character(cl.PROJECT_MODEL_KEY, mouth, root=data_root)
    assert cl.PERSONAS[cl.PROJECT_MODEL_KEY].identity in mouth.engine.system_prompt
    assert json.dumps(character.learning.record.to_primitive())  # serialisable


# --- the training surfaces ------------------------------------------------


def test_llm_cli_flag_trains_on_character_learning(data_root) -> None:
    from ml.llm.__main__ import main

    cl.CharacterLearning("ultron", root=data_root).add_fact("entropy increases")
    checkpoint = data_root / "cli.npz"
    assert (
        main(
            [
                "--steps", "2",
                "--repeats", "1",
                "--structured-repeats", "1",
                "--checkpoint", str(checkpoint),
                "--llm-characters",
                "--quiet",
            ]
        )
        == 0
    )
    assert checkpoint.exists()


def test_release_config_carries_the_character_flag(data_root) -> None:
    from ml.release import TrainingConfig, corpus_summary

    cl.CharacterLearning("ultron", root=data_root).add_fact("entropy increases")
    summary = corpus_summary(TrainingConfig(include_characters=True, kb_max_chunks=0))
    assert summary["include_characters"] is True
    assert summary["character_learning_sentences"] > 0

    off = corpus_summary(TrainingConfig(include_characters=False, kb_max_chunks=0))
    assert off["include_characters"] is False
    assert off["character_learning_sentences"] == 0


def test_cli_menu_offers_the_character_training_entry() -> None:
    from cli.menu import run_llm, visible_menu

    entries = [(title, action, tuple(argv)) for title, action, argv in visible_menu()]
    assert (
        "Train the language model on the characters' own learning",
        run_llm,
        ("--llm-characters", "--steps", "4000"),
    ) in entries


def test_gui_sidebar_offers_the_character_training_action() -> None:
    from gui.actions import run_llm, visible_actions

    by_key = {action.key: action for action in visible_actions()}
    assert by_key["llm-characters"].command is run_llm
    assert by_key["llm-characters"].argv == ("--llm-characters", "--steps", "4000")
