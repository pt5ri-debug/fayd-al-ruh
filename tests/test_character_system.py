"""Independent character learning, isolation, and controlled knowledge sharing.

The feature this pins is the task's own: seven independent characters (Jarvis,
Ultron, Vision, Banner, Natasha, Hakeem, Cloud) plus the project's own **Fayd
Al-Ruh model**, each with its **own isolated learning store**, and a controlled
knowledge-sharing interface through which the main model may learn from six of
them -- **never from Natasha**.

Everything here is headless: the isolation policy, the per-character store, the
learning pipeline, the sharing gate, the configuration registry, and the
management service all run with no display, no model, and no network.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from core import character_config as cc
from core import learning_isolation as iso
from core.character_learning_store import CharacterLearningStore, main_model_store
from core.knowledge_sharing import KnowledgeSharing
from core.learning_pipeline import LearningPipeline
from core.learning_records import (
    SOURCE_CHARACTER,
    SOURCE_INTERNET,
    SOURCE_LANGUAGE_MODEL,
    STATUS_CONTRADICTED,
    STATUS_PROVISIONAL,
    STATUS_REJECTED,
    STATUS_VERIFIED,
)

#: The seven characters, and the project's own model.
CHARACTERS = ("jarvis", "ultron", "vision", "banner", "natasha", "hakeem", "cloud")
MAIN = "fayd_al_ruh"


@pytest.fixture
def data_root(tmp_path, monkeypatch):
    """A fresh data folder for the whole program, so nothing touches the checkout."""
    root = tmp_path / "data"
    root.mkdir()
    monkeypatch.setenv("FAYD_DATA_DIR", str(root))
    return root


def _store(owner: str, root: Path) -> CharacterLearningStore:
    return CharacterLearningStore(owner, root=root)


def _learn(owner: str, root: Path, text: str, *, source: str = SOURCE_LANGUAGE_MODEL) -> None:
    """Put one statement into an owner's own store, the way the pipeline would."""
    pipeline = LearningPipeline(owner, store=_store(owner, root))
    pipeline.learn(source_type=source, question="q", answer=text, source_reference="a test")


def _internet_pipeline(owner: str, root: Path) -> LearningPipeline:
    """A pipeline whose settings allow learning from the internet."""
    config = cc.replace_learning(cc.default_config_for(owner), learn_from_internet=True)
    return LearningPipeline(owner, store=_store(owner, root), config=config)


# ==========================================================================
# A. Learning isolation
# ==========================================================================


def test_the_seven_characters_and_the_main_model_are_all_known() -> None:
    assert set(iso.CHARACTER_OWNERS) == set(CHARACTERS)
    assert iso.MAIN_MODEL_OWNER == MAIN
    for key in (*CHARACTERS, MAIN):
        assert iso.is_owner(key)


def test_owner_names_are_normalised_from_either_vocabulary() -> None:
    # Jarvis's chat mouth key is ``sources``; the owner is ``jarvis``.
    assert iso.owner_for("sources") == "jarvis"
    assert iso.owner_for("jarvis") == "jarvis"
    assert iso.owner_for("model") == MAIN
    assert iso.owner_for(MAIN) == MAIN


def test_each_character_has_its_own_separate_learning_store(data_root: Path) -> None:
    paths = {}
    for key in CHARACTERS:
        store = _store(key, data_root)
        _learn(key, data_root, f"{key} knows something true.")
        paths[key] = store.root
    # Seven distinct directories, none shared.
    assert len(set(paths.values())) == len(CHARACTERS)
    # And each store holds only its own record.
    for key in CHARACTERS:
        records = _store(key, data_root).all()
        assert len(records) == 1
        assert records[0].owner == key
        assert key in records[0].knowledge


def test_the_main_model_has_its_own_learning_store(data_root: Path) -> None:
    main = main_model_store(root=data_root)
    _learn(MAIN, data_root, "The main model learns on its own.")
    assert main.root != _store("jarvis", data_root).root
    assert len(main.all()) == 1


def test_a_character_cannot_write_a_record_owned_by_another(data_root: Path) -> None:
    from core.learning_records import new_record

    jarvis = _store("jarvis", data_root)
    record = new_record("ultron", source_type=SOURCE_LANGUAGE_MODEL, answer="not mine")
    with pytest.raises(iso.LearningAccessError):
        jarvis.append(record)
    # Nothing was written into Jarvis's store, and the refusal was logged.
    assert jarvis.all() == ()


def test_a_refused_cross_character_write_is_logged(data_root: Path) -> None:
    from core.learning_audit import LearningAudit
    from core.learning_records import new_record

    audit = LearningAudit(data_root / "audit")
    jarvis = CharacterLearningStore("jarvis", root=data_root, audit=audit)
    with pytest.raises(iso.LearningAccessError):
        jarvis.append(new_record("ultron", source_type=SOURCE_LANGUAGE_MODEL, answer="x"))
    denials = [event for event in audit.events() if event.kind == "denied"]
    assert len(denials) == 1
    assert denials[0].actor == "jarvis"
    assert denials[0].owner == "ultron"


def test_the_store_path_is_the_policys_own_layout(data_root: Path) -> None:
    for key in CHARACTERS:
        expected = data_root / "learning" / MAIN / "characters" / key
        assert _store(key, data_root).root == expected
    # The main model has its own sibling folder, not a character's.
    assert main_model_store(root=data_root).root == data_root / "learning" / MAIN / "main"


def test_learning_records_remain_independent_after_a_restart(data_root: Path) -> None:
    _learn("jarvis", data_root, "Jarvis learned this before the restart.")
    _learn("ultron", data_root, "Ultron learned something else.")
    # A *fresh* store object reads what a previous process wrote.
    jarvis = _store("jarvis", data_root)
    ultron = _store("ultron", data_root)
    assert len(jarvis.all()) == 1
    assert len(ultron.all()) == 1
    assert "restart" in jarvis.all()[0].knowledge
    assert "else" in ultron.all()[0].knowledge


def test_an_unknown_owner_is_refused() -> None:
    with pytest.raises(iso.LearningAccessError):
        CharacterLearningStore("not-a-character")


def test_ultron_cannot_read_visions_private_records(data_root: Path) -> None:
    """A character's own store is the only one it can open."""
    _learn("vision", data_root, "Vision knows a petroleum fact.")
    # Ultron's store is its own; it has no API to open Vision's, and the policy
    # refuses a read from another character.
    with pytest.raises(iso.LearningAccessError):
        iso.assert_read("ultron", "vision")
    assert _store("ultron", data_root).all() == ()


# ==========================================================================
# B. Main-model knowledge sharing (Natasha excluded)
# ==========================================================================


def _seed_shareable(data_root: Path) -> None:
    for key in CHARACTERS:
        # A provisional (trusted) record, so the share has something to import.
        pipeline = LearningPipeline(key, store=_store(key, data_root))
        pipeline.learn_from_user(
            question="q", statement=f"{key} learned a durable fact about the world."
        )


def test_natasha_is_the_only_private_character() -> None:
    assert iso.PRIVATE_OWNERS == frozenset({"natasha"})
    assert set(iso.SHAREABLE_OWNERS) == set(CHARACTERS) - {"natasha"}


@pytest.mark.parametrize("character", ["jarvis", "ultron", "vision", "banner", "hakeem", "cloud"])
def test_the_main_model_can_learn_from_each_permitted_character(
    character: str, data_root: Path
) -> None:
    _seed_shareable(data_root)
    sharing = KnowledgeSharing(data_root=data_root)
    result = sharing.learn_from(character)
    assert result.count >= 1
    assert not result.skipped_private
    # The main model's own store now holds a copy carrying the provenance.
    main = sharing.main_store
    imported = [r for r in main.all() if r.source_character == character]
    assert imported, "the main model should hold a record from this character"
    assert imported[0].source_type == SOURCE_CHARACTER
    assert imported[0].owner == MAIN
    assert imported[0].source_record_id
    assert imported[0].source_date


def test_the_main_model_cannot_retrieve_any_record_from_natasha(data_root: Path) -> None:
    _seed_shareable(data_root)
    sharing = KnowledgeSharing(data_root=data_root)
    result = sharing.learn_from("natasha")
    assert result.skipped_private is True
    assert result.count == 0
    # Nothing of Natasha's reached the main model's store.
    assert all(record.source_character != "natasha" for record in sharing.main_store.all())


def test_a_refused_natasha_share_is_logged(data_root: Path) -> None:
    from core.learning_audit import LearningAudit

    audit = LearningAudit(data_root / "audit")
    sharing = KnowledgeSharing(data_root=data_root, audit=audit)
    sharing.learn_from("natasha")
    denials = [event for event in audit.events() if event.kind == "denied"]
    assert any(event.owner == "natasha" for event in denials)


def test_learn_from_all_never_includes_natasha(data_root: Path) -> None:
    _seed_shareable(data_root)
    sharing = KnowledgeSharing(data_root=data_root)
    report = sharing.learn_from_all()
    assert {result.source_character for result in report.results} == set(iso.SHAREABLE_OWNERS)
    assert "natasha" not in {result.source_character for result in report.results}
    assert all(record.source_character != "natasha" for record in sharing.main_store.all())


def test_sharing_creates_its_own_record_and_never_modifies_the_source(data_root: Path) -> None:
    _seed_shareable(data_root)
    source_before = _store("jarvis", data_root).all()
    sharing = KnowledgeSharing(data_root=data_root)
    sharing.learn_from("jarvis")
    source_after = _store("jarvis", data_root).all()
    # The source store is byte-for-byte what it was: the share only read it.
    assert [r.to_primitive() for r in source_before] == [r.to_primitive() for r in source_after]
    # The main model's copy is a *different* record with its own id.
    imported = [r for r in sharing.main_store.all() if r.source_character == "jarvis"]
    assert imported[0].record_id != source_after[0].record_id
    assert imported[0].source_record_id == source_after[0].record_id


def test_sharing_twice_does_not_duplicate(data_root: Path) -> None:
    _seed_shareable(data_root)
    sharing = KnowledgeSharing(data_root=data_root)
    first = sharing.learn_from("jarvis")
    second = sharing.learn_from("jarvis")
    assert first.count >= 1
    assert second.count == 0
    assert second.skipped_duplicates >= 1


def test_a_shared_record_is_never_born_verified(data_root: Path) -> None:
    _seed_shareable(data_root)
    sharing = KnowledgeSharing(data_root=data_root)
    sharing.learn_from("vision")
    for record in sharing.main_store.all():
        if record.source_character:
            assert record.learning_status == STATUS_PROVISIONAL
            assert record.learning_status != STATUS_VERIFIED


def test_can_read_matches_the_policy() -> None:
    sharing = KnowledgeSharing()
    for character in CHARACTERS:
        assert sharing.can_read(character) == (character != "natasha")
    assert sharing.can_read("not-a-character") is False


def test_a_plain_share_does_not_expose_the_private_exchange(data_root: Path) -> None:
    """The knowledge is shared; the question/answer exchange is not, by default."""
    pipeline = LearningPipeline("jarvis", store=_store("jarvis", data_root))
    pipeline.learn_from_user(
        question="What did the user ask privately?", statement="Porosity is a measure of voids."
    )
    sharing = KnowledgeSharing(data_root=data_root)
    sharing.learn_from("jarvis")
    copied = [r for r in sharing.main_store.all() if r.source_character == "jarvis"]
    assert copied, "the knowledge should be shared"
    # The knowledge survives, but the private exchange was not carried over.
    assert "Porosity" in copied[0].knowledge
    assert copied[0].question == ""
    assert copied[0].answer == ""
    assert copied[0].metadata.get("history_shared") is False


def test_sharing_the_exchange_requires_an_explicit_opt_in(data_root: Path) -> None:
    from core.character_config import replace_learning
    from core.character_registry import CharacterRegistry

    registry = CharacterRegistry(data_root=data_root)
    registry.apply(
        replace_learning(registry.load("banner"), share_conversation_history=True)
    )
    pipeline = LearningPipeline("banner", store=_store("banner", data_root))
    pipeline.learn_from_user(question="The user's question", statement="Banner's knowledge.")
    sharing = KnowledgeSharing(data_root=data_root)
    sharing.learn_from("banner")
    copied = [r for r in sharing.main_store.all() if r.source_character == "banner"]
    assert copied and copied[0].question == "The user's question"
    assert copied[0].metadata.get("history_shared") is True


def test_natasha_never_shares_her_exchange_even_with_the_flag_on(data_root: Path) -> None:
    from core.character_config import replace_learning
    from core.character_registry import CharacterRegistry

    registry = CharacterRegistry(data_root=data_root)
    registry.apply(
        replace_learning(registry.load("natasha"), share_conversation_history=True)
    )
    assert registry.load("natasha").learning.share_conversation_history is True
    # The backend still refuses the read entirely, so nothing is exposed.
    sharing = KnowledgeSharing(data_root=data_root)
    assert sharing.learn_from("natasha").skipped_private is True


# ==========================================================================
# C. Learning sources and the controlled pipeline
# ==========================================================================


def test_a_language_model_answer_is_recorded_with_its_provenance(data_root: Path) -> None:
    pipeline = LearningPipeline("ultron", store=_store("ultron", data_root))
    pipeline.learn_from_model(
        question="What is entropy?",
        answer="Entropy is a measure of disorder.",
        model="qwen2.5",
    )
    record = _store("ultron", data_root).all()[0]
    assert record.source_type == SOURCE_LANGUAGE_MODEL
    assert "qwen2.5" in record.source_reference
    assert record.question == "What is entropy?"
    assert record.answer == "Entropy is a measure of disorder."
    assert record.created_at
    assert record.record_id.startswith("LR-")


def test_an_internet_result_keeps_its_url_as_evidence(data_root: Path) -> None:
    pipeline = _internet_pipeline("vision", data_root)
    pipeline.learn_from_internet(
        question="porosity",
        answer="Porosity is the void fraction of a rock.",
        url="https://example.org/porosity",
    )
    record = _store("vision", data_root).all()[0]
    assert record.source_type == SOURCE_INTERNET
    assert record.source_reference == "https://example.org/porosity"
    assert "https://example.org/porosity" in record.evidence


def test_unverified_information_is_never_automatically_verified(data_root: Path) -> None:
    """A model answer is stored ``provisional`` -- never ``verified``."""
    pipeline = LearningPipeline("banner", store=_store("banner", data_root))
    pipeline.learn_from_model(question="q", answer="A model said something.")
    record = _store("banner", data_root).all()[0]
    assert record.learning_status == STATUS_PROVISIONAL
    assert record.learning_status != STATUS_VERIFIED


def test_an_explicit_verification_is_the_only_way_to_verified(data_root: Path) -> None:
    pipeline = LearningPipeline("banner", store=_store("banner", data_root))
    pipeline.learn_from_model(question="q", answer="A model said something true.")
    record = _store("banner", data_root).all()[0]
    promoted = pipeline.verify(record.record_id)
    assert promoted is not None
    assert promoted.learning_status == STATUS_VERIFIED


def test_contradictory_information_is_preserved_not_overwritten(data_root: Path) -> None:
    pipeline = LearningPipeline("jarvis", store=_store("jarvis", data_root))
    pipeline.learn_from_model(question="q", answer="Waterflood increases recovery.")
    outcome = pipeline.learn_from_model(question="q", answer="Waterflood reduces recovery.")
    assert outcome.contradictions == 1
    records = _store("jarvis", data_root).all()
    # Both are kept: the original is untouched, the new one is marked contradicted.
    assert len(records) == 2
    assert records[0].learning_status == STATUS_PROVISIONAL
    assert records[1].learning_status == STATUS_CONTRADICTED
    assert "contradicts record" in records[1].note
    assert records[0].record_id in records[1].note


def test_a_contradiction_never_rewrites_the_original(data_root: Path) -> None:
    pipeline = LearningPipeline("jarvis", store=_store("jarvis", data_root))
    pipeline.learn_from_model(question="q", answer="Acid increases permeability.")
    original = _store("jarvis", data_root).all()[0].to_primitive()
    pipeline.learn_from_model(question="q", answer="Acid reduces permeability.")
    after = _store("jarvis", data_root).all()[0].to_primitive()
    assert original == after


def test_learning_records_retain_their_evidence_references(data_root: Path) -> None:
    pipeline = _internet_pipeline("hakeem", data_root)
    pipeline.learn_from_internet(
        question="q",
        answer="A historical claim.",
        url="https://example.org/history",
        evidence=("https://example.org/history", "https://example.org/archive"),
    )
    record = _store("hakeem", data_root).all()[0]
    assert "https://example.org/archive" in record.evidence


def test_a_disabled_source_stores_nothing(data_root: Path) -> None:
    config = cc.default_config_for("cloud")
    config = cc.replace_learning(config, learn_from_internet=False)
    pipeline = LearningPipeline("cloud", store=_store("cloud", data_root), config=config)
    outcome = pipeline.learn_from_internet(question="q", answer="from the web", url="https://x")
    assert outcome.count == 0
    assert "off" in outcome.skipped_reason
    assert _store("cloud", data_root).all() == ()


def test_learning_disabled_stores_nothing(data_root: Path) -> None:
    config = cc.replace_learning(cc.default_config_for("ultron"), enabled=False)
    pipeline = LearningPipeline("ultron", store=_store("ultron", data_root), config=config)
    outcome = pipeline.learn_from_model(question="q", answer="anything")
    assert outcome.count == 0
    assert _store("ultron", data_root).all() == ()


def test_a_low_confidence_record_is_kept_but_rejected(data_root: Path) -> None:
    config = cc.replace_learning(
        cc.default_config_for("jarvis"), confidence_threshold=0.9
    )
    pipeline = LearningPipeline("jarvis", store=_store("jarvis", data_root), config=config)
    pipeline.learn_from_model(question="q", answer="A weakly held claim.")
    record = _store("jarvis", data_root).all()[0]
    assert record.learning_status == STATUS_REJECTED
    assert record.trusted is False


def test_the_pipeline_never_writes_to_another_owners_store(data_root: Path) -> None:
    """A pipeline's store is the owner's own; its records all carry that owner."""
    for key in CHARACTERS:
        pipeline = LearningPipeline(key, store=_store(key, data_root))
        pipeline.learn_from_user(question="q", statement=f"{key} says something.")
        for record in _store(key, data_root).all():
            assert record.owner == key


# ==========================================================================
# D. Personality and configuration
# ==========================================================================


def test_each_character_loads_its_own_personality() -> None:
    names = {}
    for key in CHARACTERS:
        config = cc.default_config_for(key)
        names[key] = config.general.name
        assert config.general.role
        assert config.personality.traits
    # Seven distinct names, one per character.
    assert len(set(names.values())) == len(CHARACTERS)


def test_changing_one_personality_does_not_affect_another(data_root: Path) -> None:
    from core.character_registry import CharacterRegistry

    registry = CharacterRegistry(data_root=data_root)
    jarvis = registry.load("jarvis")
    registry.apply(cc.replace_personality(jarvis, instructions="Answer only in riddles."))
    assert "riddles" in registry.load("jarvis").personality.instructions
    assert "riddles" not in registry.load("ultron").personality.instructions


def test_changing_the_model_does_not_erase_learning_records(data_root: Path) -> None:
    from core.character_registry import CharacterRegistry

    _learn("ultron", data_root, "A fact Ultron will keep.")
    registry = CharacterRegistry(data_root=data_root)
    config = registry.load("ultron")
    registry.apply(cc.replace_model(config, key="qwen3-8b", temperature=0.2))
    assert registry.load("ultron").model.key == "qwen3-8b"
    # The learning store is untouched by a configuration change.
    assert len(_store("ultron", data_root).all()) == 1


def test_configuration_persists_across_a_restart(data_root: Path) -> None:
    from core.character_registry import CharacterRegistry

    registry = CharacterRegistry(data_root=data_root)
    config = registry.load("vision")
    registry.apply(cc.replace_general(config, name="Vision Prime", role="a test role"))
    # A fresh registry reads what the previous one wrote.
    fresh = CharacterRegistry(data_root=data_root)
    assert fresh.load("vision").general.name == "Vision Prime"
    assert fresh.load("vision").general.role == "a test role"


def test_the_main_model_has_its_own_configuration() -> None:
    config = cc.default_config_for(MAIN)
    assert config.general.name
    assert config.general.role
    assert config.learning.enabled


def test_the_personality_renders_from_the_configuration() -> None:
    config = cc.replace_personality(
        cc.default_config_for("hakeem"),
        instructions="Always name your sources.",
        traits=("patient", "precise"),
    )
    prompt = config.system_prompt()
    assert "Always name your sources." in prompt
    assert "patient" in prompt
    assert "Your role:" in prompt


def test_a_saved_configuration_renders_into_the_character_mouth(data_root: Path) -> None:
    """An edit really reaches the mouth's standing instruction."""
    from ai.characters import wrap_character
    from core.character_registry import CharacterRegistry

    registry = CharacterRegistry(data_root=data_root)
    registry.apply(
        cc.replace_personality(
            registry.load("ultron"), instructions="Speak in short, numbered points."
        )
    )

    class _FakeEngine:
        system_prompt = "BASE ENGINE PROMPT"

    class _FakeMouth:
        def __init__(self) -> None:
            self.engine = _FakeEngine()

        def ask(self, question: str, **_kwargs: object):
            class _Answer:
                text = "an answer"

            return _Answer()

    mouth = _FakeMouth()
    character = wrap_character("ultron", mouth, root=data_root, registry=registry)
    character.ask("a question")
    assert "Speak in short, numbered points." in mouth.engine.system_prompt
    assert mouth.engine.system_prompt.endswith("BASE ENGINE PROMPT")


def test_an_invalid_configuration_is_refused(data_root: Path) -> None:
    from core.character_registry import CharacterRegistry

    registry = CharacterRegistry(data_root=data_root)
    config = registry.load("jarvis")
    bad = cc.replace_model(config, temperature=9.0)
    with pytest.raises(cc.ConfigError):
        bad.validate()
    # The registry's APPLY validates first, so a bad draft raises and nothing is
    # written -- the previous configuration is intact.
    with pytest.raises(cc.ConfigError):
        registry.apply(bad)
    assert registry.load("jarvis").model.temperature != 9.0


def test_an_apply_that_fails_activation_rolls_back(data_root: Path) -> None:
    """A failing activation hook restores the previous configuration."""
    from core.character_registry import CharacterRegistry

    registry = CharacterRegistry(data_root=data_root)
    before = registry.load("jarvis")
    changed = cc.replace_general(before, name="Should Not Stick")

    def activate(_config: cc.CharacterConfig) -> tuple[str, ...]:
        raise RuntimeError("the mouth could not be rebuilt")

    report = registry.apply(changed, activate=activate)
    assert report.ok is False
    assert report.rolled_back is True
    assert "restored" in report.describe()
    # The file on disk is the previous configuration, not the failed one.
    assert registry.load("jarvis").general.name == before.general.name


# ==========================================================================
# E. Voice and language model
# ==========================================================================


def test_each_character_carries_its_own_voice_configuration() -> None:
    for key in CHARACTERS:
        config = cc.default_config_for(key)
        assert isinstance(config.voice, cc.VoiceConfig)
        assert config.voice.volume == 1.0


def test_a_voice_change_does_not_affect_another_character(data_root: Path) -> None:
    from core.character_registry import CharacterRegistry

    registry = CharacterRegistry(data_root=data_root)
    registry.apply(cc.replace_voice(registry.load("natasha"), voice="model:piper:ar_JO-kareem-medium"))
    assert registry.load("natasha").voice.voice == "model:piper:ar_JO-kareem-medium"
    # Hakeem keeps his own shipped voice; Natasha's change never touches it.
    assert registry.load("hakeem").voice.voice == cc.voice_defaults_for("hakeem")["voice"]


def test_an_unsupported_voice_engine_is_rejected(data_root: Path) -> None:
    from core.character_registry import CharacterRegistry

    registry = CharacterRegistry(data_root=data_root)
    bad = cc.replace_voice(registry.load("cloud"), engine="not-an-engine")
    with pytest.raises(cc.ConfigError):
        bad.validate()


def test_an_unsupported_model_is_rejected(data_root: Path) -> None:
    from core.character_registry import CharacterRegistry

    registry = CharacterRegistry(data_root=data_root)
    bad = cc.replace_model(registry.load("vision"), max_tokens=0)
    with pytest.raises(cc.ConfigError):
        bad.validate()


def test_a_shared_language_model_does_not_share_a_learning_store(data_root: Path) -> None:
    """Two characters may run the same model and still keep separate records."""
    from core.character_registry import CharacterRegistry

    registry = CharacterRegistry(data_root=data_root)
    for key in ("natasha", "hakeem"):
        registry.apply(cc.replace_model(registry.load(key), key="qwen3-4b"))
    assert registry.load("natasha").model.key == registry.load("hakeem").model.key
    _learn("natasha", data_root, "Natasha's private thought.")
    _learn("hakeem", data_root, "Hakeem's historical note.")
    assert len(_store("natasha", data_root).all()) == 1
    assert len(_store("hakeem", data_root).all()) == 1
    assert _store("natasha", data_root).root != _store("hakeem", data_root).root


# ==========================================================================
# F. The management service (the engine behind the window and the CLI)
# ==========================================================================


@pytest.fixture
def service(data_root: Path):
    from gui.character_manager import CharacterManagerService

    return CharacterManagerService(data_root=data_root)


def test_the_service_lists_every_character_and_the_main_model(service) -> None:
    keys = service.keys()
    assert set(keys) == set(CHARACTERS) | {MAIN}
    assert set(service.character_keys()) == set(CHARACTERS)
    rows = service.summaries()
    assert len(rows) == len(keys)
    assert any(row.private for row in rows if row.key == "natasha")


def test_the_service_reports_the_sharing_policy(service) -> None:
    assert set(service.shareable_characters()) == set(CHARACTERS) - {"natasha"}
    assert service.private_characters() == ("natasha",)
    assert service.can_share("natasha") is False
    assert service.can_share("jarvis") is True
    assert "private" in service.share_reason("natasha").lower()


def test_the_service_applies_and_reports_hot_reload(service) -> None:
    config = service.config("jarvis")
    result = service.apply(cc.replace_general(config, name="Jarvis Prime"))
    assert result.ok
    assert "general" in result.report.activated
    assert "model" in result.report.restart_required
    assert service.config("jarvis").general.name == "Jarvis Prime"


def test_the_service_rolls_back_an_invalid_change(service) -> None:
    before = service.config("ultron")
    bad = cc.replace_model(before, temperature=5.0)
    result = service.apply(bad)
    assert result.ok is False
    assert result.error is not None
    # Nothing was saved: the character is exactly as it was.
    assert service.config("ultron").to_primitive() == before.to_primitive()


def test_the_service_validates_before_applying(service) -> None:
    bad = cc.replace_voice(service.config("banner"), speed=99.0)
    assert service.validate(bad) is not None
    assert service.validate(service.config("banner")) is None


def test_the_service_manages_a_characters_own_learning(service, data_root: Path) -> None:
    _learn("cloud", data_root, "Cloud learned a fact about networks.")
    stats = service.stats("cloud")
    assert stats.total == 1
    records = service.records("cloud")
    assert len(records) == 1
    found = service.search("cloud", "networks")
    assert len(found) == 1
    assert service.search("cloud", "nothing-matches-this") == ()


def test_the_service_export_backs_up_the_records(service, data_root: Path, tmp_path) -> None:
    _learn("jarvis", data_root, "Jarvis learned something worth keeping.")
    target = tmp_path / "backup.json"
    path = service.export("jarvis", target)
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["owner"] == "jarvis"
    assert len(payload["records"]) == 1


def test_the_service_reset_learning_requires_confirmation(service, data_root: Path) -> None:
    _learn("vision", data_root, "Vision learned a petroleum fact.")
    assert service.reset_learning("vision", confirm=False) is False
    assert service.stats("vision").total == 1
    assert service.reset_learning("vision", confirm=True) is True
    assert service.stats("vision").total == 0


def test_the_service_verifies_and_rejects_records(service, data_root: Path) -> None:
    pipeline = LearningPipeline("banner", store=_store("banner", data_root))
    pipeline.learn_from_model(question="q", answer="Banner said something.")
    record = _store("banner", data_root).all()[0]
    verified = service.verify("banner", record.record_id)
    assert verified is not None and verified.learning_status == STATUS_VERIFIED
    rejected = service.reject("banner", record.record_id, note="not reliable")
    assert rejected is not None and rejected.learning_status == STATUS_REJECTED


def test_the_service_learns_from_a_character_into_the_main_model(service, data_root: Path) -> None:
    _learn("hakeem", data_root, "Hakeem learned a historical fact.")
    result = service.learn_from_character("hakeem")
    assert result.count == 1
    assert service.main_model_records()[0].source_character == "hakeem"


def test_the_service_never_learns_from_natasha(service, data_root: Path) -> None:
    _learn("natasha", data_root, "Natasha's private thought.")
    result = service.learn_from_character("natasha")
    assert result.count == 0
    assert service.main_model_records() == ()


def test_the_service_audit_records_the_denial(service, data_root: Path) -> None:
    _learn("natasha", data_root, "Natasha's private thought.")
    service.learn_from_character("natasha")
    denials = service.denials()
    assert denials
    assert any("natasha" in line for line in denials)


def test_the_service_reports_a_character_status(service) -> None:
    text = service.status()
    for key in CHARACTERS:
        assert key in text
    assert "natasha is private" in text


def test_the_service_describes_every_section(service) -> None:
    text = service.describe_config("jarvis")
    for _key, title in service.sections():
        assert title in text


def test_the_service_offers_only_real_choices(service) -> None:
    # The learning modes and languages are the project's own closed sets.
    assert set(service.learning_modes()) == set(cc.LEARNING_MODES)
    assert set(service.languages()) == set(cc.LANGUAGES)
    # Model choices for the retrieval mouth are what that mouth can run.
    assert service.model_choices("jarvis")
    # Voice choices never invent a voice (there may be none installed here).
    for key, _label, _source in service.voice_choices():
        assert key


def test_the_service_reset_returns_a_character_to_the_shipped_default(service) -> None:
    service.apply(cc.replace_general(service.config("jarvis"), name="Changed"))
    assert service.config("jarvis").general.name == "Changed"
    service.reset("jarvis")
    assert service.config("jarvis").general.name == cc.default_config_for("jarvis").general.name


# ==========================================================================
# G. The CLI twin
# ==========================================================================


def test_the_cli_lists_and_shows_characters(data_root: Path, capsys) -> None:
    from cli import characters as cli

    assert cli.main(["list"]) == 0
    out = capsys.readouterr().out
    for key in CHARACTERS:
        assert key in out
    assert cli.main(["show", "jarvis"]) == 0
    assert "General Information" in capsys.readouterr().out


def test_the_cli_shareable_reports_the_policy(data_root: Path, capsys) -> None:
    from cli import characters as cli

    assert cli.main(["shareable"]) == 0
    out = capsys.readouterr().out
    assert "natasha" in out
    assert "Never shared" in out


def test_the_cli_set_applies_and_reports(data_root: Path, capsys) -> None:
    from cli import characters as cli

    assert cli.main(["set", "vision", "--temperature", "0.2"]) == 0
    assert "Applied" in capsys.readouterr().out
    assert cli.main(["show", "vision", "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["model"]["temperature"] == 0.2


def test_the_cli_set_refuses_an_invalid_change(data_root: Path) -> None:
    from cli import characters as cli

    # An out-of-range temperature must be refused (exit 4), not saved.
    assert cli.main(["set", "vision", "--temperature", "9.0"]) == 4
    assert cli.main(["show", "vision", "--json"]) == 0


def test_the_cli_share_refuses_natasha(data_root: Path, capsys) -> None:
    from cli import characters as cli

    # A refused cross-character access is a gate refusal, so it exits 4 (not 0)
    # and is machine-detectable as well as printed.
    assert cli.main(["share", "natasha"]) == 4
    assert "Refused" in capsys.readouterr().out


def test_the_cli_audit_shows_denials(data_root: Path, capsys) -> None:
    from cli import characters as cli

    cli.main(["share", "natasha"])
    assert cli.main(["audit", "--denials"]) == 0
    assert "natasha" in capsys.readouterr().out


def test_the_cli_stats_and_records(data_root: Path, capsys) -> None:
    from cli import characters as cli

    _learn("banner", data_root, "Banner learned a fact.")
    assert cli.main(["stats", "banner"]) == 0
    assert "1 records" in capsys.readouterr().out
    assert cli.main(["records", "banner"]) == 0
    assert "Banner learned a fact" in capsys.readouterr().out


def test_the_cli_reset_learning_requires_confirm(data_root: Path) -> None:
    from cli import characters as cli

    _learn("cloud", data_root, "Cloud learned something.")
    assert cli.main(["reset-learning", "cloud"]) == 4
    assert cli.main(["reset-learning", "cloud", "--confirm"]) == 0
    assert _store("cloud", data_root).all() == ()


def test_the_cli_unknown_character_is_an_error() -> None:
    from cli import characters as cli

    assert cli.main(["show", "not-a-character"]) != 0


# ==========================================================================
# H. The sidebar entry and the Characters window
# ==========================================================================


def test_the_characters_sidebar_entry_exists_and_is_visible() -> None:
    from gui.actions import action_by_key, grouped_visible_actions, visible_actions

    action = action_by_key("characters")
    assert action is not None
    assert action.sidebar_label == "Characters"
    assert action in visible_actions()
    # It sits in a named group, so the sidebar draws it under a header.
    titles = {group.title for group, _members in grouped_visible_actions()}
    assert "Characters" in titles


def test_the_characters_action_is_in_the_characters_group() -> None:
    from gui.actions import action_group_by_key

    group = action_group_by_key("characters")
    assert group.title == "Characters"
    assert "characters" in group.keys


# ==========================================================================
# I. The held-out isolation gate (the evaluation metric)
# ==========================================================================


def test_the_isolation_gate_is_perfect_on_the_catalogue() -> None:
    from core.character_isolation_gate import (
        ISOLATION_JUDGEMENTS,
        character_isolation_accuracy,
    )

    assert character_isolation_accuracy() == 1.0
    # It is a real gate, not a trivially-empty one.
    assert len(ISOLATION_JUDGEMENTS) >= 15


def test_the_isolation_gate_reports_zero_for_an_empty_catalogue() -> None:
    from core.character_isolation_gate import character_isolation_accuracy

    assert character_isolation_accuracy(()) == 0.0


def test_the_isolation_gate_carries_negative_controls() -> None:
    """The catalogue must include the refusals, not only the permitted cases."""
    from core.character_isolation_gate import ISOLATION_JUDGEMENTS

    kinds = {judgement.kind for judgement in ISOLATION_JUDGEMENTS}
    assert "cross-read-refused" in kinds
    assert "natasha-refused" in kinds
    assert "natasha-refused-all" in kinds
    assert "config-cannot-enable-natasha" in kinds
