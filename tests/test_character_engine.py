"""Phase B: a character's Model and Voice settings actually take effect.

The Characters window lets a user set a character's model parameters (temperature,
top-p, context, output cap, an extra system prompt) and its reading voice, and
press APPLY. These tests pin that those settings reach the mouth the character
speaks through -- the gap Phase B closes -- and that every setting that *cannot*
be applied is reported rather than silently dropped.
"""

from __future__ import annotations

import pytest
from core.character_config import (
    replace_model,
    replace_voice,
)
from core.character_engine import (
    EngineSettingsReport,
    SettingsLedger,
    apply_model_settings,
    resolve_character_voice,
)


@pytest.fixture(autouse=True)
def data_root(tmp_path, monkeypatch):
    """Point the program's data folder (and so the shared registry) at a temp dir."""
    from core import character_registry

    character_registry._default = None
    root = tmp_path / "data"
    root.mkdir()
    monkeypatch.setenv("FAYD_DATA_DIR", str(root))
    yield root
    # The registry is a module-level singleton keyed to the data dir; drop it so
    # the next test (and the next data root) starts from the shipped defaults.
    character_registry._default = None


class StubEngine:
    """A mouth engine with the settings the in-process mouths carry."""

    def __init__(self) -> None:
        self.temperature = 0.7
        self.top_p = 0.95
        self.context = 4096
        self.max_tokens = 512
        self.system_prompt = "the mouth's own rules"


class StubAssistant:
    def __init__(self, engine=None) -> None:
        self.engine = engine if engine is not None else StubEngine()


class BareAssistant:
    """A mouth that exposes no engine at all (the retrieval mouth)."""


class CloudishEngine:
    """An engine that carries only some of the settings (a cloud provider)."""

    def __init__(self) -> None:
        self.temperature = 0.7
        self.max_tokens = 512
        self.system_prompt = "cloud rules"


# ------------------------------------------------------------------ helpers


def _config_for(key: str, **model):
    from core.character_registry import default_registry

    return replace_model(default_registry().load(key), **model)


# ------------------------------------------------------- apply_model_settings


def test_apply_model_settings_sets_every_numeric_field() -> None:
    assistant = StubAssistant()
    config = _config_for(
        "ultron", temperature=0.01, top_p=0.10, context_length=1024, max_tokens=33
    )
    report = apply_model_settings("ultron", assistant, config)
    assert assistant.engine.temperature == 0.01
    assert assistant.engine.top_p == 0.10
    assert assistant.engine.context == 1024
    assert assistant.engine.max_tokens == 33
    assert set(report.applied) >= {"temperature", "top_p", "context_length", "max_tokens"}
    assert report.changed


def test_apply_model_settings_appends_the_extra_prompt_never_replaces() -> None:
    assistant = StubAssistant()
    config = _config_for("ultron", system_prompt="Answer in one word.")
    report = apply_model_settings(
        "ultron", assistant, config, base_system_prompt="the mouth's own rules"
    )
    prompt = assistant.engine.system_prompt
    assert "the mouth's own rules" in prompt  # the honesty rules survive
    assert "Answer in one word." in prompt
    assert "system_prompt" in report.applied


def test_apply_model_settings_can_leave_the_prompt_to_the_caller() -> None:
    assistant = StubAssistant()
    config = _config_for("ultron", system_prompt="Answer in one word.")
    report = apply_model_settings(
        "ultron", assistant, config, base_system_prompt="rules", include_system_prompt=False
    )
    # The character wrapper renders the model prompt into its standing
    # instruction, so it must not be written onto the engine a second time.
    assert assistant.engine.system_prompt == "the mouth's own rules"
    assert "system_prompt" not in report.applied


def test_apply_model_settings_reports_a_mouth_with_no_engine() -> None:
    config = _config_for("ultron", temperature=0.01)
    report = apply_model_settings("sources", BareAssistant(), config)
    assert not report.changed
    assert any("no engine" in why for _name, why in report.skipped)


def test_apply_model_settings_reports_a_setting_the_engine_lacks() -> None:
    config = _config_for("cloud", temperature=0.11, top_p=0.10, context_length=1024)
    report = apply_model_settings("cloud", StubAssistant(CloudishEngine()), config)
    applied = set(report.applied)
    # The cloud engine carries temperature but has no top_p or context.
    assert "temperature" in applied
    assert "top_p" not in applied and "context_length" not in applied
    names = {name for name, _why in report.skipped}
    assert names == {"top_p", "context_length"}


def test_apply_model_settings_leaves_unchanged_defaults_alone() -> None:
    # A character that was never edited must not overwrite the mouth's own
    # tuning: the shipped default temperature (0.7) is the same as the generic
    # one, so nothing is written even though the engine's value differs.
    assistant = StubAssistant()
    assistant.engine.temperature = 0.6  # e.g. Vision's own default
    config = _config_for("vision")  # all fields at their shipped defaults
    report = apply_model_settings("vision", assistant, config)
    assert assistant.engine.temperature == 0.6
    assert not report.changed


def test_apply_model_settings_with_no_configuration_is_honest() -> None:
    report = apply_model_settings("ultron", StubAssistant(), None)
    assert not report.changed
    assert "no configuration" in report.describe()


def test_report_describe_names_applied_and_skipped() -> None:
    report = EngineSettingsReport(
        mouth="ultron", applied=("temperature",), skipped=(("top_p", "the engine lacks it"),)
    )
    text = report.describe()
    assert "ultron" in text and "temperature" in text and "top_p" in text


# ------------------------------------------------------- resolve_character_voice


def test_resolve_character_voice_finds_a_catalog_voice() -> None:
    config = replace_voice(_config_for("ultron"), voice="woman-2", active=True)
    choice = resolve_character_voice(config)
    assert choice.ok
    assert choice.voice.key == "woman-2"


def test_resolve_character_voice_is_honest_about_an_unknown_key() -> None:
    config = replace_voice(_config_for("ultron"), voice="not-a-real-voice")
    choice = resolve_character_voice(config)
    assert not choice.ok
    assert "unknown voice" in choice.reason


def test_resolve_character_voice_respects_the_active_flag() -> None:
    config = replace_voice(_config_for("ultron"), voice="woman-2", active=False)
    choice = resolve_character_voice(config)
    assert not choice.ok
    assert "switched off" in choice.reason


def test_resolve_character_voice_skips_a_blank_choice() -> None:
    config = replace_voice(_config_for("ultron"), voice="", engine="")
    choice = resolve_character_voice(config)
    assert not choice.ok
    assert "no voice is chosen" in choice.reason


def test_the_shipped_default_voice_resolves_for_every_character() -> None:
    """Every character ships a voice that really resolves here, not a blank one."""
    from core.character_config import owner_names

    for owner in owner_names():
        choice = resolve_character_voice(_config_for(owner))
        assert choice.ok, owner
        assert choice.voice is not None


def test_resolve_character_voice_reports_a_missing_local_voice() -> None:
    config = replace_voice(_config_for("ultron"), voice="model:audio.cpp:DoesNotExist")
    choice = resolve_character_voice(config)
    assert not choice.ok
    assert "not installed" in choice.reason


# --------------------------------------------- language, prosody, whole voice


def test_voice_language_reads_an_explicit_pin_only() -> None:
    from core.character_engine import voice_language

    assert voice_language(_config_for("ultron")) == ""
    assert voice_language(replace_voice(_config_for("ultron"), language="ar")) == "ar"
    assert voice_language(replace_voice(_config_for("ultron"), language="match")) == ""
    assert voice_language(replace_voice(_config_for("ultron"), language="fr")) == ""


def test_prosody_is_none_for_the_shipped_defaults() -> None:
    from core.character_engine import resolve_character_prosody

    assert resolve_character_prosody(_config_for("ultron")) is None


def test_prosody_folds_speed_pitch_and_volume() -> None:
    from core.character_engine import resolve_character_prosody

    config = replace_voice(
        _config_for("ultron"), speed=1.4, pitch=10.0, volume=1.5, active=True
    )
    prosody = resolve_character_prosody(config)
    assert prosody is not None
    assert prosody.speed_scale == 1.4
    assert prosody.pitch_delta == 10
    assert prosody.energy == 1.5


def test_prosody_clamps_to_its_own_bounds() -> None:
    from core.character_engine import resolve_character_prosody

    config = replace_voice(
        _config_for("ultron"), speed=9.0, pitch=999.0, volume=9.0, active=True
    )
    prosody = resolve_character_prosody(config)
    assert prosody is not None
    assert prosody.speed_scale == 1.6
    assert prosody.pitch_delta == 40
    assert prosody.energy == 1.8


def test_apply_character_voice_sets_voice_language_and_prosody() -> None:
    from core.character_engine import apply_character_voice

    class FakeReader:
        def __init__(self) -> None:
            self.voice = type("V", (), {"key": "man-1"})()
            self.language = None
            self.prosody = None

        def set_voice(self, key):
            # Like the real TextReader: resolve the key to a Voice object.
            self.voice = type("V", (), {"key": key})()

        def set_prosody(self, prosody):
            self.prosody = prosody

    reader = FakeReader()
    config = replace_voice(
        _config_for("ultron"), voice="woman-2", language="ar", speed=1.2, active=True
    )
    choice = apply_character_voice(reader, config)
    assert choice.ok
    assert reader.voice.key == "woman-2"
    assert reader.language == "ar"
    assert reader.prosody is not None and reader.prosody.speed_scale == 1.2


def test_apply_character_voice_never_raises_on_a_refusing_reader() -> None:
    from core.character_engine import apply_character_voice

    class Stubborn:
        def set_voice(self, voice):
            raise ValueError("no")

    config = replace_voice(_config_for("ultron"), voice="woman-2", active=True)
    choice = apply_character_voice(Stubborn(), config)
    assert not choice.ok
    assert "refused" in choice.reason


# --------------------------------------------------------------- the ledger


def test_settings_ledger_is_bounded() -> None:
    ledger = SettingsLedger()
    for i in range(SettingsLedger.MAX_LEDGER + 20):
        ledger.note(f"miss {i}")
    assert len(ledger.misses) == SettingsLedger.MAX_LEDGER
    assert ledger.misses[-1].endswith(str(SettingsLedger.MAX_LEDGER + 19))


def test_settings_ledger_describe_is_honest_when_empty() -> None:
    assert "nothing was skipped" in SettingsLedger().describe()


# --------------------------------------------------- the character wrapper


class _StubGrounded:
    """A minimal mouth the character wrapper can wrap (duck-typed)."""

    def __init__(self) -> None:
        self.engine = StubEngine()
        self.system_instruction = "grounded honesty rules"

    def ask(self, question, *, conversation=None, limit=4):  # pragma: no cover
        raise NotImplementedError


def test_character_mouth_applies_its_model_settings_on_build() -> None:
    from ai.characters import wrap_character
    from core.character_registry import default_registry

    registry = default_registry()
    config = replace_model(
        registry.load("ultron"), temperature=0.02, max_tokens=64, system_prompt="Be terse."
    )
    registry.save(config)

    mouth = _StubGrounded()
    character = wrap_character("ultron", mouth, in_memory=True)
    assert mouth.engine.temperature == 0.02
    assert mouth.engine.max_tokens == 64
    assert character.engine_report.changed


def test_character_mouth_reapplies_settings_on_reload() -> None:
    from ai.characters import wrap_character
    from core.character_registry import default_registry

    registry = default_registry()
    registry.reset("vision")
    mouth = _StubGrounded()
    character = wrap_character("vision", mouth, in_memory=True)

    registry.save(replace_model(registry.load("vision"), temperature=0.05))
    character.reload_config()
    assert mouth.engine.temperature == 0.05


def test_character_mouth_voice_choice_reads_its_own_configuration() -> None:
    from ai.characters import wrap_character
    from core.character_registry import default_registry

    registry = default_registry()
    registry.save(replace_voice(registry.load("banner"), voice="man-3", active=True))
    character = wrap_character("banner", _StubGrounded(), in_memory=True)
    choice = character.voice_choice()
    assert choice.ok
    assert choice.voice.key == "man-3"


# ------------------------------------------------- configured model key


def test_configured_model_key_reads_the_characters_own_choice() -> None:
    from ai import mouths
    from core.character_registry import default_registry

    registry = default_registry()
    registry.save(replace_model(registry.load("ultron"), key="qwen3-4b"))
    assert mouths.configured_model_key("ultron") == "qwen3-4b"


def test_configured_model_key_is_empty_for_a_non_character() -> None:
    from ai import mouths

    assert mouths.configured_model_key("not-a-mouth") == ""


# ------------------------------------------------- the manager hot-apply


class _LiveMouth:
    """A stand-in for a loaded character wrapper (only ``reload_config`` matters)."""

    def __init__(self) -> None:
        self.reloaded = 0

    def reload_config(self):
        self.reloaded += 1


def test_apply_hot_applies_a_loaded_mouth() -> None:
    from gui.character_manager import CharacterManagerService

    service = CharacterManagerService()
    live = _LiveMouth()
    service.set_live_lookup(lambda key: live if key == "ultron" else None)
    config = _config_for("ultron", temperature=0.03)
    result = service.apply(config)
    assert result.ok
    assert live.reloaded == 1
    assert result.report is not None
    assert "model" in result.report.activated


def test_apply_with_no_loaded_mouth_does_not_claim_the_model_activated() -> None:
    from gui.character_manager import CharacterManagerService

    service = CharacterManagerService()
    service.set_live_lookup(lambda key: None)
    result = service.apply(_config_for("vision", temperature=0.03))
    assert result.ok
    assert result.report is not None
    assert "model" not in result.report.activated


def test_apply_without_a_live_lookup_still_saves() -> None:
    from gui.character_manager import CharacterManagerService

    service = CharacterManagerService()
    result = service.apply(_config_for("banner", temperature=0.03))
    assert result.ok
    assert result.report is not None
    assert "model" not in result.report.activated


def test_apply_hot_applies_the_voice_to_an_open_window() -> None:
    from gui.character_manager import CharacterManagerService

    service = CharacterManagerService()
    seen: list[str] = []

    def sink(key: str) -> bool:
        seen.append(key)
        return True

    service.set_live_voice(sink)
    result = service.apply(_config_for("ultron", temperature=0.03))
    assert result.ok
    assert seen == ["ultron"]
    assert result.report is not None
    assert "voice" in result.report.activated


def test_apply_does_not_claim_the_voice_when_no_window_is_open() -> None:
    from gui.character_manager import CharacterManagerService

    service = CharacterManagerService()
    service.set_live_voice(lambda key: False)
    result = service.apply(_config_for("ultron", temperature=0.03))
    assert result.ok
    assert result.report is not None
    assert "voice" not in result.report.activated


def test_apply_never_raises_when_the_voice_sink_fails() -> None:
    from gui.character_manager import CharacterManagerService

    def boom(key: str) -> bool:
        raise RuntimeError("reader exploded")

    service = CharacterManagerService()
    service.set_live_voice(boom)
    result = service.apply(_config_for("ultron", temperature=0.03))
    assert result.ok
    assert result.report is not None
    assert "voice" not in result.report.activated


# ------------------------------------------------- per-mouth window voice


def test_model_chat_window_seeds_its_voice_from_the_character(data_root) -> None:
    from core.character_registry import default_registry

    registry = default_registry()
    registry.save(replace_voice(registry.load("ultron"), voice="man-4", active=True))

    from gui.model_chat import ModelChatService

    service = ModelChatService("ultron", data_root=data_root)
    assert service.voice_key() == "man-4"


def test_a_per_window_voice_still_wins_over_the_character(data_root) -> None:
    from core.character_registry import default_registry
    from core.mouth_profile import load_profile

    registry = default_registry()
    registry.save(replace_voice(registry.load("vision"), voice="man-4", active=True))

    from gui.model_chat import ModelChatService

    service = ModelChatService("vision", data_root=data_root)
    service.set_voice("woman-3")
    assert load_profile("vision").voice == "woman-3"
    assert ModelChatService("vision", data_root=data_root).voice_key() == "woman-3"


def test_reload_character_voice_honours_a_per_window_override(data_root) -> None:
    from core.character_registry import default_registry

    registry = default_registry()
    registry.save(replace_voice(registry.load("vision"), voice="man-4", active=True))

    from gui.model_chat import ModelChatService

    service = ModelChatService("vision", data_root=data_root)
    service.set_voice("woman-3")
    # A window pointed at its own voice keeps it: the character's voice is not
    # forced over a choice the user made in this window.
    assert service.reload_character_voice() is False
    assert service.voice_key() == "woman-3"


def test_reload_character_voice_applies_with_no_override(data_root) -> None:
    from core.character_registry import default_registry

    registry = default_registry()
    registry.save(replace_voice(registry.load("banner"), voice="man-2", active=True))

    from gui.model_chat import ModelChatService

    service = ModelChatService("banner", data_root=data_root)
    assert service.reload_character_voice() is True
    assert service.voice_key() == "man-2"

