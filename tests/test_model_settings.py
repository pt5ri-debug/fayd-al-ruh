"""Model-specific inference settings: validated, bounded, and resolved in one order.

Nothing here touches a model. The interesting behaviour is the *resolution order*
(a per-model choice outranks a character's configuration, which outranks the
defaults) and the validation that keeps a bad value out of a runtime call.
"""

from __future__ import annotations

import pytest
from core.model_settings import (
    CONTEXT_RANGE,
    DEFAULT_CONTEXT,
    DEFAULT_GPU_LAYERS,
    DEFAULT_MAX_TOKENS,
    DEFAULT_TEMPERATURE,
    DEFAULT_THREADS,
    DEFAULT_TOP_P,
    MAX_TOKENS_RANGE,
    NAMESPACE,
    TEMPERATURE_RANGE,
    TRAINED_IDENTITY,
    ModelSettings,
    SettingsError,
    effective_settings,
    load_settings,
    reset_settings,
    save_settings,
    settings_store_path,
    update_settings,
)


@pytest.fixture()
def store(tmp_path):
    return tmp_path / "settings-root"


def test_the_defaults_are_the_documented_ones():
    settings = ModelSettings()
    assert settings.temperature == pytest.approx(DEFAULT_TEMPERATURE)
    assert settings.top_p == pytest.approx(DEFAULT_TOP_P)
    assert settings.context == DEFAULT_CONTEXT
    assert settings.max_tokens == DEFAULT_MAX_TOKENS
    assert settings.threads == DEFAULT_THREADS
    assert settings.gpu_layers == DEFAULT_GPU_LAYERS


def test_a_value_out_of_range_is_refused():
    with pytest.raises(SettingsError):
        ModelSettings(temperature=99.0)
    with pytest.raises(SettingsError):
        ModelSettings(top_p=2.0)
    with pytest.raises(SettingsError):
        ModelSettings(context=0)
    with pytest.raises(SettingsError):
        ModelSettings(max_tokens=-1)
    with pytest.raises(SettingsError):
        ModelSettings(gpu_layers=-5)


def test_a_non_numeric_value_is_refused():
    with pytest.raises(SettingsError):
        ModelSettings(temperature="hot")  # type: ignore[arg-type]


def test_the_bounds_are_inclusive():
    assert ModelSettings(temperature=TEMPERATURE_RANGE[0]).temperature == TEMPERATURE_RANGE[0]
    assert ModelSettings(temperature=TEMPERATURE_RANGE[1]).temperature == TEMPERATURE_RANGE[1]
    assert ModelSettings(context=CONTEXT_RANGE[1]).context == CONTEXT_RANGE[1]
    assert ModelSettings(max_tokens=MAX_TOKENS_RANGE[1]).max_tokens == MAX_TOKENS_RANGE[1]


def test_with_value_changes_one_field():
    settings = ModelSettings()
    changed = settings.with_value("temperature", 0.1)
    assert changed.temperature == pytest.approx(0.1)
    assert changed.context == settings.context


def test_with_value_refuses_an_unknown_field():
    with pytest.raises(SettingsError):
        ModelSettings().with_value("nonsense", 1)


def test_serialisation_round_trips():
    settings = ModelSettings(temperature=0.25, top_p=0.5, context=2048, max_tokens=128, threads=4)
    restored = ModelSettings.from_primitive(settings.to_primitive())
    assert restored == settings


def test_reading_a_corrupt_document_falls_back_to_the_defaults():
    assert ModelSettings.from_primitive("not a mapping") == ModelSettings()
    assert ModelSettings.from_primitive({"temperature": "hot"}) == ModelSettings()


def test_reading_an_out_of_range_stored_value_clamps_rather_than_raising():
    settings = ModelSettings.from_primitive({"temperature": 99.0, "context": 10_000_000})
    assert settings.temperature == TEMPERATURE_RANGE[1]
    assert settings.context == CONTEXT_RANGE[1]


def test_describe_names_the_settings():
    line = ModelSettings(temperature=0.3, context=1024).describe()
    assert "0.3" in line
    assert "1024" in line


def test_saving_and_loading_round_trips(store):
    save_settings("some/model", ModelSettings(temperature=0.2), root=store)
    loaded = load_settings("some/model", root=store)
    assert loaded.temperature == pytest.approx(0.2)


def test_loading_an_unknown_model_gives_the_defaults(store):
    assert load_settings("never/saved", root=store) == ModelSettings()


def test_saving_an_unknown_identity_is_refused(store):
    with pytest.raises(SettingsError):
        save_settings("", ModelSettings(), root=store)


def test_updating_one_field_keeps_the_rest(store):
    save_settings("m", ModelSettings(context=2048, max_tokens=256), root=store)
    updated = update_settings("m", root=store, temperature=0.1)
    assert updated.temperature == pytest.approx(0.1)
    assert updated.context == 2048
    assert updated.max_tokens == 256


def test_updating_with_an_out_of_range_value_changes_nothing(store):
    save_settings("m", ModelSettings(context=2048), root=store)
    with pytest.raises(SettingsError):
        update_settings("m", root=store, context=0)
    assert load_settings("m", root=store).context == 2048


def test_updating_with_an_unknown_field_is_refused(store):
    with pytest.raises(SettingsError):
        update_settings("m", root=store, nonsense=1)


def test_reset_returns_to_the_defaults(store):
    save_settings("m", ModelSettings(temperature=0.1), root=store)
    reset = reset_settings("m", root=store)
    assert reset == ModelSettings()
    assert load_settings("m", root=store) == ModelSettings()


def test_the_settings_store_is_under_the_data_root(store):
    assert settings_store_path(store) == store / NAMESPACE


def test_a_corrupt_stored_document_reads_as_the_defaults(store):
    path = store / NAMESPACE
    path.mkdir(parents=True, exist_ok=True)
    (path / "broken.json").write_text("{not json", encoding="utf-8")
    assert load_settings("broken", root=store) == ModelSettings()


# ------------------------------------------------------- resolution order


def test_an_explicit_value_wins_outright(store):
    save_settings("m", ModelSettings(temperature=0.9), root=store)
    explicit = ModelSettings(temperature=0.1)
    assert effective_settings("m", root=store, explicit=explicit) is explicit


def test_a_saved_per_model_choice_outranks_a_character(store):
    class CharacterModelConfig:
        temperature = 0.1
        top_p = 0.5
        context_length = 1024
        max_tokens = 64

    save_settings("m", ModelSettings(temperature=0.8), root=store)
    resolved = effective_settings("m", root=store, character=CharacterModelConfig())
    assert resolved.temperature == pytest.approx(0.8)


def test_a_character_is_consulted_when_nothing_was_saved(store):
    class CharacterModelConfig:
        temperature = 0.15
        top_p = 0.55
        context_length = 1024
        max_tokens = 96

    resolved = effective_settings("m", root=store, character=CharacterModelConfig())
    assert resolved.temperature == pytest.approx(0.15)
    assert resolved.context == 1024
    assert resolved.max_tokens == 96


def test_the_defaults_apply_when_there_is_nothing_else(store):
    assert effective_settings("m", root=store) == ModelSettings()


def test_a_character_with_no_model_fields_is_skipped(store):
    class Blank:
        pass

    assert effective_settings("m", root=store, character=Blank()) == ModelSettings()


def test_an_out_of_range_character_value_is_skipped_not_fatal(store):
    class BadCharacter:
        temperature = 99.0
        top_p = None
        context_length = None
        max_tokens = None

    # The bad value is skipped and the model still resolves to the defaults.
    assert effective_settings("m", root=store, character=BadCharacter()) == ModelSettings()


def test_the_trained_identity_is_named():
    assert TRAINED_IDENTITY
    assert effective_settings(TRAINED_IDENTITY) == ModelSettings()
