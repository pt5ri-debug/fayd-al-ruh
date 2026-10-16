"""Tests for the text-to-speech voice catalog and the reading service.

The catalog and the reader are dependency-free; a real engine (espeak-ng) is
optional, so the tests assert the honest behaviour *without* one and only exercise
real synthesis when the engine happens to be installed.
"""

from __future__ import annotations

import shutil

import pytest

from core.reading import TextReader
from core.speech import CommandSynthesizer, SilentSynthesizer, load_synthesizer
from core.voices import (
    CATEGORY_BOY,
    CATEGORY_GIRL,
    CATEGORY_MAN,
    CATEGORY_WOMAN,
    DEFAULT_VOICE_KEY,
    VOICE_CATALOG,
    Voice,
    default_voice,
    detect_language,
    find_voice,
    voice_categories,
    voice_keys,
    voice_summaries,
    voices_for,
)

# --------------------------------------------------------------------- catalog


def test_catalog_has_several_voices_in_every_category():
    # The user asked for variety: men, women, boys, and girls, several of each.
    for category in (CATEGORY_MAN, CATEGORY_WOMAN, CATEGORY_BOY, CATEGORY_GIRL):
        assert len(voices_for(category)) >= 3, f"too few {category} voices"


def test_every_category_is_labelled_in_both_languages():
    labels = {key: (en, ar) for key, en, ar in voice_categories()}
    assert set(labels) == {CATEGORY_MAN, CATEGORY_WOMAN, CATEGORY_BOY, CATEGORY_GIRL}
    assert labels[CATEGORY_MAN][1] == "رجال"
    assert labels[CATEGORY_WOMAN][1] == "نساء"
    assert labels[CATEGORY_BOY][1] == "أولاد"
    assert labels[CATEGORY_GIRL][1] == "بنات"


def test_voice_keys_are_unique_and_resolvable():
    keys = voice_keys()
    assert len(keys) == len(set(keys)) == len(VOICE_CATALOG)
    for key in keys:
        assert find_voice(key).key == key
    assert find_voice("  man-1  ").key == "man-1"  # surrounding space tolerated


def test_find_voice_rejects_an_unknown_key():
    with pytest.raises(ValueError):
        find_voice("no-such-voice")


def test_default_voice_is_in_the_catalog():
    assert default_voice().key == DEFAULT_VOICE_KEY
    assert default_voice() in VOICE_CATALOG


def test_voice_validation_rejects_bad_data():
    with pytest.raises(ValueError):
        Voice("x", "alien", "X", "س", "m1", 50, 150)  # unknown category
    with pytest.raises(ValueError):
        Voice("x", CATEGORY_MAN, "X", "س", "", 50, 150)  # no variant
    with pytest.raises(ValueError):
        Voice("x", CATEGORY_MAN, "X", "س", "m1", 500, 150)  # pitch out of range
    with pytest.raises(ValueError):
        Voice("x", CATEGORY_MAN, "X", "س", "m1", 50, 5000)  # speed out of range


def test_voices_for_rejects_an_unknown_category():
    with pytest.raises(ValueError):
        voices_for("alien")


def test_voice_espeak_spec_uses_the_language_and_variant():
    voice = find_voice("man-1")
    assert voice.espeak_voice("ar") == "ar+m1"
    assert voice.espeak_voice("en") == "en+m1"
    options = voice.espeak_options("ar")
    assert options[0] == "-v" and options[1] == "ar+m1"
    assert "-p" in options and "-s" in options


def test_voice_labels_and_description_are_honest():
    boy = find_voice("boy-1")
    assert boy.label("en") == boy.label_en
    assert boy.label("ar") == boy.label_ar
    assert boy.category_label("ar") == "أولاد"
    # The description must not pretend a real child is speaking.
    english = boy.description("en")
    assert "not a recording" in english
    assert "real child" in english


def test_voice_summaries_match_the_catalog():
    summaries = voice_summaries()
    assert len(summaries) == len(VOICE_CATALOG)
    assert summaries[0][0] == VOICE_CATALOG[0].key
    assert all(len(row) == 4 for row in summaries)


def test_detect_language_reads_arabic_and_english():
    assert detect_language("مرحبا بالعالم") == "ar"
    assert detect_language("hello world") == "en"
    assert detect_language("hello مرحبا") == "ar"  # a mixed sentence is Arabic
    assert detect_language("   ") == "en"  # empty falls back to the default
    assert detect_language("   ", default="ar") == "ar"


# --------------------------------------------------------------- synthesizer


def test_load_synthesizer_applies_a_chosen_voice():
    # Only meaningful when espeak-ng is installed; otherwise the loader raises.
    if shutil.which("espeak-ng") is None:
        pytest.skip("espeak-ng is not installed")
    synth = load_synthesizer("espeak-ng", voice="boy-2")
    assert isinstance(synth, CommandSynthesizer)
    assert synth.voice is not None and synth.voice.key == "boy-2"
    options = synth._options_for("مرحبا")
    assert "ar+m5" in options


def test_load_synthesizer_defaults_to_the_catalog_default_voice():
    if shutil.which("espeak-ng") is None:
        pytest.skip("espeak-ng is not installed")
    synth = load_synthesizer("espeak-ng")
    assert synth.voice is not None
    assert synth.voice.key == DEFAULT_VOICE_KEY


def test_load_synthesizer_rejects_an_unknown_voice_key():
    with pytest.raises(ValueError):
        load_synthesizer("espeak-ng", voice="no-such-voice")


def test_command_synthesizer_voice_options_are_inserted_before_arguments():
    seen: dict[str, list[str]] = {}

    def _builder(executable, text, path):
        argv = [executable, "-w", str(path), text]
        seen["argv"] = argv
        return argv

    def _runner(argv, **_kwargs):
        seen["ran"] = argv

        class _Result:
            returncode = 1  # force the honest silent fallback

        return _Result()

    synth = CommandSynthesizer(
        "espeak-ng",
        "espeak-ng",
        _builder,
        voice=find_voice("girl-2"),
        language="ar",
        voice_options=lambda voice, language: voice.espeak_options(language),
    )
    # Patch subprocess.run through the module the synthesizer uses.
    import core.speech as speech

    original = speech.subprocess.run
    speech.subprocess.run = _runner
    try:
        clip = synth.synthesize("مرحبا")
    finally:
        speech.subprocess.run = original
    assert clip.is_silent  # returncode 1 -> silent fallback, never fabricated audio
    argv = seen["ran"]
    assert argv[0] == "espeak-ng"
    assert argv[1:3] == ["-v", "ar+f2"]  # the voice flags sit right after the program
    assert argv[-1] == "مرحبا"


def test_command_synthesizer_without_a_voice_adds_no_flags():
    synth = CommandSynthesizer("espeak-ng", "espeak-ng", lambda e, t, p: [e, t, p])
    assert synth._options_for("مرحبا") == ()


# -------------------------------------------------------------------- reader


def test_reader_reports_availability_honestly():
    reader = TextReader(engine="auto", voice="man-1")
    assert reader.voice.key == "man-1"
    # available() is exactly "a real engine was resolved", never a guess.
    assert reader.available() == (reader.engine_name != SilentSynthesizer.name)
    assert isinstance(reader.describe(), str)


def test_reader_with_a_silent_engine_says_so_and_still_renders():
    reader = TextReader(synthesizer=SilentSynthesizer(), voice="woman-1")
    assert reader.available() is False
    assert "No speech engine is installed" in reader.describe()
    clip = reader.render("hello")
    assert clip.is_silent


def test_reader_set_voice_changes_the_voice_and_raises_on_unknown():
    reader = TextReader(synthesizer=SilentSynthesizer(), voice="man-1")
    assert reader.set_voice("girl-4").key == "girl-4"
    assert reader.voice.key == "girl-4"
    with pytest.raises(ValueError):
        reader.set_voice("no-such-voice")


def test_reader_speak_does_not_raise_without_an_audio_player(monkeypatch):
    # No player: speak() must still return the clip instead of failing.
    reader = TextReader(synthesizer=SilentSynthesizer(), voice="boy-1")
    monkeypatch.setattr(reader.speaker, "available", lambda: False)
    clip = reader.speak("some text")
    assert clip.is_silent


def test_reader_save_writes_a_wav(tmp_path):
    reader = TextReader(synthesizer=SilentSynthesizer(), voice="girl-1")
    target = reader.save("hello world", tmp_path / "out.wav")
    assert target.exists() and target.stat().st_size > 0


@pytest.mark.skipif(shutil.which("espeak-ng") is None, reason="espeak-ng is not installed")
def test_reader_renders_real_audio_with_a_catalog_voice():
    reader = TextReader(engine="espeak-ng", voice="man-2")
    assert reader.available()
    clip = reader.render("مرحبا بكم في فيض الروح")
    assert not clip.is_silent
    assert clip.duration_seconds > 0.1


# ------------------------------------------------------------------ front ends


def test_speak_module_lists_the_voices(capsys):
    from ai import speak

    assert speak.main(["list"]) == 0
    out = capsys.readouterr().out
    assert "Men" in out and "Women" in out and "Boys" in out and "Girls" in out
    for voice in VOICE_CATALOG:
        assert voice.key in out


@pytest.mark.skipif(shutil.which("espeak-ng") is None, reason="espeak-ng is not installed")
def test_speak_module_writes_a_wav_with_a_chosen_voice(tmp_path):
    from ai import speak

    target = tmp_path / "spoken.wav"
    code = speak.main(["hello there", "--voice", "girl-4", "--write-wav", str(target)])
    assert code == 0
    assert target.exists() and target.stat().st_size > 0


def test_speak_module_reports_a_missing_engine(monkeypatch, capsys):
    from ai import speak

    monkeypatch.setattr(TextReader, "available", lambda self: False)
    code = speak.main(["hello", "--voice", "man-1"])
    assert code == 2
    out = capsys.readouterr().out
    assert "No speech engine is installed" in out
    assert "espeak-ng" in out


def test_speak_module_can_install_the_engine(monkeypatch, capsys):
    from ai import speak
    from core import speech

    calls = []

    def fake_install(*, runner=None):
        calls.append(runner)
        return speech.SpeechInstallResult(ok=True, message="Installed a speech engine.")

    monkeypatch.setattr(speak, "speech_available", lambda: False)
    monkeypatch.setattr(speak, "install_speech_engine", fake_install)
    assert speak.main(["--install-engine"]) == 0
    assert calls == [None]
    assert "Installed a speech engine" in capsys.readouterr().out


# ------------------------------------------------------- installing the engine


def test_speech_install_plan_knows_a_linux_package_manager(monkeypatch):
    from core import speech

    monkeypatch.setattr(speech.platform, "system", lambda: "Linux")
    monkeypatch.setattr(speech.shutil, "which", lambda name: "/usr/bin/apt-get" if name == "apt-get" else None)
    plan = speech.speech_install_plan()
    assert plan.can_install
    assert plan.command[:3] == ("sudo", "apt-get", "install")
    assert plan.command[-1] == "espeak-ng"


def test_speech_install_plan_is_honest_when_it_cannot_install(monkeypatch):
    from core import speech

    monkeypatch.setattr(speech.platform, "system", lambda: "Windows")
    monkeypatch.setattr(speech.shutil, "which", lambda name: None)
    plan = speech.speech_install_plan()
    assert not plan.can_install
    assert "espeak-ng" in plan.hint


def test_install_speech_engine_runs_the_plan_with_an_injected_runner(monkeypatch):
    from core import speech

    monkeypatch.setattr(speech.platform, "system", lambda: "Linux")
    monkeypatch.setattr(speech.shutil, "which", lambda name: "/usr/bin/apt-get" if name == "apt-get" else None)
    state = {"installed": False}
    monkeypatch.setattr(speech, "speech_available", lambda: state["installed"])
    seen = []

    def runner(argv):
        seen.append(tuple(argv))
        state["installed"] = True  # the package manager succeeded
        return 0, "ok"

    result = speech.install_speech_engine(runner=runner)
    assert seen == [("sudo", "apt-get", "install", "-y", "espeak-ng")]
    assert result.ok is True


def test_install_speech_engine_reports_a_failed_command(monkeypatch):
    from core import speech

    monkeypatch.setattr(speech.platform, "system", lambda: "Linux")
    monkeypatch.setattr(speech.shutil, "which", lambda name: "/usr/bin/apt-get" if name == "apt-get" else None)
    monkeypatch.setattr(speech, "speech_available", lambda: False)
    monkeypatch.setattr(speech, "discover_synthesizer_name", lambda: "silent")
    result = speech.install_speech_engine(runner=lambda argv: (100, "no permission"))
    assert result.ok is False
    assert result.returncode == 100
    assert "failed" in result.message.lower()


def test_reader_refresh_rediscovers_a_newly_installed_engine(monkeypatch):
    from core import reading
    from core.reading import TextReader

    class _FakeSynth:
        def __init__(self, name: str) -> None:
            self.name = name

        def synthesize(self, text: str):  # pragma: no cover - not called here
            raise NotImplementedError

    state = {"name": "silent"}
    monkeypatch.setattr(reading, "load_synthesizer", lambda *a, **k: _FakeSynth(state["name"]))

    # A reader built while no engine existed reports silent...
    reader = TextReader()
    assert reader.available() is False
    # ...and a refresh (after the user installs one) finds the real engine.
    state["name"] = "espeak-ng"
    reader.refresh()
    assert reader.available() is True


def test_cli_exposes_the_speak_and_voices_commands():
    from cli.__main__ import _COMMANDS
    from cli.menu import MENU, run_speak

    assert _COMMANDS["speak"] is run_speak
    assert _COMMANDS["voices"] is run_speak
    assert any("Read text aloud" in label for label, _action, _args in MENU)


def test_gui_window_owns_the_reading_voice_control():
    # The window reads messages aloud through its own controls, so the reading
    # entry is a window control rather than a mirrored sidebar action.
    from gui import actions

    labels = [action.label for action in actions.ACTIONS]
    assert not any("Read text aloud" in label for label in labels)
