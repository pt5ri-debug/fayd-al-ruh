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


def test_every_voice_is_tuned_for_a_natural_reading():
    """Men's and women's voices alike must not sound like a flat machine.

    Two settings do that: a wide intonation range (a living sentence melody, above
    the engine's flat default of 50) and a small word gap (a natural pause between
    words). Every catalog entry carries both, for both genders.
    """
    from core.voices import CATEGORY_MAN, CATEGORY_WOMAN

    for voice in VOICE_CATALOG:
        assert voice.pitch_range > 50, f"{voice.key} would read as a flat monotone"
        assert voice.word_gap > 0, f"{voice.key} would read with no word rhythm"
        assert voice.espeak_options("ar")[-4:] == (
            "-P",
            str(voice.pitch_range),
            "-g",
            str(voice.word_gap),
        )
    # The user asked specifically for men's and women's voices to read naturally.
    for category in (CATEGORY_MAN, CATEGORY_WOMAN):
        assert all(voice.pitch_range > 50 for voice in voices_for(category))
        assert all(voice.word_gap > 0 for voice in voices_for(category))


def test_voice_prosody_validation_rejects_bad_data():
    with pytest.raises(ValueError):
        Voice("x", CATEGORY_MAN, "X", "س", "m1", 50, 150, 500, 2)  # pitch range out of range
    with pytest.raises(ValueError):
        Voice("x", CATEGORY_MAN, "X", "س", "m1", 50, 150, 60, 500)  # word gap out of range


def test_onecore_ssml_carries_the_catalog_pitch():
    from pathlib import Path

    from core import speech
    from core.voices import find_voice

    # A catalog voice sets its register through the SSML, so the four voices of a
    # category stay distinct instead of all sharing the engine default.
    script = speech._windows_onecore_script(
        Path("C:/tmp/t.txt"), Path("C:/tmp/a.wav"), find_voice("woman-1"), "ar"
    )
    assert 'pitch="+10%"' in script  # woman-1 pitch 60 -> +10% from the neutral 50
    assert "prosody rate=" in script


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


def test_speak_module_prints_the_hint_only_once_when_it_cannot_install(monkeypatch, capsys):
    """A platform the app cannot install on must not print the same hint twice."""
    from ai import speak
    from core import speech

    hint = "Install espeak-ng with winget, then reopen the terminal."
    monkeypatch.setattr(speak, "speech_available", lambda: False)
    monkeypatch.setattr(speak, "speech_install_plan", lambda: speech.SpeechInstallPlan("windows", (), hint))
    # install_speech_engine returns the plan's own hint when it cannot install.
    monkeypatch.setattr(
        speak,
        "install_speech_engine",
        lambda **kwargs: speech.SpeechInstallResult(ok=False, ran=False, message=hint),
    )
    code = speak.main(["--install-engine"])
    assert code == 1
    out = capsys.readouterr().out
    assert out.count(hint) == 1


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


def test_speech_install_plan_uses_winget_on_windows(monkeypatch):
    from core import speech

    monkeypatch.setattr(speech.platform, "system", lambda: "Windows")
    # No PowerShell (so no built-in SAPI), but winget is available.
    monkeypatch.setattr(speech.shutil, "which", lambda name: "winget" if name == "winget" else None)
    plan = speech.speech_install_plan()
    assert plan.can_install
    assert plan.command == ("winget", "install", "-e", "--id", "eSpeak-NG.eSpeak-NG")


def _pin_piper_off(monkeypatch):
    """These tests exercise the system/Windows engine choice, so Piper is hidden."""
    from core import piper_speech

    monkeypatch.setattr(piper_speech, "available_synthesizer", lambda *a, **k: False)


def test_windows_builtin_engine_needs_no_install(monkeypatch):
    from core import speech
    _pin_piper_off(monkeypatch)

    monkeypatch.setattr(speech.platform, "system", lambda: "Windows")
    monkeypatch.setattr(speech.shutil, "which", lambda name: "powershell" if name == "powershell" else None)
    # Pin the engine choice so the test is the same on any host: the OneCore
    # preference has its own tests, and this one is about the built-in fallback.
    monkeypatch.setattr(speech, "windows_onecore_available", lambda **_: False)
    # Windows ships a speech engine, so the plan honestly says there is nothing to
    # install...
    plan = speech.speech_install_plan()
    assert not plan.can_install
    assert "already provides a speech engine" in plan.hint
    # ...the built-in engine is discovered and loaded without any install...
    assert speech.windows_speech_available() is True
    assert speech.discover_synthesizer_name() == "windows-sapi"
    synth = speech.load_synthesizer("auto")
    assert isinstance(synth, speech.WindowsSapiSynthesizer)
    # ...and the built-in Windows player is used to make it audible.
    assert speech.Speaker().available() is True


def test_windows_sapi_script_renders_text_and_selects_a_gendered_voice():
    from pathlib import Path

    from core import speech
    from core.voices import find_voice

    script = speech._windows_sapi_script(Path("C:/tmp/t.txt"), Path("C:/tmp/a.wav"), find_voice("girl-2"))
    assert "System.Speech" in script
    assert "SetOutputToWaveFile('C:/tmp/a.wav')" in script
    # The text is read from a UTF-8 file, not embedded in the command line, so a
    # long Arabic answer is neither truncated nor dependent on the console encoding.
    assert "ReadAllText('C:/tmp/t.txt',[System.Text.Encoding]::UTF8)" in script
    assert "$s.Speak($text)" in script
    assert "Female" in script  # girl-2 maps onto a female SAPI voice
    # The natural catalog speeds sit near the neutral 175 wpm reference, so the
    # SAPI rate lands at (or next to) zero instead of a rushed positive value.
    assert "$s.Rate=0" in script  # speed 178 -> rate 0

    male = speech._windows_sapi_script(Path("C:/tmp/t.txt"), Path("C:/tmp/b.wav"), find_voice("man-4"))
    assert "Male" in male


def test_windows_sapi_script_escapes_single_quotes():
    from pathlib import Path

    from core import speech

    script = speech._windows_sapi_script(Path("C:/tmp/it's.txt"), Path("C:/tmp/a.wav"), None)
    assert "ReadAllText('C:/tmp/it''s.txt'" in script


def test_windows_sapi_synthesizer_falls_back_when_powershell_fails(monkeypatch):
    from core import speech

    synth = speech.WindowsSapiSynthesizer()
    monkeypatch.setattr(
        speech.subprocess,
        "run",
        lambda *a, **k: speech.subprocess.CompletedProcess(a[0], 1, b"", b"boom"),
    )
    clip = synth.synthesize("hello")
    assert clip.is_silent  # honest fallback, never fabricated audio


def test_windows_sapi_synthesizer_rejects_empty_text():
    from core import speech

    with pytest.raises(ValueError):
        speech.WindowsSapiSynthesizer().synthesize("   ")


def test_windows_speaker_plays_through_powershell(monkeypatch):
    from core import speech

    monkeypatch.setattr(speech.platform, "system", lambda: "Windows")
    monkeypatch.setattr(speech.shutil, "which", lambda name: "powershell" if name == "powershell" else None)
    seen: list[list[str]] = []

    class FakeProcess:
        returncode = 0

        def poll(self):
            return 0

        def wait(self, timeout=None):
            return 0

    def fake_popen(argv, **kwargs):
        seen.append(list(argv))
        return FakeProcess()

    monkeypatch.setattr(speech.subprocess, "Popen", fake_popen)
    speaker = speech.Speaker()
    assert speaker.available() is True
    speaker.play(speech.AudioClip.silence(0.1))
    assert seen and seen[0][0] == "powershell"
    assert "System.Media.SoundPlayer" in seen[0][-1]


def test_silent_engine_still_reports_the_platform_hint(monkeypatch):
    from core import speech
    from core.reading import TextReader

    monkeypatch.setattr(speech.platform, "system", lambda: "Windows")
    monkeypatch.setattr(speech.shutil, "which", lambda name: None)
    reader = TextReader(synthesizer=speech.SilentSynthesizer())
    # On Windows the honest hint names a real install command (winget), not a bare
    # "download a build" instruction.
    assert "winget" in reader.describe().lower()


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


# ------------------------------------------------- Arabic on the system engines


def test_windows_sapi_script_selects_a_voice_for_the_spoken_language():
    from pathlib import Path

    from core import speech
    from core.voices import find_voice

    # Arabic narrows SAPI to an Arabic-culture voice, then to the chosen gender.
    arabic = speech._windows_sapi_script(Path("C:/tmp/t.txt"), Path("C:/tmp/a.wav"), find_voice("man-1"), "ar")
    assert "TwoLetterISOLanguageName -eq 'ar'" in arabic
    assert "VoiceInfo.Gender -eq 'Male'" in arabic

    # English leaves the language filter out, so the engine default is used.
    english = speech._windows_sapi_script(Path("C:/tmp/t.txt"), Path("C:/tmp/b.wav"), find_voice("man-1"), "en")
    assert "TwoLetterISOLanguageName" not in english


def test_windows_sapi_synthesizer_detects_arabic_and_selects_a_voice(monkeypatch):
    from core import speech

    seen: dict[str, str] = {}

    def fake_run(argv, **_kwargs):
        seen["script"] = argv[-1]
        return speech.subprocess.CompletedProcess(argv, 1, b"", b"")

    monkeypatch.setattr(speech.subprocess, "run", fake_run)
    speech.WindowsSapiSynthesizer().synthesize("مرحبا بكم")
    assert "TwoLetterISOLanguageName -eq 'ar'" in seen["script"]


# --------------------------------------------------- Windows 11 OneCore voices


def test_windows_onecore_script_uses_winrt_and_selects_an_arabic_voice():
    from pathlib import Path

    from core import speech
    from core.voices import find_voice

    script = speech._windows_onecore_script(
        Path("C:/tmp/t.txt"), Path("C:/tmp/a.wav"), find_voice("man-1"), "ar"
    )
    # The WinRT engine is what can see the voices added through Settings.
    assert "Windows.Media.SpeechSynthesis.SpeechSynthesizer" in script
    assert "AllVoices" in script
    # Arabic narrows to an Arabic voice, then to the chosen gender.
    assert "Language -like 'ar*'" in script
    assert "Gender.ToString() -eq 'Male'" in script
    # The text is read as UTF-8 from a file, not embedded in the command line.
    assert "ReadAllText('C:/tmp/t.txt',[System.Text.Encoding]::UTF8)" in script
    # The rate is applied through SSML, with a plain-text retry if that fails.
    assert "SynthesizeSsmlToStreamAsync" in script
    assert "SynthesizeTextToStreamAsync" in script
    assert "prosody rate=" in script
    assert "[System.IO.File]::Create('C:/tmp/a.wav')" in script

    # English leaves the language filter out, so the engine default is used.
    english = speech._windows_onecore_script(
        Path("C:/tmp/t.txt"), Path("C:/tmp/b.wav"), find_voice("man-1"), "en"
    )
    assert "Language -like" not in english


def test_windows_onecore_engine_is_preferred_over_sapi(monkeypatch):
    from core import speech
    _pin_piper_off(monkeypatch)

    monkeypatch.setattr(speech.platform, "system", lambda: "Windows")
    monkeypatch.setattr(
        speech.shutil, "which", lambda name: "powershell" if name == "powershell" else None
    )
    monkeypatch.setattr(
        speech.subprocess,
        "run",
        lambda *a, **k: speech.subprocess.CompletedProcess(a[0], 0, b"ok", b""),
    )
    speech._onecore_cache[0] = None
    try:
        assert speech.windows_onecore_available() is True
        assert speech.discover_synthesizer_name() == "windows-onecore"
        synth = speech.load_synthesizer("auto")
        assert isinstance(synth, speech.WindowsOneCoreSynthesizer)
    finally:
        speech._onecore_cache[0] = None


def test_windows_onecore_falls_back_to_sapi_when_the_probe_fails(monkeypatch):
    from core import speech
    _pin_piper_off(monkeypatch)

    monkeypatch.setattr(speech.platform, "system", lambda: "Windows")
    monkeypatch.setattr(
        speech.shutil, "which", lambda name: "powershell" if name == "powershell" else None
    )
    monkeypatch.setattr(
        speech.subprocess,
        "run",
        lambda *a, **k: speech.subprocess.CompletedProcess(a[0], 1, b"", b"boom"),
    )
    speech._onecore_cache[0] = None
    try:
        assert speech.windows_onecore_available() is False
        assert speech.discover_synthesizer_name() == "windows-sapi"
        assert isinstance(speech.load_synthesizer("auto"), speech.WindowsSapiSynthesizer)
    finally:
        speech._onecore_cache[0] = None


def test_windows_voice_report_names_an_installed_arabic_voice(monkeypatch):
    from core import speech

    monkeypatch.setattr(speech.platform, "system", lambda: "Windows")
    monkeypatch.setattr(
        speech.shutil, "which", lambda name: "powershell" if name == "powershell" else None
    )
    listing = "Microsoft Hamed|ar-SA|Male\nMicrosoft Zira|en-US|Female\n".encode()

    def fake_run(argv, **_kwargs):
        script = argv[-1]
        if "AllVoices.Count" in script:
            return speech.subprocess.CompletedProcess(argv, 0, b"ok", b"")
        return speech.subprocess.CompletedProcess(argv, 0, listing, b"")

    monkeypatch.setattr(speech.subprocess, "run", fake_run)
    speech._onecore_cache[0] = None
    speech._windows_voices_cache[0] = None
    try:
        report = speech.windows_voice_report()
        assert "windows-onecore" in report
        assert "Arabic" in report and "Microsoft Hamed" in report
    finally:
        speech._onecore_cache[0] = None
        speech._windows_voices_cache[0] = None


def test_windows_voice_report_says_how_to_add_arabic_when_missing(monkeypatch):
    from core import speech

    monkeypatch.setattr(speech.platform, "system", lambda: "Windows")
    monkeypatch.setattr(
        speech.shutil, "which", lambda name: "powershell" if name == "powershell" else None
    )
    listing = b"Microsoft Zira|en-US|Female\n"

    def fake_run(argv, **_kwargs):
        if "AllVoices.Count" in argv[-1]:
            return speech.subprocess.CompletedProcess(argv, 0, b"ok", b"")
        return speech.subprocess.CompletedProcess(argv, 0, listing, b"")

    monkeypatch.setattr(speech.subprocess, "run", fake_run)
    speech._onecore_cache[0] = None
    speech._windows_voices_cache[0] = None
    try:
        report = speech.windows_voice_report()
        assert "no Arabic" in report
        assert "Add voices" in report
    finally:
        speech._onecore_cache[0] = None
        speech._windows_voices_cache[0] = None


def test_windows_powershell_synthesizer_reads_utf8_text_and_returns_audio(monkeypatch):
    from pathlib import Path

    from core import speech

    seen: dict[str, str] = {}

    class _FakeEngine(speech._WindowsPowerShellSynthesizer):
        name = "fake"

        def _script(self, text_path: Path, target: Path, language: str) -> str:
            seen["text"] = Path(text_path).read_text(encoding="utf-8")
            seen["language"] = language
            # Emulate the engine writing a WAV where the real script would have.
            speech.AudioClip(tuple([1200] * 800), sample_rate=8000).write_wav(target)
            return "dummy"

    monkeypatch.setattr(
        speech.subprocess,
        "run",
        lambda *a, **k: speech.subprocess.CompletedProcess(a[0], 0, b"", b""),
    )
    clip = _FakeEngine().synthesize("مرحبا بكم في فيض الروح")
    assert seen["text"] == "مرحبا بكم في فيض الروح"  # Arabic survives the round trip
    assert seen["language"] == "ar"  # detected from the text
    assert not clip.is_silent
    assert clip.sample_rate == 8000


def test_windows_powershell_synthesizer_rejects_empty_text():
    from core import speech

    with pytest.raises(ValueError):
        speech.WindowsOneCoreSynthesizer().synthesize("   ")


def test_pcm_decode_mixes_stereo_and_converts_bit_depths():
    from core import speech

    # 16-bit little-endian stereo: frames (256, 768) then (1280, 1792).
    stereo = b"\x00\x01\x00\x03\x00\x05\x00\x07"
    assert speech._pcm_to_mono16(stereo, 2, 2) == (512, 1536)
    # 8-bit PCM is unsigned, centred at 128.
    assert speech._pcm_to_mono16(bytes([128, 255, 0]), 1, 1) == (0, 32512, -32768)
    # 32-bit PCM is shifted down to 16-bit.
    assert speech._pcm_to_mono16((1 << 16).to_bytes(4, "little"), 1, 4) == (1,)
    with pytest.raises(ValueError):
        speech._pcm_to_mono16(b"\x00", 1, 3)


def test_audio_clip_reads_a_stereo_wav(tmp_path):
    import wave

    from core.speech import AudioClip

    target = tmp_path / "stereo.wav"
    with wave.open(str(target), "wb") as handle:
        handle.setnchannels(2)
        handle.setsampwidth(2)
        handle.setframerate(22_050)
        handle.writeframes(b"\x10\x00\x20\x00" * 4)
    clip = AudioClip.read_wav(target)
    assert clip.sample_rate == 22_050
    assert not clip.is_silent
    assert clip.samples[0] == 0x18  # (16 + 32) // 2


def test_refresh_windows_speech_caches_rediscovers_a_new_voice(monkeypatch):
    from core import speech

    monkeypatch.setattr(speech.platform, "system", lambda: "Windows")
    monkeypatch.setattr(
        speech.shutil, "which", lambda name: "powershell" if name == "powershell" else None
    )
    listings = [b"Microsoft Zira|en-US|Female\n"]

    def fake_run(argv, **_kwargs):
        if "AllVoices.Count" in argv[-1]:
            return speech.subprocess.CompletedProcess(argv, 0, b"ok", b"")
        return speech.subprocess.CompletedProcess(argv, 0, listings[0], b"")

    monkeypatch.setattr(speech.subprocess, "run", fake_run)
    speech._onecore_cache[0] = None
    speech._windows_voices_cache[0] = None
    try:
        assert "no Arabic" in speech.windows_voice_report()
        # The user installs an Arabic voice in Windows Settings...
        listings[0] = b"Microsoft Hamed|ar-SA|Male\n"
        # ...and a reader refresh picks it up instead of the stale cache.
        speech.refresh_windows_speech_caches()
        assert "Microsoft Hamed" in speech.windows_voice_report()
    finally:
        speech._onecore_cache[0] = None
        speech._windows_voices_cache[0] = None


def test_say_selects_an_installed_voice_for_a_non_english_language(monkeypatch):
    from core import speech
    from core.voices import find_voice

    listing = "Majed               ar_SA    # مرحبا\nSamantha            en_US    # Hello\n"

    def fake_run(argv, **_kwargs):
        assert argv[:3] == ["say", "-v", "?"]
        return speech.subprocess.CompletedProcess(argv, 0, listing.encode(), b"")

    monkeypatch.setattr(speech.subprocess, "run", fake_run)
    speech._macos_voice_cache.clear()
    try:
        options = speech._say_voice_options(find_voice("man-1"), "ar")
        assert "-v" in options and "Majed" in options
        assert "-r" in options  # the catalog rate still carries over
        # English keeps the engine default, so no voice name is forced.
        english = speech._say_voice_options(find_voice("man-1"), "en")
        assert "-v" not in english
    finally:
        speech._macos_voice_cache.clear()


def test_say_leaves_the_default_when_no_voice_for_the_language_exists(monkeypatch):
    from core import speech
    from core.voices import find_voice

    def fake_run(argv, **_kwargs):
        return speech.subprocess.CompletedProcess(argv, 0, b"Samantha  en_US  # Hello\n", b"")

    monkeypatch.setattr(speech.subprocess, "run", fake_run)
    speech._macos_voice_cache.clear()
    try:
        options = speech._say_voice_options(find_voice("man-1"), "ar")
        assert "-v" not in options
    finally:
        speech._macos_voice_cache.clear()


# ------------------------------------------------------- real installed voices


def test_system_voice_key_round_trips_and_is_marked():
    from core.voices import SYSTEM_VOICE_PREFIX, SystemVoice, is_system_voice_key

    voice = SystemVoice("Microsoft Hoda", "ar-SA", "Female", engine="windows-onecore")
    assert voice.key == f"{SYSTEM_VOICE_PREFIX}windows-onecore:Microsoft Hoda"
    assert is_system_voice_key(voice.key)
    assert not is_system_voice_key("man-1")
    assert voice.arabic and voice.language_code == "ar"
    assert voice.label_en == voice.label_ar == "Microsoft Hoda"


def test_system_voice_rejects_an_empty_name():
    from core.voices import SystemVoice

    with pytest.raises(ValueError):
        SystemVoice("", "ar-SA")


def test_system_voice_description_names_the_real_voice():
    from core.voices import SystemVoice

    voice = SystemVoice("Majed", "ar-SA", "Male")
    assert "Majed" in voice.description("en")
    assert "real voice" in voice.description("en")
    assert "Majed" in voice.description("ar")


def test_system_voice_language_label_groups_arabic():
    from core.voices import SystemVoice

    arabic = SystemVoice("Hoda", "ar-SA")
    english = SystemVoice("Zira", "en-US")
    assert arabic.language_label() == "Arabic"
    assert arabic.language_label("ar") == "العربية"
    assert english.language_label() == "English"
    assert SystemVoice("Odd", "fr-FR").language_label() == "fr-FR"  # honest fallback


def test_system_voices_lists_windows_onecore_voices(monkeypatch):
    from core import speech

    monkeypatch.setattr(speech.platform, "system", lambda: "Windows")
    monkeypatch.setattr(speech.shutil, "which", lambda name: "powershell" if name == "powershell" else None)
    monkeypatch.setattr(speech, "windows_onecore_available", lambda **_: True)
    monkeypatch.setattr(
        speech,
        "windows_installed_voices",
        lambda **_: (
            speech.WindowsVoice("Microsoft Hamed", "ar-SA", "Male", engine=speech._WINDOWS_ONE_CORE),
            speech.WindowsVoice("Microsoft Zira", "en-US", "Female", engine=speech._WINDOWS_ONE_CORE),
        ),
    )
    speech._system_voices_cache[0] = None
    try:
        voices = speech.system_voices()
    finally:
        speech._system_voices_cache[0] = None
    assert [voice.name for voice in voices] == ["Microsoft Hamed", "Microsoft Zira"]
    assert voices[0].arabic and voices[0].engine == speech._WINDOWS_ONE_CORE
    assert voices[1].language_code == "en"


def test_system_voices_is_empty_off_windows_and_macos(monkeypatch):
    from core import speech

    monkeypatch.setattr(speech.platform, "system", lambda: "Linux")
    speech._system_voices_cache[0] = None
    try:
        assert speech.system_voices() == ()
    finally:
        speech._system_voices_cache[0] = None


def test_windows_installed_voices_merges_both_stores(monkeypatch):
    """The union of the two engines: a voice only SAPI sees is not lost.

    Neither store is a superset of the other, so listing only the WinRT engine can
    silently drop a voice that only SAPI reports (and vice versa).
    """
    from core import speech

    monkeypatch.setattr(speech, "windows_speech_available", lambda: True)
    monkeypatch.setattr(speech, "windows_onecore_available", lambda **_: True)

    def fake_run(argv, **_kwargs):
        script = argv[-1]
        if "AllVoices" in script:
            # OneCore reports the Arabic voice plus one English voice.
            out = b"Microsoft Hoda|ar-SA|Female\nMicrosoft David|en-US|Male\n"
        else:
            # SAPI cannot see the OneCore Arabic voice, but knows a legacy one.
            out = b"Microsoft David|en-US|Male\nLegacy Voice|en-GB|Female\n"
        return speech.subprocess.CompletedProcess(argv, 0, out, b"")

    monkeypatch.setattr(speech.subprocess, "run", fake_run)
    speech._windows_voices_cache[0] = None
    try:
        voices = speech.windows_installed_voices()
    finally:
        speech._windows_voices_cache[0] = None
    names = [voice.name for voice in voices]
    assert names == ["Microsoft Hoda", "Microsoft David", "Legacy Voice"]
    # A voice in both stores keeps the modern engine; the SAPI-only voice is tagged
    # with the engine that can actually speak it.
    assert voices[0].engine == speech._WINDOWS_ONE_CORE
    assert voices[1].engine == speech._WINDOWS_ONE_CORE
    assert voices[2].engine == speech._WINDOWS_SAPI


def test_windows_installed_voices_recovers_onecore_voices_when_probe_fails(monkeypatch):
    """A false-negative probe must not hide the OneCore voices.

    The WinRT listing is attempted regardless of the probe (its script is tolerant),
    so an Arabic voice the OS offers to every other app is still listed here even
    when the availability probe wrongly says "no OneCore".
    """
    from core import speech

    monkeypatch.setattr(speech, "windows_speech_available", lambda: True)
    monkeypatch.setattr(speech, "windows_onecore_available", lambda **_: False)
    attempts: list[str] = []

    def fake_run(argv, **_kwargs):
        script = argv[-1]
        attempts.append("onecore" if "AllVoices" in script else "sapi")
        if "AllVoices" in script:
            return speech.subprocess.CompletedProcess(argv, 0, b"Microsoft Hoda|ar-SA|Female\n", b"")
        return speech.subprocess.CompletedProcess(argv, 0, b"Microsoft Zira|en-US|Female\n", b"")

    monkeypatch.setattr(speech.subprocess, "run", fake_run)
    speech._windows_voices_cache[0] = None
    try:
        voices = speech.windows_installed_voices()
    finally:
        speech._windows_voices_cache[0] = None
    assert attempts == ["onecore", "sapi"]  # both stores are asked
    assert [voice.name for voice in voices] == ["Microsoft Hoda", "Microsoft Zira"]
    assert voices[0].engine == speech._WINDOWS_ONE_CORE
    assert voices[1].engine == speech._WINDOWS_SAPI


def test_windows_installed_voices_lists_sapi_when_the_onecore_listing_is_empty(monkeypatch):
    """An unavailable WinRT store yields no lines, not an error: SAPI still answers."""
    from core import speech

    monkeypatch.setattr(speech, "windows_speech_available", lambda: True)

    def fake_run(argv, **_kwargs):
        if "AllVoices" in argv[-1]:
            # The tolerant script prints nothing when the store is unreachable.
            return speech.subprocess.CompletedProcess(argv, 0, b"", b"")
        return speech.subprocess.CompletedProcess(argv, 0, b"Microsoft Zira|en-US|Female\n", b"")

    monkeypatch.setattr(speech.subprocess, "run", fake_run)
    speech._windows_voices_cache[0] = None
    try:
        voices = speech.windows_installed_voices()
    finally:
        speech._windows_voices_cache[0] = None
    assert [voice.name for voice in voices] == ["Microsoft Zira"]
    assert voices[0].engine == speech._WINDOWS_SAPI


def test_parse_voice_lines_tolerates_a_missing_gender():
    """A listing without a gender column still yields a usable voice."""
    from core import speech

    voices = speech._parse_voice_lines(b"Microsoft Hoda|ar-SA\n\nMalformed\n", engine="windows-sapi")
    assert [voice.name for voice in voices] == ["Microsoft Hoda"]
    assert voices[0].gender == "" and voices[0].arabic


def test_onecore_probe_tolerates_a_non_terminating_warning(monkeypatch):
    """The probe must not abort on a warning, which would hide the Arabic voice.

    The WinRT projection can emit a non-terminating warning; with
    ``$ErrorActionPreference='Stop'`` that aborted the probe and reported "no
    OneCore", so a voice the OS offered to every other app was invisible here.
    """
    from core import speech

    seen: list[str] = []

    def fake_run(argv, **_kwargs):
        seen.append(argv[-1])
        return speech.subprocess.CompletedProcess(argv, 0, b"ok\n", b"")

    monkeypatch.setattr(speech, "windows_speech_available", lambda: True)
    monkeypatch.setattr(speech.subprocess, "run", fake_run)
    speech._onecore_cache[0] = None
    try:
        assert speech.windows_onecore_available()
    finally:
        speech._onecore_cache[0] = None
    assert "ErrorActionPreference='Continue'" in seen[0]
    assert "catch{}" in seen[0]


def test_engine_for_voice_picks_the_engine_that_saw_the_voice(monkeypatch):
    """A chosen real voice is spoken by the engine that listed it."""
    from core import speech
    from core.voices import SystemVoice

    monkeypatch.setattr(speech, "windows_speech_available", lambda: True)
    monkeypatch.setattr(speech, "windows_onecore_available", lambda **_: True)
    assert (
        speech._engine_for_voice(SystemVoice("Microsoft Hoda", "ar-SA", engine=speech._WINDOWS_ONE_CORE))
        == speech._WINDOWS_ONE_CORE
    )
    assert (
        speech._engine_for_voice(SystemVoice("Legacy", "en-GB", engine=speech._WINDOWS_SAPI))
        == speech._WINDOWS_SAPI
    )
    # A catalog voice leaves the choice to discovery.
    assert speech._engine_for_voice("man-1") is None


def test_engine_for_voice_skips_a_onecore_voice_when_the_probe_fails(monkeypatch):
    """A OneCore voice is not handed to SAPI (which cannot see it) blindly."""
    from core import speech
    from core.voices import SystemVoice

    monkeypatch.setattr(speech, "windows_speech_available", lambda: True)
    monkeypatch.setattr(speech, "windows_onecore_available", lambda **_: False)
    assert (
        speech._engine_for_voice(SystemVoice("Microsoft Hoda", "ar-SA", engine=speech._WINDOWS_ONE_CORE))
        is None
    )


def test_windows_onecore_usable_survives_a_false_negative_probe(monkeypatch):
    """A listing that returned a OneCore voice is proof the engine works."""
    from core import speech

    monkeypatch.setattr(speech, "windows_onecore_available", lambda **_: False)
    monkeypatch.setattr(
        speech,
        "windows_installed_voices",
        lambda **_: (
            speech.WindowsVoice("Microsoft Hoda", "ar-SA", "", engine=speech._WINDOWS_ONE_CORE),
        ),
    )
    assert speech.windows_onecore_usable() is True


def test_windows_onecore_usable_is_false_when_truly_absent(monkeypatch):
    from core import speech

    monkeypatch.setattr(speech, "windows_onecore_available", lambda **_: False)
    monkeypatch.setattr(
        speech,
        "windows_installed_voices",
        lambda **_: (
            speech.WindowsVoice("Microsoft David", "en-US", "", engine=speech._WINDOWS_SAPI),
        ),
    )
    assert speech.windows_onecore_usable() is False


def test_load_synthesizer_accepts_onecore_when_the_probe_lied(monkeypatch):
    """The explicit OneCore gate must not trust a false-negative probe either."""
    from core import speech

    monkeypatch.setattr(speech, "windows_speech_available", lambda: True)
    monkeypatch.setattr(speech, "windows_onecore_available", lambda **_: False)
    monkeypatch.setattr(
        speech,
        "windows_installed_voices",
        lambda **_: (
            speech.WindowsVoice("Microsoft Hoda", "ar-SA", "", engine=speech._WINDOWS_ONE_CORE),
        ),
    )
    assert isinstance(
        speech.load_synthesizer("windows-onecore"), speech.WindowsOneCoreSynthesizer
    )


def test_onecore_falls_back_to_sapi_when_its_own_store_is_empty(monkeypatch):
    """The mirror of SAPI's fallback: a SAPI voice can still read the text."""
    from core import speech

    monkeypatch.setattr(speech, "windows_speech_available", lambda: True)
    monkeypatch.setattr(speech._WindowsPowerShellSynthesizer, "_language_in_sapi", lambda self, l: True)
    monkeypatch.setattr(
        speech._WindowsPowerShellSynthesizer, "_render", lambda self, text, lang: None
    )
    good = speech.AudioClip((7, 7, 7), sample_rate=speech.SAMPLE_RATE)
    monkeypatch.setattr(speech.WindowsSapiSynthesizer, "_render", lambda self, text, lang: good)
    clip = speech.WindowsOneCoreSynthesizer(language="ar").synthesize("مرحبا")
    assert clip.samples == good.samples


def test_onecore_does_not_hand_an_unserved_language_to_sapi(monkeypatch):
    """A language SAPI cannot serve is not passed on; it would be mangled, not blank."""
    from core import speech

    monkeypatch.setattr(speech, "windows_speech_available", lambda: True)
    monkeypatch.setattr(
        speech._WindowsPowerShellSynthesizer, "_language_in_sapi", lambda self, l: False
    )
    assert speech.WindowsOneCoreSynthesizer(language="ar")._fallback() is None


def test_windows_engine_name_prefers_the_arabic_capable_engine(monkeypatch):
    """One place decides the engine, so discovery and the fallback cannot disagree."""
    from core import speech

    monkeypatch.setattr(speech, "windows_arabic_engine", lambda: speech._WINDOWS_ONE_CORE)
    monkeypatch.setattr(speech, "windows_onecore_available", lambda **_: False)
    assert speech.windows_engine_name() == speech._WINDOWS_ONE_CORE

    # With no Arabic voice the probe decides, so an English-only machine still works.
    monkeypatch.setattr(speech, "windows_arabic_engine", lambda: None)
    monkeypatch.setattr(speech, "windows_onecore_available", lambda **_: True)
    assert speech.windows_engine_name() == speech._WINDOWS_ONE_CORE
    monkeypatch.setattr(speech, "windows_onecore_available", lambda **_: False)
    assert speech.windows_engine_name() == speech._WINDOWS_SAPI


def test_windows_arabic_engine_picks_the_store_that_has_an_arabic_voice(monkeypatch):
    """The engine that can see an Arabic voice is reported, OneCore first."""
    from core import speech

    monkeypatch.setattr(speech, "windows_speech_available", lambda: True)
    monkeypatch.setattr(
        speech,
        "windows_installed_voices",
        lambda **_: (
            speech.WindowsVoice("Microsoft Hoda", "ar-SA", "Female", engine=speech._WINDOWS_ONE_CORE),
            speech.WindowsVoice("Microsoft David", "en-US", "Male", engine=speech._WINDOWS_SAPI),
        ),
    )
    assert speech.windows_arabic_engine() == speech._WINDOWS_ONE_CORE


def test_windows_arabic_engine_is_none_without_an_arabic_voice(monkeypatch):
    from core import speech

    monkeypatch.setattr(speech, "windows_speech_available", lambda: True)
    monkeypatch.setattr(
        speech,
        "windows_installed_voices",
        lambda **_: (
            speech.WindowsVoice("Microsoft David", "en-US", "Male", engine=speech._WINDOWS_SAPI),
        ),
    )
    assert speech.windows_arabic_engine() is None


def test_discovery_prefers_the_engine_that_can_see_an_arabic_voice(monkeypatch):
    """An all-OneCore Arabic machine is not read by SAPI, which cannot see it.

    Discovery used to follow the OneCore availability probe; a false-negative probe
    therefore picked SAPI, which reads Arabic with an English voice -- audibly but
    mangled. The engine that actually lists an Arabic voice now wins.
    """
    from core import speech
    _pin_piper_off(monkeypatch)

    monkeypatch.setattr(speech, "windows_speech_available", lambda: True)
    monkeypatch.setattr(speech, "windows_onecore_available", lambda **_: False)  # false negative
    monkeypatch.setattr(
        speech,
        "windows_installed_voices",
        lambda **_: (
            speech.WindowsVoice("Microsoft Hoda", "ar-SA", "Female", engine=speech._WINDOWS_ONE_CORE),
            speech.WindowsVoice("Microsoft David", "en-US", "Male", engine=speech._WINDOWS_SAPI),
        ),
    )
    assert speech.discover_synthesizer_name() == speech._WINDOWS_ONE_CORE


def test_discovery_prefers_piper_when_it_is_installed(monkeypatch):
    """A natural neural voice beats the robotic system engines once it is present.

    The whole point of Piper is that the reading sounds like a person; discovery
    therefore prefers it over espeak/SAPI/OneCore whenever the extra and a voice are
    both installed.
    """
    from core import piper_speech, speech

    monkeypatch.setattr(piper_speech, "available_synthesizer", lambda *a, **k: True)
    assert speech.discover_synthesizer_name() == "piper"


def test_sapi_defers_arabic_to_onecore_when_it_has_no_arabic_voice(monkeypatch):
    """SAPI does not fail on Arabic without an Arabic voice -- it mangles it.

    So the sibling engine is tried *before* returning SAPI's own output, not only
    when SAPI produced nothing.
    """
    from core import speech

    monkeypatch.setattr(speech, "windows_speech_available", lambda: True)
    monkeypatch.setattr(speech.WindowsSapiSynthesizer, "_serves_language", lambda self, lang: False)

    sapi_audio = speech.AudioClip((1, 2, 3), sample_rate=speech.SAMPLE_RATE)
    onecore_audio = speech.AudioClip((9, 9, 9), sample_rate=speech.SAMPLE_RATE)
    monkeypatch.setattr(
        speech._WindowsPowerShellSynthesizer, "_render", lambda self, text, lang: sapi_audio
    )
    monkeypatch.setattr(
        speech.WindowsOneCoreSynthesizer, "_render", lambda self, text, lang: onecore_audio
    )
    clip = speech.WindowsSapiSynthesizer().synthesize("مرحبا بالعالم")
    assert clip.samples == onecore_audio.samples  # the Arabic-capable engine won


def test_sapi_keeps_its_own_output_when_it_can_serve_the_language(monkeypatch):
    """An installed Arabic SAPI voice is used; no needless WinRT round trip."""
    from core import speech

    monkeypatch.setattr(speech, "windows_speech_available", lambda: True)
    monkeypatch.setattr(speech.WindowsSapiSynthesizer, "_serves_language", lambda self, lang: True)

    sapi_audio = speech.AudioClip((1, 2, 3), sample_rate=speech.SAMPLE_RATE)
    monkeypatch.setattr(
        speech._WindowsPowerShellSynthesizer, "_render", lambda self, text, lang: sapi_audio
    )

    def boom(self, text, lang):  # pragma: no cover - must not be called
        raise AssertionError("OneCore should not be tried when SAPI serves the language")

    monkeypatch.setattr(speech.WindowsOneCoreSynthesizer, "_render", boom)
    clip = speech.WindowsSapiSynthesizer().synthesize("مرحبا بالعالم")
    assert clip.samples == sapi_audio.samples


def test_engine_for_voice_does_not_send_an_arabic_voice_to_sapi(monkeypatch):
    """A SAPI Arabic voice is re-routed to OneCore when only OneCore sees Arabic."""
    from core import speech
    from core.voices import SystemVoice

    monkeypatch.setattr(speech, "windows_speech_available", lambda: True)
    monkeypatch.setattr(speech, "windows_arabic_engine", lambda: speech._WINDOWS_ONE_CORE)
    arabic = SystemVoice("Microsoft Hoda", "ar-SA", engine=speech._WINDOWS_SAPI)
    assert speech._engine_for_voice(arabic) == speech._WINDOWS_ONE_CORE
    # An English SAPI voice stays on SAPI.
    english = SystemVoice("Microsoft David", "en-US", engine=speech._WINDOWS_SAPI)
    assert speech._engine_for_voice(english) == speech._WINDOWS_SAPI


def test_sapi_script_language_probe_checks_installed_voices():
    """`_serves_language` asks SAPI itself, so the answer is what SAPI really has."""
    from core import speech

    script = speech._sapi_language_probe_script("ar")
    assert "GetInstalledVoices()" in script
    assert "TwoLetterISOLanguageName -eq 'ar'" in script
    assert "$_.Enabled" in script  # a disabled voice cannot serve the language


def test_load_synthesizer_auto_uses_the_engine_that_reported_the_voice(monkeypatch):
    """Selecting a real voice in 'auto' uses its own engine, not the discovered one."""
    from core import speech
    from core.voices import SystemVoice

    monkeypatch.setattr(speech, "windows_speech_available", lambda: True)
    monkeypatch.setattr(speech, "windows_onecore_available", lambda **_: True)
    # Discovery would say SAPI, but the chosen voice is a OneCore voice.
    monkeypatch.setattr(speech, "discover_synthesizer_name", lambda: speech._WINDOWS_SAPI)
    synthesizer = speech.load_synthesizer(
        "auto", voice=SystemVoice("Microsoft Hoda", "ar-SA", engine=speech._WINDOWS_ONE_CORE)
    )
    assert synthesizer.name == speech._WINDOWS_ONE_CORE


def test_system_voices_parses_macos_say_listing(monkeypatch):
    from core import speech

    monkeypatch.setattr(speech.platform, "system", lambda: "Darwin")
    listing = "Majed               ar_SA    # مرحبا\nSamantha            en_US    # Hello\n"
    monkeypatch.setattr(
        speech.subprocess,
        "run",
        lambda argv, **_kwargs: speech.subprocess.CompletedProcess(argv, 0, listing.encode(), b""),
    )
    speech._system_voices_cache[0] = None
    try:
        voices = speech.system_voices()
    finally:
        speech._system_voices_cache[0] = None
    assert [voice.name for voice in voices] == ["Majed", "Samantha"]
    assert voices[0].language == "ar-SA" and voices[0].engine == "say"


def test_sapi_script_selects_a_system_voice_by_exact_name():
    from pathlib import Path

    from core import speech
    from core.voices import SystemVoice

    voice = SystemVoice("Microsoft Hoda", "ar-SA", "Female")
    script = speech._windows_sapi_script(Path("C:/tmp/t.txt"), Path("C:/tmp/a.wav"), voice)
    assert "$s.SelectVoice('Microsoft Hoda');" in script


def test_onecore_script_selects_a_system_voice_by_display_name():
    from pathlib import Path

    from core import speech
    from core.voices import SystemVoice

    voice = SystemVoice("Microsoft Hamed", "ar-SA", "Male")
    script = speech._windows_onecore_script(
        Path("C:/tmp/t.txt"), Path("C:/tmp/a.wav"), voice, "ar"
    )
    assert "$_.DisplayName -eq 'Microsoft Hamed'" in script
    assert "$speak.Voice=$pick" in script


def test_onecore_script_names_the_voice_inside_the_ssml():
    """A catalog voice names the picked voice in the SSML, so it cannot fall back.

    The WinRT engine takes the voice from the markup, so a bare ``xml:lang`` alone
    overrode ``$speak.Voice`` and read every Arabic entry with the language's default
    voice — a woman's entry and a man's entry both came out as the same male voice.
    """
    from pathlib import Path

    from core import speech
    from core.voices import find_voice

    woman = speech._windows_onecore_script(
        Path("C:/tmp/t.txt"), Path("C:/tmp/a.wav"), find_voice("woman-1"), "ar"
    )
    assert '<voice name="' in woman
    assert "$pick.DisplayName" in woman
    # The SSML's language follows the picked voice, not the bare request language.
    assert "$lang=$pick.Language" in woman
    assert "$speak.Voice=$pick" in woman


def test_onecore_script_speaks_an_installed_voice_as_plain_text():
    """An installed voice uses the documented path, so nothing can override it.

    Microsoft's documented way to choose a voice is the ``Voice`` property with
    ``SynthesizeTextToStreamAsync``. That is used for an installed voice (whose rate
    is already neutral), so no SSML is built that could override the pick — which is
    exactly how Hoda and Naayf both collapsed onto the language default before.
    """
    from pathlib import Path

    from core import speech
    from core.voices import SystemVoice

    for name in ("Microsoft Hoda", "Microsoft Naayf"):
        script = speech._windows_onecore_script(
            Path("C:/tmp/t.txt"),
            Path("C:/tmp/a.wav"),
            SystemVoice(name, "ar-SA", "Female"),
            "ar",
        )
        # Selected by its exact display name, set on the property, spoken as text.
        assert f"$_.DisplayName -eq '{name}'" in script
        assert "$speak.Voice=$pick" in script
        assert "SynthesizeTextToStreamAsync" in script
        # No SSML at all, so nothing in markup can override the chosen voice.
        assert "SynthesizeSsmlToStreamAsync" not in script
        assert "xml:lang" not in script


def test_say_selects_a_system_voice_by_exact_name():
    from core import speech
    from core.voices import SystemVoice

    options = speech._say_voice_options(SystemVoice("Majed", "ar-SA", "Male"), "ar")
    assert options == ("-v", "Majed")


def test_resolve_voice_accepts_a_system_voice_key(monkeypatch):
    from core import speech

    monkeypatch.setattr(speech, "system_voices", lambda **_: ())
    voice = speech._resolve_voice("system:windows-onecore:Microsoft Hoda")
    # Not installed here, so the catalog default is used rather than a bad name.
    assert voice is not None and voice.key == "woman-1"
    # An unknown *catalog* key still raises, so a typo is reported.
    with pytest.raises(ValueError):
        speech._resolve_voice("no-such-voice")


def test_reader_accepts_a_system_voice_key(monkeypatch):
    from core import speech
    from core.reading import TextReader
    from core.voices import SystemVoice

    installed = SystemVoice("Microsoft Hoda", "ar-SA", "Female", engine="windows-onecore")
    monkeypatch.setattr(speech, "system_voices", lambda **_: (installed,))
    reader = TextReader(synthesizer=SilentSynthesizer(), voice=installed.key)
    assert reader.voice.key == installed.key
    # A saved key whose voice is gone falls back to the catalog default.
    monkeypatch.setattr(speech, "system_voices", lambda **_: ())
    gone = TextReader(synthesizer=SilentSynthesizer(), voice=installed.key)
    assert gone.voice.key == DEFAULT_VOICE_KEY


def test_speak_list_shows_installed_voices_when_present(monkeypatch, capsys):
    from ai import speak
    from core.voices import SystemVoice

    monkeypatch.setattr(
        speak,
        "system_voices",
        lambda: (SystemVoice("Microsoft Hoda", "ar-SA", "Female"),),
    )
    assert speak.main(["list"]) == 0
    out = capsys.readouterr().out
    assert "Installed voices" in out
    assert "Microsoft Hoda" in out
    # The synthetic catalog is still listed below it.
    assert "Men" in out and "man-1" in out


def test_speak_list_says_so_when_no_installed_voice_is_reported(monkeypatch, capsys):
    """With no OS voice reported, say it plainly instead of only the catalog.

    This is the reported symptom (the machine's real Arabic voice never appeared):
    the list must not look as though the invented catalog is all there is.
    """
    from ai import speak

    monkeypatch.setattr(speak, "system_voices", lambda: ())
    monkeypatch.setattr(speak, "windows_voice_report", lambda: "Windows note here")
    assert speak.main(["list"]) == 0
    out = capsys.readouterr().out
    assert "none were reported" in out
    assert "Windows note here" in out
    assert "not the voices installed in the OS" in out


def test_sapi_synthesizer_falls_back_to_onecore_when_it_has_no_voice(monkeypatch):
    from core import speech

    # SAPI renders nothing (all voices are OneCore on this machine)...
    monkeypatch.setattr(speech.WindowsSapiSynthesizer, "_render", lambda self, text, lang: None)
    # The fallback does not gate on the OneCore probe (a false-negative probe is one
    # of the ways the Arabic voice goes missing); Windows speech being available is
    # enough, so the sibling is attempted and fails harmlessly if truly absent.
    monkeypatch.setattr(speech, "windows_speech_available", lambda: True)

    good = speech.AudioClip((1000, -1000, 2000), sample_rate=speech.SAMPLE_RATE)

    def fake_render(self, text, lang):
        assert self.name == speech._WINDOWS_ONE_CORE
        return good

    monkeypatch.setattr(speech.WindowsOneCoreSynthesizer, "_render", fake_render)
    clip = speech.WindowsSapiSynthesizer().synthesize("مرحبا بالعالم")
    assert clip.samples == good.samples  # the sibling engine's audio, not silence


def test_windows_synthesizer_returns_silence_only_when_both_engines_fail(monkeypatch):
    from core import speech

    monkeypatch.setattr(speech.WindowsSapiSynthesizer, "_render", lambda self, text, lang: None)
    monkeypatch.setattr(speech, "windows_onecore_available", lambda **_: False)
    clip = speech.WindowsSapiSynthesizer().synthesize("مرحبا بالعالم")
    assert clip.is_silent  # honest last resort, not a crash


def test_onecore_script_keeps_the_default_voice_when_no_language_match():
    from pathlib import Path

    from core import speech
    from core.voices import find_voice

    # A catalog voice on a machine with no Arabic voice: the engine default is kept
    # rather than failing, so the reading is audible (just not Arabic).
    script = speech._windows_onecore_script(
        Path("C:/tmp/t.txt"), Path("C:/tmp/a.wav"), find_voice("woman-1"), "ar"
    )
    assert "$m=@($voices|?{$_.Language -like 'ar*'});" in script
