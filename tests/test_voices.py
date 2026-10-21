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


def test_windows_builtin_engine_needs_no_install(monkeypatch):
    from core import speech

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
    assert "$s.Rate=1" in script  # speed 186 -> rate 1

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
