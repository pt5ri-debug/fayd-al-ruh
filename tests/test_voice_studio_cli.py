"""The speech-engine studio CLI: the headless twin of the studio window.

It drives the *same* :class:`gui.voice_studio.VoiceStudioService` the window does,
so the CLI and the window share one voice choice, one mood, and one document reader.
These tests run with no display and no speech engine: a reader is injected, so a
`say` can be exercised without audio hardware.
"""

from __future__ import annotations

import json

import pytest
from cli.voice_studio import (
    EXIT_ENGINE,
    EXIT_OK,
    EXIT_REFUSED,
    EXIT_USAGE,
    main,
)
from core import voice_studio_profile as profiles

from tests.test_voice_studio import FakeReader


@pytest.fixture
def data_root(tmp_path, monkeypatch):
    root = tmp_path / "data"
    root.mkdir()
    monkeypatch.setenv("FAYD_DATA_DIR", str(root))
    return root


@pytest.fixture
def fake_reader(monkeypatch):
    """Make every studio the CLI builds use a fake, recording reader."""
    readers: list[FakeReader] = []

    def build(source, **kwargs):
        from gui.voice_studio import VoiceStudioService

        reader = FakeReader()
        readers.append(reader)
        return VoiceStudioService(source, reader=reader)

    monkeypatch.setattr("cli.voice_studio.VoiceStudioService", build)
    return readers


def test_cli_lists_the_four_studios(data_root, capsys) -> None:
    assert main(["list"]) == EXIT_OK
    out = capsys.readouterr().out
    for token in ("system", "audio.cpp", "piper", "qwen3-tts"):
        assert token in out
    assert "VoiceTut" in out and "Piper" in out and "Qwen3-TTS" in out


def test_cli_lists_the_moods(data_root, capsys) -> None:
    assert main(["moods"]) == EXIT_OK
    out = capsys.readouterr().out
    for key in ("happy", "sad", "excited", "rapturous", "in_love", "angry"):
        assert key in out


def test_cli_moods_as_json(data_root, capsys) -> None:
    assert main(["moods", "--json"]) == EXIT_OK
    rows = json.loads(capsys.readouterr().out)
    keys = [row["key"] for row in rows]
    assert "neutral" in keys and "happy" in keys
    assert all("label_en" in row and "description" in row for row in rows)


def test_cli_status_reports_honestly(data_root, capsys) -> None:
    assert main(["status", "piper"]) == EXIT_OK
    out = capsys.readouterr().out
    assert "Piper" in out


def test_cli_lists_voices_and_marks_the_choice(data_root, capsys) -> None:
    assert main(["voices", "system"]) == EXIT_OK
    out = capsys.readouterr().out
    assert "●" in out  # the chosen voice is marked


def test_cli_voices_as_json(data_root, capsys) -> None:
    assert main(["voices", "system", "--json"]) == EXIT_OK
    rows = json.loads(capsys.readouterr().out)
    assert rows and any(row["chosen"] for row in rows)


def test_cli_sets_the_voice_and_keeps_it_exclusive(data_root, capsys) -> None:
    assert main(["voice", "system", "man-2"]) == EXIT_OK
    assert profiles.load_profile("system").voice == "man-2"
    assert profiles.load_profile("piper").voice == ""
    capsys.readouterr()


def test_cli_rejects_a_voice_this_engine_does_not_offer(data_root, capsys) -> None:
    # `woman-1` is a catalog voice; the Piper studio does not offer it.
    assert main(["voice", "piper", "woman-1"]) == EXIT_REFUSED
    capsys.readouterr()


def test_cli_sets_the_mood_and_remembers_it(data_root, capsys) -> None:
    assert main(["mood", "piper", "happy"]) == EXIT_OK
    assert profiles.load_profile("piper").mood == "happy"
    capsys.readouterr()


def test_cli_rejects_an_unknown_mood(data_root, capsys) -> None:
    assert main(["mood", "piper", "nope"]) == EXIT_REFUSED
    capsys.readouterr()


def test_cli_sets_and_clears_the_name(data_root, capsys) -> None:
    assert main(["name", "piper", "My Reader"]) == EXIT_OK
    assert profiles.load_profile("piper").name == "My Reader"
    capsys.readouterr()


def test_cli_sets_a_logo_and_reports_a_bad_one(data_root, tmp_path, capsys) -> None:
    from tests.test_voice_studio import _png

    assert main(["logo", "piper", str(_png(tmp_path / "l.png"))]) == EXIT_OK
    assert profiles.load_profile("piper").logo
    bad = tmp_path / "no.txt"
    bad.write_text("x")
    assert main(["logo", "piper", str(bad)]) == EXIT_REFUSED
    capsys.readouterr()


def test_cli_say_reads_a_document(data_root, tmp_path, fake_reader, capsys) -> None:
    doc = tmp_path / "report.txt"
    doc.write_text("Porosity increases permeability.\n")
    assert main(["say", "piper", str(doc)]) == EXIT_OK
    out = capsys.readouterr().out
    assert "Read" in out
    assert fake_reader and fake_reader[-1].spoken


def test_cli_say_with_text_and_a_wav(data_root, tmp_path, fake_reader, capsys) -> None:
    target = tmp_path / "out.wav"
    assert main(["say", "piper", "--text", "hello there", "--write-wav", str(target)]) == EXIT_OK
    assert target.is_file()
    capsys.readouterr()


def test_cli_say_without_text_is_a_usage_error(data_root, fake_reader, capsys) -> None:
    assert main(["say", "piper"]) == EXIT_USAGE
    capsys.readouterr()


def test_cli_say_with_no_engine_reports_the_next_step(data_root, monkeypatch, capsys) -> None:
    from gui.voice_studio import VoiceStudioService

    def build(source, **kwargs):
        return VoiceStudioService(source, reader=FakeReader(available=False))

    monkeypatch.setattr("cli.voice_studio.VoiceStudioService", build)
    assert main(["say", "piper", "--text", "hello"]) == EXIT_ENGINE
    out = capsys.readouterr().out
    assert "cannot read aloud" in out.lower() or "no speech engine" in out.lower()
