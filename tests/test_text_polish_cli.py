"""The polish / tashkeel CLI, the headless twin of the shared polisher."""

from __future__ import annotations

from cli.menu import run_polish, run_tashkeel
from cli.text_polish import main


def test_polish_command_defaults_to_the_polish_subcommand(capsys) -> None:
    code = run_polish(["المسامية في الصخر , وهي مهمة ؟", "--quiet"])
    assert code == 0
    out = capsys.readouterr().out
    assert "المسامية في الصخر، وهي مهمة؟" in out


def test_polish_command_vocalizes_arabic(capsys) -> None:
    code = main(["polish", "ضغط المكمن", "--tashkeel", "--quiet"])
    assert code == 0
    assert "ضَغْطُ الْمَكْمَنِ" in capsys.readouterr().out


def test_polish_command_colloquial_mode(capsys) -> None:
    code = main(["polish", "ضغط المكمن", "--tashkeel", "--tashkeel-mode", "colloquial", "--quiet"])
    assert code == 0
    assert "ضَغْط الْمَكْمَن" in capsys.readouterr().out


def test_polish_command_leaves_english_alone(capsys) -> None:
    code = main(["polish", "hello there ,friend", "--tashkeel", "--quiet"])
    assert code == 0
    assert "Hello there, friend." in capsys.readouterr().out


def test_polish_reports_its_changes(capsys) -> None:
    code = main(["polish", "hello there ,friend"])
    assert code == 0
    captured = capsys.readouterr()
    assert "polish (en)" in captured.err


def test_polish_without_text_is_a_usage_error(capsys) -> None:
    code = main(["polish"])
    assert code == 2
    assert "Nothing to polish" in capsys.readouterr().err


def test_tashkeel_command_vocalizes(capsys) -> None:
    code = run_tashkeel(["كيف أقيس النفاذية"])
    assert code == 0
    out = capsys.readouterr().out
    assert "كَيْفَ" in out


def test_tashkeel_status_reports_the_saved_setting(capsys, tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("FAYD_DATA_DIR", str(tmp_path))
    code = main(["tashkeel", "--status"])
    assert code == 0
    assert "tashkeel: off" in capsys.readouterr().out


def test_tashkeel_without_text_is_a_usage_error(capsys) -> None:
    code = main(["tashkeel"])
    assert code == 2
    assert "Nothing to vocalize" in capsys.readouterr().err
