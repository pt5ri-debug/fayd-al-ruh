"""The asking CLI: the headless twin of the learner-asks-the-teacher loop."""

import json

from cli import asking as asking_cli
from cli.__main__ import _COMMANDS
from cli.menu import MENU, run_ask_teacher


def test_ask_teacher_is_registered_in_the_cli_and_the_menu():
    assert _COMMANDS["ask-teacher"] is run_ask_teacher
    assert _COMMANDS["ask_teacher"] is run_ask_teacher
    assert any(entry[1] is run_ask_teacher for entry in MENU)


def test_status_names_the_learner_and_the_teacher(capsys):
    code = asking_cli.main(["status"])
    assert code == asking_cli.EXIT_OK
    out = capsys.readouterr().out
    assert "learner:" in out
    assert "teacher:" in out


def test_status_json_carries_the_honest_fields(capsys):
    code = asking_cli.main(["status", "--json"])
    assert code == asking_cli.EXIT_OK
    payload = json.loads(capsys.readouterr().out)
    assert "learner" in payload
    assert "teacher" in payload
    assert "next_step" in payload
    assert isinstance(payload["teacher_available"], bool)


def test_a_bad_count_is_a_usage_error(capsys):
    code = asking_cli.main(["run", "--count", "0"])
    assert code == asking_cli.EXIT_USAGE


def test_run_reports_the_next_step_when_no_teacher_is_available(capsys):
    # On a machine with no local teacher model, `run` must report the real next
    # step rather than pretending to have run; when one *is* available the same
    # path simply runs, so the assertion is on the honest message.
    code = asking_cli.main(["run", "--count", "1"])
    out = capsys.readouterr().out
    if code == asking_cli.EXIT_MODEL:
        assert "Not available" in out
    else:
        assert code == asking_cli.EXIT_OK
        assert "teacher" in out
