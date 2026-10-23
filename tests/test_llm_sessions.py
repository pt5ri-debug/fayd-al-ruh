"""Tests for training the in-project language model on the mouths' own sessions.

Jarvis, Ultron, Vision, and Banner each persist what the user asked and what the
mouth answered under the data directory. Those transcripts are the only record
of the voices the user actually talks to, so ``ml.llm.corpus.session_sentences``
can fold them into the training text. The behaviour is deliberately opt-in:
a transcript is personal, so nothing is read unless a caller asks for it. These
tests pin the reader, the exclusion of system/context turns, the two role
switches, the off-by-default wiring through ``build_corpus``/``corpus_units``,
and the CLI/`TrainingConfig` plumbing.
"""

from __future__ import annotations

import pytest
from core.conversation import ConversationMemory
from core.paths import data_dir
from ml.llm.corpus import (
    DEFAULT_MAX_SESSION_CHARS,
    MOUTH_SESSION_SOURCES,
    build_corpus,
    corpus_units,
    session_sentences,
)
from storage.json_store import JsonStore

MOUTHS = ("jarvis", "ultron", "vision", "banner")


def _write_session(mouth: str, *, user: str, assistant: str) -> None:
    """Persist one two-turn session under ``mouth``'s own folder."""
    memory = ConversationMemory(JsonStore(data_dir().joinpath(*MOUTH_SESSION_SOURCES[mouth])))
    conversation = memory.start(f"probe {mouth}")
    conversation.add_user(user)
    conversation.add_assistant(assistant)
    memory.save(conversation)


@pytest.fixture()
def mouths(tmp_path, monkeypatch):
    """Four mouths, each with a session whose wording is unique to it."""
    monkeypatch.setenv("FAYD_DATA_DIR", str(tmp_path))
    for mouth in MOUTHS:
        _write_session(
            mouth,
            user=f"what is the {mouth} question about porosity?",
            assistant=f"The {mouth} answer says porosity measures the voids in rock.",
        )
    return tmp_path


def test_all_four_mouths_are_read(mouths):
    sentences = session_sentences()
    for mouth in MOUTHS:
        assert any(f"{mouth} question" in s for s in sentences)
        assert any(f"{mouth} answer" in s for s in sentences)


def test_system_and_context_turns_are_never_read(mouths):
    memory = ConversationMemory(JsonStore(data_dir().joinpath(*MOUTH_SESSION_SOURCES["jarvis"])))
    conversation = memory.start("with reference material")
    conversation.add_system("SYSTEM-TURN-MUST-NOT-APPEAR")
    conversation.add_context("CONTEXT-FILE-MUST-NOT-APPEAR")
    conversation.add_user("a real question")
    conversation.add_assistant("a real answer")
    memory.save(conversation)

    sentences = session_sentences()
    assert not any("SYSTEM-TURN" in s for s in sentences)
    assert not any("CONTEXT-FILE" in s for s in sentences)
    assert any("a real answer" in s for s in sentences)


def test_role_switches_select_one_side_only(mouths):
    assistant_only = session_sentences(include_user=False)
    user_only = session_sentences(include_assistant=False)
    assert any("jarvis answer" in s for s in assistant_only)
    assert not any("jarvis question" in s for s in assistant_only)
    assert any("jarvis question" in s for s in user_only)
    assert not any("jarvis answer" in s for s in user_only)
    assert session_sentences(include_user=False, include_assistant=False) == []


def test_a_source_subset_reads_only_that_mouth(mouths):
    jarvis = session_sentences(sources=("jarvis",))
    assert any("jarvis answer" in s for s in jarvis)
    assert not any("banner answer" in s for s in jarvis)


def test_an_unknown_source_is_rejected(mouths):
    with pytest.raises(ValueError):
        session_sentences(sources=("telepathy",))


def test_reading_is_bounded_by_the_character_budget(mouths):
    assert session_sentences(max_chars=1) == []
    assert len(session_sentences(max_chars=DEFAULT_MAX_SESSION_CHARS)) > 0


def test_sentences_are_de_duplicated(mouths):
    sentences = session_sentences()
    assert len(sentences) == len(set(sentences))


def test_build_corpus_is_off_by_default_and_on_when_asked(mouths):
    default = build_corpus(repeats=1, structured_repeats=1)
    opted_in = build_corpus(repeats=1, structured_repeats=1, include_sessions=True)
    assert "jarvis answer" not in default
    assert "jarvis answer" in opted_in


def test_corpus_units_track_the_session_switch(mouths):
    plain = corpus_units()
    with_sessions = corpus_units(include_sessions=True)
    assert len(with_sessions) > len(plain)
    assert any("jarvis answer" in unit for unit in with_sessions)
    assert not any("jarvis answer" in unit for unit in plain)


def test_no_saved_sessions_contribute_nothing(tmp_path, monkeypatch):
    monkeypatch.setenv("FAYD_DATA_DIR", str(tmp_path))
    assert session_sentences() == []
    assert build_corpus(repeats=1, structured_repeats=1, include_sessions=True)


def test_llm_cli_flag_trains_on_sessions(mouths, tmp_path):
    from ml.llm.__main__ import main

    checkpoint = tmp_path / "cli.npz"
    assert (
        main(
            [
                "--steps", "2",
                "--repeats", "1",
                "--structured-repeats", "1",
                "--checkpoint", str(checkpoint),
                "--llm-sessions",
                "--quiet",
            ]
        )
        == 0
    )
    assert checkpoint.exists()


def test_release_config_carries_the_flag(mouths):
    from ml.release import TrainingConfig, corpus_summary

    config = TrainingConfig(include_sessions=True, kb_max_chunks=0)
    summary = corpus_summary(config)
    assert summary["include_sessions"] is True
    assert summary["session_sentences"] > 0

    off = corpus_summary(TrainingConfig(include_sessions=False, kb_max_chunks=0))
    assert off["include_sessions"] is False
    assert off["session_sentences"] == 0


def test_cli_menu_offers_both_session_training_entries():
    from cli.menu import run_llm, run_release, visible_menu

    entries = [(title, action, tuple(argv)) for title, action, argv in visible_menu()]
    assert (
        "Train the language model on the local mouths' own chats",
        run_llm,
        ("--llm-sessions", "--steps", "4000"),
    ) in entries
    assert (
        "Train everything for a release (including the local mouths' chats)",
        run_release,
        ("--llm-sessions",),
    ) in entries


def test_gui_sidebar_offers_both_session_training_actions():
    from gui.actions import run_llm, run_release, visible_actions

    by_key = {action.key: action for action in visible_actions()}
    assert by_key["llm-sessions"].command is run_llm
    assert by_key["llm-sessions"].argv == ("--llm-sessions", "--steps", "4000")
    assert by_key["release-sessions"].command is run_release
    assert by_key["release-sessions"].argv == ("--llm-sessions",)
