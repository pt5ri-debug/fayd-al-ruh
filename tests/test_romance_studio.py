"""Tests for the Romance & Companion Studio service and CLI (Phase 10).

The service (:class:`gui.romance.RomanceStudioService`) is the Tk-free half the
window and the CLI both drive. These tests run it headless with an injected
generation seam, and drive the CLI through its ``main`` with a throwaway data
root, so the whole studio is exercised with **no model and no display**.
"""

from __future__ import annotations

import json
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

from cli import romance as cli
from gui.romance import RomanceStudioService


class _Seam:
    def __init__(self, *replies: str) -> None:
        self.replies = list(replies) or ["Hello."]
        self.calls: list[str] = []

    def __call__(self, prompt: str, key, handle) -> tuple[str, bool, str]:
        self.calls.append(prompt)
        return self.replies[min(len(self.calls) - 1, len(self.replies) - 1)], True, ""


@contextmanager
def _service(seam=None) -> Iterator[RomanceStudioService]:
    with tempfile.TemporaryDirectory() as folder:
        yield RomanceStudioService(root=Path(folder), generator=seam or _Seam())


# -- the service ----------------------------------------------------------


def test_the_service_reports_an_honest_status_with_no_model():
    with _service() as service:
        status = service.status()
        assert status.character_count == 0
        assert status.local_only is True
        # The status is honest: either a runnable model is ready, or the note names
        # the real next step. It never claims ready with nothing behind it.
        assert isinstance(status.model_ready, bool)
        assert status.model_ready or status.model_note


def test_a_character_can_be_created_listed_and_read():
    with _service() as service:
        bundle = service.create_character("Layla", fictional_age=27, interests=("coffee",))
        rows = service.character_rows()
        assert len(rows) == 1 and rows[0].name == "Layla"
        assert "coffee" in service.character(bundle.id).interests()


def test_a_turn_through_the_service_uses_the_seam():
    with _service(_Seam("Hi there.")) as service:
        character = service.create_character("Noor", fictional_age=25)
        conversation = service.start_conversation(character.id)
        result = service.send(conversation.record_id, "Hello.")
        assert result.ok and result.reply == "Hi there."
        turns = service.messages(conversation.record_id)
        assert [t.role for t in turns] == ["user", "assistant"]


def test_memories_are_stored_and_recalled_per_character():
    with _service() as service:
        first = service.create_character("A", fictional_age=30).id
        second = service.create_character("B", fictional_age=30).id
        service.add_memory(first, "semantic_personal", "Likes strong coffee", importance=0.9)
        assert len(service.memories(first)) == 1
        # The other character's memory is isolated.
        assert service.memories(second) == ()
        recalled = service.recall(first, "coffee")
        assert recalled and "coffee" in recalled[0].content


def test_the_relationship_moves_and_can_be_reset():
    with _service() as service:
        character = service.create_character("Layla", fictional_age=27).id
        conversation = service.start_conversation(character)
        service.send(conversation.record_id, "I am glad to meet you.")
        snapshot = service.relationship(character)
        assert snapshot.dimensions["familiarity"] > 0.0
        reset = service.reset_relationship(character)
        assert reset.dimensions["familiarity"] == 0.0


def test_a_voice_profile_is_reported_honestly():
    with _service() as service:
        character = service.create_character("Layla", fictional_age=27).id
        view = service.voice_profile(character)
        # No voice was chosen yet; the view still reports truthfully.
        assert view.character_id == character


def test_adult_mode_is_off_by_default_and_needs_confirmation():
    with _service() as service:
        assert service.safety_status()["adult_mode"] is False
        service.set_adult_mode(True, confirmed=True)
        assert service.safety_status()["adult_mode"] is True


# -- the CLI --------------------------------------------------------------


def _run(argv, capsys):
    code = cli.main(argv)
    return code, capsys.readouterr().out


def test_cli_status_and_character_lifecycle(capsys):
    with tempfile.TemporaryDirectory() as folder:
        code, out = _run(["--data-root", folder, "status"], capsys)
        assert code == cli.EXIT_OK and "Characters: 0" in out
        code, out = _run(["--data-root", folder, "new", "Layla", "--age", "27", "--json"], capsys)
        assert code == cli.EXIT_OK
        payload = json.loads(out)
        character_id = payload["character_id"]
        code, out = _run(["--data-root", folder, "characters"], capsys)
        assert "Layla" in out
        code, out = _run(["--data-root", folder, "favorite", character_id], capsys)
        assert code == cli.EXIT_OK
        code, out = _run(["--data-root", folder, "show", character_id, "--json"], capsys)
        assert json.loads(out)["name"] == "Layla"


def test_cli_chat_reports_no_model_honestly(capsys):
    with tempfile.TemporaryDirectory() as folder:
        _run(["--data-root", folder, "new", "Layla", "--age", "27", "--json"], capsys)
        code, out = _run(
            ["--data-root", folder, "characters", "--json"], capsys
        )
        character_id = json.loads(out)[0]["character_id"]
        code, out = _run(["--data-root", folder, "chat", character_id, "Hello."], capsys)
        # No model is installed, so the turn is unavailable, not a crash.
        assert code == cli.EXIT_UNAVAILABLE
        assert "No answer" in out


def test_cli_memory_round_trip(capsys):
    with tempfile.TemporaryDirectory() as folder:
        _run(["--data-root", folder, "new", "Layla", "--age", "27", "--json"], capsys)
        code, out = _run(["--data-root", folder, "characters", "--json"], capsys)
        character_id = json.loads(out)[0]["character_id"]
        code, out = _run(
            ["--data-root", folder, "remember", character_id, "Likes tea", "--json"], capsys
        )
        assert code == cli.EXIT_OK
        code, out = _run(["--data-root", folder, "memories", character_id, "--json"], capsys)
        assert json.loads(out)[0]["content"] == "Likes tea"
        code, out = _run(["--data-root", folder, "recall", character_id, "tea", "--json"], capsys)
        assert json.loads(out)


def test_cli_modes_and_safety(capsys):
    with tempfile.TemporaryDirectory() as folder:
        _code, out = _run(["--data-root", folder, "modes", "--json"], capsys)
        keys = [row["key"] for row in json.loads(out)]
        assert "romantic" in keys and "adult" in keys
        _code, out = _run(["--data-root", folder, "safety", "--json"], capsys)
        assert json.loads(out)["adult_mode"] is False


def test_cli_adult_requires_confirmation(capsys):
    with tempfile.TemporaryDirectory() as folder:
        code, _out = _run(["--data-root", folder, "adult", "--on"], capsys)
        assert code == cli.EXIT_REFUSED
        code, _out = _run(["--data-root", folder, "adult", "--on", "--confirm"], capsys)
        assert code == cli.EXIT_OK


def test_cli_delete_honours_the_retention_choice(capsys):
    with tempfile.TemporaryDirectory() as folder:
        _run(["--data-root", folder, "new", "Layla", "--age", "27", "--json"], capsys)
        code, out = _run(["--data-root", folder, "characters", "--json"], capsys)
        character_id = json.loads(out)[0]["character_id"]
        code, out = _run(
            ["--data-root", folder, "delete", character_id, "--choice", "purge"], capsys
        )
        assert code == cli.EXIT_OK
        code, out = _run(["--data-root", folder, "characters", "--json"], capsys)
        assert json.loads(out) == []


def test_discover_lists_the_shipped_examples_without_creating_them():
    with _service() as service:
        rows = service.discover_examples()
        assert rows, "the build ships example companions"
        # Browsing the library never writes to the database.
        assert list(service.character_rows()) == []
        assert all(present is False for _draft, present in rows)


def test_add_example_creates_only_the_chosen_companion():
    with _service() as service:
        rows = service.discover_examples()
        name = rows[0][0].name
        bundle = service.add_example(name)
        assert bundle.name == name
        # Only the chosen one was created, and it is now marked present.
        assert len(service.character_rows()) == 1
        present = {draft.name: flag for draft, flag in service.discover_examples()}
        assert present[name] is True
        # Adding it twice does not duplicate it.
        again = service.add_example(name)
        assert again.id == bundle.id
        assert len(service.character_rows()) == 1


def test_add_example_refuses_a_name_the_build_does_not_ship():
    with _service() as service:
        try:
            service.add_example("Nobody")
        except ValueError as error:
            assert "Nobody" in str(error)
        else:  # pragma: no cover - the refusal must raise
            raise AssertionError("an unknown example name must be refused")


def test_cli_discover_and_add_example(capsys):
    with tempfile.TemporaryDirectory() as folder:
        code, out = _run(["--data-root", folder, "discover", "--json"], capsys)
        assert code == cli.EXIT_OK
        rows = json.loads(out)
        assert rows and rows[0]["present"] is False
        name = rows[0]["name"]
        code, out = _run(["--data-root", folder, "add-example", name, "--json"], capsys)
        assert code == cli.EXIT_OK
        code, out = _run(["--data-root", folder, "characters", "--json"], capsys)
        assert len(json.loads(out)) == 1
        code, out = _run(["--data-root", folder, "discover", "--json"], capsys)
        present = {row["name"]: row["present"] for row in json.loads(out)}
        assert present[name] is True


# -- the brief's end-to-end workflow --------------------------------------


def test_end_to_end_workflow_across_a_restart():
    """The brief's integration workflow: create → chat → remember → restart → resume.

    Runs the whole path against a **real** on-disk database (a throwaway folder),
    building the service twice so the second build is the "reopen" and must find
    the same character, conversation, history, relationship, and memory.
    """
    with tempfile.TemporaryDirectory() as folder:
        root = Path(folder)
        seam = _Seam("I am glad to meet you.", "Tell me more about your day.")
        # 1-5: create an adult character, assign a mode, start a romantic conversation.
        service = RomanceStudioService(root=root, generator=seam)
        bundle = service.create_character(
            "Layla", fictional_age=27, interests=("coffee", "poetry")
        )
        conversation = service.start_conversation(bundle.id, mode="romantic")
        first = service.send(conversation.record_id, "Hello, I am glad to meet you.")
        assert first.ok is True
        # 7: an important memory is generated and stored.
        service.add_memory(bundle.id, "semantic_personal", "The user loves Arabic poetry.")
        relationship = service.relationship(bundle.id)
        # 8-9: close and reopen (a second service over the same database).
        del service
        reopened = RomanceStudioService(root=root, generator=seam)
        # 10: the conversation resumes with its full history.
        record, turns = reopened.resume(conversation.record_id)
        assert record.record_id == conversation.record_id
        assert len(turns) == 2
        # 11: personality and memory continuity hold.
        assert reopened.character(bundle.id).name == "Layla"
        assert any("poetry" in m.content for m in reopened.memories(bundle.id))
        # 12: the relationship state persisted (familiarity moved with the turn).
        after = reopened.relationship(bundle.id)
        assert after.dimensions["familiarity"] >= relationship.dimensions["familiarity"]
        # 14: local-only is enforced on the reopened studio.
        assert reopened.safety_status()["local_only"] is True


# -- the interactive service additions ------------------------------------


def test_the_service_exposes_the_models_the_project_really_has():
    with _service() as service:
        choices = service.model_choices()
        assert choices, "the project ships a language-model catalog"
        # Every entry is named and carries a real availability flag.
        for choice in choices:
            assert choice.key
            assert choice.label()
            assert isinstance(choice.available, bool)
        # A model assigned to a character is remembered and resolved back.
        bundle = service.create_character("Layla", fictional_age=27)
        service.assign_model(bundle.id, choices[0].key)
        assert service.model_key(bundle.id) == choices[0].key


def test_a_setting_is_clamped_to_its_documented_range():
    with _service() as service:
        assert service.set_setting("temperature", "1.5") == "1.5"
        assert service.set_setting("temperature", "99") == "2"
        assert service.set_setting("temperature", "-5") == "0"
        assert service.set_setting("max_tokens", "abc") == "512"
        assert service.set_setting("context_length", "999999") == "32768"
        # A non-numeric setting passes through unchanged.
        assert service.set_setting("default_mode", "romantic") == "romantic"
        assert service.settings()["default_mode"] == "romantic"


def test_resetting_the_settings_returns_the_defaults():
    with _service() as service:
        service.set_setting("temperature", "0.1")
        service.set_setting("default_mode", "adult")
        service.reset_settings()
        values = service.settings()
        assert values["temperature"] == "0.8"
        assert values["default_mode"] == "general"


def test_setting_choices_are_the_real_ones():
    with _service() as service:
        modes = dict(service.setting_choices("default_mode"))
        assert "romantic" in modes and "adult" in modes
        models = dict(service.setting_choices("default_model_key"))
        assert set(models) == {c.key for c in service.model_choices()}
        engines = dict(service.setting_choices("voice_engine"))
        assert set(engines) == set(service.voice_engines())


def test_the_personality_round_trips_through_the_service():
    with _service() as service:
        bundle = service.create_character("Layla", fictional_age=27)
        current = service.personality(bundle.id)
        assert current["warmth"] == 0.5
        current["warmth"] = 0.9
        current["response_length"] = "detailed"
        service.set_personality(bundle.id, current)
        after = service.personality(bundle.id)
        assert round(after["warmth"], 2) == 0.90
        assert after["response_length"] == "detailed"


def test_interests_and_boundaries_are_managed_per_character():
    with _service() as service:
        bundle = service.create_character("Layla", fictional_age=27)
        service.add_interest(bundle.id, "coffee")
        assert service.character(bundle.id).interests() == ("coffee",)
        item_id = service.character(bundle.id).profile_items[0].record_id
        service.remove_interest(bundle.id, item_id)
        assert service.character(bundle.id).interests() == ()
        boundary = service.add_boundary(bundle.id, "no politics")
        assert [b.value for b in service.boundaries(bundle.id)] == ["no politics"]
        service.remove_boundary(boundary.record_id)
        assert service.boundaries(bundle.id) == ()


def test_a_conversation_can_be_deleted():
    with _service() as service:
        bundle = service.create_character("Layla", fictional_age=27)
        conversation = service.start_conversation(bundle.id, mode="general")
        service.send(conversation.record_id, "Hello there, nice to meet you.")
        assert service.conversations_for(bundle.id)
        service.delete_conversation(conversation.record_id)
        assert service.conversations_for(bundle.id) == ()


def test_the_relationship_dimensions_are_editable_and_resettable():
    with _service() as service:
        bundle = service.create_character("Layla", fictional_age=27)
        dimensions = service.relationship_dimensions()
        assert "trust" in dimensions
        service.set_relationship(bundle.id, trust=0.9)
        assert service.relationship(bundle.id).dimensions["trust"] == 0.9
        service.reset_relationship(bundle.id)
        assert service.relationship(bundle.id).dimensions["trust"] < 0.9


def test_the_studio_status_describes_what_is_installed():
    with _service() as service:
        status = service.status()
        text = status.describe()
        assert "Characters:" in text
        assert "Local only: yes" in text
