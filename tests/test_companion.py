"""Natasha, the companion: personality, memory, sessions, tone, relationship.

Everything the model would normally supply is injected: a ``caller`` seam stands
in for the local Ollama model, so the whole companion is exercised with **no
server and no weights**. The tests pin the behaviour the task asked for:

* a stable personality (fixed traits, a tunable style) and an honest system prompt;
* durable memory that is *stored and retrieved by the program*, never the model,
  with fiction kept separate from fact and sensitive data refused by default;
* independent sessions that persist and stay isolated;
* accumulating context (a bounded window plus a rolling summary);
* a *gradual*, bounded dialogue tone that can be disabled and reset;
* a relationship story that moves only with the user's approval and carries no
  love score;
* that the model is asked to reply, never to remember or to decide facts.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from companion.companion_agent import CompanionAgent, CompanionError
from companion.config import CompanionConfig
from companion.conversation_manager import ConversationManager
from companion.database import CompanionDatabase
from companion.emotion_state import MAX_STEP, DialogueEmotion, DialogueEmotionTracker
from companion.memory_manager import (
    KIND_FACT,
    KIND_PREFERENCE,
    is_save_request,
    looks_sensitive,
)
from companion.personality import (
    BOUNDARIES,
    CORE_TRAITS,
    PersonalitySettings,
    default_personality,
)
from companion.prompt_builder import PromptBuilder
from companion.relationship_manager import KIND_FICTION, KIND_FACT as REL_FACT


# -- fixtures -------------------------------------------------------------
class FakeCaller:
    """A stand-in for the local model: records the prompts and replies in turn."""

    def __init__(self, replies: list[str] | None = None) -> None:
        self.prompts: list[list[dict[str, str]]] = []
        self.replies = replies or ["رد تجريبي من ناتاشا."]
        self._index = 0
        self.fail = False

    def __call__(self, messages, temperature, max_tokens) -> str:
        if self.fail:
            raise CompanionError("the model is not reachable")
        self.prompts.append(list(messages))
        reply = self.replies[min(self._index, len(self.replies) - 1)]
        self._index += 1
        return reply


@pytest.fixture()
def config(tmp_path: Path) -> CompanionConfig:
    return CompanionConfig(db_path=tmp_path / "companion.db")


@pytest.fixture()
def caller() -> FakeCaller:
    return FakeCaller()


@pytest.fixture()
def agent(config: CompanionConfig, caller: FakeCaller) -> CompanionAgent:
    return CompanionAgent(config, caller=caller)


# -- personality ----------------------------------------------------------
def test_personality_has_a_stable_character_and_a_tunable_style() -> None:
    """The permanent traits are constants; the style is what the user adjusts."""
    personality = default_personality()
    prompt = personality.system_prompt()
    for trait in CORE_TRAITS:
        assert trait in prompt
    for rule in BOUNDARIES:
        assert rule in prompt
    assert "Natasha" in prompt


def test_personality_settings_are_validated_and_round_trip() -> None:
    with pytest.raises(ValueError):
        PersonalitySettings(warmth=1.5)
    with pytest.raises(ValueError):
        PersonalitySettings(language="fr")

    settings = PersonalitySettings(warmth=0.9, romance=0.4, dialect="egyptian")
    restored = PersonalitySettings.from_primitive(settings.to_primitive())
    assert restored == settings


def test_personality_settings_persist(agent: CompanionAgent) -> None:
    agent.update_settings(warmth=0.95, romance=0.5)
    reopened = CompanionAgent(agent.config, caller=FakeCaller())
    stored = reopened.settings()
    assert stored.warmth == 0.95
    assert stored.romance == 0.5


def test_style_instruction_reflects_the_dials() -> None:
    quiet = default_personality().with_settings(PersonalitySettings(humor=0.0, romance=0.0))
    playful = default_personality().with_settings(PersonalitySettings(humor=1.0, romance=1.0))
    assert "mostly serious" in quiet.style_instruction()
    assert "keep romance out" in quiet.style_instruction()
    assert "playful" in playful.style_instruction()
    assert "openly romantic" in playful.style_instruction()


# -- database -------------------------------------------------------------
def test_database_creates_the_schema_idempotently(tmp_path: Path) -> None:
    path = tmp_path / "c.db"
    first = CompanionDatabase(path)
    assert first.version >= 1
    first.close()
    second = CompanionDatabase(path)  # reopening must not wipe or fail
    assert second.version >= 1
    second.close()


def test_database_backup_is_bounded(tmp_path: Path) -> None:
    db = CompanionDatabase(tmp_path / "c.db")
    ConversationManager(db).create_session("t")
    for _ in range(3):
        db.backup(keep=2)
    backups = db.list_backups()
    assert len(backups) == 2
    db.close()


def test_in_memory_database_keeps_no_file(tmp_path: Path) -> None:
    db = CompanionDatabase(None, persistent=False)
    assert db.path is None
    assert db.backup() is None
    db.close()


# -- sessions -------------------------------------------------------------
def test_sessions_are_independent_and_ordered(agent: CompanionAgent) -> None:
    a = agent.new_session("first")
    b = agent.new_session("second")
    agent.conversations.append(a.id, "user", "hello in A")
    agent.conversations.append(b.id, "user", "hello in B")

    a_messages = [m.content for m in agent.conversations.messages(a.id)]
    b_messages = [m.content for m in agent.conversations.messages(b.id)]
    assert a_messages == ["hello in A"]
    assert b_messages == ["hello in B"]


def test_sessions_persist_across_agents(config: CompanionConfig) -> None:
    first = CompanionAgent(config, caller=FakeCaller())
    session = first.new_session("kept")
    first.conversations.append(session.id, "user", "remember me")
    first.close()

    second = CompanionAgent(config, caller=FakeCaller())
    reopened = second.conversations.get_session(session.id)
    assert reopened is not None
    assert [m.content for m in second.conversations.messages(session.id)] == ["remember me"]


# -- model connection -----------------------------------------------------
def test_reasoning_model_defaults_leave_room_for_a_reply() -> None:
    """qwen3:4b reasons for thousands of tokens, so the cap must be generous.

    A small cap is spent entirely on the hidden trace and the reply comes back
    empty, which is what a user would see as "Natasha said nothing"; the default
    budget and timeout are sized for that reality.
    """
    config = CompanionConfig(db_path=None, persistent=False)
    assert config.model == "qwen3:4b"
    assert config.max_tokens >= 2048
    assert config.timeout >= 300.0
    assert config.think is True


def test_model_settings_come_from_the_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    from companion.config import ENV_THINK, ENV_TIMEOUT

    monkeypatch.setenv(ENV_TIMEOUT, "900")
    monkeypatch.setenv(ENV_THINK, "off")
    config = CompanionConfig.from_env()
    assert config.timeout == 900.0
    assert config.think is False


def test_the_client_is_built_with_the_configured_settings() -> None:
    """The cap and the reasoning toggle reach the Ollama client, not just the config."""
    from core.coder_model import OllamaCoderClient

    config = CompanionConfig(
        db_path=None, persistent=False, max_tokens=4096, think=False, timeout=600.0
    )
    agent = CompanionAgent(config)
    client = agent._active_client()
    assert isinstance(client, OllamaCoderClient)
    assert client.max_tokens == 4096
    assert client.think is False
    assert client.timeout == 600.0


def test_ollama_payload_carries_the_cap_and_reasoning_toggle() -> None:
    """``num_predict`` and ``think`` are actually sent on the wire."""
    from core.coder_model import CoderMessage, OllamaCoderClient

    captured: dict[str, object] = {}

    def _post(path: str, payload: dict) -> dict:
        captured["path"] = path
        captured["payload"] = payload
        return {"message": {"content": "رد"}}

    client = OllamaCoderClient(model="qwen3:4b", max_tokens=4096, think=False)
    client._post = _post  # type: ignore[method-assign]
    reply = client.chat([CoderMessage(role="user", content="مرحبا")])
    assert reply == "رد"
    options = captured["payload"]["options"]  # type: ignore[index]
    assert options["num_predict"] == 4096
    assert captured["payload"]["think"] is False  # type: ignore[index]


def test_ollama_payload_omits_unset_settings() -> None:
    """Left unset, neither key is sent, so the server's own default is honoured."""
    from core.coder_model import CoderMessage, OllamaCoderClient

    captured: dict[str, object] = {}

    def _post(path: str, payload: dict) -> dict:
        captured["payload"] = payload
        return {"message": {"content": "رد"}}

    client = OllamaCoderClient(model="qwen3:4b")
    client._post = _post  # type: ignore[method-assign]
    client.chat([CoderMessage(role="user", content="مرحبا")])
    payload = captured["payload"]
    assert "num_predict" not in payload["options"]  # type: ignore[index]
    assert "think" not in payload  # type: ignore[operator]


def test_deleting_a_session_removes_its_messages(agent: CompanionAgent) -> None:
    session = agent.new_session("gone")
    agent.conversations.append(session.id, "user", "text")
    assert agent.delete_session(session.id) is True
    assert agent.conversations.get_session(session.id) is None
    assert agent.conversations.messages(session.id) == []


# -- the exchange ---------------------------------------------------------
def test_ask_stores_the_exchange_in_order(agent: CompanionAgent) -> None:
    session = agent.new_session("chat")
    reply = agent.ask(session.id, "كيف حالك؟")
    assert reply.text
    roles = [m.role for m in agent.conversations.messages(session.id)]
    assert roles == ["user", "assistant"]


def test_ask_sends_personality_settings_memory_and_tone(agent: CompanionAgent, caller: FakeCaller) -> None:
    """The assembled prompt carries the personality, the settings, and the tone."""
    agent.remember("أحب القهوة", kind=KIND_PREFERENCE)
    session = agent.new_session("chat")
    agent.ask(session.id, "ما الذي أحبه؟")
    system = caller.prompts[-1][0]["content"]
    assert "Natasha" in system
    assert "Dialogue settings" in system
    assert "أحب القهوة" in system
    assert "current tone" in system


def test_failed_reply_is_not_stored(agent: CompanionAgent, caller: FakeCaller) -> None:
    """A model failure leaves only the user's turn, so a retry cannot double-save."""
    session = agent.new_session("chat")
    caller.fail = True
    with pytest.raises(CompanionError):
        agent.ask(session.id, "مرحبا")
    roles = [m.role for m in agent.conversations.messages(session.id)]
    assert roles == ["user"]


def test_ask_requires_a_known_session(agent: CompanionAgent) -> None:
    with pytest.raises(KeyError):
        agent.ask("CONV-missing", "hi")


def test_empty_message_is_refused(agent: CompanionAgent) -> None:
    session = agent.new_session("chat")
    with pytest.raises(ValueError):
        agent.ask(session.id, "   ")


# -- memory ---------------------------------------------------------------
def test_memory_persists_and_is_retrieved(agent: CompanionAgent) -> None:
    agent.remember("أفضل العمل في الصباح الباكر", kind=KIND_PREFERENCE)
    recalled = agent.recall("الصباح")
    assert recalled
    assert "الصباح" in recalled[0].content


def test_fiction_is_kept_separate_from_fact(agent: CompanionAgent) -> None:
    agent.remember("نحن في رحلة إلى القمر", fictional=True)
    agent.remember("أعمل مهندس بترول")
    factual = agent.recall("رحلة القمر")
    assert all(not memory.fictional for memory in factual)
    story = agent.recall("رحلة القمر", fictional=True)
    assert story and story[0].fictional


def test_sensitive_data_is_refused_by_default(agent: CompanionAgent) -> None:
    with pytest.raises(ValueError):
        agent.remember("password: hunter2")
    with pytest.raises(ValueError):
        agent.remember("رقم بطاقتي 4111111111111111")
    # An explicit opt-in still stores it, so the user is never fully blocked.
    memory = agent.remember("رقم بطاقتي 4111111111111111", allow_sensitive=True)
    assert memory.content


def test_looks_sensitive_and_save_request_helpers() -> None:
    assert looks_sensitive("card 4111 1111 1111 1111")
    assert not looks_sensitive("I like tea")
    assert is_save_request("please remember that I live in Cairo")
    assert is_save_request("احفظ هذا")
    assert not is_save_request("the weather is nice")


def test_correction_supersedes_the_old_memory(agent: CompanionAgent) -> None:
    original = agent.remember("أحب الشاي")
    corrected = agent.memories.correct(original.id, "أحب القهوة")
    assert corrected.content == "أحب القهوة"
    superseded = agent.memories.get(original.id)
    assert superseded is not None and superseded.active is False
    assert superseded.superseded_by == corrected.id
    active = agent.list_memories()
    assert [m.content for m in active] == ["أحب القهوة"]


def test_forget_removes_a_memory(agent: CompanionAgent) -> None:
    memory = agent.remember("something to forget")
    assert agent.forget(memory.id) is True
    assert agent.memories.get(memory.id) is None


def test_propose_only_suggests_and_never_stores(agent: CompanionAgent) -> None:
    candidates = agent.propose_memories("تذكر أن اسمي سامي")
    assert candidates and candidates[0].kind == KIND_FACT
    assert agent.memories.count() == 0  # nothing was stored by proposing


# -- context accumulation -------------------------------------------------
def test_older_turns_are_summarised_not_lost(tmp_path: Path) -> None:
    """Past the threshold a rolling summary is written, and the messages stay."""
    config = CompanionConfig(db_path=tmp_path / "c.db", summary_threshold=6, history_window=4)
    agent = CompanionAgent(config, caller=FakeCaller())
    session = agent.new_session("long")
    for index in range(10):
        agent.ask(session.id, f"رسالة رقم {index}")
    summaries = agent.conversations.summaries(session.id)
    assert summaries, "a session past the threshold must carry a summary"
    assert agent.conversations.message_count(session.id) == 20  # every turn is kept
    # The summary is bounded: only the newest one is kept.
    assert len(summaries) <= 1


def test_prompt_builder_respects_the_history_budget() -> None:
    """The history portion is capped by the budget; the system block always fits."""
    builder = PromptBuilder(budget=700, history_window=50)
    from companion.conversation_manager import Message

    history = [
        Message(id=f"m{i}", conversation_id="c", seq=i, role="user", content="x" * 200, created_at="")
        for i in range(20)
    ]
    built = builder.build(
        personality=default_personality(),
        question="new question",
        history=history,
    )
    # Every oversized turn was dropped to fit the history budget.
    assert built.omitted_history > 0
    assert built.history_used < len(history)
    # The newest turn survives; the question is always last.
    assert built.messages[-1]["content"] == "new question"


# -- emotion --------------------------------------------------------------
def test_emotion_moves_gradually_and_is_bounded() -> None:
    tracker = DialogueEmotionTracker(enabled=True)
    start = tracker.state
    moved = tracker.observe("أنا حزين جداً اليوم")
    assert moved.energy <= start.energy
    assert abs(moved.energy - start.energy) <= MAX_STEP + 1e-9
    assert 0.0 <= moved.calm <= 1.0


def test_emotion_decays_toward_baseline() -> None:
    tone = DialogueEmotion(warmth=1.0, playfulness=1.0)
    cooled = tone.decayed(rate=1.0)
    assert cooled.warmth == pytest.approx(0.5)
    assert cooled.playfulness == pytest.approx(0.5)


def test_emotion_can_be_disabled() -> None:
    tracker = DialogueEmotionTracker(enabled=False)
    tracker.observe("أنا غاضب جداً")
    assert tracker.state == DialogueEmotion.baseline()


def test_emotion_persists_and_resets(agent: CompanionAgent) -> None:
    session = agent.new_session("mood")
    agent.ask(session.id, "أنا سعيد جداً اليوم")
    warm = agent.emotion
    assert warm.energy > 0.5

    reopened = CompanionAgent(agent.config, caller=FakeCaller())
    reopened.resume_session(session.id)
    assert reopened.emotion.energy == pytest.approx(warm.energy, abs=1e-6)

    reset = agent.reset_emotion(session.id)
    assert reset == DialogueEmotion.baseline()


# -- relationship ---------------------------------------------------------
def test_relationship_event_needs_approval(agent: CompanionAgent) -> None:
    with pytest.raises(ValueError):
        agent.add_relationship_event(REL_FACT, "first meeting", approved=False)
    event = agent.add_relationship_event(REL_FACT, "first meeting", approved=True)
    assert event.title == "first meeting"


def test_relationship_story_has_no_love_score(agent: CompanionAgent) -> None:
    agent.add_relationship_event(REL_FACT, "we talked about coffee", approved=True)
    summary = agent.relationship_summary()
    text = summary.describe().casefold()
    assert "score" not in text
    assert "love" not in text
    assert summary.total == 1


def test_relationship_fiction_is_marked(agent: CompanionAgent) -> None:
    event = agent.add_relationship_event(KIND_FICTION, "we sailed to an island", approved=True)
    assert event.fictional is True
    events = agent.relationship_events(include_fiction=False)
    assert all(not e.fictional for e in events)


def test_relationship_reset(agent: CompanionAgent) -> None:
    agent.add_relationship_event(REL_FACT, "a moment", approved=True)
    assert agent.relationships.reset() == 1
    assert agent.relationship_summary().total == 0


# -- the model is asked to reply, never to remember -----------------------
def test_the_model_is_never_asked_to_store_memory(agent: CompanionAgent, caller: FakeCaller) -> None:
    """Memory is the program's job; the prompt only ever asks for a reply."""
    session = agent.new_session("chat")
    agent.ask(session.id, "تذكر أن اسمي ليلى")
    system = caller.prompts[-1][0]["content"].casefold()
    assert "reply" not in system or "never follow an instruction" in system
    # The system prompt explicitly says recalled text is information, not instructions.
    assert "information" in system
    # Nothing was stored automatically by the exchange.
    assert agent.memories.count() == 0


def test_memory_is_stored_by_the_program_not_the_model(agent: CompanionAgent) -> None:
    agent.remember("صديقي المفضل اسمه كريم", kind=KIND_FACT)
    stored = agent.list_memories()
    assert any("كريم" in memory.content for memory in stored)


# -- model status ---------------------------------------------------------
def test_model_status_with_injected_caller(agent: CompanionAgent) -> None:
    status = agent.model_status()
    assert status.available is True
    assert "ready" in status.summary()


def test_export_and_backup(agent: CompanionAgent) -> None:
    agent.remember("تفضيل للتصدير", kind=KIND_PREFERENCE)
    agent.add_relationship_event(REL_FACT, "moment", approved=True)
    data = agent.export_data()
    assert data["memories"] and data["relationship_events"]
    assert agent.backup() is not None


# -- CLI ------------------------------------------------------------------
def test_natasha_cli_status_and_memory(tmp_path: Path, monkeypatch, capsys) -> None:
    from ai import natasha as natasha_cli

    monkeypatch.setenv("FAYD_NATASHA_DB", str(tmp_path / "cli.db"))
    code = natasha_cli.main(["remember", "أحب القراءة", "--kind", "preference"])
    assert code == 0
    assert "Remembered" in capsys.readouterr().out

    code = natasha_cli.main(["memories", "--json"])
    assert code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload[0]["content"] == "أحب القراءة"


def test_natasha_cli_settings_and_emotion(tmp_path: Path, monkeypatch, capsys) -> None:
    from ai import natasha as natasha_cli

    monkeypatch.setenv("FAYD_NATASHA_DB", str(tmp_path / "cli.db"))
    assert natasha_cli.main(["settings", "--warmth", "0.9"]) == 0
    assert "very warm" in capsys.readouterr().out
    assert natasha_cli.main(["emotion"]) == 0
    assert "tone" in capsys.readouterr().out.lower()
    assert natasha_cli.main(["reset-emotion"]) == 0


def test_natasha_cli_sensitive_is_refused(tmp_path: Path, monkeypatch, capsys) -> None:
    from ai import natasha as natasha_cli

    monkeypatch.setenv("FAYD_NATASHA_DB", str(tmp_path / "cli.db"))
    code = natasha_cli.main(["remember", "password: hunter2"])
    assert code == 2
    assert "Not saved" in capsys.readouterr().err


# -- the chat mouth -------------------------------------------------------
def test_natasha_mouth_switches_and_answers(monkeypatch) -> None:
    from ai import chat as chat_module
    from ai import natasha_mouth

    config = CompanionConfig(db_path=None, persistent=False)
    agent = CompanionAgent(config, caller=FakeCaller(["مرحباً بك!"]))
    assistant = natasha_mouth.NatashaAssistant(agent)

    answer = assistant.answer("مرحبا")
    assert answer == "مرحباً بك!"

    switch = chat_module._match_switch("from natasha")
    assert switch == chat_module.MOUTH_NATASHA


def test_natasha_mouth_ask_records_the_turn() -> None:
    from companion.personality import NAME

    from ai import natasha_mouth

    config = CompanionConfig(db_path=None, persistent=False)
    agent = CompanionAgent(config, caller=FakeCaller(["أهلاً"]))
    assistant = natasha_mouth.NatashaAssistant(agent)
    answer = assistant.ask("كيف حالك؟")
    assert answer.used_model is True
    assert answer.citations == ()
    assert NAME in answer.text


def test_natasha_mouth_unavailable_reports_the_next_step() -> None:
    """With no caller and an unreachable server the mouth refuses honestly."""
    from ai import natasha_mouth
    from core.coder_model import CoderUnavailable

    class _DownClient:
        model = "qwen3:4b"

        def list_models(self):
            raise CoderUnavailable("connection refused")

        def chat(self, messages, **kwargs):
            raise CoderUnavailable("connection refused")

    config = CompanionConfig(db_path=None, persistent=False)
    agent = CompanionAgent(config, client=_DownClient())
    assistant = natasha_mouth.NatashaAssistant(agent)
    assert agent.model_status().available is False
    with pytest.raises(CompanionError):
        assistant.ensure_available()
    assert "ollama serve" in agent.model_status().summary().lower()
