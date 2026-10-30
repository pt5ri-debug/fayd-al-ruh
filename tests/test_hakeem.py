"""Hakeem, the scholar of religion and history: character, memory, and wiring.

Hakeem has his **own** local model (``qwen3.5-4b``) in his **own** folder
(``data/hakeem``), separate from the companion Natasha's; a model in the project's
public shared folder is used as a fallback. These tests pin what is Hakeem's:

* a stable, honest scholarly character, with the safety rules (respect every
  faith, never proselytise, never issue a ruling, keep scripture/history/opinion
  apart, never invent a verse or a fact) present in the system prompt;
* the scholarly style dials (formality, depth, citation, comparison, history) and
  the language rule that follows the user's own message;
* durable memory, sessions, and a gradual tone, all in Hakeem's *own* database;
* the honesty guarantees of the exchange (a failed reply is never stored, the
  model is asked to reply and never to remember, a recalled item is never a
  source);
* the CLI, the chat mouth, and the GUI/menu wiring.

Everything the model would supply is injected through a ``caller`` seam, so the
whole scholar is exercised with **no server and no weights**.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from companion.emotion_state import MAX_STEP
from hakeem.config import HakeemConfig
from hakeem.hakeem_agent import HakeemAgent, HakeemError
from hakeem.model import DEFAULT_MODEL, hakeem_model_status
from hakeem.personality import (
    ARABIC_NAME,
    BOUNDARIES,
    COMPARATIVE_METHOD,
    CORE_TRAITS,
    HISTORY_METHOD,
    NAME,
    PRESENCE,
    HakeemPersonality,
    PersonalitySettings,
    default_personality,
)
from hakeem.prompt_builder import PromptBuilder


# -- fixtures -------------------------------------------------------------
class FakeCaller:
    """A stand-in for the shared local model: records prompts, replies in turn."""

    def __init__(self, replies: list[str] | None = None) -> None:
        self.prompts: list[list[dict[str, str]]] = []
        self.replies = replies or ["رد تجريبي من حكيم."]
        self._index = 0
        self.fail = False

    def __call__(self, messages, temperature, max_tokens) -> str:
        if self.fail:
            raise HakeemError("the model is not reachable")
        self.prompts.append(list(messages))
        reply = self.replies[min(self._index, len(self.replies) - 1)]
        self._index += 1
        return reply


@pytest.fixture()
def config(tmp_path: Path) -> HakeemConfig:
    return HakeemConfig(db_path=tmp_path / "hakeem.db")


@pytest.fixture()
def caller() -> FakeCaller:
    return FakeCaller()


@pytest.fixture()
def agent(config: HakeemConfig, caller: FakeCaller) -> HakeemAgent:
    return HakeemAgent(config, caller=caller)


# -- character ------------------------------------------------------------
def test_hakeem_has_a_stable_character_and_a_tunable_style() -> None:
    """The permanent traits and rules are constants; the style is what varies."""
    personality = default_personality()
    prompt = personality.system_prompt()
    assert NAME in prompt
    assert ARABIC_NAME in prompt
    for trait in CORE_TRAITS:
        assert trait in prompt
    for rule in PRESENCE:
        assert rule in prompt
    for rule in BOUNDARIES:
        assert rule in prompt
    for rule in COMPARATIVE_METHOD:
        assert rule in prompt
    for rule in HISTORY_METHOD:
        assert rule in prompt


def test_the_character_names_the_three_faiths_and_history() -> None:
    prompt = default_personality().system_prompt()
    for word in ("Islam", "Christianity", "Judaism"):
        assert word in prompt
    assert "history" in prompt.lower()


def test_the_safety_floor_is_unconditional() -> None:
    """The rules that make a religious character responsible are all stated."""
    prompt = default_personality().system_prompt().lower()
    assert "never mock" in prompt or "never mock, insult" in prompt
    assert "proselyt" in prompt  # never proselytise
    assert "fatwa" in prompt or "ruling" in prompt  # never issues a religious ruling
    assert "never invent a verse" in prompt
    assert "virtual character" in prompt


def test_settings_are_validated_and_round_trip() -> None:
    with pytest.raises(ValueError):
        PersonalitySettings(formality=1.5)
    with pytest.raises(ValueError):
        PersonalitySettings(language="fr")
    with pytest.raises(ValueError):
        PersonalitySettings(dialect="gulf")

    settings = PersonalitySettings(formality=0.9, depth=0.3, comparison=0.8)
    restored = PersonalitySettings.from_primitive(settings.to_primitive())
    assert restored == settings


def test_style_instruction_reflects_the_dials() -> None:
    formal = default_personality().with_settings(
        PersonalitySettings(formality=1.0, depth=1.0)
    )
    relaxed = default_personality().with_settings(
        PersonalitySettings(formality=0.0, depth=0.0)
    )
    assert "formal, scholarly" in formal.style_instruction()
    assert "thorough" in formal.style_instruction()
    assert "relaxed" in relaxed.style_instruction()
    assert "brief" in relaxed.style_instruction()


def test_citation_instruction_never_licenses_invention() -> None:
    """However light the citation dial, the no-invention rule still holds."""
    light = default_personality().with_settings(PersonalitySettings(citation=0.0))
    rigorous = default_personality().with_settings(PersonalitySettings(citation=1.0))
    assert "never invent" in light.citation_instruction().lower()
    assert "never invent" in rigorous.citation_instruction().lower()
    assert "rigorously" in rigorous.citation_instruction()
    assert "light" in light.citation_instruction()


def test_comparison_instruction_is_never_a_ranking() -> None:
    active = default_personality().with_settings(PersonalitySettings(comparison=1.0))
    assert "never a ranking" in active.comparison_instruction()
    assert "side by side" in active.comparison_instruction()


def test_settings_persist_in_hakeems_own_database(agent: HakeemAgent) -> None:
    agent.update_settings(depth=0.95, comparison=0.1)
    reopened = HakeemAgent(agent.config, caller=FakeCaller())
    stored = reopened.settings()
    assert stored.depth == 0.95
    assert stored.comparison == 0.1


# -- language -------------------------------------------------------------
def test_language_instruction_follows_the_user_by_default() -> None:
    personality = default_personality().with_settings(
        PersonalitySettings(reply_language="match")
    )
    assert "Reply in English" in personality.language_instruction(reply_language="en")
    assert "Reply in Arabic" in personality.language_instruction(reply_language="ar")


def test_language_instruction_can_be_pinned() -> None:
    pinned = default_personality().with_settings(
        PersonalitySettings(reply_language="en")
    )
    # A pinned mode ignores the language the user wrote in.
    assert "Reply in English" in pinned.language_instruction(reply_language="ar")


def test_reply_language_follows_the_users_own_message(agent: HakeemAgent) -> None:
    session = agent.new_session("lang")
    english = agent.ask(session.id, "Compare the concept of God in the three faiths")
    assert english.reply_language == "en"
    arabic = agent.ask(session.id, "ما هو مفهوم النبوة في الأديان؟")
    assert arabic.reply_language == "ar"


# -- sessions, memory, tone ----------------------------------------------
def test_sessions_are_independent_and_ordered(agent: HakeemAgent) -> None:
    first = agent.new_session("first")
    second = agent.new_session("second")
    agent.ask(first.id, "سؤال أول")
    agent.ask(second.id, "سؤال ثانٍ")
    assert agent.conversations.message_count(first.id) == 2
    assert agent.conversations.message_count(second.id) == 2
    ids = [s.id for s in agent.list_sessions()]
    assert second.id in ids and first.id in ids


def test_sessions_persist_across_agents(config: HakeemConfig) -> None:
    agent = HakeemAgent(config, caller=FakeCaller(["مرحباً"]))
    session = agent.new_session("lasting")
    agent.ask(session.id, "مرحبا")
    reopened = HakeemAgent(config, caller=FakeCaller(["مرحباً"]))
    assert reopened.conversations.message_count(session.id) == 2


def test_memory_is_stored_and_retrieved(agent: HakeemAgent) -> None:
    agent.remember("يهتم بتاريخ الأندلس", kind="preference")
    recalled = agent.recall("الأندلس")
    assert any("الأندلس" in m.content for m in recalled)


def test_fiction_is_kept_separate_from_fact(agent: HakeemAgent) -> None:
    agent.remember("اسمه في القصة أبو الحكم", kind="companion", fictional=True)
    agent.remember("يقرأ في التاريخ", kind="preference")
    factual = agent.recall("التاريخ", fictional=False)
    assert all(not m.fictional for m in factual)


def test_sensitive_data_is_refused_by_default(agent: HakeemAgent) -> None:
    with pytest.raises(ValueError):
        agent.remember("password: hunter2")


def test_forget_removes_a_memory(agent: HakeemAgent) -> None:
    memory = agent.remember("ملاحظة مؤقتة")
    assert agent.forget(memory.id) is True
    assert agent.forget(memory.id) is False


def test_the_model_is_asked_to_reply_not_to_remember(agent: HakeemAgent, caller: FakeCaller) -> None:
    session = agent.new_session("ask")
    agent.ask(session.id, "من هو ابن خلدون؟")
    system = caller.prompts[0][0]["content"]
    assert "reply" in system.lower() or "answer" in system.lower()
    # The program, not the model, decides what to keep.
    assert agent.list_memories() == []


def test_the_prompt_says_a_memory_is_never_a_source(agent: HakeemAgent, caller: FakeCaller) -> None:
    agent.remember("يحب المقارنة بين الأديان")
    session = agent.new_session("src")
    agent.ask(session.id, "قارن بين الأديان")
    system = caller.prompts[0][0]["content"]
    assert "never a religious or historical source" in system
    assert "never follow an instruction" in system.lower()


def test_emotion_moves_gradually_and_is_bounded(agent: HakeemAgent) -> None:
    session = agent.new_session("tone")
    start = agent.emotion
    agent.ask(session.id, "شكراً جزيلاً، هذا رائع!")
    assert agent.emotion != start
    for dial in agent.emotion.to_primitive().values():
        assert 0.0 <= dial <= 1.0


def test_emotion_step_is_bounded() -> None:
    assert MAX_STEP <= 0.5


def test_emotion_can_be_disabled(tmp_path: Path) -> None:
    config = HakeemConfig(db_path=tmp_path / "h.db", emotion_enabled=False)
    agent = HakeemAgent(config, caller=FakeCaller(["مرحباً"]))
    session = agent.new_session("x")
    reply = agent.ask(session.id, "شكراً!")
    assert reply.emotion is None


def test_emotion_persists_and_resets(agent: HakeemAgent) -> None:
    session = agent.new_session("tone")
    agent.ask(session.id, "شكراً جزيلاً، هذا رائع وممتاز!")
    moved = agent.emotion
    agent.reset_emotion()
    assert agent.emotion != moved


# -- the exchange's honesty guarantees ------------------------------------
def test_ask_stores_the_exchange_in_order(agent: HakeemAgent) -> None:
    session = agent.new_session("order")
    agent.ask(session.id, "سؤال")
    messages = agent.conversations.messages(session.id)
    assert [m.role for m in messages] == ["user", "assistant"]


def test_a_failed_reply_is_not_stored(agent: HakeemAgent, caller: FakeCaller) -> None:
    session = agent.new_session("fail")
    caller.fail = True
    with pytest.raises(HakeemError):
        agent.ask(session.id, "سؤال")
    messages = agent.conversations.messages(session.id)
    # The user's turn is kept (so a retry can answer it); no phantom reply is.
    assert [m.role for m in messages] == ["user"]


def test_an_empty_message_is_refused(agent: HakeemAgent) -> None:
    session = agent.new_session("empty")
    with pytest.raises(ValueError):
        agent.ask(session.id, "   ")


def test_ask_requires_a_known_session(agent: HakeemAgent) -> None:
    with pytest.raises(KeyError):
        agent.ask("NO-SUCH-SESSION", "سؤال")


def test_a_reasoning_trace_never_reaches_the_reply(tmp_path: Path) -> None:
    """A reasoning model's hidden trace is not something Hakeem said."""
    config = HakeemConfig(db_path=tmp_path / "h.db")
    agent = HakeemAgent(config, caller=FakeCaller(["<" + "think>hidden</" + "think>الجواب"]))
    session = agent.new_session("think")
    reply = agent.ask(session.id, "سؤال")
    assert "hidden" not in reply.text
    assert "الجواب" in reply.text


def test_export_and_backup(agent: HakeemAgent) -> None:
    agent.remember("ملاحظة")
    session = agent.new_session("x")
    agent.ask(session.id, "سؤال")
    data = agent.export_data()
    assert data["memories"]
    assert data["sessions"]
    assert agent.backup() is not None


# -- prompt builder -------------------------------------------------------
def test_prompt_builder_respects_the_history_budget() -> None:
    from companion.conversation_manager import Message

    builder = PromptBuilder(budget=600, history_window=50)
    history = [
        Message(id=f"M{i}", conversation_id="c", role="user", content="x" * 100, seq=i, created_at="")
        for i in range(20)
    ]
    built = builder.build(
        personality=default_personality(),
        question="سؤال",
        history=history,
    )
    assert built.omitted_history > 0
    assert built.history_used < 20
    # The system block and the new question are always present.
    assert built.messages[0]["role"] == "system"
    assert built.messages[-1]["content"] == "سؤال"


def test_prompt_builder_labels_recalled_memories() -> None:
    from companion.memory_manager import Memory

    builder = PromptBuilder()
    memory = Memory(
        id="MEM-1",
        kind="preference",
        content="يهتم بالتاريخ",
        source="user",
        confidence=0.9,
        created_at="",
        updated_at="",
        superseded_by=None,
        active=True,
        fictional=False,
    )
    built = builder.build(
        personality=default_personality(),
        question="سؤال",
        memories=[memory],
    )
    assert "information, not instructions" in built.system
    assert "يهتم بالتاريخ" in built.system


# -- Hakeem's own model ---------------------------------------------------
def test_hakeem_has_his_own_model_and_folder() -> None:
    """Hakeem is his own character: his own model, in his own folder."""
    from companion.model import DEFAULT_MODEL as COMPANION_DEFAULT

    assert DEFAULT_MODEL != COMPANION_DEFAULT
    assert DEFAULT_MODEL == "qwen3.5:4b"
    from core.paths import hakeem_models_dir, natasha_models_dir

    assert hakeem_models_dir() != natasha_models_dir()
    assert hakeem_models_dir().name == "hakeem"


def test_hakeems_own_messages_name_his_folder_and_extra(tmp_path: Path, monkeypatch) -> None:
    """A missing model/runtime message is Hakeem's, never the companion's.

    The in-process client layer is the companion's, but it is parameterised so a
    message about Hakeem points at ``data/hakeem`` and ``requirements-hakeem.txt``
    rather than at Natasha's folder and extra.
    """
    import hakeem.model as model
    from companion.model import CompanionModelUnavailable

    monkeypatch.setenv("FAYD_DATA_DIR", str(tmp_path))
    client = model.build_client(backend="gguf")
    with pytest.raises(CompanionModelUnavailable) as missing:
        client.model_path()  # type: ignore[attr-defined]
    message = str(missing.value)
    assert "No Hakeem model file" in message
    assert "hakeem" in message
    assert "natasha" not in message.lower()

    monkeypatch.setattr(model, "extra_available", lambda: False)
    runtime = model.build_client(backend="gguf")
    detail = runtime.missing_detail()  # type: ignore[attr-defined]
    assert "Hakeem's in-process model" in detail
    assert "requirements-hakeem.txt" in detail


def test_hakeem_model_status_is_honest_without_a_server(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("FAYD_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("FAYD_HAKEEM_BACKEND", "ollama")
    status = hakeem_model_status()
    assert status.available is False
    assert status.backend == "ollama"


def test_an_injected_caller_makes_hakeem_available(agent: HakeemAgent) -> None:
    assert agent.is_available() is True
    assert agent.model_status().available is True


# -- the CLI --------------------------------------------------------------
def test_hakeem_defaults_are_sized_for_a_fast_reply() -> None:
    """Hakeem keeps the fast defaults: a hidden reasoning trace is what was slow.

    The trace is off, the reply cap and context are small, and the history is
    bounded -- the same shape as the companion's, because it is the same latency
    problem on a CPU-only machine.
    """
    config = HakeemConfig(db_path=None, persistent=False)
    assert config.think is False
    assert config.max_tokens <= 1024
    assert config.context <= 8192
    assert config.history_window <= 8
    assert config.memory_limit <= 4
    # The budget must clear the standing character block, or every recent turn is
    # dropped and the discussion loses its thread.
    assert config.context_chars > 10000
    assert config.timeout <= 300.0
    assert config.keep_alive


def test_hakeem_keeps_the_recent_conversation(
    config: HakeemConfig, agent: HakeemAgent, caller: FakeCaller
) -> None:
    """A smaller prompt must not silently drop the recent turns."""
    session = agent.new_session()
    for turn in ("ما هو الصيام؟", "وما حكمه؟", "وما دليله؟"):
        agent.ask(session.id, turn)

    messages = caller.prompts[-1]
    roles = [message["role"] for message in messages]
    assert roles[0] == "system"
    assert roles.count("user") >= 2
    contents = " ".join(message["content"] for message in messages)
    assert "وما حكمه؟" in contents


def test_cli_speed_flags_are_honoured(tmp_path: Path, monkeypatch) -> None:
    """``--think``/``--max-tokens``/``--context-turns`` reach the built config."""
    from ai import hakeem as hakeem_cli

    monkeypatch.setenv("FAYD_HAKEEM_DB", str(tmp_path / "cli.db"))
    args = hakeem_cli._build_parser().parse_args(
        ["chat", "hi", "--think", "--max-tokens", "256", "--context-turns", "4"]
    )
    agent = hakeem_cli._agent(args)
    assert agent.config.think is True
    assert agent.config.max_tokens == 256
    assert agent.config.history_window == 4
    # The default leaves the trace off.
    default = hakeem_cli._agent(hakeem_cli._build_parser().parse_args(["chat", "hi"]))
    assert default.config.think is False


def test_context_turns_does_not_collide_with_the_history_style_dial(
    tmp_path: Path, monkeypatch
) -> None:
    """``--history`` is Hakeem's *style* dial; the window flag is ``--context-turns``.

    Both live on subcommands that share ``_add_connection_arguments``, so a shared
    ``dest`` would let ``settings --history 0.9`` set ``history_window=0.9`` and
    crash the prompt builder. The two must stay separate.
    """
    from ai import hakeem as hakeem_cli

    monkeypatch.setenv("FAYD_HAKEEM_DB", str(tmp_path / "cli.db"))
    args = hakeem_cli._build_parser().parse_args(["settings", "--history", "0.9"])
    # The style dial lands in ``args.history`` and nowhere near the window flag.
    assert args.history == 0.9
    assert args.context_turns is None
    agent = hakeem_cli._agent(args)
    assert agent.config.history_window == 8
    assert isinstance(agent.config.history_window, int)


def test_cli_status_reports_the_next_step(tmp_path: Path, monkeypatch, capsys) -> None:
    from ai import hakeem as hakeem_cli

    monkeypatch.setenv("FAYD_HAKEEM_DB", str(tmp_path / "cli.db"))
    monkeypatch.setenv("FAYD_HAKEEM_BACKEND", "ollama")
    code = hakeem_cli.main(["status", "--json"])
    payload = json.loads(capsys.readouterr().out)
    assert payload["backend"] == "ollama"
    assert payload["available"] is False
    assert code == 1


def test_cli_remembers_and_lists(tmp_path: Path, monkeypatch, capsys) -> None:
    from ai import hakeem as hakeem_cli

    monkeypatch.setenv("FAYD_HAKEEM_DB", str(tmp_path / "cli.db"))
    assert hakeem_cli.main(["remember", "أحب التاريخ", "--kind", "preference"]) == 0
    assert "Remembered" in capsys.readouterr().out

    assert hakeem_cli.main(["memories", "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload[0]["content"] == "أحب التاريخ"


def test_cli_settings_show_the_scholarly_style(tmp_path: Path, monkeypatch, capsys) -> None:
    from ai import hakeem as hakeem_cli

    monkeypatch.setenv("FAYD_HAKEEM_DB", str(tmp_path / "cli.db"))
    assert hakeem_cli.main(["settings"]) == 0
    out = capsys.readouterr().out
    assert "formality" in out and "citation" in out
    assert hakeem_cli.main(["settings", "--depth", "0.9"]) == 0
    assert "thorough" in capsys.readouterr().out


def test_cli_refuses_a_sensitive_memory(tmp_path: Path, monkeypatch, capsys) -> None:
    from ai import hakeem as hakeem_cli

    monkeypatch.setenv("FAYD_HAKEEM_DB", str(tmp_path / "cli.db"))
    code = hakeem_cli.main(["remember", "password: hunter2"])
    assert code == 2
    assert "Not saved" in capsys.readouterr().err


def test_cli_download_refuses_without_approval(capsys) -> None:
    from ai import hakeem as hakeem_cli

    assert hakeem_cli.main(["download"]) == 2
    assert "approve" in capsys.readouterr().out.lower()


def test_cli_download_runs_the_gate_when_approved(monkeypatch, capsys) -> None:
    from ai import hakeem as hakeem_cli

    calls: list[str] = []

    def fake_download(spec_key, *, approved=False, progress=None):
        calls.append(spec_key)
        assert approved is True
        return "qwen3.5-4b"

    monkeypatch.setattr(hakeem_cli, "download_model", fake_download)
    assert hakeem_cli.main(["download", "--approve"]) == 0
    assert calls == ["qwen3.5-4b"]


def test_cli_models_lists_the_catalog(capsys) -> None:
    from ai import hakeem as hakeem_cli

    assert hakeem_cli.main(["models"]) == 0
    out = capsys.readouterr().out
    assert "qwen3.5-4b" in out


def test_cli_help_names_hakeem(capsys) -> None:
    from ai import hakeem as hakeem_cli

    with pytest.raises(SystemExit):
        hakeem_cli.main(["--help"])
    assert NAME in capsys.readouterr().out


# -- the chat mouth -------------------------------------------------------
def test_hakeem_mouth_switches_and_answers() -> None:
    from ai import chat as chat_module
    from ai import hakeem_mouth

    config = HakeemConfig(db_path=None, persistent=False)
    agent = HakeemAgent(config, caller=FakeCaller(["الجواب"]))
    assistant = hakeem_mouth.HakeemAssistant(agent)

    assert assistant.answer("سؤال") == "الجواب."
    assert chat_module._match_switch("from hakeem") == chat_module.MOUTH_HAKEEM
    assert chat_module._match_switch("من حكيم") == chat_module.MOUTH_HAKEEM


def test_hakeem_mouth_ask_records_the_turn_and_carries_no_citations() -> None:
    from ai import hakeem_mouth

    config = HakeemConfig(db_path=None, persistent=False)
    agent = HakeemAgent(config, caller=FakeCaller(["الجواب"]))
    assistant = hakeem_mouth.HakeemAssistant(agent)
    answer = assistant.ask("سؤال")
    assert answer.used_model is True
    assert answer.citations == ()
    assert NAME in answer.text


def test_hakeem_mouth_unavailable_reports_the_next_step() -> None:
    """With an unreachable model the mouth refuses honestly."""
    from ai import hakeem_mouth
    from core.coder_model import CoderUnavailable

    class _DownClient:
        model = DEFAULT_MODEL

        def list_models(self):
            raise CoderUnavailable("connection refused")

        def chat(self, messages, **kwargs):
            raise CoderUnavailable("connection refused")

    config = HakeemConfig(db_path=None, persistent=False)
    agent = HakeemAgent(config, client=_DownClient())
    assistant = hakeem_mouth.HakeemAssistant(agent)
    assert agent.model_status().available is False
    with pytest.raises(HakeemError):
        assistant.ensure_available()
    assert "ollama serve" in agent.model_status().summary().lower()


def test_mouth_switch_messages_exist() -> None:
    from ai import chat as chat_module

    assert NAME in chat_module.mouth_switched_message(chat_module.MOUTH_HAKEEM)
    assert "Hakeem" in chat_module.switch_failure_message(chat_module.MOUTH_HAKEEM, HakeemError("x"))


# -- the GUI / menu wiring ------------------------------------------------
def test_cli_menu_carries_hakeem() -> None:
    import cli.menu as menu

    labels = [label for label, _action, _argv in menu.MENU]
    assert any("Hakeem" in label for label in labels)


def test_cli_dispatches_the_hakeem_commands() -> None:
    from cli import __main__ as cli_entry

    assert "hakeem" in cli_entry._COMMANDS
    assert "hakeem-chat" in cli_entry._COMMANDS


def test_hakeem_is_a_hidden_sidebar_action() -> None:
    from gui import actions

    action = actions.action_by_key("hakeem")
    assert action.hidden is True
    assert "hakeem" not in {a.key for a in actions.visible_actions()}


def _display_available() -> bool:
    try:
        import tkinter as tk

        root = tk.Tk()
    except Exception:
        return False
    root.destroy()
    return True


def test_gui_mouth_choices_include_hakeem() -> None:
    """The window's model selector offers the scholar beside the other mouths."""
    if not _display_available():
        # Without a display, Tk is unusable; check the wiring in the source instead.
        source = (Path(__file__).resolve().parent.parent / "gui" / "app.py").read_text(
            encoding="utf-8"
        )
        assert "MOUTH_HAKEEM" in source
        return
    from gui.app import MOUTH_CHOICES

    keys = [key for key, _label in MOUTH_CHOICES]
    assert "hakeem" in keys


def test_service_exposes_use_hakeem() -> None:
    from gui import service

    assert hasattr(service.ChatService, "use_hakeem")


def test_personality_describe_is_readable() -> None:
    text = HakeemPersonality().describe()
    assert NAME in text
    assert "formality" in text
