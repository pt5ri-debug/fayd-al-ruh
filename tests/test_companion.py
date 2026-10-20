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
from core.coder_model import CoderMessage, CoderUnavailable


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


# -- the model layer ------------------------------------------------------
def test_the_client_is_natashas_own_ollama_client() -> None:
    """Natasha speaks through her own client, not the coding model's directly."""
    from companion.model import NatashaModelClient, OllamaNatashaClient

    agent = CompanionAgent(CompanionConfig(db_path=None, persistent=False))
    client = agent._active_client()
    assert isinstance(client, OllamaNatashaClient)
    assert isinstance(client, NatashaModelClient)  # runtime_checkable shape


def test_the_catalog_names_a_default_and_its_specs() -> None:
    from companion.model import CATALOG, DEFAULT_SPEC_KEY, resolve_spec, spec_names

    assert DEFAULT_SPEC_KEY in CATALOG
    assert resolve_spec(None).key == DEFAULT_SPEC_KEY
    assert resolve_spec("qwen3-8b").tag == "qwen3:8b"
    assert set(spec_names()) == set(CATALOG)


def test_an_unknown_model_name_is_refused() -> None:
    from companion.model import CompanionModelUnavailable, resolve_spec

    with pytest.raises(CompanionModelUnavailable):
        resolve_spec("not-a-model")


def test_status_reports_a_missing_server_honestly() -> None:
    from companion.model import natasha_model_status

    class DownClient:
        model = "qwen3:4b"

        def list_models(self) -> tuple[str, ...]:
            raise CoderUnavailable("connection refused")

        def chat(self, messages, *, temperature: float = 0.2) -> str:  # pragma: no cover
            raise CoderUnavailable("connection refused")

    status = natasha_model_status(client=DownClient())
    assert status.available is False
    assert status.server_reachable is False
    assert "ollama serve" in status.summary()


def test_status_reports_a_missing_model_honestly() -> None:
    from companion.model import natasha_model_status

    class EmptyClient:
        model = "qwen3:4b"

        def list_models(self) -> tuple[str, ...]:
            return ("llama3:8b",)

        def chat(self, messages, *, temperature: float = 0.2) -> str:  # pragma: no cover
            return ""

    status = natasha_model_status(client=EmptyClient())
    assert status.available is False
    assert status.server_reachable is True
    assert status.installed is False
    assert "ollama pull qwen3:4b" in status.summary()


def test_status_accepts_an_implicit_latest_tag() -> None:
    from companion.model import model_present, natasha_model_status

    assert model_present("qwen3:4b", ("qwen3:4b",))
    assert model_present("qwen3", ("qwen3:latest",))
    assert not model_present("qwen3:8b", ("qwen3:4b",))

    class ReadyClient:
        model = "qwen3"

        def list_models(self) -> tuple[str, ...]:
            return ("qwen3:latest",)

        def chat(self, messages, *, temperature: float = 0.2) -> str:  # pragma: no cover
            return ""

    assert natasha_model_status(client=ReadyClient()).available is True


def test_a_model_pull_is_denied_by_default() -> None:
    from companion.model import CompanionModelUnavailable, download_model, pull_model

    with pytest.raises(CompanionModelUnavailable):
        download_model("qwen3-4b")
    with pytest.raises(CompanionModelUnavailable):
        pull_model("qwen3-4b")


def test_an_approved_pull_runs_ollama_and_reports_the_tag() -> None:
    from companion.model import pull_model

    calls: list[list[str]] = []

    def runner(argv) -> tuple[int, str]:
        calls.append(list(argv))
        return 0, "success"

    tag = pull_model("qwen3-4b", approved=True, runner=runner)
    assert tag == "qwen3:4b"
    assert calls == [["ollama", "pull", "qwen3:4b"]]


def test_a_failed_pull_is_an_honest_error() -> None:
    from companion.model import CompanionModelUnavailable, pull_model

    def runner(argv) -> tuple[int, str]:
        return 1, "Error: pull access denied"

    with pytest.raises(CompanionModelUnavailable, match="pull access denied"):
        pull_model("qwen3-8b", approved=True, runner=runner)


# -- the in-process (GGUF) runtime ----------------------------------------
def test_the_default_runtime_prefers_an_installed_gguf(tmp_path: Path, monkeypatch) -> None:
    """``auto`` picks the daemon-free runtime once the extra and a file are present."""
    from companion.model import (
        BACKEND_GGUF,
        BACKEND_OLLAMA,
        LlamaCppNatashaClient,
        NatashaModelClient,
        resolve_backend,
    )

    assert resolve_backend("auto", root=tmp_path) == BACKEND_OLLAMA  # nothing installed
    (tmp_path / "qwen.gguf").write_bytes(b"x")
    monkeypatch.setattr("companion.model.extra_available", lambda: True)
    assert resolve_backend("auto", root=tmp_path) == BACKEND_GGUF
    # An explicitly named file is enough, even outside the folder scan.
    assert resolve_backend("auto", model_file=tmp_path / "qwen.gguf") == BACKEND_GGUF
    # An explicit choice is honoured either way.
    assert resolve_backend("ollama", root=tmp_path) == BACKEND_OLLAMA
    assert resolve_backend("gguf", root=tmp_path) == BACKEND_GGUF
    client = LlamaCppNatashaClient(root=tmp_path)
    assert isinstance(client, NatashaModelClient)


def test_an_unknown_backend_is_refused() -> None:
    from companion.model import CompanionModelUnavailable, resolve_backend

    with pytest.raises(CompanionModelUnavailable):
        resolve_backend("not-a-runtime")


def test_the_config_carries_the_runtime_choice(tmp_path: Path) -> None:
    from companion.model import BACKEND_GGUF

    config = CompanionConfig(db_path=None, persistent=False, backend=BACKEND_GGUF)
    assert config.backend == BACKEND_GGUF
    assert "gguf" in config.describe()
    with pytest.raises(ValueError):
        CompanionConfig(db_path=None, persistent=False, backend="nope")
    with pytest.raises(ValueError):
        CompanionConfig(db_path=None, persistent=False, context=0)


def test_the_agent_builds_the_in_process_client_when_asked(tmp_path: Path) -> None:
    from companion.model import BACKEND_GGUF, LlamaCppNatashaClient

    model_file = tmp_path / "m.gguf"
    model_file.write_bytes(b"x")
    config = CompanionConfig(
        db_path=None, persistent=False, backend=BACKEND_GGUF, model_file=model_file
    )
    client = CompanionAgent(config)._active_client()
    assert isinstance(client, LlamaCppNatashaClient)
    assert client.model_path() == model_file


def test_find_model_file_prefers_the_registry_and_picks_the_largest(tmp_path: Path) -> None:
    from companion.model import find_model_file, model_is_cached

    assert find_model_file(tmp_path) is None
    assert model_is_cached(tmp_path) is False
    small = tmp_path / "small.gguf"
    small.write_bytes(b"x")
    big = tmp_path / "big.gguf"
    big.write_bytes(b"x" * 100)
    assert find_model_file(tmp_path) == big  # largest wins with no registry
    (tmp_path / "model.json").write_text(json.dumps({"file": "small.gguf"}), encoding="utf-8")
    assert find_model_file(tmp_path) == small


def test_the_in_process_client_reports_and_refuses_honestly(
    tmp_path: Path, monkeypatch
) -> None:
    from companion.model import CompanionModelUnavailable, LlamaCppNatashaClient

    client = LlamaCppNatashaClient(root=tmp_path)
    assert client.backend == "in-process"
    assert client.list_models() == ()
    assert client.is_installed() is False
    with pytest.raises(CompanionModelUnavailable):
        client.model_path()
    # The next step names whichever half is missing: the runtime, or the file.
    monkeypatch.setattr("companion.model.extra_available", lambda: True)
    assert "no companion model file" in client.missing_detail().lower()
    monkeypatch.setattr("companion.model.extra_available", lambda: False)
    assert "requirements-natasha.txt" in client.missing_detail()


class FakeRunner:
    """A stand-in for the llama-cpp runner: records calls, returns a fixed reply."""

    def __init__(self, reply: str = "أهلاً بك.") -> None:
        self.reply_text = reply
        self.calls: list[tuple[list[dict[str, str]], dict]] = []

    def chat(self, messages: list[dict[str, str]], **kwargs) -> str:
        self.calls.append((messages, kwargs))
        return self.reply_text


def test_an_injected_runner_makes_the_in_process_client_available(tmp_path: Path) -> None:
    from companion.model import LlamaCppNatashaClient, natasha_model_status

    client = LlamaCppNatashaClient(root=tmp_path, runner=FakeRunner())
    assert client.is_installed() is True
    status = natasha_model_status(client=client)
    assert status.available is True
    assert status.backend == "in-process"
    assert "in-process" in status.summary()


def test_the_in_process_client_sends_messages_and_reads_the_reply(tmp_path: Path) -> None:
    from companion.model import LlamaCppNatashaClient

    runner = FakeRunner("مرحبا!")
    client = LlamaCppNatashaClient(root=tmp_path, runner=runner, max_tokens=64)
    answer = client.chat([CoderMessage(role="user", content="hi")], temperature=0.5)
    assert answer == "مرحبا!"
    messages, kwargs = runner.calls[0]
    assert messages == [{"role": "user", "content": "hi"}]
    assert kwargs == {"temperature": 0.5, "max_tokens": 64}


def test_a_reasoning_trace_is_stripped_from_an_in_process_reply(tmp_path: Path) -> None:
    """Qwen3's `` thinking`` block is not something Natasha said."""
    from companion.model import LlamaCppNatashaClient, strip_thinking

    assert strip_thinking("hi <thinking>a\nb</thinking>there") == "hi there"
    assert strip_thinking("<thinking>cut off") == ""

    runner = FakeRunner("<thinking>hmm</thinking>أهلاً")
    client = LlamaCppNatashaClient(root=tmp_path, runner=runner)
    assert client.chat([CoderMessage(role="user", content="hi")]) == "<thinking>hmm</thinking>أهلاً"

    # The agent is the boundary where a reply becomes "what Natasha said", so the
    # trace is dropped there -- for any runtime, and before it is stored.
    agent = CompanionAgent(
        CompanionConfig(db_path=None, persistent=False, backend="gguf"), client=client
    )
    session = agent.new_session("t")
    reply = agent.ask(session.id, "hi")
    assert reply.text == "أهلاً"
    stored = [m.content for m in agent.conversations.messages(session.id)]
    assert stored[-1] == "أهلاً"


def test_an_in_process_client_with_no_model_raises_honestly(tmp_path: Path) -> None:
    from companion.model import CompanionModelUnavailable, LlamaCppNatashaClient

    client = LlamaCppNatashaClient(root=tmp_path)  # no runner, no file
    with pytest.raises(CompanionModelUnavailable):
        client.chat([CoderMessage(role="user", content="hi")])


def test_a_gguf_download_is_denied_by_default(tmp_path: Path) -> None:
    from companion.model import CompanionModelUnavailable, download_model

    with pytest.raises(CompanionModelUnavailable):
        download_model("qwen3-4b", root=tmp_path)


def test_an_approved_gguf_download_returns_a_cached_file(tmp_path: Path) -> None:
    from companion.model import download_model, resolve_spec

    spec = resolve_spec("qwen3-4b")
    cached = tmp_path / spec.filename
    cached.write_bytes(b"x")
    assert download_model("qwen3-4b", root=tmp_path, approved=True) == cached


def test_the_catalog_pairs_each_tag_with_its_gguf(tmp_path: Path) -> None:
    """Switching runtime does not change which model she is."""
    from companion.model import CATALOG

    for spec in CATALOG.values():
        assert spec.tag and spec.repo.endswith("-GGUF") and spec.filename.endswith(".gguf")


def test_the_in_process_runtime_is_a_registered_optional_feature() -> None:
    from core import optional_deps
    from core.optional_deps import FEATURE_NATASHA

    assert FEATURE_NATASHA in optional_deps.FEATURES
    spec = optional_deps.feature(FEATURE_NATASHA)
    assert spec.requirements_file == "requirements-natasha.txt"
    assert "requirements-natasha.txt" in optional_deps.pip_install_command(FEATURE_NATASHA)


def test_the_natasha_requirements_file_pins_the_shared_runtime() -> None:
    """She shares the project's one in-process runtime, not a second copy of it."""
    root = Path(__file__).resolve().parents[1]
    text = (root / "requirements-natasha.txt").read_text(encoding="utf-8")
    assert "llama-cpp-python" in text
    assert "huggingface_hub" in text
    pyproject = (root / "pyproject.toml").read_text(encoding="utf-8")
    assert "\nnatasha = [" in pyproject


def test_natasha_models_dir_is_under_the_writable_data_folder() -> None:
    from core import paths

    assert paths.natasha_models_dir() == paths.data_dir() / "natasha"


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


# -- the CLI surface (models / download) ----------------------------------
def test_cli_models_lists_the_catalog(capsys) -> None:
    from ai import natasha

    assert natasha.main(["models"]) == 0
    out = capsys.readouterr().out
    assert "qwen3-4b" in out
    assert "qwen3:4b" not in out  # the keys are shown, not the raw tags


def test_cli_download_refuses_without_approval(capsys) -> None:
    from ai import natasha

    assert natasha.main(["download"]) == 2
    assert "approve" in capsys.readouterr().out.lower()


def test_cli_download_runs_the_gate_when_approved(monkeypatch, capsys) -> None:
    from ai import natasha

    calls: list[str] = []

    def fake_download(spec_key, *, approved=False, progress=None):
        calls.append(spec_key)
        assert approved is True
        return "qwen3:4b"

    monkeypatch.setattr(natasha, "download_model", fake_download)
    assert natasha.main(["download", "--approve"]) == 0
    assert calls == ["qwen3-4b"]
    assert "qwen3:4b" in capsys.readouterr().out


def test_cli_download_can_pull_the_tag_into_ollama(monkeypatch, capsys) -> None:
    """``--backend ollama`` is the fallback path: the same model, pulled as a tag."""
    from ai import natasha

    calls: list[str] = []

    def fake_pull(spec_key, *, approved=False, progress=None):
        calls.append(spec_key)
        assert approved is True
        return "qwen3:4b"

    monkeypatch.setattr(natasha, "pull_model", fake_pull)
    assert natasha.main(["download", "--approve", "--backend", "ollama"]) == 0
    assert calls == ["qwen3-4b"]
    out = capsys.readouterr().out
    assert "Ollama" in out
    assert "qwen3:4b" in out


def test_cli_status_names_the_runtime(tmp_path: Path, monkeypatch, capsys) -> None:
    from ai import natasha

    monkeypatch.setenv("FAYD_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("FAYD_NATASHA_BACKEND", "gguf")
    code = natasha.main(["status", "--json"])
    payload = json.loads(capsys.readouterr().out)
    assert payload["backend"] == "in-process"
    assert payload["available"] is False
    # With the extra absent the detail names the real next step (the pip install).
    assert "requirements-natasha.txt" in payload["detail"]
    assert "natasha download" in payload["hint"]
    assert code == 1


def test_cli_accepts_the_backend_choice() -> None:
    from ai import natasha

    assert natasha._build_parser().parse_args(["chat", "--backend", "ollama"]).backend == "ollama"
    with pytest.raises(SystemExit):
        natasha._build_parser().parse_args(["chat", "--backend", "nope"])


# -- bilingual replies and the linguistic polisher -------------------------

def test_reply_language_follows_the_users_own_message() -> None:
    """An English question gets English; an Arabic one gets Arabic; a bare '?' holds."""
    from companion.language import resolve_reply_language

    assert resolve_reply_language("What is porosity?") == "en"
    assert resolve_reply_language("ما هي المسامية؟") == "ar"
    # A message with no letters falls back to the previous user turn...
    assert resolve_reply_language("?", previous_user="ما هي المسامية؟") == "ar"
    assert resolve_reply_language("?", previous_user="What is porosity?") == "en"
    # ...and to the configured language when there is nothing to go on.
    assert resolve_reply_language("?", configured="en") == "en"
    # Mixed text (an English term inside an Arabic sentence) reads as Arabic.
    assert resolve_reply_language("اشرح لي permeability من فضلك") == "ar"


def test_language_instruction_names_the_reply_language() -> None:
    personality = default_personality().with_settings(PersonalitySettings(reply_language="match"))
    assert "Reply in English" in personality.language_instruction(reply_language="en")
    assert "Reply in Arabic" in personality.language_instruction(reply_language="ar")
    # A pinned mode ignores the language the user wrote in.
    pinned = personality.with_settings(PersonalitySettings(reply_language="ar"))
    assert "Reply in Arabic" in pinned.language_instruction(reply_language="en")
    english_pinned = personality.with_settings(PersonalitySettings(reply_language="en"))
    assert "Reply in English" in english_pinned.language_instruction(reply_language="ar")


def test_agent_answers_in_the_language_of_the_question(tmp_path: Path) -> None:
    seen: dict[str, str] = {}

    def caller(messages, temperature, max_tokens):  # type: ignore[no-untyped-def]
        seen["system"] = messages[0]["content"]
        seen["question"] = messages[-1]["content"]
        if "what" in seen["question"].lower():
            return "the answer is 42 ,and i think  it is right ?  "
        return "الإجابة هي 42 , وأعتقد  أنها صحيحة ؟  "

    config = CompanionConfig(db_path=tmp_path / "c.db", persistent=True)
    agent = CompanionAgent(config, caller=caller)
    session = agent.new_session("lang")

    english = agent.ask(session.id, "What is the answer?")
    assert english.reply_language == "en"
    assert "Reply in English" in seen["system"]
    # The polisher fixed the spacing and case, so the stored text reads cleanly.
    assert english.text == "The answer is 42, and I think it is right?"

    arabic = agent.ask(session.id, "ما هي الإجابة؟")
    assert arabic.reply_language == "ar"
    assert "Reply in Arabic" in seen["system"]
    assert arabic.text.startswith("الإجابة هي 42")
    assert "،" in arabic.text  # the ASCII comma became the Arabic one

    # A bare follow-up keeps the language of the exchange so far.
    follow_up = agent.ask(session.id, "?")
    assert follow_up.reply_language == "ar"


def test_polish_is_idempotent_and_preserves_numbers_and_abbreviations() -> None:
    from companion.polish import polish, polish_report

    cases = [
        "hello there ,how are you ?  i am fine .",
        "the value is 3.14 and the range is 10--20 units",
        "This is fine!!! Really??",
        "U.S.A is a country",
        "Use e.g. this and i.e. that",
        "مرحبا كيف حالك , أنا بخير ؟  هذا جيد .",
        "القيمة 3.14 والمدى 10--20",
        "Hello.\n\nthe next line , and i agree .",
    ]
    for case in cases:
        result = polish_report(case)
        assert polish(result.text, language=result.language) == result.text, case
    # Numbers, ranges, and abbreviations survive untouched.
    assert "3.14" in polish("the value is 3.14", language="en")
    assert "10--20" in polish("range 10--20", language="en")
    assert "U.S.A" in polish("U.S.A is a country", language="en")
    assert "e.g." in polish("use e.g. this", language="en")


def test_polish_never_adds_or_removes_a_word() -> None:
    """The polisher may re-case, re-punctuate, and re-space, but never adds or drops a word.

    A stray punctuation token (``psi ,``) legitimately merges into its word
    (``psi,``), so punctuation-only tokens are not counted as words; the sequence of
    word-bearing tokens must be identical.
    """
    from companion.polish import polish

    source = "the reservoir pressure is 3000 psi , and the porosity is 0.2"
    polished = polish(source, language="en")

    def words(text: str) -> list[str]:
        stripped = [word.strip(".,!?؛،").casefold() for word in text.split()]
        return [word for word in stripped if word]

    assert words(polished) == words(source)


def test_polish_report_names_what_it_changed() -> None:
    from companion.polish import polish_report

    result = polish_report("hello there ,how are you ?  i am fine .")
    assert result.language == "en"
    assert result.changed
    assert any("space" in change for change in result.changes)
    assert any("capital" in change for change in result.changes)
    # A reply that is already clean reports no change.
    assert not polish_report("This is already fine.").changed


def test_intimacy_dial_is_bounded_and_keeps_the_boundaries() -> None:
    with pytest.raises(ValueError):
        PersonalitySettings(intimacy=1.5)
    close = default_personality().with_settings(PersonalitySettings(intimacy=0.9))
    instruction = close.intimacy_instruction()
    assert "closeness" in instruction
    # The expressiveness dial never removes the honesty/consent rules.
    assert "consent rules" in instruction
    assert "never use affection to pressure" in instruction
    # A low setting keeps a measured distance.
    reserved = default_personality().with_settings(PersonalitySettings(intimacy=0.0))
    assert "measured distance" in reserved.intimacy_instruction()


def test_settings_round_trip_includes_the_new_dials() -> None:
    settings = PersonalitySettings(reply_language="en", intimacy=0.75)
    restored = PersonalitySettings.from_primitive(settings.to_primitive())
    assert restored.reply_language == "en"
    assert restored.intimacy == 0.75
    # An older payload without the new keys still loads (forward compatible).
    legacy = PersonalitySettings.from_primitive({"language": "ar", "warmth": 0.5})
    assert legacy.reply_language == "match"
    assert legacy.intimacy == 0.3


def test_cli_settings_accepts_reply_language_and_intimacy() -> None:
    from ai import natasha

    args = natasha._build_parser().parse_args(
        ["settings", "--reply-language", "en", "--intimacy", "0.8"]
    )
    assert args.reply_language == "en"
    assert args.intimacy == 0.8
    with pytest.raises(SystemExit):
        natasha._build_parser().parse_args(["settings", "--reply-language", "nope"])


# -- adult romance: open wording, intact safety floor ----------------------

def test_explicitness_dial_is_bounded_and_round_trips() -> None:
    with pytest.raises(ValueError):
        PersonalitySettings(explicitness=1.5)
    with pytest.raises(ValueError):
        PersonalitySettings(explicitness=-0.1)
    settings = PersonalitySettings(romance=0.9, intimacy=0.8, explicitness=1.0)
    restored = PersonalitySettings.from_primitive(settings.to_primitive())
    assert restored == settings
    assert restored.explicitness == 1.0
    # An older payload without the key still loads at the default.
    legacy = PersonalitySettings.from_primitive({"romance": 0.5})
    assert legacy.explicitness == 0.2


def test_sensuality_opens_the_wording_at_the_top_level() -> None:
    open_setting = default_personality().with_settings(PersonalitySettings(explicitness=1.0))
    text = open_setting.sensuality_instruction()
    assert "explicitly" in text
    assert "never fade to black" in text
    # The safety floor is restated in the very same instruction.
    assert "adults only" in text
    assert "18+" in text
    assert "consent" in text
    assert "no minors" in text

    middle = default_personality().with_settings(PersonalitySettings(explicitness=0.5))
    assert "tasteful rather than graphic" in middle.sensuality_instruction()

    chaste = default_personality().with_settings(PersonalitySettings(explicitness=0.0))
    assert "tender and chaste" in chaste.sensuality_instruction()


def test_escalation_is_user_led_and_reversible() -> None:
    open_setting = default_personality().with_settings(
        PersonalitySettings(romance=1.0, explicitness=1.0)
    )
    text = open_setting.escalation_instruction()
    assert "build naturally" in text
    assert "do not hold back" in text
    # The safe direction is explicit: user-led and reversible.
    assert "user-led" in text
    assert "reversible" in text
    assert "never push past it" in text

    # With romance off, escalation stays off.
    off = default_personality().with_settings(PersonalitySettings(romance=0.0))
    assert "Romance is currently off" in off.escalation_instruction()


def test_boundaries_are_a_safety_floor_not_a_wording_limit() -> None:
    """The rules keep adults-only/consent/honesty; they do not forbid open wording."""
    joined = " ".join(BOUNDARIES).casefold()
    # Safety floor present...
    assert "18+" in joined
    assert "minor" in joined
    assert "consent" in joined
    assert "virtual character" in joined
    assert "manipulation" in joined
    # ...and the open-wording permission is stated explicitly.
    assert "fully welcome" in joined
    assert "do not moralise" in joined


def test_system_prompt_includes_the_new_instructions() -> None:
    personality = default_personality().with_settings(
        PersonalitySettings(romance=1.0, intimacy=1.0, explicitness=1.0)
    )
    prompt = personality.system_prompt()
    assert "explicitly" in prompt
    assert "build naturally" in prompt
    assert "adults only" in prompt
    # Every boundary still rides along on every request.
    for rule in BOUNDARIES:
        assert rule in prompt


def test_cli_settings_accepts_explicitness() -> None:
    from ai import natasha

    args = natasha._build_parser().parse_args(["settings", "--explicitness", "1.0"])
    assert args.explicitness == 1.0
    assert natasha._build_parser().parse_args(["settings"]).explicitness is None
