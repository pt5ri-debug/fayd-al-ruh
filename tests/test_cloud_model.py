"""Cloud: the remote, OpenAI-compatible mouth -- engine, gates, CLI, chat, and GUI.

Everything remote is injected: the engine takes a ``transport`` callable, so the
whole client is exercised with **no network and no account**, and the mouth is
built with an explicit key. These tests also pin the two honesty rules that make
the layer safe to ship: a bare engine is *denied by default* (the permission and
human-approval gates must both say yes), and the API key is never written to a
message, a log, or the repository.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest

from ai import chat as chat_module
from ai import cloud as cloud_cli
from ai import cloud_mouth
from ai.chat import (
    MOUTH_BANNER,
    MOUTH_CLOUD,
    MOUTH_MODEL,
    MOUTH_SOURCES,
    MOUTH_ULTRON,
    MOUTH_VISION,
)
from cli import menu as cli_menu
from core.cloud_model import (
    BLURB,
    CATALOG,
    CHAT_TOOL,
    DEFAULT_PROVIDER_KEY,
    HEADER,
    MAX_TEMPERATURE,
    MAX_TOKENS_CEILING,
    NAME,
    CloudEngine,
    CloudMessage,
    CloudUnavailable,
    OpenAICompatibleClient,
    approved_gate,
    cloud_status,
    granted_permissions,
    load_model_key,
    load_provider_key,
    provider_names,
    provider_summaries,
    redact_secret,
    resolve_provider,
    save_model_key,
    save_provider_key,
    selected_model,
    selected_provider,
)
from core.conversation import Conversation
from core.permissions import Capability
from gui import actions as gui_actions

#: A minimal, well-formed OpenAI-compatible chat completion.
_REPLY = json.dumps({"choices": [{"message": {"content": "Hello from the cloud."}}]})
_MODELS = json.dumps({"data": [{"id": "gpt-4o-mini"}, {"id": "gpt-4o"}]})


def _display_available() -> bool:
    """True when a Tk display can be opened (the GUI tests skip without one)."""
    try:
        import tkinter as tk

        root = tk.Tk()
    except Exception:
        return False
    root.destroy()
    return True


class FakeTransport:
    """A stand-in for the HTTP transport: records calls, returns canned replies."""

    def __init__(self, *, status: int = 200, body: str = _REPLY) -> None:
        self.status = status
        self.body = body
        self.calls: list[tuple[str, dict[str, str], bytes | None, float]] = []

    def __call__(self, url, headers, data, timeout):
        self.calls.append((url, headers, data, timeout))
        return self.status, self.body


class FakeClient:
    """A stand-in for a cloud client: satisfies the engine's protocol."""

    def __init__(self, reply: str = "Hi there.", *, provider: str = "openai") -> None:
        self.model = "fake-model"
        self.provider = resolve_provider(provider)
        self.reply_text = reply
        self.calls: list[tuple[list[CloudMessage], dict]] = []

    def chat(self, messages, *, temperature=None, max_tokens=None) -> str:
        self.calls.append((list(messages), {"temperature": temperature, "max_tokens": max_tokens}))
        return self.reply_text

    def list_models(self) -> tuple[str, ...]:
        return ("fake-model",)


class KeylessClient(FakeClient):
    """A client that reports it has no key, like a provider with an empty env var."""

    def key_available(self) -> bool:
        return False

    def missing_key_hint(self) -> str:
        return "Set OPENAI_API_KEY to your OpenAI (ChatGPT) API key, then try again."


# --- the catalog -----------------------------------------------------------


def test_catalog_has_a_default_and_named_providers() -> None:
    assert DEFAULT_PROVIDER_KEY in CATALOG
    assert resolve_provider(None) is CATALOG[DEFAULT_PROVIDER_KEY]
    assert "openai" in provider_names()
    assert "anthropic" in provider_names()
    assert "gemini" in provider_names()
    for _key, title, model, summary, _local, _free in provider_summaries():
        assert title and model and summary


def test_every_remote_provider_names_its_key_variable() -> None:
    """A remote provider must say which environment variable holds its key."""
    for spec in CATALOG.values():
        if spec.local:
            assert spec.api_key_env == ""
        else:
            assert spec.api_key_env.endswith("_API_KEY")


def test_local_providers_need_no_key() -> None:
    assert resolve_provider("ollama").needs_key is False
    assert resolve_provider("lmstudio").needs_key is False
    assert resolve_provider("ollama").local is True


def test_resolve_provider_is_case_insensitive_and_rejects_unknown() -> None:
    assert resolve_provider("OPENAI").key == "openai"
    with pytest.raises(ValueError):
        resolve_provider("nope")


def test_resolve_provider_passes_a_provider_back_unchanged() -> None:
    spec = resolve_provider("groq")
    assert resolve_provider(spec) is spec


def test_blurb_is_short_and_honest() -> None:
    assert "remote" in BLURB.lower()


# --- the client ------------------------------------------------------------


def test_client_posts_the_openai_chat_shape() -> None:
    transport = FakeTransport()
    client = OpenAICompatibleClient("gpt-4o-mini", provider="openai", api_key="k", transport=transport)
    reply = client.chat([CloudMessage("user", "hi")])
    assert reply == "Hello from the cloud."
    url, headers, data, _timeout = transport.calls[0]
    assert url.endswith("/chat/completions")
    assert headers["Authorization"] == "Bearer k"
    assert json.loads(cast(bytes, data))["model"] == "gpt-4o-mini"


def test_client_sends_no_authorization_without_a_key() -> None:
    transport = FakeTransport()
    client = OpenAICompatibleClient(provider="ollama", transport=transport)
    client.chat([CloudMessage("user", "hi")])
    _url, headers, _data, _timeout = transport.calls[0]
    assert "Authorization" not in headers


def test_client_refuses_without_a_key_and_names_the_variable() -> None:
    client = OpenAICompatibleClient(provider="openai", api_key="")
    with pytest.raises(CloudUnavailable) as error:
        client.chat([CloudMessage("user", "hi")])
    assert "OPENAI_API_KEY" in str(error.value)


def test_client_reads_the_key_from_the_environment(monkeypatch) -> None:
    monkeypatch.setenv("FAYD_TEST_KEY", "env-key")
    transport = FakeTransport()
    client = OpenAICompatibleClient(
        provider="openai", api_key_env="FAYD_TEST_KEY", transport=transport
    )
    client.chat([CloudMessage("user", "hi")])
    assert transport.calls[0][1]["Authorization"] == "Bearer env-key"


def test_client_reports_an_http_error_without_leaking_the_key() -> None:
    secret = "sk-super-secret"
    transport = FakeTransport(status=401, body=f"invalid key {secret}")
    client = OpenAICompatibleClient(
        "gpt-4o-mini", provider="openai", api_key=secret, transport=transport
    )
    with pytest.raises(CloudUnavailable) as error:
        client.chat([CloudMessage("user", "hi")])
    assert secret not in str(error.value)
    assert "***" in str(error.value)
    assert "401" in str(error.value)


def test_client_reports_a_non_json_reply_honestly() -> None:
    transport = FakeTransport(body="<html>not json</html>")
    client = OpenAICompatibleClient(provider="ollama", transport=transport)
    with pytest.raises(CloudUnavailable):
        client.chat([CloudMessage("user", "hi")])


def test_client_rejects_an_empty_reply() -> None:
    transport = FakeTransport(body=json.dumps({"choices": [{"message": {"content": ""}}]}))
    client = OpenAICompatibleClient(provider="ollama", transport=transport)
    with pytest.raises(CloudUnavailable):
        client.chat([CloudMessage("user", "hi")])


def test_client_validates_temperature_and_max_tokens() -> None:
    client = OpenAICompatibleClient(provider="ollama", transport=FakeTransport())
    with pytest.raises(ValueError):
        client.chat([CloudMessage("user", "hi")], temperature=MAX_TEMPERATURE + 1)
    with pytest.raises(ValueError):
        client.chat([CloudMessage("user", "hi")], max_tokens=MAX_TOKENS_CEILING + 1)


def test_client_lists_models() -> None:
    client = OpenAICompatibleClient(provider="openai", api_key="k", transport=FakeTransport(body=_MODELS))
    assert client.list_models() == ("gpt-4o", "gpt-4o-mini")


def test_message_rejects_a_bad_role_and_empty_content() -> None:
    with pytest.raises(ValueError):
        CloudMessage("system2", "x")
    with pytest.raises(ValueError):
        CloudMessage("user", "   ")


def test_redact_secret_is_safe_with_no_secret() -> None:
    assert redact_secret("plain text", None) == "plain text"
    assert redact_secret("key abc here", "abc") == "key *** here"


# --- status ----------------------------------------------------------------


def test_status_reports_a_missing_key_without_a_network_call() -> None:
    transport = FakeTransport()
    client = OpenAICompatibleClient(provider="openai", api_key="", transport=transport)
    status = cloud_status(client)
    assert status.available is False
    assert status.key_present is False
    assert "OPENAI_API_KEY" in status.hint
    assert transport.calls == []  # no probe, no request


def test_status_probe_reports_reachable_and_lists_models() -> None:
    client = OpenAICompatibleClient(
        provider="openai", api_key="k", transport=FakeTransport(body=_MODELS)
    )
    status = cloud_status(client, probe=True)
    assert status.reachable is True
    assert status.available is True
    assert "gpt-4o" in status.models


def test_status_probe_reports_an_unreachable_endpoint() -> None:
    client = OpenAICompatibleClient(
        provider="openai", api_key="k", transport=FakeTransport(status=500, body="boom")
    )
    status = cloud_status(client, probe=True)
    assert status.reachable is False
    assert status.available is False
    assert "500" in status.detail


def test_status_summary_is_honest() -> None:
    client = OpenAICompatibleClient(provider="ollama", transport=FakeTransport())
    assert "ready" in cloud_status(client).summary()
    missing = OpenAICompatibleClient(provider="openai", api_key="")
    assert "not ready" in cloud_status(missing).summary() or "no API key" in cloud_status(missing).summary()


# --- the engine and its gates ----------------------------------------------


def test_engine_is_denied_by_default() -> None:
    """A bare engine must reach nothing: the permission gate is closed."""
    client = FakeClient()
    engine = CloudEngine(client)
    assert engine.is_available() is False
    with pytest.raises(CloudUnavailable) as error:
        engine.reply("hello")
    assert "denied" in str(error.value).lower()
    assert client.calls == []  # the client was never called


def test_engine_requires_approval_even_with_permission() -> None:
    client = FakeClient()
    engine = CloudEngine(client, permissions=granted_permissions())
    assert engine.is_available() is False
    with pytest.raises(CloudUnavailable):
        engine.reply("hello")
    assert client.calls == []


def test_engine_requires_a_key_even_when_approved() -> None:
    engine = CloudEngine(KeylessClient(), approved=True)
    assert engine.is_available() is False
    with pytest.raises(CloudUnavailable) as error:
        engine.reply("hello")
    assert "OPENAI_API_KEY" in str(error.value)


def test_engine_answers_once_approved_and_keyed() -> None:
    client = FakeClient("Cloud reply.")
    engine = CloudEngine(client, approved=True)
    assert engine.is_available() is True
    assert engine.reply("hello") == "Cloud reply."
    messages, kwargs = client.calls[0]
    assert messages[0].role == "system"
    assert messages[-1] == CloudMessage("user", "hello")


def test_engine_builds_system_then_history_then_prompt() -> None:
    engine = CloudEngine(FakeClient(), approved=True)
    messages = engine.build_messages([("user", "hi"), ("assistant", "hello")], "and?")
    assert [m.role for m in messages] == ["system", "user", "assistant", "user"]
    assert messages[0].content == engine.system_prompt


def test_engine_skips_an_empty_history_turn() -> None:
    engine = CloudEngine(FakeClient(), approved=True)
    messages = engine.build_messages([("user", "hi"), ("assistant", "   ")], "and?")
    assert [m.role for m in messages] == ["system", "user", "user"]


def test_engine_rejects_an_empty_prompt() -> None:
    engine = CloudEngine(FakeClient(), approved=True)
    with pytest.raises(ValueError):
        engine.reply("   ")


def test_engine_streams_when_the_client_can() -> None:
    class StreamingClient(FakeClient):
        def stream_chat(self, messages, *, temperature=None, max_tokens=None):
            yield "a"
            yield "b"

    engine = CloudEngine(StreamingClient(), approved=True)
    assert list(engine.stream_reply("hi")) == ["a", "b"]


def test_engine_describe_names_the_provider_and_where_it_runs() -> None:
    remote = CloudEngine(FakeClient(provider="openai"), approved=True)
    assert "remote" in remote.describe()
    local = CloudEngine(FakeClient(provider="ollama"), approved=True)
    assert "on this machine" in local.describe()


def test_engine_audits_its_decision() -> None:
    """Every cloud attempt is recorded, allowed or not."""
    from core.audit_log import AuditLog

    audit = AuditLog()
    engine = CloudEngine(FakeClient(), approved=True, audit=audit)
    engine.reply("hi")
    records = audit.for_capability(Capability.EXTERNAL_TOOL)
    assert any(record.target == CHAT_TOOL for record in records)


def test_granted_permissions_and_gate_are_scoped_to_the_chat_tool() -> None:
    policy = granted_permissions()
    assert policy.decide(Capability.EXTERNAL_TOOL, CHAT_TOOL).allowed
    gate = approved_gate()
    assert gate.decide(CHAT_TOOL).approved


# --- the chat mouth --------------------------------------------------------


def _cloud_assistant(reply: str = "Cloud reply.") -> cloud_mouth.CloudAssistant:
    return cloud_mouth.load_cloud_assistant(client=FakeClient(reply), api_key="k")


def test_load_cloud_assistant_needs_a_usable_endpoint() -> None:
    with pytest.raises(CloudUnavailable):
        cloud_mouth.load_cloud_assistant(client=KeylessClient())


def test_cloud_assistant_ask_records_the_turn_with_no_citations() -> None:
    assistant = _cloud_assistant("A remote answer.")
    conversation = Conversation()
    answer = assistant.ask("what is porosity?", conversation=conversation)
    assert "A remote answer." in answer.text
    assert HEADER in answer.text
    assert answer.citations == ()
    assert answer.passages == ()
    assert answer.used_model is True
    assert conversation.messages[-1].role.value == "assistant"


def test_cloud_assistant_passes_prior_turns_as_history() -> None:
    client = FakeClient()
    assistant = cloud_mouth.load_cloud_assistant(client=client, api_key="k")
    conversation = Conversation()
    conversation.add_user("first")
    conversation.add_assistant("first answer")
    assistant.ask("second", conversation=conversation)
    messages, _kwargs = client.calls[0]
    assert ("user", "first") in [(m.role, m.content) for m in messages]
    assert ("assistant", "first answer") in [(m.role, m.content) for m in messages]


def test_cloud_assistant_strips_a_prior_honesty_header_from_history() -> None:
    client = FakeClient()
    assistant = cloud_mouth.load_cloud_assistant(client=client, api_key="k")
    conversation = Conversation()
    conversation.add_assistant(f"{HEADER}\n\nearlier answer")
    assistant.ask("next", conversation=conversation)
    messages, _kwargs = client.calls[0]
    assert all(HEADER not in m.content for m in messages)


def test_cloud_assistant_includes_attached_reference_material() -> None:
    client = FakeClient()
    assistant = cloud_mouth.load_cloud_assistant(client=client, api_key="k")
    conversation = Conversation()
    conversation.add_context("The casing is 9-5/8 inch.")
    assistant.ask("what size is the casing?", conversation=conversation)
    messages, _kwargs = client.calls[0]
    prompt = messages[-1].content
    assert "9-5/8 inch" in prompt
    assert cloud_mouth.REFERENCE_LABEL in prompt


def test_cloud_assistant_substitutes_a_body_when_the_model_is_silent() -> None:
    assistant = _cloud_assistant("   ")
    answer = assistant.ask("anything")
    assert cloud_mouth.CLOUD_EMPTY in answer.text


def test_cloud_assistant_rejects_an_empty_question() -> None:
    assistant = _cloud_assistant()
    with pytest.raises(ValueError):
        assistant.ask("   ")


def test_unavailable_message_names_the_next_step() -> None:
    engine = CloudEngine(OpenAICompatibleClient(provider="openai", api_key=""))
    message = cloud_mouth.unavailable_message(engine)
    assert "OPENAI_API_KEY" in message


# --- the switcher ----------------------------------------------------------


class FakeSession:
    def __init__(self) -> None:
        self.assistant: Any = SimpleNamespace(name="grounded")
        self.help_text = "help"
        self.voice = SimpleNamespace(assistant=self.assistant)


def _switcher(reply: str = "Cloud reply."):
    session = FakeSession()
    switcher = chat_module.ModelSwitcher(
        cast(Any, session),
        loader=lambda: (_ for _ in ()).throw(AssertionError("the trained model must not load")),
        cloud_loader=lambda: cloud_mouth.load_cloud_assistant(client=FakeClient(reply), api_key="k"),
    )
    return session, switcher


def test_switcher_cloud_swaps_the_mouth_and_keeps_the_grounded_one() -> None:
    session, switcher = _switcher()
    grounded = switcher.grounded
    grounded_help = switcher.grounded_help

    switcher.use_cloud()

    assert switcher.mouth == MOUTH_CLOUD
    assert switcher.active is True
    assert session.assistant is switcher.cloud
    assert session.voice.assistant is switcher.cloud
    assert cloud_mouth.CLOUD_HELP in session.help_text

    switcher.use_sources()
    assert switcher.mouth == MOUTH_SOURCES
    assert session.assistant is grounded
    assert session.help_text == grounded_help


def test_switcher_loads_cloud_lazily_and_reuses_it() -> None:
    session = FakeSession()
    calls: list[int] = []

    def _loader():
        calls.append(1)
        return cloud_mouth.load_cloud_assistant(client=FakeClient(), api_key="k")

    switcher = chat_module.ModelSwitcher(cast(Any, session), loader=lambda: None, cloud_loader=_loader)
    assert calls == []
    switcher.use_cloud()
    switcher.use_cloud()
    assert calls == [1]


def test_switcher_reports_a_cloud_failure_with_the_real_next_step() -> None:
    session = FakeSession()
    switcher = chat_module.ModelSwitcher(
        cast(Any, session),
        loader=lambda: None,
        cloud_loader=lambda: cloud_mouth.load_cloud_assistant(client=KeylessClient()),
    )
    message = chat_module._apply_switch(switcher, MOUTH_CLOUD)
    assert "OPENAI_API_KEY" in message
    assert switcher.mouth == MOUTH_SOURCES  # the working mouth stays


def test_switch_recognises_the_cloud_phrases() -> None:
    assert chat_module._match_switch("from cloud") == MOUTH_CLOUD
    assert chat_module._match_switch("Cloud on.") == MOUTH_CLOUD
    assert chat_module._match_switch("من السحابة") == MOUTH_CLOUD
    assert chat_module._match_switch("from banner") == MOUTH_BANNER
    assert chat_module._match_switch("from vision") == MOUTH_VISION
    assert chat_module._match_switch("from ultron") == MOUTH_ULTRON
    assert chat_module._match_switch("from the model") == MOUTH_MODEL
    assert chat_module._match_switch("from the sources") == MOUTH_SOURCES
    # A real question is never captured as a switch.
    assert chat_module._match_switch("what is cloud computing?") is None


def test_mouth_switched_message_describes_the_cloud() -> None:
    assert "cloud" in chat_module.mouth_switched_message(MOUTH_CLOUD).lower()


# --- the CLI ---------------------------------------------------------------


def test_cli_providers_lists_the_catalog(capsys, tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(cloud_cli, "_settings_root", lambda: tmp_path)
    assert cloud_cli.run_providers(cast(argparse.Namespace, SimpleNamespace(as_json=False))) == 0
    assert "openai" in capsys.readouterr().out


def test_cli_providers_json(capsys, tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(cloud_cli, "_settings_root", lambda: tmp_path)
    assert cloud_cli.run_providers(cast(argparse.Namespace, SimpleNamespace(as_json=True))) == 0
    data = json.loads(capsys.readouterr().out)
    assert data["chosen"] in CATALOG
    assert any(entry["key"] == "openai" for entry in data["providers"])


def test_cli_status_is_nonzero_without_a_key(monkeypatch, capsys, tmp_path) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setattr(cloud_cli, "_settings_root", lambda: tmp_path)
    args = cast(argparse.Namespace, SimpleNamespace(provider=None, model=None, base_url=None, api_key_env=None, probe=False, as_json=False))
    assert cloud_cli.run_status(args) == 1
    assert "OPENAI_API_KEY" in capsys.readouterr().out


def test_cli_status_is_zero_with_a_key(monkeypatch, capsys, tmp_path) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "k")
    monkeypatch.setattr(cloud_cli, "_settings_root", lambda: tmp_path)
    args = cast(argparse.Namespace, SimpleNamespace(provider=None, model=None, base_url=None, api_key_env=None, probe=False, as_json=False))
    assert cloud_cli.run_status(args) == 0
    assert NAME in capsys.readouterr().out


def test_cli_bare_question_is_normalized_to_chat() -> None:
    assert cloud_cli._normalize_argv(["who wrote Hamlet?"])[0] == "chat"
    assert cloud_cli._normalize_argv(["status"]) == ["status"]
    assert cloud_cli._normalize_argv(["providers"]) == ["providers"]
    assert cloud_cli._normalize_argv(["models"]) == ["models"]
    assert cloud_cli._normalize_argv(["use", "gemini"]) == ["use", "gemini"]
    assert cloud_cli._normalize_argv(["--help"]) == ["--help"]


def test_saved_provider_defaults_to_openai(tmp_path) -> None:
    assert load_provider_key(tmp_path) == DEFAULT_PROVIDER_KEY


def test_saved_provider_round_trips(tmp_path) -> None:
    assert save_provider_key(tmp_path, "gemini") == "gemini"
    assert load_provider_key(tmp_path) == "gemini"
    # And a second choice replaces the first.
    save_provider_key(tmp_path, "groq")
    assert load_provider_key(tmp_path) == "groq"


def test_saved_provider_rejects_an_unknown_name(tmp_path) -> None:
    with pytest.raises(ValueError):
        save_provider_key(tmp_path, "not-a-provider")


def test_saved_provider_ignores_a_stale_name(tmp_path) -> None:
    """A stored provider that left the catalog falls back, never breaks."""
    from storage.json_store import JsonStore

    JsonStore(tmp_path).save("cloud", "provider", {"provider": "removed-provider"})
    assert load_provider_key(tmp_path) == DEFAULT_PROVIDER_KEY


def test_selected_provider_prefers_an_explicit_choice(tmp_path) -> None:
    save_provider_key(tmp_path, "gemini")
    assert selected_provider(tmp_path).key == "gemini"
    assert selected_provider(tmp_path, "groq").key == "groq"


def test_gemini_default_model_is_a_current_one() -> None:
    """A retired model must not be the shipped default.

    Google shuts models down (`gemini-2.0-flash` is gone), so the catalog's
    default has to be a model the endpoint still serves, or every Gemini call
    fails with a 404 out of the box.
    """
    assert CATALOG["gemini"].default_model != "gemini-2.0-flash"
    assert CATALOG["gemini"].default_model.startswith("gemini-")


def test_selected_model_prefers_explicit_then_saved_then_default(tmp_path) -> None:
    spec = CATALOG["gemini"]
    # Nothing saved: the provider's own default.
    assert selected_model(tmp_path, "gemini") == spec.default_model
    # A saved model wins over the default.
    save_model_key(tmp_path, "gemini", "gemini-3.8-flash")
    assert selected_model(tmp_path, "gemini") == "gemini-3.8-flash"
    # An explicit --model wins over the saved one, for one turn.
    assert selected_model(tmp_path, "gemini", "gemini-2.5-flash") == "gemini-2.5-flash"


def test_saved_model_is_per_provider(tmp_path) -> None:
    save_model_key(tmp_path, "gemini", "gemini-3.8-flash")
    assert load_model_key(tmp_path, "gemini") == "gemini-3.8-flash"
    # Another provider is unaffected and keeps its own default.
    assert load_model_key(tmp_path, "groq") is None
    assert selected_model(tmp_path, "groq") == CATALOG["groq"].default_model


def test_saved_model_rejects_an_unknown_provider_and_empty_name(tmp_path) -> None:
    with pytest.raises(ValueError):
        save_model_key(tmp_path, "not-a-provider", "some-model")
    with pytest.raises(ValueError):
        save_model_key(tmp_path, "gemini", "   ")


def test_load_model_key_never_raises_on_a_broken_store(tmp_path) -> None:
    """A bad stored model, or an unknown provider, is simply "no saved model"."""
    from storage.json_store import JsonStore

    JsonStore(tmp_path).save("cloud", "models", {"gemini": "gemini-3.8-flash"})
    assert load_model_key(tmp_path, "not-a-provider") is None


def test_cli_use_remembers_the_provider(tmp_path, monkeypatch, capsys) -> None:
    monkeypatch.setattr(cloud_cli, "_settings_root", lambda: tmp_path)
    args = cast(argparse.Namespace, SimpleNamespace(provider_key="gemini", model=None))
    assert cloud_cli.run_use(args) == 0
    assert "Gemini" in capsys.readouterr().out
    assert load_provider_key(tmp_path) == "gemini"


def test_cli_use_can_remember_a_model_too(tmp_path, monkeypatch, capsys) -> None:
    """`cloud use gemini --model X` remembers the model, so a retired default is fixed once."""
    monkeypatch.setattr(cloud_cli, "_settings_root", lambda: tmp_path)
    args = cast(argparse.Namespace, SimpleNamespace(provider_key="gemini", model="gemini-3.8-flash"))
    assert cloud_cli.run_use(args) == 0
    out = capsys.readouterr().out
    assert "gemini-3.8-flash" in out
    assert load_model_key(tmp_path, "gemini") == "gemini-3.8-flash"


def test_cli_engine_honours_the_saved_model(tmp_path, monkeypatch) -> None:
    """With no --model, the engine uses the model saved for the chosen provider."""
    save_provider_key(tmp_path, "gemini")
    save_model_key(tmp_path, "gemini", "gemini-3.8-flash")
    monkeypatch.setattr(cloud_cli, "_settings_root", lambda: tmp_path)
    args = cast(
        argparse.Namespace,
        SimpleNamespace(
            provider=None, model=None, base_url=None, api_key_env=None,
            temperature=None, max_tokens=None,
        ),
    )
    engine = cloud_cli._engine(args)
    assert engine.provider.key == "gemini"
    assert engine.client.model == "gemini-3.8-flash"


def test_cli_models_lists_what_the_provider_serves(tmp_path, monkeypatch, capsys) -> None:
    """`cloud models` prints the endpoint's real list and marks the one in use."""
    save_provider_key(tmp_path, "gemini")
    save_model_key(tmp_path, "gemini", "gemini-3.8-flash")
    monkeypatch.setattr(cloud_cli, "_settings_root", lambda: tmp_path)
    client = FakeClient("x", provider="gemini")
    client.model = "gemini-3.8-flash"
    client.list_models = lambda: ("gemini-2.0-flash", "gemini-3.8-flash")  # type: ignore[method-assign]
    monkeypatch.setattr(cloud_cli, "_engine", lambda _args: CloudEngine(client, approved=True))
    args = cast(argparse.Namespace, SimpleNamespace(as_json=False))
    assert cloud_cli.run_models(args) == 0
    out = capsys.readouterr().out
    assert "gemini-3.8-flash" in out
    assert "in use" in out


def test_cli_models_json(tmp_path, monkeypatch, capsys) -> None:
    monkeypatch.setattr(cloud_cli, "_settings_root", lambda: tmp_path)
    client = FakeClient("x", provider="gemini")
    client.model = "gemini-3.8-flash"
    client.list_models = lambda: ("gemini-3.8-flash",)  # type: ignore[method-assign]
    monkeypatch.setattr(cloud_cli, "_engine", lambda _args: CloudEngine(client, approved=True))
    args = cast(argparse.Namespace, SimpleNamespace(as_json=True))
    assert cloud_cli.run_models(args) == 0
    data = json.loads(capsys.readouterr().out)
    assert data["provider"] == "gemini"
    assert data["models"] == ["gemini-3.8-flash"]


def test_cli_models_needs_a_key(tmp_path, monkeypatch, capsys) -> None:
    monkeypatch.setattr(cloud_cli, "_settings_root", lambda: tmp_path)
    monkeypatch.setattr(
        cloud_cli, "_engine", lambda _args: CloudEngine(KeylessClient(provider="openai"), approved=True)
    )
    args = cast(argparse.Namespace, SimpleNamespace(as_json=False))
    assert cloud_cli.run_models(args) == 1
    assert "OPENAI_API_KEY" in capsys.readouterr().out


def test_retired_model_error_names_the_next_step() -> None:
    """A 404 for a retired model must say how to recover, not just show JSON.

    This is the exact failure a user hits when a provider shuts a model down
    (Google retired ``gemini-2.0-flash``): the bare HTTP code leaves them stuck,
    so the message points at `cloud models` and at `--model`.
    """
    body = json.dumps({
        "error": {
            "code": 404,
            "message": "This model models/gemini-2.0-flash is no longer available. "
                       "Please update your code to use models/gemini-3.8-flash.",
        }
    })

    def transport(url: str, headers: dict[str, str], data: bytes | None, timeout: float):
        return 404, body

    client = OpenAICompatibleClient(
        "gemini-2.0-flash", provider="gemini", api_key="k", transport=transport
    )
    with pytest.raises(CloudUnavailable) as info:
        client.chat([CloudMessage("user", "hi")])
    message = str(info.value)
    assert "gemini-2.0-flash" in message
    assert "cloud models" in message
    assert "--model" in message


def test_an_unrelated_http_error_keeps_the_plain_message() -> None:
    """Only a *model* problem gets the model advice; a 500 stays a 500."""

    def transport(url: str, headers: dict[str, str], data: bytes | None, timeout: float):
        return 500, "internal server error"

    client = OpenAICompatibleClient(
        "gpt-4o-mini", provider="openai", api_key="k", transport=transport
    )
    with pytest.raises(CloudUnavailable) as info:
        client.chat([CloudMessage("user", "hi")])
    message = str(info.value)
    assert "HTTP 500" in message
    assert "cloud models" not in message


def test_cloud_mouth_honours_the_saved_model(tmp_path, monkeypatch) -> None:
    """The chat mouth answers from the saved model, not the retired default.

    The window and the CLI must agree, so a model saved once (fixing a provider
    that retired its default) is used by the mouth too, not just by `cli cloud`.
    """
    save_provider_key(tmp_path, "gemini")
    save_model_key(tmp_path, "gemini", "gemini-3.8-flash")
    monkeypatch.setattr("core.paths.data_dir", lambda: tmp_path)
    monkeypatch.setenv("GEMINI_API_KEY", "k")
    assistant = cloud_mouth.load_cloud_assistant()
    assert assistant.engine.provider.key == "gemini"
    assert assistant.engine.client.model == "gemini-3.8-flash"


def test_cloud_mouth_explicit_model_still_wins(tmp_path, monkeypatch) -> None:
    """An explicit model passed to the loader is not overridden by the saved one."""
    save_model_key(tmp_path, "gemini", "gemini-3.8-flash")
    monkeypatch.setattr("core.paths.data_dir", lambda: tmp_path)
    monkeypatch.setenv("GEMINI_API_KEY", "k")
    assistant = cloud_mouth.load_cloud_assistant(provider="gemini", model="gemini-2.5-flash")
    assert assistant.engine.client.model == "gemini-2.5-flash"


def test_cli_use_rejects_an_unknown_provider(tmp_path, monkeypatch, capsys) -> None:
    monkeypatch.setattr(cloud_cli, "_settings_root", lambda: tmp_path)
    args = cast(argparse.Namespace, SimpleNamespace(provider_key="nope"))
    assert cloud_cli.run_use(args) == 2
    assert "Unknown cloud provider" in capsys.readouterr().out


def test_cli_engine_honours_the_saved_provider(tmp_path, monkeypatch) -> None:
    """With no --provider, the engine uses what the user chose and saved."""
    save_provider_key(tmp_path, "groq")
    monkeypatch.setattr(cloud_cli, "_settings_root", lambda: tmp_path)
    args = cast(
        argparse.Namespace,
        SimpleNamespace(
            provider=None, model=None, base_url=None, api_key_env=None,
            temperature=None, max_tokens=None,
        ),
    )
    assert cloud_cli._engine(args).provider.key == "groq"


def test_cli_one_shot_uses_an_injected_client(monkeypatch, tmp_path, capsys) -> None:
    client = FakeClient("The cloud says hello.")
    monkeypatch.setattr(cloud_cli, "_engine", lambda _args: CloudEngine(client, approved=True))
    monkeypatch.setenv("FAYD_DATA_DIR", str(tmp_path))
    assert cloud_cli.main(["who wrote Hamlet?"]) == 0
    out = capsys.readouterr().out
    assert "The cloud says hello." in out
    assert HEADER in out


def test_cli_interactive_quits_immediately(monkeypatch, tmp_path, capsys) -> None:
    monkeypatch.setattr(cloud_cli, "_engine", lambda _args: CloudEngine(FakeClient(), approved=True))
    monkeypatch.setenv("FAYD_DATA_DIR", str(tmp_path))
    monkeypatch.setattr("builtins.input", lambda _prompt: "quit")
    assert cloud_cli.main([]) == 0


# --- menu and CLI dispatch -------------------------------------------------


def test_menu_shows_cloud_directly_below_banner() -> None:
    labels = [title for title, _action, _argv in cli_menu.visible_menu()]
    banner = next(i for i, label in enumerate(labels) if label.startswith("Banner"))
    assert labels[banner + 1].startswith("Cloud (remote model")


def test_menu_action_runs_the_cloud_module(monkeypatch) -> None:
    captured: dict[str, object] = {}

    def fake_run_module(name: str, argv) -> int:
        captured["name"] = name
        captured["argv"] = list(argv)
        return 0

    monkeypatch.setattr(cli_menu, "_run_module", fake_run_module)
    assert cli_menu.run_cloud(["--session", "s"]) == 0
    assert captured["name"] == "ai.cloud"
    assert captured["argv"] == ["--session", "s"]


def test_run_cloud_chat_adds_the_flag_once(monkeypatch) -> None:
    captured: dict[str, object] = {}

    def fake_run_module(name: str, argv) -> int:
        captured["name"] = name
        captured["argv"] = list(argv)
        return 0

    monkeypatch.setattr(cli_menu, "_run_module", fake_run_module)
    assert cli_menu.run_cloud_chat([]) == 0
    assert captured["name"] == "ai.chat"
    assert captured["argv"] == ["--cloud"]
    cli_menu.run_cloud_chat(["--cloud", "--json"])
    assert captured["argv"] == ["--cloud", "--json"]


def test_cli_dispatches_cloud() -> None:
    from cli import __main__ as cli_entry

    assert cli_entry._COMMANDS["cloud"] is cli_menu.run_cloud
    assert cli_entry._COMMANDS["cloud-chat"] is cli_menu.run_cloud_chat


def test_gui_sidebar_has_a_cloud_action() -> None:
    keys = {action.key for action in gui_actions.ACTIONS}
    assert "cloud" in keys
    action = gui_actions.action_by_key("cloud")
    assert action.command is cli_menu.run_cloud_chat
    assert "Cloud" in action.label


def test_gui_mouth_choices_include_cloud() -> None:
    if not _display_available():
        pytest.skip("no display available")
    from gui import app as gui_app

    keys = [key for key, _label in gui_app.MOUTH_CHOICES]
    assert MOUTH_CLOUD in keys


def test_free_providers_are_named() -> None:
    """Gemini and Groq are offered as no-card choices, so a user need not pay."""
    from core.cloud_model import free_provider_names

    free = free_provider_names()
    assert "gemini" in free
    assert "groq" in free
    # A paid-only provider is not mislabelled as free.
    assert "openai" not in free


def test_service_remembers_the_cloud_provider(tmp_path) -> None:
    """The window's provider choice is persisted through the one service."""
    from gui.service import ChatService

    service = ChatService(data_root=tmp_path)
    assert service.cloud_provider() == DEFAULT_PROVIDER_KEY
    assert service.set_cloud_provider("groq") == "groq"
    assert service.cloud_provider() == "groq"
    # A fresh service (a new launch) reads the saved choice.
    assert ChatService(data_root=tmp_path).cloud_provider() == "groq"


def test_service_remembers_the_cloud_model(tmp_path) -> None:
    """The window's model choice is persisted too, so a retired default is fixed once."""
    from gui.service import ChatService

    service = ChatService(data_root=tmp_path)
    service.set_cloud_provider("gemini")
    # With nothing saved, the provider's own default is reported.
    assert service.cloud_model() == CATALOG["gemini"].default_model
    assert service.set_cloud_model("gemini-3.8-flash") == "gemini-3.8-flash"
    assert service.cloud_model() == "gemini-3.8-flash"
    # A fresh service (a new launch) reads the saved model.
    assert ChatService(data_root=tmp_path).cloud_model() == "gemini-3.8-flash"


def test_service_rejects_an_empty_cloud_model(tmp_path) -> None:
    from gui.service import ChatService

    service = ChatService(data_root=tmp_path)
    with pytest.raises(ValueError):
        service.set_cloud_model("   ")


def test_gui_provider_picker_changes_the_provider(tmp_path) -> None:
    """The settings picker stores the choice through the service (no network)."""
    if not _display_available():
        pytest.skip("no display available")
    import tkinter as tk

    from gui.app import FaydGUI
    from gui.service import ChatService

    root = tk.Tk()
    service = ChatService(data_root=tmp_path)
    window = FaydGUI(root, service=service)
    try:
        window._toggle_settings()  # build the settings panel (and the picker)
        window._on_cloud_provider_change("gemini")
        assert service.cloud_provider() == "gemini"
    finally:
        root.destroy()


def test_gui_model_field_sets_the_cloud_model(tmp_path) -> None:
    """Typing a model in the settings field stores it, so the window can fix a 404."""
    if not _display_available():
        pytest.skip("no display available")
    import tkinter as tk

    from gui.app import FaydGUI
    from gui.service import ChatService

    root = tk.Tk()
    service = ChatService(data_root=tmp_path)
    service.set_cloud_provider("gemini")
    window = FaydGUI(root, service=service)
    try:
        window._toggle_settings()
        assert window._cloud_model_var.get() == CATALOG["gemini"].default_model
        window._cloud_model_var.set("gemini-3.8-flash")
        window._on_cloud_model_change()
        assert service.cloud_model() == "gemini-3.8-flash"
    finally:
        root.destroy()


def test_service_rejects_an_unknown_cloud_provider(tmp_path) -> None:
    from gui.service import ChatService

    service = ChatService(data_root=tmp_path)
    with pytest.raises(ValueError):
        service.set_cloud_provider("not-a-provider")


def test_chat_parser_accepts_cloud_flags(monkeypatch, capsys) -> None:
    """The chat's own argument parser accepts the cloud flags."""
    from ai import chat

    class FakeMode:
        def list_sessions(self) -> list[str]:
            return []

    monkeypatch.setattr(chat, "build_session", lambda **_kw: FakeMode())
    assert chat.main(["--cloud", "--cloud-provider", "groq", "--list-sessions"]) == 0
    assert "No saved sessions yet." in capsys.readouterr().out


def test_gui_parser_accepts_cloud_flags() -> None:
    from gui.options import build_parser

    args = build_parser().parse_args(["--cloud", "--cloud-model", "gpt-4o"])
    assert args.cloud is True
    assert args.cloud_model == "gpt-4o"


# --- no dependency, no committed key ---------------------------------------


def test_the_layer_adds_no_third_party_dependency() -> None:
    """The client is stdlib only, so a base install reaches a cloud model."""
    root = Path(__file__).resolve().parents[1]
    text = (root / "core" / "cloud_model.py").read_text(encoding="utf-8")
    assert "import requests" not in text
    assert "import urllib.request" in text


def test_no_api_key_is_committed_anywhere_in_the_source() -> None:
    """The key is read from the environment; it must never be written to a file."""
    root = Path(__file__).resolve().parents[1]
    for name in ("core/cloud_model.py", "ai/cloud.py", "ai/cloud_mouth.py"):
        text = (root / name).read_text(encoding="utf-8")
        assert "api_key_env" in text
        # No hard-coded credential of any common shape.
        for marker in ("sk-proj-", "sk-ant-", "AIzaSy"):
            assert marker not in text


def test_engine_reads_the_key_from_the_environment_only(monkeypatch) -> None:
    monkeypatch.setenv("FAYD_TEST_KEY", "env-key")
    client = OpenAICompatibleClient(provider="openai", api_key_env="FAYD_TEST_KEY")
    assert client.api_key == "env-key"
    monkeypatch.delenv("FAYD_TEST_KEY")
    assert client.api_key is None


def test_default_api_key_env_is_named() -> None:
    from core.cloud_model import DEFAULT_API_KEY_ENV

    assert DEFAULT_API_KEY_ENV == "FAYD_CLOUD_API_KEY"


def test_no_env_key_is_present_by_default() -> None:
    """The test environment itself holds no key, so the default path is denied."""
    assert os.environ.get("OPENAI_API_KEY") in (None, "")
