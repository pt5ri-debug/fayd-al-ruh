"""Streamed turns: the answer arrives piece by piece, and a trace is filtered out.

A companion answers with a reasoning model, and two things went wrong for the
reader: the hidden reasoning block ate the answer's token budget (so a reply
looked cut off) and the answer appeared only once it was finished. These tests pin
the fix -- a raised, *stored* answer limit, a streaming path through the runtime,
the manager, the router, the orchestrator, and the service, and a filter that keeps
a reasoning block out of what is shown -- without needing a model, a display, or a
network: every one drives the real code through an injected seam.
"""

from __future__ import annotations

from pathlib import Path

from core.composition import ThinkingFilter, strip_thinking
from core.model_manager import ModelManager
from core.model_registry import ModelRecord
from core.model_runtime import InferenceResult, RuntimeCapabilities
from core.model_settings import ModelSettings
from core.romance.characters import CharacterDraft, CharacterManager
from core.romance.conversation import ConversationManager
from core.romance.orchestrator import (
    KIND_DELTA,
    KIND_DONE,
    KIND_ERROR,
    KIND_REFUSED,
    CompanionOrchestrator,
)
from core.romance.routing import CompanionModelRouter, settings_from
from core.romance.storage import CompanionStorageService
from gui.romance import (
    SETTING_MAX_TOKENS,
    SETTING_MODEL_THINKING,
    RomanceStudioService,
    _default_setting,
)

OPEN = chr(60) + "thinking" + chr(62)
CLOSE = chr(60) + "/thinking" + chr(62)
SHORT_OPEN = chr(60) + "think" + chr(62)
SHORT_CLOSE = chr(60) + "/think" + chr(62)


def _storage(tmp_path: Path) -> CompanionStorageService:
    return CompanionStorageService(root=tmp_path / "romance")


def _conversation(storage: CompanionStorageService) -> tuple[str, str]:
    character = CharacterManager(storage).create(
        CharacterDraft(name="Noor", fictional_age=27)
    ).character
    conversation = ConversationManager(storage).start(character.record_id)
    return character.record_id, conversation.record_id


class _RefusingSafety:
    """A safety floor that refuses every turn (a deterministic refusal control)."""

    def check_content(self, character_id, level, *, text=""):
        return _Decision(allowed=False, reason="refused by the test floor")


class _Decision:
    def __init__(self, *, allowed: bool, reason: str = "") -> None:
        self.allowed = allowed
        self.reason = reason


# -- the reasoning-trace filter ---------------------------------------------


def test_strip_thinking_removes_a_closed_block_and_a_trailing_one():
    assert strip_thinking(f"Answer. {OPEN}why{CLOSE} Done.") == "Answer.  Done."
    assert strip_thinking(f"Answer. {SHORT_OPEN}why{SHORT_CLOSE} Done.") == "Answer.  Done."
    # An unterminated block is a trace that never closed: drop it to the end.
    assert strip_thinking(f"Answer. {OPEN}why") == "Answer."
    assert strip_thinking("no trace here") == "no trace here"


def test_the_streaming_filter_holds_a_tag_split_across_pieces():
    pieces = [
        "Hello ",
        "there. " + chr(60) + "thi",
        "nk" + chr(62) + "hidden" + chr(60) + "/th",
        "ink" + chr(62) + " How ",
        "are you?",
    ]
    stream = ThinkingFilter()
    shown = "".join(stream.feed(piece) for piece in pieces) + stream.flush()
    assert shown == "Hello there.  How are you?"


def test_the_streaming_filter_drops_an_unterminated_trace_at_flush():
    stream = ThinkingFilter()
    pieces = ["Only. " + chr(60) + "thi", "nk" + chr(62) + "all trace"]
    shown = "".join(stream.feed(piece) for piece in pieces) + stream.flush()
    assert shown == "Only. "


def test_the_streaming_filter_passes_a_plain_answer_untouched():
    stream = ThinkingFilter()
    shown = "".join(stream.feed(piece) for piece in ["Plain ", "answer."]) + stream.flush()
    assert shown == "Plain answer."


# -- the runtime and the manager --------------------------------------------


def test_the_llama_adapter_exposes_a_stream_seam():
    from core.model_runtime import LlamaCppAdapter

    assert hasattr(LlamaCppAdapter, "stream")


class _StreamingAdapter:
    """A fake runtime that streams, so the manager is exercised with no weights."""

    def __init__(self) -> None:
        self.calls = 0

    def capabilities(self) -> RuntimeCapabilities:
        return RuntimeCapabilities(can_load=True, can_infer=True, streaming=True)

    def loaded(self) -> bool:
        return True

    def load(self, model_path, *, settings=None):  # pragma: no cover - not needed
        raise AssertionError("a loaded adapter must not be loaded again")

    def infer(self, prompt, *, settings=None):  # pragma: no cover - not needed
        raise AssertionError("a streaming runtime must stream")

    def stream(self, prompt, *, settings=None):
        self.calls += 1
        yield "one "
        yield "two"


def test_the_manager_streams_through_the_adapter(tmp_path: Path):
    adapter = _StreamingAdapter()
    manager = ModelManager(tmp_path, adapter_factory=lambda runtime: adapter)
    manager.registry.register_record(
        ModelRecord(
            model_id="fake-model",
            identity="fake/model.gguf",
            display_name="Fake",
            runtime="llama.cpp",
            path=tmp_path / "f.gguf",
        )
    )
    pieces = [piece.text for piece in manager.stream("fake-model", prompt="hi")]
    assert pieces == ["one ", "two"]
    assert adapter.calls == 1


def test_the_manager_reports_a_runtime_that_cannot_stream(tmp_path: Path):
    class _OneShotAdapter:
        def capabilities(self) -> RuntimeCapabilities:
            return RuntimeCapabilities(can_infer=True, streaming=False)

        def loaded(self) -> bool:
            return True

        def infer(self, prompt, *, settings=None):
            return InferenceResult(ok=True, runtime="one-shot", text="whole reply")

    manager = ModelManager(tmp_path, adapter_factory=lambda runtime: _OneShotAdapter())
    manager.registry.register_record(
        ModelRecord(
            model_id="one-shot",
            identity="fake/one-shot.gguf",
            display_name="One-shot",
            runtime="llama.cpp",
            path=tmp_path / "o.gguf",
        )
    )
    pieces = [piece.text for piece in manager.stream("one-shot", prompt="hi")]
    assert pieces == ["whole reply"]


# -- the router --------------------------------------------------------------


def test_settings_from_clamps_and_keeps_the_answer_inside_the_context():
    settings = settings_from(context=2048, max_tokens=4096, temperature=0.9)
    assert isinstance(settings, ModelSettings)
    assert settings.context == 2048
    # A reply has to fit the window it is generated in.
    assert settings.max_tokens == 2048
    assert settings.temperature == 0.9
    default = settings_from(max_tokens=1024)
    assert default.context >= default.max_tokens


def test_the_hidden_reasoning_trace_is_off_by_default_in_a_conversation():
    # The trace is not something the character says, and it spends the answer's
    # own token budget -- the cause of a reply arriving cut short. A conversational
    # surface therefore turns it off, while the low-level default stays unchanged
    # so every non-conversational caller is unaffected.
    assert ModelSettings().thinking is True
    assert settings_from(max_tokens=256).thinking is False
    assert CompanionModelRouter().safe_settings().thinking is False
    assert settings_from(thinking=True).thinking is True


def test_thinking_off_asks_the_chat_template_not_to_write_a_trace():
    from core.model_runtime import _inference_options

    off = _inference_options(settings_from(max_tokens=128))
    assert off["chat_template_kwargs"] == {"enable_thinking": False}
    # Thinking on (or an older settings object with no field) adds no such option.
    assert "chat_template_kwargs" not in _inference_options(ModelSettings())
    assert "chat_template_kwargs" not in _inference_options(None)


def test_the_settings_record_round_trips_the_thinking_choice():
    on = ModelSettings(thinking=True)
    off = ModelSettings(thinking=False)
    assert on.to_primitive()["thinking"] is True
    assert off.to_primitive()["thinking"] is False
    assert ModelSettings.from_primitive(off.to_primitive()).thinking is False
    # A record written before the field existed keeps the safe default.
    assert ModelSettings.from_primitive({"temperature": 0.5}).thinking is True
    # A hand-edited string is read, and anything else falls back rather than raising.
    assert ModelSettings.from_primitive({"thinking": "off"}).thinking is False
    assert ModelSettings.from_primitive({"thinking": "on"}).thinking is True
    assert ModelSettings.from_primitive({"thinking": 7}).thinking is True
    assert "thinking off" in off.describe()


def test_the_studio_defaults_thinking_off_and_a_user_can_turn_it_back_on(tmp_path: Path):
    service = RomanceStudioService(root=tmp_path / "studio")
    assert service.model_thinking() is False
    assert service.model_settings().thinking is False
    service.set_setting(SETTING_MODEL_THINKING, "on")
    assert service.model_thinking() is True
    assert service.model_settings().thinking is True


def test_the_llama_runner_retries_without_an_option_an_old_build_rejects():
    from core.model_runtime import _LlamaChatRunner

    class _OldBuild:
        """A llama-cpp-python that does not accept ``chat_template_kwargs``."""

        def __init__(self) -> None:
            self.calls = 0

        def create_chat_completion(self, messages, **kwargs):
            self.calls += 1
            if "chat_template_kwargs" in kwargs:
                raise TypeError("unexpected keyword argument 'chat_template_kwargs'")
            return {"choices": [{"message": {"content": "ok"}}]}

    build = _OldBuild()
    runner = _LlamaChatRunner(build)
    reply = runner.chat([{"role": "user", "content": "hi"}], chat_template_kwargs={"enable_thinking": False})
    assert reply == "ok"
    assert build.calls == 2


def test_the_router_settings_from_delegates_to_the_module_function():
    router = CompanionModelRouter()
    assert router.settings_from(max_tokens=256).max_tokens == 256
    assert router.settings_from(max_tokens=256, thinking=True).thinking is True


def test_the_router_stream_reports_an_uninstalled_model_honestly(tmp_path: Path):
    # The catalog seam is the documented injection point: with no model to choose,
    # the router must report the honest failure rather than pretending to answer.
    router = CompanionModelRouter(ModelManager(tmp_path), catalog=lambda: ())
    pieces = list(router.stream("hello", key="definitely-not-a-model"))
    assert pieces
    assert all(not piece.ok for piece in pieces)
    assert pieces[-1].detail


# -- the orchestrator --------------------------------------------------------


def test_send_stream_yields_pieces_then_the_stored_reply(tmp_path: Path):
    storage = _storage(tmp_path)
    _, conversation_id = _conversation(storage)

    def seam(prompt, key, handle):
        yield from [
            "Hello ",
            "there. " + chr(60) + "thi",
            "nk" + chr(62) + "secret" + chr(60) + "/th",
            "ink" + chr(62) + " How ",
            "are you?",
        ]

    orchestrator = CompanionOrchestrator(storage, stream_generator=seam)
    events = list(orchestrator.send_stream(conversation_id, "Hi."))
    deltas = [event.text for event in events if event.kind == KIND_DELTA]
    final = events[-1]

    assert "secret" not in "".join(deltas)
    assert "".join(deltas) == "Hello there.  How are you?"
    assert final.kind == KIND_DONE
    # The stored reply is that same filtered text.
    assert final.result is not None
    assert final.result.reply == "Hello there.  How are you?"
    stored = ConversationManager(storage).messages(conversation_id)
    assert [message.role for message in stored] == ["user", "assistant"]
    assert stored[-1].content == "Hello there.  How are you?"


def test_send_stream_stores_nothing_when_generation_fails(tmp_path: Path):
    storage = _storage(tmp_path)
    _, conversation_id = _conversation(storage)

    def seam(prompt, key, handle):
        yield InferenceResult(ok=True, runtime="seam", text="Half an ans")
        yield InferenceResult(ok=False, runtime="seam", detail="the model stopped")

    orchestrator = CompanionOrchestrator(storage, stream_generator=seam)
    events = list(orchestrator.send_stream(conversation_id, "Hi."))
    assert events[-1].kind == KIND_ERROR
    assert events[-1].reason == "the model stopped"
    # Only the user's own message was written -- no half answer is recorded.
    stored = ConversationManager(storage).messages(conversation_id)
    assert [message.role for message in stored] == ["user"]


def test_send_stream_reports_an_answer_that_was_all_reasoning(tmp_path: Path):
    storage = _storage(tmp_path)
    _, conversation_id = _conversation(storage)

    def seam(prompt, key, handle):
        yield OPEN + "only a trace"
        yield CLOSE

    orchestrator = CompanionOrchestrator(storage, stream_generator=seam)
    events = list(orchestrator.send_stream(conversation_id, "Hi."))
    assert events[-1].kind == KIND_ERROR
    assert "empty" in events[-1].reason
    assert ConversationManager(storage).messages(conversation_id)[-1].role == "user"


def test_send_stream_honours_the_one_shot_seam_too(tmp_path: Path):
    storage = _storage(tmp_path)
    _, conversation_id = _conversation(storage)
    orchestrator = CompanionOrchestrator(
        storage, generator=lambda prompt, key, handle: ("One piece.", True, "")
    )
    events = list(orchestrator.send_stream(conversation_id, "Hi."))
    assert [event.text for event in events if event.kind == KIND_DELTA] == ["One piece."]
    assert events[-1].kind == KIND_DONE


def test_send_stream_refuses_before_writing_anything(tmp_path: Path):
    storage = _storage(tmp_path)
    _, conversation_id = _conversation(storage)
    orchestrator = CompanionOrchestrator(
        storage, safety=_RefusingSafety(), stream_generator=lambda p, k, h: ["never"]
    )
    events = list(orchestrator.send_stream(conversation_id, "Hi."))
    assert len(events) == 1
    assert events[0].kind == KIND_REFUSED
    assert ConversationManager(storage).messages(conversation_id) == ()


def test_send_stream_passes_the_settings_it_was_given(tmp_path: Path):
    storage = _storage(tmp_path)
    _, conversation_id = _conversation(storage)
    seen: dict[str, object] = {}

    class _Router:
        def stream(self, prompt, *, key=None, settings=None, handle=None):
            seen["settings"] = settings
            yield InferenceResult(ok=True, runtime="fake", text="Answer.")

    settings = ModelSettings(context=4096, max_tokens=1234, temperature=0.5)
    orchestrator = CompanionOrchestrator(
        storage, router=_Router(), settings=settings  # type: ignore[arg-type]
    )
    list(orchestrator.send_stream(conversation_id, "Hi."))
    assert seen["settings"] is settings


# -- the service -------------------------------------------------------------


def test_the_service_hands_its_stored_settings_to_the_model(tmp_path: Path):
    service = RomanceStudioService(root=tmp_path / "studio")
    settings = service.model_settings()
    assert settings.max_tokens == int(float(_default_setting(SETTING_MAX_TOKENS)))
    assert settings.max_tokens <= settings.context
    service.set_setting(SETTING_MAX_TOKENS, "2048")
    service._apply_model_settings()
    assert service.orchestrator.settings.max_tokens == 2048


def test_the_service_streams_a_turn_and_stores_it(tmp_path: Path):
    service = RomanceStudioService(root=tmp_path / "studio")

    def seam(prompt, key, handle):
        yield from ["Once ", "upon a time."]

    service.orchestrator._stream_generator = seam
    character = service.create_character("Layla", fictional_age=30)
    conversation = service.start_conversation(character.id)
    events = list(service.send_stream(conversation.record_id, "Tell me."))
    deltas = [event.text for event in events if event.kind == KIND_DELTA]
    final = events[-1]
    assert "".join(deltas) == "Once upon a time."
    assert final.kind == KIND_DONE
    assert service.messages(conversation.record_id)[-1].content == "Once upon a time."


def test_the_default_answer_limit_is_raised_above_the_old_cap():
    # The reader's complaint was a truncated answer, so the shipped default must be
    # comfortably above the old 512-token cap.
    assert float(_default_setting(SETTING_MAX_TOKENS)) >= 1024


# -- the CLI ---------------------------------------------------------------


def _run_cli(argv, capsys):
    from cli import romance as cli

    code = cli.main(argv)
    return code, capsys.readouterr().out


def test_cli_chat_streams_the_reply_as_it_is_written(tmp_path: Path, capsys):
    import json

    from cli import romance as cli

    root = str(tmp_path / "cli")
    _run_cli(["--data-root", root, "new", "Layla", "--age", "27"], capsys)
    _, out = _run_cli(["--data-root", root, "characters", "--json"], capsys)
    character_id = json.loads(out)[0]["character_id"]
    # The service the CLI builds is patched to stream through a seam, so the whole
    # path -- stream, filter, store, report -- is exercised with no model.
    original = cli._service

    def _service(args):
        service = original(args)
        service.orchestrator._stream_generator = lambda p, k, h: iter(
            ["Hello ", "there. " + OPEN, "hidden", CLOSE + " Done."]
        )
        return service

    cli._service = _service
    try:
        code, out = _run_cli(["--data-root", root, "chat", character_id, "Hi."], capsys)
    finally:
        cli._service = original
    assert code == cli.EXIT_OK
    # The reply was streamed to stdout, with the reasoning trace kept out of it.
    assert "Hello there.  Done." in out
    assert "hidden" not in out


def test_the_studios_settings_really_reach_the_runtime(tmp_path: Path):
    """The Studio's answer limit, context, and temperature reach the runtime.

    This is the heart of the fix: the window's controls used to be saved and
    ignored, so a reply was capped by the engine's own default no matter what the
    user chose. The settings now travel Studio -> router -> manager -> adapter.
    """
    from core.model_runtime import RuntimeCapabilities

    seen: dict[str, object] = {}

    class _Adapter:
        def capabilities(self) -> RuntimeCapabilities:
            return RuntimeCapabilities(can_infer=True, streaming=True)

        def loaded(self) -> bool:
            return True

        def stream(self, prompt, *, settings=None):
            seen["settings"] = settings
            yield "ok"

    class _CatalogModel:
        key = "fake"
        title = "Fake"
        engine = "llama.cpp"
        available = True
        detail = ""
        path: Path

    model = tmp_path / "fake.gguf"
    model.write_bytes(b"GGUF" + b"\x00" * 64)
    _CatalogModel.path = model
    manager = ModelManager(tmp_path, adapter_factory=lambda runtime: _Adapter())
    manager.registry.register_record(
        ModelRecord(
            model_id="fake",
            identity="fake/x.gguf",
            display_name="Fake",
            runtime="llama.cpp",
            path=model,
        )
    )
    router = CompanionModelRouter(manager, catalog=lambda: (_CatalogModel(),))
    settings = router.settings_from(context=8192, max_tokens=2048, temperature=0.6)
    list(router.stream("hi", settings=settings))
    assert seen["settings"] is settings
    assert seen["settings"].max_tokens == 2048
    assert seen["settings"].context == 8192


def test_a_reply_that_hit_the_output_cap_is_recorded_not_discarded(tmp_path: Path):
    """A truncated reply is kept, and the log says why it stopped.

    The reader's complaint was a reply that looked cut off. When the model really
    did hit the answer limit, the partial reply is still a real answer, so it is
    kept -- and the reason is recorded rather than left for the reader to guess.
    """
    from core.model_runtime import RuntimeCapabilities

    class _CappedAdapter:
        last_finish_reason = ""

        def capabilities(self) -> RuntimeCapabilities:
            return RuntimeCapabilities(can_infer=True, streaming=True)

        def loaded(self) -> bool:
            return True

        def stream(self, prompt, *, settings=None):
            yield "A long reply that ran out"
            self.last_finish_reason = "length"

    adapter = _CappedAdapter()
    manager = ModelManager(tmp_path, adapter_factory=lambda runtime: adapter)
    manager.registry.register_record(
        ModelRecord(
            model_id="capped",
            identity="fake/capped.gguf",
            display_name="Capped",
            runtime="llama.cpp",
            path=tmp_path / "c.gguf",
        )
    )
    pieces = list(manager.stream("capped", prompt="hi"))
    # The partial reply is kept, not turned into a failure.
    assert [piece.text for piece in pieces] == ["A long reply that ran out"]
    assert all(piece.ok for piece in pieces)
    messages = " ".join(event.message for event in manager.log.tail(limit=5))
    assert "output limit" in messages


# -- the in-process trained runtime streams too -----------------------------


def test_the_trained_generator_streams_exactly_what_it_generates():
    """The in-process generator can be watched, and streaming changes nothing.

    The companion's own trained checkpoint is the one model a clean install really
    has, and its runtime used to report ``streaming=False`` -- so a reply arrived
    whole, at the end. ``LocalGenerator.generate_stream`` yields the same
    continuation as ``generate``, piece by piece, so a reader watches the answer
    grow without the text changing.
    """
    from ml.llm.generator import LocalGenerator
    from ml.llm.transformer import TransformerConfig, TransformerLM

    config = TransformerConfig(vocab_size=258, block_size=32, n_layer=1, n_head=2, n_embd=16)
    model = TransformerLM(config)
    generator = LocalGenerator(model)
    prompt = "Porosity is"
    full = generator.generate(prompt, max_new_tokens=12, temperature=0.0)[len(prompt):]
    pieces = list(generator.generate_stream(prompt, max_new_tokens=12, temperature=0.0))
    # The concatenation is *exactly* the one-shot reply, and it arrives in pieces.
    assert "".join(pieces) == full
    assert len(pieces) > 1


def test_the_trained_generator_streams_a_seeded_run_identically():
    """A seeded stream and a seeded one-shot draw the same sequence."""
    from ml.llm.generator import LocalGenerator
    from ml.llm.transformer import TransformerConfig, TransformerLM

    config = TransformerConfig(vocab_size=258, block_size=32, n_layer=1, n_head=2, n_embd=16)
    generator = LocalGenerator(TransformerLM(config))
    prompt = "The reservoir"
    full = generator.generate(prompt, max_new_tokens=10, temperature=0.9, seed=11)[len(prompt):]
    pieces = "".join(
        generator.generate_stream(prompt, max_new_tokens=10, temperature=0.9, seed=11)
    )
    assert pieces == full


def test_the_streamed_pieces_never_split_a_multibyte_character(monkeypatch):
    """No piece is a broken glyph, even when a character spans two byte tokens.

    The vocabulary is bytes, so an Arabic letter is two tokens; decoding each
    token on its own would turn every half into ``�``. The stream decodes through
    an incremental UTF-8 decoder, so a whole character is held until both its
    bytes have arrived and every yielded piece is whole. The sampler is pinned to
    a known byte sequence so the property is tested on its own, not on whatever an
    untrained model happens to emit.
    """
    from ml.llm.generator import LocalGenerator
    from ml.llm.tokenizer import BOS_ID, ByteTokenizer
    from ml.llm.transformer import TransformerConfig, TransformerLM

    generator = LocalGenerator(TransformerLM(
        TransformerConfig(vocab_size=258, block_size=64, n_layer=1, n_head=2, n_embd=16)
    ))
    word = "مس"  # two Arabic letters, four bytes, i.e. four byte tokens
    sequence = iter([*list(word.encode("utf-8")), BOS_ID])

    import ml.llm.generator as generator_module

    monkeypatch.setattr(generator_module, "sample", lambda *a, **k: next(sequence))
    pieces = list(generator.generate_stream("x", max_new_tokens=8, temperature=0.0))

    assert "".join(pieces) == word
    # No replacement character is ever emitted, and no piece holds half a glyph:
    # the first letter only appears once both of its bytes have arrived.
    assert all("\ufffd" not in piece for piece in pieces)
    assert pieces[0] == word[:1]
    # The tokenizer round-trips the Arabic word itself (sanity on the harness).
    assert ByteTokenizer().decode(ByteTokenizer().encode(word)) == word


def test_the_trained_adapter_reports_streaming_and_has_a_stream_seam():
    from core.model_runtime import InProcessTrainedAdapter

    adapter = InProcessTrainedAdapter(loader=lambda path: object())
    assert adapter.capabilities().streaming is True
    assert hasattr(adapter, "stream")


def test_the_trained_adapter_streams_through_the_manager(tmp_path: Path, monkeypatch):
    """The manager streams a trained checkpoint instead of returning it whole."""
    from core.model_runtime import InProcessTrainedAdapter

    class _Model:
        pass

    class _Generator:
        def __init__(self, model):
            self._model = model

        def generate_stream(self, prompt, *, max_new_tokens, temperature):
            yield "one "
            yield "two"

    adapter = InProcessTrainedAdapter(loader=lambda path: _Model())
    # Drive the adapter directly: load is bypassed by injecting the loaded model.
    adapter._model = _Model()
    import ml.llm.generator as generator_module

    monkeypatch.setattr(generator_module, "LocalGenerator", _Generator)
    pieces = list(adapter.stream("hi"))
    assert pieces == ["one ", "two"]


def test_the_ollama_client_exposes_a_stream_seam():
    from core.coder_model import OllamaCoderClient

    assert hasattr(OllamaCoderClient, "chat_stream")


def test_the_ollama_adapter_reports_streaming_and_has_a_stream_seam():
    from core.model_runtime import OllamaAdapter

    adapter = OllamaAdapter(client=object())
    assert adapter.capabilities().streaming is True
    assert hasattr(adapter, "stream")


def test_the_ollama_adapter_streams_through_a_client_that_can():
    """A client that streams is used; the pieces are passed straight through."""
    from core.model_runtime import OllamaAdapter

    class _Client:
        def chat(self, messages, *, temperature=0.2):  # pragma: no cover
            raise AssertionError("a streaming client must stream")

        def chat_stream(self, messages, *, temperature=0.2):
            yield "a "
            yield "b"

    adapter = OllamaAdapter(client=_Client())
    assert list(adapter.stream("hi")) == ["a ", "b"]


def test_the_ollama_adapter_falls_back_when_its_client_cannot_stream():
    """A client with no streaming method is called once, never refused."""
    from core.model_runtime import OllamaAdapter

    class _Client:
        def chat(self, messages, *, temperature=0.2):
            return "the whole reply"

    adapter = OllamaAdapter(client=_Client())
    assert list(adapter.stream("hi")) == ["the whole reply"]


def test_the_shipped_answer_limit_is_comfortably_larger_than_the_old_cap():
    # The reader asked for more words: the shipped default is well above the old
    # 512-token cap and above the earlier 1024, so a long answer is not cut off.
    assert float(_default_setting(SETTING_MAX_TOKENS)) >= 2048


def test_the_bubble_cap_allows_a_long_companion_answer():
    # The reader's second complaint was a reply drawn only up to the old cap, so a
    # complete answer looked truncated. The cap must be far above a long reply.
    from gui.romance_dialog import _MAX_BUBBLE_LINES

    assert _MAX_BUBBLE_LINES >= 1000
