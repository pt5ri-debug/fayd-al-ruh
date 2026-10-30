"""The runtime adapters, exercised through their injection seams.

Nothing here needs ``llama-cpp-python``, numpy, a server, a GPU, or a model: each
adapter takes a fake runner/client/loader, exactly as the project's other engines
do. The tests that must know whether a real extra is installed are written to pass
either way, and say which branch they took.
"""

from __future__ import annotations

import struct
from pathlib import Path

import pytest
from core.model_files import (
    FORMAT_FAYD_CHECKPOINT,
    FORMAT_GGUF,
    probe_file,
)
from core.model_runtime import (
    RUNTIME_FOR_ENGINE,
    RUNTIME_FOR_FORMAT,
    RUNTIME_INPROCESS_TRAINED,
    RUNTIME_LLAMA_CPP,
    RUNTIME_NAMES,
    RUNTIME_OLLAMA,
    RUNTIME_UNKNOWN,
    InferenceResult,
    InProcessTrainedAdapter,
    LlamaCppAdapter,
    LoadResult,
    OllamaAdapter,
    RuntimeAdapter,
    RuntimeCapabilities,
    RuntimeStatus,
    RuntimeUnavailable,
    UnsupportedCapability,
    adapter_for,
    describe_runtimes,
    iter_messages,
    llama_cpp_available,
    llama_cpp_gpu_offload,
    llama_cpp_version,
    runtime_for_engine,
    runtime_for_format,
    runtime_names,
)

# ------------------------------------------------------------------ fixtures


def _gguf_key(text: str) -> bytes:
    raw = text.encode("utf-8")
    return struct.pack("<Q", len(raw)) + raw


def _gguf_string(text: str) -> bytes:
    raw = text.encode("utf-8")
    return struct.pack("<I", 8) + struct.pack("<Q", len(raw)) + raw


def _gguf_uint32(value: int) -> bytes:
    return struct.pack("<I", 4) + struct.pack("<I", value)


def write_gguf(path, *, architecture="llama", context=4096):
    entries = [
        ("general.name", _gguf_string("Runtime Model")),
        ("general.architecture", _gguf_string(architecture)),
        ("general.file_type", _gguf_uint32(15)),
        (f"{architecture}.context_length", _gguf_uint32(context)),
    ]
    body = b"".join(_gguf_key(key) + payload for key, payload in entries)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"GGUF" + struct.pack("<IQQ", 3, 4, len(entries)) + body)
    return path


class FakeRunner:
    """The seam a real ``llama-cpp-python`` runner occupies."""

    def __init__(self, reply="ready", error=None):
        self.reply = reply
        self.error = error
        self.calls: list[tuple] = []

    def chat(self, messages, **kwargs):
        self.calls.append((messages, kwargs))
        if self.error is not None:
            raise self.error
        return self.reply


class FakeOllamaClient:
    """The seam a real ``OllamaCoderClient`` occupies (it carries a ``model`` name)."""

    def __init__(self, reply="ready", models=("qwen2.5-coder:7b",), error=None):
        self.model = models[0] if models else "qwen2.5-coder:7b"
        self.reply = reply
        self._models = models
        self.error = error
        self.calls: list[tuple] = []

    def list_models(self):
        if self.error is not None:
            raise self.error
        return list(self._models)

    def chat(self, messages, **kwargs):
        self.calls.append((messages, kwargs))
        if self.error is not None:
            raise self.error
        return self.reply


# ------------------------------------------------------------ capabilities


def test_capabilities_report_what_a_runtime_can_do():
    caps = RuntimeCapabilities(can_load=True, can_infer=True)
    assert caps.supports("can_load") is True
    assert caps.supports("can_stop") is False
    assert "can_stop" in caps.unsupported()
    assert "load" in caps.describe()


def test_full_capabilities_describe_themselves():
    caps = RuntimeCapabilities(
        can_load=True, can_unload=True, can_stop=True, can_discover=True,
        can_validate=True, can_infer=True, gpu_offload=True, streaming=True, vision=True,
    )
    assert caps.describe() == "full capabilities"
    assert caps.unsupported() == ()


def test_status_summary_is_honest_about_gpu_support():
    ready = RuntimeStatus(runtime="x", available=True, gpu_offload=True)
    assert "gpu offload available" in ready.summary()
    cpu = RuntimeStatus(runtime="x", available=True, gpu_offload=False)
    assert "cpu only" in cpu.summary()
    unknown = RuntimeStatus(runtime="x", available=True, gpu_offload=None)
    assert "gpu support unknown" in unknown.summary()


def test_status_summary_names_the_reason_when_unavailable():
    status = RuntimeStatus(runtime="x", available=False, detail="nothing installed")
    assert "nothing installed" in status.summary()


def test_results_describe_themselves_without_raising():
    ok = LoadResult(ok=True, runtime="x", model_path=Path("/a.gguf"), seconds=0.5)
    assert "loaded" in ok.describe()
    bad = LoadResult(ok=False, runtime="x", detail="it broke")
    assert "it broke" in bad.describe()
    answered = InferenceResult(ok=True, runtime="x", text="hello world", seconds=0.1)
    assert "hello world" in answered.describe()
    failed = InferenceResult(ok=False, runtime="x", detail="no model")
    assert "no model" in failed.describe()


def test_llama_cpp_adapter_satisfies_the_protocol():
    adapter = LlamaCppAdapter(runner=FakeRunner())
    assert isinstance(adapter, RuntimeAdapter)


def test_ollama_adapter_satisfies_the_protocol():
    assert isinstance(OllamaAdapter(client=FakeOllamaClient()), RuntimeAdapter)


def test_trained_adapter_satisfies_the_protocol():
    assert isinstance(InProcessTrainedAdapter(loader=lambda path: object()), RuntimeAdapter)


# ------------------------------------------------------------- llama.cpp


def test_llama_cpp_capabilities_are_honest():
    caps = LlamaCppAdapter(runner=FakeRunner()).capabilities()
    assert caps.can_load is True
    assert caps.can_unload is True
    assert caps.can_infer is True
    # There is no process or server to stop.
    assert caps.can_stop is False
    assert caps.resident is True


def test_an_injected_runner_reports_gpu_support_as_unknown():
    """A fake runner must not be reported as a GPU build."""
    adapter = LlamaCppAdapter(runner=FakeRunner())
    assert adapter.capabilities().gpu_offload is None
    assert adapter.status().gpu_offload is None
    assert adapter.status().available is True
    assert adapter.status().version == "injected runner"


def test_llama_cpp_status_without_the_extra_names_the_install():
    adapter = LlamaCppAdapter()
    status = adapter.status()
    if llama_cpp_available():  # pragma: no cover - environment dependent
        assert status.available is True
    else:
        assert status.available is False
        assert "llama-cpp-python" in status.detail
        assert "pip install" in status.hint


def test_llama_cpp_loads_and_tests_through_the_runner(tmp_path):
    target = write_gguf(tmp_path / "m.gguf")
    runner = FakeRunner(reply="hello there")
    adapter = LlamaCppAdapter(runner=runner)
    loaded = adapter.load(target)
    assert loaded.ok is True
    assert adapter.loaded() is True
    assert adapter.loaded_path() == target
    result = adapter.infer("say hi")
    assert result.ok is True
    assert result.text == "hello there"
    assert runner.calls


def test_llama_cpp_refuses_an_empty_test_prompt(tmp_path):
    adapter = LlamaCppAdapter(runner=FakeRunner())
    adapter.load(write_gguf(tmp_path / "m.gguf"))
    result = adapter.infer("   ")
    assert result.ok is False
    assert "prompt" in result.detail


def test_llama_cpp_refuses_a_test_before_a_load():
    adapter = LlamaCppAdapter(runner=FakeRunner())
    result = adapter.infer("hi")
    assert result.ok is False
    assert "no model is loaded" in result.detail.lower()


def test_llama_cpp_reports_a_runner_failure_instead_of_raising(tmp_path):
    adapter = LlamaCppAdapter(runner=FakeRunner(error=RuntimeError("boom")))
    adapter.load(write_gguf(tmp_path / "m.gguf"))
    result = adapter.infer("hi")
    assert result.ok is False
    assert "boom" in result.detail


def test_llama_cpp_unload_is_honest_about_in_process_memory():
    adapter = LlamaCppAdapter(runner=FakeRunner())
    adapter.load(Path("/tmp/whatever.gguf"))
    result = adapter.unload()
    assert result.ok is True
    assert adapter.loaded() is False
    # It does not claim the memory came back.
    assert "collects them" in result.detail


def test_llama_cpp_unload_before_a_load_is_a_no_op():
    adapter = LlamaCppAdapter(runner=FakeRunner())
    result = adapter.unload()
    assert result.ok is True
    assert "no model was loaded" in result.detail.lower()


def test_llama_cpp_validation_accepts_a_gguf_and_refuses_another_format(tmp_path):
    adapter = LlamaCppAdapter(runner=FakeRunner())
    good = probe_file(write_gguf(tmp_path / "m.gguf"))
    assert adapter.validate(good) == (True, "")
    # A real file of another format is refused because of its format...
    from tests.test_model_files import write_checkpoint

    checkpoint_path = tmp_path / "fayd-llm.npz"
    write_checkpoint(checkpoint_path)
    checkpoint = probe_file(checkpoint_path)
    ok, reason = adapter.validate(checkpoint)
    assert ok is False
    assert "GGUF" in reason
    # ...while a file that is not there at all is refused for that reason.
    missing_ok, missing_reason = adapter.validate(probe_file(tmp_path / "gone.gguf"))
    assert missing_ok is False
    assert "no file" in missing_reason.lower()


def test_llama_cpp_validation_refuses_a_missing_file(tmp_path):
    adapter = LlamaCppAdapter(runner=FakeRunner())
    ok, reason = adapter.validate(probe_file(tmp_path / "gone.gguf"))
    assert ok is False
    assert reason


def test_llama_cpp_discovery_lists_only_gguf_files(tmp_path):
    write_gguf(tmp_path / "a.gguf")
    write_gguf(tmp_path / "b.gguf")
    (tmp_path / "notes.txt").write_text("hi")
    adapter = LlamaCppAdapter(runner=FakeRunner(), root=tmp_path)
    assert adapter.discover() == ("a.gguf", "b.gguf")


def test_llama_cpp_discovery_without_a_root_is_empty():
    assert LlamaCppAdapter(runner=FakeRunner()).discover() == ()


def test_llama_cpp_load_without_the_extra_fails_honestly(tmp_path):
    if llama_cpp_available():  # pragma: no cover - environment dependent
        pytest.skip("llama-cpp-python is installed here")
    adapter = LlamaCppAdapter()
    result = adapter.load(write_gguf(tmp_path / "m.gguf"))
    assert result.ok is False
    assert "llama-cpp-python" in result.detail


def test_llama_cpp_inference_options_come_from_the_settings():
    from core.model_runtime import _inference_options
    from core.model_settings import ModelSettings

    settings = ModelSettings(temperature=0.25, top_p=0.8, max_tokens=64)
    options = _inference_options(settings)
    assert options["temperature"] == pytest.approx(0.25)
    assert options["top_p"] == pytest.approx(0.8)
    assert options["max_tokens"] == 64
    assert _inference_options(None) == {}


def test_llama_cpp_version_and_gpu_probe_never_raise():
    assert isinstance(llama_cpp_version(), str)
    assert llama_cpp_gpu_offload() in (True, False, None)


# ----------------------------------------------------------------- Ollama


def test_ollama_capabilities_are_honest_about_the_server():
    caps = OllamaAdapter(client=FakeOllamaClient()).capabilities()
    assert caps.can_load is True
    assert caps.can_unload is True
    # The project never stops the user's daemon.
    assert caps.can_stop is False
    # Ollama decides its own GPU usage, so the honest answer is unknown.
    assert caps.gpu_offload is None


def test_ollama_status_uses_the_injected_client():
    adapter = OllamaAdapter(client=FakeOllamaClient())
    status = adapter.status()
    assert status.runtime == RUNTIME_OLLAMA
    assert isinstance(status.available, bool)


def test_ollama_status_reports_a_server_that_did_not_answer():
    from core.coder_model import CoderUnavailable

    adapter = OllamaAdapter(client=FakeOllamaClient(error=CoderUnavailable("no server")))
    status = adapter.status()
    assert status.available is False


def test_ollama_discovery_returns_the_pulled_models():
    adapter = OllamaAdapter(client=FakeOllamaClient(models=("a:1b", "b:7b")))
    assert adapter.discover() == ("a:1b", "b:7b")


def test_ollama_discovery_is_empty_when_the_server_is_down():
    from core.coder_model import CoderUnavailable

    adapter = OllamaAdapter(client=FakeOllamaClient(error=CoderUnavailable("down")))
    assert adapter.discover() == ()


def test_ollama_refuses_a_local_gguf_path_with_an_explanation(tmp_path):
    adapter = OllamaAdapter(client=FakeOllamaClient())
    ok, reason = adapter.validate(probe_file(write_gguf(tmp_path / "m.gguf")))
    assert ok is False
    assert "pulled" in reason


def test_ollama_load_explains_that_it_loads_on_first_request():
    adapter = OllamaAdapter(client=FakeOllamaClient())
    result = adapter.load(Path("qwen2.5-coder:7b"))
    assert result.ok is True
    assert "first request" in result.detail
    assert adapter.loaded() is True


def test_ollama_unload_says_it_does_not_stop_the_server():
    adapter = OllamaAdapter(client=FakeOllamaClient())
    adapter.load(Path("a:1b"))
    result = adapter.unload()
    assert result.ok is True
    assert "does not stop the server" in result.detail
    assert adapter.loaded() is False


def test_ollama_infers_through_the_client():
    client = FakeOllamaClient(reply="ollama says hi")
    adapter = OllamaAdapter(client=client)
    result = adapter.infer("hi")
    assert result.ok is True
    assert result.text == "ollama says hi"
    assert client.calls


def test_ollama_inference_failure_is_reported():
    from core.coder_model import CoderUnavailable

    adapter = OllamaAdapter(client=FakeOllamaClient(error=CoderUnavailable("down")))
    result = adapter.infer("hi")
    assert result.ok is False
    assert "down" in result.detail


def test_ollama_refuses_an_empty_prompt():
    result = OllamaAdapter(client=FakeOllamaClient()).infer("")
    assert result.ok is False


# ---------------------------------------------------- in-process trained


def test_trained_capabilities_report_cpu_only():
    caps = InProcessTrainedAdapter(loader=lambda path: object()).capabilities()
    assert caps.can_load is True
    assert caps.can_infer is True
    # The numpy transformer is CPU only by design, and says so.
    assert caps.gpu_offload is False
    # There is no folder of checkpoints to browse.
    assert caps.can_discover is False


def test_trained_discovery_raises_an_unsupported_capability():
    adapter = InProcessTrainedAdapter(loader=lambda path: object())
    with pytest.raises(UnsupportedCapability):
        adapter.discover()


def test_trained_validation_refuses_a_gguf(tmp_path):
    adapter = InProcessTrainedAdapter(loader=lambda path: object())
    ok, reason = adapter.validate(probe_file(write_gguf(tmp_path / "m.gguf")))
    assert ok is False
    assert ".npz" in reason


def test_trained_loads_a_checkpoint_through_the_injected_loader(tmp_path):
    from tests.test_model_files import write_checkpoint

    target = tmp_path / "fayd-llm.npz"
    write_checkpoint(target)
    seen: list[Path] = []

    def loader(path):
        seen.append(path)
        return "the model"

    adapter = InProcessTrainedAdapter(loader=loader)
    result = adapter.load(target)
    assert result.ok is True
    assert adapter.loaded() is True
    assert seen == [target]


def test_trained_loader_failure_is_reported(tmp_path):
    from tests.test_model_files import write_checkpoint

    target = tmp_path / "fayd-llm.npz"
    write_checkpoint(target)

    def loader(path):
        raise ValueError("bad checkpoint")

    adapter = InProcessTrainedAdapter(loader=loader)
    result = adapter.load(target)
    assert result.ok is False
    assert "bad checkpoint" in result.detail
    assert adapter.loaded() is False


def test_trained_refuses_a_file_that_is_not_a_checkpoint(tmp_path):
    adapter = InProcessTrainedAdapter(loader=lambda path: object())
    result = adapter.load(tmp_path / "notes.txt")
    assert result.ok is False
    assert adapter.loaded() is False


def test_trained_refuses_a_test_before_a_load():
    result = InProcessTrainedAdapter(loader=lambda path: object()).infer("hi")
    assert result.ok is False
    assert "no checkpoint" in result.detail.lower()


def test_trained_unload_drops_the_reference(tmp_path):
    from tests.test_model_files import write_checkpoint

    target = tmp_path / "fayd-llm.npz"
    write_checkpoint(target)
    adapter = InProcessTrainedAdapter(loader=lambda path: object())
    adapter.load(target)
    result = adapter.unload()
    assert result.ok is True
    assert adapter.loaded() is False


def test_trained_status_reports_whether_numpy_is_present():
    status = InProcessTrainedAdapter(loader=lambda path: object()).status()
    assert status.runtime == RUNTIME_INPROCESS_TRAINED
    assert status.gpu_offload is False
    assert isinstance(status.available, bool)


# ---------------------------------------------------------------- factory


def test_adapter_for_builds_each_known_runtime():
    assert isinstance(adapter_for(RUNTIME_LLAMA_CPP, runner=FakeRunner()), LlamaCppAdapter)
    assert isinstance(adapter_for(RUNTIME_OLLAMA, client=FakeOllamaClient()), OllamaAdapter)
    assert isinstance(
        adapter_for(RUNTIME_INPROCESS_TRAINED, loader=lambda path: object()),
        InProcessTrainedAdapter,
    )


def test_adapter_for_refuses_an_unknown_runtime_with_the_known_names():
    with pytest.raises(RuntimeUnavailable) as caught:
        adapter_for("not-a-runtime")
    assert "not-a-runtime" in str(caught.value)
    assert RUNTIME_LLAMA_CPP in str(caught.value)


def test_runtime_for_format_maps_both_supported_formats():
    assert runtime_for_format(FORMAT_GGUF) == RUNTIME_LLAMA_CPP
    assert runtime_for_format(FORMAT_FAYD_CHECKPOINT) == RUNTIME_INPROCESS_TRAINED
    assert runtime_for_format("something-else") == RUNTIME_UNKNOWN


def test_runtime_for_engine_records_what_each_mouth_uses():
    for engine in ("ultron", "vision", "banner", "natasha", "hakeem", "coder", "docs"):
        assert runtime_for_engine(engine) == RUNTIME_LLAMA_CPP
    assert runtime_for_engine("model") == RUNTIME_INPROCESS_TRAINED
    assert runtime_for_engine("nonsense") == RUNTIME_UNKNOWN


def test_the_runtime_maps_agree_with_the_named_runtimes():
    for runtime in RUNTIME_FOR_FORMAT.values():
        assert runtime in RUNTIME_NAMES
    # Every runtime an engine maps to is either adaptable here or is one the
    # project already has an owner for (a cloud endpoint, or no model at all).
    from core.model_runtime import RUNTIME_VALUES

    for runtime in RUNTIME_FOR_ENGINE.values():
        assert runtime in RUNTIME_VALUES


def test_runtime_names_are_stable():
    assert runtime_names() == RUNTIME_NAMES
    assert RUNTIME_LLAMA_CPP in RUNTIME_NAMES


def test_describe_runtimes_reports_every_runtime():
    report = describe_runtimes()
    for name in RUNTIME_NAMES:
        assert name in report


# ------------------------------------------------------------------ helpers


def test_iter_messages_normalises_dicts_and_objects():
    from core.coder_model import CoderMessage

    messages = [
        {"role": "system", "content": "be brief"},
        CoderMessage(role="user", content="hello"),
    ]
    assert list(iter_messages(messages)) == [
        {"role": "system", "content": "be brief"},
        {"role": "user", "content": "hello"},
    ]


def test_iter_messages_skips_an_object_with_no_role():
    class Empty:
        pass

    assert list(iter_messages([Empty()])) == []


def test_unsupported_capability_is_a_runtime_unavailable():
    assert issubclass(UnsupportedCapability, RuntimeUnavailable)
    assert issubclass(RuntimeUnavailable, RuntimeError)
