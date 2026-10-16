"""Tests for the in-process coding model (``core/coder_llama.py``).

The point of this module is that the Code Studio can write code with **no Ollama
server left running in the background**. The runtime itself is
``llama-cpp-python``, an optional extra, so the tests never import it: they drive
the client with an injected runner and check the discovery, the honest status
reporting, and the download gate. A test that needs the real model is skipped
when the extra is absent.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from core.coder_llama import (
    CATALOG,
    DEFAULT_SPEC_KEY,
    LlamaCppCoderClient,
    CoderModelUnavailable,
    coder_backend_status,
    download_model,
    extra_available,
    find_model_file,
    model_is_cached,
    resolve_spec,
    spec_names,
    spec_summaries,
)
from core.coder_model import CoderMessage, coder_status

# --- a fake runner, so no llama_cpp and no model file are needed -----------


class FakeRunner:
    """A scripted stand-in for ``LlamaCppRunner``."""

    def __init__(self, reply: str = '{"action": "done", "summary": "ok"}') -> None:
        self.reply = reply
        self.calls: list[tuple[list[dict[str, str]], dict]] = []

    def chat(self, messages: list[dict[str, str]], **kwargs) -> str:
        self.calls.append((messages, kwargs))
        return self.reply


class ExplodingRunner:
    def chat(self, messages, **kwargs):
        raise RuntimeError("the model blew up")


def _make_model(root: Path, name: str = "Qwen2.5-Coder-7B-Instruct-Q4_K_M.gguf") -> Path:
    path = root / name
    path.write_bytes(b"gguf" * 32)
    return path


# --- catalog ---------------------------------------------------------------


def test_default_spec_is_the_coder_7b() -> None:
    assert resolve_spec(None).key == DEFAULT_SPEC_KEY
    assert DEFAULT_SPEC_KEY in CATALOG


def test_unknown_spec_names_the_available_ones() -> None:
    with pytest.raises(CoderModelUnavailable) as excinfo:
        resolve_spec("gpt-9")
    assert "gpt-9" in str(excinfo.value)
    assert "qwen2.5-coder-7b" in str(excinfo.value)


def test_spec_summaries_cover_the_catalog() -> None:
    keys = [row[0] for row in spec_summaries()]
    assert keys == list(spec_names())
    assert set(keys) == set(CATALOG)


# --- discovery -------------------------------------------------------------


def test_no_model_file_reports_nothing(tmp_path: Path) -> None:
    assert find_model_file(tmp_path) is None
    assert model_is_cached(tmp_path) is False


def test_registry_file_wins_over_the_largest(tmp_path: Path) -> None:
    """The downloaded model is the one loaded, even if a bigger file is present."""
    _make_model(tmp_path, "chosen.gguf")
    bigger = tmp_path / "bigger.gguf"
    bigger.write_bytes(b"x" * 10_000)
    (tmp_path / "model.json").write_text(json.dumps({"file": "chosen.gguf"}), encoding="utf-8")
    assert find_model_file(tmp_path) == tmp_path / "chosen.gguf"


def test_largest_model_is_the_fallback(tmp_path: Path) -> None:
    _make_model(tmp_path, "small.gguf")
    big = tmp_path / "big.gguf"
    big.write_bytes(b"x" * 5_000)
    assert find_model_file(tmp_path) == big


def test_corrupt_registry_falls_back_to_the_glob(tmp_path: Path) -> None:
    _make_model(tmp_path, "only.gguf")
    (tmp_path / "model.json").write_text("{not json", encoding="utf-8")
    assert find_model_file(tmp_path) == tmp_path / "only.gguf"


# --- the client ------------------------------------------------------------


def test_chat_returns_the_runner_reply_and_passes_temperature(tmp_path: Path) -> None:
    _make_model(tmp_path)
    runner = FakeRunner("hello")
    client = LlamaCppCoderClient(root=tmp_path, runner=runner)
    assert client.chat([CoderMessage("user", "hi")], temperature=0.35) == "hello"
    assert runner.calls[0][1]["temperature"] == 0.35


def test_client_is_lazy_until_the_first_chat(tmp_path: Path) -> None:
    """Constructing a client must never load a model or touch the disk."""
    _make_model(tmp_path)
    client = LlamaCppCoderClient(root=tmp_path, runner=FakeRunner())
    assert client._runner is not None  # injected; nothing loaded yet
    # An empty folder constructs fine too, and only fails when it must resolve.
    empty = LlamaCppCoderClient(root=tmp_path / "nope")
    with pytest.raises(CoderModelUnavailable):
        empty.model_path()


def test_missing_model_file_is_an_actionable_error(tmp_path: Path) -> None:
    client = LlamaCppCoderClient(root=tmp_path)
    with pytest.raises(CoderModelUnavailable) as excinfo:
        client.model_path()
    assert "studio download" in str(excinfo.value)


def test_a_runner_failure_becomes_a_clear_error(tmp_path: Path) -> None:
    _make_model(tmp_path)
    client = LlamaCppCoderClient(root=tmp_path, runner=ExplodingRunner())
    with pytest.raises(CoderModelUnavailable):
        client.chat([CoderMessage("user", "hi")])


def test_chat_needs_at_least_one_message(tmp_path: Path) -> None:
    _make_model(tmp_path)
    client = LlamaCppCoderClient(root=tmp_path, runner=FakeRunner())
    with pytest.raises(ValueError):
        client.chat([])


# --- status: the wording must match the runtime ----------------------------


def test_status_says_in_process_and_not_ollama(tmp_path: Path) -> None:
    _make_model(tmp_path)
    client = LlamaCppCoderClient(root=tmp_path, runner=FakeRunner())
    status = coder_status(client)
    assert status.available is True
    assert status.backend == "in-process"
    assert "in-process" in status.summary()
    assert "Ollama" not in status.summary()


def test_status_when_the_extra_is_missing_is_actionable(tmp_path: Path) -> None:
    client = LlamaCppCoderClient(root=tmp_path)
    status = coder_status(client)
    assert status.available is False
    assert "requirements-coder.txt" in status.summary()


def test_ollama_wording_is_unchanged(tmp_path: Path) -> None:
    """The default backend stays Ollama, so existing users see no change."""
    from core.coder_model import CoderStatus

    status = CoderStatus(False, False, "qwen2.5-coder:7b", False)
    assert "Ollama" in status.summary()


def test_backend_status_prefers_in_process(tmp_path: Path, monkeypatch) -> None:
    _make_model(tmp_path)
    monkeypatch.setattr("core.coder_llama.extra_available", lambda: True)
    status = coder_backend_status(root=tmp_path, probe_ollama=False)
    assert status.backend == "in-process"
    assert status.available is True


def test_backend_status_offers_the_extra_when_the_model_exists(
    tmp_path: Path, monkeypatch
) -> None:
    _make_model(tmp_path)
    monkeypatch.setattr("core.coder_llama.extra_available", lambda: False)
    status = coder_backend_status(root=tmp_path, probe_ollama=False)
    assert status.available is False
    assert status.hint == "python -m pip install -r requirements-coder.txt"


def test_backend_status_points_at_the_download_when_the_extra_exists(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setattr("core.coder_llama.extra_available", lambda: True)
    status = coder_backend_status(root=tmp_path, probe_ollama=False)
    assert status.available is False
    assert "studio download" in status.hint


# --- the download gate -----------------------------------------------------


def test_download_is_denied_without_approval(tmp_path: Path) -> None:
    with pytest.raises(CoderModelUnavailable) as excinfo:
        download_model("qwen2.5-coder-7b", root=tmp_path)
    assert "denied" in str(excinfo.value).lower() or "approval" in str(excinfo.value).lower()


def test_download_returns_an_already_present_file(tmp_path: Path) -> None:
    """A file already on disk is reused, so a second run does not re-download."""
    spec = CATALOG[DEFAULT_SPEC_KEY]
    existing = tmp_path / spec.filename
    existing.write_bytes(b"gguf")
    assert download_model(DEFAULT_SPEC_KEY, root=tmp_path, approved=True) == existing


def test_download_needs_the_hub_when_approved(tmp_path: Path) -> None:
    if _hub_available():
        pytest.skip("huggingface_hub is installed, so the missing-extra path is not reached")
    with pytest.raises(CoderModelUnavailable) as excinfo:
        download_model(DEFAULT_SPEC_KEY, root=tmp_path, approved=True)
    assert "huggingface_hub" in str(excinfo.value)


def _hub_available() -> bool:
    import importlib.util

    return importlib.util.find_spec("huggingface_hub") is not None


# --- the real runtime, only when the extra is present ----------------------


@pytest.mark.skipif(not extra_available(), reason="the [coder] extra is not installed")
def test_extra_reports_available_when_installed() -> None:
    assert extra_available() is True
