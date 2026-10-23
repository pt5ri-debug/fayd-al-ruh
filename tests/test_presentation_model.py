"""Tests for the Presentation Studio's own authoring model.

The model is an optional Qwen3 GGUF loaded in-process through
``llama-cpp-python``, mirroring ``core.coder_model``'s pattern. These tests never
load a real model: they exercise the *contract* -- the catalog, the resolver, the
download gate, and the honest status -- plus the client and the author callback
through an injected runner, which is the seam that makes the whole feature
testable with no extra and no weights.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from core import presentation_model as module
from core.presentation_model import (
    CATALOG,
    DEFAULT_PRESENTATION_MODEL,
    DEFAULT_SPEC_KEY,
    LlamaCppPresentationClient,
    PresentationModelUnavailable,
    download_model,
    find_model_file,
    model_is_cached,
    presentation_backend_status,
    presentation_model_status,
    resolve_spec,
    spec_names,
    spec_summaries,
)


class ScriptedRunner:
    """A ``PresentationRunner`` that returns canned text and records its calls."""

    def __init__(self, replies: list[str] | None = None) -> None:
        self.replies = list(replies or ["Reservoir\n- Porosity"])
        self.calls: list[tuple[list[dict[str, str]], dict[str, object]]] = []

    def __call__(self, messages, **kwargs):  # pragma: no cover - simple shim
        return self.chat(messages, **kwargs)

    def chat(self, messages, **kwargs):
        self.calls.append((list(messages), dict(kwargs)))
        if len(self.replies) > 1:
            return self.replies.pop(0)
        return self.replies[0]


# ------------------------------------------------------------------ the catalog


def test_the_catalog_has_a_default_and_the_default_is_in_it() -> None:
    assert DEFAULT_SPEC_KEY in CATALOG
    assert CATALOG[DEFAULT_SPEC_KEY].key == DEFAULT_SPEC_KEY


def test_every_catalog_entry_is_a_gguf_from_a_hugging_face_repo() -> None:
    for key, spec in CATALOG.items():
        assert spec.key == key
        assert spec.filename.endswith(".gguf")
        assert "/" in spec.repo, "a repo must be owner/name so huggingface_hub can fetch it"
        assert spec.approx_size_gb > 0
        assert spec.summary


def test_spec_names_and_summaries_are_ordered_and_complete() -> None:
    names = spec_names()
    assert names == tuple(sorted(CATALOG))
    summaries = spec_summaries()
    assert [row[0] for row in summaries] == list(names)
    assert all(len(row) == 4 for row in summaries)
    assert DEFAULT_SPEC_KEY in names


def test_resolve_spec_accepts_a_key_and_the_empty_default_but_rejects_the_unknown() -> None:
    assert resolve_spec("qwen3-4b").key == "qwen3-4b"
    assert resolve_spec(None).key == DEFAULT_SPEC_KEY
    assert resolve_spec("").key == DEFAULT_SPEC_KEY
    assert resolve_spec("Qwen3-8B").key == "qwen3-8b", "a key is matched case-insensitively"
    # A catalog key is required: an arbitrary string is never silently fetched.
    with pytest.raises(PresentationModelUnavailable):
        resolve_spec("some-owner/Some-Model-GGUF")


# ----------------------------------------------------------------- the model file


def test_find_model_file_prefers_the_registry_named_file(tmp_path: Path) -> None:
    bigger = tmp_path / "zzz-bigger.gguf"
    bigger.write_bytes(b"x" * 2048)
    named = tmp_path / "chosen.gguf"
    named.write_bytes(b"x" * 16)
    (tmp_path / "model.json").write_text(
        '{"file": "chosen.gguf", "repo": "o/r"}', encoding="utf-8"
    )
    assert find_model_file(tmp_path) == named


def test_find_model_file_falls_back_to_the_largest_gguf(tmp_path: Path) -> None:
    small = tmp_path / "small.gguf"
    small.write_bytes(b"x" * 16)
    large = tmp_path / "large.gguf"
    large.write_bytes(b"x" * 4096)
    assert find_model_file(tmp_path) == large


def test_find_model_file_is_none_for_an_empty_or_missing_directory(tmp_path: Path) -> None:
    assert find_model_file(tmp_path) is None
    assert find_model_file(tmp_path / "missing") is None
    # A stray non-GGUF file must never be offered as a model.
    (tmp_path / "notes.txt").write_text("hi", encoding="utf-8")
    assert find_model_file(tmp_path) is None


def test_find_model_file_ignores_a_registry_that_names_a_missing_file(tmp_path: Path) -> None:
    present = tmp_path / "present.gguf"
    present.write_bytes(b"x" * 8)
    (tmp_path / "model.json").write_text('{"file": "gone.gguf"}', encoding="utf-8")
    assert find_model_file(tmp_path) == present


def test_model_is_cached_reports_truthfully(tmp_path: Path) -> None:
    assert not model_is_cached(tmp_path)
    spec = resolve_spec("qwen3-8b")
    (tmp_path / spec.filename).write_bytes(b"x" * 8)
    assert model_is_cached(tmp_path)


# ------------------------------------------------------------------- the client


def test_the_client_returns_the_runner_text_and_forwards_the_knobs() -> None:
    from core.coder_model import CoderMessage

    runner = ScriptedRunner(["  Reservoir\n- Porosity  "])
    client = LlamaCppPresentationClient(model="qwen3:8b", runner=runner)
    assert client.model == "qwen3:8b"
    assert client.backend == "in-process"
    reply = client.chat([CoderMessage(role="user", content="hi")])
    # The runner's text is returned verbatim; tidying is the author's job.
    assert reply == "  Reservoir\n- Porosity  "
    messages, kwargs = runner.calls[0]
    assert messages == [{"role": "user", "content": "hi"}]
    assert kwargs["temperature"] == 0.2
    assert kwargs["max_tokens"] == client.max_tokens


def test_the_client_refuses_to_run_without_a_model_file() -> None:
    """No weights is an honest error, never a silent empty answer."""
    from core.coder_model import CoderMessage

    client = LlamaCppPresentationClient(root="/nonexistent/presentation-root")
    with pytest.raises(PresentationModelUnavailable):
        client.chat([CoderMessage(role="user", content="hi")])


def test_the_client_rejects_an_empty_message_list() -> None:
    client = LlamaCppPresentationClient(model="m", runner=ScriptedRunner())
    with pytest.raises(ValueError):
        client.chat([])


def test_the_client_honours_an_explicit_model_path(tmp_path: Path) -> None:
    path = tmp_path / "mine.gguf"
    path.write_bytes(b"x" * 8)
    client = LlamaCppPresentationClient(path, runner=ScriptedRunner())
    assert client.model_path() == path
    assert client.is_installed()


def test_the_client_reports_a_missing_explicit_path(tmp_path: Path) -> None:
    client = LlamaCppPresentationClient(tmp_path / "gone.gguf", runner=ScriptedRunner())
    with pytest.raises(PresentationModelUnavailable):
        client.model_path()


def test_list_models_reports_what_is_on_disk(tmp_path: Path) -> None:
    client = LlamaCppPresentationClient(root=tmp_path)
    assert client.list_models() == ()
    spec = resolve_spec("qwen3-8b")
    (tmp_path / spec.filename).write_bytes(b"x" * 8)
    assert client.list_models() == (spec.filename,)


def test_the_client_returns_a_non_string_reply_unchanged_for_the_caller() -> None:
    """The client is a transport: it does not invent or reshape a runner's reply."""

    class OddRunner:
        def chat(self, messages, **kwargs):
            return {"nonsense": True}

    from core.coder_model import CoderMessage

    client = LlamaCppPresentationClient(model="m", runner=OddRunner())
    assert client.chat([CoderMessage(role="user", content="hi")]) == {"nonsense": True}


def test_the_client_wraps_a_runner_failure_in_the_honest_type() -> None:
    class FailingRunner:
        def chat(self, messages, **kwargs):
            raise RuntimeError("model exploded")

    from core.coder_model import CoderMessage

    client = LlamaCppPresentationClient(model="m", runner=FailingRunner())
    with pytest.raises(PresentationModelUnavailable) as excinfo:
        client.chat([CoderMessage(role="user", content="hi")])
    assert "model exploded" in str(excinfo.value)


# ------------------------------------------------------------------- the status


def test_status_reports_the_in_process_model_when_the_file_is_present(tmp_path: Path) -> None:
    spec = resolve_spec("qwen3-8b")
    (tmp_path / spec.filename).write_bytes(b"x" * 8)
    if not module.extra_available():
        pytest.skip("the presentation extra is not installed in this environment")
    status = presentation_backend_status(root=tmp_path, probe_ollama=False)
    assert status.available
    assert status.backend == "in-process"
    assert spec.filename in status.detail


def test_status_names_the_missing_extra_when_the_file_exists_without_it(tmp_path: Path) -> None:
    (tmp_path / "Qwen3-8B-Q4_K_M.gguf").write_bytes(b"x" * 8)
    if module.extra_available():
        pytest.skip("the presentation extra is installed in this environment")
    status = presentation_backend_status(root=tmp_path, probe_ollama=False)
    assert not status.available
    assert "extra is not installed" in status.detail
    assert "requirements-presentation.txt" in status.hint


def test_status_offers_the_download_when_the_extra_is_present_but_empty(tmp_path: Path) -> None:
    if not module.extra_available():
        pytest.skip("the presentation extra is not installed in this environment")
    status = presentation_backend_status(root=tmp_path, probe_ollama=False)
    assert not status.available
    assert "model-download" in status.hint


def test_status_never_mentions_a_coding_model(tmp_path: Path) -> None:
    """The status is this feature's own wording, not the Code Studio's."""
    status = presentation_model_status(root=tmp_path, probe_ollama=False)
    assert not status.available
    assert "coding model" not in status.summary()
    assert "qwen2.5-coder" not in status.summary()
    assert "presentation" in status.summary().lower()


def test_status_of_an_injected_available_client_is_available() -> None:
    client = LlamaCppPresentationClient(model="qwen3:8b", runner=ScriptedRunner())
    status = presentation_backend_status(client=client, probe_ollama=False)
    assert status.available
    assert status.backend == "injected"


def test_the_model_status_shape_matches_what_the_window_reads(tmp_path: Path) -> None:
    status = presentation_model_status(root=tmp_path, probe_ollama=False)
    assert status.model == DEFAULT_PRESENTATION_MODEL
    assert isinstance(status.installed, bool)
    assert isinstance(status.backend, str)


# ----------------------------------------------------------------- the download


def test_download_is_denied_without_the_human_approval(tmp_path: Path) -> None:
    """A download is an external action: it must pass the approval gate."""
    with pytest.raises(PresentationModelUnavailable) as excinfo:
        download_model("qwen3-8b", root=tmp_path, approved=False)
    message = str(excinfo.value).lower()
    assert "denied" in message
    assert "external" in message


def test_download_reports_the_missing_runtime_instead_of_failing_obscurely(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """With approval but no ``huggingface_hub``, the error names the real next step."""
    monkeypatch.setattr(module, "extra_available", lambda: False)
    with pytest.raises(PresentationModelUnavailable) as excinfo:
        download_model("qwen3-8b", root=tmp_path, approved=True)
    assert "requirements-presentation.txt" in str(excinfo.value)


def test_download_rejects_an_unknown_model_before_any_network_use(tmp_path: Path) -> None:
    with pytest.raises(PresentationModelUnavailable):
        download_model("nope-not-a-repo!", root=tmp_path, approved=True)


# ------------------------------------------------------------------- the author


class _AuthorRunner:
    """A runner whose replies are scripted per call."""

    def __init__(self, replies: list[str]) -> None:
        self.replies = list(replies)
        self.messages: list[list[dict[str, str]]] = []

    def chat(self, messages, **kwargs):
        self.messages.append(list(messages))
        return self.replies.pop(0) if len(self.replies) > 1 else self.replies[0]


def test_strip_thinking_removes_a_complete_block_and_its_whitespace() -> None:
    from ai.presentation_author import strip_thinking

    text = "before\n<thinking>a lot of private reasoning\nacross lines</thinking>\nafter"
    # The block goes; the surrounding lines stay, so the answer reads normally.
    assert strip_thinking(text) == "before\n\nafter"


def test_strip_thinking_drops_a_leading_block_so_the_answer_starts_clean() -> None:
    """The first surviving line becomes the slide title, so no blank lead remains."""
    from ai.presentation_author import strip_thinking

    text = "<thinking>plan the outline first</thinking>\n\nReservoir\n- Porosity"
    assert strip_thinking(text) == "Reservoir\n- Porosity"


def test_strip_thinking_removes_an_unterminated_block_to_the_end() -> None:
    """A reply cut off mid-thought must not leak the reasoning into a slide."""
    from ai.presentation_author import strip_thinking

    assert strip_thinking("Answer\n<thinking>still going and never closed") == "Answer"


def test_strip_thinking_leaves_ordinary_text_untouched() -> None:
    from ai.presentation_author import strip_thinking

    text = "Porosity\n- Void fraction\n- Measured by logs"
    assert strip_thinking(text) == text


def test_strip_thinking_removes_several_blocks() -> None:
    from ai.presentation_author import strip_thinking

    text = "one <thinking>x</thinking> two <thinking>y</thinking> three"
    assert strip_thinking(text) == "one  two  three"


def test_the_author_tidies_the_model_reply_before_returning_it(tmp_path: Path) -> None:
    from ai.presentation_author import load_presentation_author

    runner = _AuthorRunner(["<thinking>plan the outline</thinking>\n\nReservoir\n- Porosity"])
    author = load_presentation_author(runner=runner, root=tmp_path)
    assert author("propose an outline") == "Reservoir\n- Porosity"


def test_the_author_sends_a_system_prompt_then_the_question(tmp_path: Path) -> None:
    from ai.presentation_author import SYSTEM_PROMPT, load_presentation_author

    runner = _AuthorRunner(["Reservoir"])
    author = load_presentation_author(runner=runner, root=tmp_path)
    author("Draft the outline")
    messages = runner.messages[0]
    assert messages[0]["role"] == "system"
    assert messages[0]["content"] == SYSTEM_PROMPT
    assert messages[-1] == {"role": "user", "content": "Draft the outline"}


def test_the_author_is_stateless_so_the_service_owns_the_context(tmp_path: Path) -> None:
    """Each call is one self-contained question; the Studio supplies any history.

    Keeping the adapter stateless means a retry or a re-draft cannot be skewed by
    an earlier exchange, and the Studio's own prompts (which already carry the deck
    title and heading) are the single source of context.
    """
    from ai.presentation_author import load_presentation_author

    runner = _AuthorRunner(["first answer", "second answer", "third answer"])
    author = load_presentation_author(runner=runner, root=tmp_path)
    assert author("one") == "first answer"
    assert author("two") == "second answer"
    assert author("three") == "third answer"
    # Every call carries exactly a system prompt and the one question.
    for messages in runner.messages:
        assert [message["role"] for message in messages] == ["system", "user"]
    assert [messages[-1]["content"] for messages in runner.messages] == ["one", "two", "three"]


def test_the_author_is_a_plain_callable_for_the_service(tmp_path: Path) -> None:
    from ai.presentation_author import load_presentation_author

    author = load_presentation_author(runner=_AuthorRunner(["Reservoir"]), root=tmp_path)
    assert callable(author)
    assert isinstance(author("hi"), str)
    assert "in-process" in author.describe() or "Ollama" in author.describe()


def test_loading_the_author_without_a_runtime_or_file_raises_honestly(tmp_path: Path) -> None:
    """No runner, no extra, no weights: the caller must be told, not given silence."""
    from ai.presentation_author import load_presentation_author

    with pytest.raises(PresentationModelUnavailable):
        load_presentation_author(root=tmp_path)
