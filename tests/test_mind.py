"""The mind: answering from the trained model instead of the retrieved sources."""

from __future__ import annotations

import pytest
from ai import mind
from ai.local_model import LocalLanguageModel
from ml.llm.tokenizer import ByteTokenizer
from ml.llm.transformer import TransformerConfig, TransformerLM, train_language_model

_SMALL = TransformerConfig(n_layer=1, n_head=2, n_embd=16, block_size=16)


def _tiny_model(tmp_path) -> LocalLanguageModel:
    """A tiny but real model, so the mind runs the same code path as the release."""
    model = TransformerLM(_SMALL, seed=0)
    stream = ByteTokenizer().encode("Porosity measures the voids in rock. " * 20, bos=False)
    train_language_model(model, stream, steps=20, batch_size=4, seed=0)
    checkpoint = tmp_path / "llm.npz"
    model.save(checkpoint)
    return LocalLanguageModel(checkpoint, max_new_tokens=8, temperature=0.0, seed=0)


def test_render_prompt_frames_a_question() -> None:
    assert mind.render_prompt("what is porosity?") == "Question: what is porosity?\nAnswer:"


def test_render_prompt_strips_surrounding_whitespace() -> None:
    assert mind.render_prompt("  porosity  ") == "Question: porosity\nAnswer:"


def test_mind_answer_returns_the_continuation_only(tmp_path) -> None:
    assistant = mind.MindAssistant(_tiny_model(tmp_path), max_new_tokens=8)
    result = assistant.answer("porosity")
    assert result.text
    assert "Question:" not in result.text
    assert "Answer:" not in result.text


class _EchoModel:
    """A fake mind model whose generator returns a fixed continuation."""

    def __init__(self, continuation: str) -> None:
        self._continuation = continuation

        class _Generator:
            def __init__(self, continuation: str) -> None:
                self._continuation = continuation

            def generate(
                self,
                prompt: str,
                *,
                max_new_tokens: int = 64,
                temperature: float = 0.8,
                seed: int | None = None,
            ) -> str:
                return prompt + self._continuation

        self.generator = _Generator(continuation)


def test_mind_answer_composes_the_continuation_into_sentences() -> None:
    """The mouth tidies ragged model output into whole, punctuated sentences."""
    assistant = mind.MindAssistant(
        _EchoModel("the mud weight is 1.2 sg. The procedure for Prod"), max_new_tokens=8
    )
    result = assistant.answer("porosity")
    assert result.text == "The mud weight is 1.2 sg."
    # The untouched continuation is kept for inspection.
    assert result.raw_text == "the mud weight is 1.2 sg. The procedure for Prod"


def test_mind_answer_centres_the_answer_on_the_question() -> None:
    """The question steers which whole sentence opens the model's answer."""
    assistant = mind.MindAssistant(
        _EchoModel(
            "Compute the surface facilities. The mud weight controls the pressure."
        ),
        max_new_tokens=8,
    )
    result = assistant.answer("How does mud weight work?")
    assert result.text == (
        "The mud weight controls the pressure. Compute the surface facilities."
    )


def test_mind_answer_collapses_a_repeat_run_the_model_stalls_on() -> None:
    """A byte model that loops on a phrase shows it once, not several times."""
    assistant = mind.MindAssistant(
        _EchoModel("schedule the casing the casing the casing the casing the lamin"),
        max_new_tokens=8,
    )
    result = assistant.answer("how is casing designed?")
    assert result.text == "Schedule the casing the lamin."
    # The untouched continuation keeps the loop, for inspection.
    assert result.raw_text == "schedule the casing the casing the casing the casing the lamin"


def test_mind_answer_rejects_an_empty_question(tmp_path) -> None:
    assistant = mind.MindAssistant(_tiny_model(tmp_path))
    with pytest.raises(ValueError):
        assistant.answer("   ")


class _RecordingGenerator:
    """Records the settings each generation call is made with."""

    def __init__(self, continuation: str) -> None:
        self._continuation = continuation
        self.calls: list[dict] = []

    def generate(
        self,
        prompt: str,
        *,
        max_new_tokens: int = 64,
        temperature: float = 0.8,
        seed: int | None = None,
    ) -> str:
        self.calls.append(
            {"max_new_tokens": max_new_tokens, "temperature": temperature, "seed": seed}
        )
        return prompt + self._continuation


class _RecordingModel:
    def __init__(self, continuation: str, *, temperature: float, seed: int | None) -> None:
        self.temperature = temperature
        self.seed = seed
        self.generator = _RecordingGenerator(continuation)


def test_mind_uses_the_models_temperature_and_seed() -> None:
    """The mind must not fall back to the generator's own sampling defaults.

    The generator defaults to ``temperature=0.8`` and no seed; if the mind did
    not forward the loaded settings, ``--mind-temperature 0`` would still sample
    and answers would not be reproducible.
    """
    model = _RecordingModel("The well flows.", temperature=0.0, seed=7)
    assistant = mind.MindAssistant(model)
    assistant.answer("porosity")

    call = model.generator.calls[-1]
    assert call["temperature"] == 0.0
    assert call["seed"] == 7


def test_mind_greedy_generation_is_deterministic(tmp_path) -> None:
    """With a real greedy model, two runs of the same question are identical."""
    assistant = mind.MindAssistant(_tiny_model(tmp_path), max_new_tokens=8)
    assert assistant.answer("porosity").text == assistant.answer("porosity").text


def test_mind_assistant_can_override_the_generation_settings() -> None:
    model = _RecordingModel("The well flows.", temperature=0.0, seed=0)
    assistant = mind.MindAssistant(model, temperature=0.9, seed=3)
    assistant.answer("porosity")
    call = model.generator.calls[-1]
    assert call["temperature"] == 0.9
    assert call["seed"] == 3


def test_mind_ask_matches_the_grounded_assistant_shape(tmp_path) -> None:
    from core.conversation import Conversation

    assistant = mind.MindAssistant(_tiny_model(tmp_path), max_new_tokens=8)
    conversation = Conversation(title="test")
    answer = assistant.ask("porosity", conversation=conversation)

    assert answer.used_model is True
    assert answer.citations == ()
    assert answer.passages == ()
    assert answer.confidence == 0.0
    assert answer.text.startswith(mind.MIND_HEADER)
    assert answer.conversation_id == conversation.id
    last_user = conversation.last_user_message()
    assert last_user is not None and last_user.text == "porosity"


def test_mind_ask_starts_a_conversation_when_none_is_given(tmp_path) -> None:
    assistant = mind.MindAssistant(_tiny_model(tmp_path), max_new_tokens=8)
    answer = assistant.ask("porosity")
    assert answer.conversation_id


def test_mind_ask_never_returns_a_header_with_no_body() -> None:
    """An empty continuation must not leave a blank answer bubble in the GUI."""
    assistant = mind.MindAssistant(_EchoModel(""), max_new_tokens=8)
    answer = assistant.ask("porosity")
    assert answer.text == f"{mind.MIND_HEADER}\n\n{mind.MIND_EMPTY}"


def test_mind_ask_replaces_wordless_model_output() -> None:
    """A stopping run such as ')))))' is reported, not rendered as punctuation."""
    assistant = mind.MindAssistant(_EchoModel(")))))"), max_new_tokens=8)
    answer = assistant.ask("porosity")
    assert answer.text == f"{mind.MIND_HEADER}\n\n{mind.MIND_EMPTY}"


def test_load_mind_model_reports_a_missing_checkpoint(tmp_path) -> None:
    with pytest.raises(FileNotFoundError):
        mind.load_mind_model(tmp_path / "absent.npz")


def test_mind_status_reports_absence(tmp_path) -> None:
    status = mind.mind_status(tmp_path / "absent.npz")
    assert status["exists"] is False
    assert "absent.npz" in status["checkpoint"]


def test_mind_status_reports_a_present_checkpoint(tmp_path) -> None:
    _tiny_model(tmp_path)
    status = mind.mind_status(tmp_path / "llm.npz")
    assert status["exists"] is True
    assert status["size_bytes"] > 0
    assert "modified" in status


def test_status_line_names_the_checkpoint(tmp_path) -> None:
    _tiny_model(tmp_path)
    line = mind.status_line(mind.mind_status(tmp_path / "llm.npz"))
    assert "Model mode" in line
    assert "llm.npz" in line


def test_switch_phrases_are_disjoint() -> None:
    model_phrases = {p.casefold() for p in mind.SWITCH_TO_MODEL}
    source_phrases = {p.casefold() for p in mind.SWITCH_TO_SOURCES}
    assert model_phrases.isdisjoint(source_phrases)
