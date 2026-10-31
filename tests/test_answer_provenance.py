"""The answer provenance line: which model answered, and what it rested on.

Every window must show, after an answer, the model that produced it and the
sources it used (``core.answer_provenance``). These tests pin the builder, its
honest fallbacks, the bilingual labels, the turn wiring, and the footer widget
that paints the line -- all without a display, a model, or a download.
"""

from __future__ import annotations

import pytest
from core.answer_provenance import (
    MODEL_LABEL,
    NO_SOURCES,
    RETRIEVAL_SOURCES,
    SOURCES_LABEL,
    UNKNOWN_MODEL,
    AnswerProvenance,
    answer_provenance,
    describe_answer_model,
)


class _Described:
    """A stand-in assistant that describes itself, like every real mouth does."""

    def describe(self) -> str:
        return "Ultron: qwen2.5-7b.gguf (4.0 GB) at data/ultron."


def _display_available() -> bool:
    """True only when a real Tk root can be built (a headless run skips)."""
    try:
        import tkinter as tk

        root = tk.Tk()
    except Exception:
        return False
    root.destroy()
    return True


class _Silent:
    """An assistant with no ``describe`` at all -- the honest-fallback case."""


class _Broken:
    def describe(self) -> str:
        raise RuntimeError("the model is gone")


# --- describe_answer_model ------------------------------------------------


def test_describe_answer_model_prefers_the_assistants_own_description():
    assert describe_answer_model(_Described()) == "Ultron: qwen2.5-7b.gguf (4.0 GB) at data/ultron."


def test_describe_answer_model_survives_a_broken_describe():
    # A mouth whose describe() raises must not break the answer -- it falls back.
    assert describe_answer_model(_Broken()) == describe_answer_model(None) == ""


def test_describe_answer_model_falls_back_to_the_catalog():
    # No assistant, but a mouth key: the shared catalog names its default model.
    text = describe_answer_model(None, mouth="ultron")
    assert text
    assert "Ultron" in text or "qwen" in text.lower()


def test_describe_answer_model_never_invents_a_name():
    assert describe_answer_model(_Silent(), mouth="") == ""
    assert describe_answer_model(None, mouth="not-a-mouth") == ""


# --- AnswerProvenance rendering -------------------------------------------


def test_model_line_names_the_model_or_says_it_is_unnamed():
    named = AnswerProvenance(model_label="Ultron: qwen2.5-7b.")
    assert named.model_line() == "Answer model: Ultron: qwen2.5-7b."
    blank = AnswerProvenance()
    assert blank.model_line() == f"Answer model: {UNKNOWN_MODEL['en']}"


def test_sources_line_lists_the_citations_when_there_are_any():
    provenance = AnswerProvenance(sources=("Doc A", "Doc B"))
    assert provenance.sources_line() == "Sources: Doc A · Doc B"


def test_sources_line_says_what_a_model_answer_without_citations_rests_on():
    provenance = AnswerProvenance(model_label="Ultron.")
    assert provenance.sources_line() == f"Sources: {NO_SOURCES['en']}"


def test_sources_line_says_a_retrieval_answer_used_the_corpus():
    provenance = AnswerProvenance(retrieval_only=True)
    assert provenance.sources_line() == f"Sources: {RETRIEVAL_SOURCES['en']}"


def test_describe_includes_extra_lines_after_model_and_sources():
    provenance = AnswerProvenance(
        model_label="Doc's Studio: qwen3-vl-4b.", sources=("page 3",),
        extra=("Pages read: 3",),
    )
    lines = provenance.describe().splitlines()
    assert lines[0].startswith(MODEL_LABEL["en"])
    assert lines[1].startswith(SOURCES_LABEL["en"])
    assert lines[2] == "Pages read: 3"


def test_describe_can_omit_the_sources_line_for_a_caller_that_drew_it():
    # The turn renderer draws its own "Sources:" block; the provenance must not
    # repeat the citations a second time.
    provenance = AnswerProvenance(model_label="Ultron.", sources=("Doc A",))
    assert "Doc A" not in provenance.describe(include_sources=False)
    assert "Doc A" in provenance.describe(include_sources=True)


# --- bilingual labels -----------------------------------------------------


def test_an_arabic_question_gets_arabic_labels():
    provenance = answer_provenance(
        _Described(), citations=("مصدر أ", "مصدر ب"), question="ما هي المسامية؟"
    )
    assert provenance.language == "ar"
    assert provenance.model_line().startswith(MODEL_LABEL["ar"])
    assert provenance.sources_line().startswith(SOURCES_LABEL["ar"])
    assert "، " in provenance.sources_line()


def test_an_explicit_language_wins_over_the_questions_script():
    provenance = answer_provenance(None, question="ما هي المسامية؟", language="en")
    assert provenance.language == "en"


def test_an_english_question_gets_english_labels():
    provenance = answer_provenance(_Described(), question="what is porosity?")
    assert provenance.language == "en"
    assert provenance.model_line().startswith(MODEL_LABEL["en"])


# --- answer_provenance assembly -------------------------------------------


def test_answer_provenance_names_the_model_and_lists_the_citations():
    provenance = answer_provenance(
        _Described(), citations=("Doc A", "Doc B"), question="what is porosity?"
    )
    assert provenance.model_label == "Ultron: qwen2.5-7b.gguf (4.0 GB) at data/ultron."
    assert provenance.sources == ("Doc A", "Doc B")
    assert provenance.mouth == ""


def test_answer_provenance_records_web_and_learned_flags():
    provenance = answer_provenance(None, used_web=True, learned=True, question="q")
    assert provenance.used_web is True
    assert provenance.learned is True


def test_answer_provenance_is_honest_when_the_model_cannot_be_named():
    provenance = answer_provenance(None, mouth="", question="q")
    assert provenance.model_label == ""
    assert provenance.model_line().endswith(UNKNOWN_MODEL["en"])


# --- the turn wiring ------------------------------------------------------


def _assistant():
    from core.assistant import GroundedAssistant
    from core.curriculum_knowledge import seed_curriculum
    from core.documents import build_document_store
    from core.domain_knowledge import seed_ontology
    from core.ontology import Ontology
    from core.rag import RagRetriever

    ontology = Ontology()
    seed_ontology(ontology)
    seed_curriculum(ontology)
    return GroundedAssistant(RagRetriever(build_document_store(), ontology=ontology))


def _mode(tmp_path):
    from core.command_router import NaturalLanguageRouter
    from core.jarvis import TEXT_HELP_TEXT, JarvisMode
    from core.speech import SilentSynthesizer, TranscriptRecognizer, VoiceAssistant
    from core.tool_agent import ToolAgent
    from core.tools import build_sandbox

    sandbox = build_sandbox(tmp_path)
    assistant = _assistant()
    return JarvisMode(
        assistant,
        router=NaturalLanguageRouter(sandbox),
        agent=ToolAgent(sandbox),
        voice=VoiceAssistant(
            assistant, recognizer=TranscriptRecognizer(), synthesizer=SilentSynthesizer()
        ),
        help_text=TEXT_HELP_TEXT,
    )


class _FakeMindModel:
    """A stand-in trained model that describes itself and echoes fixed prose."""

    def __init__(self, continuation: str = "the model's own words.") -> None:
        self.model_name = "fayd-local-llm:test"
        self._continuation = continuation

        class _Generator:
            def generate(self, prompt, *, max_new_tokens=64, temperature=0.8, seed=None):
                return continuation

        self.generator = _Generator()


def _switcher_mode(tmp_path):
    """A chat mode plus its switcher, over a fake model (no numpy needed)."""
    from ai import chat

    mode = _mode(tmp_path)
    switcher = chat.ModelSwitcher(
        mode, loader=lambda: _FakeMindModel(), max_new_tokens=16
    )
    return mode, switcher


def test_a_retrieval_answer_carries_its_model_and_sources(tmp_path, monkeypatch):
    mode, _switcher = _switcher_mode(tmp_path)
    turn = mode.handle("what is porosity?", require_wake=False, speak=False)

    assert turn.provenance
    assert turn.answer_model
    # The sources are the turn's own citations, so the line and the block agree.
    assert turn.citations


def test_a_control_turn_carries_no_provenance(tmp_path):
    mode, _switcher = _switcher_mode(tmp_path)
    turn = mode.handle("help", require_wake=False, speak=False)

    assert turn.routed == "control"
    assert turn.provenance == ""
    assert turn.answer_model == ""


def test_format_turn_renders_the_provenance_after_the_sources(tmp_path):
    from ai import chat

    mode, _switcher = _switcher_mode(tmp_path)
    turn = mode.handle("what is porosity?", require_wake=False, speak=False)
    text = chat.format_turn(turn)

    assert MODEL_LABEL["en"] + ":" in text
    # A reader sees the answer, then its sources, then what answered it.
    assert text.index(turn.answer_text.splitlines()[0]) < text.index("Sources:")
    assert text.index("Sources:") < text.index(MODEL_LABEL["en"])


def test_turn_payload_exposes_the_provenance(tmp_path):
    from ai import chat

    mode, _switcher = _switcher_mode(tmp_path)
    turn = mode.handle("what is porosity?", require_wake=False, speak=False)
    payload = chat.turn_payload(turn)

    assert payload["answer_model"] == turn.answer_model
    assert payload["provenance"] == turn.provenance


def test_the_provenance_names_the_selected_mouth(tmp_path):
    # A model-backed mouth must be named as itself, not as the retrieval default.
    from ai import chat

    mode, switcher = _switcher_mode(tmp_path)
    switcher.use_model()
    turn = mode.handle("what is porosity?", require_wake=False, speak=False)

    assert turn.provenance
    assert MODEL_LABEL["en"] in turn.provenance
    assert chat.turn_payload(turn)["answer_model"] == turn.answer_model


# --- the GUI service reply ------------------------------------------------


def test_chat_reply_carries_the_provenance(tmp_path):
    from gui.service import ChatService

    service = ChatService(data_root=tmp_path)
    reply = service.ask("what is porosity?")

    assert reply.answer_model
    assert reply.provenance
    assert MODEL_LABEL["en"] in reply.provenance


def test_chat_reply_provenance_is_empty_for_a_control(tmp_path):
    from gui.service import ChatService

    service = ChatService(data_root=tmp_path)
    reply = service.ask("help")

    assert reply.is_control
    assert reply.provenance == ""
    assert reply.answer_model == ""


def test_add_provenance_label_adds_no_widget_for_empty_text():
    if not _display_available():
        pytest.skip("no display available")
    import tkinter as tk

    root = tk.Tk()
    try:
        from gui.app import DARK_THEME
        from gui.message_footer import add_provenance_label
        from gui.typography import for_root

        assert add_provenance_label(root, "", theme=DARK_THEME, typography=for_root(root)) is None
        label = add_provenance_label(
            root, "Answer model: X.", theme=DARK_THEME, typography=for_root(root)
        )
        assert label is not None
        assert "Answer model" in label.cget("text")
    finally:
        root.destroy()


def test_message_footer_accepts_a_provenance_line():
    if not _display_available():
        pytest.skip("no display available")
    import tkinter as tk

    root = tk.Tk()
    try:
        from gui.app import DARK_THEME
        from gui.message_footer import MessageFooter
        from gui.typography import for_root

        widget = tk.Text(root)
        footer = MessageFooter(
            root, text="hello", widget=widget, theme=DARK_THEME,
            typography=for_root(root), on_toggle=lambda _f: None,
            provenance="Answer model: X.\nSources: Doc A",
        )
        assert footer.provenance is not None
        assert "Answer model" in footer.provenance.cget("text")
    finally:
        root.destroy()
