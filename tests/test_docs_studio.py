"""Tests for Doc's Studio: the document engine, the model layer, and the service.

The feature reads any document -- text, Word, PDF, a *scanned* PDF, or an image --
with a local Qwen3-VL vision model and answers questions about it. Every test here
runs with **no model, no extra, and no display**: the vision runner is injected
(the same seam ``core.docs_model`` exposes), so the whole Studio is exercised on a
fresh clone. The parts that genuinely need an optional dependency (a raster
renderer, ``pypdfium2``) skip honestly rather than pretending.
"""

from __future__ import annotations

import json
import urllib.parse
import urllib.request
from pathlib import Path

import pytest
from core import docs_model
from core.docs_model import (
    CATALOG,
    HEADER,
    DocsModelUnavailable,
    LlamaCppDocsClient,
    docs_model_status,
    find_mmproj_file,
    find_model_file,
    model_is_cached,
    resolve_spec,
    spec_names,
    spec_summaries,
)
from core.docs_studio import (
    SUPPORTED_SUFFIXES,
    UNSUPPORTED_SUFFIXES,
    DocsStudioError,
    DocumentPage,
    LoadedDocument,
    batch_pages,
    build_vision_messages,
    file_uri,
    load_document,
    parse_page_range,
    select_pages,
)
from gui.docs_studio import (
    DOCS_EMPTY,
    DocsStudioService,
    picker_filetypes,
)

# --- helpers ---------------------------------------------------------------


class RecordingRunner:
    """A deterministic stand-in for the vision model (no extra, no weights)."""

    def __init__(self, answer: str = "The document says so.") -> None:
        self.answer = answer
        self.calls: list[list[dict]] = []

    def chat(self, messages: list[dict], **kwargs) -> str:
        self.calls.append(messages)
        return self.answer


def write(path: Path, text: str, *, encoding: str = "utf-8") -> Path:
    path.write_text(text, encoding=encoding)
    return path


# --- the model layer -------------------------------------------------------


def test_catalog_has_the_requested_default_and_names_are_sorted() -> None:
    assert docs_model.DEFAULT_DOCS_MODEL == "qwen3-vl-4b"
    assert docs_model.DEFAULT_SPEC_KEY in CATALOG
    assert spec_names() == tuple(sorted(CATALOG))


def test_every_spec_names_both_halves_of_a_vision_model() -> None:
    # A vision model without its projector cannot see a page image, so every
    # catalog entry must carry a distinct mmproj filename.
    for spec in CATALOG.values():
        assert spec.filename.endswith(".gguf")
        assert spec.mmproj_filename.endswith(".gguf")
        assert spec.mmproj_filename.startswith(docs_model.MMPROJ_PREFIX)
        assert spec.filename != spec.mmproj_filename
        assert spec.repo
        assert spec.approx_size_gb > 0


def test_resolve_spec_defaults_and_rejects_unknown() -> None:
    assert resolve_spec(None).key == docs_model.DEFAULT_SPEC_KEY
    assert resolve_spec("").key == docs_model.DEFAULT_SPEC_KEY
    assert resolve_spec("QWEN3-VL-2B").key == "qwen3-vl-2b"
    with pytest.raises(DocsModelUnavailable):
        resolve_spec("nope")


def test_spec_summaries_are_four_field_rows_in_order() -> None:
    rows = spec_summaries()
    assert len(rows) == len(CATALOG)
    assert [row[0] for row in rows] == list(spec_names())
    for key, title, size, summary in rows:
        assert key and title and summary
        assert isinstance(size, float)


def test_find_model_file_excludes_the_projector(tmp_path: Path) -> None:
    (tmp_path / "mmproj-model-F16.gguf").write_bytes(b"x" * 100)
    (tmp_path / "language-model-Q4.gguf").write_bytes(b"x" * 50)
    # The largest .gguf is the projector, but the language model must still win:
    # the projector is excluded by name, never chosen as the model.
    model = find_model_file(tmp_path)
    projector = find_mmproj_file(tmp_path)
    assert model is not None and model.name == "language-model-Q4.gguf"
    assert projector is not None and projector.name == "mmproj-model-F16.gguf"


def test_find_model_file_prefers_the_registry(tmp_path: Path) -> None:
    (tmp_path / "mmproj-model-F16.gguf").write_bytes(b"x" * 100)
    (tmp_path / "small.gguf").write_bytes(b"x" * 10)
    (tmp_path / "big.gguf").write_bytes(b"x" * 500)
    (tmp_path / docs_model.REGISTRY_FILENAME).write_text(
        json.dumps({"file": "small.gguf", "mmproj": "mmproj-model-F16.gguf"}),
        encoding="utf-8",
    )
    model = find_model_file(tmp_path)
    assert model is not None and model.name == "small.gguf"


def test_model_is_cached_requires_both_halves(tmp_path: Path) -> None:
    assert model_is_cached(tmp_path) is False
    (tmp_path / "m.gguf").write_bytes(b"x")
    assert model_is_cached(tmp_path) is False, "a model without a projector is not ready"
    (tmp_path / "mmproj-m.gguf").write_bytes(b"x")
    assert model_is_cached(tmp_path) is True


def test_status_reports_a_missing_projector_honestly(tmp_path: Path) -> None:
    (tmp_path / "m.gguf").write_bytes(b"x")
    status = docs_model_status(root=tmp_path)
    assert status.available is False
    assert "mmproj" in status.detail
    assert status.hint


def test_status_reports_no_model_honestly(tmp_path: Path) -> None:
    status = docs_model_status(root=tmp_path)
    assert status.available is False
    assert status.installed is False
    assert status.detail


def test_status_is_ready_with_an_injected_runner(tmp_path: Path) -> None:
    client = LlamaCppDocsClient(root=tmp_path, runner=RecordingRunner())
    status = docs_model_status(root=tmp_path, client=client)
    assert status.available is True
    assert status.backend == "in-process"
    assert "ready" in status.summary().lower()


def test_client_missing_detail_names_the_real_next_step(tmp_path: Path) -> None:
    client = LlamaCppDocsClient(root=tmp_path)
    detail = client.missing_detail()
    assert "docs" in detail.lower()


def test_download_is_denied_by_default(tmp_path: Path) -> None:
    with pytest.raises(DocsModelUnavailable):
        docs_model.download_model("qwen3-vl-4b", root=tmp_path, approved=False)


def test_client_requires_a_projector_and_says_why(tmp_path: Path) -> None:
    (tmp_path / "m.gguf").write_bytes(b"x")
    client = LlamaCppDocsClient(root=tmp_path)
    with pytest.raises(DocsModelUnavailable) as error:
        client.mmproj_path()
    message = str(error.value).lower()
    assert "projector" in message or "mmproj" in message


def test_injected_client_chat_needs_no_model(tmp_path: Path) -> None:
    runner = RecordingRunner("Porosity is the void fraction.")
    client = LlamaCppDocsClient(root=tmp_path, runner=runner)
    assert client.chat([{"role": "user", "content": "hi"}]) == "Porosity is the void fraction."
    assert runner.calls


# --- the document engine ---------------------------------------------------


def test_supported_and_unsupported_lists_are_disjoint() -> None:
    assert not set(SUPPORTED_SUFFIXES) & set(UNSUPPORTED_SUFFIXES)
    assert ".pdf" in SUPPORTED_SUFFIXES
    assert ".docx" in SUPPORTED_SUFFIXES
    assert ".png" in SUPPORTED_SUFFIXES
    assert ".doc" in UNSUPPORTED_SUFFIXES


def test_load_text_document(tmp_path: Path) -> None:
    path = write(tmp_path / "notes.md", "# Notes\n\nPorosity controls storage.\n")
    document = load_document(path)
    assert document.filename == "notes.md"
    assert document.page_count == 1
    assert "Porosity controls storage." in document.text
    assert document.word_count >= 4
    assert document.vision_pages == ()


def test_load_plain_data_file_is_decoded_strictly(tmp_path: Path) -> None:
    path = write(tmp_path / "rows.csv", "a,b\n1,2\n")
    document = load_document(path)
    assert "a,b" in document.text
    assert document.kind == "csv"


def test_load_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(DocsStudioError):
        load_document(tmp_path / "nope.txt")


def test_load_legacy_word_file_names_the_next_step(tmp_path: Path) -> None:
    legacy = write(tmp_path / "old.doc", "x")
    with pytest.raises(DocsStudioError) as error:
        load_document(legacy)
    assert ".docx" in str(error.value)


def test_load_rejects_an_unknown_suffix(tmp_path: Path) -> None:
    path = write(tmp_path / "thing.xyz", "x")
    with pytest.raises(DocsStudioError):
        load_document(path)


def test_load_plain_image_is_one_vision_page(tmp_path: Path) -> None:
    path = tmp_path / "shot.png"
    path.write_bytes(b"\x89PNG\r\n\x1a\nnot-really-a-png")
    document = load_document(path)
    assert document.kind == "image"
    assert document.scanned is True
    assert document.vision_pages == (1,)
    assert document.pages[0].from_image is True


def test_select_pages_prefers_named_pages_and_is_bounded() -> None:
    document = LoadedDocument(
        path="x", filename="x.pdf", kind="pdf",
        pages=[
            DocumentPage(number=1, text="the casing design"),
            DocumentPage(number=2, text="the cementing programme"),
            DocumentPage(number=3, text=""),
        ],
    )
    assert select_pages(document, "cementing")[0] == 2
    assert len(select_pages(document, "anything", limit=2)) == 2


def test_select_pages_never_empty_for_a_non_empty_document() -> None:
    document = LoadedDocument(
        path="x", filename="x.pdf", kind="pdf", pages=[DocumentPage(number=1, text="hi")]
    )
    assert select_pages(document, "") == (1,)


def test_build_vision_messages_carries_text_and_images(tmp_path: Path) -> None:
    image = tmp_path / "page.png"
    image.write_bytes(b"\x89PNG")
    document = LoadedDocument(
        path="x", filename="scan.pdf", kind="pdf",
        pages=[DocumentPage(number=1, image_path=str(image), from_image=True)],
    )
    messages = build_vision_messages(document, "what is this?", pages=(1,))
    assert messages[-1]["role"] == "user"
    parts = messages[-1]["content"]
    kinds = [part["type"] for part in parts]
    assert "image_url" in kinds
    urls = [part["image_url"]["url"] for part in parts if part["type"] == "image_url"]
    assert urls and all(url.startswith("file://") for url in urls)
    # The URL must be one urllib can actually open on every OS -- the bug was a
    # Windows drive path landing in the URI authority, so ``urlopen`` looked for
    # the empty path and failed with WinError 3.
    assert urllib.request.urlopen(urls[0]).read() == b"\x89PNG"


def test_file_uri_keeps_the_path_out_of_the_authority(tmp_path: Path) -> None:
    image = tmp_path / "page 1.png"  # a space must be percent-encoded
    image.write_bytes(b"\x89PNG")
    uri = file_uri(image)
    split = urllib.parse.urlsplit(uri)
    assert split.scheme == "file"
    assert split.netloc == ""  # never the path (the Windows failure)
    assert split.path.startswith("/") and "page%201.png" in split.path
    assert urllib.request.urlopen(uri).read() == b"\x89PNG"


def test_file_uri_is_the_path_as_uri(tmp_path: Path) -> None:
    image = tmp_path / "page.png"
    image.write_bytes(b"x")
    assert file_uri(image) == image.resolve().as_uri()


def test_build_vision_messages_states_when_there_is_no_text_layer() -> None:
    document = LoadedDocument(
        path="x", filename="scan.pdf", kind="pdf",
        pages=[DocumentPage(number=1, image_path="/tmp/none.png", from_image=True)],
    )
    messages = build_vision_messages(document, "summarise", pages=(1,))
    assert "no text layer" in messages[-1]["content"][0]["text"]


def test_build_vision_messages_includes_a_system_prompt_when_given() -> None:
    document = LoadedDocument(
        path="x", filename="a.txt", kind="txt", pages=[DocumentPage(number=1, text="hi")]
    )
    messages = build_vision_messages(document, "q", system_prompt="Be careful.")
    assert messages[0]["role"] == "system"
    assert messages[-1]["role"] == "user"


def test_build_vision_messages_caps_the_pages_it_attaches(tmp_path: Path) -> None:
    pages = []
    for number in range(1, 6):
        image = tmp_path / f"p{number}.png"
        image.write_bytes(b"\x89PNG")
        pages.append(DocumentPage(number=number, image_path=str(image), from_image=True))
    document = LoadedDocument(path="x", filename="s.pdf", kind="pdf", pages=pages)
    messages = build_vision_messages(document, "q", pages=(1, 2, 3, 4, 5), max_pages=2)
    kinds = [part["type"] for part in messages[-1]["content"]]
    assert kinds.count("image_url") == 2


def test_parse_page_range_accepts_the_forms_users_type() -> None:
    assert parse_page_range("7") == (7,)
    assert parse_page_range("1-3") == (1, 2, 3)
    assert parse_page_range("3-1") == (1, 2, 3)
    assert parse_page_range("1 to 3") == (1, 2, 3)
    assert parse_page_range("2,5,7") == (2, 5, 7)
    assert parse_page_range("2, 2, 5") == (2, 5)
    assert parse_page_range("") == ()


def test_parse_page_range_rejects_nonsense() -> None:
    for bad in ("zero", "0", "1-x"):
        with pytest.raises(DocsStudioError):
            parse_page_range(bad)


def test_document_page_is_one_based() -> None:
    with pytest.raises(ValueError):
        DocumentPage(number=0)


def test_loaded_document_summary_is_honest_about_scanned_pages() -> None:
    document = LoadedDocument(
        path="x", filename="scan.pdf", kind="pdf", scanned=True,
        pages=[DocumentPage(number=1, image_path="/tmp/a.png", from_image=True)],
    )
    assert "scanned" in document.summary()


# --- the service -----------------------------------------------------------


def test_service_opens_and_answers_with_the_injected_runner(tmp_path: Path) -> None:
    runner = RecordingRunner("Porosity is the void fraction.")
    service = DocsStudioService(runner=runner, model_root=tmp_path)
    path = write(tmp_path / "notes.md", "Porosity is the void fraction of a rock.")
    opened = service.open_document(path)
    answer = service.ask(opened.key, "What is porosity?")
    assert answer.ok is True
    assert answer.text == "Porosity is the void fraction."
    assert answer.pages_used
    assert HEADER in answer.summary()


def test_service_reports_a_missing_model_honestly(tmp_path: Path) -> None:
    service = DocsStudioService(model_root=tmp_path)  # no runner, no extra
    path = write(tmp_path / "notes.md", "hello")
    opened = service.open_document(path)
    answer = service.ask(opened.key, "what is this?")
    assert answer.ok is False
    assert answer.message


def test_service_refuses_an_empty_question(tmp_path: Path) -> None:
    service = DocsStudioService(runner=RecordingRunner(), model_root=tmp_path)
    path = write(tmp_path / "n.md", "x")
    opened = service.open_document(path)
    assert service.ask(opened.key, "   ").ok is False


def test_service_refuses_an_unknown_document_key(tmp_path: Path) -> None:
    service = DocsStudioService(runner=RecordingRunner(), model_root=tmp_path)
    assert service.ask("nope", "q").ok is False


def test_service_empty_model_reply_becomes_a_named_message(tmp_path: Path) -> None:
    service = DocsStudioService(runner=RecordingRunner("   "), model_root=tmp_path)
    path = write(tmp_path / "n.md", "x")
    opened = service.open_document(path)
    answer = service.ask(opened.key, "q")
    assert answer.ok is True
    assert answer.text == DOCS_EMPTY


def test_service_tracks_and_closes_documents(tmp_path: Path) -> None:
    service = DocsStudioService(runner=RecordingRunner(), model_root=tmp_path)
    first = service.open_document(write(tmp_path / "a.md", "a"))
    service.open_document(write(tmp_path / "b.md", "b"))
    assert [doc.filename for doc in service.documents()] == ["a.md", "b.md"]
    assert service.close_document(first.key) is True
    assert service.close_document(first.key) is False
    assert [doc.filename for doc in service.documents()] == ["b.md"]
    service.close_all()
    assert service.documents() == ()


def test_service_pages_for_is_deterministic(tmp_path: Path) -> None:
    service = DocsStudioService(runner=RecordingRunner(), model_root=tmp_path)
    opened = service.open_document(write(tmp_path / "n.md", "porosity and permeability"))
    assert service.pages_for(opened.key, "porosity") == (1,)
    with pytest.raises(DocsStudioError):
        service.pages_for("nope", "q")


def test_service_uses_the_callers_own_assistant(tmp_path: Path) -> None:
    asked: list[str] = []

    def ask(question: str) -> str:
        asked.append(question)
        return "A grounded answer."

    service = DocsStudioService(runner=RecordingRunner(), model_root=tmp_path)
    service.set_ask(ask)
    assert service.ask_assistant("what is porosity?") == "A grounded answer."
    assert asked == ["what is porosity?"]


def test_service_says_so_when_no_assistant_was_supplied(tmp_path: Path) -> None:
    service = DocsStudioService(runner=RecordingRunner(), model_root=tmp_path)
    with pytest.raises(DocsStudioError):
        service.ask_assistant("q")


def test_service_saves_the_extracted_text(tmp_path: Path) -> None:
    service = DocsStudioService(runner=RecordingRunner(), model_root=tmp_path)
    opened = service.open_document(write(tmp_path / "n.md", "the casing design"))
    target = service.save_text(opened.key, tmp_path / "out.txt")
    assert "the casing design" in target.read_text(encoding="utf-8")


def test_service_model_status_and_names(tmp_path: Path) -> None:
    service = DocsStudioService(runner=RecordingRunner(), model_root=tmp_path)
    assert service.model_status().available is True
    assert service.model_names() == spec_names()
    assert service.model_summaries() == spec_summaries()


def test_service_download_is_governed(tmp_path: Path) -> None:
    service = DocsStudioService(runner=RecordingRunner(), model_root=tmp_path)
    with pytest.raises(DocsModelUnavailable):
        service.download_model("qwen3-vl-4b", approved=False)


def test_picker_filetypes_lists_the_supported_formats() -> None:
    rows = picker_filetypes()
    labels = [label for label, _pattern in rows]
    assert "PDF" in labels
    assert "Word" in labels
    assert "Images" in labels
    assert rows[-1][1] == "*.*"


def test_picker_patterns_cover_every_supported_suffix() -> None:
    patterns = " ".join(pattern for _label, pattern in picker_filetypes())
    for suffix in SUPPORTED_SUFFIXES:
        assert f"*{suffix}" in patterns


# --- reading a scanned document's contents --------------------------------


def test_batch_pages_bounds_and_keeps_order() -> None:
    assert batch_pages([1, 2, 3, 4, 5], limit=2) == ((1, 2), (3, 4), (5,))
    assert batch_pages([], limit=2) == ()
    with pytest.raises(ValueError):
        batch_pages([1], limit=0)


def test_service_read_says_a_text_document_needs_no_reading(tmp_path: Path) -> None:
    service = DocsStudioService(runner=RecordingRunner(), model_root=tmp_path)
    opened = service.open_document(write(tmp_path / "n.md", "the casing design"))
    reading = service.read(opened.key)
    assert reading.ok is True
    assert reading.text == ""
    assert reading.pages_read == ()
    assert "text layer" in reading.message


def test_service_read_asks_the_model_to_transcribe(tmp_path: Path) -> None:
    # A plain image is a one-page vision document, so `read` must send the
    # transcribe instruction -- not an answer-a-question prompt.
    from core.docs_studio import TRANSCRIBE_QUESTION

    image = tmp_path / "shot.png"
    image.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 64)
    runner = RecordingRunner(answer="POROSITY REPORT")
    service = DocsStudioService(runner=runner, model_root=tmp_path)
    opened = service.open_document(str(image))
    reading = service.read(opened.key)
    assert reading.ok is True
    assert reading.pages_read == (1,)
    assert "POROSITY REPORT" in reading.text
    parts = runner.calls[-1][-1]["content"]
    text = " ".join(part.get("text", "") for part in parts if part["type"] == "text")
    assert TRANSCRIBE_QUESTION in text
    assert any(part["type"] == "image_url" for part in parts)


def test_service_read_reports_a_missing_model_honestly(tmp_path: Path) -> None:
    image = tmp_path / "shot.png"
    image.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 64)
    service = DocsStudioService(model_root=tmp_path)  # no runner -> no model
    opened = service.open_document(str(image))
    reading = service.read(opened.key)
    assert reading.ok is False
    assert "docs extra" in reading.message or "model" in reading.message.lower()


def test_service_read_records_the_reading_for_saving(tmp_path: Path) -> None:
    image = tmp_path / "shot.png"
    image.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 64)
    service = DocsStudioService(
        runner=RecordingRunner(answer="the casing is set at 2000 ft"), model_root=tmp_path
    )
    opened = service.open_document(str(image))
    assert service.document_text(opened.key) == ""  # nothing readable yet
    service.read(opened.key)
    held = service.get(opened.key)
    assert held is not None and held.reading is not None
    target = service.save_text(opened.key, tmp_path / "out.txt")
    assert "the casing is set at 2000 ft" in target.read_text(encoding="utf-8")


def test_read_failure_names_the_real_reason_not_the_ocr_red_herring(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A scanned PDF whose pages cannot be rendered must name *that* reason.

    Reporting the generic "no extractable text ... needs OCR" instead is exactly
    what made a scanned file look like an empty document with no explanation.
    """
    pytest.importorskip("pypdf", reason="pypdf (the [pdf] extra) is not installed")
    from core import docs_studio as engine

    from tests.pdf_fixtures import image_page, write_pdf

    def refuse(*_args, **_kwargs):
        raise engine.DocsStudioError("Rendering a scanned page needs the docs extra.")

    monkeypatch.setattr(engine, "render_page_image", refuse)
    path = write_pdf(tmp_path / "scan.pdf", [image_page()])
    with pytest.raises(DocsStudioError) as error:
        load_document(path, render=True)
    message = str(error.value)
    assert "could not be turned into images" in message
    assert "docs extra" in message


# --- a real PDF (only when the [pdf] extra is present) ---------------------


def test_digital_pdf_is_read_page_by_page(tmp_path: Path) -> None:
    pytest.importorskip("pypdf", reason="pypdf (the [pdf] extra) is not installed")
    from tests.pdf_fixtures import latin_page, write_pdf

    path = write_pdf(
        tmp_path / "report.pdf",
        [latin_page("Porosity is the void fraction."), latin_page("Permeability measures flow.")],
    )
    document = load_document(path, render=False)
    assert document.kind == "pdf"
    assert document.page_count == 2
    assert "Porosity" in document.text
    assert document.scanned is False
    assert document.vision_pages == ()


def test_scanned_pdf_needs_rendering_and_says_so_without_it(tmp_path: Path) -> None:
    pytest.importorskip("pypdf", reason="pypdf (the [pdf] extra) is not installed")
    from tests.pdf_fixtures import image_page, write_pdf

    path = write_pdf(tmp_path / "scan.pdf", [image_page(), image_page()])
    # Without rendering there is no text layer at all, so the honest next step is
    # named rather than an empty document being reported as success.
    with pytest.raises(DocsStudioError) as error:
        load_document(path, render=False)
    assert "scanned" in str(error.value).lower()


def test_scanned_pdf_pages_are_rendered_for_the_vision_model(tmp_path: Path) -> None:
    pytest.importorskip("pypdf", reason="pypdf (the [pdf] extra) is not installed")
    pytest.importorskip("pypdfium2", reason="pypdfium2 (the [docs] extra) is not installed")
    from tests.pdf_fixtures import image_page, write_pdf

    path = write_pdf(tmp_path / "scan.pdf", [image_page(), image_page()])
    document = load_document(path, render=True)
    assert document.scanned is True
    assert document.vision_pages == (1, 2)
    for page in document.pages:
        assert page.image_path
        assert Path(page.image_path).is_file()
    messages = build_vision_messages(document, "what does this say?")
    kinds = [part["type"] for part in messages[-1]["content"]]
    assert kinds.count("image_url") == 2
