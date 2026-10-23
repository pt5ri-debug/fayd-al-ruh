"""Doc's Studio service: the Tk-free half of the document window.

This is where the Studio's work actually happens: open a document, hold it, ask
the local vision model about it, and report honestly when the model cannot
answer. The window only collects a file and a question and shows the result, so
all of this is testable with no display -- and, through the injected
``runner``/``client`` seam, with no model and no weights either.

The service never stands up a second AI: :meth:`DocsStudioService.set_ask` lets
the chat window hand it *its own* assistant, so a document can also be discussed
with the project's grounded assistant and its knowledge base. The vision model is
the Studio's own reading, and every answer carries
:data:`core.docs_model.HEADER` so it is never mistaken for a cited extract.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from core.docs_model import (
    DEFAULT_DOCS_MODEL,
    HEADER,
    DocsModelStatus,
    DocsModelUnavailable,
    LlamaCppDocsClient,
    docs_model_status,
    download_model,
    spec_names,
    spec_summaries,
)
from core.docs_studio import (
    MAX_PAGES_PER_REQUEST,
    SUPPORTED_SUFFIXES,
    DocsStudioError,
    LoadedDocument,
    build_vision_messages,
    load_document,
    select_pages,
)
from core.paths import docs_models_dir

#: The standing instruction given to the vision model. It is asked to answer from
#: the document only, to quote when it helps, and to say plainly when the answer
#: is not in the pages it was shown -- the honest failure mode.
SYSTEM_PROMPT = (
    "You are Doc's Studio, a careful document-reading assistant. You are shown a "
    "document's extracted text and images of its pages. Answer the user's question "
    "using only what the document contains. Quote the document's own words when "
    "they matter. If the answer is not in the pages you were given, say so plainly "
    "instead of guessing. Answer in the language of the question."
)

#: The placeholder used when the model produced nothing wordbearing, so an answer
#: is never a header with an empty body.
DOCS_EMPTY = "The model returned no text for this question."


@dataclass(frozen=True)
class DocAnswer:
    """One answer about a document, ready for the window or the terminal."""

    ok: bool
    question: str
    text: str
    filename: str = ""
    pages_used: tuple[int, ...] = ()
    message: str = ""
    header: str = HEADER

    def summary(self) -> str:
        """A one-block, human-readable report."""
        if not self.ok:
            return f"Could not answer: {self.message}"
        pages = ", ".join(str(number) for number in self.pages_used) or "—"
        return f"{self.header}\n{self.text}\n\n(pages read: {pages})"


@dataclass
class OpenDocument:
    """A document the service is holding open, plus what it knows about it."""

    document: LoadedDocument
    key: str = ""

    @property
    def filename(self) -> str:
        return self.document.filename

    @property
    def summary(self) -> str:
        return self.document.summary()


class DocsStudioService:
    """Open documents and answer questions about them with the local vision model."""

    def __init__(
        self,
        *,
        root: Path | str | None = None,
        model_root: Path | str | None = None,
        client: LlamaCppDocsClient | None = None,
        runner: Any = None,
        ask: Callable[[str], str] | None = None,
    ) -> None:
        self.root = Path(root) if root is not None else None
        self.model_root = model_root
        self._documents: dict[str, OpenDocument] = {}
        self._order: list[str] = []
        self._ask = ask
        self._client = client or LlamaCppDocsClient(
            root=model_root, runner=runner, model=DEFAULT_DOCS_MODEL
        )

    # --- documents ------------------------------------------------------
    def open_document(
        self,
        path: str | Path,
        *,
        key: str = "",
        render: bool = True,
        normalize: bool = False,
    ) -> OpenDocument:
        """Load ``path`` and remember it under ``key`` (defaults to its path)."""
        document = load_document(path, render=render, normalize=normalize)
        identifier = key or str(Path(path).expanduser().resolve())
        if identifier not in self._documents:
            self._order.append(identifier)
        opened = OpenDocument(document=document, key=identifier)
        self._documents[identifier] = opened
        return opened

    def documents(self) -> tuple[OpenDocument, ...]:
        """The open documents, in the order they were opened."""
        return tuple(self._documents[key] for key in self._order if key in self._documents)

    def get(self, key: str) -> OpenDocument | None:
        return self._documents.get(key)

    def close_document(self, key: str) -> bool:
        """Forget one document. Returns whether it was open."""
        if key not in self._documents:
            return False
        del self._documents[key]
        self._order = [item for item in self._order if item != key]
        return True

    def close_all(self) -> None:
        self._documents.clear()
        self._order.clear()

    # --- asking ---------------------------------------------------------
    def set_ask(self, ask: Callable[[str], str] | None) -> None:
        """Use the caller's own assistant for :meth:`ask_assistant`.

        The window hands its grounded assistant in, so a document can be discussed
        with the same knowledge base the chat uses -- never a second AI.
        """
        self._ask = ask

    def pages_for(self, key: str, question: str, *, limit: int = MAX_PAGES_PER_REQUEST) -> tuple[int, ...]:
        """The pages the service would attach for ``question`` (deterministic)."""
        opened = self._documents.get(key)
        if opened is None:
            raise DocsStudioError(f"No document is open under {key!r}.")
        return select_pages(opened.document, question, limit=limit)

    def ask(
        self,
        key: str,
        question: str,
        *,
        pages: tuple[int, ...] | None = None,
        max_pages: int = MAX_PAGES_PER_REQUEST,
    ) -> DocAnswer:
        """Answer ``question`` about the open document ``key`` with the vision model."""
        opened = self._documents.get(key)
        if opened is None:
            return DocAnswer(
                ok=False, question=question, text="", message=f"No document is open under {key!r}."
            )
        if not question.strip():
            return DocAnswer(
                ok=False, question=question, text="", filename=opened.filename,
                message="Ask a question about the document.",
            )
        document = opened.document
        chosen = pages if pages is not None else select_pages(document, question, limit=max_pages)
        messages = build_vision_messages(
            document, question, pages=chosen, system_prompt=SYSTEM_PROMPT, max_pages=max_pages
        )
        try:
            text = self._client.chat(messages)
        except DocsModelUnavailable as error:
            return DocAnswer(
                ok=False, question=question, text="", filename=opened.filename,
                pages_used=tuple(chosen), message=str(error),
            )
        text = (text or "").strip() or DOCS_EMPTY
        return DocAnswer(
            ok=True,
            question=question,
            text=text,
            filename=opened.filename,
            pages_used=tuple(chosen),
        )

    def ask_assistant(self, question: str) -> str:
        """Answer with the caller's own assistant, when one was supplied.

        This is the *grounded* path: the question goes to the project's assistant
        with its citations, so a document can also be discussed against the
        knowledge base. It is deliberately separate from :meth:`ask`.
        """
        if self._ask is None:
            raise DocsStudioError(
                "No assistant is available in this Studio. Open it from the chat "
                "window, or use the vision model instead."
            )
        return self._ask(question)

    # --- model ----------------------------------------------------------
    def model_status(self) -> DocsModelStatus:
        """An honest status of the local vision model."""
        return docs_model_status(root=self.model_root, client=self._client)

    def model_names(self) -> tuple[str, ...]:
        return spec_names()

    def model_summaries(self) -> tuple[tuple[str, str, float, str], ...]:
        return spec_summaries()

    def model_dir(self) -> Path:
        return docs_models_dir() if self.model_root is None else Path(self.model_root)

    def download_model(self, spec_key: str = "", *, approved: bool = False, progress: Any = None) -> tuple[Path, Path]:
        """Fetch a vision model's weights and projector (governed by the gates)."""
        return download_model(
            spec_key or DEFAULT_DOCS_MODEL,
            root=self.model_root,
            approved=approved,
            progress=progress,
        )

    # --- text -----------------------------------------------------------
    def document_text(self, key: str) -> str:
        opened = self._documents.get(key)
        if opened is None:
            raise DocsStudioError(f"No document is open under {key!r}.")
        return opened.document.text

    def save_text(self, key: str, target: str | Path) -> Path:
        """Write a document's extracted text to ``target`` (a UTF-8 .txt file)."""
        text = self.document_text(key)
        path = Path(target)
        try:
            path.write_text(text, encoding="utf-8")
        except OSError as error:
            raise DocsStudioError(f"Could not write {path}: {error}") from error
        return path


#: The formats the Studio accepts, for a file picker. Built from one list so the
#: picker and the loader can never disagree.
def picker_filetypes() -> tuple[tuple[str, str], ...]:
    """Tk ``filetypes`` tuples built from the supported list, no new dependency."""
    documents = " ".join(f"*{suffix}" for suffix in SUPPORTED_SUFFIXES)
    return (
        ("Documents", documents),
        ("Text", "*.txt *.md *.markdown"),
        ("Word", "*.docx *.docm"),
        ("PDF", "*.pdf"),
        ("Images", "*.png *.jpg *.jpeg *.webp *.bmp *.tif *.tiff"),
        ("Data", "*.csv *.tsv *.json *.jsonl *.ndjson"),
        ("All files", "*.*"),
    )


__all__ = [
    "DOCS_EMPTY",
    "DocAnswer",
    "DocsStudioService",
    "OpenDocument",
    "SYSTEM_PROMPT",
    "picker_filetypes",
]
