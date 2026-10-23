"""The document engine behind Doc's Studio: load a document, then read it.

Doc's Studio has to turn *any* document a user has into something the local
vision model can read, and then let the user ask about it. The two halves are
kept apart on purpose:

* **Extraction** (:func:`load_document`) is deterministic and dependency-light.
  Text, Markdown, CSV/TSV, JSON/JSONL, Word ``.docx``/``.docm``, and digital PDFs
  are read with the project's existing readers (:mod:`core.document_reader`,
  :mod:`core.pdf_extract`). Nothing is guessed and nothing is executed.

* **Rendering** (``render_page_image``) is what makes a *scanned* document
  readable. A scanned PDF page has no text layer, so the page is rasterised to a
  PNG with ``pypdfium2`` and handed to the vision model as an image. The same
  path renders a plain image file. The project's raster guards
  (:func:`core.pdf_ocr.check_raster_limits`, :data:`core.pdf_ocr.OcrConfig`) are
  reused, so a pathological page is refused *before* allocation.

The engine also builds the multimodal message (:func:`build_vision_messages`) --
the text prompt plus the page images -- in the one shape ``llama-cpp-python``'s
``MTMDChatHandler`` expects, so :mod:`core.docs_model` stays a thin loader.

Design rules, matching the rest of the project:

* **No silent substitution.** A page that has real text is never replaced by an
  image; an image page is *added* for the model, and every page records whether
  its text came from the text layer, from an image, or from neither.
* **Bounded.** Only the pages a question actually needs are attached, up to a
  documented maximum, because image tokens are expensive and a 4B model on a CPU
  is slow. A document larger than the cap is reported as such, never truncated
  silently.
* **Nothing is executed.** A reader only decodes bytes.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

from core.document_reader import DocumentReadError, DocumentText, read_document
from core.paths import docs_pages_dir
from core.pdf_ocr import OcrConfig, check_raster_limits
from core.text_normalize import normalize_text

#: Text formats read with the project's own readers.
TEXT_SUFFIXES: tuple[str, ...] = (".txt", ".md", ".markdown", ".csv", ".tsv", ".json", ".jsonl", ".ndjson")

#: Word formats.
WORD_SUFFIXES: tuple[str, ...] = (".docx", ".docm")

#: Formats the reader turns into page images directly.
IMAGE_SUFFIXES: tuple[str, ...] = (".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff")

#: Everything the Studio accepts.
SUPPORTED_SUFFIXES: tuple[str, ...] = TEXT_SUFFIXES + WORD_SUFFIXES + (".pdf",) + IMAGE_SUFFIXES

#: Recognised but deliberately unsupported, refused with a clear message.
UNSUPPORTED_SUFFIXES: tuple[str, ...] = (".doc", ".rtf")

#: The default resolution for rasterising a PDF page.
DEFAULT_DPI = 150

#: The most pages of one document handed to the model in a single question. Image
#: tokens are expensive and a CPU vision model is slow, so a whole 300-page report
#: is never attached at once -- the engine picks the pages the question names.
MAX_PAGES_PER_REQUEST = 8

#: The most pages the engine will rasterise for one document. Beyond this the
#: document is still read, but only the first pages are available as images.
MAX_RENDERED_PAGES = 200

#: A character cap on the extracted text kept per document, mirroring the reader.
MAX_TEXT_CHARS = 2_000_000

#: Below this many characters a PDF page is treated as scanned (no usable text),
#: matching :data:`core.pdf_ocr.OcrConfig.min_chars_per_page`.
MIN_CHARS_PER_PAGE = 16


class DocsStudioError(ValueError):
    """A document could not be loaded, rendered, or read."""


@dataclass(frozen=True)
class DocumentPage:
    """One page of a document: its text, and the image that shows it, if any.

    ``text`` is whatever the text layer yielded (possibly empty for a scanned
    page). ``image_path`` is the rendered PNG for a scanned or image page, or
    ``None`` for a text page that needs no image. ``from_image`` records that the
    page's *only* usable content is its picture, so a caller can label it.
    """

    number: int
    text: str = ""
    image_path: str = ""
    from_image: bool = False

    def __post_init__(self) -> None:
        if self.number < 1:
            raise ValueError("DocumentPage number is 1-based and must be >= 1.")

    @property
    def has_text(self) -> bool:
        return bool(self.text.strip())

    @property
    def needs_vision(self) -> bool:
        """True when this page can only be understood by looking at it."""
        return self.from_image or not self.has_text


@dataclass
class LoadedDocument:
    """A document the Studio has loaded: its pages, its text, and its provenance."""

    path: str
    filename: str
    kind: str
    pages: list[DocumentPage] = field(default_factory=list)
    title: str = ""
    encoding: str = "utf-8"
    truncated: bool = False
    scanned: bool = False
    """True when at least one page had to be read as an image (no text layer)."""

    notes: list[str] = field(default_factory=list)

    @property
    def text(self) -> str:
        """The whole extracted text, page by page, blank pages omitted."""
        return "\n\n".join(page.text for page in self.pages if page.has_text)

    @property
    def page_count(self) -> int:
        return len(self.pages)

    @property
    def vision_pages(self) -> tuple[int, ...]:
        """The 1-based numbers of the pages that need the vision model."""
        return tuple(page.number for page in self.pages if page.needs_vision)

    @property
    def word_count(self) -> int:
        return len(self.text.split())

    def page(self, number: int) -> DocumentPage | None:
        for page in self.pages:
            if page.number == number:
                return page
        return None

    def summary(self) -> str:
        """A short, honest description of what was loaded."""
        bits = [f"{self.page_count} page(s)"]
        if self.text.strip():
            bits.append(f"{self.word_count} words")
        if self.scanned:
            bits.append("some pages read as images (scanned)")
        if self.truncated:
            bits.append("text capped")
        return f"{self.filename} — {', '.join(bits)}"


def _suffix(path: Path) -> str:
    return path.suffix.lower()


def _page_dir(path: Path) -> Path:
    """A stable, per-document folder for its rendered page images.

    Keyed by the resolved path and the file's size and mtime, so re-opening an
    unchanged document reuses its renders and a changed one gets fresh ones --
    without hashing a multi-megabyte file on every open.
    """
    try:
        stat = path.stat()
        stamp = f"{stat.st_size}:{int(stat.st_mtime)}"
    except OSError:
        stamp = "0:0"
    digest = hashlib.blake2b(f"{path.resolve()}:{stamp}".encode(), digest_size=8).hexdigest()
    return docs_pages_dir() / digest


def render_page_image(
    path: Path,
    number: int,
    *,
    out_dir: Path,
    dpi: int = DEFAULT_DPI,
    config: OcrConfig | None = None,
) -> Path:
    """Rasterise PDF ``path``'s page ``number`` (1-based) to a PNG and return it.

    The raster is bounded by the project's own guards: the page's pixel size is
    checked *before* the bitmap is allocated, and the finished PNG is written only
    after it has been serialised in memory, so an over-large page raises instead of
    filling memory or disk. ``pypdfium2`` renders in memory, so no external
    poppler binary is needed.
    """
    settings = config or OcrConfig(enabled=True, dpi=dpi)
    try:
        import pypdfium2  # type: ignore[import-not-found]
    except ImportError as error:
        raise DocsStudioError(
            "Rendering a scanned page needs the optional docs extra. Install it "
            "with: python -m pip install -r requirements-docs.txt"
        ) from error

    out_dir.mkdir(parents=True, exist_ok=True)
    target = out_dir / f"page-{number:04d}.png"
    if target.is_file():
        return target

    scale = settings.dpi / 72.0
    document = pypdfium2.PdfDocument(str(path))
    try:
        if number < 1 or number > len(document):
            raise DocsStudioError(
                f"Page {number} is out of range (the document has {len(document)} pages)."
            )
        page = document[number - 1]
        width_pt, height_pt = page.get_size()
        width = int(width_pt * scale)
        height = int(height_pt * scale)
        try:
            check_raster_limits(width, height, settings, number)
        except Exception as error:  # OcrLimitError - surfaced as an honest message
            raise DocsStudioError(f"Page {number} is too large to render: {error}") from error
        bitmap = page.render(scale=scale)
        image = bitmap.to_pil()
        try:
            image.save(target, format="PNG")
        finally:
            image.close()
    finally:
        document.close()
    return target


def _copy_image(path: Path, out_dir: Path) -> Path:
    """Register a plain image file as a page image, without re-encoding it."""
    out_dir.mkdir(parents=True, exist_ok=True)
    target = out_dir / f"page-0001{path.suffix.lower()}"
    if not target.is_file():
        try:
            target.write_bytes(path.read_bytes())
        except OSError as error:
            raise DocsStudioError(f"Could not read the image {path}: {error}") from error
    return target


def _load_pdf(path: Path, *, render: bool, max_pages: int, dpi: int) -> LoadedDocument:
    """Read a PDF: native text per page, plus an image for the pages that need one.

    A *digital* PDF yields its text page by page. A page without a usable text
    layer is rendered to an image. A wholly **scanned** PDF is the case that
    matters most: ``PdfPageStream`` yields every (empty-text) page and only then
    raises "no extractable text", so the pages are already in hand and each one is
    rendered -- which is exactly how a scanned report becomes readable.
    """
    from core.pdf_extract import PdfExtractionError, PdfPageStream

    out_dir = _page_dir(path)
    pages: list[DocumentPage] = []
    notes: list[str] = []
    scanned = False
    failure: PdfExtractionError | None = None
    try:
        for native in PdfPageStream(path):
            text = native.text or ""
            needs_image = len(text.strip()) < MIN_CHARS_PER_PAGE
            image_path = ""
            if needs_image:
                scanned = True
                if render and native.number <= max_pages:
                    try:
                        image_path = str(
                            render_page_image(path, native.number, out_dir=out_dir, dpi=dpi)
                        )
                    except DocsStudioError as error:
                        # A page that cannot be rendered keeps its (thin) text and the
                        # reason is recorded, never hidden and never fatal.
                        notes.append(f"page {native.number}: {error}")
            pages.append(
                DocumentPage(
                    number=native.number,
                    text=text,
                    image_path=image_path,
                    from_image=bool(image_path),
                )
            )
    except PdfExtractionError as error:
        # Raised after every page was yielded, only when *no* page had any text:
        # a scanned (image-only) PDF, or a blank one. Keep the pages we collected.
        failure = error

    if failure is not None and not any(page.has_text for page in pages):
        if not render:
            raise DocsStudioError(
                f"{path.name!r} is a scanned PDF with no text layer. Render its pages "
                "(drop --no-render) so the vision model can read them."
            ) from failure
        if not any(page.image_path for page in pages):
            # Rendering was attempted for every page and none succeeded, so this is
            # an honest failure rather than an empty document.
            raise DocsStudioError(str(failure)) from failure
    elif failure is not None:
        raise DocsStudioError(str(failure)) from failure

    return LoadedDocument(
        path=str(path),
        filename=path.name,
        kind="pdf",
        pages=pages,
        title=path.stem,
        truncated=len(pages) >= max_pages,
        scanned=scanned,
        notes=notes,
    )


def _load_image(path: Path) -> LoadedDocument:
    """A plain image is a one-page document whose content is the picture itself."""
    out_dir = _page_dir(path)
    image = _copy_image(path, out_dir)
    return LoadedDocument(
        path=str(path),
        filename=path.name,
        kind="image",
        pages=[DocumentPage(number=1, image_path=str(image), from_image=True)],
        title=path.stem,
        scanned=True,
    )


def _load_text_document(path: Path) -> LoadedDocument:
    """Read a text/Word document through the project's own reader."""
    try:
        document: DocumentText = read_document(path)
    except DocumentReadError as error:
        raise DocsStudioError(str(error)) from error
    return LoadedDocument(
        path=str(path),
        filename=path.name,
        kind=document.kind,
        pages=[DocumentPage(number=1, text=document.text)],
        title=document.title or path.stem,
        encoding=document.encoding,
        truncated=document.truncated,
        scanned=document.scanned,
    )


def _load_plain_text(path: Path) -> LoadedDocument:
    """Read a tabular/structured text file (CSV, TSV, JSON, JSONL) as one page.

    The project's document reader handles the *documents* (text, Markdown, Word);
    a data file is read here with the same strict, explainable decoding
    (:func:`core.text_encoding.decode_bytes`), so a mis-encoded file raises rather
    than filling with replacement characters. It is kept as one page because its
    value is the whole table, not a page split.
    """
    from core.text_encoding import EncodingDetectionError, decode_bytes

    try:
        raw = path.read_bytes()
    except OSError as error:
        raise DocsStudioError(f"Could not read {path}: {error}") from error
    try:
        decoded = decode_bytes(raw)
    except EncodingDetectionError as error:
        raise DocsStudioError(f"Could not decode {path}: {error}") from error
    text = decoded.text[:MAX_TEXT_CHARS]
    return LoadedDocument(
        path=str(path),
        filename=path.name,
        kind=path.suffix.lower().lstrip(".") or "text",
        pages=[DocumentPage(number=1, text=text)],
        title=path.stem,
        encoding=decoded.encoding,
        truncated=len(decoded.text) > MAX_TEXT_CHARS,
    )


def load_document(
    path: str | Path,
    *,
    render: bool = True,
    max_pages: int = MAX_RENDERED_PAGES,
    dpi: int = DEFAULT_DPI,
    normalize: bool = False,
) -> LoadedDocument:
    """Load ``path`` into a :class:`LoadedDocument`, or raise :class:`DocsStudioError`.

    Text, Markdown, CSV/TSV, JSON/JSONL, and Word files are read with the
    project's readers. A digital PDF is read page by page, and any page without a
    usable text layer is *rendered* to an image (when ``render``) so the vision
    model can read it. A plain image becomes a one-page document.

    ``render=False`` reads only the text layer -- useful for a fast preview on a
    machine that has no raster dependency. ``normalize=True`` applies the
    project's text normalisation (off by default, because it changes the text).
    """
    source = Path(path).expanduser()
    if not source.exists():
        raise DocsStudioError(f"No such file: {source}")
    if not source.is_file():
        raise DocsStudioError(f"Not a file: {source}")
    suffix = _suffix(source)
    if suffix in UNSUPPORTED_SUFFIXES:
        raise DocsStudioError(
            f"{source.suffix} is an older format this Studio cannot read. Save it as "
            ".docx (Word: File > Save As > Word Document) and try again."
        )
    if suffix not in SUPPORTED_SUFFIXES:
        raise DocsStudioError(
            f"Unsupported file type {source.suffix or '(none)'}. Supported: "
            f"{', '.join(sorted(SUPPORTED_SUFFIXES))}."
        )

    if suffix == ".pdf":
        document = _load_pdf(source, render=render, max_pages=max_pages, dpi=dpi)
    elif suffix in IMAGE_SUFFIXES:
        document = _load_image(source)
    elif suffix in {".csv", ".tsv", ".json", ".jsonl", ".ndjson"}:
        document = _load_plain_text(source)
    else:
        document = _load_text_document(source)

    if normalize:
        document.pages = [
            DocumentPage(
                number=page.number,
                text=normalize_text(page.text) if page.text else "",
                image_path=page.image_path,
                from_image=page.from_image,
            )
            for page in document.pages
        ]
    return document


# ------------------------------------------------------------------- selection


def _keywords(question: str) -> set[str]:
    """The wordbearing terms of a question, lower-cased and de-punctuated."""
    cleaned = "".join(ch if ch.isalnum() or ch.isspace() else " " for ch in question.lower())
    return {word for word in cleaned.split() if len(word) > 2}


def select_pages(
    document: LoadedDocument,
    question: str,
    *,
    limit: int = MAX_PAGES_PER_REQUEST,
) -> tuple[int, ...]:
    """The pages worth attaching for ``question``, deterministically chosen.

    A page that names the question's words comes first; a page that needs vision
    (no text layer) outranks a text-only page at the same score, because a text
    page contributes its text anyway; ties keep document order. When nothing
    matches -- or the document is short -- the first pages are used. The result is
    always bounded by ``limit``, never empty for a non-empty document, and ordered
    most-relevant first.
    """
    if not document.pages:
        return ()
    terms = _keywords(question)
    scored: list[tuple[int, int, int]] = []
    for index, page in enumerate(document.pages):
        haystack = page.text.lower()
        score = sum(1 for term in terms if term in haystack)
        vision_rank = 0 if page.needs_vision else 1
        scored.append((-score, vision_rank, index))
    scored.sort()
    chosen = [index for _score, _rank, index in scored[: max(1, limit)]]
    return tuple(document.pages[index].number for index in chosen)


def parse_page_range(text: str) -> tuple[int, ...]:
    """Parse a plain page selection into 1-based page numbers.

    Accepts the forms a user actually types: ``"7"``, ``"1-3"``, ``"1 to 3"``,
    and a comma-separated list ``"2,5,7"`` (ranges and single numbers may be
    mixed). A reversed range is normalised, duplicates are dropped, and the result
    keeps the order the numbers appear in. An empty string yields ``()``; anything
    that is not a page number raises :class:`DocsStudioError`.
    """
    cleaned = (text or "").strip()
    if not cleaned:
        return ()
    pages: list[int] = []
    for chunk in cleaned.replace("–", "-").split(","):
        part = chunk.strip()
        if not part:
            continue
        lowered = part.lower()
        for separator in (" to ", "-", ".."):
            if separator in lowered:
                start_text, _, end_text = lowered.partition(separator)
                try:
                    start, end = int(start_text.strip()), int(end_text.strip())
                except ValueError as error:
                    raise DocsStudioError(f"Not a page range: {part!r}.") from error
                if start < 1 or end < 1:
                    raise DocsStudioError("Page numbers are 1-based.")
                low, high = min(start, end), max(start, end)
                pages.extend(range(low, high + 1))
                break
        else:
            try:
                number = int(lowered)
            except ValueError as error:
                raise DocsStudioError(f"Not a page number: {part!r}.") from error
            if number < 1:
                raise DocsStudioError("Page numbers are 1-based.")
            pages.append(number)
    return tuple(dict.fromkeys(pages))


def build_vision_messages(
    document: LoadedDocument,
    question: str,
    *,
    pages: Sequence[int] | None = None,
    system_prompt: str = "",
    max_text_chars: int = 12000,
    max_pages: int = MAX_PAGES_PER_REQUEST,
) -> list[dict[str, Any]]:
    """Build the multimodal chat messages for one question about a document.

    The shape is what ``llama-cpp-python``'s ``MTMDChatHandler`` expects: a user
    message whose ``content`` is a list of parts, each either ``{"type": "text",
    ...}`` or ``{"type": "image_url", "image_url": {"url": "file://…"}}``. The
    extracted text is included as a text part, and the selected pages' images
    follow, so the model can read the words *and* look at the page.
    """
    wanted = tuple(pages) if pages is not None else select_pages(document, question, limit=max_pages)
    chosen: list[DocumentPage] = []
    for number in wanted:
        page = document.page(number)
        if page is not None:
            chosen.append(page)
        if len(chosen) >= max_pages:
            break
    # The pages are attached in document order, so the model reads the document as
    # it is laid out rather than in the order relevance happened to rank it.
    chosen.sort(key=lambda page: page.number)

    parts: list[dict[str, Any]] = []
    header = f"Document: {document.filename} ({document.kind}, {document.page_count} page(s))."
    body = document.text[:max_text_chars]
    if body.strip():
        parts.append(
            {
                "type": "text",
                "text": (
                    f"{header}\n\nExtracted text:\n{body}\n\n"
                    f"Question: {question}"
                ),
            }
        )
    else:
        parts.append(
            {
                "type": "text",
                "text": (
                    f"{header}\n\nThis document has no text layer; read the page "
                    f"image(s) below.\n\nQuestion: {question}"
                ),
            }
        )
    for page in chosen:
        if not page.image_path:
            continue
        parts.append(
            {
                "type": "text",
                "text": f"Page {page.number}:",
            }
        )
        parts.append(
            {
                "type": "image_url",
                "image_url": {"url": f"file://{Path(page.image_path).resolve()}"},
            }
        )

    messages: list[dict[str, Any]] = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": parts})
    return messages


__all__ = [
    "DEFAULT_DPI",
    "DocsStudioError",
    "DocumentPage",
    "IMAGE_SUFFIXES",
    "LoadedDocument",
    "MAX_PAGES_PER_REQUEST",
    "MAX_RENDERED_PAGES",
    "MAX_TEXT_CHARS",
    "MIN_CHARS_PER_PAGE",
    "SUPPORTED_SUFFIXES",
    "TEXT_SUFFIXES",
    "UNSUPPORTED_SUFFIXES",
    "WORD_SUFFIXES",
    "build_vision_messages",
    "load_document",
    "parse_page_range",
    "render_page_image",
    "select_pages",
]
