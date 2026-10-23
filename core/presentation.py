"""Presentation Studio: a persistent, reviewable PowerPoint authoring workflow.

This module is the deterministic engine behind the window's "Presentation
Studio" screen and its ``python -m cli presentation`` twin. It owns four things
and nothing else:

* the **project model** -- a :class:`PresentationProject` holding the workflow
  state, the approved outline, the numbered slides, the revision history, and
  the export history;
* the **state machine** -- the named states from DISCUSSION to EXPORTED and the
  legal transitions between them (:data:`STATE_ORDER`, :func:`can_transition`);
* the **approval gate** -- a slide keeps its content *and* its approval flag, so
  editing an approved slide marks that approval **outdated** rather than
  silently discarding it, and the export refuses to run until every slide is
  approved again;
* the **writer** -- a standard-library ``.pptx`` (Office Open XML) writer built
  from ``zipfile`` and ``xml.etree``, the direct analogue of
  :func:`core.imagegen.encode_png`. No third-party dependency is added.

The design follows the rest of the project: the model is a plain dataclass that
round-trips through ``to_primitive``/``from_primitive``, persistence reuses
:class:`storage.json_store.JsonStore`, and every operation that could lose work
refuses instead of overwriting. Content is plain Unicode, so Arabic, English,
and mixed-language text all round-trip unchanged.

Nothing here talks to a model. The discussion, outline, and slide text are
produced by the *existing* assistant (see :mod:`gui.presentation`), so this
module stays pure, testable, and free of a second AI.
"""

from __future__ import annotations

import io
import re
import zipfile
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from xml.etree import ElementTree as ET

from core.paths import data_dir

# --------------------------------------------------------------------------- #
# Workflow states
# --------------------------------------------------------------------------- #

#: DISCUSSION -> ... -> EXPORTED, with EXPORT_FAILED as the safe-retry side state.
STATE_DISCUSSION = "DISCUSSION"
STATE_OUTLINE_DRAFT = "OUTLINE_DRAFT"
STATE_OUTLINE_REVIEW = "OUTLINE_REVIEW"
STATE_SLIDE_DRAFTING = "SLIDE_DRAFTING"
STATE_SLIDE_REVIEW = "SLIDE_REVIEW"
STATE_CONTENT_APPROVED = "CONTENT_APPROVED"
STATE_EXPORTING = "EXPORTING"
STATE_EXPORTED = "EXPORTED"
STATE_EXPORT_FAILED = "EXPORT_FAILED"

#: The ordered happy path. ``EXPORT_FAILED`` sits outside it on purpose: it is
#: reachable from ``EXPORTING`` and returns there, so a failed export is a
#: detour, not a step that reorders the workflow.
STATE_ORDER: tuple[str, ...] = (
    STATE_DISCUSSION,
    STATE_OUTLINE_DRAFT,
    STATE_OUTLINE_REVIEW,
    STATE_SLIDE_DRAFTING,
    STATE_SLIDE_REVIEW,
    STATE_CONTENT_APPROVED,
    STATE_EXPORTING,
    STATE_EXPORTED,
)

STATE_LABELS: dict[str, str] = {
    STATE_DISCUSSION: "Discussion",
    STATE_OUTLINE_DRAFT: "Outline drafted",
    STATE_OUTLINE_REVIEW: "Outline review",
    STATE_SLIDE_DRAFTING: "Drafting slides",
    STATE_SLIDE_REVIEW: "Slide review",
    STATE_CONTENT_APPROVED: "Content approved",
    STATE_EXPORTING: "Exporting",
    STATE_EXPORTED: "Exported",
    STATE_EXPORT_FAILED: "Export failed (safe to retry)",
}

#: The forward edges of the happy path, plus the two detour edges around a
#: failed export. A transition not listed here is refused, so the workflow can
#: never be walked backwards by accident (the service asks to *reopen* a phase,
#: which resets the approval that depended on it).
_TRANSITIONS: dict[str, frozenset[str]] = {
    STATE_DISCUSSION: frozenset({STATE_OUTLINE_DRAFT}),
    STATE_OUTLINE_DRAFT: frozenset({STATE_OUTLINE_REVIEW}),
    STATE_OUTLINE_REVIEW: frozenset({STATE_SLIDE_DRAFTING}),
    STATE_SLIDE_DRAFTING: frozenset({STATE_SLIDE_REVIEW}),
    STATE_SLIDE_REVIEW: frozenset({STATE_CONTENT_APPROVED}),
    STATE_CONTENT_APPROVED: frozenset({STATE_EXPORTING}),
    STATE_EXPORTING: frozenset({STATE_EXPORTED, STATE_EXPORT_FAILED}),
    STATE_EXPORT_FAILED: frozenset({STATE_EXPORTING}),
    # An exported deck may be written again (the same approved content, another
    # file) without reopening it; a change still requires a fresh approval.
    STATE_EXPORTED: frozenset({STATE_EXPORTING}),
}


class PresentationError(ValueError):
    """A refused Presentation Studio operation (never a silent failure)."""


def can_transition(current: str, target: str) -> bool:
    """Whether the workflow may move from ``current`` to ``target``."""
    if current not in STATE_ORDER and current != STATE_EXPORT_FAILED:
        return False
    return target in _TRANSITIONS.get(current, frozenset())


def require_transition(current: str, target: str) -> None:
    """Raise :class:`PresentationError` unless the transition is legal."""
    if not can_transition(current, target):
        raise PresentationError(
            f"Cannot move the presentation from {STATE_LABELS.get(current, current)} "
            f"to {STATE_LABELS.get(target, target)}."
        )


# --------------------------------------------------------------------------- #
# Content types
# --------------------------------------------------------------------------- #

CONTENT_BULLETS = "bullets"
CONTENT_PARAGRAPH = "paragraph"
CONTENT_TABLE = "table"
CONTENT_EQUATION = "equation"
CONTENT_IMAGE = "image"
CONTENT_SECTION = "section"
CONTENT_TITLE = "title"

CONTENT_TYPES: tuple[str, ...] = (
    CONTENT_TITLE,
    CONTENT_SECTION,
    CONTENT_BULLETS,
    CONTENT_PARAGRAPH,
    CONTENT_TABLE,
    CONTENT_EQUATION,
    CONTENT_IMAGE,
)

CONTENT_TYPE_LABELS: dict[str, str] = {
    CONTENT_TITLE: "Title slide",
    CONTENT_SECTION: "Section header",
    CONTENT_BULLETS: "Bulleted list",
    CONTENT_PARAGRAPH: "Paragraph",
    CONTENT_TABLE: "Table",
    CONTENT_EQUATION: "Equation",
    CONTENT_IMAGE: "Image",
}


def resolve_content_type(value: str) -> str:
    """Normalise a content-type name, raising for an unknown one."""
    key = (value or "").strip().lower()
    if key not in CONTENT_TYPES:
        raise PresentationError(
            f"Unknown content type {value!r}; expected one of {', '.join(CONTENT_TYPES)}."
        )
    return key


#: How many slides one drafting batch produces unless the user says otherwise.
#: Five keeps the *first* review pass small, which is the whole point of batching;
#: the caller may raise it (ten is a common choice) with no other change.
DEFAULT_SLIDE_BATCH = 5
#: An upper bound on one batch, so a typo cannot ask for an unbounded run.
MAX_SLIDE_BATCH = 50


def resolve_batch_size(value: int | None, *, default: int = DEFAULT_SLIDE_BATCH) -> int:
    """Validate a batch size, defaulting and clamping to the allowed range."""
    if value is None:
        return default
    size = int(value)
    if size < 1:
        raise PresentationError("A batch size must be 1 or greater.")
    return min(size, MAX_SLIDE_BATCH)


# --------------------------------------------------------------------------- #
# Slide geometry, themes, and presentation settings
# --------------------------------------------------------------------------- #

#: English Metric Units per inch (the unit DrawingML uses for every length).
EMU_PER_INCH = 914400
#: EMU per point (a point is 1/72 inch).
EMU_PER_POINT = 12700

#: 16:9 widescreen -- the default for a modern deck.
SLIDE_16_9 = (12192000, 6858000)
#: 4:3 -- the legacy projector aspect, offered when a venue needs it.
SLIDE_4_3 = (9144000, 6858000)

ASPECT_16_9 = "16:9"
ASPECT_4_3 = "4:3"
ASPECTS: dict[str, tuple[int, int]] = {ASPECT_16_9: SLIDE_16_9, ASPECT_4_3: SLIDE_4_3}
DEFAULT_ASPECT = ASPECT_16_9


def resolve_aspect(value: str | None) -> str:
    """Normalise an aspect name, defaulting to 16:9 and refusing an unknown one."""
    key = (value or "").strip().lower().replace(" ", "")
    if not key:
        return DEFAULT_ASPECT
    aliases = {
        "16:9": ASPECT_16_9,
        "16x9": ASPECT_16_9,
        "widescreen": ASPECT_16_9,
        "wide": ASPECT_16_9,
        "4:3": ASPECT_4_3,
        "4x3": ASPECT_4_3,
        "standard": ASPECT_4_3,
    }
    if key not in aliases:
        raise PresentationError(
            f"Unknown aspect {value!r}; expected one of {', '.join(sorted(ASPECTS))}."
        )
    return aliases[key]


def slide_size(aspect: str) -> tuple[int, int]:
    """The ``(cx, cy)`` EMU geometry for an aspect name."""
    return ASPECTS[resolve_aspect(aspect)]


@dataclass(frozen=True)
class DeckTheme:
    """Fonts, sizes, colours, and spacing for a rendered deck.

    Sizes are in points; lengths are in EMU. The values are deliberately
    conservative so text stays readable on a projector and never crowds the
    slide edges.
    """

    name: str = "professional"
    #: Latin/body and heading typefaces.
    heading_font: str = "Calibri Light"
    body_font: str = "Calibri"
    #: Complex-script (Arabic) typeface; empty means "use the heading/body font".
    arabic_font: str = "Arial"
    #: Point sizes.
    title_size: float = 40.0
    subtitle_size: float = 22.0
    section_size: float = 34.0
    heading_size: float = 30.0
    body_size: float = 20.0
    small_size: float = 16.0
    notes_size: float = 12.0
    table_size: float = 16.0
    equation_size: float = 20.0
    #: Colours as ``RRGGBB``.
    background: str = "FFFFFF"
    foreground: str = "1F2933"
    muted: str = "52606D"
    accent: str = "C8A24A"
    accent_dark: str = "2E5A88"
    section_background: str = "1F2933"
    section_foreground: str = "FFFFFF"
    #: Spacing.
    margin_x: int = 838200
    margin_y: int = 685800
    title_gap: int = 274320
    line_spacing: float = 1.15
    space_after: float = 0.35
    #: Slide-number placement.
    show_slide_numbers: bool = True

    def script_font(self, font: str) -> str:
        """The complex-script typeface to pair with ``font`` (Arabic text)."""
        return self.arabic_font or font

    def to_primitive(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "heading_font": self.heading_font,
            "body_font": self.body_font,
            "arabic_font": self.arabic_font,
            "title_size": self.title_size,
            "subtitle_size": self.subtitle_size,
            "section_size": self.section_size,
            "heading_size": self.heading_size,
            "body_size": self.body_size,
            "small_size": self.small_size,
            "notes_size": self.notes_size,
            "table_size": self.table_size,
            "equation_size": self.equation_size,
            "background": self.background,
            "foreground": self.foreground,
            "muted": self.muted,
            "accent": self.accent,
            "accent_dark": self.accent_dark,
            "section_background": self.section_background,
            "section_foreground": self.section_foreground,
            "margin_x": self.margin_x,
            "margin_y": self.margin_y,
            "title_gap": self.title_gap,
            "line_spacing": self.line_spacing,
            "space_after": self.space_after,
            "show_slide_numbers": self.show_slide_numbers,
        }


#: Named themes a user may choose. "professional" is the default.
THEMES: dict[str, DeckTheme] = {
    "professional": DeckTheme(),
    "light": DeckTheme(
        name="light",
        background="FFFFFF",
        foreground="111827",
        muted="6B7280",
        accent="2563EB",
        accent_dark="1E40AF",
        section_background="1E3A8A",
        section_foreground="FFFFFF",
    ),
    "academic": DeckTheme(
        name="academic",
        heading_font="Cambria",
        body_font="Cambria",
        accent="7A1F1F",
        accent_dark="1F3A5F",
        section_background="1F3A5F",
        section_foreground="FFFFFF",
    ),
    "dark": DeckTheme(
        name="dark",
        background="111827",
        foreground="F9FAFB",
        muted="9CA3AF",
        accent="F59E0B",
        accent_dark="60A5FA",
        section_background="000000",
        section_foreground="F9FAFB",
    ),
}

DEFAULT_THEME = "professional"


def resolve_theme(value: str | None) -> DeckTheme:
    """Return a :class:`DeckTheme` by name, defaulting to "professional"."""
    key = (value or "").strip().lower()
    if not key:
        return THEMES[DEFAULT_THEME]
    if key not in THEMES:
        raise PresentationError(
            f"Unknown theme {value!r}; expected one of {', '.join(sorted(THEMES))}."
        )
    return THEMES[key]


def theme_names() -> tuple[str, ...]:
    """The available theme names, default first."""
    return (DEFAULT_THEME, *tuple(sorted(name for name in THEMES if name != DEFAULT_THEME)))


@dataclass(frozen=True)
class PresentationSettings:
    """How a deck should be rendered: geometry, theme, numbering, and metadata."""

    aspect: str = DEFAULT_ASPECT
    theme: str = DEFAULT_THEME
    show_slide_numbers: bool = True
    footer: str = ""
    author: str = ""
    company: str = ""
    subject: str = ""
    language: str = "en"
    #: Force right-to-left paragraph direction on every body paragraph.
    rtl: bool = False
    #: Refuse to overwrite an existing file unless the caller says otherwise.
    allow_overwrite: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "aspect", resolve_aspect(self.aspect))
        object.__setattr__(self, "theme", (self.theme or DEFAULT_THEME).strip().lower())
        if self.theme not in THEMES:
            raise PresentationError(
                f"Unknown theme {self.theme!r}; expected one of {', '.join(sorted(THEMES))}."
            )

    @property
    def theme_spec(self) -> DeckTheme:
        """The resolved :class:`DeckTheme` for this deck."""
        return THEMES[self.theme]

    @property
    def size(self) -> tuple[int, int]:
        """The ``(cx, cy)`` EMU geometry for this deck's aspect."""
        return ASPECTS[self.aspect]

    def replace(self, **changes: Any) -> PresentationSettings:
        """A copy with ``changes`` applied (validated like a fresh instance)."""
        data = self.to_primitive()
        data.update(changes)
        return PresentationSettings.from_primitive(data)

    def to_primitive(self) -> dict[str, Any]:
        return {
            "aspect": self.aspect,
            "theme": self.theme,
            "show_slide_numbers": self.show_slide_numbers,
            "footer": self.footer,
            "author": self.author,
            "company": self.company,
            "subject": self.subject,
            "language": self.language,
            "rtl": self.rtl,
            "allow_overwrite": self.allow_overwrite,
        }

    @classmethod
    def from_primitive(cls, data: dict[str, Any] | None) -> PresentationSettings:
        """Rebuild settings from a stored dict (missing keys take defaults)."""
        if not data:
            return cls()
        return cls(
            aspect=str(data.get("aspect", DEFAULT_ASPECT)),
            theme=str(data.get("theme", DEFAULT_THEME)),
            show_slide_numbers=bool(data.get("show_slide_numbers", True)),
            footer=str(data.get("footer", "")),
            author=str(data.get("author", "")),
            company=str(data.get("company", "")),
            subject=str(data.get("subject", "")),
            language=str(data.get("language", "en")),
            rtl=bool(data.get("rtl", False)),
            allow_overwrite=bool(data.get("allow_overwrite", False)),
        )


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

#: A stable, filesystem-safe slug. Keeps Arabic letters (they are word
#: characters in Unicode) so a mixed-language title still yields a name.
_SLUG_RE = re.compile(r"[^\w]+", re.UNICODE)


def _now() -> str:
    """An ISO-8601 UTC timestamp, second resolution (stable for tests)."""
    return datetime.now(UTC).replace(microsecond=0).isoformat()


def slugify(text: str, *, fallback: str = "presentation") -> str:
    """A short, filesystem-safe slug from a title (Unicode-aware)."""
    value = _SLUG_RE.sub("-", (text or "").strip()).strip("-_")
    value = re.sub(r"-{2,}", "-", value)
    return value[:60] or fallback


def default_presentation_dir() -> Path:
    """Where generated ``.pptx`` files are written by default (git-ignored)."""
    return data_dir() / "presentations"


def presentation_filename(title: str, *, suffix: str = ".pptx") -> str:
    """A filename for a deck, derived from its title."""
    return f"{slugify(title)}{suffix}"


def _clean_lines(values: Iterable[str]) -> list[str]:
    """Drop blank lines and normalise whitespace inside each, order preserved."""
    out: list[str] = []
    for value in values:
        text = " ".join(str(value).split())
        if text:
            out.append(text)
    return out


def slide_topic_key(section: str, subsection: str = "") -> str:
    """The key naming the outline heading a slide covers.

    A subsection is qualified by its section, because the same subsection name
    can occur under two sections; the key therefore stays unique across the
    outline. An empty key means the slide is not tied to an outline heading.
    """
    head = (section or "").strip()
    sub = (subsection or "").strip()
    if sub:
        return f"{head} / {sub}" if head else sub
    return head


# --------------------------------------------------------------------------- #
# Outline
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class OutlineSection:
    """One outline section: a heading, its subsections, and a slide estimate."""

    heading: str
    subsections: tuple[str, ...] = ()
    estimated_slides: int = 1

    def __post_init__(self) -> None:
        if not self.heading.strip():
            raise PresentationError("An outline section needs a heading.")
        if self.estimated_slides < 1:
            raise PresentationError("An outline section must estimate at least one slide.")

    def to_primitive(self) -> dict[str, Any]:
        return {
            "heading": self.heading,
            "subsections": list(self.subsections),
            "estimated_slides": self.estimated_slides,
        }

    @classmethod
    def from_primitive(cls, data: dict[str, Any]) -> OutlineSection:
        return cls(
            heading=str(data["heading"]),
            subsections=tuple(str(item) for item in data.get("subsections", ())),
            estimated_slides=int(data.get("estimated_slides", 1)),
        )


@dataclass(frozen=True)
class OutlineRevision:
    """One immutable revision of the outline, kept so a draft can be restored."""

    revision: int
    sections: tuple[OutlineSection, ...]
    note: str = ""
    created_at: str = field(default_factory=_now)

    @property
    def estimated_slides(self) -> int:
        """The total slide estimate across every section."""
        return sum(section.estimated_slides for section in self.sections)

    def to_primitive(self) -> dict[str, Any]:
        return {
            "revision": self.revision,
            "sections": [section.to_primitive() for section in self.sections],
            "note": self.note,
            "created_at": self.created_at,
        }

    @classmethod
    def from_primitive(cls, data: dict[str, Any]) -> OutlineRevision:
        return cls(
            revision=int(data["revision"]),
            sections=tuple(OutlineSection.from_primitive(item) for item in data["sections"]),
            note=str(data.get("note", "")),
            created_at=str(data.get("created_at") or _now()),
        )


# --------------------------------------------------------------------------- #
# Slides
# --------------------------------------------------------------------------- #

#: The slide fields a content edit may change. ``approved`` and ``order`` are
#: deliberately absent: approval has its own transition and order is separate.
#: ``section``/``subsection`` are editable too, so a slide can be re-homed under
#: a different outline heading without losing its stable id.
EDITABLE_FIELDS: tuple[str, ...] = (
    "title",
    "content",
    "content_type",
    "table",
    "equations",
    "images",
    "sources",
    "notes",
    "section",
    "subsection",
    "layout",
)


@dataclass
class Slide:
    """One slide: a stable id, a displayed number, its content, and its state.

    The :attr:`slide_id` never changes, so reordering (which only rewrites
    :attr:`order`) cannot lose a slide's identity. The content fields and the
    :attr:`approved` flag are stored *together* on purpose: an edit clears the
    flag (recording the previous approved snapshot first), which is what makes
    "never overwrite an approved version" true by construction.
    """

    slide_id: str
    order: int
    title: str
    content: list[str] = field(default_factory=list)
    content_type: str = CONTENT_BULLETS
    table: list[list[str]] = field(default_factory=list)
    equations: list[str] = field(default_factory=list)
    images: list[str] = field(default_factory=list)
    sources: list[str] = field(default_factory=list)
    notes: str = ""
    #: The outline section (and optional subsection) this slide belongs to. They
    #: travel with the slide, so a reorder never loses where a slide came from.
    section: str = ""
    subsection: str = ""
    #: An optional explicit layout hint (e.g. "two_column", "chart"); empty lets
    #: the exporter infer the layout from the content type and the title.
    layout: str = ""
    approved: bool = False
    revision: int = 0
    #: The last approved content snapshot, kept so an edit can be undone.
    approved_snapshot: dict[str, Any] | None = None
    created_at: str = field(default_factory=_now)
    updated_at: str = field(default_factory=_now)

    def __post_init__(self) -> None:
        if not self.slide_id.strip():
            raise PresentationError("A slide needs a stable id.")
        if self.order < 1:
            raise PresentationError("A slide's order must be 1 or greater.")
        self.content_type = resolve_content_type(self.content_type)
        self.title = str(self.title)
        self.content = _clean_lines(self.content)
        self.equations = _clean_lines(self.equations)
        self.images = _clean_lines(self.images)
        self.sources = _clean_lines(self.sources)
        self.table = [[str(cell) for cell in row] for row in self.table if row]
        self.notes = str(self.notes)
        self.section = str(self.section).strip()
        self.subsection = str(self.subsection).strip()
        self.layout = str(self.layout).strip().lower().replace("-", "_").replace(" ", "_")

    @property
    def number(self) -> int:
        """The displayed slide number (the same value as :attr:`order`)."""
        return self.order

    @property
    def is_empty(self) -> bool:
        """True when the slide carries no body content of any kind."""
        return not (self.title.strip() or self.content or self.table or self.equations)

    @property
    def topic(self) -> str:
        """The outline heading this slide covers (see :func:`slide_topic_key`)."""
        return slide_topic_key(self.section, self.subsection)

    def content_snapshot(self) -> dict[str, Any]:
        """The editable content of this slide, as a plain dict."""
        return {
            "title": self.title,
            "content": list(self.content),
            "content_type": self.content_type,
            "table": [list(row) for row in self.table],
            "equations": list(self.equations),
            "images": list(self.images),
            "sources": list(self.sources),
            "notes": self.notes,
            "section": self.section,
            "subsection": self.subsection,
            "layout": self.layout,
        }

    def to_primitive(self) -> dict[str, Any]:
        data = self.content_snapshot()
        data.update(
            {
                "slide_id": self.slide_id,
                "order": self.order,
                "approved": self.approved,
                "revision": self.revision,
                "approved_snapshot": self.approved_snapshot,
                "created_at": self.created_at,
                "updated_at": self.updated_at,
            }
        )
        return data

    @classmethod
    def from_primitive(cls, data: dict[str, Any]) -> Slide:
        snapshot = data.get("approved_snapshot")
        return cls(
            slide_id=str(data["slide_id"]),
            order=int(data["order"]),
            title=str(data.get("title", "")),
            content=list(data.get("content", ())),
            content_type=str(data.get("content_type", CONTENT_BULLETS)),
            table=[list(row) for row in data.get("table", ())],
            equations=list(data.get("equations", ())),
            images=list(data.get("images", ())),
            sources=list(data.get("sources", ())),
            notes=str(data.get("notes", "")),
            section=str(data.get("section", "")),
            subsection=str(data.get("subsection", "")),
            layout=str(data.get("layout", "")),
            approved=bool(data.get("approved", False)),
            revision=int(data.get("revision", 0)),
            approved_snapshot=dict(snapshot) if isinstance(snapshot, dict) else None,
            created_at=str(data.get("created_at") or _now()),
            updated_at=str(data.get("updated_at") or _now()),
        )


# --------------------------------------------------------------------------- #
# Revisions and exports
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class SlideRevision:
    """An immutable record of a slide's content at one point in time."""

    revision: int
    slide_id: str
    content: dict[str, Any]
    approved: bool
    reason: str
    created_at: str = field(default_factory=_now)

    def to_primitive(self) -> dict[str, Any]:
        return {
            "revision": self.revision,
            "slide_id": self.slide_id,
            "content": self.content,
            "approved": self.approved,
            "reason": self.reason,
            "created_at": self.created_at,
        }

    @classmethod
    def from_primitive(cls, data: dict[str, Any]) -> SlideRevision:
        return cls(
            revision=int(data["revision"]),
            slide_id=str(data["slide_id"]),
            content=dict(data["content"]),
            approved=bool(data.get("approved", False)),
            reason=str(data.get("reason", "")),
            created_at=str(data.get("created_at") or _now()),
        )


@dataclass(frozen=True)
class ExportRecord:
    """One export attempt: where it wrote, whether it validated, and why not."""

    export_id: str
    path: str
    ok: bool
    message: str
    slide_count: int
    size_bytes: int = 0
    created_at: str = field(default_factory=_now)

    def to_primitive(self) -> dict[str, Any]:
        return {
            "export_id": self.export_id,
            "path": self.path,
            "ok": self.ok,
            "message": self.message,
            "slide_count": self.slide_count,
            "size_bytes": self.size_bytes,
            "created_at": self.created_at,
        }

    @classmethod
    def from_primitive(cls, data: dict[str, Any]) -> ExportRecord:
        return cls(
            export_id=str(data["export_id"]),
            path=str(data["path"]),
            ok=bool(data["ok"]),
            message=str(data.get("message", "")),
            slide_count=int(data.get("slide_count", 0)),
            size_bytes=int(data.get("size_bytes", 0)),
            created_at=str(data.get("created_at") or _now()),
        )


# --------------------------------------------------------------------------- #
# The project
# --------------------------------------------------------------------------- #

_SCHEMA_VERSION = 1


@dataclass
class PresentationProject:
    """One independent presentation project and everything it has been through."""

    project_id: str
    title: str
    state: str = STATE_DISCUSSION
    subtitle: str = ""
    audience: str = ""
    language: str = "en"
    #: Conversation ids the user discussed this deck in (references, not copies).
    conversation_ids: list[str] = field(default_factory=list)
    outline_revisions: list[OutlineRevision] = field(default_factory=list)
    approved_outline_revision: int | None = None
    slides: list[Slide] = field(default_factory=list)
    revisions: list[SlideRevision] = field(default_factory=list)
    exports: list[ExportRecord] = field(default_factory=list)
    content_approved: bool = False
    #: How the deck should be rendered (geometry, theme, numbering, metadata).
    #: Kept on the project so a deck's look survives a restart and is shared by
    #: the window and the CLI.
    settings: PresentationSettings = field(default_factory=PresentationSettings)
    created_at: str = field(default_factory=_now)
    updated_at: str = field(default_factory=_now)
    schema_version: int = _SCHEMA_VERSION

    def __post_init__(self) -> None:
        if not self.project_id.strip():
            raise PresentationError("A project needs an id.")
        if not self.title.strip():
            raise PresentationError("A project needs a title.")

    # ------------------------------------------------------------- reading

    @property
    def outline(self) -> OutlineRevision | None:
        """The approved outline revision, or the newest draft when none is set."""
        if self.approved_outline_revision is not None:
            for revision in self.outline_revisions:
                if revision.revision == self.approved_outline_revision:
                    return revision
        return self.outline_revisions[-1] if self.outline_revisions else None

    @property
    def latest_outline(self) -> OutlineRevision | None:
        """The newest outline revision, approved or not."""
        return self.outline_revisions[-1] if self.outline_revisions else None

    def outline_revision(self, revision: int) -> OutlineRevision:
        """The outline revision numbered ``revision``; raises when unknown."""
        for item in self.outline_revisions:
            if item.revision == revision:
                return item
        raise PresentationError(f"No outline revision {revision}.")

    def ordered_slides(self) -> list[Slide]:
        """Every slide in display order (a fresh list, so it cannot be mutated)."""
        return sorted(self.slides, key=lambda slide: slide.order)

    def slide(self, slide_id: str) -> Slide:
        """The slide with ``slide_id``; raises when it is unknown."""
        for slide in self.slides:
            if slide.slide_id == slide_id:
                return slide
        raise PresentationError(f"No slide with id {slide_id!r}.")

    def slide_by_number(self, number: int) -> Slide:
        """The slide displayed as ``number``; raises when there is none."""
        for slide in self.slides:
            if slide.order == number:
                return slide
        raise PresentationError(f"No slide numbered {number}.")

    @property
    def approved_slides(self) -> list[Slide]:
        """The approved slides, in display order."""
        return [slide for slide in self.ordered_slides() if slide.approved]

    @property
    def outdated_slides(self) -> list[Slide]:
        """Slides whose content was edited after their last approval."""
        return [slide for slide in self.ordered_slides() if not slide.approved]

    @property
    def all_slides_approved(self) -> bool:
        """True when there is at least one slide and every slide is approved."""
        return bool(self.slides) and all(slide.approved for slide in self.slides)

    @property
    def approval_outdated(self) -> bool:
        """True when content changed after the project was approved."""
        return self.content_approved and not self.all_slides_approved

    @property
    def total_estimated_slides(self) -> int:
        """The slide estimate from the current outline."""
        outline = self.outline
        return outline.estimated_slides if outline is not None else len(self.slides)

    def status(self) -> str:
        """An honest, one-line description of where the project stands."""
        return (
            f"{self.title} - {STATE_LABELS.get(self.state, self.state)}: "
            f"{len(self.slides)} slide(s), "
            f"{len(self.approved_slides)} approved"
            + (", approval outdated" if self.approval_outdated else "")
        )

    # ------------------------------------------------------------- writing

    def touch(self) -> None:
        """Stamp the project as modified now."""
        self.updated_at = _now()

    def next_revision(self) -> int:
        """The next revision number (monotonic across the whole project)."""
        return (self.revisions[-1].revision + 1) if self.revisions else 1

    def next_export_id(self) -> str:
        """A fresh, human-readable export id."""
        return f"EXP-{len(self.exports) + 1:03d}"

    def next_slide_id(self) -> str:
        """A fresh, stable slide id that does not depend on its number."""
        highest = 0
        for slide in self.slides:
            if slide.slide_id.startswith("SLD-"):
                try:
                    highest = max(highest, int(slide.slide_id[4:]))
                except ValueError:
                    continue
        return f"SLD-{highest + 1:03d}"

    def renumber(self) -> None:
        """Renumber every slide 1..n in its current display order."""
        for index, slide in enumerate(self.ordered_slides(), start=1):
            slide.order = index

    def slide_numbers(self) -> list[int]:
        """The displayed numbers, 1..n, in current order (the numbering is dense)."""
        return list(range(1, len(self.slides) + 1))

    def slides_in_range(self, first: int, last: int | None = None) -> list[Slide]:
        """The slides numbered ``first``..``last`` (inclusive), in display order.

        A range is clamped to the slides that exist rather than raising, so
        "shorten slides 11-15" still does the sensible thing on a shorter deck;
        the caller decides whether an empty result is an error.
        """
        if last is None:
            last = first
        low, high = (first, last) if first <= last else (last, first)
        ordered = self.ordered_slides()
        low = max(1, low)
        high = min(len(ordered), high)
        return ordered[low - 1 : high] if low <= high else []

    def insert_slide(self, slide: Slide, position: int | None = None) -> None:
        """Insert ``slide`` at ``position`` (1-based; appended when omitted)."""
        ordered = self.ordered_slides()
        target = len(ordered) + 1 if position is None else max(1, min(position, len(ordered) + 1))
        ordered.insert(target - 1, slide)
        for index, item in enumerate(ordered, start=1):
            item.order = index
        self.slides = ordered

    def record_revision(self, slide: Slide, reason: str) -> SlideRevision:
        """Append a revision snapshot of ``slide`` and return it."""
        revision = SlideRevision(
            revision=self.next_revision(),
            slide_id=slide.slide_id,
            content=slide.content_snapshot(),
            approved=slide.approved,
            reason=reason,
        )
        self.revisions.append(revision)
        slide.revision = revision.revision
        return revision

    def revisions_for(self, slide_id: str) -> list[SlideRevision]:
        """Every revision recorded for one slide, oldest first."""
        return [item for item in self.revisions if item.slide_id == slide_id]

    def to_primitive(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "project_id": self.project_id,
            "title": self.title,
            "subtitle": self.subtitle,
            "audience": self.audience,
            "language": self.language,
            "state": self.state,
            "conversation_ids": list(self.conversation_ids),
            "outline_revisions": [item.to_primitive() for item in self.outline_revisions],
            "approved_outline_revision": self.approved_outline_revision,
            "slides": [slide.to_primitive() for slide in self.slides],
            "revisions": [item.to_primitive() for item in self.revisions],
            "exports": [item.to_primitive() for item in self.exports],
            "content_approved": self.content_approved,
            "settings": self.settings.to_primitive(),
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }

    @classmethod
    def from_primitive(cls, data: dict[str, Any]) -> PresentationProject:
        version = int(data.get("schema_version", 0))
        if version > _SCHEMA_VERSION:
            raise PresentationError(
                f"This project was written by a newer version ({version} > "
                f"{_SCHEMA_VERSION}); update the program before opening it."
            )
        return cls(
            project_id=str(data["project_id"]),
            title=str(data["title"]),
            state=str(data.get("state", STATE_DISCUSSION)),
            subtitle=str(data.get("subtitle", "")),
            audience=str(data.get("audience", "")),
            language=str(data.get("language", "en")),
            conversation_ids=[str(item) for item in data.get("conversation_ids", ())],
            outline_revisions=[
                OutlineRevision.from_primitive(item)
                for item in data.get("outline_revisions", ())
            ],
            approved_outline_revision=(
                int(data["approved_outline_revision"])
                if data.get("approved_outline_revision") is not None
                else None
            ),
            slides=[Slide.from_primitive(item) for item in data.get("slides", ())],
            revisions=[SlideRevision.from_primitive(item) for item in data.get("revisions", ())],
            exports=[ExportRecord.from_primitive(item) for item in data.get("exports", ())],
            content_approved=bool(data.get("content_approved", False)),
            settings=PresentationSettings.from_primitive(data.get("settings")),
            created_at=str(data.get("created_at") or _now()),
            updated_at=str(data.get("updated_at") or _now()),
            schema_version=version or _SCHEMA_VERSION,
        )


# --------------------------------------------------------------------------- #
# Persistence
# --------------------------------------------------------------------------- #

#: The JsonStore namespace presentation projects live under. Separate from
#: ``documents``/``jarvis``, so a project is never entangled with the store.
PRESENTATIONS_NAMESPACE = "presentations"


class PresentationStore:
    """Namespaced persistence for presentation projects.

    Reuses :class:`storage.json_store.JsonStore` -- the same adapter the
    conversation memory and the preferences use -- so no new storage layer is
    introduced. Reads tolerate a missing file and a newer schema is refused
    loudly rather than half-read, so an upgrade never silently corrupts a deck.
    """

    def __init__(self, root: str | Path | None = None) -> None:
        from storage.json_store import JsonStore

        self.root = Path(root) if root is not None else data_dir()
        self._store = JsonStore(self.root)

    def save(self, project: PresentationProject) -> None:
        """Persist one project (atomic: JsonStore writes a temp then replaces)."""
        self._store.save(PRESENTATIONS_NAMESPACE, project.project_id, project.to_primitive())

    def load(self, project_id: str) -> PresentationProject | None:
        """Load a project by id, or ``None`` when it does not exist."""
        record = self._store.load(PRESENTATIONS_NAMESPACE, project_id)
        if record is None:
            return None
        return PresentationProject.from_primitive(record)

    def list_ids(self) -> list[str]:
        """Every stored project id, sorted."""
        return sorted(self._store.list_keys(PRESENTATIONS_NAMESPACE))

    def list_projects(self) -> list[PresentationProject]:
        """Every stored project, skipping any file that cannot be read."""
        projects: list[PresentationProject] = []
        for project_id in self.list_ids():
            try:
                project = self.load(project_id)
            except (PresentationError, ValueError, KeyError, TypeError, OSError):
                continue
            if project is not None:
                projects.append(project)
        return projects

    def delete(self, project_id: str) -> bool:
        """Remove a project. Returns whether a file was removed."""
        path = self._store.root / self._store._encode(PRESENTATIONS_NAMESPACE)
        target = path / f"{self._store._encode(project_id)}.json"
        if target.exists():
            target.unlink()
            return True
        return False


# --------------------------------------------------------------------------- #
# The .pptx writer (standard library only)
# --------------------------------------------------------------------------- #

_NS_P = "http://schemas.openxmlformats.org/presentationml/2006/main"
_NS_A = "http://schemas.openxmlformats.org/drawingml/2006/main"
_NS_R = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
_NS_CT = "http://schemas.openxmlformats.org/package/2006/content-types"
_NS_REL = "http://schemas.openxmlformats.org/package/2006/relationships"

_CT_PRESENTATION = (
    "application/vnd.openxmlformats-officedocument.presentationml.presentation.main+xml"
)
_CT_SLIDE = "application/vnd.openxmlformats-officedocument.presentationml.slide+xml"
_CT_MASTER = "application/vnd.openxmlformats-officedocument.presentationml.slideMaster+xml"
_CT_LAYOUT = "application/vnd.openxmlformats-officedocument.presentationml.slideLayout+xml"
_CT_NOTES = "application/vnd.openxmlformats-officedocument.presentationml.notesSlide+xml"
_CT_THEME = "application/vnd.openxmlformats-officedocument.theme+xml"
_CT_RELS = "application/vnd.openxmlformats-package.relationships+xml"
_CT_XML = "application/xml"

#: 4:3 slide geometry in EMU, matching the reference writer.
_SLIDE_CX = 9144000
_SLIDE_CY = 6858000
_MARGIN_X = 457200
_MARGIN_Y = 457200

#: A table style id every consumer recognises (medium style 2 - accent 1).
_TABLE_STYLE_ID = "{5C22544A-7EE6-4342-B048-85BDC9FD1C3A}"


def _xml_escape(text: str) -> str:
    """Escape the five XML metacharacters in text content or an attribute."""
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
        .replace("'", "&apos;")
    )


def _rpr(size: int, *, bold: bool = False, italic: bool = False) -> str:
    """Run properties at ``size`` hundredths of a point."""
    attrs = f'lang="en-US" sz="{size}" dirty="0"'
    if bold:
        attrs += ' b="1"'
    if italic:
        attrs += ' i="1"'
    return f"<a:rPr {attrs}/>"


def _paragraph(
    text: str,
    size: int,
    *,
    bold: bool = False,
    italic: bool = False,
    level: int = 0,
) -> str:
    """One DrawingML paragraph holding a single run."""
    if not text:
        return f'<a:p><a:pPr lvl="{level}"/><a:endParaRPr lang="en-US" sz="{size}"/></a:p>'
    return (
        f'<a:p><a:pPr lvl="{level}"/><a:r>{_rpr(size, bold=bold, italic=italic)}'
        f"<a:t>{_xml_escape(text)}</a:t></a:r></a:p>"
    )


def _bullet(text: str, size: int) -> str:
    """A paragraph rendered as a bullet (the layout supplies the glyph)."""
    return (
        '<a:p><a:pPr marL="285750" indent="-285750" lvl="0"/>'
        f"<a:r>{_rpr(size)}<a:t>{_xml_escape(text)}</a:t></a:r></a:p>"
    )


def _text_box(
    x: int,
    y: int,
    cx: int,
    cy: int,
    paragraphs: str,
    *,
    name: str,
    sid: int,
) -> str:
    """A rectangular text shape holding ``paragraphs``."""
    return (
        f'<p:sp><p:nvSpPr><p:cNvPr id="{sid}" name="{_xml_escape(name)}"/>'
        '<p:cNvSpPr><a:spLocks noGrp="1"/></p:cNvSpPr>'
        f'<p:nvPr><p:ph type="body" idx="{sid}"/></p:nvPr></p:nvSpPr>'
        f'<p:spPr><a:xfrm><a:off x="{x}" y="{y}"/><a:ext cx="{cx}" cy="{cy}"/></a:xfrm>'
        '<a:prstGeom prst="rect"><a:avLst/></a:prstGeom></p:spPr>'
        '<p:txBody><a:bodyPr wrap="square" rtlCol="0"><a:normAutofit/></a:bodyPr>'
        f"<a:lstStyle/>{paragraphs}</p:txBody></p:sp>"
    )


def _table_shape(slide: Slide, x: int, y: int, cx: int, *, sid: int) -> str:
    """A real DrawingML table shape for a slide's ``table`` content."""
    rows = slide.table
    cols = max(len(row) for row in rows)
    row_h = 370840
    grid = "".join(f'<a:gridCol w="{cx // cols}"/>' for _ in range(cols))
    body: list[str] = []
    for r_index, row in enumerate(rows):
        cells: list[str] = []
        for c_index in range(cols):
            cell_text = row[c_index] if c_index < len(row) else ""
            cells.append(
                "<a:tc><a:txBody><a:bodyPr/><a:lstStyle/>"
                + _paragraph(cell_text, 1400, bold=(r_index == 0))
                + "</a:txBody><a:tcPr/></a:tc>"
            )
        body.append(f'<a:tr h="{row_h}">' + "".join(cells) + "</a:tr>")
    return (
        f'<p:graphicFrame><p:nvGraphicFramePr><p:cNvPr id="{sid}" name="Table"/>'
        '<p:cNvGraphicFramePr><a:graphicFrameLocks noGrp="1"/></p:cNvGraphicFramePr>'
        "<p:nvPr/></p:nvGraphicFramePr>"
        f'<p:xfrm><a:off x="{x}" y="{y}"/><a:ext cx="{cx}" cy="{row_h * len(rows)}"/></p:xfrm>'
        '<a:graphic><a:graphicData uri="http://schemas.openxmlformats.org/drawingml/2006/table">'
        f'<a:tbl><a:tblPr firstRow="1" bandRow="1"><a:tableStyleId>{_TABLE_STYLE_ID}'
        f"</a:tableStyleId></a:tblPr><a:tblGrid>{grid}</a:tblGrid>"
        + "".join(body)
        + "</a:tbl></a:graphicData></a:graphic></p:graphicFrame>"
    )


def _image_shape(slide: Slide, x: int, y: int, cx: int, cy: int, *, sid: int) -> str:
    """A framed placeholder naming the slide's first image reference.

    The image *file* is not embedded: the engine writes a deterministic deck and
    records the reference (in the alt text and the frame), exactly as the rest of
    the project keeps a reference rather than copying a user's data.
    """
    ref = slide.images[0] if slide.images else ""
    return (
        f'<p:sp><p:nvSpPr><p:cNvPr id="{sid}" name="Image reference" '
        f'descr="{_xml_escape(ref)}"/><p:cNvSpPr/><p:nvPr/></p:nvSpPr>'
        f'<p:spPr><a:xfrm><a:off x="{x}" y="{y}"/><a:ext cx="{cx}" cy="{cy}"/></a:xfrm>'
        '<a:prstGeom prst="rect"><a:avLst/></a:prstGeom>'
        '<a:ln><a:solidFill><a:srgbClr val="C8A24A"/></a:solidFill>'
        '<a:prstDash val="dash"/></a:ln></p:spPr>'
        "<p:txBody><a:bodyPr wrap=\"square\"/><a:lstStyle/>"
        + _paragraph(f"[image: {ref}]", 1200, italic=True)
        + "</p:txBody></p:sp>"
    )


def _slide_xml(slide: Slide) -> str:
    """The ``ppt/slides/slideN.xml`` part for one slide."""
    shapes: list[str] = []
    y = _MARGIN_Y
    body_cx = _SLIDE_CX - 2 * _MARGIN_X

    title_cy = 1005840
    shapes.append(
        _text_box(
            _MARGIN_X,
            y,
            body_cx,
            title_cy,
            _paragraph(slide.title or f"Slide {slide.number}", 2800, bold=True),
            name="Title",
            sid=2,
        )
    )
    y += title_cy + 228600

    if slide.content_type == CONTENT_TABLE and slide.table:
        shapes.append(_table_shape(slide, _MARGIN_X, y, body_cx, sid=3))
        y += 370840 * len(slide.table) + 228600

    if slide.content_type == CONTENT_IMAGE and slide.images:
        image_cy = 2743200
        shapes.append(_image_shape(slide, _MARGIN_X, y, body_cx, image_cy, sid=3))
        y += image_cy + 228600

    if slide.content_type in (CONTENT_BULLETS, CONTENT_SECTION, CONTENT_TITLE):
        paragraphs = [_bullet(line, 1800) for line in slide.content]
    elif slide.content_type == CONTENT_EQUATION:
        paragraphs = [_paragraph(line, 1800, italic=True) for line in slide.content]
    else:
        paragraphs = [_paragraph(line, 1800) for line in slide.content]
    paragraphs += [_paragraph(line, 1600, italic=True) for line in slide.equations]

    if paragraphs:
        remaining = max(1000000, _SLIDE_CY - y - _MARGIN_Y)
        shapes.append(
            _text_box(
                _MARGIN_X, y, body_cx, remaining, "".join(paragraphs), name="Content", sid=4
            )
        )

    return (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        f'<p:sld xmlns:a="{_NS_A}" xmlns:r="{_NS_R}" xmlns:p="{_NS_P}">'
        "<p:cSld><p:spTree>"
        '<p:nvGrpSpPr><p:cNvPr id="1" name=""/><p:cNvGrpSpPr/><p:nvPr/></p:nvGrpSpPr>'
        '<p:grpSpPr><a:xfrm><a:off x="0" y="0"/><a:ext cx="0" cy="0"/>'
        '<a:chOff x="0" y="0"/><a:chExt cx="0" cy="0"/></a:xfrm></p:grpSpPr>'
        + "".join(shapes)
        + "</p:spTree></p:cSld>"
        "<p:clrMapOvr><a:masterClrMapping/></p:clrMapOvr></p:sld>"
    )


def _notes_xml(slide: Slide) -> str:
    """The ``ppt/notesSlides/notesSlideN.xml`` part carrying the speaker notes."""
    return (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        f'<p:notes xmlns:a="{_NS_A}" xmlns:r="{_NS_R}" xmlns:p="{_NS_P}">'
        "<p:cSld><p:spTree>"
        '<p:nvGrpSpPr><p:cNvPr id="1" name=""/><p:cNvGrpSpPr/><p:nvPr/></p:nvGrpSpPr>'
        '<p:grpSpPr><a:xfrm><a:off x="0" y="0"/><a:ext cx="0" cy="0"/>'
        '<a:chOff x="0" y="0"/><a:chExt cx="0" cy="0"/></a:xfrm></p:grpSpPr>'
        + _text_box(
            _MARGIN_X,
            _MARGIN_Y,
            _SLIDE_CX - 2 * _MARGIN_X,
            4572000,
            _paragraph(slide.notes, 1200),
            name="Notes",
            sid=2,
        )
        + "</p:spTree></p:cSld>"
        "<p:clrMapOvr><a:masterClrMapping/></p:clrMapOvr></p:notes>"
    )


_EMPTY_TREE = (
    "<p:cSld><p:spTree>"
    '<p:nvGrpSpPr><p:cNvPr id="1" name=""/><p:cNvGrpSpPr/><p:nvPr/></p:nvGrpSpPr>'
    '<p:grpSpPr><a:xfrm><a:off x="0" y="0"/><a:ext cx="0" cy="0"/>'
    '<a:chOff x="0" y="0"/><a:chExt cx="0" cy="0"/></a:xfrm></p:grpSpPr>'
    "</p:spTree></p:cSld>"
)


def _master_xml() -> str:
    return (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        f'<p:sldMaster xmlns:a="{_NS_A}" xmlns:r="{_NS_R}" xmlns:p="{_NS_P}">'
        + _EMPTY_TREE
        + '<p:clrMap bg1="lt1" tx1="dk1" bg2="lt2" tx2="dk2" accent1="accent1" '
        'accent2="accent2" accent3="accent3" accent4="accent4" accent5="accent5" '
        'accent6="accent6" hlink="hlink" folHlink="folHlink"/>'
        '<p:sldLayoutIdLst><p:sldLayoutId id="2147483649" r:id="rId1"/></p:sldLayoutIdLst>'
        "</p:sldMaster>"
    )


def _layout_xml() -> str:
    return (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        f'<p:sldLayout xmlns:a="{_NS_A}" xmlns:r="{_NS_R}" xmlns:p="{_NS_P}" '
        'type="blank" preserve="1">'
        + _EMPTY_TREE
        + "</p:sldLayout>"
    )


_THEME_XML = (
    '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
    f'<a:theme xmlns:a="{_NS_A}" name="Fayd"><a:themeElements>'
    '<a:clrScheme name="Fayd">'
    '<a:dk1><a:sysClr val="windowText" lastClr="000000"/></a:dk1>'
    '<a:lt1><a:sysClr val="window" lastClr="FFFFFF"/></a:lt1>'
    '<a:dk2><a:srgbClr val="1F1F1F"/></a:dk2><a:lt2><a:srgbClr val="EEEEEE"/></a:lt2>'
    '<a:accent1><a:srgbClr val="C8A24A"/></a:accent1>'
    '<a:accent2><a:srgbClr val="2E5A88"/></a:accent2>'
    '<a:accent3><a:srgbClr val="7A9A5B"/></a:accent3>'
    '<a:accent4><a:srgbClr val="9A5B5B"/></a:accent4>'
    '<a:accent5><a:srgbClr val="5B5B9A"/></a:accent5>'
    '<a:accent6><a:srgbClr val="9A8A5B"/></a:accent6>'
    '<a:hlink><a:srgbClr val="2E5A88"/></a:hlink>'
    '<a:folHlink><a:srgbClr val="7A5B9A"/></a:folHlink>'
    "</a:clrScheme>"
    '<a:fontScheme name="Fayd">'
    '<a:majorFont><a:latin typeface="Calibri Light"/><a:ea typeface=""/>'
    '<a:cs typeface=""/></a:majorFont>'
    '<a:minorFont><a:latin typeface="Calibri"/><a:ea typeface=""/>'
    '<a:cs typeface=""/></a:minorFont>'
    "</a:fontScheme>"
    '<a:fmtScheme name="Fayd">'
    '<a:fillStyleLst><a:solidFill><a:schemeClr val="phClr"/></a:solidFill>'
    '<a:gradFill rotWithShape="1"><a:gsLst><a:gs pos="0"><a:schemeClr val="phClr"/></a:gs>'
    '<a:gs pos="100000"><a:schemeClr val="phClr"/></a:gs></a:gsLst>'
    '<a:lin ang="5400000" scaled="0"/></a:gradFill>'
    '<a:solidFill><a:schemeClr val="phClr"/></a:solidFill></a:fillStyleLst>'
    "<a:lnStyleLst>"
    '<a:ln w="6350" cap="flat" cmpd="sng" algn="ctr">'
    '<a:solidFill><a:schemeClr val="phClr"/></a:solidFill>'
    '<a:prstDash val="solid"/></a:ln>'
    '<a:ln w="12700" cap="flat" cmpd="sng" algn="ctr">'
    '<a:solidFill><a:schemeClr val="phClr"/></a:solidFill>'
    '<a:prstDash val="solid"/></a:ln>'
    '<a:ln w="19050" cap="flat" cmpd="sng" algn="ctr">'
    '<a:solidFill><a:schemeClr val="phClr"/></a:solidFill>'
    '<a:prstDash val="solid"/></a:ln>'
    "</a:lnStyleLst>"
    "<a:effectStyleLst><a:effectStyle><a:effectLst/></a:effectStyle>"
    "<a:effectStyle><a:effectLst/></a:effectStyle>"
    "<a:effectStyle><a:effectLst/></a:effectStyle></a:effectStyleLst>"
    '<a:bgFillStyleLst><a:solidFill><a:schemeClr val="phClr"/></a:solidFill>'
    '<a:solidFill><a:schemeClr val="phClr"/></a:solidFill>'
    '<a:solidFill><a:schemeClr val="phClr"/></a:solidFill></a:bgFillStyleLst>'
    "</a:fmtScheme></a:themeElements></a:theme>"
)


def _relationships(items: Iterable[tuple[str, str, str]]) -> str:
    """A ``.rels`` part from ``(id, type, target)`` triples."""
    parts = [
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>',
        f'<Relationships xmlns="{_NS_REL}">',
    ]
    for rid, rtype, target in items:
        parts.append(
            f'<Relationship Id="{rid}" Type="{rtype}" Target="{_xml_escape(target)}"/>'
        )
    parts.append("</Relationships>")
    return "".join(parts)


def _content_types(count: int, with_notes: Iterable[int]) -> str:
    parts = [
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>',
        f'<Types xmlns="{_NS_CT}">',
        f'<Default Extension="rels" ContentType="{_CT_RELS}"/>',
        f'<Default Extension="xml" ContentType="{_CT_XML}"/>',
        f'<Override PartName="/ppt/presentation.xml" ContentType="{_CT_PRESENTATION}"/>',
        f'<Override PartName="/ppt/slideMasters/slideMaster1.xml" ContentType="{_CT_MASTER}"/>',
        f'<Override PartName="/ppt/slideLayouts/slideLayout1.xml" ContentType="{_CT_LAYOUT}"/>',
        f'<Override PartName="/ppt/theme/theme1.xml" ContentType="{_CT_THEME}"/>',
    ]
    for index in range(1, count + 1):
        parts.append(
            f'<Override PartName="/ppt/slides/slide{index}.xml" ContentType="{_CT_SLIDE}"/>'
        )
    for index in with_notes:
        parts.append(
            f'<Override PartName="/ppt/notesSlides/notesSlide{index}.xml" '
            f'ContentType="{_CT_NOTES}"/>'
        )
    parts.append("</Types>")
    return "".join(parts)


def _presentation_xml(count: int) -> str:
    slide_ids = "".join(
        f'<p:sldId id="{255 + index}" r:id="rId{index + 1}"/>' for index in range(1, count + 1)
    )
    return (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        f'<p:presentation xmlns:a="{_NS_A}" xmlns:r="{_NS_R}" xmlns:p="{_NS_P}" '
        'saveSubsetFonts="1">'
        '<p:sldMasterIdLst><p:sldMasterId id="2147483648" r:id="rId1"/></p:sldMasterIdLst>'
        f"<p:sldIdLst>{slide_ids}</p:sldIdLst>"
        f'<p:sldSz cx="{_SLIDE_CX}" cy="{_SLIDE_CY}" type="screen4x3"/>'
        '<p:notesSz cx="6858000" cy="9144000"/></p:presentation>'
    )


@dataclass(frozen=True)
class Deck:
    """The slides to write, in the order the writer must emit them."""

    title: str
    slides: tuple[Slide, ...]

    def __post_init__(self) -> None:
        if not self.title.strip():
            raise PresentationError("A deck needs a title.")
        if not self.slides:
            raise PresentationError("A deck needs at least one slide.")

    @classmethod
    def from_project(cls, project: PresentationProject, *, approved_only: bool = True) -> Deck:
        """Build a deck from a project's slides (approved ones by default)."""
        slides = project.approved_slides if approved_only else project.ordered_slides()
        return cls(title=project.title, slides=tuple(slides))


class PresentationWriter:
    """Write a :class:`Deck` to a valid ``.pptx`` with the standard library only."""

    def write(self, deck: Deck, path: str | Path) -> Path:
        """Write ``deck`` to ``path`` and return the resolved path.

        The package is assembled in memory and written atomically (a temporary
        file replaced into place), so a failure never leaves a half-written deck.
        """
        target = Path(path).expanduser()
        if target.suffix.lower() != ".pptx":
            target = target.with_suffix(".pptx")
        payload = self.render(deck)
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_suffix(target.suffix + ".tmp")
        temporary.write_bytes(payload)
        temporary.replace(target)
        return target

    def render(self, deck: Deck) -> bytes:
        """Render ``deck`` to the bytes of a ``.pptx`` package."""
        slides = list(deck.slides)
        count = len(slides)
        notes_indices = [
            index for index, slide in enumerate(slides, start=1) if slide.notes.strip()
        ]
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
            archive.writestr("[Content_Types].xml", _content_types(count, notes_indices))
            archive.writestr(
                "_rels/.rels",
                _relationships([("rId1", f"{_NS_R}/officeDocument", "ppt/presentation.xml")]),
            )
            archive.writestr("ppt/presentation.xml", _presentation_xml(count))
            archive.writestr(
                "ppt/_rels/presentation.xml.rels",
                _relationships(
                    [("rId1", f"{_NS_R}/slideMaster", "slideMasters/slideMaster1.xml")]
                    + [
                        (f"rId{index + 1}", f"{_NS_R}/slide", f"slides/slide{index}.xml")
                        for index in range(1, count + 1)
                    ]
                ),
            )
            archive.writestr("ppt/slideMasters/slideMaster1.xml", _master_xml())
            archive.writestr(
                "ppt/slideMasters/_rels/slideMaster1.xml.rels",
                _relationships(
                    [
                        ("rId1", f"{_NS_R}/slideLayout", "../slideLayouts/slideLayout1.xml"),
                        ("rId2", f"{_NS_R}/theme", "../theme/theme1.xml"),
                    ]
                ),
            )
            archive.writestr("ppt/slideLayouts/slideLayout1.xml", _layout_xml())
            archive.writestr(
                "ppt/slideLayouts/_rels/slideLayout1.xml.rels",
                _relationships(
                    [("rId1", f"{_NS_R}/slideMaster", "../slideMasters/slideMaster1.xml")]
                ),
            )
            archive.writestr("ppt/theme/theme1.xml", _THEME_XML)
            for index, slide in enumerate(slides, start=1):
                archive.writestr(f"ppt/slides/slide{index}.xml", _slide_xml(slide))
                if slide.notes.strip():
                    archive.writestr(
                        f"ppt/notesSlides/notesSlide{index}.xml", _notes_xml(slide)
                    )
                    archive.writestr(
                        f"ppt/slides/_rels/slide{index}.xml.rels",
                        _relationships(
                            [
                                (
                                    "rId1",
                                    f"{_NS_R}/notesSlide",
                                    f"../notesSlides/notesSlide{index}.xml",
                                )
                            ]
                        ),
                    )
        return buffer.getvalue()


@dataclass(frozen=True)
class ValidationResult:
    """What a ``.pptx`` validation found."""

    ok: bool
    slide_count: int
    parts: int
    message: str
    size_bytes: int = 0


def validate_pptx(path: str | Path) -> ValidationResult:
    """Re-open a written deck and confirm it is a well-formed OOXML package.

    The check is deliberately self-contained (no third-party reader): the file
    must be a ZIP, every ``.xml``/``.rels`` part must parse, and the slide count
    in ``ppt/presentation.xml`` must match the ``ppt/slides/`` parts. This is what
    the EXPORTED state means -- the file exists *and* passed validation.
    """
    target = Path(path).expanduser()
    if not target.exists():
        return ValidationResult(False, 0, 0, f"No file was written at {target}.")
    size = target.stat().st_size
    try:
        with zipfile.ZipFile(target) as archive:
            names = archive.namelist()
            for name in names:
                if name.endswith(".xml") or name.endswith(".rels"):
                    ET.fromstring(archive.read(name))
            slides = [name for name in names if name.startswith("ppt/slides/slide")]
            if "ppt/presentation.xml" not in names:
                return ValidationResult(
                    False, 0, len(names), "The deck has no presentation part.", size
                )
            slide_list = ET.fromstring(archive.read("ppt/presentation.xml")).find(
                f"{{{_NS_P}}}sldIdLst"
            )
            declared = len(slide_list) if slide_list is not None else 0
    except (zipfile.BadZipFile, ET.ParseError, OSError) as error:
        return ValidationResult(False, 0, 0, f"The deck is not a readable .pptx: {error}", size)
    if declared != len(slides):
        return ValidationResult(
            False,
            len(slides),
            len(names),
            f"The deck declares {declared} slide(s) but contains {len(slides)}.",
            size,
        )
    if not slides:
        return ValidationResult(False, 0, len(names), "The deck contains no slides.", size)
    return ValidationResult(
        True, len(slides), len(names), f"Valid deck with {len(slides)} slide(s).", size
    )




# --------------------------------------------------------------------------- #
# Deterministic text shaping (used to turn assistant prose into deck content)
# --------------------------------------------------------------------------- #

#: A leading list marker: "1.", "1)", "-", "*", "•", "1.2", "Section 3:".
_OUTLINE_ITEM_RE = re.compile(
    r"^\s*(?P<marker>(?:\d+(?:\.\d+)*[.)]?|[-*\u2022\u2013\u2014]|#+)\s+)?(?P<text>.*\S)\s*$"
)
#: A trailing slide estimate: "(4 slides)", "[3 slides]", "- 2 slides".
_SLIDE_ESTIMATE_RE = re.compile(r"[\[(]?\s*(\d{1,2})\s*slides?\s*[\])]?\s*$", re.IGNORECASE)


def _outline_line(raw: str) -> tuple[int, str, int] | None:
    """Split one outline line into ``(depth, text, estimated_slides)``.

    ``depth`` is 0 for a section heading and 1 for a subsection (an indented or
    bulleted sub-line). Returns ``None`` for a blank line.
    """
    if not raw.strip():
        return None
    depth = 1 if raw[:1].isspace() else 0
    stripped = raw.strip()
    estimate = 1
    match = _SLIDE_ESTIMATE_RE.search(stripped)
    if match:
        estimate = max(1, int(match.group(1)))
        stripped = stripped[: match.start()].rstrip(" -:\u2013\u2014")
    item = _OUTLINE_ITEM_RE.match(stripped)
    text = item.group("text") if item else stripped
    # A numbered item at depth 0 ("1. Introduction") is a section; a nested
    # item ("  1.1 Darcy's law") is a subsection.
    if depth == 0 and item and item.group("marker") and re.match(r"^\d+\.\d", item.group("marker")):
        depth = 1
    text = text.strip().rstrip(":")
    if not text:
        return None
    return depth, text, estimate


def parse_outline(text: str) -> list[OutlineSection]:
    """Parse assistant prose into outline sections (deterministic, no model).

    A non-indented line opens a section; an indented or sub-numbered line under
    it becomes one of its subsections. A trailing ``(N slides)`` sets that
    section's estimate. A leading line that ends in sentence punctuation is
    treated as the model's preamble and ignored, so only a real heading opens a
    section; when *no* line looks like a heading every non-blank line is used,
    so a plain list still parses. Returns ``[]`` when nothing parses.
    """
    parsed_lines = [line for line in (_outline_line(raw) for raw in (text or "").splitlines()) if line]
    headings = [
        (index, item) for index, item in enumerate(parsed_lines)
        if item[0] == 0 and not item[1].endswith((".", "!", "?", ";"))
    ]
    if not headings:
        headings = [(index, item) for index, item in enumerate(parsed_lines) if item[0] == 0]
    first_heading = headings[0][0] if headings else 0

    sections: list[tuple[str, list[str], int]] = []
    for index, (depth, item_text, estimate) in enumerate(parsed_lines):
        if index < first_heading:
            continue
        if depth == 0 or not sections:
            sections.append((item_text, [], estimate))
        else:
            heading, subs, est = sections[-1]
            subs.append(item_text)
            sections[-1] = (heading, subs, max(est, estimate))
    return [
        OutlineSection(heading=heading, subsections=tuple(subs), estimated_slides=est)
        for heading, subs, est in sections
    ]


def bullets_from_text(text: str, *, limit: int = 6) -> list[str]:
    """Turn an assistant answer into clean bullet lines.

    Existing bullet lines are kept as they are; a prose answer is split into
    whole sentences with :func:`core.composition.sentences`. Blank lines are
    dropped and the result is capped at ``limit`` so a slide never overflows.
    """
    # A line counts as bulleted when it opens with a list marker that is not a
    # bare number (which is more likely a heading or a figure than a bullet).
    def _is_bulleted(raw: str) -> bool:
        item = _OUTLINE_ITEM_RE.match(raw.strip())
        if item is None or not item.group("marker"):
            return False
        return re.match(r"^\d+(?:\.\d+)*[.)]?$", item.group("marker").strip()) is None

    bulleted = [line for line in (text or "").splitlines() if _is_bulleted(line)]
    if len(bulleted) >= 2:
        lines: list[str] = []
        seen_bullet = False
        for raw in (text or "").splitlines():
            stripped = raw.strip()
            if not stripped:
                continue
            item = _OUTLINE_ITEM_RE.match(stripped)
            # ``_OUTLINE_ITEM_RE`` matches every line (its marker group is
            # optional), so "has a marker" is the test, not "matched".
            marked = item is not None and bool(item.group("marker"))
            # A line before the first bullet is the answer's title line, which
            # the slide prompt asked for; it is the slide's *title*, not a bullet,
            # so repeating it here would put the same words on the slide twice.
            if not marked and not seen_bullet:
                continue
            seen_bullet = seen_bullet or marked
            value = (item.group("text") if item else stripped).strip()
            if value and not is_scaffolding(value):
                lines.append(strip_discourse_opener(value))
        return lines[:limit]
    return [strip_discourse_opener(s) for s in content_sentences(text)[:limit]]


def title_from_text(text: str, *, fallback: str = "Slide") -> str:
    """A short slide title from a chunk of text (first line, trimmed)."""
    for raw in (text or "").splitlines():
        stripped = raw.strip().lstrip("#").strip()
        if stripped:
            return stripped[:80]
    return fallback


#: The extractive fallback's own scaffolding, which is not deck content.
_META_SENTENCE_RE = re.compile(
    r"^(no web source answered|internal knowledge on file|here is what the "
    r"available sources say|i answered this|note:|there is not enough evidence)",
    re.IGNORECASE,
)


def is_scaffolding(text: str) -> bool:
    """Whether a sentence is the assistant's own scaffolding, not deck content.

    The no-model extractive fallback announces itself ("No web source answered
    the question…", "Internal knowledge on file: …"). Those sentences describe
    the *process*, so they must never become a slide bullet or heading.
    """
    return bool(_META_SENTENCE_RE.match((text or "").strip()))


def content_sentences(text: str) -> list[str]:
    """The whole sentences of ``text`` that carry deck content (scaffolding out)."""
    from core.composition import sentences

    return [
        sentence
        for sentence in sentences(strip_scaffolding_lines(text))
        if not is_scaffolding(sentence)
    ]


def strip_scaffolding_lines(text: str) -> str:
    """Drop the assistant's own label lines before the text is split into sentences.

    The no-model fallback writes its label on a line of its own
    ("Internal knowledge on file:") ahead of the content. Sentence splitting
    would glue that label onto the first content sentence, and the scaffolding
    filter would then discard both -- silently losing real deck content. So a
    label line is removed, and when a label carries content after its colon that
    content is kept on its own.
    """
    kept: list[str] = []
    for raw in (text or "").splitlines():
        line = raw.strip()
        if not line:
            kept.append("")
            continue
        if is_scaffolding(line):
            _, separator, remainder = line.partition(":")
            remainder = remainder.strip()
            if separator and remainder and not is_scaffolding(remainder):
                kept.append(remainder)
            continue
        kept.append(line)
    return "\n".join(kept)


def looks_like_prose_outline(sections: Sequence[OutlineSection]) -> bool:
    """Whether parsed sections are really prose, not an outline.

    The extractive fallback answers in paragraphs, so ``parse_outline`` reads
    every paragraph as a heading. A heading that is a whole sentence (or very
    long) is a sign of that, and the caller should derive an outline instead.
    """
    if not sections:
        return True
    for section in sections:
        heading = section.heading
        if len(heading) > 90 or heading.endswith((".", "!", "?")):
            return True
    return False


def outline_from_prose(
    text: str,
    *,
    title: str,
    subtitle: str = "",
    limit: int = 6,
) -> list[OutlineSection]:
    """Derive a small outline from prose (the no-model fallback).

    A title section is followed by one section per content sentence of the
    answer, each with a short heading taken from the sentence and the sentence
    itself as its first subsection. The assistant's own scaffolding sentences are
    dropped. It is deliberately simple and deterministic, and the caller labels
    the revision as derived, so a user can edit or replace it.
    """
    sections: list[OutlineSection] = [
        OutlineSection(heading=title.strip() or "Overview", estimated_slides=1)
    ]
    if subtitle.strip():
        sections.append(OutlineSection(heading=subtitle.strip(), estimated_slides=1))
    for sentence in content_sentences(text)[: max(1, limit)]:
        sections.append(
            OutlineSection(
                heading=topic_sentence(sentence),
                subsections=(sentence,),
                estimated_slides=1,
            )
        )
    return sections


#: The answer's own joining words, which are not the topic of the sentence.
_DISCOURSE_OPENER_RE = re.compile(
    r"^(?:first(?:ly)?|second(?:ly)?|third(?:ly)?|finally|also|next|then|"
    r"in addition|additionally|moreover|furthermore|however|therefore|thus|"
    r"for example|for instance|in summary|overall|meanwhile|besides)\s*,?\s+",
    re.IGNORECASE,
)


def dedup_key(text: str) -> str:
    """A comparison key for a bullet, so a repeat is recognised across slides.

    Case, whitespace, and a trailing stop are ignored, and a leading discourse
    connective is dropped, so "First, Casing is steel pipe." and "casing is steel
    pipe" count as the same bullet and the second is not added to the deck.
    """
    cleaned = strip_discourse_opener(text).casefold().rstrip(" .,;:!?")
    return " ".join(cleaned.split())


def strip_discourse_opener(text: str) -> str:
    """Drop a leading discourse connective from a sentence or bullet.

    "First,", "In addition,", "Finally," and friends are the *answer's* joining
    words, not its content, so a bullet that opens with one reads as the
    assistant's narration rather than as a fact about the topic. Only a known
    connective at the very start is removed; a word that merely looks like one
    inside the sentence is left alone, and text with no opener is returned
    exactly as it was (so a bullet is never re-cased by this pass).
    """
    cleaned = " ".join((text or "").split())
    stripped = _DISCOURSE_OPENER_RE.sub("", cleaned)
    if stripped == cleaned:
        return cleaned
    stripped = stripped.strip()
    if not stripped:
        return cleaned
    # Removing an opener can leave a lower-case start; a bullet reads as a phrase.
    return stripped[0].upper() + stripped[1:] if stripped[0].islower() else stripped


def topic_sentence(sentence: str, *, words: int = 6) -> str:
    """A short heading from a sentence (its first few words, no trailing stop).

    A leading discourse connective ("First,", "In addition,", "Moreover,") is the
    answer's own joining word, not the topic, so it is dropped before the heading
    is taken -- otherwise every derived section reads "In addition, ...".
    """
    cleaned = strip_discourse_opener(sentence)
    tokens = cleaned.split(" ")
    heading = " ".join(tokens[:words]).rstrip(".,;:!?")
    if not heading:
        return "Section"
    # Dropping an opener can leave the sentence starting lower case; a heading is
    # capitalised (presentation only -- the words are the assistant's own).
    return heading[0].upper() + heading[1:]


# --------------------------------------------------------------------------- #
# Slide commands
# --------------------------------------------------------------------------- #

#: The slide commands a user may type or say, each of which maps to exactly one
#: deterministic service operation. Parsing them here (rather than asking a model
#: to interpret "shorten slides 11-15") keeps the edit predictable and testable.
SLIDE_COMMANDS: tuple[str, ...] = (
    "revise",
    "shorten",
    "expand",
    "add-example",
    "add-slide",
    "move",
    "renumber",
    "regenerate",
    "approve",
    "approve-batch",
)

#: "11-15", "11–15", "11 to 15", "11..15", optionally prefixed with "slides".
_RANGE_RE = re.compile(
    r"(?:slides?\s+)?(?P<first>\d+)\s*(?:-{1,2}|\u2013|\u2014|\bto\b|\bthrough\b)\s*(?P<last>\d+)",
    re.IGNORECASE,
)
_NUMBER_RE = re.compile(r"(?:slides?\s+)?(?P<number>\d+)", re.IGNORECASE)
_MOVE_RE = re.compile(
    r"move\s+slides?\s+(?P<a>\d+)\s+(?P<where>before|after|above|below|ahead of|in front of)\s+"
    r"slides?\s+(?P<b>\d+)",
    re.IGNORECASE,
)
_ADD_RE = re.compile(
    r"add\s+(?:a\s+|an\s+)?(?P<what>.+?)\s+(?P<where>after|before|following|preceding)\s+"
    r"slides?\s+(?P<n>\d+)",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class SlideCommand:
    """One parsed slide instruction: the action and the slides it targets."""

    action: str
    slides: tuple[int, ...] = ()
    anchor: int | None = None
    before: bool = False
    text: str = ""

    def __post_init__(self) -> None:
        if self.action not in SLIDE_COMMANDS:
            raise PresentationError(
                f"Unknown slide command {self.action!r}; expected one of "
                f"{', '.join(SLIDE_COMMANDS)}."
            )

    def to_primitive(self) -> dict[str, Any]:
        return {
            "action": self.action,
            "slides": list(self.slides),
            "anchor": self.anchor,
            "before": self.before,
            "text": self.text,
        }


def parse_slide_range(text: str) -> tuple[int, int] | None:
    """Read a slide range ("7", "slides 11-15", "11 to 15") as ``(first, last)``.

    A single number is the one-slide range ``(n, n)``. Returns ``None`` when the
    text holds no number, so a caller can tell "no range here" from "the range
    1-1".
    """
    match = _RANGE_RE.search(text or "")
    if match:
        first = int(match.group("first"))
        last = int(match.group("last"))
        if first < 1 or last < 1:
            return None
        return (first, last) if first <= last else (last, first)
    single = _NUMBER_RE.search(text or "")
    if not single:
        return None
    number = int(single.group("number"))
    return (number, number) if number >= 1 else None


def _slide_numbers(text: str) -> tuple[int, ...]:
    """Every slide number the text names (a range expands to each number)."""
    span = parse_slide_range(text)
    if span is not None:
        return tuple(range(span[0], span[1] + 1))
    match = _NUMBER_RE.search(text or "")
    if not match:
        return ()
    return (int(match.group("number")),)


def parse_slide_command(text: str) -> SlideCommand | None:
    """Parse one slide instruction, or return ``None`` when it is not a command.

    Recognised (case-insensitive, English)::

        Revise slide 7.
        Shorten slides 11-15.            Expand slides 3-4.
        Add a worked example after slide 12.
        Move slide 18 before slide 17.   Move slide 3 after slide 9.
        Renumber all slides after reordering.
        Regenerate slides 21-25.
        Approve slide 2.                 Approve slides 5-8.

    The order of the checks matters: "renumber ... after reordering" contains the
    word "after" but is a renumber, not a move, so it is tested first.
    """
    raw = (text or "").strip()
    if not raw:
        return None
    low = raw.lower()

    if "renumber" in low:
        return SlideCommand(action="renumber", text=raw)

    move = _MOVE_RE.search(raw)
    if move:
        before = move.group("where").lower() in ("before", "above", "ahead of", "in front of")
        return SlideCommand(
            action="move",
            slides=(int(move.group("a")),),
            anchor=int(move.group("b")),
            before=before,
            text=raw,
        )

    if low.startswith("add") or low.startswith("insert"):
        add = _ADD_RE.search(raw)
        if add:
            what = add.group("what").strip().lower()
            action = "add-example" if "example" in what else "add-slide"
            return SlideCommand(
                action=action,
                slides=(int(add.group("n")),),
                before=add.group("where").lower() in ("before", "preceding"),
                text=raw,
            )

    if "regenerate" in low or "redo" in low:
        return SlideCommand(action="regenerate", slides=_slide_numbers(raw), text=raw)
    if "shorten" in low or "condense" in low or "trim" in low:
        return SlideCommand(action="shorten", slides=_slide_numbers(raw), text=raw)
    if "expand" in low or "lengthen" in low or "elaborate" in low:
        return SlideCommand(action="expand", slides=_slide_numbers(raw), text=raw)
    if low.startswith("approve"):
        numbers = _slide_numbers(raw)
        action = "approve-batch" if "batch" in low or len(numbers) > 1 else "approve"
        return SlideCommand(action=action, slides=numbers, text=raw)
    if "revise" in low or "rewrite" in low or "rework" in low:
        return SlideCommand(action="revise", slides=_slide_numbers(raw), text=raw)
    return None


__all__ = [
    "CONTENT_BULLETS",
    "CONTENT_EQUATION",
    "CONTENT_IMAGE",
    "CONTENT_PARAGRAPH",
    "CONTENT_SECTION",
    "CONTENT_TABLE",
    "CONTENT_TITLE",
    "CONTENT_TYPES",
    "CONTENT_TYPE_LABELS",
    "DEFAULT_SLIDE_BATCH",
    "EDITABLE_FIELDS",
    "MAX_SLIDE_BATCH",
    "PRESENTATIONS_NAMESPACE",
    "SLIDE_COMMANDS",
    "STATE_CONTENT_APPROVED",
    "STATE_DISCUSSION",
    "STATE_EXPORTED",
    "STATE_EXPORTING",
    "STATE_EXPORT_FAILED",
    "STATE_LABELS",
    "STATE_ORDER",
    "STATE_OUTLINE_DRAFT",
    "STATE_OUTLINE_REVIEW",
    "STATE_SLIDE_DRAFTING",
    "STATE_SLIDE_REVIEW",
    "Deck",
    "ExportRecord",
    "OutlineRevision",
    "OutlineSection",
    "PresentationError",
    "PresentationProject",
    "PresentationStore",
    "PresentationWriter",
    "Slide",
    "SlideCommand",
    "SlideRevision",
    "ValidationResult",
    "bullets_from_text",
    "can_transition",
    "content_sentences",
    "dedup_key",
    "default_presentation_dir",
    "is_scaffolding",
    "looks_like_prose_outline",
    "outline_from_prose",
    "parse_outline",
    "parse_slide_command",
    "parse_slide_range",
    "presentation_filename",
    "require_transition",
    "resolve_batch_size",
    "resolve_content_type",
    "slide_topic_key",
    "slugify",
    "strip_scaffolding_lines",
    "title_from_text",
    "topic_sentence",
    "validate_pptx",
]
