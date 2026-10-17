"""The Presentation Studio service: the Tk-free workflow behind the window.

This is the headless half of the "Presentation Studio" screen, and the one
engine the window and ``python -m cli presentation`` share. It drives the
deterministic model in :mod:`core.presentation` through the named workflow
states (DISCUSSION -> OUTLINE_DRAFT -> OUTLINE_REVIEW -> SLIDE_DRAFTING ->
SLIDE_REVIEW -> CONTENT_APPROVED -> EXPORTING -> EXPORTED, with EXPORT_FAILED as
a safe-retry detour) and persists every change through
:class:`core.presentation.PresentationStore`.

Two design points carry the requirements:

* **The existing assistant is reused, not replaced.** The service takes an
  ``ask`` callable -- the window passes ``ChatService.ask`` and the CLI can pass
  the same session -- so the discussion, the outline draft, and the slide drafts
  all come from the assistant the rest of the program already answers with. No
  second model, retriever, or knowledge base is created here.
* **Approval is enforced by the model, not by the caller.** Editing an approved
  slide clears its approval flag (recording the previous approved snapshot
  first), ``approve_content`` refuses until every slide is approved, and
  ``export`` refuses while any approval is outdated. Nothing here can silently
  overwrite an approved version.

The service is dependency-free and needs no display, so it is exercised directly
by the tests; the window only collects input and shows the result.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

from core.presentation import (
    CONTENT_BULLETS,
    EDITABLE_FIELDS,
    STATE_CONTENT_APPROVED,
    STATE_DISCUSSION,
    STATE_EXPORT_FAILED,
    STATE_EXPORTED,
    STATE_EXPORTING,
    STATE_LABELS,
    STATE_OUTLINE_DRAFT,
    STATE_OUTLINE_REVIEW,
    STATE_SLIDE_DRAFTING,
    STATE_SLIDE_REVIEW,
    Deck,
    ExportRecord,
    OutlineRevision,
    OutlineSection,
    PresentationError,
    PresentationProject,
    PresentationStore,
    PresentationWriter,
    Slide,
    SlideRevision,
    ValidationResult,
    bullets_from_text,
    dedup_key,
    default_presentation_dir,
    looks_like_prose_outline,
    outline_from_prose,
    parse_outline,
    parse_slide_command,
    require_transition,
    resolve_batch_size,
    resolve_content_type,
    slide_topic_key,
    title_from_text,
    validate_pptx,
)
from core.presentation_courses import (
    DEFAULT_MIN_MATCHES,
    CourseDetail,
    CourseMatch,
)
from core.presentation_layout import (
    PresentationSettings,
    prepare_deck,
    safe_filename,
    unique_export_path,
)

#: An ``ask`` callable: a question in, the assistant's answer out.
AskCallback = Callable[[str], str]
ProgressCallback = Callable[[str], None]

#: Export engines the service can use, in preference order. ``pptx`` needs the
#: optional python-pptx extra; ``stdlib`` always works.
ENGINE_PPTX = "pptx"
ENGINE_STDLIB = "stdlib"
ENGINES: tuple[str, ...] = (ENGINE_PPTX, ENGINE_STDLIB)
DEFAULT_ENGINE = ENGINE_PPTX


def _is_title_line(candidate: str, bullets: list[str]) -> bool:
    """Whether a line the assistant wrote is a *title*, not the first bullet.

    The slide prompt asks for a title line, and a model that follows it writes a
    short label. The grounded assistant answers in prose instead, so its first
    line is a whole sentence that also appears as the first bullet -- using it as
    the title would put the same sentence on the slide twice.
    """
    line = candidate.strip()
    if not line:
        return False
    if bullets and line == bullets[0].strip():
        return False
    return len(line.split()) <= 8 and not line.endswith((".", "!", "?"))


#: The prompt shapes the service sends to the assistant. They ask for the
#: structure the deterministic parsers understand, so no model-specific
#: post-processing is needed.
OUTLINE_PROMPT = (
    "{title!r}\n"
    "Propose a slide outline for a presentation titled {title!r}{audience}. "
    "List the sections as headings, with any subsections indented beneath them, "
    "and give an estimated slide count for each section."
)
SLIDE_PROMPT = (
    "{heading}\n"
    "Write 3 to 5 concise slide bullets about {heading} for a presentation on "
    "{title!r}. Give a short slide title on the first line."
)

#: Prompts for the review-time edits. Each asks for the *same* shape as
#: :data:`SLIDE_PROMPT`, so the deterministic parsers read the answer unchanged.
SHORTEN_PROMPT = (
    "{heading}\n"
    "Shorten the slide {heading!r} of the presentation {title!r} to its key points. "
    "Keep the same technical meaning and every number. Give a short slide title on "
    "the first line, then 2 to 3 concise bullets."
)
EXPAND_PROMPT = (
    "{heading}\n"
    "Expand the slide {heading!r} of the presentation {title!r} with the missing "
    "scientific or engineering detail. Give a short slide title on the first line, "
    "then 4 to 6 concise bullets."
)
EXAMPLE_PROMPT = (
    "{heading}\n"
    "Give one short worked example about {heading} for the presentation {title!r}. "
    "Give a short slide title on the first line, then the example as 2 to 4 concise "
    "bullets, naming every quantity and its units."
)

#: A drafted slide's note when the retriever returned no new material for its
#: heading, so the slide is flagged for the user instead of silently repeated.
REPEATED_HEADING_NOTE = (
    "The retriever returned no new material for this heading; "
    "only its strongest line is kept. Replace it during slide review."
)

REVISION_PROMPT = (
    "Revise this {kind} for the presentation {title!r}.\n"
    "Current content:\n{current}\n\n"
    "Requested change: {feedback}"
)


@dataclass(frozen=True)
class DiscussResult:
    """One discussion turn: the question, the answer, and the conversation."""

    question: str
    answer: str
    conversation_id: str = ""
    project_id: str = ""


@dataclass
class PresentationReport:
    """The outcome of an operation the window or CLI renders."""

    ok: bool
    project_id: str
    message: str
    path: str = ""
    slide_count: int = 0
    size_bytes: int = 0
    export_id: str = ""
    state: str = ""
    #: Slides that are approved (the ones an export would write).
    approved_slides: int = 0
    #: Slides whose approval is outdated (edited after approval).
    outdated_slides: int = 0
    #: The engine that produced the file ("pptx" or "stdlib").
    engine: str = ""
    #: The generated file's basename (the reference a download action uses).
    filename: str = ""
    #: Slide warnings (an overflowing slide, a missing image) the user must see.
    warnings: tuple[str, ...] = ()

    def summary(self) -> str:
        """A one-line, human-readable description of the outcome."""
        return self.message

    @property
    def download(self) -> dict[str, str]:
        """The reference and action a front end needs to serve the file.

        There is no web server in this desktop app, so "download" is a real,
        local action: the exact path on disk, its basename, and the folder that
        holds it (what an "open the folder" or "save a copy" action uses).
        """
        if not self.path:
            return {}
        path = Path(self.path)
        return {
            "path": str(path),
            "filename": self.filename or path.name,
            "directory": str(path.parent),
            "size_bytes": str(self.size_bytes),
        }

    def as_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "project_id": self.project_id,
            "message": self.message,
            "path": self.path,
            "slide_count": self.slide_count,
            "size_bytes": self.size_bytes,
            "export_id": self.export_id,
            "state": self.state,
            "approved_slides": self.approved_slides,
            "outdated_slides": self.outdated_slides,
            "engine": self.engine,
            "filename": self.filename,
            "warnings": list(self.warnings),
            "download": self.download,
        }


@dataclass
class PresentationStatus:
    """A snapshot of one project, ready to render in a sidebar or a report."""

    project_id: str
    title: str
    state: str
    state_label: str
    slides: int
    approved_slides: int
    outdated_slides: int
    outline_revisions: int
    approved_outline: int | None
    exports: int
    last_export: str = ""
    approval_outdated: bool = False
    estimated_slides: int = 0
    #: When the project was created and last changed (ISO-8601 strings).
    created_at: str = ""
    updated_at: str = ""
    #: The workflow step the user is on (1..8), for a dashboard column.
    step: int = 1
    #: The approved deck's download reference, when one is available.
    download: dict[str, str] = field(default_factory=dict)

    def summary(self) -> str:
        """A one-line description, including any outstanding approval."""
        note = " (approval outdated)" if self.approval_outdated else ""
        return (
            f"{self.title} [{self.state_label}]: {self.slides} slide(s), "
            f"{self.approved_slides} approved{note}"
        )

    def dashboard_row(self) -> str:
        """A fixed-width row for the dashboard list: date, counts, and status."""
        date = (self.updated_at or self.created_at or "")[:10] or "----------"
        return (
            f"{self.project_id}  {date}  "
            f"{self.slides:>3} slide(s)  {self.approved_slides:>3} approved  "
            f"{self.state_label}"
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "project_id": self.project_id,
            "title": self.title,
            "state": self.state,
            "state_label": self.state_label,
            "slides": self.slides,
            "approved_slides": self.approved_slides,
            "outdated_slides": self.outdated_slides,
            "outline_revisions": self.outline_revisions,
            "approved_outline": self.approved_outline,
            "exports": self.exports,
            "last_export": self.last_export,
            "approval_outdated": self.approval_outdated,
            "estimated_slides": self.estimated_slides,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "step": self.step,
            "download": self.download,
        }


@dataclass
class ProjectValidation:
    """The result of checking a project's structure before an export.

    The check is read-only and never mutates the project: it reports what a user
    must fix (a numbering gap, an empty slide, an unapproved slide, an outdated
    approval) so the CLI can *validate* a deck without exporting it.
    """

    ok: bool
    project_id: str
    state: str
    issues: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()
    slides: int = 0
    approved_slides: int = 0

    def summary(self) -> str:
        """A one-line description of the verdict."""
        if self.ok:
            return (
                f"{self.project_id} is valid: {self.slides} slide(s), "
                f"{self.approved_slides} approved, ready to export."
            )
        return f"{self.project_id} is not ready: " + "; ".join(self.issues)

    def as_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "project_id": self.project_id,
            "state": self.state,
            "slides": self.slides,
            "approved_slides": self.approved_slides,
            "issues": list(self.issues),
            "warnings": list(self.warnings),
        }


@dataclass
class SlideCommandResult:
    """The outcome of one parsed slide command (what the review box reports)."""

    ok: bool
    action: str
    message: str
    state: str = ""
    slide_id: str = ""

    def summary(self) -> str:
        """A one-line description of the outcome."""
        return self.message

    def as_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "action": self.action,
            "message": self.message,
            "state": self.state,
            "slide_id": self.slide_id,
        }


#: The workflow step each state corresponds to (for a dashboard's "step" column).
_WORKFLOW_STEPS: dict[str, int] = {
    STATE_DISCUSSION: 1,
    STATE_OUTLINE_DRAFT: 2,
    STATE_OUTLINE_REVIEW: 3,
    STATE_SLIDE_DRAFTING: 5,
    STATE_SLIDE_REVIEW: 6,
    STATE_CONTENT_APPROVED: 7,
    STATE_EXPORTING: 8,
    STATE_EXPORTED: 8,
    STATE_EXPORT_FAILED: 8,
}


def _workflow_step(state: str) -> int:
    """The 1..8 workflow step a project's state corresponds to."""
    return _WORKFLOW_STEPS.get(state, 1)


def render_slide_text(slide: Slide) -> str:
    """A slide's content as plain text (used to prompt a revision)."""
    lines = [slide.title]
    lines.extend(slide.content)
    lines.extend(slide.equations)
    return "\n".join(line for line in lines if line)


@dataclass(frozen=True)
class SlideSpec:
    """The content for one new slide (used for single and batch drafting)."""

    title: str = ""
    content: Sequence[str] = ()
    content_type: str = CONTENT_BULLETS
    table: Sequence[Sequence[str]] = ()
    equations: Sequence[str] = ()
    images: Sequence[str] = ()
    sources: Sequence[str] = ()
    notes: str = ""
    section: str = ""
    subsection: str = ""
    #: An optional layout hint (e.g. "chart", "two_column"); empty infers it.
    layout: str = ""
    approved: bool = False


class PresentationService:
    """Build, review, approve, and export a presentation, one state at a time."""

    def __init__(
        self,
        *,
        store: PresentationStore | None = None,
        root: str | Path | None = None,
        ask: AskCallback | None = None,
        output_dir: str | Path | None = None,
        writer: PresentationWriter | None = None,
        on_progress: ProgressCallback | None = None,
        batch_size: int | None = None,
    ) -> None:
        self._store = store if store is not None else PresentationStore(root)
        self._ask = ask
        self._author: AskCallback | None = None
        self._writer = writer if writer is not None else PresentationWriter()
        self._output_dir = Path(output_dir) if output_dir is not None else None
        self._on_progress = on_progress
        # How many slides one drafting batch adds. Five by default, so the first
        # review pass is small; the user may raise it (ten is common).
        self._batch_size = resolve_batch_size(batch_size)

    # ------------------------------------------------------------ properties

    @property
    def store(self) -> PresentationStore:
        """The persistence adapter (a JSON namespace, shared with the CLI)."""
        return self._store

    @property
    def has_assistant(self) -> bool:
        """Whether a discussion/assistant callback is wired in."""
        return self._ask is not None

    @property
    def output_dir(self) -> Path:
        """Where exports are written when no explicit path is given."""
        return self._output_dir if self._output_dir is not None else default_presentation_dir()

    @property
    def batch_size(self) -> int:
        """How many slides one drafting batch adds unless a call overrides it."""
        return self._batch_size

    def set_batch_size(self, size: int | None) -> int:
        """Change the drafting batch size (returns the validated value)."""
        self._batch_size = resolve_batch_size(size)
        return self._batch_size

    def set_ask(self, ask: AskCallback | None) -> None:
        """Wire (or clear) the assistant callback the discussion uses."""
        self._ask = ask

    def set_author(self, author: AskCallback | None) -> None:
        """Wire (or clear) the Studio's own authoring model.

        When set, drafting an *outline* or *slide content* uses this model instead
        of the grounded assistant: those two jobs want generated, structured prose,
        not a cited extract. The discussion itself still uses the assistant, so the
        user keeps talking to the same grounded brain it always did.
        """
        self._author = author

    @property
    def author(self) -> AskCallback | None:
        """The authoring model's ``ask`` callback, or ``None`` when none is wired."""
        return self._author

    @property
    def has_author(self) -> bool:
        """Whether a dedicated authoring model is wired in."""
        return self._author is not None

    def set_progress(self, on_progress: ProgressCallback | None) -> None:
        """Wire (or clear) the progress callback long operations report to."""
        self._on_progress = on_progress

    # ----------------------------------------------------------- project CRUD

    def new_project(
        self,
        title: str,
        *,
        subtitle: str = "",
        audience: str = "",
        language: str = "en",
    ) -> PresentationProject:
        """Create and persist an independent project in the DISCUSSION state."""
        project = PresentationProject(
            project_id=self._next_project_id(),
            title=title.strip(),
            subtitle=subtitle.strip(),
            audience=audience.strip(),
            language=language or "en",
        )
        self._store.save(project)
        return project

    def get(self, project_id: str) -> PresentationProject:
        """Load a project; raises when it does not exist.

        A project found mid-export (state ``EXPORTING``) was interrupted between
        the state being written and the deck being finished -- a crash or a kill
        during the write. ``EXPORTING`` has no outgoing edge except to
        ``EXPORTED``/``EXPORT_FAILED``, so without recovery the project would be
        stranded: no export, edit, or reopen would ever be accepted again. The
        approved content is still intact, so it is returned to
        ``CONTENT_APPROVED`` and the export may simply be run again.
        """
        project = self._store.load(project_id)
        if project is None:
            raise PresentationError(f"No presentation project {project_id!r}.")
        if project.state == STATE_EXPORTING:
            project.state = STATE_CONTENT_APPROVED
            self._save(project)
        return project

    def list_projects(self) -> list[PresentationProject]:
        """Every stored project, newest first by update time."""
        projects = self._store.list_projects()
        projects.sort(key=lambda item: item.updated_at, reverse=True)
        return projects

    def list_ids(self) -> list[str]:
        """Every stored project id, sorted."""
        return self._store.list_ids()

    def delete(self, project_id: str) -> bool:
        """Remove a project; returns whether a file was removed."""
        return self._store.delete(project_id)

    def status(self, project_id: str) -> PresentationStatus:
        """A renderable snapshot of one project."""
        project = self.get(project_id)
        last = next((item for item in reversed(project.exports) if item.ok), None)
        download: dict[str, str] = {}
        if last is not None and last.path and Path(last.path).exists():
            path = Path(last.path)
            download = {
                "path": str(path),
                "filename": path.name,
                "directory": str(path.parent),
                "size_bytes": str(last.size_bytes),
            }
        return PresentationStatus(
            project_id=project.project_id,
            title=project.title,
            state=project.state,
            state_label=STATE_LABELS.get(project.state, project.state),
            slides=len(project.slides),
            approved_slides=len(project.approved_slides),
            outdated_slides=len(project.outdated_slides),
            outline_revisions=len(project.outline_revisions),
            approved_outline=project.approved_outline_revision,
            exports=len(project.exports),
            last_export=last.path if last is not None else "",
            approval_outdated=project.approval_outdated,
            estimated_slides=project.total_estimated_slides,
            created_at=project.created_at,
            updated_at=project.updated_at,
            step=_workflow_step(project.state),
            download=download,
        )

    def rename_project(self, project_id: str, title: str) -> PresentationProject:
        """Rename a project (its id, slides, and exports are untouched)."""
        project = self.get(project_id)
        new_title = title.strip()
        if not new_title:
            raise PresentationError("A project needs a non-empty title.")
        project.title = new_title
        return self._save(project)

    def duplicate_project(self, project_id: str, *, title: str = "") -> PresentationProject:
        """Copy a project under a fresh id, preserving its slides and settings.

        The copy starts in the state the original is in, so a duplicated deck can
        be edited and re-exported without touching the original's exports. The
        new project never carries the original's export records (each deck owns
        its own history) and its slides are re-keyed so the two never collide.
        """
        source = self.get(project_id)
        clone = PresentationProject.from_primitive(source.to_primitive())
        clone.project_id = self._next_project_id()
        clone.title = (title.strip() or f"{source.title} (copy)")
        clone.exports = []
        clone.conversation_ids = []
        clone.slides = [replace(slide) for slide in source.slides]
        return self._save(clone)

    def validate_project(
        self, project_id: str, *, settings: PresentationSettings | None = None
    ) -> ProjectValidation:
        """Check a project's structure and numbering without exporting it.

        Reports the *blocking* issues (an empty slide, an unapproved slide, a
        numbering gap, an outdated approval) and the *advisory* warnings (an
        over-full slide, a missing image) an export would produce. Never mutates
        the project, so the CLI can validate a deck in any state.
        """
        project = self.get(project_id)
        issues: list[str] = []
        warnings: list[str] = []

        if not project.slides:
            issues.append("the project has no slides")
        numbers = [slide.number for slide in project.ordered_slides()]
        expected = list(range(1, len(numbers) + 1))
        if numbers != expected:
            issues.append(
                f"the slide numbers are not dense 1..n (found {numbers or 'none'})"
            )
        empty = [slide.number for slide in project.ordered_slides() if slide.is_empty]
        if empty:
            issues.append(
                "slide(s) " + ", ".join(str(number) for number in empty) + " are empty"
            )
        unapproved = [
            slide.number for slide in project.ordered_slides() if not slide.approved
        ]
        if unapproved and project.state in (STATE_CONTENT_APPROVED, STATE_EXPORTED):
            issues.append(
                "slide(s) "
                + ", ".join(str(number) for number in unapproved)
                + " are not approved"
            )
        if project.approval_outdated:
            issues.append("the content approval is outdated (a slide changed)")

        allowed, reason = self.can_export(project_id)
        if not allowed:
            issues.append(reason)

        if project.slides:
            try:
                warnings.extend(self.export_warnings(project_id, settings=settings))
            except PresentationError:
                pass

        return ProjectValidation(
            ok=not issues,
            project_id=project.project_id,
            state=project.state,
            issues=tuple(issues),
            warnings=tuple(warnings),
            slides=len(project.slides),
            approved_slides=len(project.approved_slides),
        )

    def _next_project_id(self) -> str:
        """A fresh project id that does not collide with a stored one."""
        existing = set(self._store.list_ids())
        index = len(existing) + 1
        while True:
            candidate = f"PRS-{index:03d}"
            if candidate not in existing:
                return candidate
            index += 1

    def _save(self, project: PresentationProject) -> PresentationProject:
        """Stamp and persist a project, returning it for chaining."""
        project.touch()
        self._store.save(project)
        return project

    def _progress(self, message: str) -> None:
        if self._on_progress is not None:
            self._on_progress(message)

    def _drafting_callback(self, what: str) -> AskCallback:
        """The callback that drafts ``what``: the authoring model, else the assistant.

        Outline and slide-content drafting prefer the Studio's own authoring model
        (:mod:`core.presentation_model`): those jobs want generated, structured
        prose, and a model built for instruction following does that better than
        the grounded assistant, which answers with cited extracts. The discussion
        and the review-time instructions keep using the assistant. When no
        authoring model is wired the assistant answers, so the feature works with
        no optional extra installed.
        """
        if self._author is not None:
            return self._author
        return self._require_assistant(what)

    def _require_assistant(self, what: str) -> AskCallback:
        if self._ask is None:
            raise PresentationError(
                f"No assistant is connected, so the {what} cannot be drafted. "
                "Open Presentation Studio from the chat window, or supply the "
                "text directly."
            )
        return self._ask

    # ------------------------------------------------------------- discussion

    def discuss(
        self,
        project_id: str,
        question: str,
        *,
        conversation_id: str = "",
    ) -> DiscussResult:
        """Ask the existing assistant and record the conversation reference.

        The answer is returned as-is (the caller decides how to show it); the
        project stores only the *reference* to the conversation, never a copy of
        the transcript, so the one conversation store stays authoritative.
        """
        ask = self._require_assistant("discussion")
        project = self.get(project_id)
        answer = ask(question)
        if conversation_id and conversation_id not in project.conversation_ids:
            project.conversation_ids.append(conversation_id)
        self._save(project)
        return DiscussResult(
            question=question,
            answer=answer,
            conversation_id=conversation_id,
            project_id=project.project_id,
        )

    def attach_conversation(self, project_id: str, conversation_id: str) -> PresentationProject:
        """Record a conversation id the project was discussed in."""
        project = self.get(project_id)
        if conversation_id and conversation_id not in project.conversation_ids:
            project.conversation_ids.append(conversation_id)
        return self._save(project)

    # ----------------------------------------------------------- the outline

    def draft_outline(
        self,
        project_id: str,
        *,
        text: str | None = None,
        note: str = "",
    ) -> OutlineRevision:
        """Propose an outline (from the assistant, or from supplied text).

        Moves DISCUSSION -> OUTLINE_DRAFT and appends an outline revision. The
        caller may pass ``text`` directly (the CLI and the tests do), which keeps
        the whole path exercisable with no model.
        """
        project = self.get(project_id)
        if project.state == STATE_DISCUSSION:
            require_transition(project.state, STATE_OUTLINE_DRAFT)
            project.state = STATE_OUTLINE_DRAFT
        elif project.state not in (STATE_OUTLINE_DRAFT, STATE_OUTLINE_REVIEW):
            raise PresentationError(
                "The outline can only be drafted from the discussion or review stage."
            )
        source = text
        if source is None:
            audience = f" for {project.audience}" if project.audience else ""
            prompt = OUTLINE_PROMPT.format(title=project.title, audience=audience)
            self._progress("Asking the assistant for an outline...")
            source = self._drafting_callback("outline")(prompt)
        sections = parse_outline(source)
        derived = False
        if looks_like_prose_outline(sections):
            # The answer was prose (the extractive fallback, or a model that
            # ignored the requested shape), so derive a simple outline from it
            # and say so in the note -- the user reviews and edits it either way.
            sections = outline_from_prose(
                source, title=project.title, subtitle=project.subtitle
            )
            derived = True
        if not sections:
            raise PresentationError(
                "The outline could not be read; no section headings were found."
            )
        revision = self._next_outline_revision(
            project, sections, note or ("derived from the assistant's answer" if derived else "")
        )
        self._save(project)
        return revision

    def _next_outline_revision(
        self,
        project: PresentationProject,
        sections: Iterable[OutlineSection],
        note: str,
    ) -> OutlineRevision:
        """Append a new outline revision to ``project`` and return it."""
        revision = OutlineRevision(
            revision=(project.outline_revisions[-1].revision + 1)
            if project.outline_revisions
            else 1,
            sections=tuple(sections),
            note=note,
        )
        project.outline_revisions.append(revision)
        return revision

    def review_outline(self, project_id: str) -> PresentationProject:
        """Move OUTLINE_DRAFT -> OUTLINE_REVIEW (the user is now reviewing)."""
        project = self.get(project_id)
        require_transition(project.state, STATE_OUTLINE_REVIEW)
        project.state = STATE_OUTLINE_REVIEW
        return self._save(project)

    def edit_outline(
        self,
        project_id: str,
        sections: Iterable[OutlineSection],
        *,
        note: str = "edited by the user",
    ) -> OutlineRevision:
        """Replace the outline with the user's own sections (a new revision).

        The previous revision is kept, so an edit is always reversible; approving
        an earlier revision is how a user restores it.
        """
        project = self.get(project_id)
        if project.state not in (STATE_OUTLINE_DRAFT, STATE_OUTLINE_REVIEW):
            raise PresentationError("The outline can only be edited during outline review.")
        items = tuple(sections)
        if not items:
            raise PresentationError("An outline needs at least one section.")
        revision = self._next_outline_revision(project, items, note)
        if project.state == STATE_OUTLINE_DRAFT:
            project.state = STATE_OUTLINE_REVIEW
        self._save(project)
        return revision

    def request_outline_revision(self, project_id: str, feedback: str) -> OutlineRevision:
        """Ask the assistant to revise the outline from the user's feedback."""
        ask = self._require_assistant("outline revision")
        project = self.get(project_id)
        current = self.render_outline(project.latest_outline)
        prompt = REVISION_PROMPT.format(
            kind="outline", title=project.title, current=current, feedback=feedback
        )
        self._progress("Asking the assistant to revise the outline...")
        sections = parse_outline(ask(prompt))
        if not sections:
            raise PresentationError("The revised outline could not be read.")
        revision = self._next_outline_revision(project, sections, f"revision requested: {feedback}")
        if project.state == STATE_OUTLINE_DRAFT:
            project.state = STATE_OUTLINE_REVIEW
        self._save(project)
        return revision

    def approve_outline(self, project_id: str, revision: int | None = None) -> PresentationProject:
        """Approve an outline revision and open slide drafting.

        OUTLINE_REVIEW -> SLIDE_DRAFTING. Approving an *earlier* revision is the
        supported way to restore a previous outline.
        """
        project = self.get(project_id)
        if project.state not in (STATE_OUTLINE_REVIEW, STATE_OUTLINE_DRAFT):
            raise PresentationError("The outline can only be approved during outline review.")
        target = project.latest_outline if revision is None else project.outline_revision(revision)
        if target is None:
            raise PresentationError("There is no outline to approve yet.")
        project.approved_outline_revision = target.revision
        if project.state == STATE_OUTLINE_DRAFT:
            require_transition(project.state, STATE_OUTLINE_REVIEW)
            project.state = STATE_OUTLINE_REVIEW
        require_transition(project.state, STATE_SLIDE_DRAFTING)
        project.state = STATE_SLIDE_DRAFTING
        return self._save(project)

    @staticmethod
    def render_outline(revision: OutlineRevision | None) -> str:
        """Render an outline revision back to text (for a revision prompt)."""
        if revision is None:
            return "(none yet)"
        lines: list[str] = []
        for section in revision.sections:
            lines.append(f"{section.heading} ({section.estimated_slides} slides)")
            lines.extend(f"  {sub}" for sub in section.subsections)
        return "\n".join(lines)

    # ------------------------------------------------------------ the slides

    def _new_slide(self, project: PresentationProject, spec: SlideSpec) -> Slide:
        """Build one slide from a spec, assigning a stable id and a number."""
        slide = Slide(
            slide_id=project.next_slide_id(),
            order=len(project.slides) + 1,
            title=spec.title,
            content=list(spec.content),
            content_type=spec.content_type,
            table=[list(row) for row in spec.table],
            equations=list(spec.equations),
            images=list(spec.images),
            sources=list(spec.sources),
            notes=spec.notes,
            section=spec.section,
            subsection=spec.subsection,
            layout=spec.layout,
            approved=spec.approved,
        )
        if slide.approved:
            slide.approved_snapshot = slide.content_snapshot()
        return slide

    def add_slide(self, project_id: str, spec: SlideSpec) -> Slide:
        """Append one slide."""
        project = self.get(project_id)
        if project.state not in (STATE_SLIDE_DRAFTING, STATE_SLIDE_REVIEW):
            raise PresentationError("Slides can only be added while drafting or reviewing slides.")
        slide = self._new_slide(project, spec)
        project.slides.append(slide)
        project.renumber()
        self._save(project)
        return slide

    def add_slides(self, project_id: str, specs: Iterable[SlideSpec]) -> list[Slide]:
        """Append a batch of slides (batch-level editing, one save).

        Each slide is appended as it is built, so ``next_slide_id`` sees the
        slides already added and a batch can never hand out a duplicate id.
        """
        project = self.get(project_id)
        if project.state not in (STATE_SLIDE_DRAFTING, STATE_SLIDE_REVIEW):
            raise PresentationError("Slides can only be added while drafting or reviewing slides.")
        added: list[Slide] = []
        for spec in specs:
            slide = self._new_slide(project, spec)
            project.slides.append(slide)
            added.append(slide)
        project.renumber()
        self._save(project)
        return added

    def _outline_topics(self, project: PresentationProject) -> list[tuple[str, str]]:
        """The ``(section, subsection)`` topics the approved outline implies.

        A section with subsections yields one topic per subsection (so a long
        section is developed piece by piece); a section without any yields one
        topic for the section itself. The outline's order is preserved exactly,
        so no topic is added, dropped, or merged.
        """
        outline = project.outline
        topics: list[tuple[str, str]] = []
        for section in outline.sections if outline is not None else ():
            if section.subsections:
                topics.extend((section.heading, sub) for sub in section.subsections)
            else:
                topics.append((section.heading, ""))
        return topics

    def draft_slides_from_outline(
        self,
        project_id: str,
        *,
        heading: str | None = None,
        batch: int | None = None,
    ) -> list[Slide]:
        """Ask the assistant for slide content, one manageable batch at a time.

        With no ``heading`` every not-yet-drafted outline topic is drafted; with
        a ``heading`` only that section is. ``batch`` caps how many slides one
        call may add -- it defaults to :attr:`batch_size` (5), so a long deck is
        developed in reviewable pieces rather than in one long response. Every
        drafted slide keeps the outline heading it came from (its
        ``section``/``subsection``), so the topic sequence is preserved.
        """
        ask = self._drafting_callback("slide content")
        project = self.get(project_id)
        if project.state not in (STATE_SLIDE_DRAFTING, STATE_SLIDE_REVIEW):
            raise PresentationError("Slides can only be drafted after the outline is approved.")
        topics = self._outline_topics(project)
        if heading is not None:
            topics = [topic for topic in topics if topic[0] == heading]
            if not topics:
                raise PresentationError(f"The outline has no section {heading!r}.")
        limit = resolve_batch_size(batch, default=self._batch_size)
        drafted = {slide.topic for slide in project.slides if slide.topic}
        # Bullets already used on another slide, so a new slide does not repeat a
        # sentence the deck has already shown (the retriever can return the same
        # passages for two related headings).
        used: set[str] = {
            dedup_key(line)
            for slide in project.slides
            for line in slide.content
        }
        specs: list[SlideSpec] = []
        for section, subsection in topics:
            topic = slide_topic_key(section, subsection)
            if heading is None and topic in drafted:
                continue
            label = subsection or section
            self._progress(f"Drafting slides for {label!r}...")
            prompt = SLIDE_PROMPT.format(title=project.title, heading=label)
            answer = ask(prompt)
            drafted_bullets = bullets_from_text(answer)
            bullets = [line for line in drafted_bullets if dedup_key(line) not in used]
            repeated = False
            if not bullets and drafted_bullets:
                # Every bullet repeated an earlier slide (the retriever can return
                # the same passages for two related headings). A blank slide is
                # worse than a flagged one, so the topic keeps its single
                # strongest line and the note says what happened -- slide review is
                # where the user replaces it.
                bullets = drafted_bullets[:1]
                repeated = True
            used.update(dedup_key(line) for line in bullets)
            # A grounded assistant answers in prose, so its first line is a
            # sentence, not the requested title. The outline heading is the title
            # in that case -- taking the first sentence as the title too would
            # repeat it verbatim as the slide's first bullet.
            explicit = title_from_text(answer, fallback="")
            if explicit and _is_title_line(explicit, bullets):
                title = explicit
            else:
                title = label
            specs.append(
                SlideSpec(
                    title=title,
                    content=bullets,
                    sources=(label,),
                    notes=REPEATED_HEADING_NOTE if repeated else "",
                    section=section,
                    subsection=subsection,
                )
            )
            if len(specs) >= limit:
                break
        if not specs:
            return []
        return self.add_slides(project_id, specs)

    def review_slides(self, project_id: str) -> PresentationProject:
        """Move SLIDE_DRAFTING -> SLIDE_REVIEW (the user reviews the slides)."""
        project = self.get(project_id)
        require_transition(project.state, STATE_SLIDE_REVIEW)
        project.state = STATE_SLIDE_REVIEW
        return self._save(project)

    def edit_slide(self, project_id: str, slide_id: str, **fields: Any) -> Slide:
        """Edit one slide's content.

        Editing clears the slide's approval (after recording the approved
        snapshot), which marks the project's approval outdated -- so an approved
        version is never overwritten silently and the export gate re-arms.
        """
        project = self.get(project_id)
        self._require_editable(project)
        return self._apply_edit(project, project.slide(slide_id), fields)

    def edit_slides(
        self,
        project_id: str,
        edits: Iterable[tuple[str, dict[str, Any]]],
    ) -> list[Slide]:
        """Edit several slides in one batch (one revision pass, one save)."""
        project = self.get(project_id)
        self._require_editable(project)
        changed = [
            self._apply_edit(project, project.slide(slide_id), fields, save=False)
            for slide_id, fields in edits
        ]
        self._save(project)
        return changed

    @staticmethod
    def _require_editable(project: PresentationProject) -> None:
        if project.state not in (
            STATE_SLIDE_DRAFTING,
            STATE_SLIDE_REVIEW,
            STATE_CONTENT_APPROVED,
            STATE_EXPORTED,
            STATE_EXPORT_FAILED,
        ):
            raise PresentationError(
                "Slide content can only be edited while drafting, reviewing, or after "
                "reopening approved content."
            )

    def _reopen_for_edit(self, project: PresentationProject) -> None:
        """Reopen approved/exported content before an edit changes it.

        A change after approval is never silent: the project returns to
        SLIDE_REVIEW, the content approval is cleared, and the export gate then
        requires a fresh approval. This is what makes "if the content changes
        after approval, mark the affected approval state as outdated" true.
        """
        if project.state in (STATE_CONTENT_APPROVED, STATE_EXPORTED, STATE_EXPORT_FAILED):
            project.state = STATE_SLIDE_REVIEW
            project.content_approved = False

    def _apply_edit(
        self,
        project: PresentationProject,
        slide: Slide,
        fields: dict[str, Any],
        *,
        save: bool = True,
    ) -> Slide:
        """Apply a validated content edit to one slide and record the change."""
        unknown = set(fields) - set(EDITABLE_FIELDS)
        if unknown:
            raise PresentationError(
                f"Cannot edit {', '.join(sorted(unknown))}; editable fields are "
                f"{', '.join(EDITABLE_FIELDS)}."
            )
        if not fields:
            return slide
        self._reopen_for_edit(project)
        # Snapshot the *approved* content before it is replaced, so a later edit
        # can always be undone back to what the user signed off on.
        if slide.approved:
            slide.approved_snapshot = slide.content_snapshot()
            project.record_revision(slide, reason="approved content before edit")
        slide.approved = False
        for name, value in fields.items():
            if name == "content_type":
                value = resolve_content_type(str(value))
            elif name in ("content", "equations", "images", "sources"):
                value = list(value)
            elif name == "table":
                value = [list(row) for row in value]
            setattr(slide, name, value)
        # Re-run the model's own normalisation over the edited fields, so an edit
        # cannot smuggle in an unnormalised value the writer would mis-render.
        normalized = Slide(
            slide_id=slide.slide_id,
            order=slide.order,
            title=slide.title,
            content=slide.content,
            content_type=slide.content_type,
            table=slide.table,
            equations=slide.equations,
            images=slide.images,
            sources=slide.sources,
            notes=slide.notes,
            section=slide.section,
            subsection=slide.subsection,
        )
        for name in EDITABLE_FIELDS:
            setattr(slide, name, getattr(normalized, name))
        slide.updated_at = normalized.updated_at
        project.record_revision(slide, reason="edited")
        if save:
            self._save(project)
        return slide

    def reorder_slide(self, project_id: str, slide_id: str, position: int) -> PresentationProject:
        """Move a slide to ``position`` (1-based), keeping its identity.

        Only the displayed ``order`` changes; the slide's stable id, content, and
        approval all travel with it, which is what lets a reorder never lose a
        slide's identity. Reordering approved content reopens it, so the order
        the user finally approves is the order that is exported.
        """
        project = self.get(project_id)
        self._require_editable(project)
        self._reopen_for_edit(project)
        ordered = project.ordered_slides()
        slide = project.slide(slide_id)
        target = max(1, min(position, len(ordered)))
        ordered.remove(slide)
        ordered.insert(target - 1, slide)
        for index, item in enumerate(ordered, start=1):
            item.order = index
        self._save(project)
        return project

    def remove_slide(self, project_id: str, slide_id: str) -> PresentationProject:
        """Remove a slide; the remaining slides are renumbered."""
        project = self.get(project_id)
        self._require_editable(project)
        self._reopen_for_edit(project)
        project.slides.remove(project.slide(slide_id))
        project.renumber()
        self._save(project)
        return project

    def approve_slide(self, project_id: str, slide_id: str) -> Slide:
        """Approve one slide's content and remember the approved snapshot."""
        project = self.get(project_id)
        slide = project.slide(slide_id)
        if slide.is_empty:
            raise PresentationError("An empty slide cannot be approved.")
        slide.approved = True
        slide.approved_snapshot = slide.content_snapshot()
        project.record_revision(slide, reason="approved")
        self._save(project)
        return slide

    def unapprove_slide(self, project_id: str, slide_id: str) -> Slide:
        """Withdraw a slide's approval (the export gate re-arms)."""
        project = self.get(project_id)
        slide = project.slide(slide_id)
        slide.approved = False
        project.record_revision(slide, reason="approval withdrawn")
        self._save(project)
        return slide

    def approve_slides(self, project_id: str) -> PresentationProject:
        """Approve every slide at once (batch approval).

        An empty slide is refused *by number* rather than skipped: silently
        leaving one unapproved made the content gate fail later with a message
        that did not say which slide was at fault, which is exactly how the
        workflow appeared to break between reviewing the slides and exporting.
        """
        project = self.get(project_id)
        self._require_editable(project)
        self._reopen_for_edit(project)
        empty = [slide.number for slide in project.ordered_slides() if slide.is_empty]
        if empty:
            raise PresentationError(
                f"Slide(s) {', '.join(str(number) for number in empty)} are empty; "
                "fill or remove them before approving."
            )
        for slide in project.ordered_slides():
            slide.approved = True
            slide.approved_snapshot = slide.content_snapshot()
            project.record_revision(slide, reason="approved (batch)")
        self._save(project)
        return project

    def approve_slides_in_range(
        self, project_id: str, first: int, last: int | None = None
    ) -> list[Slide]:
        """Approve the slides numbered ``first``..``last`` (batch, one save).

        A range is clamped to the slides that exist, so approving 5-12 on a
        ten-slide deck approves 5-10 rather than failing.
        """
        project = self.get(project_id)
        self._require_editable(project)
        targets = project.slides_in_range(first, last)
        if not targets:
            raise PresentationError(f"There are no slides in the range {first}-{last or first}.")
        empty = [slide.number for slide in targets if slide.is_empty]
        if empty:
            raise PresentationError(
                f"Slide(s) {', '.join(str(number) for number in empty)} are empty; "
                "fill or remove them before approving."
            )
        for slide in targets:
            slide.approved = True
            slide.approved_snapshot = slide.content_snapshot()
            project.record_revision(slide, reason="approved (range)")
        self._save(project)
        return targets

    def insert_slide(
        self, project_id: str, spec: SlideSpec, *, position: int | None = None
    ) -> Slide:
        """Insert one slide at ``position`` (1-based; appended when omitted).

        Inserting renumbers the slides after it, which is what "add a slide after
        slide 12" means; the new slide gets a fresh stable id, never a reused one.
        """
        project = self.get(project_id)
        self._require_editable(project)
        self._reopen_for_edit(project)
        slide = self._new_slide(project, spec)
        project.insert_slide(slide, position)
        self._save(project)
        return slide

    def move_slide(
        self,
        project_id: str,
        slide_id: str,
        *,
        before: int | None = None,
        after: int | None = None,
    ) -> PresentationProject:
        """Move a slide to sit before or after another slide (by number).

        The move is expressed against another slide's number rather than a raw
        position, so "move slide 18 before slide 17" stays meaningful while the
        slides are being renumbered. Reordering approved content reopens it, so
        the order the user finally approves is the order that is exported.
        """
        project = self.get(project_id)
        self._require_editable(project)
        if (before is None) == (after is None):
            raise PresentationError("A move needs exactly one of before/after.")
        anchor = project.slide_by_number(before if before is not None else after or 0)
        moving = project.slide(slide_id)
        self._reopen_for_edit(project)
        ordered = project.ordered_slides()
        ordered.remove(moving)
        anchor_index = ordered.index(anchor)
        target = anchor_index if before is not None else anchor_index + 1
        ordered.insert(target, moving)
        for index, item in enumerate(ordered, start=1):
            item.order = index
        self._save(project)
        return project

    def renumber_slides(self, project_id: str) -> PresentationProject:
        """Renumber every slide 1..n in display order (the numbering is dense)."""
        project = self.get(project_id)
        self._require_editable(project)
        self._reopen_for_edit(project)
        project.renumber()
        self._save(project)
        return project

    def _slides_for_numbers(
        self, project: PresentationProject, numbers: Iterable[int]
    ) -> list[Slide]:
        """Resolve slide numbers to slides, raising for an unknown one."""
        wanted = list(numbers)
        if not wanted:
            raise PresentationError("Name at least one slide by its number.")
        slides = [project.slide_by_number(number) for number in wanted]
        return slides

    def _redraft_slide(
        self,
        project: PresentationProject,
        slide: Slide,
        answer: str,
        *,
        fallback_title: str,
    ) -> None:
        """Apply an assistant answer to an existing slide (no save).

        The slide keeps its stable id, number, and outline heading; only its
        title and bullets change, and its approval is cleared like any edit.
        """
        bullets = bullets_from_text(answer)
        explicit = title_from_text(answer, fallback="")
        title = explicit if explicit and _is_title_line(explicit, bullets) else fallback_title
        self._apply_edit(
            project,
            slide,
            {"title": title, "content": bullets},
            save=False,
        )

    def regenerate_slides(
        self,
        project_id: str,
        numbers: Iterable[int],
        *,
        instruction: str | None = None,
    ) -> list[Slide]:
        """Re-draft the named slides, keeping their number, id, and heading.

        With no ``instruction`` the assistant is asked for the topic again (the
        same prompt the first draft used); with one, it is asked to apply that
        change. Regenerating is an edit, so the affected approvals are cleared
        and the export gate re-arms.
        """
        ask = self._drafting_callback("slide content")
        project = self.get(project_id)
        self._require_editable(project)
        targets = self._slides_for_numbers(project, numbers)
        self._reopen_for_edit(project)
        for slide in targets:
            label = slide.subsection or slide.section or slide.title
            self._progress(f"Regenerating slide {slide.number} ({label!r})...")
            if instruction:
                prompt = REVISION_PROMPT.format(
                    kind="slide",
                    title=project.title,
                    current=render_slide_text(slide),
                    feedback=instruction,
                )
            else:
                prompt = SLIDE_PROMPT.format(title=project.title, heading=label)
            answer = ask(prompt)
            self._redraft_slide(project, slide, answer, fallback_title=label)
        self._save(project)
        return targets

    def revise_slides(
        self,
        project_id: str,
        numbers: Iterable[int],
        *,
        instruction: str | None = None,
        template: str = REVISION_PROMPT,
    ) -> list[Slide]:
        """Revise the named slides with an instruction (shorten, expand, ...).

        The default template asks for the current content to be revised; the
        shortcut commands pass :data:`SHORTEN_PROMPT`/:data:`EXPAND_PROMPT`,
        which re-ask for the topic in fewer or more bullets. Either way the slide
        keeps its number and heading and its approval is cleared.
        """
        ask = self._drafting_callback("slide content")
        project = self.get(project_id)
        self._require_editable(project)
        targets = self._slides_for_numbers(project, numbers)
        self._reopen_for_edit(project)
        for slide in targets:
            label = slide.subsection or slide.section or slide.title
            self._progress(f"Revising slide {slide.number} ({label!r})...")
            if template is REVISION_PROMPT:
                prompt = template.format(
                    kind="slide",
                    title=project.title,
                    current=render_slide_text(slide),
                    feedback=instruction or "Improve the scientific accuracy and clarity.",
                )
            else:
                prompt = template.format(title=project.title, heading=label)
            answer = ask(prompt)
            self._redraft_slide(project, slide, answer, fallback_title=label)
        self._save(project)
        return targets

    def add_example_after(
        self, project_id: str, number: int, *, instruction: str | None = None
    ) -> Slide:
        """Draft a worked-example slide and insert it after the named slide."""
        ask = self._drafting_callback("slide content")
        project = self.get(project_id)
        self._require_editable(project)
        anchor = project.slide_by_number(int(number))
        label = anchor.subsection or anchor.section or anchor.title
        self._progress(f"Drafting a worked example after slide {anchor.number}...")
        prompt = EXAMPLE_PROMPT.format(title=project.title, heading=label)
        answer = ask(prompt)
        bullets = bullets_from_text(answer)
        explicit = title_from_text(answer, fallback="")
        title = (
            explicit
            if explicit and _is_title_line(explicit, bullets)
            else f"Worked example: {label}"
        )
        spec = SlideSpec(
            title=title,
            content=bullets,
            section=anchor.section,
            subsection=anchor.subsection,
            notes=instruction or "",
        )
        self._reopen_for_edit(project)
        slide = self._new_slide(project, spec)
        project.insert_slide(slide, anchor.number + 1)
        self._save(project)
        return slide

    def apply_slide_command(self, project_id: str, text: str) -> SlideCommandResult:
        """Parse and run one slide instruction (the review-time command).

        This is the one entry point the window's command box and the CLI's
        ``command`` subcommand share, so a typed "shorten slides 11-15" and a
        spoken one do exactly the same thing. A text that is not a command
        returns ``ok=False`` rather than raising, so a caller can tell the user
        what it understood.
        """
        command = parse_slide_command(text)
        if command is None:
            return SlideCommandResult(
                ok=False,
                action="",
                message=f"I did not recognise a slide instruction in {text!r}.",
            )
        project = self.get(project_id)
        action = command.action

        if action == "renumber":
            self.renumber_slides(project_id)
            return self._command_result(
                project_id, action, f"Renumbered {len(project.slides)} slide(s)."
            )

        if action == "move":
            if not command.slides or command.anchor is None:
                raise PresentationError("A move needs a slide number and an anchor.")
            source = project.slide_by_number(command.slides[0])
            self.move_slide(
                project_id,
                source.slide_id,
                before=command.anchor if command.before else None,
                after=None if command.before else command.anchor,
            )
            where = "before" if command.before else "after"
            return self._command_result(
                project_id,
                action,
                f"Moved slide {command.slides[0]} {where} slide {command.anchor}.",
            )

        if action in ("add-example", "add-slide"):
            if not command.slides:
                raise PresentationError("An add command needs the slide to follow.")
            anchor = command.slides[0]
            slide = self.add_example_after(project_id, anchor)
            return self._command_result(
                project_id,
                action,
                f"Added slide {slide.number} ({slide.slide_id}) after slide {anchor}.",
                slide_id=slide.slide_id,
            )

        if action in ("revise", "shorten", "expand", "regenerate"):
            if action == "regenerate":
                changed = self.regenerate_slides(
                    project_id, command.slides, instruction=command.text
                )
            else:
                template = {
                    "shorten": SHORTEN_PROMPT,
                    "expand": EXPAND_PROMPT,
                }.get(action, REVISION_PROMPT)
                changed = self.revise_slides(
                    project_id,
                    command.slides,
                    instruction=command.text,
                    template=template,
                )
            numbers = ", ".join(str(slide.number) for slide in changed)
            verb = {
                "shorten": "Shortened",
                "expand": "Expanded",
                "regenerate": "Regenerated",
            }.get(action, "Revised")
            return self._command_result(project_id, action, f"{verb} slide(s) {numbers}.")

        if action in ("approve", "approve-batch"):
            if not command.slides:
                raise PresentationError("Name the slide(s) to approve by number.")
            if len(command.slides) > 1:
                approved = self.approve_slides_in_range(
                    project_id, command.slides[0], command.slides[-1]
                )
            else:
                target = project.slide_by_number(command.slides[0])
                approved = [self.approve_slide(project_id, target.slide_id)]
            numbers = ", ".join(str(slide.number) for slide in approved)
            return self._command_result(project_id, action, f"Approved slide(s) {numbers}.")

        raise PresentationError(f"Unhandled slide command {action!r}.")

    def _command_result(
        self,
        project_id: str,
        action: str,
        message: str,
        *,
        slide_id: str = "",
    ) -> SlideCommandResult:
        """Build the report for one executed slide command."""
        project = self.get(project_id)
        return SlideCommandResult(
            ok=True,
            action=action,
            message=message,
            state=project.state,
            slide_id=slide_id,
        )

    # ----------------------------------------------------------- the history

    def revisions(self, project_id: str, slide_id: str) -> list[SlideRevision]:
        """Every recorded revision of one slide, oldest first."""
        return self.get(project_id).revisions_for(slide_id)

    def restore_slide_revision(self, project_id: str, slide_id: str, revision: int) -> Slide:
        """Restore a slide to a recorded revision (the safe undo path).

        Restoring is an edit: it clears the slide's approval, so a restored
        version still has to be approved again before it can be exported.
        """
        project = self.get(project_id)
        self._require_editable(project)
        slide = project.slide(slide_id)
        match = next(
            (item for item in project.revisions_for(slide_id) if item.revision == revision),
            None,
        )
        if match is None:
            raise PresentationError(f"Slide {slide_id} has no revision {revision}.")
        return self._apply_edit(project, slide, dict(match.content))

    def restore_approved(self, project_id: str, slide_id: str) -> Slide:
        """Undo an edit by restoring the slide's last approved content."""
        project = self.get(project_id)
        slide = project.slide(slide_id)
        if slide.approved_snapshot is None:
            raise PresentationError("This slide has no approved version to restore.")
        return self._apply_edit(project, slide, dict(slide.approved_snapshot))

    def export_history(self, project_id: str) -> list[ExportRecord]:
        """Every export attempt recorded for a project, oldest first."""
        return list(self.get(project_id).exports)

    # ----------------------------------------------------------- curriculum

    def course_matches(
        self, project_id: str, *, min_matches: int = DEFAULT_MIN_MATCHES, limit: int | None = None
    ) -> tuple[CourseMatch, ...]:
        """The project's own curriculum courses, strongest first.

        The deck's *content* -- its title, outline, and slides -- decides which of
        the project's courses it covers. The discussion is deliberately excluded:
        it may wander, and what matters is what the deck actually says. This is a
        read-only report, so it never changes the project or its state.
        """
        from core.presentation_courses import match_project

        return match_project(self.get(project_id), min_matches=min_matches, limit=limit)

    def best_course(self, project_id: str) -> CourseMatch | None:
        """The one course the deck is *about*, or ``None`` when none clears the bar."""
        from core.presentation_courses import best_course

        return best_course(self.course_matches(project_id))

    def course_detail(self, project_id: str, course_key: str) -> CourseDetail:
        """A course's concepts and relations, marking what this deck already covers."""
        from core.presentation_courses import course_detail, project_text

        project = self.get(project_id)
        return course_detail(course_key, text=project_text(project))

    def course_concepts_text(self, project_id: str, course_key: str) -> str:
        """A course's *missing* concepts, one per line, for pasting into a draft.

        The Course window offers this so the user can put the ideas the deck has
        not covered yet into the Studio's own question box -- the material comes
        from the curriculum, and the user stays in control of what is added.
        """
        detail = self.course_detail(project_id, course_key)
        return "\n".join(f"- {concept.name}" for concept in detail.missing)

    def slides_for_concept(self, project_id: str, concept: str) -> list[int]:
        """The numbers of the slides whose own text mentions a concept.

        Used to jump the user to the slides that already cover a concept, so the
        Course window's coverage marks are actionable rather than decorative.
        """
        from core.rag import _query_terms

        wanted = _query_terms(concept)
        if not wanted:
            return []
        numbers: list[int] = []
        for slide in self.get(project_id).ordered_slides():
            parts = [slide.title, slide.section, slide.subsection, *slide.content, *slide.equations]
            for row in slide.table or ():
                if isinstance(row, (list, tuple)):
                    parts.extend(str(cell) for cell in row)
            text = " ".join(part for part in parts if part)
            if wanted <= _query_terms(text):
                numbers.append(slide.number)
        return numbers

    # ------------------------------------------------------------- approvals

    def approve_content(self, project_id: str) -> PresentationProject:
        """Approve the complete slide content (SLIDE_REVIEW -> CONTENT_APPROVED).

        Refuses unless every slide is approved, so the export gate cannot be
        opened by a project that still has unreviewed or outdated slides.
        """
        project = self.get(project_id)
        if project.state not in (STATE_SLIDE_REVIEW, STATE_SLIDE_DRAFTING):
            raise PresentationError(
                "Content can only be approved once the slides are under review."
            )
        if not project.slides:
            raise PresentationError("There is no slide content to approve.")
        empty = [slide.number for slide in project.ordered_slides() if slide.is_empty]
        if empty:
            raise PresentationError(
                f"Slide(s) {', '.join(str(number) for number in empty)} are empty; "
                "fill or remove them before approving."
            )
        not_approved = [slide.number for slide in project.ordered_slides() if not slide.approved]
        if not_approved:
            raise PresentationError(
                f"Slide(s) {', '.join(str(number) for number in not_approved)} are not "
                "approved yet; approve every slide before approving the content."
            )
        if project.state == STATE_SLIDE_DRAFTING:
            require_transition(project.state, STATE_SLIDE_REVIEW)
            project.state = STATE_SLIDE_REVIEW
        require_transition(project.state, STATE_CONTENT_APPROVED)
        project.state = STATE_CONTENT_APPROVED
        project.content_approved = True
        return self._save(project)

    def reopen_content(self, project_id: str) -> PresentationProject:
        """Reopen approved content for editing (clears the content approval)."""
        project = self.get(project_id)
        if project.state not in (STATE_CONTENT_APPROVED, STATE_EXPORTED, STATE_EXPORT_FAILED):
            raise PresentationError("Only approved or exported content can be reopened.")
        project.state = STATE_SLIDE_REVIEW
        project.content_approved = False
        return self._save(project)

    def can_export(self, project_id: str) -> tuple[bool, str]:
        """Whether an export may run now, and why not when it may not."""
        project = self.get(project_id)
        if project.state not in (STATE_CONTENT_APPROVED, STATE_EXPORTED, STATE_EXPORT_FAILED):
            return False, (
                "The content must be approved before it can be exported "
                f"(the project is in {STATE_LABELS.get(project.state, project.state)})."
            )
        if not project.slides:
            return False, "There are no slides to export."
        if not project.all_slides_approved:
            outdated = ", ".join(str(slide.number) for slide in project.outdated_slides)
            return False, (
                f"Slide(s) {outdated} changed after approval; approve them again "
                "before exporting."
            )
        return True, ""

    # --------------------------------------------------------------- exporting

    def export(
        self,
        project_id: str,
        *,
        path: str | Path | None = None,
        approved_only: bool = True,
        engine: str | None = None,
        settings: PresentationSettings | None = None,
        overwrite: bool | None = None,
    ) -> PresentationReport:
        """Write the approved slides to a ``.pptx`` and validate the result.

        The state machine drives this: CONTENT_APPROVED (or a retry from
        EXPORT_FAILED) -> EXPORTING -> EXPORTED or EXPORT_FAILED. A failure is
        recorded in the export history and leaves the project in EXPORT_FAILED,
        from which ``export`` may safely be called again.

        ``engine`` selects the renderer: ``"pptx"`` (the optional python-pptx
        engine, the default when it is installed) draws real, editable shapes,
        tables, charts, embedded images, notes, and RTL text; ``"stdlib"`` is the
        dependency-free fallback that always works. When ``engine`` is not given
        the python-pptx engine is preferred and the fallback is used only when it
        is absent -- so a base install and CI never break.

        ``settings`` overrides the project's stored look for this one export.
        The output is never allowed to overwrite an existing file unless
        ``overwrite`` is true (or ``settings.allow_overwrite`` is set): when no
        explicit ``path`` is given a unique, safe name is chosen instead.
        """
        project = self.get(project_id)
        allowed, reason = self.can_export(project_id)
        if not allowed:
            raise PresentationError(reason)

        spec = settings or project.settings
        chosen = self.resolve_engine(engine)

        require_transition(project.state, STATE_EXPORTING)
        project.state = STATE_EXPORTING
        self._save(project)

        export_id = project.next_export_id()
        allow_overwrite = spec.allow_overwrite if overwrite is None else overwrite
        if path is not None:
            target = Path(path).expanduser()
            if target.suffix.lower() != ".pptx":
                target = target.with_suffix(".pptx")
            if target.exists() and not allow_overwrite:
                return self._fail_export(
                    project,
                    export_id,
                    target,
                    f"{target.name} already exists; choose another name or allow an overwrite.",
                )
        else:
            directory = self.output_dir
            filename = safe_filename(project.title, export_id=export_id)
            if allow_overwrite:
                target = directory / filename
            else:
                target = unique_export_path(directory, project.title, export_id=export_id)

        self._progress(f"Writing {target}...")
        try:
            deck = Deck.from_project(project, approved_only=approved_only)
            plan = prepare_deck(deck, settings=spec, project=project)
            warnings = plan.warnings
            if chosen == ENGINE_PPTX:
                from core.pptx_engine import PptxEngine, validate_deck

                written = PptxEngine().write(plan, target)
                validation = validate_deck(written, plan)
                result = ValidationResult(
                    ok=validation.ok,
                    slide_count=validation.slide_count,
                    parts=len(validation.layouts),
                    message=(
                        f"Valid deck with {validation.slide_count} slide(s)."
                        if validation.ok
                        else "; ".join(validation.issues) or "The deck did not validate."
                    ),
                    size_bytes=validation.size_bytes,
                )
                if validation.issues:
                    warnings = warnings + validation.issues
            else:
                written = self._writer.write(deck, target)
                result = validate_pptx(written)
        except PresentationError as error:
            return self._fail_export(project, export_id, target, str(error))
        except OSError as error:
            return self._fail_export(project, export_id, target, str(error))
        except Exception as error:  # noqa: BLE001 - reported, project preserved
            return self._fail_export(project, export_id, target, str(error))

        if not result.ok:
            return self._fail_export(project, export_id, target, result.message)

        record = ExportRecord(
            export_id=export_id,
            path=str(written),
            ok=True,
            message=result.message,
            slide_count=result.slide_count,
            size_bytes=result.size_bytes,
        )
        project.exports.append(record)
        require_transition(project.state, STATE_EXPORTED)
        project.state = STATE_EXPORTED
        self._save(project)
        message = f"Exported {result.slide_count} slide(s) to {written}."
        if warnings:
            message += f" {len(warnings)} note(s) to review."
        return PresentationReport(
            ok=True,
            project_id=project.project_id,
            message=message,
            path=str(written),
            slide_count=result.slide_count,
            size_bytes=result.size_bytes,
            export_id=export_id,
            state=project.state,
            approved_slides=len(project.approved_slides),
            outdated_slides=len(project.outdated_slides),
            engine=chosen,
            filename=Path(written).name,
            warnings=warnings,
        )

    def resolve_engine(self, engine: str | None) -> str:
        """Pick a render engine, falling back honestly when the extra is absent.

        An explicit request for an engine that is unavailable raises; the
        *automatic* choice (``engine is None``) falls back to the standard-library
        writer so a base install still exports.
        """
        from core.pptx_engine import pptx_available

        if engine is None:
            return ENGINE_PPTX if pptx_available() else ENGINE_STDLIB
        key = str(engine).strip().lower()
        if key not in ENGINES:
            raise PresentationError(
                f"Unknown export engine {engine!r}; expected one of {', '.join(ENGINES)}."
            )
        if key == ENGINE_PPTX and not pptx_available():
            raise PresentationError(
                "The python-pptx engine is not installed; install it with "
                "`python -m cli deps --install pptx`, or export with engine='stdlib'."
            )
        return key

    def export_warnings(self, project_id: str, *, settings: PresentationSettings | None = None) -> tuple[str, ...]:
        """The slide warnings an export *would* produce (without writing a file).

        Used by the review screen to flag an over-full slide or a missing image
        *before* the user commits to an export, so nothing is silently clipped.
        """
        project = self.get(project_id)
        spec = settings or project.settings
        deck = Deck.from_project(project, approved_only=True)
        plan = prepare_deck(deck, settings=spec, project=project)
        return plan.warnings

    def update_settings(self, project_id: str, **changes: Any) -> PresentationSettings:
        """Change a project's presentation settings (theme, aspect, metadata)."""
        project = self.get(project_id)
        project.settings = project.settings.replace(**changes)
        project.touch()
        self._save(project)
        return project.settings

    def _fail_export(
        self,
        project: PresentationProject,
        export_id: str,
        target: Path,
        message: str,
    ) -> PresentationReport:
        """Record a failed export and leave the project safely retryable."""
        record = ExportRecord(
            export_id=export_id,
            path=str(target),
            ok=False,
            message=message,
            slide_count=0,
        )
        project.exports.append(record)
        require_transition(project.state, STATE_EXPORT_FAILED)
        project.state = STATE_EXPORT_FAILED
        self._save(project)
        return PresentationReport(
            ok=False,
            project_id=project.project_id,
            message=f"Export failed: {message}",
            path=str(target),
            slide_count=0,
            export_id=export_id,
            state=project.state,
            approved_slides=len(project.approved_slides),
            outdated_slides=len(project.outdated_slides),
        )

    def preview(self, project_id: str, *, approved_only: bool = True) -> Deck:
        """The exact deck an export would write (used by the review screen)."""
        return Deck.from_project(self.get(project_id), approved_only=approved_only)


__all__ = [
    "AskCallback",
    "DiscussResult",
    "EXAMPLE_PROMPT",
    "EXPAND_PROMPT",
    "OUTLINE_PROMPT",
    "PresentationReport",
    "PresentationService",
    "PresentationStatus",
    "ProgressCallback",
    "ProjectValidation",
    "REVISION_PROMPT",
    "SHORTEN_PROMPT",
    "SLIDE_PROMPT",
    "SlideCommandResult",
    "SlideSpec",
    "render_slide_text",
]
