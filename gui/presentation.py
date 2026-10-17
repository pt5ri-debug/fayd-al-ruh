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
from dataclasses import dataclass
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
    bullets_from_text,
    dedup_key,
    default_presentation_dir,
    looks_like_prose_outline,
    outline_from_prose,
    parse_outline,
    presentation_filename,
    require_transition,
    resolve_content_type,
    title_from_text,
    validate_pptx,
)

#: An ``ask`` callable: a question in, the assistant's answer out.
AskCallback = Callable[[str], str]
ProgressCallback = Callable[[str], None]


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

    def summary(self) -> str:
        """A one-line, human-readable description of the outcome."""
        return self.message

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

    def summary(self) -> str:
        """A one-line description, including any outstanding approval."""
        note = " (approval outdated)" if self.approval_outdated else ""
        return (
            f"{self.title} [{self.state_label}]: {self.slides} slide(s), "
            f"{self.approved_slides} approved{note}"
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
        }


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
    ) -> None:
        self._store = store if store is not None else PresentationStore(root)
        self._ask = ask
        self._writer = writer if writer is not None else PresentationWriter()
        self._output_dir = Path(output_dir) if output_dir is not None else None
        self._on_progress = on_progress

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

    def set_ask(self, ask: AskCallback | None) -> None:
        """Wire (or clear) the assistant callback the discussion uses."""
        self._ask = ask

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
        """Load a project; raises when it does not exist."""
        project = self._store.load(project_id)
        if project is None:
            raise PresentationError(f"No presentation project {project_id!r}.")
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
            source = self._require_assistant("outline")(prompt)
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

    def draft_slides_from_outline(
        self,
        project_id: str,
        *,
        heading: str | None = None,
        batch: int | None = None,
    ) -> list[Slide]:
        """Ask the assistant for slide content, one manageable batch at a time.

        With no ``heading`` every not-yet-drafted outline section is drafted;
        with a ``heading`` only that section is. ``batch`` caps how many slides
        one call may add, so a long deck is developed in reviewable pieces.
        """
        ask = self._require_assistant("slide content")
        project = self.get(project_id)
        if project.state not in (STATE_SLIDE_DRAFTING, STATE_SLIDE_REVIEW):
            raise PresentationError("Slides can only be drafted after the outline is approved.")
        outline = project.outline
        sections = list(outline.sections) if outline is not None else []
        if heading is not None:
            sections = [item for item in sections if item.heading == heading]
            if not sections:
                raise PresentationError(f"The outline has no section {heading!r}.")
        drafted = {slide.sources[0] for slide in project.slides if slide.sources}
        # Bullets already used on another slide, so a new slide does not repeat a
        # sentence the deck has already shown (the retriever can return the same
        # passages for two related headings).
        used: set[str] = {
            dedup_key(line)
            for slide in project.slides
            for line in slide.content
        }
        specs: list[SlideSpec] = []
        for section in sections:
            if heading is None and section.heading in drafted:
                continue
            self._progress(f"Drafting slides for {section.heading!r}...")
            prompt = SLIDE_PROMPT.format(title=project.title, heading=section.heading)
            answer = ask(prompt)
            drafted_bullets = bullets_from_text(answer)
            bullets = [line for line in drafted_bullets if dedup_key(line) not in used]
            repeated = False
            if not bullets and drafted_bullets:
                # Every bullet repeated an earlier slide (the retriever can return
                # the same passages for two related headings). A blank slide is
                # worse than a flagged one, so the section keeps its single
                # strongest line and the note says what happened -- slide review is
                # where the user replaces it.
                bullets = drafted_bullets[:1]
                repeated = True
            used.update(dedup_key(line) for line in bullets)
            # A grounded assistant answers in prose, so its first line is a
            # sentence, not the requested title. The section heading is the title
            # in that case -- taking the first sentence as the title too would
            # repeat it verbatim as the slide's first bullet.
            explicit = title_from_text(answer, fallback="")
            if explicit and _is_title_line(explicit, bullets):
                title = explicit
            else:
                title = section.heading
            specs.append(
                SlideSpec(
                    title=title,
                    content=bullets,
                    sources=(section.heading,),
                    notes=REPEATED_HEADING_NOTE if repeated else "",
                )
            )
            if batch is not None and len(specs) >= batch:
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
        """Approve every non-empty slide at once (batch approval)."""
        project = self.get(project_id)
        for slide in project.ordered_slides():
            if not slide.is_empty:
                slide.approved = True
                slide.approved_snapshot = slide.content_snapshot()
                project.record_revision(slide, reason="approved (batch)")
        self._save(project)
        return project

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
    ) -> PresentationReport:
        """Write the approved slides to a ``.pptx`` and validate the result.

        The state machine drives this: CONTENT_APPROVED (or a retry from
        EXPORT_FAILED) -> EXPORTING -> EXPORTED or EXPORT_FAILED. A failure is
        recorded in the export history and leaves the project in EXPORT_FAILED,
        from which ``export`` may safely be called again.
        """
        project = self.get(project_id)
        allowed, reason = self.can_export(project_id)
        if not allowed:
            raise PresentationError(reason)

        require_transition(project.state, STATE_EXPORTING)
        project.state = STATE_EXPORTING
        self._save(project)

        target = (
            Path(path).expanduser()
            if path is not None
            else self.output_dir / presentation_filename(project.title)
        )
        export_id = project.next_export_id()
        self._progress(f"Writing {target}...")
        try:
            deck = Deck.from_project(project, approved_only=approved_only)
            written = self._writer.write(deck, target)
            result = validate_pptx(written)
        except (PresentationError, OSError, ValueError) as error:
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
        return PresentationReport(
            ok=True,
            project_id=project.project_id,
            message=f"Exported {result.slide_count} slide(s) to {written}.",
            path=str(written),
            slide_count=result.slide_count,
            size_bytes=result.size_bytes,
            export_id=export_id,
            state=project.state,
            approved_slides=len(project.approved_slides),
            outdated_slides=len(project.outdated_slides),
        )

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
    "OUTLINE_PROMPT",
    "PresentationReport",
    "PresentationService",
    "PresentationStatus",
    "ProgressCallback",
    "REVISION_PROMPT",
    "SLIDE_PROMPT",
    "SlideSpec",
]
