"""Command line for Presentation Studio (the headless twin of its window).

    python -m cli presentation new "Reservoir Porosity"   # a new project
    python -m cli presentation list                       # the projects so far
    python -m cli presentation show PRS-001               # state + slides
    python -m cli presentation rename PRS-001 "New title"
    python -m cli presentation duplicate PRS-001 [--title "A copy"]
    python -m cli presentation delete PRS-001
    python -m cli presentation validate PRS-001           # structure + numbering
    python -m cli presentation test                       # run the suite
    python -m cli presentation discuss PRS-001 "what is porosity?"
    python -m cli presentation outline PRS-001 --text-file plan.txt
    python -m cli presentation outline-draft PRS-001      # ask the assistant
    python -m cli presentation review-outline PRS-001
    python -m cli presentation approve-outline PRS-001 [--revision 2]
    python -m cli presentation draft PRS-001 [--heading "Porosity"] [--batch 3]
    python -m cli presentation add PRS-001 --title "Porosity" --bullet "..."
    python -m cli presentation slides PRS-001               # the numbered slides
    python -m cli presentation edit PRS-001 SLD-001 --title "New title"
    python -m cli presentation move PRS-001 18 --before 17
    python -m cli presentation reorder PRS-001 SLD-002 --position 1
    python -m cli presentation renumber PRS-001
    python -m cli presentation regenerate PRS-001 21-25
    python -m cli presentation revise PRS-001 11-15 --instruction "add units"
    python -m cli presentation command PRS-001 "shorten slides 11-15"
    python -m cli presentation review-slides PRS-001
    python -m cli presentation approve-slide PRS-001 SLD-001
    python -m cli presentation approve-range PRS-001 5-8
    python -m cli presentation unapprove-slide PRS-001 SLD-001
    python -m cli presentation approve-slides PRS-001
    python -m cli presentation approve-content PRS-001
    python -m cli presentation revisions PRS-001 SLD-001
    python -m cli presentation restore PRS-001 SLD-001 --revision 3
    python -m cli presentation export PRS-001 [--output deck.pptx]
    python -m cli presentation history PRS-001

It drives the *same* :class:`gui.presentation.PresentationService` the window
uses, so the CLI and the window share one set of projects, one revision history,
and one export gate. No workflow logic is reimplemented here.

The discussion, outline draft, and slide draft call the project's *existing*
assistant (through ``ai.jarvis.build_session``); with no model installed they
still work when you supply the text yourself (``--text`` / ``--text-file`` /
``--title`` / ``--bullet``).

Exit codes: ``0`` success, ``2`` invalid arguments, ``3`` the assistant is not
available, ``4`` a gate refused the operation, ``5`` an unexpected failure.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from collections.abc import Sequence
from pathlib import Path

from core.paths import bundle_root
from core.presentation import (
    CONTENT_BULLETS,
    CONTENT_TYPES,
    DEFAULT_SLIDE_BATCH,
    STATE_LABELS,
    OutlineRevision,
    OutlineSection,
    PresentationError,
    Slide,
    default_presentation_dir,
    parse_slide_range,
)
from core.presentation_layout import LAYOUTS
from gui.presentation import PresentationService, SlideSpec


def _read_text(path: str | None) -> str | None:
    """Read a text file, or return ``None`` when no path was given."""
    if not path:
        return None
    return Path(path).expanduser().read_text(encoding="utf-8")


def _make_ask(root: Path | None = None, project_id: str | None = None):
    """Build the assistant callback from the project's own session.

    This is the *existing* assistant -- ``ai.jarvis.build_session`` with the
    grounded assistant and the knowledge retriever -- so a presentation is
    discussed with the same mind the chat window uses. Imported lazily, so the
    CLI never loads the model stack unless it is actually needed.

    A project that names a conversation resumes it, so a follow-up question in a
    later run keeps the earlier exchange (and a new project starts a fresh one).
    The session is saved after every turn, so the discussion survives the
    process: returning to a project never means re-arguing from the start.
    """
    from ai.jarvis import build_session

    session = build_session(text_only=True)
    if project_id:
        saved = _load_conversation_id(root, project_id)
        # Resume the project's own conversation when it exists; otherwise this
        # fresh session becomes the project's conversation from here on.
        if not (saved and session.resume_session(saved)):
            _save_conversation_id(root, project_id, session.conversation.id)

    def ask(question: str) -> str:
        turn = session.handle(question)
        if project_id:
            # ``save_session`` reports a write failure instead of raising, so a
            # read-only store never breaks the turn.
            session.save_session()
        # ``answer_text`` is the assistant's own prose; ``response`` may carry a
        # spoken wrapper or an agent summary, so the answer text is preferred.
        return turn.answer_text or turn.response

    return ask


def _conversation_root(root: Path | None) -> Path:
    """Where a project's conversation pointer lives (beside its own projects)."""
    if root is not None:
        return root / "conversations"
    return default_presentation_dir().parent / "conversations"


def _load_conversation_id(root: Path | None, project_id: str) -> str:
    """The saved conversation id for a project, or "" when none is stored."""
    from storage.json_store import JsonStore

    value = JsonStore(_conversation_root(root)).load("map", project_id)
    return value if isinstance(value, str) else ""


def _save_conversation_id(root: Path | None, project_id: str, conversation_id: str) -> None:
    """Remember which conversation a project is discussed in (best effort)."""
    from storage.json_store import JsonStore

    try:
        JsonStore(_conversation_root(root)).save("map", project_id, conversation_id)
    except (OSError, ValueError):
        pass


def _maybe_author(root: Path | None):
    """The Studio's own authoring model, or ``None`` when it is not installed.

    Never raises: a missing optional extra or a missing model file simply means
    the Studio drafts with the project's assistant, which is the documented
    fallback. Imported lazily so the CLI loads no model stack unless it can help.
    """
    try:
        from ai.presentation_author import load_presentation_author

        return load_presentation_author(root=root)
    except Exception:
        return None


def _service(args: argparse.Namespace, *, need_assistant: bool = False) -> PresentationService:
    root = Path(args.root).expanduser() if getattr(args, "root", None) else None
    output_dir = Path(args.output_dir).expanduser() if getattr(args, "output_dir", None) else None
    ask = None
    if need_assistant or getattr(args, "assistant", False):
        ask = _make_ask(root, getattr(args, "project_id", None))
    service = PresentationService(root=root, ask=ask, output_dir=output_dir)
    # Drafting prefers the Studio's own Qwen3 model when it is installed, so the
    # outline and slide bullets are generated prose rather than cited extracts.
    # Only the drafting commands look for it, so a `list` or `status` never probes
    # a runtime it will not use.
    if need_assistant:
        author = _maybe_author(root)
        if author is not None:
            service.set_author(author)
    if not getattr(args, "quiet", False):
        service.set_progress(lambda message: print(message, file=sys.stderr))
    return service


def _print_json(payload: object) -> None:
    print(json.dumps(payload, ensure_ascii=False, indent=2))


def _print_slide(slide: Slide, *, as_json: bool) -> None:
    data = {
        "slide_id": slide.slide_id,
        "number": slide.number,
        "title": slide.title,
        "content_type": slide.content_type,
        "content": list(slide.content),
        "table": [list(row) for row in slide.table],
        "equations": list(slide.equations),
        "images": list(slide.images),
        "sources": list(slide.sources),
        "notes": slide.notes,
        "section": slide.section,
        "subsection": slide.subsection,
        "approved": slide.approved,
        "revision": slide.revision,
    }
    if as_json:
        _print_json(data)
        return
    mark = "approved" if slide.approved else "not approved"
    place = f" [{slide.topic}]" if slide.topic else ""
    print(f"  [{slide.number}] {slide.slide_id} {slide.title!r} ({slide.content_type}, {mark}){place}")
    for line in slide.content:
        print(f"      - {line}")
    for row in slide.table:
        print(f"      | {' | '.join(row)}")
    for line in slide.equations:
        print(f"      = {line}")
    for ref in slide.images:
        print(f"      [image] {ref}")
    if slide.notes:
        print(f"      notes: {slide.notes}")


def _print_project(
    service: PresentationService, project_id: str, *, as_json: bool, slides: bool
) -> None:
    project = service.get(project_id)
    status = service.status(project_id)
    if as_json:
        payload = status.as_dict()
        payload["conversation_ids"] = list(project.conversation_ids)
        if slides:
            payload["slide_list"] = [
                {
                    "slide_id": item.slide_id,
                    "number": item.number,
                    "title": item.title,
                    "approved": item.approved,
                }
                for item in project.ordered_slides()
            ]
        _print_json(payload)
        return
    print(status.summary())
    print(f"  project: {project.project_id}  state: {STATE_LABELS.get(project.state, project.state)}")
    if project.conversation_ids:
        print(f"  conversations: {', '.join(project.conversation_ids)}")
    outline = project.outline
    if outline is not None:
        approved = "approved" if project.approved_outline_revision == outline.revision else "draft"
        print(
            f"  outline revision {outline.revision} ({approved}), "
            f"~{outline.estimated_slides} slide(s):"
        )
        for section in outline.sections:
            print(f"    - {section.heading} ({section.estimated_slides} slides)")
            for sub in section.subsections:
                print(f"        {sub}")
    if slides:
        if not project.slides:
            print("  (no slides yet)")
        for slide in project.ordered_slides():
            _print_slide(slide, as_json=False)


# ----------------------------------------------------------------- handlers


def _run_new(service: PresentationService, args: argparse.Namespace) -> int:
    project = service.new_project(
        args.title,
        subtitle=args.subtitle or "",
        audience=args.audience or "",
        language=args.language,
    )
    if args.as_json:
        _print_json({"ok": True, "project_id": project.project_id, "title": project.title})
    else:
        print(f"Created {project.project_id}: {project.title!r} ({STATE_LABELS[project.state]}).")
    return 0


def _run_list(service: PresentationService, args: argparse.Namespace) -> int:
    projects = service.list_projects()
    if args.as_json:
        _print_json([service.status(item.project_id).as_dict() for item in projects])
        return 0
    if not projects:
        print('No presentation projects yet. Create one with: python -m cli presentation new "Title"')
        return 0
    print(f"{'ID':<9} {'UPDATED':<11} {'SLIDES':>6} {'APPROVED':>8}  STATUS")
    for project in projects:
        status = service.status(project.project_id)
        date = (status.updated_at or status.created_at or "")[:10] or "----------"
        print(
            f"{status.project_id:<9} {date:<11} {status.slides:>6} "
            f"{status.approved_slides:>8}  {status.state_label}  {status.title}"
        )
    return 0


def _run_rename(service: PresentationService, args: argparse.Namespace) -> int:
    project = service.rename_project(args.project_id, args.title)
    if args.as_json:
        _print_json(service.status(project.project_id).as_dict())
        return 0
    print(f"Renamed {project.project_id} to {project.title!r}.")
    return 0


def _run_duplicate(service: PresentationService, args: argparse.Namespace) -> int:
    clone = service.duplicate_project(args.project_id, title=getattr(args, "title", "") or "")
    if args.as_json:
        _print_json(service.status(clone.project_id).as_dict())
        return 0
    print(
        f"Duplicated {args.project_id} as {clone.project_id} ({clone.title!r}) "
        f"with {len(clone.slides)} slide(s)."
    )
    return 0


def _run_delete(service: PresentationService, args: argparse.Namespace) -> int:
    removed = service.delete(args.project_id)
    if args.as_json:
        _print_json({"project_id": args.project_id, "removed": removed})
        return 0
    if removed:
        print(f"Removed {args.project_id}.")
    else:
        print(f"No stored project {args.project_id}.")
    return 0


def _run_validate(service: PresentationService, args: argparse.Namespace) -> int:
    """Check a project's structure and numbering without exporting it."""
    result = service.validate_project(args.project_id)
    if args.as_json:
        _print_json(result.as_dict())
        return 0 if result.ok else 4
    print(result.summary())
    for issue in result.issues:
        print(f"  issue: {issue}")
    for warning in result.warnings:
        print(f"  warning: {warning}")
    return 0 if result.ok else 4


#: The test files that cover Presentation Studio (the ``test`` command's default).
PRESENTATION_TEST_FILES: tuple[str, ...] = (
    "tests/test_presentation.py",
    "tests/test_presentation_authoring.py",
    "tests/test_presentation_engine.py",
    "tests/test_presentation_integration.py",
    "tests/test_presentation_window.py",
    "tests/test_presentation_e2e.py",
    "tests/test_presentation_e2e_window.py",
    "tests/test_presentation_finalize.py",
    "tests/test_presentation_finalize_window.py",
    "tests/test_presentation_model.py",
    "tests/test_presentation_courses.py",
    "tests/test_presentation_course_window.py",
)


def _run_test(service: PresentationService, args: argparse.Namespace) -> int:
    """Run the presentation test suite through the project's own test runner.

    This is the same ``python -m pytest`` the repository uses (never a second
    test framework); it exists so a user can verify the feature from the CLI.
    """
    argv = [sys.executable, "-m", "pytest", "-q"]
    argv.extend(args.files or PRESENTATION_TEST_FILES)
    if args.keyword:
        argv.extend(["-k", args.keyword])
    # Run from the repository root so the default test paths resolve, wherever
    # the CLI was launched from (``core.paths`` owns the same resolution).
    completed = subprocess.run(
        argv, capture_output=args.as_json, text=True, cwd=str(bundle_root())
    )
    if args.as_json:
        _print_json(
            {
                "ok": completed.returncode == 0,
                "command": argv,
                "returncode": completed.returncode,
                "output": completed.stdout,
            }
        )
    return 0 if completed.returncode == 0 else 4


def _run_show(service: PresentationService, args: argparse.Namespace) -> int:
    _print_project(service, args.project_id, as_json=args.as_json, slides=True)
    return 0


def _run_status(service: PresentationService, args: argparse.Namespace) -> int:
    _print_project(service, args.project_id, as_json=args.as_json, slides=False)
    return 0


def _run_discuss(service: PresentationService, args: argparse.Namespace) -> int:
    conversation = getattr(args, "conversation", "") or _load_conversation_id(
        Path(args.root).expanduser() if getattr(args, "root", None) else None,
        args.project_id,
    )
    result = service.discuss(args.project_id, args.question, conversation_id=conversation)
    if args.as_json:
        _print_json(
            {
                "ok": True,
                "project_id": result.project_id,
                "conversation_id": result.conversation_id,
                "question": result.question,
                "answer": result.answer,
            }
        )
    else:
        print(result.answer)
    return 0


def _outline_text(args: argparse.Namespace) -> str | None:
    """The outline text from ``--text`` or ``--text-file``, if either was given."""
    return _read_text(args.text_file) if args.text_file else args.text


def _print_outline(revision: OutlineRevision) -> None:
    print(f"Outline revision {revision.revision}: ~{revision.estimated_slides} slide(s)")
    for section in revision.sections:
        print(f"  - {section.heading} ({section.estimated_slides} slides)")
        for sub in section.subsections:
            print(f"      {sub}")


def _run_outline(service: PresentationService, args: argparse.Namespace) -> int:
    revision = service.draft_outline(args.project_id, text=_outline_text(args), note=args.note or "")
    if args.as_json:
        _print_json({"ok": True, **revision.to_primitive()})
        return 0
    _print_outline(revision)
    return 0


def _run_outline_draft(service: PresentationService, args: argparse.Namespace) -> int:
    revision = service.draft_outline(args.project_id, note=args.note or "")
    if args.as_json:
        _print_json({"ok": True, **revision.to_primitive()})
        return 0
    _print_outline(revision)
    return 0


def _run_outline_edit(service: PresentationService, args: argparse.Namespace) -> int:
    sections: list[OutlineSection] = []
    for heading in args.section:
        parts = heading.split("|", 1)
        subs = (
            tuple(item.strip() for item in parts[1].split(",") if item.strip())
            if len(parts) > 1
            else ()
        )
        sections.append(OutlineSection(heading=parts[0].strip(), subsections=subs))
    revision = service.edit_outline(args.project_id, sections, note=args.note or "edited by the user")
    if args.as_json:
        _print_json({"ok": True, **revision.to_primitive()})
        return 0
    print(f"Outline revision {revision.revision} saved with {len(revision.sections)} section(s).")
    return 0


def _run_outline_revise(service: PresentationService, args: argparse.Namespace) -> int:
    revision = service.request_outline_revision(args.project_id, args.feedback)
    if args.as_json:
        _print_json({"ok": True, **revision.to_primitive()})
        return 0
    _print_outline(revision)
    return 0


def _run_review_outline(service: PresentationService, args: argparse.Namespace) -> int:
    project = service.review_outline(args.project_id)
    print(f"{project.project_id} is now in {STATE_LABELS.get(project.state, project.state)}.")
    return 0


def _run_approve_outline(service: PresentationService, args: argparse.Namespace) -> int:
    project = service.approve_outline(args.project_id, revision=args.revision)
    print(
        f"Approved outline revision {project.approved_outline_revision}; "
        f"the project is now in {STATE_LABELS.get(project.state, project.state)}."
    )
    return 0


def _run_draft(service: PresentationService, args: argparse.Namespace) -> int:
    slides = service.draft_slides_from_outline(
        args.project_id, heading=args.heading, batch=args.batch
    )
    if args.as_json:
        _print_json({"ok": True, "added": [slide.slide_id for slide in slides]})
        return 0
    if not slides:
        print("Nothing new to draft (every outline section already has a slide).")
        return 0
    print(f"Drafted {len(slides)} slide(s):")
    for slide in slides:
        _print_slide(slide, as_json=False)
    return 0


def _run_add(service: PresentationService, args: argparse.Namespace) -> int:
    spec = SlideSpec(
        title=args.title or "",
        content=tuple(args.bullet or ()),
        content_type=args.type,
        table=tuple(
            tuple(cell.strip() for cell in row.split("|")) for row in (args.table_row or ())
        ),
        equations=tuple(args.equation or ()),
        images=tuple(args.image or ()),
        sources=tuple(args.source or ()),
        notes=args.notes or "",
        section=args.section or "",
        subsection=args.subsection or "",
        layout=args.layout or "",
    )
    if args.position is not None:
        slide = service.insert_slide(args.project_id, spec, position=args.position)
    else:
        slide = service.add_slide(args.project_id, spec)
    if args.as_json:
        _print_json({"ok": True, "slide_id": slide.slide_id, "number": slide.number})
        return 0
    print(f"Added slide {slide.number} ({slide.slide_id}): {slide.title!r}")
    return 0


def _run_edit(service: PresentationService, args: argparse.Namespace) -> int:
    fields: dict[str, object] = {}
    if args.title is not None:
        fields["title"] = args.title
    if args.bullet is not None:
        fields["content"] = list(args.bullet)
    if args.type is not None:
        fields["content_type"] = args.type
    if args.notes is not None:
        fields["notes"] = args.notes
    if args.table_row is not None:
        fields["table"] = [row.split("|") for row in args.table_row]
    if args.equation is not None:
        fields["equations"] = list(args.equation)
    if args.source is not None:
        fields["sources"] = list(args.source)
    if args.image is not None:
        fields["images"] = list(args.image)
    if args.section is not None:
        fields["section"] = args.section
    if args.subsection is not None:
        fields["subsection"] = args.subsection
    slide = service.edit_slide(args.project_id, args.slide_id, **fields)
    project = service.get(args.project_id)
    if args.as_json:
        _print_json(
            {"ok": True, "slide_id": slide.slide_id, "state": project.state, "approved": slide.approved}
        )
        return 0
    print(f"Updated {slide.slide_id}; it is now {'approved' if slide.approved else 'not approved'}.")
    if project.approval_outdated:
        print("The content approval is now outdated; approve the slides again before exporting.")
    return 0


def _run_reorder(service: PresentationService, args: argparse.Namespace) -> int:
    service.reorder_slide(args.project_id, args.slide_id, args.position)
    if args.as_json:
        _print_json({"ok": True, "slide_id": args.slide_id, "position": args.position})
        return 0
    print(f"Moved {args.slide_id} to position {args.position}.")
    return 0


def _run_remove(service: PresentationService, args: argparse.Namespace) -> int:
    service.remove_slide(args.project_id, args.slide_id)
    print(f"Removed {args.slide_id}.")
    return 0


def _run_approve_slide(service: PresentationService, args: argparse.Namespace) -> int:
    slide = service.approve_slide(args.project_id, args.slide_id)
    print(f"Approved {slide.slide_id} (slide {slide.number}).")
    return 0


def _run_review_slides(service: PresentationService, args: argparse.Namespace) -> int:
    project = service.review_slides(args.project_id)
    print(f"{project.project_id} is now in {STATE_LABELS.get(project.state, project.state)}.")
    return 0


def _run_unapprove_slide(service: PresentationService, args: argparse.Namespace) -> int:
    slide = service.unapprove_slide(args.project_id, args.slide_id)
    print(f"Withdrew approval from slide {slide.number} ({slide.slide_id}).")
    return 0


def _run_approve_slides(service: PresentationService, args: argparse.Namespace) -> int:
    project = service.approve_slides(args.project_id)
    print(f"Approved {len(project.approved_slides)} slide(s).")
    return 0


def _run_approve_content(service: PresentationService, args: argparse.Namespace) -> int:
    project = service.approve_content(args.project_id)
    print(f"Content approved; the project is now in {STATE_LABELS.get(project.state, project.state)}.")
    return 0


def _run_reopen(service: PresentationService, args: argparse.Namespace) -> int:
    project = service.reopen_content(args.project_id)
    print(
        f"Reopened for editing; the project is now in "
        f"{STATE_LABELS.get(project.state, project.state)}."
    )
    return 0


def _run_revisions(service: PresentationService, args: argparse.Namespace) -> int:
    items = service.revisions(args.project_id, args.slide_id)
    if args.as_json:
        _print_json([item.to_primitive() for item in items])
        return 0
    if not items:
        print(f"No revisions recorded for {args.slide_id}.")
        return 0
    for item in items:
        mark = "approved" if item.approved else "draft"
        print(f"  r{item.revision} ({mark}, {item.reason}): {item.content.get('title')!r}")
    return 0


def _run_restore(service: PresentationService, args: argparse.Namespace) -> int:
    slide = service.restore_slide_revision(args.project_id, args.slide_id, args.revision)
    print(f"Restored {slide.slide_id} to revision {args.revision}; approve it again before exporting.")
    return 0


def _run_restore_approved(service: PresentationService, args: argparse.Namespace) -> int:
    slide = service.restore_approved(args.project_id, args.slide_id)
    print(f"Restored the approved content of {slide.slide_id}; approve it again before exporting.")
    return 0


def _run_export(service: PresentationService, args: argparse.Namespace) -> int:
    settings = None
    if any(
        value is not None
        for value in (args.aspect, args.theme, args.author, args.company, args.subject, args.footer)
    ) or args.rtl or args.no_slide_numbers:
        current = service.get(args.project_id).settings
        settings = current.replace(
            aspect=args.aspect if args.aspect is not None else current.aspect,
            theme=args.theme if args.theme is not None else current.theme,
            author=args.author if args.author is not None else current.author,
            company=args.company if args.company is not None else current.company,
            subject=args.subject if args.subject is not None else current.subject,
            footer=args.footer if args.footer is not None else current.footer,
            rtl=args.rtl or current.rtl,
            show_slide_numbers=False if args.no_slide_numbers else current.show_slide_numbers,
        )
    report = service.export(
        args.project_id,
        path=args.output,
        engine=args.engine,
        settings=settings,
        overwrite=True if args.overwrite else None,
    )
    if args.as_json:
        _print_json(report.as_dict())
        return 0 if report.ok else 4
    print(report.message)
    for warning in report.warnings:
        print(f"  note: {warning}", file=sys.stderr)
    if report.ok:
        print(f"  engine: {report.engine}")
        print(f"  file: {report.path}")
    return 0 if report.ok else 4


def _run_warnings(service: PresentationService, args: argparse.Namespace) -> int:
    """Report the notes an export would raise, without writing a file."""
    warnings = service.export_warnings(args.project_id)
    if args.as_json:
        _print_json({"ok": True, "warnings": list(warnings)})
        return 0
    if not warnings:
        print("No slide needs attention; the deck will export as approved.")
        return 0
    for warning in warnings:
        print(f"  - {warning}")
    return 0


def _run_settings(service: PresentationService, args: argparse.Namespace) -> int:
    """Show or change a project's presentation settings."""
    project = service.get(args.project_id)
    if args.set_aspect is not None or args.set_theme is not None or args.set_footer is not None:
        project.settings = service.update_settings(
            args.project_id,
            **{
                key: value
                for key, value in (
                    ("aspect", args.set_aspect),
                    ("theme", args.set_theme),
                    ("footer", args.set_footer),
                )
                if value is not None
            },
        )
    settings = service.get(args.project_id).settings
    if args.as_json:
        _print_json(settings.to_primitive())
        return 0
    print(f"Project {project.project_id}:")
    print(f"  aspect: {settings.aspect}")
    print(f"  theme: {settings.theme}")
    print(f"  slide numbers: {'on' if settings.show_slide_numbers else 'off'}")
    print(f"  footer: {settings.footer or '(none)'}")
    print(f"  author: {settings.author or '(none)'}")
    return 0

#: A complete sample deck that exercises every layout, so a user can see the
#: whole pipeline end to end without discussing anything with the assistant.
#: Each tuple is ``(title, content_type, content, extra)``.
_DEMO_SLIDES: tuple[tuple[str, str, tuple[str, ...], dict], ...] = (
    (
        "Reservoir Characterisation",
        "title",
        ("A petroleum engineering primer",),
        {"notes": "Introduce the deck and the audience."},
    ),
    (
        "Learning objectives",
        "bullets",
        ("Define porosity and permeability", "Relate pore geometry to flow"),
        {"notes": "Set expectations for the session."},
    ),
    ("Fundamentals", "section", (), {"notes": "Open the first section."}),
    (
        "Porosity and permeability",
        "bullets",
        ("Porosity is the void fraction | Permeability is the flow capacity",
         "Primary intergranular | Secondary fracture"),
        {"section": "Fundamentals", "subsection": "Pore system",
         "notes": "Contrast storage and flow."},
    ),
    (
        "Typical rock properties",
        "table",
        (),
        {"table": (("Rock", "Porosity", "Permeability (mD)"),
                    ("Sandstone", "0.20", "100"),
                    ("Limestone", "0.15", "10"),
                    ("Shale", "0.05", "0.001")),
         "section": "Fundamentals", "subsection": "Pore system",
         "notes": "Values are indicative; cite the source in the review."},
    ),
    (
        "Porosity versus permeability",
        "table",
        (),
        {"table": (("Rock", "Permeability"), ("Sandstone", "100"), ("Limestone", "10")),
         "layout": "chart", "section": "Fundamentals", "subsection": "Pore system",
         "notes": "The trend is not always monotonic; discuss."},
    ),
    (
        "Darcy's law",
        "equation",
        ("Q = k A (ΔP) / (μ L)", "k is permeability, μ is viscosity"),
        {"section": "Fundamentals", "subsection": "Flow",
         "notes": "Walk through each term."},
    ),
    ("Summary", "bullets", ("Porosity stores, permeability flows",
                            "Both depend on the pore network"),
     {"section": "Fundamentals", "notes": "Recap before questions."}),
    ("Thank you", "title", ("Questions and discussion",), {"notes": "Invite questions."}),
)


def _run_demo(service: PresentationService, args: argparse.Namespace) -> int:
    """Build a complete sample deck and export it in one command.

    This walks the *same* state machine a real project walks (draft the outline,
    approve it, add the numbered slides, approve each, approve the content, then
    export), so it is a real demonstration of the pipeline rather than a
    shortcut. Nothing is discussed with the assistant, so it works with no model.
    """
    project = service.new_project(args.title or "Reservoir Characterisation Demo")
    service.draft_outline(project.project_id, text="Fundamentals\n- Pore system\n- Flow")
    service.review_outline(project.project_id)
    service.approve_outline(project.project_id)
    for title, content_type, content, extra in _DEMO_SLIDES:
        spec = SlideSpec(
            title=title,
            content=content,
            content_type=content_type,
            table=extra.get("table", ()),
            notes=extra.get("notes", ""),
            section=extra.get("section", ""),
            subsection=extra.get("subsection", ""),
            layout=extra.get("layout", ""),
        )
        service.add_slide(project.project_id, spec)
    service.review_slides(project.project_id)
    service.approve_slides(project.project_id)
    service.approve_content(project.project_id)
    report = service.export(
        project.project_id,
        path=args.output,
        engine=args.engine,
        overwrite=True if args.overwrite else None,
    )
    if args.as_json:
        _print_json({"ok": report.ok, "project_id": project.project_id, **report.as_dict()})
        return 0 if report.ok else 4
    print(f"Built project {project.project_id} with {report.slide_count} slide(s).")
    print(report.message)
    if report.ok:
        print(f"  engine: {report.engine}")
        print(f"  file: {report.path}")
    return 0 if report.ok else 4


def _run_history(service: PresentationService, args: argparse.Namespace) -> int:
    records = service.export_history(args.project_id)
    if args.as_json:
        _print_json([item.to_primitive() for item in records])
        return 0
    if not records:
        print("No export attempts yet.")
        return 0
    for item in records:
        mark = "ok" if item.ok else "failed"
        print(f"  {item.export_id} [{mark}] {item.path} ({item.slide_count} slides) - {item.message}")
    return 0


def _run_slides(service: PresentationService, args: argparse.Namespace) -> int:
    """List the numbered slides (the review view)."""
    project = service.get(args.project_id)
    slides = project.ordered_slides()
    if args.as_json:
        _print_json(
            {
                "ok": True,
                "project_id": project.project_id,
                "count": len(slides),
                "slides": [
                    {
                        "slide_id": slide.slide_id,
                        "number": slide.number,
                        "section": slide.section,
                        "subsection": slide.subsection,
                        "title": slide.title,
                        "approved": slide.approved,
                    }
                    for slide in slides
                ],
            }
        )
        return 0
    if not slides:
        print("No slides yet.")
        return 0
    for slide in slides:
        mark = "approved" if slide.approved else "not approved"
        place = f" [{slide.topic}]" if slide.topic else ""
        print(f"  {slide.number:>3}. {slide.slide_id} {slide.title or '(untitled)'} ({mark}){place}")
    print(f"{len(slides)} slide(s), {len(project.approved_slides)} approved.")
    return 0


def _parse_slide_numbers(value: str) -> tuple[int, int]:
    """Parse a CLI slide range ("7", "11-15", "11 to 15") into ``(first, last)``."""
    span = parse_slide_range(value)
    if span is not None:
        return span
    raise PresentationError(
        f"{value!r} is not a slide number or range; write e.g. 7 or 11-15."
    )


def _run_approve_range(service: PresentationService, args: argparse.Namespace) -> int:
    first, last = _parse_slide_numbers(args.range)
    approved = service.approve_slides_in_range(args.project_id, first, last)
    if args.as_json:
        _print_json({"ok": True, "approved": [slide.number for slide in approved]})
        return 0
    numbers = ", ".join(str(slide.number) for slide in approved)
    print(f"Approved slide(s) {numbers}.")
    return 0


def _run_regenerate(service: PresentationService, args: argparse.Namespace) -> int:
    first, last = _parse_slide_numbers(args.range)
    numbers = list(range(first, last + 1))
    changed = service.regenerate_slides(
        args.project_id, numbers, instruction=args.instruction
    )
    if args.as_json:
        _print_json({"ok": True, "regenerated": [slide.number for slide in changed]})
        return 0
    print(f"Regenerated {len(changed)} slide(s): {', '.join(str(s.number) for s in changed)}.")
    return 0


def _run_revise_range(service: PresentationService, args: argparse.Namespace) -> int:
    first, last = _parse_slide_numbers(args.range)
    numbers = list(range(first, last + 1))
    changed = service.revise_slides(
        args.project_id, numbers, instruction=args.instruction
    )
    if args.as_json:
        _print_json({"ok": True, "revised": [slide.number for slide in changed]})
        return 0
    print(f"Revised {len(changed)} slide(s): {', '.join(str(s.number) for s in changed)}.")
    return 0


def _run_move(service: PresentationService, args: argparse.Namespace) -> int:
    project = service.get(args.project_id)
    slide = project.slide_by_number(args.number)
    if (args.before is None) == (args.after is None):
        raise PresentationError("Give exactly one of --before/--after.")
    service.move_slide(
        args.project_id, slide.slide_id, before=args.before, after=args.after
    )
    where = "before" if args.before is not None else "after"
    target = args.before if args.before is not None else args.after
    if args.as_json:
        _print_json({"ok": True, "slide_id": slide.slide_id, "where": where, "anchor": target})
        return 0
    print(f"Moved slide {args.number} {where} slide {target}.")
    return 0


def _run_renumber(service: PresentationService, args: argparse.Namespace) -> int:
    project = service.renumber_slides(args.project_id)
    if args.as_json:
        _print_json({"ok": True, "count": len(project.slides)})
        return 0
    print(f"Renumbered {len(project.slides)} slide(s) 1..{len(project.slides)}.")
    return 0


def _run_command(service: PresentationService, args: argparse.Namespace) -> int:
    result = service.apply_slide_command(args.project_id, args.text)
    if args.as_json:
        _print_json(result.as_dict())
        return 0 if result.ok else 4
    print(result.message)
    return 0 if result.ok else 4


def _run_model_status(service: PresentationService, args: argparse.Namespace) -> int:
    """Report whether the Studio's own authoring model can draft, and the next step."""
    from core.presentation_model import presentation_model_status

    root = Path(args.root).expanduser() if getattr(args, "root", None) else None
    status = presentation_model_status(root=root)
    if getattr(args, "as_json", False):
        _print_json(
            {
                "available": status.available,
                "model": status.model,
                "backend": getattr(status, "backend", ""),
                "installed": status.installed,
                "detail": status.detail,
                "hint": status.hint,
            }
        )
        return 0 if status.available else 4
    print(status.summary())
    if not status.available and status.hint:
        print(f"Next step: {status.hint}")
    return 0 if status.available else 4


def _run_model_download(service: PresentationService, args: argparse.Namespace) -> int:
    """Download the Studio's authoring model (a human-approved external action)."""
    from core.presentation_model import download_model, spec_summaries

    if getattr(args, "list_models", False):
        for key, title, size, summary in spec_summaries():
            print(f"{key}\t{title}\t~{size:.1f} GB\t{summary}")
        return 0
    root = Path(args.root).expanduser() if getattr(args, "root", None) else None
    try:
        path = download_model(
            getattr(args, "model", None) or "qwen3-8b",
            root=root,
            approved=bool(getattr(args, "approve", False)),
            progress=None if getattr(args, "quiet", False) else lambda m: print(m, file=sys.stderr),
        )
    except Exception as error:
        print(str(error), file=sys.stderr)
        return 4
    if getattr(args, "as_json", False):
        _print_json({"path": str(path)})
    else:
        print(f"Saved the presentation model to {path}")
    return 0


def _run_courses(service: PresentationService, args: argparse.Namespace) -> int:
    """Show which of the project's own courses the deck covers.

    The headless twin of the Course window: it reports the curriculum courses the
    deck's own content matches, and -- with ``--course`` -- one course's concepts
    marked covered or missing, so a script can drive the same "reach suitable
    course content" step the window offers.
    """
    if args.course:
        try:
            detail = service.course_detail(args.project_id, args.course)
        except KeyError as error:
            print(str(error), file=sys.stderr)
            return 2
        if args.as_json:
            _print_json(
                {
                    "key": detail.key,
                    "label": detail.label,
                    "branch": detail.branch,
                    "covered": [c.name for c in detail.covered],
                    "missing": [c.name for c in detail.missing],
                    "relations": [r.describe() for r in detail.relations],
                }
            )
            return 0
        print(f"{detail.label} — {detail.branch}")
        print(f"  covered ({len(detail.covered)}): " + ", ".join(c.name for c in detail.covered))
        print(f"  missing ({len(detail.missing)}): " + ", ".join(c.name for c in detail.missing))
        for relation in detail.relations[: args.limit or 10]:
            print(f"    {relation.describe()}")
        return 0

    matches = service.course_matches(args.project_id, limit=args.limit)
    if args.as_json:
        _print_json(
            [
                {
                    "key": match.key,
                    "label": match.label,
                    "branch": match.branch,
                    "matched": list(match.matched),
                    "match_count": match.match_count,
                    "total": match.total,
                    "coverage": round(match.coverage, 4),
                    "strong": match.strong,
                }
                for match in matches
            ]
        )
        return 0
    if not matches:
        print("No curriculum course matches this deck's content yet.")
        return 0
    for match in matches:
        flag = "strong" if match.strong else "related"
        print(f"  [{flag}] {match.describe()} ({match.match_count}/{match.total})")
    best = service.best_course(args.project_id)
    if best is not None:
        print(f"\nThe deck is about: {best.label} ({best.branch})")
    return 0


_HANDLERS = {
    "new": _run_new,
    "list": _run_list,
    "show": _run_show,
    "status": _run_status,
    "rename": _run_rename,
    "duplicate": _run_duplicate,
    "delete": _run_delete,
    "validate": _run_validate,
    "test": _run_test,
    "discuss": _run_discuss,
    "outline": _run_outline,
    "outline-draft": _run_outline_draft,
    "outline-edit": _run_outline_edit,
    "outline-revise": _run_outline_revise,
    "review-outline": _run_review_outline,
    "approve-outline": _run_approve_outline,
    "draft": _run_draft,
    "add": _run_add,
    "edit": _run_edit,
    "reorder": _run_reorder,
    "move": _run_move,
    "slides": _run_slides,
    "renumber": _run_renumber,
    "regenerate": _run_regenerate,
    "revise": _run_revise_range,
    "command": _run_command,
    "remove": _run_remove,
    "review-slides": _run_review_slides,
    "approve-slide": _run_approve_slide,
    "unapprove-slide": _run_unapprove_slide,
    "approve-slides": _run_approve_slides,
    "approve-range": _run_approve_range,
    "approve-content": _run_approve_content,
    "reopen": _run_reopen,
    "revisions": _run_revisions,
    "restore": _run_restore,
    "restore-approved": _run_restore_approved,
    "export": _run_export,
    "warnings": _run_warnings,
    "settings": _run_settings,
    "demo": _run_demo,
    "history": _run_history,
    "model-status": _run_model_status,
    "model-download": _run_model_download,
    "courses": _run_courses,
}

#: The commands that need the project's own assistant wired in.
_NEEDS_ASSISTANT = frozenset(
    {"discuss", "outline-draft", "outline-revise", "draft", "regenerate", "revise", "command"}
)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="presentation", description=__doc__)
    parser.add_argument("--root", default=None, help="Where projects are stored (default: the data directory).")
    parser.add_argument("--output-dir", default=None, help="Where exports are written by default.")
    parser.add_argument(
        "--assistant", action="store_true",
        help="Load the project's assistant for drafting (implied by discuss/outline-draft/draft).",
    )
    parser.add_argument("--quiet", action="store_true", help="Suppress progress output on stderr.")
    parser.add_argument("--json", action="store_true", dest="as_json", help="Machine-readable output.")
    sub = parser.add_subparsers(dest="command")

    new = sub.add_parser("new", help="Create a new presentation project.")
    new.add_argument("title")
    new.add_argument("--subtitle", default="")
    new.add_argument("--audience", default="")
    new.add_argument("--language", default="en", help="Content language hint (default: en).")

    sub.add_parser("list", help="List the presentation projects.")

    rename = sub.add_parser("rename", help="Rename a presentation project.")
    rename.add_argument("project_id")
    rename.add_argument("title")

    duplicate = sub.add_parser("duplicate", help="Copy a project under a new id.")
    duplicate.add_argument("project_id")
    duplicate.add_argument("--title", default="", help="The copy's title (default: '<title> (copy)').")

    delete = sub.add_parser("delete", help="Remove a project and its stored file.")
    delete.add_argument("project_id")

    validate = sub.add_parser(
        "validate", help="Check a project's structure and slide numbering (no export)."
    )
    validate.add_argument("project_id")

    test = sub.add_parser("test", help="Run the Presentation Studio test suite.")
    test.add_argument("files", nargs="*", help="Test paths (default: the presentation tests).")
    test.add_argument(
        "-k", "--keyword", default="", help="Only run tests matching this pytest keyword."
    )

    show = sub.add_parser("show", help="Show a project, its outline, and its slides.")
    show.add_argument("project_id")

    status = sub.add_parser("status", help="Show a project's state and counts.")
    status.add_argument("project_id")

    discuss = sub.add_parser("discuss", help="Discuss the subject with the project's assistant.")
    discuss.add_argument("project_id")
    discuss.add_argument("question")
    discuss.add_argument("--conversation", default="", help="Record this conversation id.")

    outline = sub.add_parser("outline", help="Set the outline from text you supply.")
    outline.add_argument("project_id")
    outline.add_argument("--text", default=None, help="The outline text.")
    outline.add_argument("--text-file", default=None, help="Read the outline text from this file.")
    outline.add_argument("--note", default="")

    outline_draft = sub.add_parser("outline-draft", help="Ask the assistant to propose an outline.")
    outline_draft.add_argument("project_id")
    outline_draft.add_argument("--note", default="")

    outline_edit = sub.add_parser("outline-edit", help="Replace the outline with your own sections.")
    outline_edit.add_argument("project_id")
    outline_edit.add_argument(
        "--section", action="append", required=True,
        help='A section, e.g. --section "Reservoir|Porosity,Permeability".',
    )
    outline_edit.add_argument("--note", default="")

    outline_revise = sub.add_parser("outline-revise", help="Ask the assistant to revise the outline.")
    outline_revise.add_argument("project_id")
    outline_revise.add_argument("feedback")

    review = sub.add_parser("review-outline", help="Move the project into outline review.")
    review.add_argument("project_id")

    approve_outline = sub.add_parser("approve-outline", help="Approve an outline revision.")
    approve_outline.add_argument("project_id")
    approve_outline.add_argument(
        "--revision", type=int, default=None,
        help="Approve this revision instead of the newest (to restore one).",
    )

    draft = sub.add_parser("draft", help="Ask the assistant for slide content.")
    draft.add_argument("project_id")
    draft.add_argument("--heading", default=None, help="Draft only this outline section.")
    draft.add_argument(
        "--batch", type=int, default=None,
        help=f"Add at most this many slides (default: {DEFAULT_SLIDE_BATCH}).",
    )

    add = sub.add_parser("add", help="Add one slide.")
    add.add_argument("project_id")
    add.add_argument("--title", default="")
    add.add_argument("--bullet", action="append", help="A content line (repeatable).")
    add.add_argument("--type", default=CONTENT_BULLETS, choices=list(CONTENT_TYPES))
    add.add_argument("--table-row", action="append", help='A table row, cells split on "|" (repeatable).')
    add.add_argument("--equation", action="append", help="An equation line (repeatable).")
    add.add_argument("--image", action="append", help="An image reference (repeatable).")
    add.add_argument("--source", action="append", help="A source reference (repeatable).")
    add.add_argument("--notes", default="")
    add.add_argument("--section", default="", help="The outline section this slide covers.")
    add.add_argument("--subsection", default="", help="The outline subsection it covers.")
    add.add_argument(
        "--layout",
        default="",
        choices=["", *LAYOUTS],
        help="An explicit layout (default: inferred from the content and title).",
    )
    add.add_argument(
        "--position", type=int, default=None,
        help="Insert at this 1-based position instead of appending.",
    )

    edit = sub.add_parser("edit", help="Edit one slide's content.")
    edit.add_argument("project_id")
    edit.add_argument("slide_id")
    edit.add_argument("--title", default=None)
    edit.add_argument("--bullet", action="append", help="Replace the content lines (repeatable).")
    edit.add_argument("--type", default=None, choices=list(CONTENT_TYPES))
    edit.add_argument("--table-row", action="append", help='Replace the table rows (cells split on "|").')
    edit.add_argument("--equation", action="append")
    edit.add_argument("--image", action="append")
    edit.add_argument("--source", action="append")
    edit.add_argument("--notes", default=None)
    edit.add_argument("--section", default=None, help="Re-home the slide under this section.")
    edit.add_argument("--subsection", default=None, help="Re-home it under this subsection.")

    reorder = sub.add_parser("reorder", help="Move a slide to a new position.")
    reorder.add_argument("project_id")
    reorder.add_argument("slide_id")
    reorder.add_argument("--position", type=int, required=True)

    move = sub.add_parser("move", help="Move a slide before or after another slide.")
    move.add_argument("project_id")
    move.add_argument("number", type=int, help="The slide number to move.")
    move.add_argument("--before", type=int, default=None, help="Put it before this slide number.")
    move.add_argument("--after", type=int, default=None, help="Put it after this slide number.")

    slides = sub.add_parser("slides", help="List the numbered slides.")
    slides.add_argument("project_id")

    renumber = sub.add_parser("renumber", help="Renumber every slide 1..n in display order.")
    renumber.add_argument("project_id")

    regenerate = sub.add_parser("regenerate", help="Re-draft slides, keeping their numbers.")
    regenerate.add_argument("project_id")
    regenerate.add_argument("range", help="A slide number or range, e.g. 7 or 11-15.")
    regenerate.add_argument("--instruction", default=None, help="What to change (optional).")

    revise = sub.add_parser("revise", help="Revise slides with an instruction.")
    revise.add_argument("project_id")
    revise.add_argument("range", help="A slide number or range, e.g. 7 or 11-15.")
    revise.add_argument("--instruction", default=None, help="What to change.")

    command = sub.add_parser(
        "command", help="Run one slide instruction, e.g. 'shorten slides 11-15'."
    )
    command.add_argument("project_id")
    command.add_argument("text", help="The instruction to parse and run.")

    remove = sub.add_parser("remove", help="Remove a slide.")
    remove.add_argument("project_id")
    remove.add_argument("slide_id")

    review_slides = sub.add_parser("review-slides", help="Move the project into slide review.")
    review_slides.add_argument("project_id")

    approve_slide = sub.add_parser("approve-slide", help="Approve one slide.")
    approve_slide.add_argument("project_id")
    approve_slide.add_argument("slide_id")

    unapprove_slide = sub.add_parser("unapprove-slide", help="Withdraw a slide's approval.")
    unapprove_slide.add_argument("project_id")
    unapprove_slide.add_argument("slide_id")

    approve_slides = sub.add_parser("approve-slides", help="Approve every non-empty slide.")
    approve_slides.add_argument("project_id")

    approve_range = sub.add_parser(
        "approve-range", help="Approve the slides in a number range, e.g. 5-8."
    )
    approve_range.add_argument("project_id")
    approve_range.add_argument("range", help="A slide number or range, e.g. 7 or 5-8.")

    approve_content = sub.add_parser("approve-content", help="Approve the complete slide content.")
    approve_content.add_argument("project_id")

    reopen = sub.add_parser("reopen", help="Reopen approved content for editing.")
    reopen.add_argument("project_id")

    revisions = sub.add_parser("revisions", help="List a slide's recorded revisions.")
    revisions.add_argument("project_id")
    revisions.add_argument("slide_id")

    restore = sub.add_parser("restore", help="Restore a slide to a recorded revision.")
    restore.add_argument("project_id")
    restore.add_argument("slide_id")
    restore.add_argument("--revision", type=int, required=True)

    restore_approved = sub.add_parser("restore-approved", help="Undo an edit: restore the approved content.")
    restore_approved.add_argument("project_id")
    restore_approved.add_argument("slide_id")

    export = sub.add_parser("export", help="Write the approved slides to a .pptx.")
    export.add_argument("project_id")
    export.add_argument("--output", default=None, help="Write here (default: the data directory).")
    export.add_argument(
        "--engine",
        choices=["pptx", "stdlib"],
        default=None,
        help="Render engine: python-pptx (default when installed) or the stdlib fallback.",
    )
    export.add_argument("--aspect", choices=["16:9", "4:3"], default=None)
    export.add_argument("--theme", default=None, help="Theme name (professional, light, academic, dark).")
    export.add_argument("--author", default=None)
    export.add_argument("--company", default=None)
    export.add_argument("--subject", default=None)
    export.add_argument("--footer", default=None)
    export.add_argument("--rtl", action="store_true", help="Force right-to-left paragraphs.")
    export.add_argument(
        "--no-slide-numbers", action="store_true", help="Omit the slide number on each slide."
    )
    export.add_argument(
        "--overwrite", action="store_true", help="Replace an existing file of the same name."
    )

    warnings = sub.add_parser(
        "warnings", help="Report the notes an export would raise, without writing a file."
    )
    warnings.add_argument("project_id")

    settings = sub.add_parser("settings", help="Show or change a project's presentation settings.")
    settings.add_argument("project_id")
    settings.add_argument("--set-aspect", choices=["16:9", "4:3"], default=None)
    settings.add_argument("--set-theme", default=None)
    settings.add_argument("--set-footer", default=None)

    history = sub.add_parser("history", help="Show the export history.")
    history.add_argument("project_id")

    demo = sub.add_parser(
        "demo", help="Build a complete sample deck and export it in one command."
    )
    demo.add_argument("--title", default="", help="The project title (default: a demo title).")
    demo.add_argument("--output", default=None, help="Write here (default: the data directory).")
    demo.add_argument("--engine", choices=["pptx", "stdlib"], default=None)
    demo.add_argument("--overwrite", action="store_true", help="Replace an existing file.")

    sub.add_parser(
        "model-status", help="Show whether the Studio's own authoring model can draft."
    )
    model_download = sub.add_parser(
        "model-download",
        help="Download the Studio's own authoring model (Qwen3 8B by default).",
    )
    model_download.add_argument(
        "--model", default=None, help="Which model to fetch (see --list-models)."
    )
    model_download.add_argument(
        "--list-models", action="store_true", help="List the fetchable models and exit."
    )
    model_download.add_argument(
        "--approve", action="store_true",
        help="Record the human approval the download gate requires.",
    )

    courses = sub.add_parser(
        "courses",
        help="Show the project's own curriculum courses the deck covers.",
    )
    courses.add_argument("project_id")
    courses.add_argument(
        "--course", default=None,
        help="Show one course's concepts (covered/missing) and relations.",
    )
    courses.add_argument(
        "--limit", type=int, default=None, help="Cap how many courses (or relations) to show."
    )

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)
    if not args.command:
        parser.print_help()
        return 0
    try:
        service = _service(args, need_assistant=args.command in _NEEDS_ASSISTANT)
        return _HANDLERS[args.command](service, args)
    except PresentationError as error:
        print(str(error), file=sys.stderr)
        return 4
    except ModuleNotFoundError as error:
        print(f"The assistant is not available: {error}", file=sys.stderr)
        return 3
    except OSError as error:
        print(f"Could not complete the operation: {error}", file=sys.stderr)
        return 5


if __name__ == "__main__":
    raise SystemExit(main())
