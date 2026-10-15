"""Headless service layer behind the GUI.

The GUI window is a thin shell over this class. Keeping the logic here means it
can be exercised without a display, and it reuses the exact session the text
``chat`` command builds -- same assistant, same learner, same sandbox, same
``ModelSwitcher`` -- so a GUI answer is produced by the same code path a CLI
answer is.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, TYPE_CHECKING

from ai import chat as chat_module
from ai import mind as mind_module
from ai import ultron_mouth
from ai.jarvis import build_session
from core.command_router import RouteKind, RoutedCommand
from core.jarvis import JarvisMode, JarvisTurn
from core.paths import data_dir
from gui.attachments import Attachment, AttachmentNotice, attach_document, remove_documents
from gui.import_center import ImportReport, KnowledgeImporter
from gui.image_edit import ImageEditService
from gui.image_gen import ImageGenerationService
from gui.sql_import import SqlImportService
from gui.modes import (
    AnswerOnlyRouter,
    HistoryItem,
    ModeSpec,
    load_mode_key,
    resolve_mode,
    save_mode_key,
)

if TYPE_CHECKING:
    from gui.import_center import ProgressCallback


@dataclass
class ChatReply:
    """One rendered assistant reply, ready for the GUI to display."""

    text: str
    citations: tuple[str, ...] = ()
    used_web: bool = False
    learned: bool = False
    mode: str = "sources"
    hint: str = ""
    is_control: bool = False
    steps: list[dict[str, Any]] = field(default_factory=list)


@dataclass(frozen=True)
class ExecutionPreview:
    """What a message would do in the current mode, before it is sent.

    Used to ask for confirmation on a destructive or sensitive action: when
    :attr:`needs_confirmation` is true, the window shows the plan's steps and the
    user must approve before the tool runs. It is computed with the *real* router,
    so it describes the same decision the sandbox would make.
    """

    kind: str
    summary: str
    steps: tuple[dict[str, str], ...] = ()
    needs_confirmation: bool = False


#: Tools that change something on disk, so Jarvics Mode asks before running them.
#: Read-only tools (list/search/read/calculate) run without a prompt; a write is
#: the one sandboxed action that leaves a lasting change, so it is confirmed.
_DESTRUCTIVE_TOOLS = frozenset({"write_file"})


def _context_text(attachment: Attachment) -> str:
    """The attachment's whole text, labelled so the model knows what it is.

    A file's text joins the conversation as reference material; the header names
    the file so a reader (and the model) can tell which document the text belongs
    to when more than one is attached.
    """
    body = attachment.text.strip()
    if not body:
        return ""
    return f"Attached file: {attachment.filename}\n\n{body}"


class ChatService:
    """Owns one conversation session and answers questions for the GUI.

    ``source`` selects which mouth answers, mirroring the CLI's switches:

    - ``"sources"`` -- grounded answers with citations (default).
    - ``"model"`` -- answers from the trained language model, no citations.
    - ``"ultron"`` -- answers from the general local model, no citations.

    ``mode`` selects the *operating mode* (see :mod:`gui.modes`). In **AI Mode**
    the message is answered conversationally and no tool can run. In **Jarvics
    Mode** the same message may be routed to a sandboxed tool by the project's
    real router, under every existing guarantee.
    """

    def __init__(
        self,
        *,
        mode: JarvisMode | None = None,
        no_web: bool = False,
        no_feedback: bool = False,
        root: Path | None = None,
        read_write: bool = False,
        model_checkpoint: str | None = None,
        max_new_tokens: int = 96,
        ultron_model: str | None = None,
        ultron_tokens: int | None = None,
        ultron_temperature: float | None = None,
        session: str | None = None,
        operating_mode: str | None = None,
        data_root: Path | None = None,
        import_store_dir: str | Path | None = None,
        image_store_dir: str | Path | None = None,
    ) -> None:
        self.mode = mode or build_session(
            root=root,
            read_write=read_write,
            no_web=no_web,
            no_feedback=no_feedback,
            text_only=True,
        )
        self.switcher = chat_module.ModelSwitcher(
            self.mode,
            loader=lambda: mind_module.load_mind_model(
                model_checkpoint,
                max_new_tokens=max_new_tokens,
            ),
            max_new_tokens=max_new_tokens,
            ultron_loader=lambda: ultron_mouth.load_ultron_assistant(
                ultron_model,
                max_tokens=ultron_tokens,
                temperature=ultron_temperature,
            ),
        )
        self.session = session
        if session and not self.mode.resume_session(session):
            self.session = None
        # Files the user attached to this conversation, kept so they can be listed
        # in the UI and detached again. Their text is in the assistant's store
        # (see attach_file) *and* in the conversation as reference material, so
        # both the grounded mouth and Ultron can read it.
        self.attachments: list[Attachment] = []
        # (attachment, its reference-material turn), so detaching can take the
        # text back out of the conversation. Kept here because Attachment is
        # frozen and its turn belongs to one conversation.
        self._context_turns: list[tuple[Attachment, Any]] = []

        # The operating mode. The *real* router is kept so Jarvics Mode can run
        # tools; AI Mode swaps in an answer-only router that never plans one.
        self._data_root = data_root if data_root is not None else data_dir()
        self._tool_router = getattr(self.mode, "router", None)
        self._answer_router = (
            AnswerOnlyRouter(self._tool_router.sandbox) if self._tool_router is not None else None
        )
        key = operating_mode if operating_mode is not None else load_mode_key(self._data_root)
        self._mode_spec = resolve_mode(key)
        self._apply_router()

        # The Knowledge Import Center: the window's ＋ control imports files and
        # folders through the existing ingestion pipeline, writing to the same
        # on-disk knowledge store the CLI's `ingest add` writes to.
        self._importer = KnowledgeImporter(store_dir=import_store_dir)
        # The SQL Database Import screen drives the same resumable SQL ingestion
        # service the CLI's `ingest sql-import` command uses, writing to the same
        # store and the same `<store>/ingest/jobs-sql` job files.
        self._sql_import = SqlImportService(store_dir=import_store_dir)
        # The optional text-to-image page. The default backend renders a PNG with
        # only the standard library, under the data folder; no model is required.
        self._image_gen = ImageGenerationService(root=image_store_dir)
        # The optional "edit an uploaded image from text" page. It reuses the same
        # storage folder and needs no model: the default editor is a deterministic
        # keyword interpreter (see core/imageedit.py).
        self._image_edit = ImageEditService(root=image_store_dir)

    # ------------------------------------------------------------ operating mode

    @property
    def operating_mode(self) -> ModeSpec:
        """The current :class:`~gui.modes.ModeSpec` (AI Mode or Jarvics Mode)."""
        return self._mode_spec

    @property
    def operating_mode_key(self) -> str:
        return self._mode_spec.key

    def set_operating_mode(self, key: str) -> ModeSpec:
        """Switch the operating mode, persist it, and return the new spec."""
        spec = resolve_mode(key)
        self._mode_spec = spec
        self._apply_router()
        save_mode_key(self._data_root, spec.key)
        return spec

    def _apply_router(self) -> None:
        """Point the session at the router the mode allows (tools on or off).

        A session without a router (a lightweight test double) is left untouched,
        so the mode logic is still testable without a full Jarvis session.
        """
        if self._tool_router is None or self._answer_router is None:
            return
        self.mode.router = self._tool_router if self._mode_spec.allows_tools else self._answer_router

    @property
    def tools_enabled(self) -> bool:
        return self._mode_spec.allows_tools

    @property
    def tool_names(self) -> tuple[str, ...]:
        """The tools the sandbox would actually allow, for the sidebar's list."""
        if self._tool_router is None:
            return ()
        return tuple(sorted(self._tool_router.allowed_tools))

    def preview_execution(self, message: str) -> ExecutionPreview:
        """Describe what ``message`` would do in the current mode.

        Always computed with the *real* router, so the preview is honest even in
        AI Mode (where the message would not actually run a tool): it tells the
        user what Jarvics Mode *would* do, and whether that needs confirmation.
        """
        if not message.strip() or self._tool_router is None:
            return ExecutionPreview("answer", "")
        routed: RoutedCommand = self._tool_router.route(message)
        if routed.kind is not RouteKind.TOOL or routed.plan is None:
            return ExecutionPreview("answer", routed.reason)
        steps = tuple(
            {"tool": call.tool, "arguments": ", ".join(f"{k}={v}" for k, v in call.arguments.items())}
            for call in routed.plan.calls
        )
        needs = any(call.tool in _DESTRUCTIVE_TOOLS for call in routed.plan.calls)
        return ExecutionPreview(
            kind="tool",
            summary=f"{routed.plan.goal} — {routed.reason}",
            steps=steps,
            needs_confirmation=needs,
        )

    @property
    def source(self) -> str:
        """Which mouth currently answers: ``"model"``, ``"ultron"``, or ``"sources"``."""
        return self.switcher.mouth

    def use_model(self) -> None:
        """Answer from the trained model. Raises when the checkpoint is missing."""
        self.switcher.use_model()

    def use_ultron(self) -> None:
        """Answer from Ultron, the general local model.

        Raises :class:`~core.ultron.UltronUnavailable` when the optional runtime
        or a model file is missing, so the caller can report the real next step.
        """
        self.switcher.use_ultron()

    def use_sources(self) -> None:
        """Answer from retrieved passages with citations."""
        self.switcher.use_sources()

    def ask(self, message: str) -> ChatReply:
        """Handle one user message and return the rendered reply."""
        turn = self.mode.handle(message, require_wake=False, speak=False)
        return self._to_reply(turn)

    def save(self) -> str:
        """Persist the current session; returns the status line."""
        result = self.mode.save_session()
        if result.startswith("Saved") and self.session is None:
            self.session = self.mode.conversation.id
        return result

    def new_conversation(self) -> str:
        """Start a fresh session and return its id."""
        conversation = self.mode.reset_conversation()
        self.session = None
        # The reference-material turns belonged to the old conversation.
        self._context_turns = []
        return conversation.id

    def list_sessions(self) -> tuple[str, ...]:
        """The ids of saved conversations available to resume."""
        return self.mode.list_sessions()

    def saved_session_summaries(self) -> tuple[HistoryItem, ...]:
        """The saved conversations as :class:`HistoryItem`s, newest first.

        The title (the first user turn), the creation time, and a short snippet
        come from the stored conversation, so the sidebar's history can group by
        date and search by text. A store that cannot be read yields nothing rather
        than an exception.
        """
        ids = self.list_sessions()
        memory = getattr(self.mode, "memory", None)
        items: list[HistoryItem] = []
        for session_id in ids:
            created: datetime | None = None
            label = session_id
            snippet = ""
            try:
                conversation = memory.load(session_id) if memory is not None else None
            except Exception:
                conversation = None
            if conversation is not None:
                created = conversation.created_at
                for message in conversation.messages:
                    if message.role.value == "user":
                        snippet = message.text
                        break
                label = (snippet or conversation.title or session_id).strip().splitlines()[0][:60]
            items.append(
                HistoryItem(
                    session_id=session_id,
                    label=label or session_id,
                    created_at=created,
                    is_current=session_id == self.session,
                    snippet=snippet,
                )
            )
        items.sort(key=lambda item: item.created_at or datetime.min.replace(tzinfo=timezone.utc), reverse=True)
        return tuple(items)

    def attach_file(self, path: str | Path, *, ocr: bool = False) -> AttachmentNotice:
        """Read a file, index its text, and remember it as an attachment.

        The document's chunks join the assistant's own store, so a question can
        retrieve and cite them exactly like the built-in corpus. Its *whole* text
        is also added to the conversation as reference material, so a mouth that
        never retrieves -- Ultron -- reads the file too. A read failure comes back
        as ``ok=False`` with a message, never as an exception. ``ocr=True`` reads a
        scanned PDF (needs the ``[ocr]`` extra + Tesseract).
        """
        notice = attach_document(path, ocr=ocr)
        if not notice.ok or not notice.attachments:
            return notice
        retriever = self._retriever()
        if retriever is None:
            return AttachmentNotice(
                filename=notice.filename,
                ok=False,
                message="Attachments need the source assistant, which is not available here.",
            )
        for attachment in notice.attachments:
            for document in attachment.documents:
                try:
                    retriever.index_document(document)
                except (ValueError, OSError) as error:
                    return AttachmentNotice(
                        filename=notice.filename,
                        ok=False,
                        message=f"Could not index the file: {error}",
                    )
            self._add_context(attachment)
            self.attachments.append(attachment)
        return notice

    def _add_context(self, attachment: Attachment) -> None:
        """Put an attachment's whole text in the conversation as reference material."""
        conversation = getattr(self.mode, "conversation", None)
        add_context = getattr(conversation, "add_context", None)
        if not callable(add_context):
            return
        text = _context_text(attachment)
        if not text:
            return
        self._context_turns.append((attachment, add_context(text)))

    def attach_document(self, attachment: Attachment) -> None:
        """Index an already-read :class:`Attachment` (used when wiring in tests)."""
        retriever = self._retriever()
        if retriever is None:
            raise ValueError("Attachments need the source assistant, which is not available here.")
        for document in attachment.documents:
            retriever.index_document(document)
        self._add_context(attachment)
        self.attachments.append(attachment)

    def remove_attachment(self, attachment: Attachment) -> int:
        """Detach a file: drop its documents, and its reference material, from the chat."""
        retriever = self._retriever()
        removed = remove_documents(retriever, attachment) if retriever is not None else 0
        self._drop_context(attachment)
        self.attachments = [item for item in self.attachments if item is not attachment]
        return removed

    def _drop_context(self, attachment: Attachment) -> None:
        """Take an attachment's reference material back out of the conversation."""
        conversation = getattr(self.mode, "conversation", None)
        remove_context = getattr(conversation, "remove_context", None)
        if not callable(remove_context):
            return
        for item, message in list(self._context_turns):
            if item is attachment:
                remove_context(message)
                self._context_turns.remove((item, message))

    def clear_attachments(self) -> int:
        """Detach every file; returns how many were removed."""
        removed = 0
        for attachment in list(self.attachments):
            removed += self.remove_attachment(attachment)
        return removed

    # --------------------------------------------------- knowledge import center

    @property
    def importer(self) -> KnowledgeImporter:
        """The Knowledge Import Center engine (the ingestion pipeline front end)."""
        return self._importer

    def import_files(
        self,
        paths: Iterable[str | Path],
        *,
        ocr: bool | None = None,
        ocr_languages: str | None = None,
        progress: "ProgressCallback | None" = None,
    ) -> ImportReport:
        """Import one or many chosen files into the knowledge base.

        The window's **Upload files** path: each file is imported on its own, so a
        malformed one is recorded and the rest still succeed, and the per-file
        result can be retried. ``ocr=True`` reads scanned PDFs.
        """
        return self._importer.import_files(
            paths, ocr=ocr, ocr_languages=ocr_languages, progress=progress
        )

    def import_folder(
        self,
        folder: str | Path,
        *,
        recursive: bool | None = None,
        ocr: bool | None = None,
        ocr_languages: str | None = None,
        job_id: str | None = None,
        resume: bool = True,
        progress: "ProgressCallback | None" = None,
    ) -> ImportReport:
        """Import every supported file under ``folder`` (**Import from a folder**)."""
        return self._importer.import_folder(
            folder,
            recursive=recursive,
            ocr=ocr,
            ocr_languages=ocr_languages,
            job_id=job_id,
            resume=resume,
            progress=progress,
        )

    def retry_failed(
        self,
        failures: Sequence[Any],
        *,
        ocr: bool | None = None,
        ocr_languages: str | None = None,
        progress: "ProgressCallback | None" = None,
    ) -> ImportReport:
        """Retry the files that failed in an earlier import run (optionally with OCR)."""
        return self._importer.retry_failed(
            failures, ocr=ocr, ocr_languages=ocr_languages, progress=progress
        )

    def import_jobs(self):
        """The recorded import jobs (for the Import Center's Jobs list)."""
        return self._importer.list_jobs()

    def resume_import_job(
        self, job_id: str, *, progress: "ProgressCallback | None" = None
    ) -> ImportReport:
        """Resume an interrupted folder import from its last completed file."""
        return self._importer.resume_job(job_id, progress=progress)

    def import_stats(self) -> dict[str, Any]:
        """A small description of the store and the import bookkeeping."""
        return self._importer.stats()

    @property
    def sql_import(self) -> SqlImportService:
        """The SQL Database Import engine (the resumable SQL ingestion service)."""
        return self._sql_import

    @property
    def image_gen(self) -> ImageGenerationService:
        """The optional text-to-image engine (default: the procedural renderer)."""
        return self._image_gen

    @property
    def image_edit(self) -> ImageEditService:
        """The optional image-editing engine (default: the keyword interpreter)."""
        return self._image_edit

    def _retriever(self):
        """The grounded assistant's retriever, when one is wired in.

        Taken from the switcher's grounded assistant rather than the active one,
        so attaching a file works even while the window is answering from the
        trained model -- the evidence is there the moment the user switches back.
        """
        grounded = getattr(self.switcher, "grounded", None)
        return getattr(grounded, "retriever", None)

    def resume(self, session_id: str) -> bool:
        """Make a saved conversation current. Returns whether it was found."""
        if not self.mode.resume_session(session_id):
            return False
        self.session = session_id
        # Attachments and their reference turns belong to the conversation they
        # were added to, so a resume starts from a clean slate.
        self.attachments = []
        self._context_turns = []
        return True

    def _to_reply(self, turn: JarvisTurn) -> ChatReply:
        text = chat_module.format_turn(turn)
        if turn.routed == "control":
            return ChatReply(text=text, mode=self.source, is_control=True)
        hint = ""
        if (
            self.source == "sources"
            and turn.citations
            and chat_module._NO_SOURCE_MARKER in turn.response
        ):
            hint = chat_module.MIND_HINT
        steps = [
            {
                "tool": step.call.tool,
                "ok": step.result.ok,
                "output": step.result.output,
                "error": step.result.error,
            }
            for step in (turn.agent_run.steps if turn.agent_run is not None else [])
        ]
        return ChatReply(
            text=text,
            citations=tuple(turn.citations),
            used_web=bool(turn.used_web),
            learned=bool(turn.learned),
            mode=self.source,
            hint=hint,
            steps=steps,
        )


__all__ = ["ChatReply", "ChatService", "ExecutionPreview"]
