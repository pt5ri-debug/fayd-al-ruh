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
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

from ai import banner_mouth, cloud_mouth, hakeem_mouth, natasha_mouth, ultron_mouth, vision_mouth
from ai import chat as chat_module
from ai import mind as mind_module
from ai import mouths as mouths_module
from ai.jarvis import build_session
from core import language_models as language_models_module
from core.command_router import RoutedCommand, RouteKind
from core.jarvis import JarvisMode, JarvisTurn
from core.paths import data_dir
from core.reading import TextReader
from core.speech import (
    AudioClip,
    MicRecording,
    MicrophoneRecorder,
    Playback,
    discover_recognizer_name,
    load_recognizer,
    microphone_install_hint,
)
from core.voices import ModelVoice, SystemVoice, Voice, model_voices, voice_keys, voice_sections

from gui.attachments import Attachment, AttachmentNotice, attach_document, remove_documents
from gui.docs_studio import DocsStudioService
from gui.image_edit import ImageEditService
from gui.image_gen import ImageGenerationService
from gui.import_center import ImportReport, KnowledgeImporter
from gui.modes import (
    AnswerOnlyRouter,
    HistoryItem,
    ModeSpec,
    load_mode_key,
    resolve_mode,
    save_mode_key,
)
from gui.presentation import PresentationService
from gui.scene3d import SceneStudioService
from gui.sql_import import SqlImportService

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
    - ``"vision"`` -- answers from the petroleum local model, no citations.
    - ``"banner"`` -- answers from the presentation local model, no citations.
    - ``"cloud"`` -- answers from a remote cloud model over the network, no citations.

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
        vision_model: str | None = None,
        vision_tokens: int | None = None,
        vision_temperature: float | None = None,
        banner_model: str | None = None,
        banner_tokens: int | None = None,
        banner_temperature: float | None = None,
        cloud_provider: str | None = None,
        cloud_model: str | None = None,
        cloud_tokens: int | None = None,
        cloud_temperature: float | None = None,
        session: str | None = None,
        operating_mode: str | None = None,
        data_root: Path | None = None,
        import_store_dir: str | Path | None = None,
        image_store_dir: str | Path | None = None,
        voice: str | None = None,
        tts_engine: str = "auto",
        asr_engine: str = "auto",
        record_seconds: float = 120.0,
    ) -> None:
        self.mode = mode or build_session(
            root=root,
            read_write=read_write,
            no_web=no_web,
            no_feedback=no_feedback,
            text_only=True,
        )
        # A per-mouth language-model override, set from a dedicated model window
        # (see gui/model_chat.py). An empty map means "every mouth uses the model
        # it would pick on its own", which is the historical behaviour; a key here
        # makes that mouth load the chosen model instead.
        self._mouth_models: dict[str, str] = {}
        self.switcher = chat_module.ModelSwitcher(
            self.mode,
            loader=lambda: mind_module.load_mind_model(
                model_checkpoint,
                max_new_tokens=max_new_tokens,
            ),
            max_new_tokens=max_new_tokens,
            ultron_loader=lambda: self._load_mouth(
                chat_module.MOUTH_ULTRON,
                lambda: ultron_mouth.load_ultron_assistant(
                    ultron_model,
                    max_tokens=ultron_tokens,
                    temperature=ultron_temperature,
                ),
            ),
            vision_loader=lambda: self._load_mouth(
                chat_module.MOUTH_VISION,
                lambda: vision_mouth.load_vision_assistant(
                    vision_model,
                    max_tokens=vision_tokens,
                    temperature=vision_temperature,
                ),
            ),
            banner_loader=lambda: self._load_mouth(
                chat_module.MOUTH_BANNER,
                lambda: banner_mouth.load_banner_assistant(
                    banner_model,
                    max_tokens=banner_tokens,
                    temperature=banner_temperature,
                ),
            ),
            cloud_loader=lambda: self._load_mouth(
                chat_module.MOUTH_CLOUD,
                lambda: cloud_mouth.load_cloud_assistant(
                    cloud_model,
                    provider=cloud_provider,
                    max_tokens=cloud_tokens,
                    temperature=cloud_temperature,
                ),
            ),
            natasha_loader=lambda: self._load_mouth(
                chat_module.MOUTH_NATASHA,
                lambda: natasha_mouth.load_natasha_assistant(),
            ),
            hakeem_loader=lambda: self._load_mouth(
                chat_module.MOUTH_HAKEEM,
                lambda: hakeem_mouth.load_hakeem_assistant(),
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
        # Presentation Studio: build a .pptx from a discussion, slide by slide. It
        # drives the *same* service the CLI uses and asks the *existing* assistant
        # (this session's own loop), so no second AI is introduced.
        self._presentation = PresentationService(
            ask=self._presentation_ask, on_progress=self._on_presentation_progress
        )
        # The Studio's own authoring model (an optional Qwen3 GGUF) is built lazily
        # on first use, so constructing the window never loads a multi-gigabyte
        # model. ``None`` means "not looked for yet"; a failed load sets the
        # ``_presentation_author_loaded`` flag and leaves it ``None``, so the
        # fallback to the assistant is not retried (and re-attempted) on every draft.
        self._presentation_author: Any = None
        self._presentation_author_loaded = False
        self._presentation_model_root: Path | None = None
        # Doc's Studio: read any document -- including a scanned PDF or an image --
        # with its own local vision model (a Qwen3-VL GGUF loaded in-process), then
        # ask about it. It borrows *this* session's assistant for the grounded path
        # (see ``docs_assistant_answer``), so no second AI is introduced.
        # ``None`` means the model folder is the default ``data/docs``.
        self._docs_model_root: Path | None = None
        self._docs = DocsStudioService(model_root=self._docs_model_root)
        # The Scene Studio: build, pose, and render a 3D figure locally with the
        # project's own CPU software rasteriser. It needs no model, no extra, and
        # no network -- the figures and props are generated in code -- so it is
        # always available. The window and the CLI drive this same service.
        self._scene3d = SceneStudioService()
        # The text-to-speech reader: it reads the chat's own text aloud with a
        # chosen catalog voice (core/voices.py). It is always available; when no
        # speech engine is installed it reports honestly instead of pretending.
        self._reader = TextReader(engine=tts_engine, voice=voice)
        # Speech-to-text: the microphone icon records a clip and the *same*
        # recognizer the voice loop uses turns it into text. ``auto`` prefers the
        # local Qwen3-ASR model (it covers Arabic), so the icon drives Qwen3-ASR
        # when it is installed and degrades honestly when it is not.
        self._asr_engine = asr_engine
        self._recognizer: Any = None
        self._recording: MicRecording | None = None
        self._record_seconds = record_seconds
        # Progress lines the Studio service emits; the window drains them.
        self.presentation_progress: list[str] = []

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
        """Which mouth currently answers: ``"model"``, ``"ultron"``, ``"vision"``, ``"banner"``, ``"cloud"``, or ``"sources"``."""
        return self.switcher.mouth

    # ------------------------------------------------- per-mouth model choice

    def _load_mouth(self, mouth: str, default_loader):
        """Build ``mouth``'s assistant, honouring a chosen model when one is set.

        With no override this is exactly the loader the mouth always used, so a
        chat that never opened a dedicated model window is unchanged. With an
        override the mouth is built against the chosen model through
        :func:`ai.mouths.build_mouth`, which resolves the model to its GGUF file
        (or its cloud provider/model) rather than re-guessing it.
        """
        key = self._mouth_models.get(mouth)
        if not key:
            return default_loader()
        return mouths_module.build_mouth(mouth, model_key=key)

    def mouth_model(self, mouth: str) -> str:
        """The language-model key a mouth is set to use (``""`` means its default)."""
        return self._mouth_models.get(mouth, "")

    def set_mouth_model(self, mouth: str, key: str) -> None:
        """Point ``mouth`` at the language model ``key`` and rebuild it if active.

        An empty ``key`` clears the override (back to the mouth's own default).
        When the mouth is the one currently answering, it is reloaded at once, so
        the very next question uses the chosen model -- the same "remember it and
        apply it now" rule the cloud provider picker follows.
        """
        if key:
            self._mouth_models[mouth] = key
        else:
            self._mouth_models.pop(mouth, None)
        if self.switcher.mouth == mouth:
            self._reload_active_mouth(mouth)

    def _reload_active_mouth(self, mouth: str) -> None:
        """Rebuild the currently-answering mouth so a new model takes effect."""
        if mouth == chat_module.MOUTH_MODEL:
            self.switcher.use_model()
        elif mouth == chat_module.MOUTH_ULTRON:
            self.switcher.use_ultron()
        elif mouth == chat_module.MOUTH_VISION:
            self.switcher.use_vision()
        elif mouth == chat_module.MOUTH_BANNER:
            self.switcher.use_banner()
        elif mouth == chat_module.MOUTH_CLOUD:
            self.switcher.use_cloud()
        elif mouth == chat_module.MOUTH_NATASHA:
            self.switcher.use_natasha()
        elif mouth == chat_module.MOUTH_HAKEEM:
            self.switcher.use_hakeem()

    def mouth_model_choices(self, mouth: str):
        """The language models this mouth's window offers, with real availability."""
        return language_models_module.models_for_choice(mouth)

    def describe_mouth_model(self, mouth: str) -> str:
        """An honest one-line description of the model a mouth will answer with."""
        return mouths_module.describe_mouth_model(mouth, self._mouth_models.get(mouth, ""))

    def use_model(self) -> None:
        """Answer from the trained model. Raises when the checkpoint is missing."""
        self.switcher.use_model()

    def use_ultron(self) -> None:
        """Answer from Ultron, the general local model.

        Raises :class:`~core.ultron.UltronUnavailable` when the optional runtime
        or a model file is missing, so the caller can report the real next step.
        """
        self.switcher.use_ultron()

    def use_vision(self) -> None:
        """Answer from Vision, the petroleum local model.

        Raises :class:`~core.vision_model.VisionUnavailable` when the optional
        runtime or a model file is missing, so the caller can report the real next
        step.
        """
        self.switcher.use_vision()

    def use_banner(self) -> None:
        """Answer from Banner, the presentation local model.

        Raises :class:`~core.banner.BannerUnavailable` when the optional runtime or
        a model file is missing, so the caller can report the real next step.
        """
        self.switcher.use_banner()

    def use_cloud(self) -> None:
        """Answer from a remote cloud model over the network.

        Raises :class:`~core.cloud_model.CloudUnavailable` when no API key is set
        or the endpoint refuses, so the caller can report the real next step.
        """
        self.switcher.use_cloud()

    def use_natasha(self) -> None:
        """Talk with Natasha, the companion.

        Raises :class:`~companion.companion_agent.CompanionError` when the local
        model cannot answer, so the caller can report the real next step.
        """
        self.switcher.use_natasha()

    def use_hakeem(self) -> None:
        """Talk with Hakeem, the scholar of religion and history.

        He speaks through the same local model as the companion, so this raises
        :class:`~hakeem.hakeem_agent.HakeemError` when that model cannot answer,
        and the caller can report the real next step.
        """
        self.switcher.use_hakeem()

    def cloud_provider(self) -> str:
        """The provider key the Cloud mouth will use (the saved choice, else the default)."""
        from core.cloud_model import load_provider_key

        return load_provider_key(self._data_root)

    def set_cloud_provider(self, key: str) -> str:
        """Remember ``key`` as the Cloud provider, so the next answer uses it.

        Returns the stored key. Raises :class:`ValueError` for an unknown provider,
        so the window can report a bad choice instead of silently ignoring it. If
        the Cloud mouth is already answering, it is rebuilt against the new
        provider at once, so the very next question uses the chosen one.
        """
        from core.cloud_model import save_provider_key

        stored = save_provider_key(self._data_root, key)
        self.switcher.invalidate_cloud()
        if self.switcher.mouth == chat_module.MOUTH_CLOUD:
            self.switcher.use_cloud()
        return stored

    def cloud_model(self) -> str:
        """The model the Cloud mouth will ask for (the saved one, else the default).

        Shown beside the provider picker so a provider whose built-in default was
        retired can be pointed at a working model from the window, not only the CLI.
        """
        from core.cloud_model import load_provider_key, selected_model

        provider = load_provider_key(self._data_root)
        return selected_model(self._data_root, provider)

    def free_cloud_providers(self) -> tuple[tuple[str, str, str, str], ...]:
        """The no-card providers as ``(key, title, api_key_env, signup_url)`` rows.

        The window's "which ones are free, and where do I sign up" answer, read
        from the same catalog the CLI prints, so the two never disagree.
        """
        from core.cloud_model import free_provider_summaries

        return tuple(
            (key, title, env, signup)
            for key, title, env, signup, _summary in free_provider_summaries()
        )

    def cloud_model_choices(self) -> tuple[str, ...]:
        """The models the window offers for the current provider, default first.

        The catalog's own knowledge of the provider's lineup, so the picker is a
        real menu rather than a blank box. The saved model is always included, so a
        model the user typed (one the catalog does not list yet) stays selectable.
        """
        from core.cloud_model import load_provider_key, provider_models

        provider = load_provider_key(self._data_root)
        choices = list(provider_models(provider))
        current = self.cloud_model()
        if current and current not in choices:
            choices.insert(0, current)
        return tuple(choices)

    def set_cloud_model(self, model: str) -> str:
        """Remember ``model`` for the current Cloud provider, and rebuild the mouth.

        Returns the stored name. Raises :class:`ValueError` for an empty name, so
        the window reports a bad entry rather than storing it. This is the window's
        fix for a provider that retired its default model: type a working one once.
        """
        from core.cloud_model import load_provider_key, save_model_key

        stored = save_model_key(self._data_root, load_provider_key(self._data_root), model)
        self.switcher.invalidate_cloud()
        if self.switcher.mouth == chat_module.MOUTH_CLOUD:
            self.switcher.use_cloud()
        return stored

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
        items.sort(key=lambda item: item.created_at or datetime.min.replace(tzinfo=UTC), reverse=True)
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
        progress: ProgressCallback | None = None,
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
        progress: ProgressCallback | None = None,
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
        progress: ProgressCallback | None = None,
    ) -> ImportReport:
        """Retry the files that failed in an earlier import run (optionally with OCR)."""
        return self._importer.retry_failed(
            failures, ocr=ocr, ocr_languages=ocr_languages, progress=progress
        )

    def import_jobs(self):
        """The recorded import jobs (for the Import Center's Jobs list)."""
        return self._importer.list_jobs()

    def resume_import_job(
        self, job_id: str, *, progress: ProgressCallback | None = None
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

    @property
    def presentation(self) -> PresentationService:
        """The Presentation Studio engine (the same service the CLI drives)."""
        return self._presentation

    @property
    def docs(self) -> DocsStudioService:
        """Doc's Studio engine (the same service the CLI's ``docs`` command drives)."""
        return self._docs

    @property
    def scene3d(self) -> SceneStudioService:
        """The Scene Studio engine (the same service the CLI's ``scene3d`` drives).

        It builds, poses, and renders 3D figures locally with the project's own CPU
        software rasteriser -- no GPU, no model download, no network -- so the
        window and the CLI share one scene, one renderer, and one command box.
        """
        return self._scene3d

    def docs_assistant_answer(self, question: str) -> str:
        """Answer one Doc's Studio question with *this* session's assistant.

        The Studio must never stand up a second AI: for the *grounded* path it
        borrows the conversation's own loop (``self.mode``), so a document can also
        be discussed with the same assistant, knowledge base, and citations the chat
        window uses. Its *own* vision model is a separate, clearly-labelled reading.
        Speech is suppressed -- the Studio reads text.
        """
        turn = self.mode.handle(question, require_wake=False, speak=False)
        return turn.answer_text or turn.response

    def _presentation_ask(self, question: str) -> str:
        """Answer one Presentation Studio question with *this* session's assistant.

        The Studio must never stand up a second AI: it borrows the conversation's
        own loop (``self.mode``), so a presentation is discussed with the same
        grounded assistant, the same knowledge base, and the same citations the
        chat window uses. Speech is suppressed -- the Studio reads text.
        """
        turn = self.mode.handle(question, require_wake=False, speak=False)
        return turn.answer_text or turn.response

    def presentation_author(self) -> Any:
        """The Studio's own authoring model (Qwen3 GGUF), or ``None``.

        Drafting an outline or slide bullets is structured prose generation, which
        the grounded assistant -- built to answer with cited extracts -- does not
        do well. When the user has installed the optional authoring model the
        Studio drafts with it; otherwise it falls back to the assistant, so the
        feature works with no extra at all. Construction is lazy and never raises:
        a missing extra or missing weights simply yields ``None``.
        """
        if self._presentation_author_loaded:
            return self._presentation_author
        self._presentation_author_loaded = True
        try:
            from ai.presentation_author import load_presentation_author

            self._presentation_author = load_presentation_author(root=self._presentation_model_root)
        except Exception:
            return None
        self._presentation.set_author(self._presentation_author)
        return self._presentation_author

    def presentation_model_status(self) -> Any:
        """An honest status of the Studio's authoring model, for the window."""
        from core.presentation_model import presentation_model_status

        return presentation_model_status(root=self._presentation_model_root)

    def _on_presentation_progress(self, message: str) -> None:
        """Record a Studio progress line for the window (it polls this list)."""
        self.presentation_progress.append(message)
        del self.presentation_progress[:-32]

    # --------------------------------------------------------------- reading

    @property
    def reader(self) -> TextReader:
        """The text-to-speech reader that reads chat text aloud."""
        return self._reader

    def read_aloud(self, text: str):
        """Read ``text`` aloud with the chosen voice and return the audio clip.

        The clip is always rendered; playback is best-effort, so a machine with no
        audio player still gets the audio (and can save it) rather than an error.
        """
        return self._reader.speak(text)

    def save_speech(self, text: str, path: str | Path) -> Path:
        """Render ``text`` to a WAV file with the chosen voice and return the path."""
        return self._reader.save(text, path)

    def voice_key(self) -> str:
        """The key of the voice currently used to read."""
        return self._reader.voice.key

    def set_voice(self, key: str) -> Voice | SystemVoice | ModelVoice:
        """Choose a reading voice by key; raises for an unknown catalog key.

        A ``system:<engine>:<name>`` key names a real installed voice and is
        resolved against what the OS reports; a ``model:<engine>:<name>`` key names
        a voice of a local model and is resolved the same way; a catalog key is
        resolved as before.
        """
        return self._reader.set_voice(key)

    def voice_keys(self) -> tuple[str, ...]:
        """Every selectable voice key, in catalog order."""
        return voice_keys()

    def system_voices(self) -> tuple[SystemVoice, ...]:
        """The *real* voices installed on this machine (empty when none are known).

        The window prefers these over the synthetic catalog: they are what the
        machine will actually speak with, by their own names.
        """
        return self._reader.system_voices()

    def model_voices(self) -> tuple[ModelVoice, ...]:
        """The voices of the local speech models this machine can run.

        VoiceTut-TTS over audio.cpp, Piper, and Qwen3-TTS each report their own
        named voices, and each returns ``()`` unless it is fully installed, so this
        list is exactly what can be spoken here -- never a promise of a model that
        is absent.
        """
        return model_voices()

    def voice_sections(self):
        """The voices grouped under the source each came from, for the menus.

        One section per source -- *Installed Voices*, *VoiceTut-TTS*, *Piper*,
        *Qwen3-TTS*, and the synthetic catalog -- each carrying only the voices that
        engine actually reports here, so the sidebar can never list a voice that
        would fail to speak.
        """
        return voice_sections()

    def voice_source_notes(self) -> tuple[str, ...]:
        """One honest line per voice source that contributed no voice here.

        So a menu that lists only some of the sources says *why* the others are
        missing and what the real next step is, instead of silently omitting them.
        """
        from core.voices import SOURCE_SYNTHETIC, SOURCE_SYSTEM

        present = {section.source for section in self.voice_sections()}
        notes: list[str] = []
        if SOURCE_SYSTEM not in present:
            from core.speech import system_engine_available

            if system_engine_available():
                notes.append(
                    "Installed Voices: none were reported by this machine's speech engine."
                )
            else:
                notes.append(
                    "Installed Voices: no system speech engine is installed "
                    "(espeak-ng, say, or Windows speech)."
                )
        if SOURCE_SYNTHETIC not in present:
            notes.append(
                "Synthetic catalog: no speech engine is installed to speak these "
                "settings, so they are not offered."
            )
        for note in (
            self.model_speech_status(),
            self.piper_speech_status(),
            self.qwen_tts_voice_status(),
        ):
            if "not ready" in note:
                notes.append(note)
        return tuple(notes)

    def qwen_tts_voice_status(self) -> str:
        """The Qwen3-TTS voice line alone, for the reading-voice menu.

        :meth:`qwen_speech_status` also reports Qwen3-*ASR*, which is speech to text
        and so belongs to the microphone, not to the voice menu.
        """
        from core.qwen_speech import qwen_tts_status

        tts = qwen_tts_status()
        line = "ready" if tts.available else f"not ready — {tts.hint}"
        return f"Qwen3-TTS (English): {line}"

    def model_speech_status(self) -> str:
        """An honest note about the local model voices, for the window.

        Names the engines that can actually speak here and the real next step when
        none can, so a user who sees an absent model-voice section knows why. The
        count is VoiceTut-TTS's *own* speakers, never another engine's voices.
        """
        from core.audiocpp_speech import AUDIOCPP_ENGINE_NAME, audiocpp_status

        status = audiocpp_status()
        voices = model_voices(AUDIOCPP_ENGINE_NAME)
        if voices:
            return f"VoiceTut-TTS (Local Arabic): {len(voices)} voices ready"
        return f"VoiceTut-TTS (Local Arabic): not ready — {status.hint}"

    def reading_status(self) -> str:
        """An honest, human-readable status of the reader."""
        return self._reader.describe()

    def qwen_speech_status(self) -> str:
        """An honest note about the local Qwen3 speech models, for the window.

        Reports each family separately (ASR and TTS can be installed
        independently) and states the Arabic asymmetry plainly: Qwen3-ASR covers
        Arabic, Qwen3-TTS does not.
        """
        from core.qwen_speech import qwen_asr_status, qwen_tts_status

        asr = qwen_asr_status()
        tts = qwen_tts_status()
        asr_line = "ready" if asr.available else f"not ready — {asr.hint}"
        tts_line = "ready" if tts.available else f"not ready — {tts.hint}"
        return (
            f"Qwen3-ASR (speech to text): {asr_line}\n"
            f"Qwen3-TTS (text to speech): {tts_line}"
        )

    def piper_speech_status(self) -> str:
        """An honest note about the local Piper voices, for the window.

        Piper is the natural voice: it sounds like a person rather than the
        robotic system engine, and it speaks Arabic *and* English -- so it is
        offered as the preferred reader whenever it is installed.
        """
        from core.piper_speech import piper_status

        status = piper_status()
        line = "ready" if status.available else f"not ready — {status.hint}"
        return f"Piper (Arabic + English): {line}"

    def reading_available(self) -> bool:
        """True when a real speech engine is installed."""
        return self._reader.available()

    def refresh_reading(self) -> None:
        """Re-discover the speech engine (after one is installed mid-session)."""
        self._reader.refresh()

    # --------------------------------------------------------------- listening

    def render_aloud(self, text: str) -> AudioClip:
        """Render ``text`` to audio with the chosen voice, without playing it.

        The window renders on the worker thread, then plays (and can stop) the
        clip on the main thread, so a long answer never freezes the window and a
        second click on the speaker can end the reading.
        """
        return self._reader.render(text)

    def play_clip(self, clip: AudioClip) -> Playback | None:
        """Play an already-rendered clip in the background; ``None`` if no player."""
        return self._reader.play(clip)

    def listening_available(self) -> bool:
        """True when the microphone icon can work (a recorder *and* an engine).

        Two things are needed: a program that captures audio (``arecord``/``sox``)
        and a speech recognizer. Reported separately so the window can name the
        real missing piece instead of a vague "cannot listen".
        """
        return MicrophoneRecorder().available() and self._resolve_recognizer() is not None

    def microphone_available(self) -> bool:
        """True when a microphone recorder program is installed."""
        return MicrophoneRecorder().available()

    def recognizer_name(self) -> str:
        """The ASR engine that will be used (``'qwen3-asr'``, ``'whisper'``, ...)."""
        return self._resolve_recognizer_name()

    def asr_status(self) -> str:
        """An honest, human-readable status of the microphone path."""
        if not self.microphone_available():
            return (
                "No microphone recorder is installed, so the microphone button cannot "
                "record. " + microphone_install_hint()
            )
        name = self._resolve_recognizer_name()
        if name == "transcript":
            from core.qwen_speech import qwen_asr_status

            return (
                "No speech-to-text engine is installed. "
                + qwen_asr_status().hint
            )
        if name == "qwen3-asr":
            return "Listening through the local Qwen3-ASR model."
        return f"Listening through the {name} engine."

    def start_recording(self) -> MicRecording:
        """Begin a microphone recording; raises when no recorder is installed.

        The capture window is generous (two minutes) because the user stops it
        with a second click on the microphone; the limit is only a safety bound,
        so a forgotten recording cannot grow without end.
        """
        if self._recording is not None and self._recording.recording:
            return self._recording
        recording = MicrophoneRecorder(seconds=self._record_seconds).start()
        self._recording = recording
        return recording

    def stop_recording(self) -> MicRecording | None:
        """Stop the current recording (if any) and return it, or ``None``."""
        recording = self._recording
        self._recording = None
        if recording is not None:
            recording.stop()
        return recording

    def recording(self) -> MicRecording | None:
        """The recording in progress, or ``None``."""
        return self._recording

    def transcribe(self, clip: AudioClip) -> str:
        """Turn a recorded clip into text with the resolved recognizer.

        Raises :class:`ValueError` when no recognizer is available, so the window
        reports the real next step instead of returning an empty transcript.
        """
        recognizer = self._resolve_recognizer()
        if recognizer is None:
            raise ValueError(self.asr_status())
        return recognizer.transcribe(clip).strip()

    def refresh_listening(self) -> None:
        """Drop the cached recognizer, so a newly installed engine is found."""
        self._recognizer = None

    def _resolve_recognizer_name(self) -> str:
        if self._asr_engine == "auto":
            return discover_recognizer_name()
        return self._asr_engine

    def _resolve_recognizer(self):
        """The cached ASR engine, or ``None`` when none can be built here.

        Building an engine is deferred until the microphone is actually used, so
        constructing the window never imports torch. A requested engine that is
        unavailable leaves ``None`` (reported honestly) rather than raising.
        """
        if self._recognizer is not None:
            return self._recognizer
        name = self._resolve_recognizer_name()
        if name == "transcript":
            return None
        try:
            self._recognizer = load_recognizer(name)
        except (ValueError, OSError, ImportError):
            return None
        return self._recognizer

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
